"""
V4.0 精准片段划分 — 时间定位器 (Temporal Localizer)

核心逻辑 (三步定位):
  1. 确定逻辑断裂点: 从 Decision LLM 审计结论获取 V_break
  2. 后向路径溯源: 沿图结构回溯，寻找实体 Embedding 显著位移的起点
  3. 物理动能微调: 在逻辑起点附近搜索 motion_energy 曲线的斜率突变帧

最终输出:
  - [Start_Time, End_Time] 精准异常片段
  - 可选: 自动调用 OpenCV 截取并保存异常短视频
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import LocalizationConfig, OUTPUT_DIR
from ..association.temporal_graph import TemporalGraph

logger = logging.getLogger(__name__)


@dataclass
class AnomalySegment:
    """一个精准异常片段"""
    entity_id: int
    start_time: float                # 起始时间 (秒)
    end_time: float                  # 结束时间 (秒)
    start_frame: int = -1            # 起始帧号 (如有)
    end_frame: int = -1              # 结束帧号 (如有)
    confidence: float = 0.0          # 置信度
    reason_zh: str = ""              # 中文解释
    reason_en: str = ""              # 英文解释
    localization_method: str = ""    # 定位方法描述
    clip_path: str = ""              # 保存的视频片段路径


@dataclass
class LocalizationResult:
    """视频级定位结果"""
    segments: list[AnomalySegment] = field(default_factory=list)
    total_anomaly_duration: float = 0.0   # 总异常时长 (秒)
    video_duration: float = 0.0
    clips_saved: list[str] = field(default_factory=list)


class TemporalLocalizer:
    """
    V4.0 时间定位器。

    基于 Decision LLM 审计结论，通过语义回溯 + 动能微调
    精确定位异常片段的 [Start_Time, End_Time]。
    """

    def __init__(self, cfg: Optional[LocalizationConfig] = None):
        self.cfg = cfg or LocalizationConfig()

    def localize_anomaly(
        self,
        entity_id: int,
        graph: TemporalGraph,
        break_timestamp: Optional[float],
        confidence: float = 0.0,
        reason_zh: str = "",
        reason_en: str = "",
        video_fps: float = 30.0,
        video_duration: float = 0.0,
    ) -> Optional[AnomalySegment]:
        """
        对单个异常实体进行精准片段定位。

        Args:
            entity_id:        实体 ID
            graph:            时间演化图
            break_timestamp:  Decision LLM 给出的逻辑断裂时间
            confidence:       审计置信度
            reason_zh:        中文审计结论
            reason_en:        英文审计结论
            video_fps:        视频帧率
            video_duration:   视频总时长

        Returns:
            AnomalySegment 或 None (无法定位)
        """
        path = graph.get_entity_path(entity_id)
        edges = graph.get_entity_edges(entity_id)

        if len(path) == 0:
            return None

        # 单帧实体 — 直接用 break_timestamp 或帧时间戳生成片段
        if len(path) < 2:
            ts = break_timestamp if break_timestamp is not None else path[0].get("timestamp", 0.0)
            half_dur = max(self.cfg.min_segment_duration_sec, 3.0)
            s = max(0.0, ts - half_dur)
            e = ts + half_dur
            if video_duration > 0:
                e = min(video_duration, e)
            return AnomalySegment(
                entity_id=entity_id,
                start_time=round(s, 2), end_time=round(e, 2),
                start_frame=int(s * video_fps), end_frame=int(e * video_fps),
                confidence=confidence, reason_zh=reason_zh, reason_en=reason_en,
                localization_method="single_frame_expansion",
            )

        # ── Step 1: 确定逻辑断裂点 ──
        break_idx = self._find_break_index(path, break_timestamp)
        if break_idx is None:
            # 如果无法确定断裂点，使用最后一个节点
            break_idx = len(path) - 1
            logger.info(
                f"Entity #{entity_id}: no exact break point, "
                f"using last node at idx={break_idx}"
            )

        break_node = path[break_idx]
        end_time = break_node["timestamp"]

        # ── Step 2: 后向路径溯源 ──
        start_idx = self._backward_trace(path, edges, break_idx)
        start_time_logic = path[start_idx]["timestamp"]

        # ── Step 3: 物理动能微调 ──
        start_time_phys = self._energy_fine_tune(
            graph, start_time_logic, path, video_fps
        )

        # 使用更精确的起始时间
        start_time = min(start_time_logic, start_time_phys)

        # 添加前后 padding
        start_time = max(0.0, start_time - self.cfg.segment_padding_sec)
        end_time = end_time + self.cfg.segment_padding_sec
        if video_duration > 0:
            end_time = min(video_duration, end_time)

        # 验证片段时长
        duration = end_time - start_time
        if duration < self.cfg.min_segment_duration_sec:
            # 扩展到最小时长
            mid = (start_time + end_time) / 2.0
            half = self.cfg.min_segment_duration_sec / 2.0
            start_time = max(0.0, mid - half)
            end_time = mid + half
            if video_duration > 0:
                end_time = min(video_duration, end_time)

        if duration > self.cfg.max_segment_duration_sec:
            # 截断到最大时长 (从 break 点往前截)
            start_time = max(0.0, end_time - self.cfg.max_segment_duration_sec)

        # 计算帧号
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)

        method = (
            f"semantic_backtrack(idx {start_idx}->{break_idx}) + "
            f"energy_finetune({start_time_phys:.2f}s)"
        )

        segment = AnomalySegment(
            entity_id=entity_id,
            start_time=round(start_time, 2),
            end_time=round(end_time, 2),
            start_frame=start_frame,
            end_frame=end_frame,
            confidence=confidence,
            reason_zh=reason_zh,
            reason_en=reason_en,
            localization_method=method,
        )

        logger.info(
            f"Entity #{entity_id} anomaly segment: "
            f"[{segment.start_time:.2f}s, {segment.end_time:.2f}s] "
            f"confidence={confidence:.2f}"
        )

        return segment

    def localize_all_anomalies(
        self,
        graph: TemporalGraph,
        entity_verdicts: list,
        video_fps: float = 30.0,
        video_duration: float = 0.0,
        video_path: str = "",
        video_name: str = "",
    ) -> LocalizationResult:
        """
        批量定位所有异常实体的片段。

        Args:
            graph:           时间演化图
            entity_verdicts: AuditVerdict 列表 (仅异常实体)
            video_fps:       视频帧率
            video_duration:  视频总时长
            video_path:      原始视频路径 (用于切片)
            video_name:      视频名 (用于输出命名)

        Returns:
            LocalizationResult
        """
        segments = []
        clips_saved = []

        for verdict in entity_verdicts:
            if not verdict.is_anomaly or verdict.is_cinematic_false_alarm:
                continue

            segment = self.localize_anomaly(
                entity_id=verdict.entity_id,
                graph=graph,
                break_timestamp=verdict.break_timestamp,
                confidence=verdict.confidence,
                reason_zh=verdict.reason_zh,
                reason_en=verdict.reason_en,
                video_fps=video_fps,
                video_duration=video_duration,
            )

            if segment is None:
                continue

            # 自动切片
            if self.cfg.save_anomaly_clips and video_path and os.path.isfile(video_path):
                clip_path = self._cut_clip(
                    video_path, segment, video_name, len(segments)
                )
                if clip_path:
                    segment.clip_path = clip_path
                    clips_saved.append(clip_path)

            segments.append(segment)

        # ── 异常区间合并与扩展 ──
        # 多个实体的短片段合并为连续区间，提高帧级 IoU
        segments = self._merge_segments(segments, video_duration)

        total_duration = sum(s.end_time - s.start_time for s in segments)

        return LocalizationResult(
            segments=segments,
            total_anomaly_duration=round(total_duration, 2),
            video_duration=video_duration,
            clips_saved=clips_saved,
        )

    def _merge_segments(
        self,
        segments: list[AnomalySegment],
        video_duration: float,
        merge_gap: float = 10.0,
    ) -> list[AnomalySegment]:
        """
        合并相邻异常片段为连续区间。

        1. 按起始时间排序
        2. 间隔 < merge_gap 秒的片段合并
        3. 合并后的置信度取最大值

        这样可以将多个实体的碎片异常合并为覆盖完整异常过程的区间。
        """
        if len(segments) <= 1:
            return segments

        # 按起始时间排序
        sorted_segs = sorted(segments, key=lambda s: s.start_time)

        merged = []
        current = sorted_segs[0]

        for seg in sorted_segs[1:]:
            # 如果当前片段和下一个片段间隔 < merge_gap，合并
            if seg.start_time <= current.end_time + merge_gap:
                # 扩展区间
                current = AnomalySegment(
                    entity_id=current.entity_id,
                    start_time=current.start_time,
                    end_time=max(current.end_time, seg.end_time),
                    start_frame=current.start_frame,
                    end_frame=max(current.end_frame, seg.end_frame),
                    confidence=max(current.confidence, seg.confidence),
                    reason_zh=current.reason_zh,
                    reason_en=current.reason_en,
                    localization_method="merged",
                )
            else:
                merged.append(current)
                current = seg

        merged.append(current)

        if len(merged) < len(segments):
            logger.info(
                f"Merged {len(segments)} segments → {len(merged)} "
                f"(gap threshold={merge_gap}s)"
            )

        return merged

    # ── Step 1: 断裂点索引 ───────────────────────────────
    def _find_break_index(
        self, path: list[dict], break_timestamp: Optional[float]
    ) -> Optional[int]:
        """找到与 break_timestamp 最近的节点索引"""
        if break_timestamp is None:
            return None

        min_diff = float("inf")
        best_idx = None
        for i, node in enumerate(path):
            diff = abs(node["timestamp"] - break_timestamp)
            if diff < min_diff:
                min_diff = diff
                best_idx = i

        return best_idx

    # ── Step 2: 后向路径溯源 ─────────────────────────────
    def _backward_trace(
        self,
        path: list[dict],
        edges: list[dict],
        break_idx: int,
    ) -> int:
        """
        从断裂点向前回溯，寻找 Embedding/语义显著变化的起始点。

        策略:
          - 检查边的 action_score: 低合理性 = 语义位移
          - 检查边的 energy_avg:   高能量 = 物理变化
          - 在回溯窗口内找到第一个"正常→异常"转折点
        """
        if break_idx <= 0:
            return 0

        window = self.cfg.embedding_shift_window
        search_start = max(0, break_idx - window)

        # 从 break 点向前搜索
        # 找最早的"异常边" (action_score 低于阈值)
        earliest_anomalous_idx = break_idx

        for i in range(break_idx - 1, search_start - 1, -1):
            if i >= len(edges):
                continue

            edge = edges[i]
            action_score = edge.get("action_score", 0.5)
            energy_avg = edge.get("energy_avg", 0.0)

            # 语义位移: 低合理性
            semantic_shift = action_score < (1.0 - self.cfg.embedding_shift_threshold)
            # 物理位移: 高能量
            physical_shift = energy_avg > self.cfg.embedding_shift_threshold

            if semantic_shift or physical_shift:
                earliest_anomalous_idx = i
            else:
                # 找到正常边，停止回溯
                break

        return earliest_anomalous_idx

    # ── Step 3: 物理动能微调 ─────────────────────────────
    def _energy_fine_tune(
        self,
        graph: TemporalGraph,
        start_time_logic: float,
        path: list[dict],
        video_fps: float,
    ) -> float:
        """
        在逻辑起点附近搜索 motion_energy 斜率突变帧。

        在 [start_logic - radius, start_logic + radius] 范围内:
          - 计算 motion_energy 的一阶差分
          - 找到斜率绝对值最大的点
          - 如果该点斜率超过阈值，将其作为物理起始时间
        """
        radius = self.cfg.energy_search_radius_sec
        slope_threshold = self.cfg.energy_slope_threshold

        # 收集逻辑起点附近的能量数据
        t_lo = start_time_logic - radius
        t_hi = start_time_logic + radius

        nearby = []
        for fid, energy in sorted(graph.get_frame_energies().items()):
            t = fid / video_fps if video_fps > 0 else 0.0
            if t_lo <= t <= t_hi:
                nearby.append((t, energy))

        if len(nearby) < 3:
            return start_time_logic

        times = np.array([x[0] for x in nearby])
        energies = np.array([x[1] for x in nearby])

        # 一阶差分
        dt = np.diff(times)
        de = np.diff(energies)
        slopes = np.where(dt > 0, de / dt, 0.0)

        # 找最大斜率突变点
        abs_slopes = np.abs(slopes)
        max_slope_idx = np.argmax(abs_slopes)
        max_slope = abs_slopes[max_slope_idx]

        if max_slope >= slope_threshold:
            # 斜率突变点的时间
            phys_time = float(times[max_slope_idx])
            logger.debug(
                f"Energy slope break at {phys_time:.2f}s "
                f"(slope={max_slope:.4f})"
            )
            return phys_time

        return start_time_logic

    # ── 视频片段切分 ─────────────────────────────────────
    def _cut_clip(
        self,
        video_path: str,
        segment: AnomalySegment,
        video_name: str,
        clip_idx: int,
    ) -> str:
        """
        使用 OpenCV 截取异常视频片段并保存。

        Returns:
            保存的视频文件路径，失败返回空字符串
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV (cv2) not available, skipping clip saving")
            return ""

        # 确定输出目录
        if self.cfg.clip_output_dir:
            out_dir = self.cfg.clip_output_dir
        else:
            out_dir = str(OUTPUT_DIR / video_name / "clips")

        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(
            out_dir,
            f"anomaly_{clip_idx:03d}_e{segment.entity_id}"
            f"_{segment.start_time:.1f}s-{segment.end_time:.1f}s.mp4",
        )

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return ""

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            start_frame = int(segment.start_time * fps)
            end_frame = int(segment.end_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_count = 0
            max_frames = end_frame - start_frame
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                frame_count += 1

            cap.release()
            writer.release()

            if frame_count > 0:
                logger.info(
                    f"Saved anomaly clip: {out_path} "
                    f"({frame_count} frames, {frame_count/fps:.1f}s)"
                )
                return out_path
            else:
                # 删除空文件
                if os.path.exists(out_path):
                    os.remove(out_path)
                return ""

        except Exception as e:
            logger.error(f"Failed to cut clip: {e}")
            return ""

    # ── 格式化输出 ───────────────────────────────────────
    @staticmethod
    def segments_to_dict(segments: list[AnomalySegment]) -> list[dict]:
        """将 AnomalySegment 列表转为可序列化的字典列表"""
        result = []
        for seg in segments:
            # 格式化时间为 mm:ss.f
            def fmt(t):
                mins = int(t // 60)
                secs = t % 60
                return f"{mins:02d}:{secs:04.1f}"

            result.append({
                "entity_id": seg.entity_id,
                "start": fmt(seg.start_time),
                "end": fmt(seg.end_time),
                "start_sec": seg.start_time,
                "end_sec": seg.end_time,
                "confidence": round(seg.confidence, 2),
                "reason": seg.reason_zh or seg.reason_en,
                "clip_path": seg.clip_path,
            })
        return result
