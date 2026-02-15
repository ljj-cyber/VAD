"""
V4.0 语义因果决策 — 主管线

数据流 (V4.0):
  视频 → 自适应采样 → VLLM 增强感知 → 语义Re-ID → 时间图(含动能)
       → 叙事引擎(因果文本) → Decision LLM 审计 → 精准片段定位 → 结构化报告

  Fallback (V3):
  视频 → 自适应采样 → VLLM 感知 → Re-ID → 时间图 → 加权异常分数

Usage:
    python -m v3.pipeline --video path/to/video.mp4
    python -m v3.pipeline --video path/to/video.mp4 --mode v4
    python -m v3.pipeline --video_dir path/to/videos/ --mode v4
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .config import (
    PerceptionConfig,
    SamplerConfig,
    AssociationConfig,
    AnalysisConfig,
    DecisionConfig,
    NarrativeConfig,
    LocalizationConfig,
    SystemConfig,
    OUTPUT_DIR,
)
from .perception.frame_sampler import AdaptiveFrameSampler
from .perception.vllm_client import VLLMClient
from .association.entity_pool import EntityPool
from .association.temporal_graph import TemporalGraph
from .analysis.anomaly_detector import AnomalyDetector, AnomalyResult, VideoAnomalyResult
from .analysis.temporal_localizer import TemporalLocalizer, AnomalySegment, LocalizationResult
from .utils.result_schema import validate_result

logger = logging.getLogger(__name__)


# ── V4.0 管线核心类 ────────────────────────────────────
class VideoAnomalyPipeline:
    """
    V4.0 端到端视频异常检测管线。

    阶段:
      1. 读取视频，自适应采样关键帧
      2. 对每个关键帧调用 VLLM 提取增强语义快照
         (含 is_cinematic, visual_danger_score)
      3. 通过 Sentence-BERT 实体池进行跨帧 Re-ID
      4. 构建时间演化有向图 (含积分动能边)
      5. 异常检测 (Decision LLM 审计 或 V3 加权 Fallback)
      6. 精准片段定位 (语义回溯 + 动能微调)
      7. 输出结构化报告 (段落 + 解释 + 视频片段)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        save_intermediate: bool = True,
        mode: str = "v4",
        backend: str = "server",
        api_base: str = "http://localhost:8000",
        max_workers: int = 16,
    ):
        """
        Args:
            model_name: VLLM 模型名
            device: 推理设备
            save_intermediate: 是否保存中间结果
            mode: "v4" (Decision LLM) 或 "v3" (加权 Fallback)
            backend: "server" (vLLM API, 推荐) 或 "local" (transformers)
            api_base: vLLM API 地址 (backend=server 时使用)
            max_workers: 并行请求线程数 (backend=server 时使用)
        """
        # 配置
        self.perception_cfg = PerceptionConfig()
        self.sampler_cfg = SamplerConfig()
        self.association_cfg = AssociationConfig()
        self.analysis_cfg = AnalysisConfig()
        self.decision_cfg = DecisionConfig()
        self.narrative_cfg = NarrativeConfig()
        self.localization_cfg = LocalizationConfig()

        self.mode = mode
        self.save_intermediate = save_intermediate

        # 感知模块
        self.vllm_client = VLLMClient(
            model_name=model_name,
            device=device,
            backend=backend,
            api_base=api_base,
            max_workers=max_workers,
        )

        # 每个视频独立的模块 (延迟初始化)
        self.entity_pool: Optional[EntityPool] = None
        self.temporal_graph: Optional[TemporalGraph] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.localizer: Optional[TemporalLocalizer] = None

    # ── 处理单个视频 (V4.0) ──────────────────────────
    def process_video(self, video_path: str) -> dict:
        """
        处理单个视频文件，返回 V4.0 完整分析结果。

        Returns:
            V4.0 结构化结果字典 (包含 anomaly_segments, explanation 等)
        """
        video_path = str(video_path)
        video_name = Path(video_path).stem
        logger.info(f"[V4.0 Pipeline] Processing: {video_path}")
        t_start = time.time()

        # ── Stage 1: 视频读取 ──
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return self._empty_result(video_path)

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        logger.info(
            f"Video: {total_frames} frames, {video_fps:.1f} FPS, "
            f"{duration:.1f}s"
        )

        frames_bgr = self._read_all_frames(cap)
        cap.release()

        if not frames_bgr:
            logger.error("No frames read from video.")
            return self._empty_result(video_path)

        # ── Stage 2: 自适应采样 ──
        sampler = AdaptiveFrameSampler(
            video_fps, len(frames_bgr), self.sampler_cfg
        )
        sampled_indices = sampler.compute_adaptive_schedule(frames_bgr)
        logger.info(f"Sampled {len(sampled_indices)} frames for VLLM.")

        # ── Stage 3: VLLM 增强感知 ──
        snapshots = self._run_perception(frames_bgr, sampled_indices, video_fps)

        # ── Stage 4: Re-ID + 时间图 + 动能注入 ──
        self.entity_pool = EntityPool(self.association_cfg)
        self.temporal_graph = TemporalGraph(self.association_cfg)
        frame_energies = {}

        for snapshot in snapshots:
            frame_id = snapshot["frame_id"]
            timestamp = snapshot["timestamp"]
            entities = snapshot["entities"]
            motion_energy = snapshot.get("motion_energy", 0.0)

            frame_energies[frame_id] = motion_energy

            # Re-ID
            matched = self.entity_pool.match_entities(
                entities, frame_id, timestamp
            )

            # 时间图
            self.temporal_graph.add_frame_snapshot(
                matched, frame_id, timestamp
            )

        # 注入帧级动能到图 (供边计算和定位使用)
        self.temporal_graph.register_frame_energies(frame_energies)

        # ── Stage 5: 异常检测 ──
        scene_type = self._get_dominant_scene_type(snapshots)
        use_decision_llm = (self.mode == "v4")

        self.anomaly_detector = AnomalyDetector(
            cfg=self.analysis_cfg,
            decision_cfg=self.decision_cfg,
            narrative_cfg=self.narrative_cfg,
            vllm_client=self.vllm_client,
        )

        video_result: VideoAnomalyResult = self.anomaly_detector.analyze_video(
            graph=self.temporal_graph,
            scene_type=scene_type,
            frame_energies=frame_energies,
            use_decision_llm=use_decision_llm,
            snapshots=snapshots,
        )

        # ── Stage 6: 精准片段定位 ──
        localization = LocalizationResult()
        if video_result.video_anomaly and video_result.entity_results:
            self.localizer = TemporalLocalizer(self.localization_cfg)

            anomaly_results = [
                r for r in video_result.entity_results if r.is_anomaly
            ]

            # 分离: 有实体路径的 vs 规则兜底的
            entity_results = [r for r in anomaly_results if r.entity_id >= 0]
            rule_results = [r for r in anomaly_results if r.entity_id < 0]

            if entity_results:
            localization = self._localize_anomalies(
                    anomaly_results=entity_results,
                video_fps=video_fps,
                video_duration=duration,
                video_path=video_path,
                video_name=video_name,
            )

            # 规则兜底检出: 用全局动能信号划定异常区间
            if rule_results and not localization.segments:
                from .analysis.temporal_localizer import AnomalySegment
                for r in rule_results:
                    intervals = getattr(r, "_anomaly_intervals", None) or []
                    # 优先使用全局异常区间
                    a_start = getattr(r, '_anomaly_start', None)
                    a_end = getattr(r, '_anomaly_end', None)
                    padding = 1.5

                    candidate_intervals = []
                    if intervals:
                        candidate_intervals.extend(intervals)
                    elif a_start is not None and a_end is not None and a_end > a_start:
                        candidate_intervals.append((a_start, a_end))
                    else:
                        ts = r.break_timestamp or 0.0
                        candidate_intervals.append((max(0.0, ts - 1.5), ts + 1.5))

                    for istart, iend in candidate_intervals:
                        s = max(0.0, float(istart) - padding)
                        e = float(iend) + padding
                        if duration > 0:
                            e = min(duration, e)
                        if e <= s:
                            continue
                        seg = AnomalySegment(
                            entity_id=-1,
                            start_time=round(s, 2),
                            end_time=round(e, 2),
                            start_frame=int(s * video_fps),
                            end_frame=int(e * video_fps),
                            confidence=r.anomaly_score,
                            reason_zh=r.reason_zh,
                            reason_en=r.reason,
                            localization_method="global_energy_interval_topk",
                        )
                        localization.segments.append(seg)

                localization.total_anomaly_duration = sum(
                    seg.end_time - seg.start_time for seg in localization.segments
                )

            # 与实体定位结果统一做一次 merge，避免碎片段拉低 IoU
            if localization.segments and self.localizer is not None:
                localization.segments = self.localizer._merge_segments(
                    localization.segments, duration, merge_gap=4.0
                )
                localization.total_anomaly_duration = sum(
                    seg.end_time - seg.start_time for seg in localization.segments
                )

        # ── Stage 7: 结构化报告 ──
        elapsed = time.time() - t_start
        result = self._build_v4_result(
            video_path=video_path,
            video_name=video_name,
            video_result=video_result,
            localization=localization,
            scene_type=scene_type,
            duration=duration,
            total_frames=total_frames,
            sampled_count=len(sampled_indices),
            elapsed=elapsed,
        )

        # 保存
        if self.save_intermediate:
            self._save_results(video_name, result, snapshots)

        # 日志摘要
        seg_count = len(localization.segments)
        logger.info(
            f"✓ {video_name}: anomaly={video_result.video_anomaly}, "
            f"score={video_result.video_score:.4f}, "
            f"segments={seg_count}, "
            f"entities={self.entity_pool.size}, "
            f"mode={video_result.mode}, "
            f"time={elapsed:.1f}s"
        )

        # 释放
        self.entity_pool = None
        self.temporal_graph = None
        self.anomaly_detector = None
        self.localizer = None

        return result

    # ── 精准片段定位 ──────────────────────────────────
    def _localize_anomalies(
        self,
        anomaly_results: list[AnomalyResult],
        video_fps: float,
        video_duration: float,
        video_path: str,
        video_name: str,
    ) -> LocalizationResult:
        """对所有异常实体进行精准片段定位"""
        segments = []
        clips_saved = []

        for result in anomaly_results:
            seg = self.localizer.localize_anomaly(
                entity_id=result.entity_id,
                graph=self.temporal_graph,
                break_timestamp=result.break_timestamp,
                confidence=result.anomaly_score,
                reason_zh=result.reason_zh,
                reason_en=result.reason,
                video_fps=video_fps,
                video_duration=video_duration,
            )

            if seg is None:
                continue

            # 视频片段切分
            if (
                self.localization_cfg.save_anomaly_clips
                and video_path
                and Path(video_path).is_file()
            ):
                clip_path = self.localizer._cut_clip(
                    video_path, seg, video_name, len(segments)
                )
                if clip_path:
                    seg.clip_path = clip_path
                    clips_saved.append(clip_path)

            segments.append(seg)

        total_dur = sum(s.end_time - s.start_time for s in segments)

        return LocalizationResult(
            segments=segments,
            total_anomaly_duration=round(total_dur, 2),
            video_duration=video_duration,
            clips_saved=clips_saved,
        )

    # ── 构建 V4.0 结构化结果 ─────────────────────────
    def _build_v4_result(
        self,
        video_path: str,
        video_name: str,
        video_result: VideoAnomalyResult,
        localization: LocalizationResult,
        scene_type: str,
        duration: float,
        total_frames: int,
        sampled_count: int,
        elapsed: float,
    ) -> dict:
        """构建 V4.0 最终输出字典"""

        # 异常片段
        segments_dict = TemporalLocalizer.segments_to_dict(
            localization.segments
        )

        # 实体详情
        entity_details = []
        for r in video_result.entity_results:
            detail = {
                "entity_id": r.entity_id,
                "is_anomaly": r.is_anomaly,
                "anomaly_score": round(r.anomaly_score, 4),
                "confidence": round(r.anomaly_score, 4),
                "reason": r.reason,
                "reason_zh": r.reason_zh,
                "mode": r.mode,
            }
            if r.break_timestamp is not None:
                detail["break_timestamp"] = round(r.break_timestamp, 2)
            if r.is_cinematic_false_alarm:
                detail["cinematic_false_alarm"] = True
            if r.violated_contracts:
                detail["violated_contracts"] = r.violated_contracts
            if r.action_sequence:
                detail["action_sequence"] = r.action_sequence[:15]
            # V3 信号分 (Fallback 模式下有值)
            if r.mode == "fallback_v3":
                detail["v3_signals"] = {
                    "path_score": round(r.path_score, 4),
                    "edge_score": round(r.edge_score, 4),
                    "breakage_score": round(r.breakage_score, 4),
                    "energy_score": round(r.energy_score, 4),
                }
                detail["matched_template"] = r.matched_template
            entity_details.append(detail)

        # 电影特征过滤
        cinematic_filter = "Negative (Real-world Event)"
        if video_result.cinematic_filtered:
            cinematic_filter = (
                f"Positive ({len(video_result.cinematic_filtered)} entities "
                f"filtered as cinematic)"
            )

        # 最终结果
        result = {
            # V4.0 核心输出
            "status": (
                "Anomaly Detected"
                if video_result.video_anomaly
                else "Normal"
            ),
            "anomaly_segments": segments_dict,
            "cinematic_filter": cinematic_filter,

            # 解释文本
            "anomaly_explanation": video_result.summary_zh,
            "anomaly_explanation_en": video_result.summary_en,

            # 元数据
            "video_path": video_path,
            "video_name": video_name,
            "anomaly_score": round(video_result.video_score, 4),
            "duration_sec": round(duration, 2),
            "total_frames": total_frames,
            "sampled_frames": sampled_count,
            "num_entities_tracked": (
                self.entity_pool.size if self.entity_pool else 0
            ),
            "scene_type": scene_type,
            "mode": video_result.mode,

            # 实体详情
            "entity_results": entity_details,

            # 定位统计
            "localization": {
                "total_anomaly_duration_sec": localization.total_anomaly_duration,
                "num_segments": len(localization.segments),
                "clips_saved": localization.clips_saved,
            },

            # 图 & 池统计
            "graph_stats": (
                self.temporal_graph.get_stats()
                if self.temporal_graph else {}
            ),
            "pool_stats": (
                self.entity_pool.get_stats()
                if self.entity_pool else {}
            ),

            "processing_time_sec": round(elapsed, 2),
        }

        return result

    # ── 帧读取 ────────────────────────────────────────
    def _read_all_frames(self, cap: cv2.VideoCapture) -> list[np.ndarray]:
        """读取视频所有帧"""
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        logger.info(f"Read {len(frames)} frames from video.")
        return frames

    # ── VLLM 增强感知 ─────────────────────────────────
    def _run_perception(
        self,
        frames_bgr: list[np.ndarray],
        sampled_indices: list[int],
        video_fps: float,
        batch_size: int = 8,
    ) -> list[dict]:
        """
        对采样帧运行 VLLM 推理 (V4.0 增强)。
        使用批量推理加速：一次 forward pass 处理多帧。
        """
        total = len(sampled_indices)
        logger.info(
            f"Running VLLM perception on {total} frames "
            f"(batch_size={batch_size})..."
        )

        # 预处理所有帧: BGR → RGB → PIL → resize
        pil_images = []
        frame_ids = []
        timestamps = []

        for frame_idx in sampled_indices:
            frame_rgb = cv2.cvtColor(frames_bgr[frame_idx], cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image = pil_image.resize(self.perception_cfg.frame_size)
            pil_images.append(pil_image)
            frame_ids.append(frame_idx)
            timestamps.append(frame_idx / video_fps)

        # 批量推理
        t0 = time.time()
        snapshots = self.vllm_client.infer_batch(
            pil_images, frame_ids, timestamps, batch_size=batch_size
        )
        elapsed = time.time() - t0

                logger.info(
            f"  VLLM perception done: {total} frames in {elapsed:.1f}s "
            f"({total / elapsed:.1f} frames/s)"
                )

        return snapshots

    # ── 辅助方法 ──────────────────────────────────────
    @staticmethod
    def _get_dominant_scene_type(snapshots: list[dict]) -> str:
        """从快照中提取最常见的场景类型"""
        scene_counts: dict[str, int] = {}
        for s in snapshots:
            scene = s.get("scene", {})
            stype = scene.get("type", "").strip()
            if stype:
                scene_counts[stype] = scene_counts.get(stype, 0) + 1

        if not scene_counts:
            return ""

        return max(scene_counts, key=scene_counts.get)

    @staticmethod
    def _empty_result(video_path: str) -> dict:
        """返回空结果 (V4.0 格式)"""
        return {
            "status": "Error",
            "anomaly_segments": [],
            "cinematic_filter": "N/A",
            "anomaly_explanation": "视频无法打开或无有效帧。",
            "anomaly_explanation_en": "Video could not be opened or has no valid frames.",
            "video_path": video_path,
            "video_name": Path(video_path).stem,
            "anomaly_score": 0.0,
            "duration_sec": 0.0,
            "total_frames": 0,
            "sampled_frames": 0,
            "num_entities_tracked": 0,
            "scene_type": "",
            "mode": "error",
            "entity_results": [],
            "localization": {
                "total_anomaly_duration_sec": 0.0,
                "num_segments": 0,
                "clips_saved": [],
            },
            "graph_stats": {},
            "pool_stats": {},
            "processing_time_sec": 0.0,
        }

    def _save_results(
        self, video_name: str, result: dict, snapshots: list[dict]
    ):
        """保存 V4.0 分析结果和快照"""
        out_dir = OUTPUT_DIR / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 校验 V4.0 输出 Schema
        is_valid, errors = validate_result(result)
        if not is_valid:
            logger.warning(f"Result schema validation: {errors}")

        # 保存分析结果 (V4.0 格式)
        result_file = out_dir / "analysis_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 保存帧快照
        snapshots_file = out_dir / "frame_snapshots.json"
        with open(snapshots_file, "w", encoding="utf-8") as f:
            json.dump(snapshots, f, indent=2, ensure_ascii=False)

        # 保存实体时间线
        if self.temporal_graph is not None:
            timelines = {}
            for eid in self.temporal_graph.get_all_entity_ids():
                path = self.temporal_graph.get_entity_path(eid)
                edges = self.temporal_graph.get_entity_edges(eid)
                timelines[str(eid)] = {
                    "path": path,
                    "edges": edges,
                }

            timeline_file = out_dir / "entity_timelines.json"
            with open(timeline_file, "w", encoding="utf-8") as f:
                json.dump(
                    timelines, f, indent=2, ensure_ascii=False, default=str
                )

        logger.info(f"Results saved to {out_dir}")

    def cleanup(self):
        """释放所有资源"""
        self.vllm_client.unload()
        if self.anomaly_detector is not None:
            self.anomaly_detector.cleanup()
        self.entity_pool = None
        self.temporal_graph = None
        self.anomaly_detector = None
        self.localizer = None
        logger.info("Pipeline cleanup complete.")


# ── 批量处理 ──────────────────────────────────────────
def process_video_directory(
    video_dir: str,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    mode: str = "v4",
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov"),
) -> list[dict]:
    """
    批量处理目录中的所有视频。

    Returns:
        list of V4.0 result dicts
    """
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        logger.error(f"Not a directory: {video_dir}")
        return []

    video_files = sorted(
        f for f in video_dir.rglob("*") if f.suffix.lower() in extensions
    )
    logger.info(f"Found {len(video_files)} video files in {video_dir}")

    pipeline = VideoAnomalyPipeline(
        model_name=model_name,
        device=device,
        save_intermediate=True,
        mode=mode,
    )

    results = []
    for i, vf in enumerate(video_files):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(video_files)}] Processing: {vf.name}")
        logger.info(f"{'='*60}")

        try:
            result = pipeline.process_video(str(vf))
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {vf.name}: {e}", exc_info=True)
            results.append(VideoAnomalyPipeline._empty_result(str(vf)))

    pipeline.cleanup()

    # 保存汇总
    _save_summary(results, video_dir, mode)

    return results


def _save_summary(results: list[dict], video_dir: Path, mode: str):
    """保存批量处理汇总 (V4.0)"""
    summary_file = OUTPUT_DIR / "batch_summary.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    anomaly_count = sum(1 for r in results if r.get("status") == "Anomaly Detected")
    total_segments = sum(
        len(r.get("anomaly_segments", [])) for r in results
    )

    summary = {
        "source_dir": str(video_dir),
        "mode": mode,
        "total_videos": len(results),
        "anomaly_videos": anomaly_count,
        "total_anomaly_segments": total_segments,
        "avg_anomaly_score": round(
            np.mean([r.get("anomaly_score", 0) for r in results]), 4
        ) if results else 0.0,
        "videos": [
            {
                "name": r.get("video_name", ""),
                "status": r.get("status", "Error"),
                "anomaly_score": r.get("anomaly_score", 0),
                "segments": len(r.get("anomaly_segments", [])),
                "entities": r.get("num_entities_tracked", 0),
                "scene": r.get("scene_type", ""),
                "mode": r.get("mode", ""),
                "time_sec": r.get("processing_time_sec", 0),
            }
            for r in results
        ],
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 控制台汇总
    logger.info(f"\n{'='*60}")
    logger.info("BATCH SUMMARY (V4.0)")
    logger.info(f"{'='*60}")
    logger.info(f"Total videos: {len(results)}")
    logger.info(f"Anomaly videos: {anomaly_count}")
    logger.info(f"Total segments: {total_segments}")
    logger.info(f"Mode: {mode}")

    if results:
        scores = [r.get("anomaly_score", 0) for r in results]
        logger.info(
            f"Anomaly score — mean: {np.mean(scores):.4f}, "
            f"max: {np.max(scores):.4f}, min: {np.min(scores):.4f}"
        )

        # Top-5 最异常视频
        sorted_results = sorted(
            results, key=lambda r: r.get("anomaly_score", 0), reverse=True
        )
        logger.info("\nTop-5 anomalous videos:")
        for r in sorted_results[:5]:
            seg_count = len(r.get("anomaly_segments", []))
            logger.info(
                f"  [{r.get('anomaly_score', 0):.4f}] {r.get('video_name', '?')} "
                f"({r.get('num_entities_tracked', 0)} entities, "
                f"{seg_count} segments)"
            )

    logger.info(f"\nSummary saved to {summary_file}")


# ── CLI 入口 ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="V4.0 Semantic Causal Decision — Video Anomaly Detection"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a single video file",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Path to a directory of video files",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["qwen2-vl-7b", "moondream2"],
        help="VLLM model name (default: qwen2-vl-7b)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (e.g. 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--mode", type=str, default="v4",
        choices=["v4", "v3"],
        help="Detection mode: 'v4' (Decision LLM) or 'v3' (weighted fallback)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable saving intermediate results",
    )
    parser.add_argument(
        "--no-clips", action="store_true",
        help="Disable saving anomaly video clips",
    )
    parser.add_argument(
        "--backend", type=str, default="server",
        choices=["server", "local"],
        help="Inference backend: 'server' (vLLM API, fast) or 'local' (transformers)",
    )
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000",
        help="vLLM API server URL (for --backend server)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=16,
        help="Max parallel request threads (for --backend server)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    # 日志配置
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.video:
        # 单视频模式
        pipeline = VideoAnomalyPipeline(
            model_name=args.model,
            device=args.device,
            save_intermediate=not args.no_save,
            mode=args.mode,
            backend=args.backend,
            api_base=args.api_base,
            max_workers=args.max_workers,
        )

        if args.no_clips:
            pipeline.localization_cfg.save_anomaly_clips = False

        result = pipeline.process_video(args.video)
        pipeline.cleanup()

        # 打印 V4.0 结果
        _print_v4_result(result)

    elif args.video_dir:
        # 批量模式
        process_video_directory(
            args.video_dir,
            model_name=args.model,
            device=args.device,
            mode=args.mode,
        )

    else:
        parser.error("Must specify either --video or --video_dir")


def _print_v4_result(result: dict):
    """在控制台打印 V4.0 结果"""
    print(f"\n{'='*60}")
    print(f"V4.0 ANALYSIS RESULT")
    print(f"{'='*60}")
    print(f"Video:    {result.get('video_name', '?')}")
    print(f"Status:   {result.get('status', '?')}")
    print(f"Score:    {result.get('anomaly_score', 0):.4f}")
    print(f"Mode:     {result.get('mode', '?')}")
    print(f"Scene:    {result.get('scene_type', 'N/A')}")
    print(f"Entities: {result.get('num_entities_tracked', 0)}")
    print(f"Time:     {result.get('processing_time_sec', 0):.1f}s")
    print(f"Cinematic Filter: {result.get('cinematic_filter', 'N/A')}")

    segments = result.get("anomaly_segments", [])
    if segments:
        print(f"\nAnomaly Segments ({len(segments)}):")
        for seg in segments:
            print(
                f"  [{seg.get('start', '?')} ~ {seg.get('end', '?')}] "
                f"conf={seg.get('confidence', 0):.2f}"
            )
            reason = seg.get("reason", "")
            if reason:
                print(f"    Reason: {reason}")
            clip = seg.get("clip_path", "")
            if clip:
                print(f"    Clip: {clip}")

    explanation = result.get("anomaly_explanation", "")
    if explanation:
        print(f"\nExplanation (zh): {explanation}")

    entities = result.get("entity_results", [])
    if entities:
        print(f"\nEntity Details (top 5):")
        for er in entities[:5]:
            prefix = "⚠️" if er.get("is_anomaly") else "✓"
            print(
                f"  {prefix} Entity #{er.get('entity_id', '?')}: "
                f"score={er.get('anomaly_score', 0):.4f} "
                f"[{er.get('mode', '?')}]"
            )
            r = er.get("reason_zh") or er.get("reason", "")
            if r:
                print(f"     {r}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
