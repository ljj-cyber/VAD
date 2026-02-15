"""
V5 Tube-Skeleton Pipeline — 视频异常检测主流程

三阶段架构:
  Stage 1: 物理追踪骨架 — MotionExtractor + CLIP Crop + EntityTracker
  Stage 2: 稀疏语义挂载 — NodeTrigger + SemanticVLLM (仅关键帧调用)
  Stage 3: 动态图审计   — GraphBuilder + NarrativeGenerator + DecisionAuditor

用法:
  python -m v5.pipeline --video /path/to/video.mp4 [--api-base http://localhost:8000]
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import (
    MotionConfig, CLIPEncoderConfig, TrackerConfig,
    NodeTriggerConfig, SemanticVLLMConfig,
    GraphConfig, NarrativeConfig, DecisionConfig,
    OUTPUT_DIR,
)

from .tracking.motion_extractor import MotionExtractor
from .tracking.clip_encoder import CropCLIPEncoder
from .tracking.entity_tracker import EntityTracker
from .tracking.visual_painter import VisualPainter
from .tracking.multi_frame_stacker import MultiFrameStacker

from .semantic.node_trigger import NodeTrigger
from .semantic.vllm_semantic import SemanticVLLM
from .semantic.discordance_checker import DiscordanceChecker
from .semantic.global_heartbeat import GlobalHeartbeat

from .graph.graph_builder import GraphBuilder
from .graph.narrative_generator import NarrativeGenerator
from .graph.decision_prompt import DecisionAuditor, VideoVerdict

logger = logging.getLogger(__name__)


class TubeSkeletonPipeline:
    """
    V5 管线。

    输入: 视频路径
    输出: VideoVerdict (异常判定 + 异常区间 + 叙事解释)
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000",
        max_workers: int = 16,
        backend: str = "server",
    ):
        # Stage 1
        self.motion_extractor = MotionExtractor(MotionConfig())
        self.clip_encoder = CropCLIPEncoder(CLIPEncoderConfig())
        self.entity_tracker = EntityTracker(TrackerConfig())
        self.visual_painter = VisualPainter(output_size=(768, 768))
        self.frame_stacker = MultiFrameStacker(buffer_interval_frames=6, grid_size=(768, 768))

        # Stage 2
        self.node_trigger = NodeTrigger(NodeTriggerConfig())
        vllm_cfg = SemanticVLLMConfig()
        vllm_cfg.backend = backend
        vllm_cfg.api_base = api_base
        vllm_cfg.max_workers = max_workers
        self.semantic_vllm = SemanticVLLM(vllm_cfg)
        self.discordance_checker = DiscordanceChecker(
            sigma_multiplier=5.0,
            min_energy_threshold=0.15,
            min_excess_ratio=2.5,
            voting_suppress_ratio=0.5,
        )
        self.global_heartbeat = GlobalHeartbeat(heartbeat_sec=2.5, drift_threshold=0.18)

        # Stage 3
        self.graph_builder = GraphBuilder(GraphConfig())
        self.narrative_gen = NarrativeGenerator(NarrativeConfig())
        self.decision_auditor = DecisionAuditor(
            decision_cfg=DecisionConfig(),
            vllm_cfg=vllm_cfg,
        )

    def reset(self):
        """重置所有模块状态"""
        self.motion_extractor.reset()
        self.entity_tracker.reset()
        self.node_trigger.reset()
        self.graph_builder.reset()
        self.global_heartbeat.reset()
        self.frame_stacker.reset()

    def analyze_video(
        self,
        video_path: str,
        sample_every_n: int = 1,
        max_frames: int = 0,
    ) -> dict:
        """
        分析视频。

        Args:
            video_path: 视频文件路径
            sample_every_n: 每 N 帧处理一帧 (加速)
            max_frames: 最大处理帧数 (0=不限)

        Returns:
            dict: 包含 verdict, graphs, trace_log, timing 等
        """
        self.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / max(fps, 1e-6)

        logger.info(
            f"Processing: {video_path} | "
            f"{total_frames} frames, {fps:.1f} fps, {video_duration:.1f}s | "
            f"sample_every={sample_every_n}"
        )

        t_start = time.time()

        # ── 背景动能校准（前 calibration_frames 帧） ──
        calibration_n = 30  # 用前 30 帧校准背景动能
        bg_energies: list[float] = []

        # ── Stage 1 + 2: 逐帧处理 ──
        frame_idx = 0
        processed = 0
        all_triggers = []
        # 缓存每个实体在相邻 trigger 之间的 trace entries
        entity_trace_buffer: dict[int, list] = {}
        # 缓存带框全图 (frame_idx → PIL.Image)
        painted_images: dict[int, object] = {}
        # 缓存 4 宫格拼图 (frame_idx → PIL.Image)
        grid_images: dict[int, object] = {}
        # 全局心跳触发的帧列表
        heartbeat_frames: list[dict] = []
        # 所有帧的全局动能（用于校准）
        all_frame_energies: list[float] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and processed >= max_frames:
                break

            if frame_idx % sample_every_n != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / max(fps, 1e-6)

            # ── Stage 1: 动能 + 连通域 ──
            regions = self.motion_extractor.extract(frame)

            # 收集帧级全局动能（用于背景校准）
            frame_energy = self.motion_extractor.compute_frame_energy(frame)
            all_frame_energies.append(frame_energy)

            # 在前 calibration_n 帧收集完毕后，校准 DiscordanceChecker
            if processed == calibration_n:
                self.discordance_checker.calibrate(all_frame_energies[:calibration_n])

            if regions:
                # CLIP encode crops (用于 EntityTracker 匹配)
                crops = [r.crop_image for r in regions]
                embeddings = self.clip_encoder.encode_crops(crops)

                # Entity tracking
                entries = self.entity_tracker.update(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    regions=regions,
                    embeddings=embeddings,
                    keep_crops=True,
                )

                # 缓存 trace entries
                for entry in entries:
                    eid = entry.entity_id
                    if eid not in entity_trace_buffer:
                        entity_trace_buffer[eid] = []
                    entity_trace_buffer[eid].append(entry)

                # ── Stage 2: NodeTrigger ──
                triggers = self.node_trigger.check_triggers(entries)

                if triggers:
                    # 带框全图
                    entity_ids = [e.entity_id for e in entries]
                    painted = self.visual_painter.paint(frame, regions, entity_ids)
                    for tr in triggers:
                        painted_images[tr.frame_idx] = painted

                    # 4 宫格拼图（多帧时序）
                    if self.frame_stacker.can_make_grid():
                        grid = self.frame_stacker.make_grid(current_frame=frame)
                        if grid is not None:
                            for tr in triggers:
                                grid_images[tr.frame_idx] = grid

                all_triggers.extend(triggers)

            # 每帧都推入 stacker（不论有无 region）
            self.frame_stacker.push(frame_idx, frame)

            # ── Stage 2: 全局心跳 (不依赖帧差) ──
            if self.global_heartbeat.should_heartbeat(timestamp):
                # 全图 CLIP embedding
                fullframe_pil = self.visual_painter.paint_fullframe_only(frame)
                fullframe_emb = self.clip_encoder.encode_single(
                    np.array(fullframe_pil)[:, :, ::-1].copy()  # RGB→BGR
                )
                hb_result = self.global_heartbeat.update(
                    frame_idx, timestamp, fullframe_emb
                )
                if hb_result:
                    heartbeat_frames.append({
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "trigger_reason": hb_result.trigger_reason,
                        "drift": hb_result.drift_from_baseline,
                    })
                    # 如果是漂移触发且没有实体 trigger，
                    # 也生成一个全图 painted_images 供 VLLM 用
                    if hb_result.trigger_reason == "drift" and frame_idx not in painted_images:
                        painted_images[frame_idx] = fullframe_pil

            processed += 1
            frame_idx += 1

        cap.release()

        t_tracking = time.time()
        logger.info(
            f"Stage 1+2 Tracking: {processed} frames processed, "
            f"{len(self.entity_tracker.get_all_entity_ids())} entities, "
            f"{len(all_triggers)} triggers, "
            f"{len(heartbeat_frames)} heartbeats | "
            f"{t_tracking - t_start:.1f}s"
        )

        # ── Stage 2: 批量 VLLM 语义推理 (带框全图 + 多帧拼图) ──
        semantic_results = self.semantic_vllm.infer_triggers(
            all_triggers,
            painted_images=painted_images,
            grid_images=grid_images,
        )

        t_semantic = time.time()
        logger.info(
            f"Stage 2 Semantic: {len(semantic_results)} nodes | "
            f"{t_semantic - t_tracking:.1f}s"
        )

        # ── Stage 2: 矛盾检测 ──
        entity_trace_energies: dict[int, list[float]] = {}
        entity_trace_time_energy: dict[int, list[tuple[float, float]]] = {}
        for eid, entries in entity_trace_buffer.items():
            entity_trace_energies[eid] = [e.kinetic_energy for e in entries]
            entity_trace_time_energy[eid] = [
                (e.timestamp, e.kinetic_energy) for e in entries
            ]

        discordance_alerts = self.discordance_checker.check_video(
            semantic_results, entity_trace_energies,
            entity_trace_time_energy=entity_trace_time_energy,
        )

        # CLIP 漂移信息
        drift_info = {
            "max_drift": self.global_heartbeat.get_max_drift(),
            "heartbeats": len(heartbeat_frames),
        }

        if discordance_alerts:
            logger.info(f"Discordance alerts: {len(discordance_alerts)}")
        if drift_info["max_drift"] > 0.15:
            logger.info(f"Scene drift detected: max={drift_info['max_drift']:.3f}")

        # ── Stage 3: 构建图 ──
        for sr in semantic_results:
            eid = sr["entity_id"]
            trace_buf = entity_trace_buffer.get(eid, [])
            self.graph_builder.add_semantic_node(sr, trace_buf)

        # ── Stage 3: Decision Audit ──
        graphs = self.graph_builder.graphs

        # 确定场景类型 (取最常见的 scene_context)
        scene_type = self._detect_scene(graphs)

        verdict = self.decision_auditor.audit_video(
            graphs, scene_type,
            discordance_alerts=discordance_alerts,
            drift_info=drift_info,
        )

        t_end = time.time()
        logger.info(
            f"Stage 3 Decision: anomaly={verdict.is_anomaly}, "
            f"conf={verdict.confidence:.2f} | {t_end - t_semantic:.1f}s"
        )
        logger.info(
            f"Total: {t_end - t_start:.1f}s | "
            f"Verdict: {'ANOMALY' if verdict.is_anomaly else 'NORMAL'} "
            f"({verdict.confidence:.2f})"
        )

        # ── 构建 danger_timeline: 所有语义节点的 (timestamp, danger_score) ──
        danger_timeline = []
        for sr in semantic_results:
            danger_timeline.append({
                "timestamp": round(sr.get("timestamp", 0.0), 3),
                "danger_score": round(sr.get("danger_score", 0.0), 4),
                "is_suspicious": sr.get("is_suspicious", False),
                "entity_id": sr.get("entity_id", -1),
                "action": sr.get("action", "unknown"),
            })
        danger_timeline.sort(key=lambda x: x["timestamp"])

        # ── 构建帧级动能时间线 (下采样到每秒) ──
        energy_per_sec = []
        if all_frame_energies:
            sec_bins = int(video_duration) + 1
            for s in range(sec_bins):
                frame_start = int(s * fps / sample_every_n)
                frame_end = int((s + 1) * fps / sample_every_n)
                bin_energies = all_frame_energies[frame_start:frame_end]
                if bin_energies:
                    energy_per_sec.append(round(max(bin_energies), 4))
                else:
                    energy_per_sec.append(0.0)

        # ── 心跳漂移时间线 ──
        drift_timeline = self.global_heartbeat.get_drift_timeline()

        # ── 构建输出 ──
        result = {
            "video_path": str(video_path),
            "video_duration_sec": round(video_duration, 2),
            "total_frames": total_frames,
            "processed_frames": processed,
            "fps": round(fps, 2),
            "verdict": {
                "is_anomaly": verdict.is_anomaly,
                "confidence": round(verdict.confidence, 3),
                "anomaly_entity_ids": verdict.anomaly_entity_ids,
                "scene_type": scene_type,
                "summary": verdict.summary,
                "entity_verdicts": [
                    {
                        "entity_id": v.entity_id,
                        "is_anomaly": v.is_anomaly,
                        "confidence": round(v.confidence, 3),
                        "break_timestamp": v.break_timestamp,
                        "reason": v.reason,
                        "anomaly_start_sec": round(v.anomaly_start_sec, 2),
                        "anomaly_end_sec": round(v.anomaly_end_sec, 2),
                    }
                    for v in verdict.entity_verdicts
                ],
            },
            "temporal_signals": {
                "danger_timeline": danger_timeline,
                "energy_per_sec": energy_per_sec,
                "drift_timeline": [
                    {"timestamp": round(t, 3), "drift": round(d, 4)}
                    for t, d in drift_timeline
                ],
                "heartbeat_frames": heartbeat_frames,
            },
            "timing": {
                "tracking_sec": round(t_tracking - t_start, 2),
                "semantic_sec": round(t_semantic - t_tracking, 2),
                "decision_sec": round(t_end - t_semantic, 2),
                "total_sec": round(t_end - t_start, 2),
            },
            "stats": {
                "entities": len(graphs),
                "triggers": len(all_triggers),
                "nodes": sum(g.num_nodes for g in graphs.values()),
                "edges": sum(g.num_edges for g in graphs.values()),
            },
            "graphs": self.graph_builder.export_all(),
            "trace_log": self.entity_tracker.export_trace_log(),
        }

        return result

    @staticmethod
    def _detect_scene(graphs: dict) -> str:
        """从图中检测最常见的场景类型"""
        from collections import Counter
        scenes = Counter()
        for g in graphs.values():
            for n in g.nodes:
                if n.scene_context and n.scene_context != "unknown":
                    scenes[n.scene_context] += 1

        if scenes:
            return scenes.most_common(1)[0][0]
        return "unknown"


# ── CLI 入口 ──────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="V5 Tube-Skeleton Pipeline")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--api-base", default="http://localhost:8000", help="vLLM API base URL")
    parser.add_argument("--backend", default="server", choices=["server", "local"])
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--sample-every", type=int, default=2, help="Process every N-th frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0=all)")
    parser.add_argument("--output", default="", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = TubeSkeletonPipeline(
        api_base=args.api_base,
        max_workers=args.max_workers,
        backend=args.backend,
    )

    result = pipeline.analyze_video(
        video_path=args.video,
        sample_every_n=args.sample_every,
        max_frames=args.max_frames,
    )

    # 保存结果
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = OUTPUT_DIR / Path(args.video).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # trace_log 可能很大，不保存 embedding
    for entry in result.get("trace_log", []):
        entry.pop("embedding", None)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResult saved to: {out_path}")
    print(f"Verdict: {'ANOMALY' if result['verdict']['is_anomaly'] else 'NORMAL'}")
    print(f"Confidence: {result['verdict']['confidence']}")
    print(f"Summary: {result['verdict']['summary']}")


if __name__ == "__main__":
    main()
