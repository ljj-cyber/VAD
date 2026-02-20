"""
V5 Tube-Skeleton Pipeline — 视频异常检测主流程

三阶段架构:
  Stage 1: 物理追踪骨架 — HybridDetector(帧差+YOLO) + CLIP Crop + EntityTracker
  Stage 2: 稀疏语义挂载 — NodeTrigger + SemanticVLLM (仅关键帧调用)
  Stage 3: 动态图审计   — GraphBuilder + NarrativeGenerator + DecisionAuditor

用法:
  python -m v5.pipeline --video /path/to/video.mp4 [--api-base http://localhost:8000]
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import (
    MotionConfig, CLIPEncoderConfig, TrackerConfig,
    YoloDetectorConfig, HybridDetectorConfig,
    NodeTriggerConfig, SemanticVLLMConfig,
    GraphConfig, NarrativeConfig, DecisionConfig,
    OUTPUT_DIR,
)

from .tracking.motion_extractor import MotionExtractor
from .tracking.clip_encoder import CropCLIPEncoder
from .tracking.entity_tracker import EntityTracker
from .tracking.visual_painter import VisualPainter
from .tracking.multi_frame_stacker import MultiFrameStacker
from .tracking.hybrid_detector import HybridDetector

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

    # Regex: matches filenames like "Salt.2010__#..." or "Deadpool.2.2018__#..."
    _MOVIE_FILENAME_RE = re.compile(
        r'^(?!v=)[A-Z][\w.]+\.\d{4}__#', re.IGNORECASE
    )

    # Scene-context keywords that suggest cinematic/broadcast content
    _CINEMATIC_SCENE_KEYWORDS = frozenset({
        "movie", "film", "cinema", "studio", "stage", "set",
        "broadcast", "tv show", "television", "news studio",
        "sports arena", "basketball court", "soccer field",
        "football field", "stadium", "ice rink", "wrestling ring",
        "boxing ring", "concert", "performance", "theater",
    })

    def __init__(
        self,
        api_base: str = "http://localhost:8000",
        max_workers: int = 16,
        backend: str = "server",
        disable_discordance: bool = False,
        use_yolo: bool = False,
        cinematic_filter: bool = False,
    ):
        self.disable_discordance = disable_discordance
        self.use_yolo = use_yolo
        self.cinematic_filter = cinematic_filter

        # Stage 1 — 始终使用 HybridDetector (帧差 + YOLO 融合)
        #   use_yolo=False → lazy_yolo=True  (帧差优先, YOLO 按需 fallback)
        #   use_yolo=True  → lazy_yolo=False (帧差 + YOLO 全程并行)
        hybrid_cfg = HybridDetectorConfig()
        hybrid_cfg.lazy_yolo = not use_yolo
        if use_yolo:
            logger.info("Using HybridDetector (MotionExtractor + YOLO-World, always-on)")
        else:
            logger.info(
                f"Using HybridDetector (MotionExtractor + YOLO-World lazy fallback, "
                f"streak_threshold={hybrid_cfg.empty_streak_for_yolo_fallback})"
            )
        self.hybrid_detector = HybridDetector(
            motion_cfg=MotionConfig(),
            yolo_cfg=YoloDetectorConfig(),
            hybrid_cfg=hybrid_cfg,
        )
        self.motion_extractor = self.hybrid_detector.motion_extractor

        self.clip_encoder = CropCLIPEncoder(CLIPEncoderConfig())
        self.entity_tracker = EntityTracker(TrackerConfig())
        self.visual_painter = VisualPainter(output_size=(768, 768))
        # ★ 增大 stacker 间隔以减少内存占用 (6→10)
        self.frame_stacker = MultiFrameStacker(buffer_interval_frames=10, grid_size=(768, 768))

        # Stage 2
        self.node_trigger = NodeTrigger(NodeTriggerConfig())
        vllm_cfg = SemanticVLLMConfig()
        vllm_cfg.backend = backend
        vllm_cfg.api_base = api_base
        vllm_cfg.max_workers = max_workers
        self.semantic_vllm = SemanticVLLM(vllm_cfg)
        if not disable_discordance:
            self.discordance_checker = DiscordanceChecker(
                sigma_multiplier=5.0,
                min_energy_threshold=0.20,
                min_excess_ratio=3.5,
                voting_suppress_ratio=0.35,
            )
        else:
            self.discordance_checker = None
            logger.info("Discordance checker DISABLED (ablation mode)")
        # ★ 增大全局心跳间隔以减少 CLIP 编码频率 (2.5→4.0)
        self.global_heartbeat = GlobalHeartbeat(heartbeat_sec=4.0, drift_threshold=0.18)

        # Stage 3
        self.graph_builder = GraphBuilder(GraphConfig())
        self.narrative_gen = NarrativeGenerator(NarrativeConfig())
        self.decision_auditor = DecisionAuditor(
            decision_cfg=DecisionConfig(),
            vllm_cfg=vllm_cfg,
        )

    def reset(self):
        """重置所有模块状态"""
        self.hybrid_detector.reset()
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

            # ── Stage 1: 实体检测 (帧差 + YOLO 融合, YOLO 按需激活) ──
            regions = self.hybrid_detector.detect(frame)
            frame_energy = self.hybrid_detector.compute_frame_energy()

            # 收集帧级全局动能（用于背景校准）
            all_frame_energies.append(frame_energy)

            # 在前 calibration_n 帧收集完毕后，校准 DiscordanceChecker
            if processed == calibration_n and self.discordance_checker is not None:
                self.discordance_checker.calibrate(all_frame_energies[:calibration_n])

            # 记录本帧触发的实体 ID（用于心跳去重）
            triggered_eids_this_frame: set[int] = set()

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
                        triggered_eids_this_frame.add(tr.entity_id)

                    # 4 宫格拼图（多帧时序）
                    if self.frame_stacker.can_make_grid():
                        grid = self.frame_stacker.make_grid(current_frame=frame)
                        if grid is not None:
                            for tr in triggers:
                                grid_images[tr.frame_idx] = grid

                all_triggers.extend(triggers)

            # ── Stage 2: 实体级心跳扫描（不依赖当前帧是否有 regions）──
            # 对当前帧未触发的活跃实体做心跳检查，确保动态图能产生 Edge
            active_eids = self.entity_tracker.get_active_entity_ids()
            if active_eids:
                hb_triggers = self.node_trigger.check_heartbeat_for_active(
                    active_entity_ids=active_eids,
                    current_timestamp=timestamp,
                    current_frame_idx=frame_idx,
                    entity_trace_buffer=entity_trace_buffer,
                    triggered_eids=triggered_eids_this_frame,
                )
                if hb_triggers:
                    for tr in hb_triggers:
                        # 用当前帧全图作为 painted image
                        if tr.frame_idx not in painted_images:
                            painted_images[tr.frame_idx] = \
                                self.visual_painter.paint_fullframe_only(frame)
                        # 尝试生成 grid
                        if tr.frame_idx not in grid_images and self.frame_stacker.can_make_grid():
                            grid = self.frame_stacker.make_grid(current_frame=frame)
                            if grid is not None:
                                grid_images[tr.frame_idx] = grid
                    all_triggers.extend(hb_triggers)

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
        yolo_stats = self.hybrid_detector.yolo_stats
        yolo_info = ""
        if yolo_stats["total_yolo_frames"] > 0:
            yolo_info = (
                f", YOLO fallbacks={yolo_stats['fallback_count']}, "
                f"YOLO frames={yolo_stats['total_yolo_frames']}"
            )
        logger.info(
            f"Stage 1+2 Tracking: {processed} frames processed, "
            f"{len(self.entity_tracker.get_all_entity_ids())} entities, "
            f"{len(all_triggers)} triggers, "
            f"{len(heartbeat_frames)} heartbeats{yolo_info} | "
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
        discordance_alerts = []
        if self.discordance_checker is not None:
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
            if discordance_alerts:
                logger.info(f"Discordance alerts: {len(discordance_alerts)}")
        else:
            logger.info("Discordance checker disabled — skipping contradiction detection")

        # CLIP 漂移信息
        drift_info = {
            "max_drift": self.global_heartbeat.get_max_drift(),
            "heartbeats": len(heartbeat_frames),
        }

        if drift_info["max_drift"] > 0.15:
            logger.info(f"Scene drift detected: max={drift_info['max_drift']:.3f}")

        # ── Stage 3: 构建图 ──
        for sr in semantic_results:
            if sr is None:
                logger.warning("Skipping None semantic result in graph building")
                continue
            eid = sr["entity_id"]
            trace_buf = entity_trace_buffer.get(eid, [])
            self.graph_builder.add_semantic_node(sr, trace_buf)

        # ── Stage 3: Decision Audit ──
        graphs = self.graph_builder.graphs

        # 确定场景类型 (取最常见的 scene_context)
        scene_type = self._detect_scene(graphs)

        # ── 电影场景检测 (cinematic filter) ──
        is_cinematic = False
        cinematic_reason = ""
        if self.cinematic_filter:
            fn_hit = self._detect_cinematic_by_filename(video_path)
            sem_hit, sem_ratio = self._detect_cinematic_by_semantics(semantic_results)
            if fn_hit:
                is_cinematic = True
                cinematic_reason = f"filename_match (ratio={sem_ratio:.2f})"
            elif sem_hit:
                is_cinematic = True
                cinematic_reason = f"semantic_scene (ratio={sem_ratio:.2f})"

            if is_cinematic:
                logger.info(
                    f"Cinematic scene detected: {cinematic_reason} → "
                    f"suppressing anomaly verdicts"
                )

        verdict = self.decision_auditor.audit_video(
            graphs, scene_type,
            discordance_alerts=discordance_alerts,
            drift_info=drift_info,
            entity_trace_buffer=entity_trace_buffer,
            is_cinematic=is_cinematic,
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
                "is_cinematic": is_cinematic if self.cinematic_filter else None,
                "cinematic_reason": cinematic_reason if self.cinematic_filter else None,
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

    @classmethod
    def _detect_cinematic_by_filename(cls, video_path: str) -> bool:
        """Detect movie/film clips via filename pattern (e.g. 'Salt.2010__#...')."""
        stem = Path(video_path).stem
        return bool(cls._MOVIE_FILENAME_RE.match(stem))

    @classmethod
    def _detect_cinematic_by_semantics(
        cls, semantic_results: list[dict], threshold: float = 0.3,
    ) -> tuple[bool, float]:
        """
        Detect cinematic/broadcast content from VLLM scene_context fields.

        Returns (is_cinematic, cinematic_ratio).
        """
        if not semantic_results:
            return False, 0.0

        total = 0
        cinematic_hits = 0
        for sr in semantic_results:
            if sr is None:
                continue
            ctx = str(sr.get("scene_context", "")).lower().strip()
            if not ctx or ctx == "unknown":
                continue
            total += 1
            for kw in cls._CINEMATIC_SCENE_KEYWORDS:
                if kw in ctx:
                    cinematic_hits += 1
                    break

        if total == 0:
            return False, 0.0

        ratio = cinematic_hits / total
        return ratio >= threshold, ratio


# ── CLI 入口 ──────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="V5 Tube-Skeleton Pipeline")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--api-base", default="http://localhost:8000", help="vLLM API base URL")
    parser.add_argument("--backend", default="server", choices=["server", "local"])
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--sample-every", type=int, default=2, help="Process every N-th frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0=all)")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--no-discordance", action="store_true",
                        help="Disable discordance checker (ablation: pure semantic)")
    parser.add_argument("--yolo", action="store_true",
                        help="Enable YOLO-World hybrid detection (motion + semantic)")
    parser.add_argument("--cinematic-filter", action="store_true",
                        help="Enable cinematic/movie scene filter")
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
        disable_discordance=args.no_discordance,
        use_yolo=args.yolo,
        cinematic_filter=args.cinematic_filter,
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
