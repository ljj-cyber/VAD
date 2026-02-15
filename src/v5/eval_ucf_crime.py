"""
V5 Tube-Skeleton Pipeline — UCF-Crime 评估脚本

评估指标:
  1. Frame-level AUC-ROC (核心)
  2. Video-level Accuracy / Precision / Recall / F1
  3. Mean Anomaly Segment IoU

用法:
  python -m v5.eval_ucf_crime --max-videos 20 --sample-every 2 --parallel 3
"""

import argparse
import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np

# ── 日志 ──
_log_dir = "/data/liuzhe/EventVAD/output/v5/eval_ucf_crime"
os.makedirs(_log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{_log_dir}/eval_v5.log", mode="w", encoding="utf-8"),
    ],
)
for mod in ["httpx", "v5.tracking.clip_encoder", "sentence_transformers"]:
    logging.getLogger(mod).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from v5.pipeline import TubeSkeletonPipeline
from v5.config import OUTPUT_DIR


# ── 数据加载 ──────────────────────────────────────────
UCF_ROOT = Path("/data/liuzhe/EventVAD/src/event_seg/videos/ucf_crime")
ANN_FILE = UCF_ROOT / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
SEARCH_DIRS = [
    "Anomaly-Videos-Part-1", "Anomaly-Videos-Part-2",
    "Anomaly-Videos-Part-3", "Anomaly-Videos-Part-4",
    "Testing_Normal_Videos_Anomaly",
]


@dataclass
class VideoAnnotation:
    filename: str
    category: str
    is_anomaly: bool
    intervals: list  # [(start_frame, end_frame), ...]
    filepath: str = ""
    total_frames: int = 0


def load_annotations() -> list[VideoAnnotation]:
    file_index = {}
    for d in SEARCH_DIRS:
        base = UCF_ROOT / d
        if not base.exists():
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".mp4"):
                    file_index[f] = os.path.join(root, f)

    annotations = []
    with open(ANN_FILE) as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            fname, category = parts[0], parts[1]
            s1, e1, s2, e2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])

            intervals = []
            if s1 != -1 and e1 != -1:
                intervals.append((s1, e1))
            if s2 != -1 and e2 != -1:
                intervals.append((s2, e2))

            annotations.append(VideoAnnotation(
                filename=fname,
                category=category,
                is_anomaly=category != "Normal",
                intervals=intervals,
                filepath=file_index.get(fname, ""),
            ))

    logger.info(
        f"Loaded {len(annotations)} annotations "
        f"({sum(a.is_anomaly for a in annotations)} anomaly, "
        f"{sum(not a.is_anomaly for a in annotations)} normal)"
    )
    return annotations


# ── 帧级分数 ──────────────────────────────────────────
def _parse_time(time_str) -> float:
    if isinstance(time_str, (int, float)):
        return float(time_str)
    time_str = str(time_str).strip()
    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    try:
        return float(time_str)
    except ValueError:
        return 0.0


def build_gt_mask(intervals: list, total_frames: int) -> np.ndarray:
    mask = np.zeros(total_frames, dtype=bool)
    for s, e in intervals:
        mask[min(s, total_frames - 1): min(e, total_frames)] = True
    return mask


def compute_frame_scores(
    entity_verdicts: list[dict],
    video_confidence: float,
    total_frames: int,
    fps: float,
) -> np.ndarray:
    """
    将动态图审计的 [T1, T2] 异常结论广播到该区间内的所有帧。

    策略:
      1. 核心区间 [start, end]：所有帧获得满分 (confidence)
      2. 前后渐变区 (ramp)：用线性衰减平滑过渡，避免硬边界
      3. 多个异常实体的分数取 max 叠加
      4. 如果视频被判异常但没有精确区间，给全视频一个低底分
    """
    scores = np.zeros(total_frames, dtype=np.float32)

    ramp_sec = 4.0   # 渐变区秒数（前后各 4 秒线性衰减）

    any_anomaly = False
    for v in entity_verdicts:
        if not v.get("is_anomaly", False):
            continue

        any_anomaly = True
        conf = float(v.get("confidence", video_confidence))
        start_sec = float(v.get("anomaly_start_sec", 0.0))
        end_sec = float(v.get("anomaly_end_sec", 0.0))

        if end_sec <= start_sec:
            # 没有精确区间，跳过（后面会给底分）
            continue

        # ── 核心区间：全部帧满分 ──
        core_sf = int(start_sec * fps)
        core_ef = int(end_sec * fps)
        core_sf = max(0, min(core_sf, total_frames))
        core_ef = max(0, min(core_ef, total_frames))

        if core_sf < core_ef:
            scores[core_sf:core_ef] = np.maximum(
                scores[core_sf:core_ef], conf
            )

        # ── 前渐变区：线性从 0 升到 conf ──
        ramp_frames = int(ramp_sec * fps)
        ramp_sf = max(0, core_sf - ramp_frames)
        if ramp_sf < core_sf:
            n = core_sf - ramp_sf
            ramp_values = np.linspace(conf * 0.15, conf * 0.85, n, dtype=np.float32)
            scores[ramp_sf:core_sf] = np.maximum(scores[ramp_sf:core_sf], ramp_values)

        # ── 后渐变区：线性从 conf 降到 0 ──
        ramp_ef = min(total_frames, core_ef + ramp_frames)
        if core_ef < ramp_ef:
            n = ramp_ef - core_ef
            ramp_values = np.linspace(conf * 0.85, conf * 0.15, n, dtype=np.float32)
            scores[core_ef:ramp_ef] = np.maximum(scores[core_ef:ramp_ef], ramp_values)

    # ── 如果有异常但没有精确区间，给全视频低底分 ──
    if any_anomaly and scores.max() == 0:
        scores[:] = video_confidence * 0.3

    return scores


def compute_frame_iou(
    gt_intervals: list,
    entity_verdicts: list[dict],
    total_frames: int,
    fps: float,
) -> float:
    if total_frames <= 0:
        return 0.0

    gt_mask = build_gt_mask(gt_intervals, total_frames)
    pred_mask = np.zeros(total_frames, dtype=bool)

    for v in entity_verdicts:
        if not v.get("is_anomaly", False):
            continue
        sf = int(float(v.get("anomaly_start_sec", 0)) * fps)
        ef = int(float(v.get("anomaly_end_sec", 0)) * fps)
        pred_mask[max(0, sf): min(ef, total_frames)] = True

    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


# ── 评估主逻辑 ────────────────────────────────────────
def evaluate(
    annotations: list[VideoAnnotation],
    max_videos: int = 0,
    sample_every: int = 2,
    api_base: str = "http://localhost:8000",
    max_workers: int = 16,
    parallel_videos: int = 3,
    balanced: bool = True,
) -> dict:
    if max_videos > 0:
        if balanced:
            # 均衡采样：异常和正常各取一半
            anomaly = [a for a in annotations if a.is_anomaly]
            normal = [a for a in annotations if not a.is_anomaly]
            n_each = max_videos // 2
            annotations = anomaly[:n_each] + normal[:n_each]
            logger.info(f"Balanced sampling: {n_each} anomaly + {n_each} normal = {len(annotations)}")
        else:
            annotations = annotations[:max_videos]

    total = len(annotations)
    logger.info(f"\n{'='*60}")
    logger.info(f"  V5 Tube-Skeleton UCF-Crime Evaluation")
    logger.info(f"  Videos: {total}, sample_every={sample_every}, parallel={parallel_videos}")
    logger.info(f"{'='*60}\n")

    results = []
    results_lock = threading.Lock()
    t_start = time.time()

    def _process_one(idx_ann):
        i, ann = idx_ann
        if not ann.filepath or not Path(ann.filepath).exists():
            return {
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "pred_anomaly": False,
                "pred_score": 0.0, "error": "not found",
            }

        pipe = TubeSkeletonPipeline(
            api_base=api_base,
            max_workers=max(4, max_workers // parallel_videos),
            backend="server",
        )

        try:
            t0 = time.time()
            result = pipe.analyze_video(
                video_path=ann.filepath,
                sample_every_n=sample_every,
            )
            elapsed = time.time() - t0

            verdict = result.get("verdict", {})
            pred_anomaly = verdict.get("is_anomaly", False)
            pred_score = verdict.get("confidence", 0.0)
            entity_verdicts = verdict.get("entity_verdicts", [])

            total_frames = result.get("total_frames", 0)
            fps = result.get("fps", 30.0)

            iou = None
            if ann.is_anomaly and total_frames > 0:
                iou = compute_frame_iou(
                    ann.intervals, entity_verdicts, total_frames, fps
                )

            status = "✅" if pred_anomaly == ann.is_anomaly else "❌"
            iou_str = f"IoU={iou:.3f}" if iou is not None else ""
            logger.info(
                f"[{i+1}/{total}] {status} {ann.filename} "
                f"GT={ann.is_anomaly} Pred={pred_anomaly} "
                f"Score={pred_score:.2f} {iou_str} "
                f"entities={result['stats']['entities']} "
                f"triggers={result['stats']['triggers']} "
                f"{elapsed:.1f}s"
            )

            return {
                "filename": ann.filename,
                "category": ann.category,
                "gt_anomaly": ann.is_anomaly,
                "gt_intervals": ann.intervals,
                "pred_anomaly": pred_anomaly,
                "pred_score": pred_score,
                "entity_verdicts": entity_verdicts,
                "iou": iou,
                "fps": fps,
                "total_frames": total_frames,
                "time_sec": round(elapsed, 1),
                "stats": result.get("stats", {}),
            }
        except Exception as e:
            logger.error(f"[{i+1}/{total}] ❌ {ann.filename}: {e}")
            return {
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "pred_anomaly": False,
                "pred_score": 0.0, "error": str(e),
            }

    # 并行处理
    if parallel_videos > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=parallel_videos) as executor:
            results = list(executor.map(_process_one, enumerate(annotations)))
    else:
        results = [_process_one((i, ann)) for i, ann in enumerate(annotations)]

    total_time = time.time() - t_start

    # ── 计算指标 ──
    metrics = compute_metrics(results)
    metrics["total_time_sec"] = round(total_time, 1)
    metrics["avg_time_per_video"] = round(total_time / max(len(results), 1), 1)
    metrics["sample_every"] = sample_every
    metrics["parallel_videos"] = parallel_videos

    # ── 打印 ──
    print_metrics(metrics)

    # ── 保存 ──
    out_dir = Path(_log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 清理不可序列化的字段
    clean_results = []
    for r in results:
        cr = {k: v for k, v in r.items()}
        clean_results.append(cr)

    with open(out_dir / "results_v5.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": clean_results}, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {out_dir / 'results_v5.json'}")
    return metrics


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}

    correct = sum(1 for r in results if r.get("pred_anomaly") == r.get("gt_anomaly"))
    total = len(results)
    accuracy = correct / total

    tp = sum(1 for r in results if r.get("gt_anomaly") and r.get("pred_anomaly"))
    fn = sum(1 for r in results if r.get("gt_anomaly") and not r.get("pred_anomaly"))
    fp = sum(1 for r in results if not r.get("gt_anomaly") and r.get("pred_anomaly"))
    tn = sum(1 for r in results if not r.get("gt_anomaly") and not r.get("pred_anomaly"))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Frame-level AUC-ROC
    frame_auc = 0.0
    try:
        from sklearn.metrics import roc_auc_score
        all_gt, all_pred = [], []
        for r in results:
            tf = r.get("total_frames", 0)
            if tf <= 0:
                continue
            fps = float(r.get("fps", 30.0) or 30.0)
            gt_mask = build_gt_mask(r.get("gt_intervals", []), tf)
            pred_scores = compute_frame_scores(
                r.get("entity_verdicts", []),
                r.get("pred_score", 0.0),
                tf, fps,
            )
            all_gt.extend(gt_mask.astype(int).tolist())
            all_pred.extend(pred_scores.tolist())

        if len(set(all_gt)) > 1:
            frame_auc = roc_auc_score(all_gt, all_pred)
    except Exception as e:
        logger.warning(f"Frame AUC computation failed: {e}")

    # Video-level AUC
    video_auc = 0.0
    try:
        from sklearn.metrics import roc_auc_score
        vgt = [1 if r.get("gt_anomaly") else 0 for r in results]
        vps = [r.get("pred_score", 0.0) for r in results]
        if len(set(vgt)) > 1:
            video_auc = roc_auc_score(vgt, vps)
    except Exception:
        pass

    # IoU
    ious = [r["iou"] for r in results if r.get("iou") is not None and r["iou"] > 0]
    mean_iou = float(np.mean(ious)) if ious else 0.0

    # Per category
    cat_stats = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0, "ious": []}
        cat_stats[cat]["total"] += 1
        if r.get("pred_anomaly") == r.get("gt_anomaly"):
            cat_stats[cat]["correct"] += 1
        if r.get("iou") is not None:
            cat_stats[cat]["ious"].append(r["iou"])

    for cat, s in cat_stats.items():
        s["accuracy"] = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0
        s["mean_iou"] = round(float(np.mean(s["ious"])), 4) if s["ious"] else 0.0
        del s["ious"]

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "frame_auc": round(frame_auc, 4),
        "video_auc": round(video_auc, 4),
        "mean_iou": round(mean_iou, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "total": total,
        "category_stats": cat_stats,
    }


def print_metrics(metrics: dict):
    print(f"\n{'='*70}")
    print(f"  V5 Tube-Skeleton — UCF-Crime Evaluation Results")
    print(f"{'='*70}")
    print(f"  Video-level Accuracy:  {metrics['accuracy']:.4f} ({metrics['tp']+metrics['tn']}/{metrics['total']})")
    print(f"  Precision:             {metrics['precision']:.4f}")
    print(f"  Recall:                {metrics['recall']:.4f}")
    print(f"  F1 Score:              {metrics['f1']:.4f}")
    print(f"  ★ Frame-level AUC-ROC: {metrics.get('frame_auc', 0):.4f}")
    print(f"  Video-level AUC-ROC:   {metrics.get('video_auc', 0):.4f}")
    print(f"  Mean Anomaly IoU:      {metrics['mean_iou']:.4f}")
    print(f"  TP={metrics['tp']}  FN={metrics['fn']}  FP={metrics['fp']}  TN={metrics['tn']}")
    print(f"  Total time: {metrics.get('total_time_sec', 0)}s "
          f"({metrics.get('avg_time_per_video', 0)}s/video)")
    print(f"\n  Per-category:")
    for cat, s in sorted(metrics.get("category_stats", {}).items()):
        iou_str = f"IoU={s['mean_iou']:.3f}" if s.get("mean_iou") else ""
        print(f"    {cat:<20} acc={s['accuracy']:.2f} ({s['correct']}/{s['total']}) {iou_str}")
    print(f"{'='*70}\n")


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V5 UCF-Crime Evaluation")
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=2,
                        help="Process every N-th frame (default 2)")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--parallel", type=int, default=3,
                        help="Parallel video processing count")
    parser.add_argument("--no-balanced", action="store_true",
                        help="Disable balanced anomaly/normal sampling")
    args = parser.parse_args()

    annotations = load_annotations()
    evaluate(
        annotations,
        max_videos=args.max_videos,
        sample_every=args.sample_every,
        api_base=args.api_base,
        max_workers=args.max_workers,
        parallel_videos=args.parallel,
        balanced=not args.no_balanced,
    )


if __name__ == "__main__":
    main()
