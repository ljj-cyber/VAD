"""
ImageBind Baseline — UCF-Crime 帧级评估脚本

评估指标:
  1. Frame-level AUC-ROC (核心)
  2. Video-level Accuracy / Precision / Recall / F1
  3. Video-level AUC-ROC

用法:
  cd src
  CUDA_VISIBLE_DEVICES=0 python -m baselines.eval_imagebind --max-videos 40 --sample-every 3
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── 日志 ──
_log_base = "/date/liuzhe/EventVAD/EventVAD/output/baselines/imagebind/eval"
_run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_dir = f"{_log_base}/run_{_run_ts}"
os.makedirs(_log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{_log_dir}/eval.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── 数据加载 ──────────────────────────────────────────
UCF_ROOT = Path("/date/liuzhe/EventVAD/EventVAD/src/event_seg/videos/ucf_crime")
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


def stratified_sample(annotations: list[VideoAnnotation], max_videos: int) -> list[VideoAnnotation]:
    """分层采样：异常/正常各半，异常类别尽量均衡覆盖"""
    anomaly = [a for a in annotations if a.is_anomaly]
    normal = [a for a in annotations if not a.is_anomaly]
    n_anomaly = max_videos // 2
    n_normal = max_videos - n_anomaly

    cat_groups: dict[str, list] = defaultdict(list)
    for a in anomaly:
        cat_groups[a.category].append(a)

    selected_anomaly = []
    remaining = n_anomaly
    categories = sorted(cat_groups.keys())

    for cat in categories:
        if remaining <= 0:
            break
        selected_anomaly.append(cat_groups[cat][0])
        remaining -= 1

    if remaining > 0:
        large_cats = sorted(categories, key=lambda c: len(cat_groups[c]), reverse=True)
        idx = 0
        while remaining > 0 and idx < len(large_cats) * 10:
            cat = large_cats[idx % len(large_cats)]
            picked = sum(1 for a in selected_anomaly if a.category == cat)
            if picked < len(cat_groups[cat]):
                selected_anomaly.append(cat_groups[cat][picked])
                remaining -= 1
            idx += 1

    result = selected_anomaly + normal[:n_normal]

    cat_counts = defaultdict(int)
    for a in selected_anomaly:
        cat_counts[a.category] += 1
    cat_str = ", ".join(f"{c}={n}" for c, n in sorted(cat_counts.items()))
    logger.info(
        f"Stratified sampling: {len(selected_anomaly)} anomaly [{cat_str}] "
        f"+ {min(n_normal, len(normal))} normal = {len(result)}"
    )
    return result


# ── 帧级评估 ──────────────────────────────────────────
def build_gt_mask(intervals: list, total_frames: int) -> np.ndarray:
    mask = np.zeros(total_frames, dtype=bool)
    for s, e in intervals:
        mask[min(s, total_frames - 1): min(e, total_frames)] = True
    return mask


def evaluate(
    annotations: list[VideoAnnotation],
    max_videos: int = 40,
    sample_every: int = 3,
    batch_size: int = 32,
    device: str = "cuda",
    threshold: float = 0.5,
):
    from baselines.imagebind_baseline import ImageBindBaseline, ImageBindBaselineConfig

    if max_videos > 0:
        annotations = stratified_sample(annotations, max_videos)

    total = len(annotations)
    logger.info(f"\n{'='*60}")
    logger.info(f"  ImageBind Baseline — UCF-Crime Evaluation")
    logger.info(f"  Videos: {total}, sample_every={sample_every}, batch_size={batch_size}")
    logger.info(f"{'='*60}\n")

    cfg = ImageBindBaselineConfig()
    cfg.batch_size = batch_size
    cfg.device = device
    baseline = ImageBindBaseline(cfg)

    results = []
    all_gt_frames = []
    all_pred_frames = []
    t_start = time.time()

    for i, ann in enumerate(annotations):
        if not ann.filepath or not Path(ann.filepath).exists():
            logger.warning(f"[{i+1}/{total}] SKIP {ann.filename}: file not found")
            results.append({
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "pred_anomaly": False,
                "pred_score": 0.0, "error": "not found",
            })
            continue

        try:
            t0 = time.time()
            result = baseline.analyze_video(
                video_path=ann.filepath,
                sample_every_n=sample_every,
            )
            elapsed = time.time() - t0

            frame_scores = result["frame_scores"]
            total_frames = result["total_frames"]
            video_score = result["video_score"]
            pred_anomaly = video_score >= threshold

            gt_mask = build_gt_mask(ann.intervals, total_frames)
            all_gt_frames.extend(gt_mask.astype(int).tolist())
            all_pred_frames.extend(frame_scores.tolist())

            status = "OK" if pred_anomaly == ann.is_anomaly else "MISS"
            logger.info(
                f"[{i+1}/{total}] {status} {ann.filename} "
                f"GT={ann.is_anomaly} Pred={pred_anomaly} "
                f"Score={video_score:.3f} Mean={frame_scores.mean():.3f} "
                f"{elapsed:.1f}s"
            )

            results.append({
                "filename": ann.filename,
                "category": ann.category,
                "gt_anomaly": ann.is_anomaly,
                "gt_intervals": ann.intervals,
                "pred_anomaly": pred_anomaly,
                "pred_score": float(video_score),
                "mean_score": float(frame_scores.mean()),
                "total_frames": total_frames,
                "processed_frames": result["processed_frames"],
                "time_sec": round(elapsed, 1),
            })

        except Exception as e:
            logger.error(f"[{i+1}/{total}] ERROR {ann.filename}: {e}")
            results.append({
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "pred_anomaly": False,
                "pred_score": 0.0, "error": str(e),
            })

    total_time = time.time() - t_start

    # ── 计算指标 ──
    metrics = compute_metrics(results, all_gt_frames, all_pred_frames)
    metrics["total_time_sec"] = round(total_time, 1)
    metrics["avg_time_per_video"] = round(total_time / max(len(results), 1), 1)
    metrics["sample_every"] = sample_every
    metrics["threshold"] = threshold
    metrics["max_videos"] = max_videos
    metrics["run_timestamp"] = _run_ts

    print_metrics(metrics)

    # ── 保存 ──
    out_dir = Path(_log_dir)
    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)

    latest_link = Path(_log_base) / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(out_dir)
    except OSError:
        pass

    logger.info(f"\nResults saved to {results_path}")
    return metrics


def compute_metrics(
    results: list[dict],
    all_gt_frames: list,
    all_pred_frames: list,
) -> dict:
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
        if len(set(all_gt_frames)) > 1 and len(all_gt_frames) == len(all_pred_frames):
            frame_auc = roc_auc_score(all_gt_frames, all_pred_frames)
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

    # Per-category stats
    cat_stats = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0}
        cat_stats[cat]["total"] += 1
        if r.get("pred_anomaly") == r.get("gt_anomaly"):
            cat_stats[cat]["correct"] += 1
    for s in cat_stats.values():
        s["accuracy"] = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "frame_auc": round(frame_auc, 4),
        "video_auc": round(video_auc, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "total": total,
        "total_gt_frames": len(all_gt_frames),
        "category_stats": cat_stats,
    }


def print_metrics(metrics: dict):
    print(f"\n{'='*70}")
    print(f"  ImageBind Baseline — UCF-Crime Evaluation Results")
    print(f"{'='*70}")
    print(f"  Video-level Accuracy:  {metrics['accuracy']:.4f} ({metrics['tp']+metrics['tn']}/{metrics['total']})")
    print(f"  Precision:             {metrics['precision']:.4f}")
    print(f"  Recall:                {metrics['recall']:.4f}")
    print(f"  F1 Score:              {metrics['f1']:.4f}")
    print(f"  ★ Frame-level AUC-ROC: {metrics.get('frame_auc', 0):.4f}")
    print(f"  Video-level AUC-ROC:   {metrics.get('video_auc', 0):.4f}")
    print(f"  TP={metrics['tp']}  FN={metrics['fn']}  FP={metrics['fp']}  TN={metrics['tn']}")
    print(f"  Total frames evaluated: {metrics.get('total_gt_frames', 0)}")
    print(f"  Total time: {metrics.get('total_time_sec', 0)}s "
          f"({metrics.get('avg_time_per_video', 0)}s/video)")
    print(f"\n  Per-category:")
    for cat, s in sorted(metrics.get("category_stats", {}).items()):
        print(f"    {cat:<20} acc={s['accuracy']:.2f} ({s['correct']}/{s['total']})")
    print(f"{'='*70}\n")


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ImageBind UCF-Crime Evaluation")
    parser.add_argument("--max-videos", type=int, default=40)
    parser.add_argument("--sample-every", type=int, default=3,
                        help="Process every N-th frame")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Video-level anomaly threshold")
    args = parser.parse_args()

    annotations = load_annotations()
    evaluate(
        annotations,
        max_videos=args.max_videos,
        sample_every=args.sample_every,
        batch_size=args.batch_size,
        device=args.device,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
