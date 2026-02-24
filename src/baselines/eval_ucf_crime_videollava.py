"""
Video-LLaVA Baseline — UCF-Crime 评估脚本

评估指标:
  1. Frame-level AUC-ROC / AP / Max-F1 / EER
  2. Video-level AUC-ROC / Accuracy / Precision / Recall / F1

用法:
  conda activate videollava
  cd /date/liuzhe/EventVAD/EventVAD/src
  CUDA_VISIBLE_DEVICES=1 python -m baselines.eval_ucf_crime_videollava --max-videos 50
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
_log_base = "/date/liuzhe/EventVAD/EventVAD/output/baselines/eval_ucf_crime_videollava"
_run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_dir = f"{_log_base}/run_{_run_ts}"
os.makedirs(_log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{_log_dir}/eval_videollava.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── 数据加载 (同 baselines/eval_ucf_crime.py) ─────────
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
    intervals: list
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


# ── 评估工具 ──────────────────────────────────────────
def build_gt_mask(intervals: list, total_frames: int) -> np.ndarray:
    mask = np.zeros(total_frames, dtype=bool)
    for s, e in intervals:
        mask[min(s, total_frames - 1): min(e, total_frames)] = True
    return mask


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

    # Frame-level metrics
    frame_auc = 0.0
    frame_ap = 0.0
    frame_eer = 1.0
    frame_max_f1 = 0.0

    try:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            roc_curve, precision_recall_curve,
        )
        all_gt, all_pred = [], []
        for r in results:
            tf = r.get("total_frames", 0)
            if tf <= 0:
                continue
            gt_mask = build_gt_mask(r.get("gt_intervals", []), tf)
            pred_scores = r.get("frame_scores_array", np.zeros(tf))
            if len(pred_scores) != tf:
                pred_scores = np.interp(
                    np.arange(tf), np.linspace(0, tf - 1, len(pred_scores)), pred_scores
                )
            all_gt.extend(gt_mask.astype(int).tolist())
            all_pred.extend(pred_scores.tolist())

        if len(set(all_gt)) > 1:
            all_gt_np = np.array(all_gt)
            all_pred_np = np.array(all_pred)

            frame_auc = roc_auc_score(all_gt_np, all_pred_np)
            frame_ap = average_precision_score(all_gt_np, all_pred_np)

            fpr, tpr, _ = roc_curve(all_gt_np, all_pred_np)
            fnr = 1.0 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            frame_eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

            prec_arr, rec_arr, _ = precision_recall_curve(all_gt_np, all_pred_np)
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr)
            f1_arr = np.nan_to_num(f1_arr, nan=0.0)
            frame_max_f1 = float(f1_arr.max())

        logger.info(
            f"Frame-level stats: {len(all_gt)} total frames, "
            f"{sum(all_gt)} anomaly frames "
            f"({100*sum(all_gt)/max(len(all_gt),1):.1f}%)"
        )
    except Exception as e:
        logger.warning(f"Frame-level metric computation failed: {e}")

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

    # Per-category
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
        "frame_ap": round(frame_ap, 4),
        "frame_eer": round(frame_eer, 4),
        "frame_max_f1": round(frame_max_f1, 4),
        "video_auc": round(video_auc, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "total": total,
        "category_stats": cat_stats,
    }


def print_metrics(metrics: dict):
    print(f"\n{'='*70}")
    print(f"  Video-LLaVA Baseline — UCF-Crime Evaluation Results")
    print(f"{'='*70}")
    print(f"  Video-level Accuracy:  {metrics['accuracy']:.4f} ({metrics['tp']+metrics['tn']}/{metrics['total']})")
    print(f"  Precision:             {metrics['precision']:.4f}")
    print(f"  Recall:                {metrics['recall']:.4f}")
    print(f"  F1 Score:              {metrics['f1']:.4f}")
    print(f"  ★ Frame AUC-ROC:       {metrics.get('frame_auc', 0):.4f}")
    print(f"  ★ Frame AP:            {metrics.get('frame_ap', 0):.4f}")
    print(f"  ★ Frame Max-F1:        {metrics.get('frame_max_f1', 0):.4f}")
    print(f"  ★ Frame EER:           {metrics.get('frame_eer', 1):.4f}")
    print(f"  Video-level AUC-ROC:   {metrics.get('video_auc', 0):.4f}")
    print(f"  TP={metrics['tp']}  FN={metrics['fn']}  FP={metrics['fp']}  TN={metrics['tn']}")
    print(f"  Total time: {metrics.get('total_time_sec', 0)}s "
          f"({metrics.get('avg_time_per_video', 0)}s/video)")
    print(f"\n  Per-category:")
    for cat, s in sorted(metrics.get("category_stats", {}).items()):
        print(f"    {cat:<20} acc={s['accuracy']:.2f} ({s['correct']}/{s['total']})")
    print(f"{'='*70}\n")


# ── 评估主逻辑 ────────────────────────────────────────
def evaluate(annotations: list[VideoAnnotation], args) -> dict:
    max_videos = args.max_videos
    sample_every = args.sample_every
    balanced = not args.no_balanced
    video_threshold = args.threshold

    if max_videos > 0:
        if balanced:
            anomaly = [a for a in annotations if a.is_anomaly]
            normal = [a for a in annotations if not a.is_anomaly]
            n_anomaly = max_videos // 2
            n_normal = max_videos - n_anomaly

            cat_groups: dict[str, list] = defaultdict(list)
            for a in anomaly:
                cat_groups[a.category].append(a)

            selected_anomaly = []
            remaining_slots = n_anomaly
            categories = sorted(cat_groups.keys())

            for cat in categories:
                if remaining_slots <= 0:
                    break
                selected_anomaly.append(cat_groups[cat][0])
                remaining_slots -= 1

            if remaining_slots > 0:
                large_cats = sorted(categories, key=lambda c: len(cat_groups[c]), reverse=True)
                idx = 0
                while remaining_slots > 0 and idx < len(large_cats) * 10:
                    cat = large_cats[idx % len(large_cats)]
                    picked = sum(1 for a in selected_anomaly if a.category == cat)
                    if picked < len(cat_groups[cat]):
                        selected_anomaly.append(cat_groups[cat][picked])
                        remaining_slots -= 1
                    idx += 1

            annotations = selected_anomaly + normal[:n_normal]

            cat_counts = defaultdict(int)
            for a in selected_anomaly:
                cat_counts[a.category] += 1
            cat_str = ", ".join(f"{c}={n}" for c, n in sorted(cat_counts.items()))
            logger.info(
                f"Stratified sampling: {len(selected_anomaly)} anomaly [{cat_str}] "
                f"+ {min(n_normal, len(normal))} normal = {len(annotations)}"
            )
        else:
            annotations = annotations[:max_videos]

    total = len(annotations)
    logger.info(f"\n{'='*60}")
    logger.info(f"  Video-LLaVA Baseline — UCF-Crime Evaluation")
    logger.info(f"  Videos: {total}, sample_every={sample_every}, threshold={video_threshold}")
    logger.info(f"{'='*60}\n")

    from baselines.videollava_baseline import VideoLLaVABaseline, VideoLLaVAConfig
    cfg = VideoLLaVAConfig()
    cfg.model_path = args.model_path
    cfg.segment_sec = args.segment_sec
    baseline = VideoLLaVABaseline(cfg)

    results = []
    t_start = time.time()

    for i, ann in enumerate(annotations):
        if not ann.filepath or not Path(ann.filepath).exists():
            logger.warning(f"[{i+1}/{total}] SKIP {ann.filename}: not found")
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

            video_score = result["video_score"]
            pred_anomaly = video_score >= video_threshold

            status = "OK" if pred_anomaly == ann.is_anomaly else "XX"
            logger.info(
                f"[{i+1}/{total}] {status} {ann.filename} "
                f"GT={ann.is_anomaly} Pred={pred_anomaly} "
                f"Score={video_score:.3f} Segs={result['processed_segments']} "
                f"{elapsed:.1f}s"
            )

            results.append({
                "filename": ann.filename,
                "category": ann.category,
                "gt_anomaly": ann.is_anomaly,
                "gt_intervals": ann.intervals,
                "pred_anomaly": pred_anomaly,
                "pred_score": video_score,
                "frame_scores_array": result["frame_scores"],
                "total_frames": result["total_frames"],
                "fps": result["fps"],
                "processed_segments": result["processed_segments"],
                "time_sec": round(elapsed, 1),
            })
        except Exception as e:
            logger.error(f"[{i+1}/{total}] XX {ann.filename}: {e}", exc_info=True)
            results.append({
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "pred_anomaly": False,
                "pred_score": 0.0, "error": str(e),
            })

    total_time = time.time() - t_start

    metrics = compute_metrics(results)
    metrics["total_time_sec"] = round(total_time, 1)
    metrics["avg_time_per_video"] = round(total_time / max(len(results), 1), 1)
    metrics["method"] = "videollava"
    metrics["model"] = args.model_path
    metrics["sample_every"] = sample_every
    metrics["segment_sec"] = args.segment_sec
    metrics["video_threshold"] = video_threshold
    metrics["run_timestamp"] = _run_ts
    metrics["run_dir"] = str(_log_dir)

    print_metrics(metrics)

    # 保存
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "frame_scores_array"}
        serializable_results.append(sr)

    results_path = Path(_log_dir) / "results_videollava.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": serializable_results}, f, indent=2, ensure_ascii=False)

    latest_link = Path(_log_base) / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(Path(_log_dir))
    except OSError:
        pass

    logger.info(f"\nResults saved to {results_path}")
    return metrics


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Video-LLaVA UCF-Crime Evaluation")
    parser.add_argument("--model-path",
                        default="/date/liuzhe/EventVAD/Video-LLaVA/checkpoints/Video-LLaVA-7B")
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Process every N-th segment")
    parser.add_argument("--segment-sec", type=float, default=8.0,
                        help="Segment duration in seconds")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Video-level anomaly threshold")
    parser.add_argument("--no-balanced", action="store_true")
    args = parser.parse_args()

    annotations = load_annotations()
    evaluate(annotations, args)


if __name__ == "__main__":
    main()
