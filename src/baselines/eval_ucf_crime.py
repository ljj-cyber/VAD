"""
Baseline Methods — UCF-Crime 全量评估脚本

评估 CLIP / BLIP 零样本基线在 UCF-Crime 数据集上的性能。

帧级评估指标 (4 项核心):
  1. AUC-ROC — 帧级 ROC 曲线下面积
  2. AP      — 帧级 Average Precision
  3. Max-F1  — PR 曲线上最优 F1
  4. EER     — 等错误率 (FPR == FNR 交点)

用法:
  python -m baselines.eval_ucf_crime --method clip --max-videos 0
  python -m baselines.eval_ucf_crime --method blip --max-videos 0
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

import numpy as np

# ── 日志 ──
_log_base = "/date/liuzhe/EventVAD/EventVAD/output/baselines/eval_ucf_crime"
_run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logging(method: str):
    global _log_dir
    _log_dir = f"{_log_base}/run_{_run_ts}_{method}"
    os.makedirs(_log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"{_log_dir}/eval_{method}.log", mode="w", encoding="utf-8"
            ),
        ],
        force=True,
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
    intervals: list
    filepath: str = ""


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
                filename=fname, category=category,
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


# ── 帧级指标计算 ─────────────────────────────────────
def build_gt_mask(intervals: list, total_frames: int) -> np.ndarray:
    mask = np.zeros(total_frames, dtype=bool)
    for s, e in intervals:
        mask[min(s, total_frames - 1): min(e, total_frames)] = True
    return mask


def compute_frame_metrics(all_gt: np.ndarray, all_pred: np.ndarray) -> dict:
    """计算帧级 AUC-ROC, AP, Max-F1, EER"""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve,
    )

    auc = roc_auc_score(all_gt, all_pred)
    ap = average_precision_score(all_gt, all_pred)

    # EER
    fpr, tpr, _ = roc_curve(all_gt, all_pred)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    # Max-F1
    prec_arr, rec_arr, _ = precision_recall_curve(all_gt, all_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr)
    f1_arr = np.nan_to_num(f1_arr, nan=0.0)
    max_f1 = float(f1_arr.max())

    return {
        "frame_auc": round(auc, 4),
        "frame_ap": round(ap, 4),
        "frame_max_f1": round(max_f1, 4),
        "frame_eer": round(eer, 4),
    }


def compute_all_metrics(results: list[dict]) -> dict:
    if not results:
        return {}

    # Video-level
    tp = sum(1 for r in results if r["gt_anomaly"] and r["pred_anomaly"])
    fn = sum(1 for r in results if r["gt_anomaly"] and not r["pred_anomaly"])
    fp = sum(1 for r in results if not r["gt_anomaly"] and r["pred_anomaly"])
    tn = sum(1 for r in results if not r["gt_anomaly"] and not r["pred_anomaly"])
    total = len(results)
    accuracy = (tp + tn) / total

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Frame-level 指标
    all_gt, all_pred = [], []
    for r in results:
        tf = r.get("total_frames", 0)
        if tf <= 0:
            continue
        gt_mask = build_gt_mask(r.get("gt_intervals", []), tf)
        scores = r.get("frame_scores_array")
        if scores is None or len(scores) == 0:
            scores = np.zeros(tf, dtype=np.float32)
        if len(scores) != tf:
            scores = np.interp(
                np.arange(tf), np.linspace(0, tf - 1, len(scores)), scores
            )
        all_gt.extend(gt_mask.astype(int).tolist())
        all_pred.extend(scores.tolist())

    frame_metrics = {"frame_auc": 0, "frame_ap": 0, "frame_max_f1": 0, "frame_eer": 1.0}
    if len(set(all_gt)) > 1:
        try:
            frame_metrics = compute_frame_metrics(
                np.array(all_gt), np.array(all_pred)
            )
        except Exception as e:
            logger.warning(f"Frame metric computation failed: {e}")

    # Video-level AUC
    video_auc = 0.0
    try:
        from sklearn.metrics import roc_auc_score
        vgt = [1 if r["gt_anomaly"] else 0 for r in results]
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
        if r["pred_anomaly"] == r["gt_anomaly"]:
            cat_stats[cat]["correct"] += 1
    for s in cat_stats.values():
        s["accuracy"] = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "video_auc": round(video_auc, 4),
        **frame_metrics,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "total": total,
        "category_stats": cat_stats,
    }


def print_metrics(metrics: dict, method: str):
    print(f"\n{'='*70}")
    print(f"  {method.upper()} Baseline — UCF-Crime Results")
    print(f"{'='*70}")
    print(f"  Video Accuracy:        {metrics['accuracy']:.4f} ({metrics['tp']+metrics['tn']}/{metrics['total']})")
    print(f"  Precision / Recall:    {metrics['precision']:.4f} / {metrics['recall']:.4f}")
    print(f"  F1:                    {metrics['f1']:.4f}")
    print(f"  Video AUC-ROC:         {metrics.get('video_auc', 0):.4f}")
    print(f"  ─── Frame-level (core metrics) ───")
    print(f"  ★ AUC-ROC:             {metrics.get('frame_auc', 0):.4f}")
    print(f"  ★ AP:                  {metrics.get('frame_ap', 0):.4f}")
    print(f"  ★ Max-F1:              {metrics.get('frame_max_f1', 0):.4f}")
    print(f"  ★ EER:                 {metrics.get('frame_eer', 1):.4f}")
    print(f"  TP={metrics['tp']}  FN={metrics['fn']}  FP={metrics['fp']}  TN={metrics['tn']}")
    if metrics.get("total_time_sec"):
        print(f"  Time: {metrics['total_time_sec']}s ({metrics.get('avg_time_per_video',0)}s/video)")
    print(f"\n  Per-category:")
    for cat, s in sorted(metrics.get("category_stats", {}).items()):
        print(f"    {cat:<20} acc={s['accuracy']:.2f} ({s['correct']}/{s['total']})")
    print(f"{'='*70}\n")


# ── 基线工厂 ──────────────────────────────────────────
def create_baseline(method: str, args):
    if method == "clip":
        from baselines.clip_baseline import CLIPBaseline, CLIPBaselineConfig
        cfg = CLIPBaselineConfig()
        cfg.model_name = args.model
        cfg.batch_size = args.batch_size
        return CLIPBaseline(cfg)
    elif method == "blip":
        from baselines.blip_baseline import BLIPBaseline, BLIPBaselineConfig
        cfg = BLIPBaselineConfig()
        cfg.hf_model_name = args.model
        cfg.loader = args.loader
        cfg.lavis_model_type = args.lavis_type
        cfg.batch_size = args.batch_size
        return BLIPBaseline(cfg)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── 评估主函数 ────────────────────────────────────────
def evaluate(method: str, annotations: list[VideoAnnotation], args) -> dict:
    max_videos = args.max_videos
    sample_every = args.sample_every
    video_threshold = args.threshold

    if max_videos > 0:
        annotations = annotations[:max_videos]

    total = len(annotations)
    logger.info(f"\n{'='*60}")
    logger.info(f"  {method.upper()} Baseline — UCF-Crime Evaluation")
    logger.info(f"  Videos: {total}, sample_every={sample_every}")
    logger.info(f"{'='*60}\n")

    # 创建一个共享的 baseline 实例（单线程串行处理）
    baseline = create_baseline(method, args)

    results = []
    t_start = time.time()

    for i, ann in enumerate(annotations):
        if not ann.filepath or not Path(ann.filepath).exists():
            results.append({
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "gt_intervals": ann.intervals,
                "pred_anomaly": False, "pred_score": 0.0,
                "frame_scores_array": None, "total_frames": 0, "fps": 30.0,
                "error": "not found",
            })
            continue

        try:
            t0 = time.time()
            result = baseline.analyze_video(
                video_path=ann.filepath, sample_every_n=sample_every,
            )
            elapsed = time.time() - t0

            frame_scores = result["frame_scores"]
            video_score = result["video_score"]
            pred_anomaly = video_score >= video_threshold

            status = "OK" if pred_anomaly == ann.is_anomaly else "XX"
            logger.info(
                f"[{i+1}/{total}] {status} {ann.filename} "
                f"GT={ann.is_anomaly} Pred={pred_anomaly} "
                f"Score={video_score:.3f} {elapsed:.1f}s"
            )

            results.append({
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "gt_intervals": ann.intervals,
                "pred_anomaly": pred_anomaly, "pred_score": video_score,
                "frame_scores_array": frame_scores,
                "total_frames": result["total_frames"], "fps": result["fps"],
                "time_sec": round(elapsed, 1),
            })
        except Exception as e:
            logger.error(f"[{i+1}/{total}] XX {ann.filename}: {e}")
            results.append({
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "gt_intervals": ann.intervals,
                "pred_anomaly": False, "pred_score": 0.0,
                "frame_scores_array": None, "total_frames": 0, "fps": 30.0,
                "error": str(e),
            })

    total_time = time.time() - t_start

    metrics = compute_all_metrics(results)
    metrics["total_time_sec"] = round(total_time, 1)
    metrics["avg_time_per_video"] = round(total_time / max(len(results), 1), 1)
    metrics["method"] = method
    metrics["model"] = args.model
    metrics["sample_every"] = sample_every
    metrics["video_threshold"] = video_threshold
    metrics["dataset"] = "ucf_crime"

    print_metrics(metrics, method)

    # 保存
    out_dir = Path(_log_dir)
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "frame_scores_array"}
        serializable.append(sr)

    results_path = out_dir / f"results_{method}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": serializable}, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {results_path}")
    return metrics


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Baseline UCF-Crime Evaluation")
    parser.add_argument("--method", required=True, choices=["clip", "blip"])
    parser.add_argument("--model", default="")
    parser.add_argument("--loader", default="transformers", choices=["transformers", "lavis"])
    parser.add_argument("--lavis-type", default="pretrain")
    parser.add_argument("--max-videos", type=int, default=0, help="0 = all 290 videos")
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.model:
        args.model = ("openai/clip-vit-base-patch16" if args.method == "clip"
                       else "Salesforce/blip-itm-base-coco")
    if args.batch_size == 0:
        args.batch_size = 64 if args.method == "clip" else 32

    _setup_logging(args.method)
    annotations = load_annotations()
    evaluate(args.method, annotations, args)


if __name__ == "__main__":
    main()
