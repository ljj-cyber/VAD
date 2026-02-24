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
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np

# ── 日志（以时间戳命名，避免覆盖） ──
_log_base = "/date/liuzhe/EventVAD/EventVAD/output/v5/eval_ucf_crime"
_run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_dir = f"{_log_base}/run_{_run_ts}"
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
    帧级异常分数生成器 — 纯 Signal 1。

    Signal 1: Decision LLM 输出的 anomaly_start_sec / anomaly_end_sec 区间，
              直接广播 confidence 值，前后加 ramp 渐变，最后高斯平滑。
    """
    try:
        from scipy.ndimage import gaussian_filter1d
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    scores = np.zeros(total_frames, dtype=np.float32)
    ramp_sec = 5.0

    any_entity_anomaly = False
    for v in entity_verdicts:
        if not v.get("is_anomaly", False):
            continue

        any_entity_anomaly = True
        conf = float(v.get("confidence", video_confidence))
        start_sec = float(v.get("anomaly_start_sec", 0.0))
        end_sec = float(v.get("anomaly_end_sec", 0.0))

        if end_sec <= start_sec:
            continue

        core_sf = max(0, min(int(start_sec * fps), total_frames))
        core_ef = max(0, min(int(end_sec * fps), total_frames))

        if core_sf < core_ef:
            scores[core_sf:core_ef] = np.maximum(scores[core_sf:core_ef], conf)

        # 前后渐变
        ramp_frames = int(ramp_sec * fps)
        ramp_sf = max(0, core_sf - ramp_frames)
        if ramp_sf < core_sf:
            n = core_sf - ramp_sf
            ramp_values = np.linspace(conf * 0.1, conf * 0.8, n, dtype=np.float32)
            scores[ramp_sf:core_sf] = np.maximum(scores[ramp_sf:core_sf], ramp_values)

        ramp_ef = min(total_frames, core_ef + ramp_frames)
        if core_ef < ramp_ef:
            n = ramp_ef - core_ef
            ramp_values = np.linspace(conf * 0.8, conf * 0.1, n, dtype=np.float32)
            scores[core_ef:ramp_ef] = np.maximum(scores[core_ef:ramp_ef], ramp_values)

    # 高斯平滑 (σ = 2秒)
    if total_frames > 10 and _has_scipy:
        smooth_sigma = max(1, int(2.0 * fps))
        scores = gaussian_filter1d(scores, sigma=smooth_sigma).astype(np.float32)

    # 如果视频判异常但所有信号都为0，给全视频低底分
    if any_entity_anomaly and scores.max() < 0.01:
        scores[:] = video_confidence * 0.2

    # 裁剪到 [0, 1]
    scores = np.clip(scores, 0.0, 1.0)

    return scores


def _hysteresis_threshold(
    scores: np.ndarray,
    tau_high: float = 0.35,
    tau_low: float = 0.15,
) -> np.ndarray:
    """
    滞回阈值：score >= tau_high 启动异常段，score < tau_low 结束。
    桥接异常段中间的短暂 score 下降，减少碎片化。
    """
    mask = np.zeros(len(scores), dtype=bool)
    in_segment = False
    for i in range(len(scores)):
        if not in_segment:
            if scores[i] >= tau_high:
                in_segment = True
                mask[i] = True
        else:
            if scores[i] >= tau_low:
                mask[i] = True
            else:
                in_segment = False
    return mask


def compute_frame_iou(
    gt_intervals: list,
    entity_verdicts: list[dict],
    total_frames: int,
    fps: float,
    mode: str = "soft",
    tau_high: float = 0.35,
    tau_low: float = 0.15,
) -> float:
    """
    帧级 IoU 计算（基于 compute_frame_scores 生成的连续分数）。

    模式:
      - "soft":       Soft IoU = Σmin(gt, pred) / Σmax(gt, pred)，无阈值
      - "hysteresis": 滞回阈值二值化后计算 hard IoU
    """
    if total_frames <= 0:
        return 0.0

    gt_mask = build_gt_mask(gt_intervals, total_frames).astype(np.float32)

    video_confidence = max(
        (float(v.get("confidence", 0.0)) for v in entity_verdicts if v.get("is_anomaly")),
        default=0.0,
    )
    scores = compute_frame_scores(
        entity_verdicts, video_confidence,
        total_frames, fps,
    )

    if mode == "soft":
        # Soft IoU: 无阈值，信息保留最完整
        numerator = np.minimum(gt_mask, scores).sum()
        denominator = np.maximum(gt_mask, scores).sum()
        if denominator < 1e-8:
            return 1.0 if numerator < 1e-8 else 0.0
        return float(numerator / denominator)

    else:  # "hysteresis"
        pred_mask = _hysteresis_threshold(scores, tau_high, tau_low)
        inter = np.logical_and(gt_mask > 0.5, pred_mask).sum()
        union = np.logical_or(gt_mask > 0.5, pred_mask).sum()
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
    backend: str = "server",
    model_name: str = "",
    disable_discordance: bool = False,
    use_yolo: bool = False,
) -> dict:
    # 本地推理时强制串行（模型单线程）
    if backend == "local" and parallel_videos > 1:
        logger.info(f"Local backend: forcing parallel=1 (was {parallel_videos})")
        parallel_videos = 1

    if max_videos > 0:
        if balanced:
            # 分层采样：每个异常类别至少 1 个视频，剩余按类别大小分配
            from collections import defaultdict
            anomaly = [a for a in annotations if a.is_anomaly]
            normal = [a for a in annotations if not a.is_anomaly]
            n_anomaly = max_videos // 2
            n_normal = max_videos - n_anomaly

            # 按类别分组
            cat_groups: dict[str, list] = defaultdict(list)
            for a in anomaly:
                cat_groups[a.category].append(a)

            # 分层采样: 每个类别至少 1 个
            selected_anomaly = []
            remaining_slots = n_anomaly
            categories = sorted(cat_groups.keys())

            # 第一轮: 每个类别选 1 个
            for cat in categories:
                if remaining_slots <= 0:
                    break
                selected_anomaly.append(cat_groups[cat][0])
                remaining_slots -= 1

            # 第二轮: 剩余名额按类别大小比例分配（从大类别再选）
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

            # 统计
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
    ablation_tag = " [ABLATION: no-discordance]" if disable_discordance else ""
    yolo_tag = " [YOLO hybrid]" if use_yolo else ""
    logger.info(f"\n{'='*60}")
    logger.info(f"  V5 Tube-Skeleton UCF-Crime Evaluation{ablation_tag}{yolo_tag}")
    logger.info(f"  Videos: {total}, sample_every={sample_every}, parallel={parallel_videos}")
    if disable_discordance:
        logger.info(f"  ★ Discordance DISABLED — pure semantic anomaly detection")
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
            backend=backend,
            model_name=model_name,
            disable_discordance=disable_discordance,
            use_yolo=use_yolo,
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

            iou_soft = None
            iou_hyst = None
            if ann.is_anomaly and total_frames > 0:
                iou_soft = compute_frame_iou(
                    ann.intervals, entity_verdicts, total_frames, fps,
                    mode="soft",
                )
                iou_hyst = compute_frame_iou(
                    ann.intervals, entity_verdicts, total_frames, fps,
                    mode="hysteresis",
                )

            status = "✅" if pred_anomaly == ann.is_anomaly else "❌"
            iou_str = ""
            if iou_soft is not None:
                iou_str = f"SoftIoU={iou_soft:.3f} HystIoU={iou_hyst:.3f}"
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
                "iou": iou_soft,
                "iou_soft": iou_soft,
                "iou_hysteresis": iou_hyst,
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
    metrics["max_videos"] = max_videos
    metrics["run_timestamp"] = _run_ts
    metrics["run_dir"] = str(_log_dir)
    metrics["disable_discordance"] = disable_discordance
    metrics["use_yolo"] = use_yolo

    # ── 打印 ──
    print_metrics(metrics)

    # ── 保存 ──
    out_dir = Path(_log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results_v5.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)

    # 创建/更新 latest 软链接，方便快速查看最新结果
    latest_link = Path(_log_base) / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(out_dir)
    except OSError:
        pass  # 软链接创建失败不影响主流程

    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Run directory: {out_dir}")
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

    # Frame-level metrics: AUC-ROC, AP, EER, Max F1
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
            all_gt_np = np.array(all_gt)
            all_pred_np = np.array(all_pred)

            frame_auc = roc_auc_score(all_gt_np, all_pred_np)
            frame_ap = average_precision_score(all_gt_np, all_pred_np)

            # EER: FPR == FNR (i.e., FPR == 1 - TPR)
            fpr, tpr, _ = roc_curve(all_gt_np, all_pred_np)
            fnr = 1.0 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            frame_eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

            # Max F1: best F1 across all thresholds on PR curve
            prec_arr, rec_arr, _ = precision_recall_curve(all_gt_np, all_pred_np)
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr)
            f1_arr = np.nan_to_num(f1_arr, nan=0.0)
            frame_max_f1 = float(f1_arr.max())

        logger.info(
            f"Frame-level stats: {len(all_gt)} total frames, "
            f"{sum(all_gt)} anomaly frames ({100*sum(all_gt)/max(len(all_gt),1):.1f}%)"
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

    # IoU (Soft + Hysteresis)
    ious_soft = [r["iou_soft"] for r in results
                 if r.get("iou_soft") is not None and r["iou_soft"] > 0]
    ious_hyst = [r["iou_hysteresis"] for r in results
                 if r.get("iou_hysteresis") is not None and r["iou_hysteresis"] > 0]
    mean_iou_soft = float(np.mean(ious_soft)) if ious_soft else 0.0
    mean_iou_hyst = float(np.mean(ious_hyst)) if ious_hyst else 0.0

    # Per category
    cat_stats = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0,
                              "ious_soft": [], "ious_hyst": []}
        cat_stats[cat]["total"] += 1
        if r.get("pred_anomaly") == r.get("gt_anomaly"):
            cat_stats[cat]["correct"] += 1
        if r.get("iou_soft") is not None:
            cat_stats[cat]["ious_soft"].append(r["iou_soft"])
        if r.get("iou_hysteresis") is not None:
            cat_stats[cat]["ious_hyst"].append(r["iou_hysteresis"])

    for cat, s in cat_stats.items():
        s["accuracy"] = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0
        s["mean_iou_soft"] = round(float(np.mean(s["ious_soft"])), 4) if s["ious_soft"] else 0.0
        s["mean_iou_hyst"] = round(float(np.mean(s["ious_hyst"])), 4) if s["ious_hyst"] else 0.0
        del s["ious_soft"]
        del s["ious_hyst"]

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
        "mean_iou": round(mean_iou_soft, 4),
        "mean_iou_soft": round(mean_iou_soft, 4),
        "mean_iou_hysteresis": round(mean_iou_hyst, 4),
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
    print(f"  ★ Frame-level AP:      {metrics.get('frame_ap', 0):.4f}")
    print(f"  ★ Frame-level EER:     {metrics.get('frame_eer', 1):.4f}")
    print(f"  ★ Frame-level Max F1:  {metrics.get('frame_max_f1', 0):.4f}")
    print(f"  Video-level AUC-ROC:   {metrics.get('video_auc', 0):.4f}")
    print(f"  Mean Soft IoU:         {metrics.get('mean_iou_soft', 0):.4f}")
    print(f"  Mean Hysteresis IoU:   {metrics.get('mean_iou_hysteresis', 0):.4f}")
    print(f"  TP={metrics['tp']}  FN={metrics['fn']}  FP={metrics['fp']}  TN={metrics['tn']}")
    print(f"  Total time: {metrics.get('total_time_sec', 0)}s "
          f"({metrics.get('avg_time_per_video', 0)}s/video)")
    print(f"\n  Per-category:")
    for cat, s in sorted(metrics.get("category_stats", {}).items()):
        iou_s = f"SoftIoU={s['mean_iou_soft']:.3f}" if s.get("mean_iou_soft") else ""
        iou_h = f"HystIoU={s['mean_iou_hyst']:.3f}" if s.get("mean_iou_hyst") else ""
        print(f"    {cat:<20} acc={s['accuracy']:.2f} ({s['correct']}/{s['total']}) {iou_s} {iou_h}")
    print(f"{'='*70}\n")


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V5 UCF-Crime Evaluation")
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=2,
                        help="Process every N-th frame (default 2)")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--parallel", type=int, default=6,
                        help="Parallel video processing count")
    parser.add_argument("--backend", default="server", choices=["server", "local"],
                        help="Inference backend: server (vLLM API) or local (transformers)")
    parser.add_argument("--model", default="", help="VLM model name (e.g. qwen2.5-vl-32b)")
    parser.add_argument("--no-balanced", action="store_true",
                        help="Disable balanced anomaly/normal sampling")
    parser.add_argument("--no-discordance", action="store_true",
                        help="Disable discordance checker (ablation: pure semantic)")
    parser.add_argument("--yolo", action="store_true",
                        help="Enable YOLO-World hybrid detection (motion + semantic)")
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
        backend=args.backend,
        model_name=args.model,
        disable_discordance=args.no_discordance,
        use_yolo=args.yolo,
    )


if __name__ == "__main__":
    main()
