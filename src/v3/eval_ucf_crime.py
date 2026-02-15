"""
UCF-Crime 测试集评估 — V4.0 消融契约模式

评估指标:
  1. Video-level Accuracy: 视频级异常/正常分类准确率
  2. Frame-level AUC: 帧级异常分数 ROC-AUC
  3. Anomaly Segment IoU: 检测到的异常片段与 GT 的交并比

数据集: UCF-Crime Testing Set (140 异常 + 150 正常 = 290 视频)

用法:
  python -m v3.eval_ucf_crime [--max-videos N] [--mode v4|v3] [--no-contracts]
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_log_file = "/data/liuzhe/EventVAD/output/v4/eval_ucf_crime/eval_detailed.log"
import os as _os
_os.makedirs(_os.path.dirname(_log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file, mode="w", encoding="utf-8"),
    ],
)
# 降低非关键日志
for mod in ["httpx", "v3.perception.frame_sampler",
            "v3.association.temporal_graph", "sentence_transformers"]:
    logging.getLogger(mod).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from v3.pipeline import VideoAnomalyPipeline
from v3.config import OUTPUT_DIR


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
    intervals: list[tuple[int, int]]  # (start_frame, end_frame)
    filepath: str = ""
    total_frames: int = 0


def load_annotations() -> list[VideoAnnotation]:
    """加载 UCF-Crime 测试集标注"""
    # 建立文件索引
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
            fname = parts[0]
            category = parts[1]
            s1, e1 = int(parts[2]), int(parts[3])
            s2, e2 = int(parts[4]), int(parts[5])

            intervals = []
            if s1 != -1 and e1 != -1:
                intervals.append((s1, e1))
            if s2 != -1 and e2 != -1:
                intervals.append((s2, e2))

            is_anomaly = category != "Normal"

            ann = VideoAnnotation(
                filename=fname,
                category=category,
                is_anomaly=is_anomaly,
                intervals=intervals,
                filepath=file_index.get(fname, ""),
            )
            annotations.append(ann)

    logger.info(f"Loaded {len(annotations)} annotations "
                f"({sum(a.is_anomaly for a in annotations)} anomaly, "
                f"{sum(not a.is_anomaly for a in annotations)} normal)")
    return annotations


def get_total_frames(video_path: str) -> int:
    """获取视频总帧数"""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ── 评估指标 ──────────────────────────────────────────
def _build_pred_mask(pred_segments: list[dict], total_frames: int, fps: float) -> np.ndarray:
    """将预测片段转为帧级 bool mask"""
    mask = np.zeros(total_frames, dtype=bool)
    for seg in pred_segments:
        start_sec = _parse_time(seg.get("start", "0"))
        end_sec = _parse_time(seg.get("end", "0"))
        sf = int(start_sec * fps)
        ef = int(end_sec * fps)
        mask[min(sf, total_frames - 1): min(ef, total_frames)] = True
    return mask


def _build_gt_mask(gt_intervals: list[tuple[int, int]], total_frames: int) -> np.ndarray:
    """将 GT 区间转为帧级 bool mask"""
    mask = np.zeros(total_frames, dtype=bool)
    for s, e in gt_intervals:
        mask[min(s, total_frames - 1): min(e, total_frames)] = True
    return mask


def compute_frame_iou(
    gt_intervals: list[tuple[int, int]],
    pred_segments: list[dict],
    total_frames: int,
    fps: float,
) -> float:
    """计算帧级 IoU"""
    if total_frames <= 0:
        return 0.0

    gt_mask = _build_gt_mask(gt_intervals, total_frames)
    pred_mask = _build_pred_mask(pred_segments, total_frames, fps)

    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def compute_frame_scores(
    pred_segments: list[dict],
    pred_score: float,
    total_frames: int,
    fps: float,
) -> np.ndarray:
    """生成帧级异常分数：片段内高分，边界缓冲区低分，片段外=0。"""
    scores = np.zeros(total_frames, dtype=np.float32)
    # 轻度时间扩展，缓解边界误差同时避免过宽区间污染
    expand_sec = 1.0
    for seg in pred_segments:
        start_sec = _parse_time(seg.get("start", "0"))
        end_sec = _parse_time(seg.get("end", "0"))
        conf = seg.get("confidence", pred_score)
        sf = int(max(0.0, start_sec - expand_sec) * fps)
        ef = int(min((total_frames - 1) / max(fps, 1e-6), end_sec + expand_sec) * fps)
        core_sf = int(start_sec * fps)
        core_ef = int(end_sec * fps)

        sf = min(sf, total_frames - 1)
        ef = min(ef, total_frames)
        core_sf = min(core_sf, total_frames - 1)
        core_ef = min(core_ef, total_frames)

        if sf < core_sf:
            scores[sf:core_sf] = np.maximum(scores[sf:core_sf], float(conf) * 0.6)
        if core_sf < core_ef:
            scores[core_sf:core_ef] = np.maximum(scores[core_sf:core_ef], float(conf))
        if core_ef < ef:
            scores[core_ef:ef] = np.maximum(scores[core_ef:ef], float(conf) * 0.6)
    return scores


def _parse_time(time_str) -> float:
    """解析 'MM:SS.S' 或数字字符串为秒数"""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    time_str = str(time_str).strip()
    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    try:
        return float(time_str)
    except ValueError:
        return 0.0


# ── 主评估逻辑 ────────────────────────────────────────
def evaluate(
    annotations: list[VideoAnnotation],
    mode: str = "v4",
    no_contracts: bool = True,
    backend: str = "server",
    api_base: str = "http://localhost:8000",
    max_workers: int = 16,
    max_videos: int = 0,
    parallel_videos: int = 3,
) -> dict:
    """运行评估"""

    if max_videos > 0:
        annotations = annotations[:max_videos]

    total = len(annotations)
    logger.info(f"\n{'='*60}")
    logger.info(f"  UCF-Crime Evaluation")
    logger.info(f"  Mode: {mode}, Contracts: {'OFF' if no_contracts else 'ON'}")
    logger.info(f"  Videos: {total}")
    logger.info(f"{'='*60}\n")

    # ── 加速配置 ──
    from v3.config import SamplerConfig, PerceptionConfig, DecisionConfig, LocalizationConfig
    SamplerConfig.base_fps = 0.6             # 提升时间对齐质量，优先帧级 AUC
    PerceptionConfig.frame_size = (336, 336) # 保持适中视觉分辨率
    PerceptionConfig.max_new_tokens = 896    # 适当提高结构化输出稳定性
    PerceptionConfig.max_retries = 2
    DecisionConfig.decision_max_tokens = 384 # 允许更稳定的断点与理由输出
    DecisionConfig.max_audit_entities = 12
    LocalizationConfig.segment_padding_sec = 2.0

    # ── 多视频并行处理 ──
    n_parallel = parallel_videos  # 同时处理的视频数

    results = []
    t_start = time.time()
    results_lock = __import__("threading").Lock()

    def _process_one(idx_ann: tuple[int, VideoAnnotation]) -> dict:
        """处理单个视频（线程安全）"""
        i, ann = idx_ann
        if not ann.filepath or not Path(ann.filepath).exists():
            return {"filename": ann.filename, "error": "not found",
                    "gt_anomaly": ann.is_anomaly, "pred_anomaly": False, "pred_score": 0.0}

        # 每个线程创建独立的 pipeline 实例
        pipe = VideoAnomalyPipeline(
            model_name="qwen2-vl-7b",
            mode=mode,
            backend=backend,
            api_base=api_base,
            max_workers=max(4, max_workers // n_parallel),
            save_intermediate=False,
        )
        if no_contracts:
            pipe.decision_cfg.business_contracts = {"default": []}
        # 评估不需要导出 clip，减少磁盘 IO 与编码开销
        pipe.localization_cfg.save_anomaly_clips = False

        try:
            t0 = time.time()
            result = pipe.process_video(ann.filepath)
            elapsed = time.time() - t0

            total_frames = result.get("total_frames", 0)
            fps = total_frames / result.get("duration_sec", 1) if result.get("duration_sec") else 30.0
            pred_anomaly = result.get("status") == "Anomaly Detected"
            pred_score = result.get("anomaly_score", 0.0)
            pred_segments = result.get("anomaly_segments", [])
            iou = compute_frame_iou(ann.intervals, pred_segments, total_frames, fps) if ann.is_anomaly else None

            status = "✅" if pred_anomaly == ann.is_anomaly else "❌"
            iou_str = f"IoU={iou:.3f}" if iou is not None else "N/A"
            logger.info(f"[{i+1}/{total}] {status} {ann.filename} "
                        f"GT={ann.is_anomaly} Pred={pred_anomaly} "
                        f"Score={pred_score:.3f} {iou_str} {elapsed:.0f}s")

            return {
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "gt_intervals": ann.intervals,
                "pred_anomaly": pred_anomaly, "pred_score": pred_score,
                "pred_segments": len(pred_segments),
                "pred_segments_raw": pred_segments,
                "iou": iou,
                "fps": fps,
                "time_sec": round(elapsed, 1), "total_frames": total_frames,
            }
        except Exception as e:
            logger.error(f"[{i+1}/{total}] ❌ {ann.filename}: {e}")
            return {
                "filename": ann.filename, "category": ann.category,
                "gt_anomaly": ann.is_anomaly, "pred_anomaly": False,
                "pred_score": 0.0, "error": str(e),
            }
        finally:
            pipe.entity_pool = None
            pipe.temporal_graph = None
            pipe.anomaly_detector = None
            pipe.localizer = None
            # 不调 cleanup 以保持 server 模式下无模型需要释放

    if n_parallel > 1:
        logger.info(f"Processing {total} videos with {n_parallel} parallel workers")
        from concurrent.futures import ThreadPoolExecutor as TPE
        with TPE(max_workers=n_parallel) as executor:
            results = list(executor.map(_process_one, enumerate(annotations)))
    else:
        for i, ann in enumerate(annotations):
            results.append(_process_one((i, ann)))
    total_time = time.time() - t_start

    # ── 计算指标 ──
    metrics = compute_metrics(results)
    metrics["total_time_sec"] = round(total_time, 1)
    metrics["avg_time_per_video"] = round(total_time / max(len(results), 1), 1)
    metrics["mode"] = mode
    metrics["contracts"] = not no_contracts

    # ── 打印结果 ──
    print_metrics(metrics, results)

    # ── 保存结果 ──
    out_dir = OUTPUT_DIR / "eval_ucf_crime"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{mode}_{'with' if not no_contracts else 'no'}_contracts"
    with open(out_dir / f"results_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {out_dir}/results_{suffix}.json")
    return metrics


def compute_metrics(results: list[dict]) -> dict:
    """计算评估指标"""
    if not results:
        return {}

    # Video-level accuracy
    correct = sum(1 for r in results if r.get("pred_anomaly") == r.get("gt_anomaly"))
    total = len(results)
    accuracy = correct / total

    # 分类别
    tp = sum(1 for r in results if r.get("gt_anomaly") and r.get("pred_anomaly"))
    fn = sum(1 for r in results if r.get("gt_anomaly") and not r.get("pred_anomaly"))
    fp = sum(1 for r in results if not r.get("gt_anomaly") and r.get("pred_anomaly"))
    tn = sum(1 for r in results if not r.get("gt_anomaly") and not r.get("pred_anomaly"))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Frame-level AUC — 用片段级分数（片段内=conf，片段外=0）
    try:
        from sklearn.metrics import roc_auc_score
        all_gt_frames, all_pred_frames = [], []
        for r in results:
            tf = r.get("total_frames", 0)
            if tf <= 0:
                continue
            fps = float(r.get("fps", 30.0) or 30.0)
            gt_mask = _build_gt_mask(r.get("gt_intervals", []), tf)
            pred_scores_f = compute_frame_scores(
                r.get("pred_segments_raw", []), r.get("pred_score", 0.0), tf, fps
            )
            all_gt_frames.extend(gt_mask.astype(int).tolist())
            all_pred_frames.extend(pred_scores_f.tolist())

        if len(set(all_gt_frames)) > 1:
            auc = roc_auc_score(all_gt_frames, all_pred_frames)
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    # Video-level AUC
    try:
        vgt = [1 if r.get("gt_anomaly") else 0 for r in results]
        vps = [r.get("pred_score", 0.0) for r in results]
        video_auc = roc_auc_score(vgt, vps) if len(set(vgt)) > 1 else 0.0
    except Exception:
        video_auc = 0.0

    # Anomaly segment IoU (仅异常视频)
    ious = [r["iou"] for r in results if r.get("iou") is not None and r["iou"] > 0]
    mean_iou = np.mean(ious) if ious else 0.0

    # 按类别统计
    category_stats = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0, "ious": []}
        category_stats[cat]["total"] += 1
        if r.get("pred_anomaly") == r.get("gt_anomaly"):
            category_stats[cat]["correct"] += 1
        if r.get("iou") is not None:
            category_stats[cat]["ious"].append(r["iou"])

    for cat, stats in category_stats.items():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0
        stats["mean_iou"] = round(np.mean(stats["ious"]), 4) if stats["ious"] else None
        del stats["ious"]

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "frame_auc": round(auc, 4),
        "video_auc": round(video_auc, 4),
        "mean_iou": round(mean_iou, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "total": total,
        "category_stats": category_stats,
    }


def print_metrics(metrics: dict, results: list[dict]):
    """打印评估结果"""
    print(f"\n{'='*70}")
    print(f"  UCF-Crime Test Set Evaluation Results")
    print(f"  Mode: {metrics.get('mode')}, Contracts: {'ON' if metrics.get('contracts') else 'OFF'}")
    print(f"{'='*70}")
    print(f"  Video-level Accuracy:  {metrics['accuracy']:.4f} ({metrics['tp']+metrics['tn']}/{metrics['total']})")
    print(f"  Precision:             {metrics['precision']:.4f}")
    print(f"  Recall:                {metrics['recall']:.4f}")
    print(f"  F1 Score:              {metrics['f1']:.4f}")
    print(f"  Frame-level AUC-ROC:   {metrics.get('frame_auc', 0):.4f}")
    print(f"  Video-level AUC-ROC:   {metrics.get('video_auc', 0):.4f}")
    print(f"  Mean Anomaly IoU:      {metrics['mean_iou']:.4f}")
    print(f"  TP={metrics['tp']}  FN={metrics['fn']}  FP={metrics['fp']}  TN={metrics['tn']}")
    print(f"  Total time: {metrics.get('total_time_sec', 0)}s ({metrics.get('avg_time_per_video', 0)}s/video)")
    print(f"\n  Per-category accuracy:")

    for cat, stats in sorted(metrics.get("category_stats", {}).items()):
        iou_str = f"IoU={stats['mean_iou']:.3f}" if stats.get("mean_iou") is not None else ""
        print(f"    {cat:<20} acc={stats['accuracy']:.2f} ({stats['correct']}/{stats['total']}) {iou_str}")

    print(f"{'='*70}\n")


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UCF-Crime V4 Evaluation")
    parser.add_argument("--max-videos", type=int, default=0, help="Max videos (0=all)")
    parser.add_argument("--mode", default="v4", choices=["v4", "v3"])
    parser.add_argument("--no-contracts", action="store_true", default=True, help="Disable business contracts")
    parser.add_argument("--with-contracts", action="store_true", help="Enable business contracts")
    parser.add_argument("--backend", default="server", choices=["server", "local"])
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--parallel", type=int, default=3, help="Parallel video processing count")
    args = parser.parse_args()

    no_contracts = not args.with_contracts

    annotations = load_annotations()
    evaluate(
        annotations,
        mode=args.mode,
        no_contracts=no_contracts,
        backend=args.backend,
        api_base=args.api_base,
        max_workers=args.max_workers,
        max_videos=args.max_videos,
        parallel_videos=args.parallel,
    )


if __name__ == "__main__":
    main()
