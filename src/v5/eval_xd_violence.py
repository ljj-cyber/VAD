"""
V5 Tube-Skeleton Pipeline — XD-Violence 评估脚本

评估指标:
  1. Frame-level AUC-ROC (核心)
  2. Frame-level AP (Average Precision)
  3. Video-level Accuracy / Precision / Recall / F1

标注格式 (annotations.txt):
  每行: video_name frame_start frame_end [frame_start frame_end ...]
  - 帧号成对出现，表示异常区间
  - label_A 视频为正常视频（不在标注文件中）

用法:
  python -m v5.eval_xd_violence --max-videos 50 --sample-every 2 --parallel 3
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
from dataclasses import dataclass, field

import cv2
import numpy as np

# ── 日志 ──
_log_base = "/data/liuzhe/EventVAD/output/v5/eval_xd_violence"
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
XD_ROOT = Path("/data/liuzhe/EventVAD/src/event_seg/videos/xdviolence")
XD_VIDEOS = XD_ROOT / "videos"
ANN_FILE = Path("/data/liuzhe/EventVAD/src/event_seg/videos/annotations.txt")


@dataclass
class VideoAnnotation:
    filename: str          # e.g., "v=S-7rRLrxnVQ__#1_label_B4-0-0"
    label: str             # e.g., "B4-0-0", "A" (normal)
    is_anomaly: bool
    intervals: list = field(default_factory=list)  # [(start_frame, end_frame), ...]
    filepath: str = ""
    total_frames: int = 0


def _extract_label(filename: str) -> str:
    """从文件名中提取标签，如 _label_B4-0-0 -> B4-0-0"""
    idx = filename.find("_label_")
    if idx >= 0:
        return filename[idx + 7:]
    return "unknown"


def _extract_primary_category(label: str) -> str:
    """提取主类别，如 B4-0-0 -> B4, G-B2-0 -> G, B1-B2-B6 -> B1"""
    parts = label.split("-")
    return parts[0] if parts else label


def load_annotations() -> list[VideoAnnotation]:
    """加载 XD-Violence 测试集标注"""
    # 构建视频文件索引
    file_index = {}
    if XD_VIDEOS.exists():
        for f in XD_VIDEOS.iterdir():
            if f.suffix == ".mp4":
                stem = f.stem  # 不含 .mp4
                file_index[stem] = str(f)

    annotations = []
    ann_names = set()

    # 1. 加载异常视频标注
    with open(ANN_FILE) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            video_name = parts[0]
            # 去掉可能的 .mp4 后缀
            video_name_clean = video_name.replace(".mp4", "")
            frame_numbers = [int(x) for x in parts[1:]]

            # 解析帧号对
            intervals = []
            for i in range(0, len(frame_numbers), 2):
                if i + 1 < len(frame_numbers):
                    s, e = frame_numbers[i], frame_numbers[i + 1]
                    if s < e:
                        intervals.append((s, e))

            label = _extract_label(video_name_clean)
            filepath = file_index.get(video_name_clean, "")

            annotations.append(VideoAnnotation(
                filename=video_name_clean,
                label=label,
                is_anomaly=True,
                intervals=intervals,
                filepath=filepath,
            ))
            ann_names.add(video_name_clean)

    # 2. 加载正常视频（label_A，不在标注文件中）
    for stem, fpath in file_index.items():
        if stem not in ann_names and "_label_A" in stem:
            annotations.append(VideoAnnotation(
                filename=stem,
                label="A",
                is_anomaly=False,
                intervals=[],
                filepath=fpath,
            ))

    n_anomaly = sum(1 for a in annotations if a.is_anomaly)
    n_normal = sum(1 for a in annotations if not a.is_anomaly)
    n_found = sum(1 for a in annotations if a.filepath)
    logger.info(
        f"Loaded {len(annotations)} annotations: "
        f"{n_anomaly} anomaly + {n_normal} normal, "
        f"{n_found} files found"
    )
    return annotations


# ── 帧级分数 (复用 eval_ucf_crime 中的逻辑) ──────────
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
    """帧级异常分数：基于 entity verdict 的时间区间广播 + 高斯平滑"""
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

    # 高斯平滑
    if total_frames > 10 and _has_scipy:
        smooth_sigma = max(1, int(2.0 * fps))
        scores = gaussian_filter1d(scores, sigma=smooth_sigma).astype(np.float32)

    # 视频判异常但帧分数全 0 → 给低底分
    if any_entity_anomaly and scores.max() < 0.01:
        scores[:] = video_confidence * 0.2

    return np.clip(scores, 0.0, 1.0)


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
    disable_discordance: bool = False,
    use_yolo: bool = False,
    cinematic_filter: bool = False,
) -> dict:
    if backend == "local" and parallel_videos > 1:
        logger.info(f"Local backend: forcing parallel=1 (was {parallel_videos})")
        parallel_videos = 1

    if max_videos > 0:
        if balanced:
            from collections import defaultdict
            anomaly = [a for a in annotations if a.is_anomaly]
            normal = [a for a in annotations if not a.is_anomaly]
            n_anomaly = max_videos // 2
            n_normal = max_videos - n_anomaly

            # 按主类别分组
            cat_groups: dict[str, list] = defaultdict(list)
            for a in anomaly:
                cat = _extract_primary_category(a.label)
                cat_groups[cat].append(a)

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
                    picked = sum(1 for a in selected_anomaly
                                 if _extract_primary_category(a.label) == cat)
                    if picked < len(cat_groups[cat]):
                        selected_anomaly.append(cat_groups[cat][picked])
                        remaining_slots -= 1
                    idx += 1

            annotations = selected_anomaly + normal[:n_normal]

            cat_counts = defaultdict(int)
            for a in selected_anomaly:
                cat_counts[_extract_primary_category(a.label)] += 1
            cat_str = ", ".join(f"{c}={n}" for c, n in sorted(cat_counts.items()))
            logger.info(
                f"Stratified sampling: {len(selected_anomaly)} anomaly [{cat_str}] "
                f"+ {min(n_normal, len(normal))} normal = {len(annotations)}"
            )
        else:
            annotations = annotations[:max_videos]

    total = len(annotations)
    logger.info(f"\n{'='*60}")
    logger.info(f"  V5 Tube-Skeleton XD-Violence Evaluation")
    logger.info(f"  Videos: {total}, sample_every={sample_every}, parallel={parallel_videos}")
    logger.info(f"  cinematic_filter={cinematic_filter}")
    logger.info(f"{'='*60}\n")

    results = []
    t_start = time.time()

    def _process_one(idx_ann):
        i, ann = idx_ann
        if not ann.filepath or not Path(ann.filepath).exists():
            logger.warning(f"[{i+1}/{total}] SKIP {ann.filename}: file not found")
            return {
                "filename": ann.filename, "label": ann.label,
                "gt_anomaly": ann.is_anomaly, "pred_anomaly": False,
                "pred_score": 0.0, "error": "not found",
            }

        pipe = TubeSkeletonPipeline(
            api_base=api_base,
            max_workers=max(4, max_workers // parallel_videos),
            backend=backend,
            disable_discordance=disable_discordance,
            use_yolo=use_yolo,
            cinematic_filter=cinematic_filter,
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

            is_cine = verdict.get("is_cinematic", False)
            cine_tag = " [CINE]" if is_cine else ""
            status = "✅" if pred_anomaly == ann.is_anomaly else "❌"
            logger.info(
                f"[{i+1}/{total}] {status} {ann.filename} "
                f"GT={ann.is_anomaly} Pred={pred_anomaly} "
                f"Score={pred_score:.2f} "
                f"entities={result['stats']['entities']} "
                f"triggers={result['stats']['triggers']} "
                f"{elapsed:.1f}s{cine_tag}"
            )

            return {
                "filename": ann.filename,
                "label": ann.label,
                "gt_anomaly": ann.is_anomaly,
                "gt_intervals": ann.intervals,
                "pred_anomaly": pred_anomaly,
                "pred_score": pred_score,
                "entity_verdicts": entity_verdicts,
                "fps": fps,
                "total_frames": total_frames,
                "time_sec": round(elapsed, 1),
                "stats": result.get("stats", {}),
                "is_cinematic": is_cine,
                "cinematic_reason": verdict.get("cinematic_reason", ""),
            }
        except Exception as e:
            logger.error(f"[{i+1}/{total}] ❌ {ann.filename}: {e}")
            return {
                "filename": ann.filename, "label": ann.label,
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

    # ── 打印 ──
    print_metrics(metrics)

    # ── 保存 ──
    out_dir = Path(_log_dir)
    results_path = out_dir / "results_xd_violence.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)

    # latest symlink
    latest_link = Path(_log_base) / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(out_dir)
    except OSError:
        pass

    logger.info(f"\nResults saved to {results_path}")
    return metrics


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}

    # Video-level metrics
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

    # Frame-level AUC-ROC & AP
    frame_auc = 0.0
    frame_ap = 0.0
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
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
            frame_ap = average_precision_score(all_gt, all_pred)
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

    # Per-category stats
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"total": 0, "correct": 0, "tp": 0, "fn": 0, "fp": 0})
    for r in results:
        cat = _extract_primary_category(r.get("label", "unknown"))
        cat_stats[cat]["total"] += 1
        if r.get("pred_anomaly") == r.get("gt_anomaly"):
            cat_stats[cat]["correct"] += 1
        if r.get("gt_anomaly") and r.get("pred_anomaly"):
            cat_stats[cat]["tp"] += 1
        if r.get("gt_anomaly") and not r.get("pred_anomaly"):
            cat_stats[cat]["fn"] += 1
        if not r.get("gt_anomaly") and r.get("pred_anomaly"):
            cat_stats[cat]["fp"] += 1

    for cat, s in cat_stats.items():
        s["accuracy"] = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "frame_auc": round(frame_auc, 4),
        "frame_ap": round(frame_ap, 4),
        "video_auc": round(video_auc, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "total": total,
        "category_stats": dict(cat_stats),
    }


def print_metrics(metrics: dict):
    print(f"\n{'='*70}")
    print(f"  V5 Tube-Skeleton — XD-Violence Evaluation Results")
    print(f"{'='*70}")
    print(f"  Video-level Accuracy:  {metrics['accuracy']:.4f} ({metrics['tp']+metrics['tn']}/{metrics['total']})")
    print(f"  Precision:             {metrics['precision']:.4f}")
    print(f"  Recall:                {metrics['recall']:.4f}")
    print(f"  F1 Score:              {metrics['f1']:.4f}")
    print(f"  ★ Frame-level AUC-ROC: {metrics.get('frame_auc', 0):.4f}")
    print(f"  ★ Frame-level AP:      {metrics.get('frame_ap', 0):.4f}")
    print(f"  Video-level AUC-ROC:   {metrics.get('video_auc', 0):.4f}")
    print(f"  TP={metrics['tp']}  FN={metrics['fn']}  FP={metrics['fp']}  TN={metrics['tn']}")
    print(f"  Total time: {metrics.get('total_time_sec', 0)}s "
          f"({metrics.get('avg_time_per_video', 0)}s/video)")
    print(f"\n  Per-category:")
    for cat, s in sorted(metrics.get("category_stats", {}).items()):
        print(f"    {cat:<12} acc={s['accuracy']:.2f} "
              f"({s['correct']}/{s['total']}) "
              f"TP={s['tp']} FN={s['fn']} FP={s['fp']}")
    print(f"{'='*70}\n")


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V5 XD-Violence Evaluation")
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Max videos to evaluate (0=all 800)")
    parser.add_argument("--sample-every", type=int, default=3,
                        help="Process every N-th frame (default 3, ★ 2→3 for speed)")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--parallel", type=int, default=6,
                        help="Parallel video processing count")
    parser.add_argument("--backend", default="server", choices=["server", "local"])
    parser.add_argument("--no-balanced", action="store_true",
                        help="Disable balanced anomaly/normal sampling")
    parser.add_argument("--no-discordance", action="store_true",
                        help="Disable discordance checker")
    parser.add_argument("--yolo", action="store_true",
                        help="Enable YOLO-World hybrid detection")
    parser.add_argument("--cinematic-filter", action="store_true",
                        help="Enable cinematic/movie scene filter to reduce FP on film clips")
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
        disable_discordance=args.no_discordance,
        use_yolo=args.yolo,
        cinematic_filter=args.cinematic_filter,
    )


if __name__ == "__main__":
    main()
