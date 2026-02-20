"""
V5 UCF-Crime Evaluation — Comprehensive Result Analysis

Generates:
  1. Overall metrics summary table
  2. Frame-level ROC curve + AUC
  3. Video-level ROC curve + AUC
  4. Confusion matrix heatmap
  5. Per-category accuracy bar chart
  6. Per-category IoU bar chart
  7. Score distribution (anomaly vs normal)
  8. Error analysis (FN / FP breakdown)
  9. Temporal localization quality analysis
  10. Markdown report

Usage:
  python -m v5.analyze_results [--run-dir path/to/run_xxx] [--latest]
"""

import argparse
import json
import logging
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

EVAL_BASE = Path("/data/liuzhe/EventVAD/output/v5/eval_ucf_crime")


def load_results(run_dir: Path) -> dict:
    results_path = run_dir / "results_v5.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results_v5.json in {run_dir}")
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def generate_all_analyses(data: dict, out_dir: Path):
    """Master function: generate all analysis artifacts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    metrics = data["metrics"]
    details = data["details"]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── 1. Overall metrics summary ──
    print_overall_summary(metrics)

    # ── 2. Frame-level ROC ──
    plot_frame_roc(details, metrics, fig_dir)

    # ── 3. Video-level ROC ──
    plot_video_roc(details, metrics, fig_dir)

    # ── 4. Confusion matrix ──
    plot_confusion_matrix(metrics, fig_dir)

    # ── 5. Per-category accuracy ──
    plot_category_accuracy(metrics, fig_dir)

    # ── 6. Per-category IoU ──
    plot_category_iou(metrics, fig_dir)

    # ── 7. Score distribution ──
    plot_score_distribution(details, fig_dir)

    # ── 8. Error analysis ──
    error_analysis = analyze_errors(details)
    save_error_analysis(error_analysis, out_dir)

    # ── 9. Temporal localization quality ──
    plot_temporal_quality(details, fig_dir)

    # ── 10. Dashboard (combined figure) ──
    plot_dashboard(metrics, details, fig_dir)

    # ── 11. Markdown report ──
    generate_markdown_report(metrics, details, error_analysis, out_dir)

    print(f"\nAll analysis artifacts saved to: {out_dir}")
    print(f"  Figures: {fig_dir}")


def print_overall_summary(metrics: dict):
    print(f"\n{'='*70}")
    print(f"  V5 Tube-Skeleton — UCF-Crime Full Evaluation Results")
    print(f"{'='*70}")
    print(f"  Total Videos:          {metrics['total']}")
    print(f"  Video-level Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision:             {metrics['precision']:.4f}")
    print(f"  Recall:                {metrics['recall']:.4f}")
    print(f"  F1 Score:              {metrics['f1']:.4f}")
    print(f"  ★ Frame-level AUC-ROC: {metrics.get('frame_auc', 0):.4f}")
    print(f"  Video-level AUC-ROC:   {metrics.get('video_auc', 0):.4f}")
    print(f"  Mean Soft IoU:         {metrics.get('mean_iou_soft', 0):.4f}")
    print(f"  Mean Hysteresis IoU:   {metrics.get('mean_iou_hysteresis', 0):.4f}")
    print(f"  TP={metrics['tp']}  FN={metrics['fn']}  FP={metrics['fp']}  TN={metrics['tn']}")
    print(f"  Total time: {metrics.get('total_time_sec', 0)}s "
          f"({metrics.get('avg_time_per_video', 0)}s/video)")
    print(f"{'='*70}\n")


# ── ROC Curves ──

def _build_frame_arrays(details: list):
    """Build frame-level GT and prediction arrays from results."""
    from v5.eval_ucf_crime import build_gt_mask, compute_frame_scores
    all_gt, all_pred = [], []
    for r in details:
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
    return np.array(all_gt), np.array(all_pred)


def plot_frame_roc(details: list, metrics: dict, fig_dir: Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    all_gt, all_pred = _build_frame_arrays(details)
    if len(set(all_gt)) < 2:
        print("  [SKIP] Frame ROC: only one class present")
        return

    fpr, tpr, _ = roc_curve(all_gt, all_pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"Frame AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Frame-level ROC Curve")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "frame_roc.png")
    plt.close(fig)
    print(f"  [OK] Frame ROC saved (AUC={roc_auc:.4f})")


def plot_video_roc(details: list, metrics: dict, fig_dir: Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    vgt = np.array([1 if r.get("gt_anomaly") else 0 for r in details])
    vps = np.array([r.get("pred_score", 0.0) for r in details])
    if len(set(vgt)) < 2:
        print("  [SKIP] Video ROC: only one class present")
        return

    fpr, tpr, thresholds = roc_curve(vgt, vps)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#FF5722", lw=2, label=f"Video AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], c="red", s=80, zorder=5,
               label=f"Optimal threshold = {optimal_threshold:.2f}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Video-level ROC Curve")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "video_roc.png")
    plt.close(fig)
    print(f"  [OK] Video ROC saved (AUC={roc_auc:.4f})")


# ── Confusion Matrix ──

def plot_confusion_matrix(metrics: dict, fig_dir: Path):
    import matplotlib.pyplot as plt

    tp, fn, fp, tn = metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["Normal", "Anomaly"]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    ax.set_yticklabels(["GT Normal", "GT Anomaly"])
    ax.set_title("Video-level Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png")
    plt.close(fig)
    print(f"  [OK] Confusion matrix saved")


# ── Per-Category Charts ──

def plot_category_accuracy(metrics: dict, fig_dir: Path):
    import matplotlib.pyplot as plt

    cat_stats = metrics.get("category_stats", {})
    if not cat_stats:
        return

    cats = sorted(cat_stats.keys())
    accs = [cat_stats[c]["accuracy"] for c in cats]
    totals = [cat_stats[c]["total"] for c in cats]

    colors = ["#4CAF50" if c != "Normal" else "#9E9E9E" for c in cats]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(cats)), accs, color=colors, edgecolor="white", linewidth=0.5)

    for i, (bar, total) in enumerate(zip(bars, totals)):
        correct = cat_stats[cats[i]]["correct"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{correct}/{total}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Category Video-level Accuracy")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=metrics["accuracy"], color="red", linestyle="--", alpha=0.7,
               label=f"Overall: {metrics['accuracy']:.2f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "category_accuracy.png")
    plt.close(fig)
    print(f"  [OK] Category accuracy chart saved")


def plot_category_iou(metrics: dict, fig_dir: Path):
    import matplotlib.pyplot as plt

    cat_stats = metrics.get("category_stats", {})
    anomaly_cats = {c: s for c, s in cat_stats.items() if c != "Normal"}
    if not anomaly_cats:
        return

    cats = sorted(anomaly_cats.keys())
    soft_ious = [anomaly_cats[c].get("mean_iou_soft", 0) for c in cats]
    hyst_ious = [anomaly_cats[c].get("mean_iou_hyst", 0) for c in cats]

    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, soft_ious, width, label="Soft IoU", color="#2196F3")
    bars2 = ax.bar(x + width / 2, hyst_ious, width, label="Hysteresis IoU", color="#FF9800")

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Per-Category Temporal Localization IoU (Anomaly Classes Only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(soft_ious, default=0), max(hyst_ious, default=0)) * 1.3 + 0.05)

    overall_soft = metrics.get("mean_iou_soft", 0)
    ax.axhline(y=overall_soft, color="blue", linestyle="--", alpha=0.5,
               label=f"Overall Soft IoU: {overall_soft:.3f}")

    fig.tight_layout()
    fig.savefig(fig_dir / "category_iou.png")
    plt.close(fig)
    print(f"  [OK] Category IoU chart saved")


# ── Score Distribution ──

def plot_score_distribution(details: list, fig_dir: Path):
    import matplotlib.pyplot as plt

    anomaly_scores = [r["pred_score"] for r in details if r.get("gt_anomaly")]
    normal_scores = [r["pred_score"] for r in details if not r.get("gt_anomaly")]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(normal_scores, bins=bins, alpha=0.6, color="#4CAF50", label=f"Normal (n={len(normal_scores)})")
    ax.hist(anomaly_scores, bins=bins, alpha=0.6, color="#F44336", label=f"Anomaly (n={len(anomaly_scores)})")
    ax.set_xlabel("Predicted Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Video-level Score Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "score_distribution.png")
    plt.close(fig)
    print(f"  [OK] Score distribution saved")


# ── Error Analysis ──

def analyze_errors(details: list) -> dict:
    fn_cases = []  # GT=anomaly, Pred=normal (missed anomalies)
    fp_cases = []  # GT=normal, Pred=anomaly (false alarms)
    tp_cases = []
    tn_cases = []

    for r in details:
        gt = r.get("gt_anomaly", False)
        pred = r.get("pred_anomaly", False)
        entry = {
            "filename": r.get("filename", ""),
            "category": r.get("category", ""),
            "pred_score": r.get("pred_score", 0.0),
            "entities": r.get("stats", {}).get("entities", 0),
            "triggers": r.get("stats", {}).get("triggers", 0),
            "iou_soft": r.get("iou_soft"),
            "iou_hysteresis": r.get("iou_hysteresis"),
            "time_sec": r.get("time_sec", 0),
        }
        if gt and not pred:
            fn_cases.append(entry)
        elif not gt and pred:
            fp_cases.append(entry)
        elif gt and pred:
            tp_cases.append(entry)
        else:
            tn_cases.append(entry)

    fn_by_cat = defaultdict(list)
    for c in fn_cases:
        fn_by_cat[c["category"]].append(c)

    fp_by_cat = defaultdict(list)
    for c in fp_cases:
        fp_by_cat[c["category"]].append(c)

    return {
        "fn_cases": fn_cases,
        "fp_cases": fp_cases,
        "tp_cases": tp_cases,
        "tn_cases": tn_cases,
        "fn_by_category": dict(fn_by_cat),
        "fp_by_category": dict(fp_by_cat),
        "fn_zero_entity": [c for c in fn_cases if c["entities"] == 0],
        "fn_with_entities": [c for c in fn_cases if c["entities"] > 0],
    }


def save_error_analysis(error_analysis: dict, out_dir: Path):
    with open(out_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)

    print(f"\n  Error Analysis:")
    print(f"    False Negatives (missed anomalies): {len(error_analysis['fn_cases'])}")
    fn_cats = defaultdict(int)
    for c in error_analysis["fn_cases"]:
        fn_cats[c["category"]] += 1
    for cat, n in sorted(fn_cats.items(), key=lambda x: -x[1]):
        print(f"      {cat}: {n}")

    print(f"    False Positives (false alarms): {len(error_analysis['fp_cases'])}")
    for c in error_analysis["fp_cases"]:
        print(f"      {c['filename']} (score={c['pred_score']:.2f})")

    print(f"    FN with 0 entities: {len(error_analysis['fn_zero_entity'])}")
    print(f"    FN with entities but wrong verdict: {len(error_analysis['fn_with_entities'])}")


# ── Temporal Quality ──

def plot_temporal_quality(details: list, fig_dir: Path):
    import matplotlib.pyplot as plt

    tp_cases = [r for r in details if r.get("gt_anomaly") and r.get("pred_anomaly")]
    if not tp_cases:
        print("  [SKIP] Temporal quality: no TP cases")
        return

    soft_ious = [r.get("iou_soft", 0) or 0 for r in tp_cases]
    hyst_ious = [r.get("iou_hysteresis", 0) or 0 for r in tp_cases]
    filenames = [r.get("filename", "")[:20] for r in tp_cases]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(soft_ious, bins=20, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("Soft IoU")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Soft IoU Distribution (TP cases, n={len(tp_cases)})")
    axes[0].axvline(x=np.mean(soft_ious), color="red", linestyle="--",
                     label=f"Mean={np.mean(soft_ious):.3f}")
    axes[0].legend()

    axes[1].hist(hyst_ious, bins=20, color="#FF9800", alpha=0.7, edgecolor="white")
    axes[1].set_xlabel("Hysteresis IoU")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Hysteresis IoU Distribution (TP cases, n={len(tp_cases)})")
    axes[1].axvline(x=np.mean(hyst_ious), color="red", linestyle="--",
                     label=f"Mean={np.mean(hyst_ious):.3f}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "temporal_quality.png")
    plt.close(fig)
    print(f"  [OK] Temporal quality chart saved")


# ── Dashboard ──

def plot_dashboard(metrics: dict, details: list, fig_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from sklearn.metrics import roc_curve, auc

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (0,0) Video ROC
    ax1 = fig.add_subplot(gs[0, 0])
    vgt = np.array([1 if r.get("gt_anomaly") else 0 for r in details])
    vps = np.array([r.get("pred_score", 0.0) for r in details])
    if len(set(vgt)) >= 2:
        fpr, tpr, _ = roc_curve(vgt, vps)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color="#FF5722", lw=2, label=f"AUC={roc_auc:.4f}")
        ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_title("Video-level ROC")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # (0,1) Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    tp, fn, fp, tn = metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax2.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=18, fontweight="bold", color=color)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    ax2.set_yticklabels(["GT Normal", "GT Anomaly"])
    ax2.set_title("Confusion Matrix")

    # (0,2) Metrics text
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    text_lines = [
        f"Total Videos: {metrics['total']}",
        f"",
        f"Accuracy:    {metrics['accuracy']:.4f}",
        f"Precision:   {metrics['precision']:.4f}",
        f"Recall:      {metrics['recall']:.4f}",
        f"F1 Score:    {metrics['f1']:.4f}",
        f"",
        f"Frame AUC:   {metrics.get('frame_auc', 0):.4f}",
        f"Video AUC:   {metrics.get('video_auc', 0):.4f}",
        f"",
        f"Soft IoU:    {metrics.get('mean_iou_soft', 0):.4f}",
        f"Hyst IoU:    {metrics.get('mean_iou_hysteresis', 0):.4f}",
        f"",
        f"TP={tp}  FN={fn}  FP={fp}  TN={tn}",
    ]
    ax3.text(0.1, 0.95, "\n".join(text_lines), transform=ax3.transAxes,
             fontsize=12, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.8))
    ax3.set_title("Overall Metrics")

    # (1,0) Score distribution
    ax4 = fig.add_subplot(gs[1, 0])
    anomaly_scores = [r["pred_score"] for r in details if r.get("gt_anomaly")]
    normal_scores = [r["pred_score"] for r in details if not r.get("gt_anomaly")]
    bins = np.linspace(0, 1, 25)
    ax4.hist(normal_scores, bins=bins, alpha=0.6, color="#4CAF50", label="Normal")
    ax4.hist(anomaly_scores, bins=bins, alpha=0.6, color="#F44336", label="Anomaly")
    ax4.set_xlabel("Predicted Score")
    ax4.set_ylabel("Count")
    ax4.set_title("Score Distribution")
    ax4.legend()

    # (1,1) Per-category accuracy
    ax5 = fig.add_subplot(gs[1, 1])
    cat_stats = metrics.get("category_stats", {})
    cats = sorted(cat_stats.keys())
    accs = [cat_stats[c]["accuracy"] for c in cats]
    colors = ["#4CAF50" if c != "Normal" else "#9E9E9E" for c in cats]
    ax5.barh(range(len(cats)), accs, color=colors)
    ax5.set_yticks(range(len(cats)))
    ax5.set_yticklabels(cats, fontsize=9)
    ax5.set_xlabel("Accuracy")
    ax5.set_title("Per-Category Accuracy")
    ax5.axvline(x=metrics["accuracy"], color="red", linestyle="--", alpha=0.7)
    ax5.set_xlim(0, 1.1)

    # (1,2) Per-category IoU
    ax6 = fig.add_subplot(gs[1, 2])
    anomaly_cats = sorted([c for c in cats if c != "Normal"])
    if anomaly_cats:
        soft_ious = [cat_stats[c].get("mean_iou_soft", 0) for c in anomaly_cats]
        hyst_ious = [cat_stats[c].get("mean_iou_hyst", 0) for c in anomaly_cats]
        x = np.arange(len(anomaly_cats))
        w = 0.35
        ax6.barh(x - w / 2, soft_ious, w, label="Soft IoU", color="#2196F3")
        ax6.barh(x + w / 2, hyst_ious, w, label="Hyst IoU", color="#FF9800")
        ax6.set_yticks(x)
        ax6.set_yticklabels(anomaly_cats, fontsize=9)
        ax6.set_xlabel("IoU")
        ax6.set_title("Per-Category IoU")
        ax6.legend(fontsize=9)

    fig.suptitle("V5 Tube-Skeleton — UCF-Crime Evaluation Dashboard", fontsize=16, y=1.01)
    fig.savefig(fig_dir / "dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Dashboard saved")


# ── Markdown Report ──

def generate_markdown_report(metrics: dict, details: list, error_analysis: dict, out_dir: Path):
    cat_stats = metrics.get("category_stats", {})

    lines = [
        "# V5 Tube-Skeleton — UCF-Crime Evaluation Report",
        "",
        "## 1. Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Videos | {metrics['total']} |",
        f"| Video-level Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 Score | {metrics['f1']:.4f} |",
        f"| **Frame-level AUC-ROC** | **{metrics.get('frame_auc', 0):.4f}** |",
        f"| Video-level AUC-ROC | {metrics.get('video_auc', 0):.4f} |",
        f"| Mean Soft IoU | {metrics.get('mean_iou_soft', 0):.4f} |",
        f"| Mean Hysteresis IoU | {metrics.get('mean_iou_hysteresis', 0):.4f} |",
        "",
        "## 2. Confusion Matrix",
        "",
        f"| | Pred Normal | Pred Anomaly |",
        f"|---|---|---|",
        f"| GT Normal | TN={metrics['tn']} | FP={metrics['fp']} |",
        f"| GT Anomaly | FN={metrics['fn']} | TP={metrics['tp']} |",
        "",
        "## 3. Per-Category Performance",
        "",
        "| Category | Total | Correct | Accuracy | Soft IoU | Hyst IoU |",
        "|----------|-------|---------|----------|----------|----------|",
    ]

    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        siou = f"{s.get('mean_iou_soft', 0):.3f}" if s.get("mean_iou_soft") else "N/A"
        hiou = f"{s.get('mean_iou_hyst', 0):.3f}" if s.get("mean_iou_hyst") else "N/A"
        lines.append(
            f"| {cat} | {s['total']} | {s['correct']} | {s['accuracy']:.2f} | {siou} | {hiou} |"
        )

    lines.extend([
        "",
        "## 4. Error Analysis",
        "",
        f"### False Negatives (Missed Anomalies): {len(error_analysis['fn_cases'])}",
        "",
    ])

    if error_analysis["fn_cases"]:
        lines.append("| Filename | Category | Score | Entities | Triggers |")
        lines.append("|----------|----------|-------|----------|----------|")
        for c in sorted(error_analysis["fn_cases"], key=lambda x: x["category"]):
            lines.append(
                f"| {c['filename']} | {c['category']} | {c['pred_score']:.2f} | "
                f"{c['entities']} | {c['triggers']} |"
            )

    lines.extend([
        "",
        f"### False Positives (False Alarms): {len(error_analysis['fp_cases'])}",
        "",
    ])

    if error_analysis["fp_cases"]:
        lines.append("| Filename | Score | Entities | Triggers |")
        lines.append("|----------|-------|----------|----------|")
        for c in error_analysis["fp_cases"]:
            lines.append(
                f"| {c['filename']} | {c['pred_score']:.2f} | {c['entities']} | {c['triggers']} |"
            )

    fn_zero = error_analysis["fn_zero_entity"]
    fn_with = error_analysis["fn_with_entities"]
    lines.extend([
        "",
        "### FN Root Cause Breakdown",
        "",
        f"- **Zero entities detected** (tracking failure): {len(fn_zero)} videos",
        f"- **Entities detected but wrong verdict** (semantic/decision failure): {len(fn_with)} videos",
        "",
    ])

    # Per-category FN analysis
    fn_by_cat = error_analysis.get("fn_by_category", {})
    if fn_by_cat:
        lines.append("FN by category:")
        lines.append("")
        for cat, cases in sorted(fn_by_cat.items(), key=lambda x: -len(x[1])):
            lines.append(f"- **{cat}**: {len(cases)} missed")

    lines.extend([
        "",
        "## 5. Figures",
        "",
        "- `figures/dashboard.png` — Combined dashboard",
        "- `figures/frame_roc.png` — Frame-level ROC curve",
        "- `figures/video_roc.png` — Video-level ROC curve",
        "- `figures/confusion_matrix.png` — Confusion matrix",
        "- `figures/category_accuracy.png` — Per-category accuracy",
        "- `figures/category_iou.png` — Per-category IoU",
        "- `figures/score_distribution.png` — Score distribution",
        "- `figures/temporal_quality.png` — Temporal localization quality",
        "",
    ])

    report_path = out_dir / "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] Markdown report saved to {report_path}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="V5 UCF-Crime Result Analysis")
    parser.add_argument("--run-dir", type=str, default="",
                        help="Path to specific run directory")
    parser.add_argument("--latest", action="store_true",
                        help="Use the latest run (default)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        latest = EVAL_BASE / "latest"
        if latest.is_symlink() or latest.exists():
            run_dir = latest.resolve()
        else:
            runs = sorted(EVAL_BASE.glob("run_*"))
            if not runs:
                print("No evaluation runs found!")
                return
            run_dir = runs[-1]

    print(f"Analyzing: {run_dir}")
    data = load_results(run_dir)

    analysis_dir = run_dir / "analysis"
    generate_all_analyses(data, analysis_dir)


if __name__ == "__main__":
    main()
