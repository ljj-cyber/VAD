"""
三案例完整回归测试。
conda run -n eventvad python -m v5.test_all3
"""

import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
for mod in ["httpx", "v5.tracking.clip_encoder", "sentence_transformers"]:
    logging.getLogger(mod).setLevel(logging.WARNING)

from v5.pipeline import TubeSkeletonPipeline
from v5.eval_ucf_crime import compute_frame_iou

UCF = Path("/data/liuzhe/EventVAD/src/event_seg/videos/ucf_crime")
CASES = [
    ("Abuse028",    UCF / "Anomaly-Videos-Part-1/Abuse/Abuse028_x264.mp4",      [(165, 240)]),
    ("Abuse030",    UCF / "Anomaly-Videos-Part-1/Abuse/Abuse030_x264.mp4",      [(1275, 1360)]),
    ("Fighting018", UCF / "Anomaly-Videos-Part-2/Fighting/Fighting018_x264.mp4", [(80, 420)]),
]

BASELINE = {
    "Abuse028":    {"pred": True,  "conf": 0.90, "ent": 5, "nodes": 31, "iou": 0.133},
    "Abuse030":    {"pred": False, "conf": 0.00, "ent": 0, "nodes": 0,  "iou": 0.000},
    "Fighting018": {"pred": True,  "conf": 0.95, "ent": 7, "nodes": 33, "iou": 0.354},
}


def main():
    pipe = TubeSkeletonPipeline(
        api_base="http://localhost:8000",
        max_workers=48,
        backend="server",
    )

    results = []
    for name, path, gt in CASES:
        t0 = time.time()
        result = pipe.analyze_video(video_path=str(path), sample_every_n=2)
        elapsed = time.time() - t0

        v = result["verdict"]
        s = result["stats"]
        evs = v["entity_verdicts"]
        iou = compute_frame_iou(gt, evs, result["total_frames"], result["fps"], mode="soft")

        results.append({
            "name": name, "pred": v["is_anomaly"], "conf": v["confidence"],
            "ent": s["entities"], "nodes": s["nodes"], "edges": s["edges"],
            "iou": iou, "time": elapsed,
        })

    print(f"\n{'='*80}")
    print(f"{'Name':<15} {'Pred':>6} {'Conf':>5} {'E':>3} {'N':>3} {'IoU':>6} {'Time':>5}  {'vs Baseline'}")
    print(f"{'-'*80}")
    for r in results:
        b = BASELINE[r["name"]]
        status = "OK" if r["pred"] == b["pred"] else "REGRESS"
        ent_diff = r["ent"] - b["ent"]
        iou_diff = r["iou"] - b["iou"]
        print(f"{r['name']:<15} {'ANO' if r['pred'] else 'NOR':>6} {r['conf']:>5.2f} "
              f"{r['ent']:>3} {r['nodes']:>3} {r['iou']:>6.3f} {r['time']:>5.0f}s "
              f" {status} (E{ent_diff:+d} IoU{iou_diff:+.3f})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
