"""
回归测试 — 确认修复 EntityTracker 不影响 Abuse028 和 Fighting018。

conda run -n eventvad python -m v5.test_regression
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
    ("Abuse028", UCF / "Anomaly-Videos-Part-1/Abuse/Abuse028_x264.mp4", [(165, 240)]),
    ("Fighting018", UCF / "Anomaly-Videos-Part-2/Fighting/Fighting018_x264.mp4", [(80, 420)]),
]


def main():
    pipe = TubeSkeletonPipeline(
        api_base="http://localhost:8000",
        max_workers=48,
        backend="server",
    )

    for name, path, gt in CASES:
        t0 = time.time()
        result = pipe.analyze_video(video_path=str(path), sample_every_n=2)
        elapsed = time.time() - t0

        v = result["verdict"]
        s = result["stats"]
        evs = v["entity_verdicts"]
        iou = compute_frame_iou(gt, evs, result["total_frames"], result["fps"], mode="soft")

        status = "ANOMALY" if v["is_anomaly"] else "NORMAL"
        print(f"{name}: {status} conf={v['confidence']:.2f} "
              f"E={s['entities']} N={s['nodes']} IoU={iou:.3f} {elapsed:.0f}s")


if __name__ == "__main__":
    main()
