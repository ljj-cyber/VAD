"""
Abuse028 单独运行 + 详细图输出，排查回归原因。
conda run -n eventvad python -m v5.test_abuse028_solo
"""

import json
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

VIDEO = "/data/liuzhe/EventVAD/src/event_seg/videos/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse028_x264.mp4"
GT = [(165, 240)]


def main():
    pipe = TubeSkeletonPipeline(
        api_base="http://localhost:8000",
        max_workers=48,
        backend="server",
    )

    result = pipe.analyze_video(video_path=VIDEO, sample_every_n=2)
    v = result["verdict"]
    s = result["stats"]
    evs = v["entity_verdicts"]
    iou = compute_frame_iou(GT, evs, result["total_frames"], result["fps"], mode="soft")

    print(f"\nVerdict: {'ANOMALY' if v['is_anomaly'] else 'NORMAL'} (conf={v['confidence']:.2f})")
    print(f"Entities: {s['entities']}, Nodes: {s['nodes']}, Edges: {s['edges']}, Triggers: {s['triggers']}")
    print(f"Soft IoU: {iou:.4f}")

    for ev in evs:
        tag = "ANO" if ev["is_anomaly"] else "NOR"
        print(f"  E{ev['entity_id']}: {tag} conf={ev['confidence']:.2f} reason=\"{ev['reason'][:120]}\"")

    graphs = result["graphs"]
    print(f"\nGraph detail:")
    for eid, g in graphs.items():
        nodes = g.get("nodes", [])
        actions = [n["action"] for n in nodes]
        suspicious = [n for n in nodes if n.get("is_suspicious")]
        danger_nodes = [n for n in nodes if n.get("danger_score", 0) > 0.1]
        print(f"  E{eid}: {len(nodes)} nodes | actions: {' → '.join(actions[:12])}")
        if suspicious:
            print(f"    suspicious: {[(n['node_id'], n['timestamp'], n['action']) for n in suspicious]}")
        if danger_nodes:
            print(f"    danger>0.1: {[(n['node_id'], n['danger_score']) for n in danger_nodes]}")


if __name__ == "__main__":
    main()
