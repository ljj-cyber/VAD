"""
Abuse030 修复验证 — 单独跑 Abuse030 确认 YOLO regions 能创建 entity。

conda run -n eventvad python -m v5.test_abuse030_fix
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
for mod in ["httpx", "v5.tracking.clip_encoder", "sentence_transformers"]:
    logging.getLogger(mod).setLevel(logging.WARNING)

logger = logging.getLogger("test_fix")

from v5.pipeline import TubeSkeletonPipeline
from v5.eval_ucf_crime import compute_frame_iou

VIDEO = "/data/liuzhe/EventVAD/src/event_seg/videos/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse030_x264.mp4"
GT_INTERVALS = [(1275, 1360)]
OUT_DIR = Path("/data/liuzhe/EventVAD/output/v5/bad_case_test")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = TubeSkeletonPipeline(
        api_base="http://localhost:8000",
        max_workers=48,
        backend="server",
        disable_discordance=False,
        use_yolo=False,
    )

    print(f"{'='*60}")
    print(f"  Abuse030 — Fix verification")
    print(f"  GT anomaly: frames 1275-1360 (42.5s - 45.3s)")
    print(f"{'='*60}")

    t0 = time.time()
    result = pipe.analyze_video(video_path=VIDEO, sample_every_n=2)
    elapsed = time.time() - t0

    verdict = result["verdict"]
    entity_verdicts = verdict["entity_verdicts"]
    total_frames = result["total_frames"]
    fps = result["fps"]
    stats = result["stats"]

    iou_soft = compute_frame_iou(GT_INTERVALS, entity_verdicts, total_frames, fps, mode="soft")
    iou_hyst = compute_frame_iou(GT_INTERVALS, entity_verdicts, total_frames, fps, mode="hysteresis")

    status = "ANOMALY" if verdict["is_anomaly"] else "NORMAL"
    print(f"\n  Verdict: {status} (conf={verdict['confidence']:.3f})")
    print(f"  Summary: {verdict['summary'][:200]}")
    print(f"  Entities: {stats['entities']}, Nodes: {stats['nodes']}, Edges: {stats['edges']}")
    print(f"  Triggers: {stats['triggers']}")
    print(f"  Soft IoU: {iou_soft:.4f}, Hysteresis IoU: {iou_hyst:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    if entity_verdicts:
        print(f"\n  Entity Verdicts:")
        for ev in entity_verdicts:
            tag = "ANOMALY" if ev["is_anomaly"] else "NORMAL"
            print(f"    Entity {ev['entity_id']}: {tag} conf={ev['confidence']:.2f} "
                  f"[{ev['anomaly_start_sec']:.1f}s, {ev['anomaly_end_sec']:.1f}s] "
                  f"reason=\"{ev['reason'][:100]}\"")

    # 打印动态图
    graphs = result["graphs"]
    print(f"\n  Dynamic Graph: {len(graphs)} entities")
    for eid, g in graphs.items():
        nodes = g.get("nodes", [])
        edges = g.get("edges", [])
        actions = [n["action"] for n in nodes]
        print(f"    Entity {eid}: {len(nodes)} nodes, {len(edges)} edges")
        print(f"      Actions: {' → '.join(actions[:10])}")
        suspicious = [n for n in nodes if n.get("is_suspicious")]
        if suspicious:
            print(f"      Suspicious nodes: {len(suspicious)}")

    # 对比修复前后
    print(f"\n{'='*60}")
    print(f"  Before fix: 0 entities, 0 nodes, NORMAL (conf=0.00)")
    print(f"  After fix:  {stats['entities']} entities, {stats['nodes']} nodes, "
          f"{status} (conf={verdict['confidence']:.3f})")
    print(f"{'='*60}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{ts}_Abuse030_fix.json"
    for entry in result.get("trace_log", []):
        entry.pop("embedding", None)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
