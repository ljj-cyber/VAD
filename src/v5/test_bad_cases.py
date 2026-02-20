"""
Bad case 验证脚本 — 针对性跑几个 Abuse / Fighting 视频，
详细输出动态图结构 (Nodes, Edges, Action Sequence)，
验证 V5 动态图是否正常工作。

用法:
  conda run -n eventvad python -m v5.test_bad_cases
"""

import json
import logging
import os
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

logger = logging.getLogger("test_bad_cases")

from v5.pipeline import TubeSkeletonPipeline
from v5.eval_ucf_crime import compute_frame_scores, build_gt_mask, compute_frame_iou

UCF_ROOT = Path("/data/liuzhe/EventVAD/src/event_seg/videos/ucf_crime")

BAD_CASES = [
    {
        "name": "Abuse028",
        "path": UCF_ROOT / "Anomaly-Videos-Part-1/Abuse/Abuse028_x264.mp4",
        "category": "Abuse",
        "gt_intervals": [(165, 240)],
    },
    {
        "name": "Abuse030",
        "path": UCF_ROOT / "Anomaly-Videos-Part-1/Abuse/Abuse030_x264.mp4",
        "category": "Abuse",
        "gt_intervals": [(1275, 1360)],
    },
    {
        "name": "Fighting018",
        "path": UCF_ROOT / "Anomaly-Videos-Part-2/Fighting/Fighting018_x264.mp4",
        "category": "Fighting",
        "gt_intervals": [(80, 420)],
    },
]

OUT_DIR = Path("/data/liuzhe/EventVAD/output/v5/bad_case_test")
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def dump_graph_detail(graphs: dict, name: str):
    """打印动态图详细结构"""
    print(f"\n{'─'*60}")
    print(f"  Dynamic Graph Detail: {name}")
    print(f"{'─'*60}")
    print(f"  Total entities: {len(graphs)}")

    for eid, g_dict in graphs.items():
        nodes = g_dict.get("nodes", [])
        edges = g_dict.get("edges", [])
        print(f"\n  ╔═ Entity {eid} ═══════════════════════════════")
        print(f"  ║ Nodes: {len(nodes)}, Edges: {len(edges)}")
        print(f"  ║ Duration: {g_dict.get('birth_time', 0):.1f}s → {g_dict.get('last_time', 0):.1f}s "
              f"({g_dict.get('total_duration', 0):.1f}s)")
        print(f"  ║ Max danger: {g_dict.get('max_danger_score', 0):.2f}, "
              f"Has suspicious: {g_dict.get('has_suspicious', False)}")
        print(f"  ║ Total kinetic integral: {g_dict.get('total_kinetic_integral', 0):.4f}")

        actions = [n["action"] for n in nodes]
        print(f"  ║ Action sequence: {' → '.join(actions[:15])}")
        if len(actions) > 15:
            print(f"  ║   ... ({len(actions) - 15} more)")

        print(f"  ║")
        print(f"  ║ Nodes:")
        for n in nodes:
            susp = " ⚠SUSPICIOUS" if n.get("is_suspicious") else ""
            print(f"  ║   {n['node_id']} t={n['timestamp']:.1f}s "
                  f"action=\"{n['action']}\" obj=\"{n['action_object']}\" "
                  f"danger={n['danger_score']:.2f} "
                  f"trigger={n['trigger_rule']}{susp}")

        if edges:
            print(f"  ║")
            print(f"  ║ Edges:")
            for e in edges:
                print(f"  ║   {e['source']} → {e['target']} "
                      f"dt={e['duration_sec']:.1f}s "
                      f"KE∫={e['kinetic_integral']:.4f} "
                      f"transition=\"{e['action_transition']}\" "
                      f"missing={e['missing_frames']}")

        print(f"  ╚{'═'*45}")


def run_one(case: dict, pipe: TubeSkeletonPipeline) -> dict:
    """跑单个 bad case"""
    name = case["name"]
    path = str(case["path"])
    gt_intervals = case["gt_intervals"]

    print(f"\n{'='*60}")
    print(f"  Running: {name} ({case['category']})")
    print(f"  GT anomaly frames: {gt_intervals}")
    print(f"{'='*60}")

    if not Path(path).exists():
        print(f"  ❌ Video not found: {path}")
        return {"name": name, "error": "not found"}

    t0 = time.time()
    result = pipe.analyze_video(video_path=path, sample_every_n=2)
    elapsed = time.time() - t0

    verdict = result["verdict"]
    entity_verdicts = verdict["entity_verdicts"]
    total_frames = result["total_frames"]
    fps = result["fps"]

    # 计算 IoU
    iou_soft = compute_frame_iou(gt_intervals, entity_verdicts, total_frames, fps, mode="soft")
    iou_hyst = compute_frame_iou(gt_intervals, entity_verdicts, total_frames, fps, mode="hysteresis")

    # 输出 Verdict
    status = "ANOMALY" if verdict["is_anomaly"] else "NORMAL"
    print(f"\n  ★ Verdict: {status} (conf={verdict['confidence']:.3f})")
    print(f"  Summary: {verdict['summary']}")
    print(f"  Scene: {verdict['scene_type']}")
    print(f"  Anomaly entities: {verdict['anomaly_entity_ids']}")
    print(f"  Soft IoU: {iou_soft:.4f}")
    print(f"  Hysteresis IoU: {iou_hyst:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Entity verdicts
    print(f"\n  Entity Verdicts:")
    for ev in entity_verdicts:
        a_tag = "⚠" if ev["is_anomaly"] else "✓"
        print(f"    {a_tag} Entity {ev['entity_id']}: anomaly={ev['is_anomaly']} "
              f"conf={ev['confidence']:.2f} "
              f"window=[{ev['anomaly_start_sec']:.1f}s, {ev['anomaly_end_sec']:.1f}s] "
              f"reason=\"{ev['reason']}\"")

    # Pipeline stats
    stats = result["stats"]
    timing = result["timing"]
    print(f"\n  Stats: entities={stats['entities']} triggers={stats['triggers']} "
          f"nodes={stats['nodes']} edges={stats['edges']}")
    print(f"  Timing: tracking={timing['tracking_sec']:.1f}s "
          f"semantic={timing['semantic_sec']:.1f}s "
          f"decision={timing['decision_sec']:.1f}s")

    # 动态图详情
    dump_graph_detail(result["graphs"], name)

    # 保存完整 JSON
    out_path = OUT_DIR / f"{RUN_TS}_{name}.json"
    for entry in result.get("trace_log", []):
        entry.pop("embedding", None)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Result saved to: {out_path}")

    return {
        "name": name,
        "category": case["category"],
        "gt_intervals": gt_intervals,
        "pred_anomaly": verdict["is_anomaly"],
        "confidence": verdict["confidence"],
        "iou_soft": iou_soft,
        "iou_hyst": iou_hyst,
        "entities": stats["entities"],
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "triggers": stats["triggers"],
        "time_sec": round(elapsed, 1),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = TubeSkeletonPipeline(
        api_base="http://localhost:8000",
        max_workers=48,
        backend="server",
        disable_discordance=False,
        use_yolo=False,
    )

    all_results = []
    for case in BAD_CASES:
        try:
            r = run_one(case, pipe)
            all_results.append(r)
        except Exception as e:
            logger.error(f"Failed on {case['name']}: {e}", exc_info=True)
            all_results.append({"name": case["name"], "error": str(e)})

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  Bad Case Test Summary")
    print(f"{'='*70}")
    for r in all_results:
        if "error" in r:
            print(f"  ❌ {r['name']}: {r['error']}")
            continue
        status = "✅" if r["pred_anomaly"] else "❌"
        print(f"  {status} {r['name']} ({r['category']}): "
              f"anomaly={r['pred_anomaly']} conf={r['confidence']:.2f} "
              f"SoftIoU={r['iou_soft']:.3f} HystIoU={r['iou_hyst']:.3f} "
              f"E={r['entities']} N={r['nodes']} Ed={r['edges']} "
              f"{r['time_sec']}s")
    print(f"{'='*70}")

    summary_path = OUT_DIR / f"{RUN_TS}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
