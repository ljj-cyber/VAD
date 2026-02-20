"""
快速验证动态图 pipeline (GraphBuilder → NarrativeGenerator → Decision Prompt)

不需要 GPU、视频或 vLLM server，纯 CPU mock 数据即可运行。

用法:
    python tests/test_graph_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from v5.graph.graph_builder import GraphBuilder
from v5.graph.narrative_generator import NarrativeGenerator
from v5.graph.structures import EntityGraph
from v5.graph.decision_prompt import (
    DecisionAuditor, DECISION_SYSTEM_PROMPT, DECISION_USER_TEMPLATE,
)
from v5.tracking.entity_tracker import TraceEntry
from v5.config import GraphConfig, NarrativeConfig, DecisionConfig

# ── 构造 mock 数据: 模拟一个 "正常行走 → 突然打架" 的实体 ──

ENTITY_ID = 1

MOCK_SEMANTIC_RESULTS = [
    {
        "entity_id": ENTITY_ID,
        "frame_idx": 30,
        "timestamp": 1.0,
        "action": "walking",
        "action_object": "none",
        "posture": "standing",
        "scene_context": "parking lot",
        "is_suspicious": False,
        "danger_score": 0.0,
        "anomaly_category_guess": "none",
        "trigger_rule": "birth",
    },
    {
        "entity_id": ENTITY_ID,
        "frame_idx": 120,
        "timestamp": 4.0,
        "action": "walking",
        "action_object": "none",
        "posture": "standing",
        "scene_context": "parking lot",
        "is_suspicious": False,
        "danger_score": 0.0,
        "anomaly_category_guess": "none",
        "trigger_rule": "heartbeat",
    },
    {
        "entity_id": ENTITY_ID,
        "frame_idx": 240,
        "timestamp": 8.0,
        "action": "approaching another person aggressively",
        "action_object": "person",
        "posture": "lunging",
        "scene_context": "parking lot",
        "is_suspicious": True,
        "danger_score": 0.6,
        "anomaly_category_guess": "fighting",
        "trigger_rule": "change_point",
    },
    {
        "entity_id": ENTITY_ID,
        "frame_idx": 330,
        "timestamp": 11.0,
        "action": "punching",
        "action_object": "person",
        "posture": "crouching",
        "scene_context": "parking lot",
        "is_suspicious": True,
        "danger_score": 0.9,
        "anomaly_category_guess": "fighting",
        "trigger_rule": "change_point",
    },
    {
        "entity_id": ENTITY_ID,
        "frame_idx": 420,
        "timestamp": 14.0,
        "action": "running away",
        "action_object": "none",
        "posture": "running",
        "scene_context": "parking lot",
        "is_suspicious": True,
        "danger_score": 0.7,
        "anomaly_category_guess": "fighting",
        "trigger_rule": "change_point",
    },
]


def make_trace_entries(entity_id: int) -> list[TraceEntry]:
    """生成 mock TraceEntry 序列 (模拟逐帧物理轨迹)"""
    entries = []
    rng = np.random.RandomState(42)
    emb_base = rng.randn(512).astype(np.float32)
    emb_base /= np.linalg.norm(emb_base)

    for frame_idx in range(30, 450, 3):
        ts = frame_idx / 30.0
        if frame_idx < 200:
            ke = 0.05 + rng.rand() * 0.03   # 正常行走
        elif frame_idx < 350:
            ke = 0.30 + rng.rand() * 0.20   # 打架: 高动能
        else:
            ke = 0.15 + rng.rand() * 0.10   # 逃跑

        noise = rng.randn(512).astype(np.float32) * 0.02
        emb = emb_base + noise
        emb /= np.linalg.norm(emb)

        entries.append(TraceEntry(
            frame_idx=frame_idx,
            timestamp=ts,
            entity_id=entity_id,
            bbox=(100 + frame_idx // 5, 80, 70, 90),
            embedding=emb,
            kinetic_energy=round(float(ke), 4),
        ))
    return entries


def main():
    print("=" * 72)
    print("  动态图 Pipeline 快速验证")
    print("  GraphBuilder → NarrativeGenerator → Decision Prompt")
    print("=" * 72)

    # ── Step 1: GraphBuilder 构建动态图 ──
    print("\n[Step 1] 构建 EntityGraph ...")
    builder = GraphBuilder(GraphConfig())
    trace_entries = make_trace_entries(ENTITY_ID)

    for sr in MOCK_SEMANTIC_RESULTS:
        relevant_traces = [
            te for te in trace_entries
            if te.entity_id == sr["entity_id"]
            and te.timestamp <= sr["timestamp"]
        ]
        node = builder.add_semantic_node(sr, relevant_traces)
        print(f"  + Node: {node.node_id} | T={node.timestamp:.1f}s | "
              f"action={node.action} | danger={node.danger_score}")

    graph: EntityGraph = builder.get_graph(ENTITY_ID)
    print(f"\n  Graph 统计: {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"duration={graph.total_duration:.1f}s")
    print(f"  max_danger={graph.max_danger_score:.2f}, "
          f"has_suspicious={graph.has_suspicious}")
    print(f"  total_kinetic_integral={graph.total_kinetic_integral:.4f}")

    print(f"\n  Action sequence: {' → '.join(graph.get_action_sequence())}")

    print("\n  Edges:")
    for e in graph.edges:
        print(f"    {e.action_transition} | "
              f"duration={e.duration_sec:.1f}s | "
              f"kinetic_integral={e.kinetic_integral:.4f} | "
              f"missing={e.missing_frames}")

    # ── Step 2: NarrativeGenerator 生成叙事文本 ──
    print("\n" + "=" * 72)
    print("[Step 2] NarrativeGenerator 生成叙事文本 ...")
    narrator = NarrativeGenerator(NarrativeConfig())

    entity_traces = [te for te in trace_entries if te.entity_id == ENTITY_ID]
    narrative = narrator.generate(
        graph,
        discordance_alerts=None,
        drift_info=None,
        trace_entries=entity_traces,
    )
    print("\n--- Narrative Text (输入到 Decision LLM 的实体行为叙事) ---")
    print(narrative)
    print("--- End Narrative ---")

    # ── Step 3: 构建完整的 Decision Prompt ──
    print("\n" + "=" * 72)
    print("[Step 3] 构建 Decision LLM 的完整输入 ...")

    scene_type = "parking lot"
    auditor = DecisionAuditor(decision_cfg=DecisionConfig())
    contracts_text = auditor._get_contracts(scene_type)

    contracts_section = ""
    if contracts_text != "  (No specific contracts)":
        contracts_section = (
            f"\n## Scene Rules (prior knowledge):\n{contracts_text}\n"
        )

    contracts_reminder = ""
    if contracts_section:
        contracts_reminder = (
            "\nEvaluate whether the behavior described above matches "
            "any of the Scene Rules. Only flag as anomalous if the "
            "evidence is clear and specific — do NOT flag routine "
            "shopping or work activities.\n"
        )

    user_prompt = DECISION_USER_TEMPLATE.format(
        scene_type=scene_type,
        narrative=narrative,
        contracts_section=contracts_section,
        contracts_reminder=contracts_reminder,
    )

    print("\n" + "=" * 72)
    print("  [SYSTEM PROMPT]  (Decision LLM 系统提示)")
    print("=" * 72)
    print(DECISION_SYSTEM_PROMPT)

    print("\n" + "=" * 72)
    print("  [USER PROMPT]  (Decision LLM 用户输入 — 即最终审计输入)")
    print("=" * 72)
    print(user_prompt)

    # ── Step 4: 序列化验证 ──
    print("\n" + "=" * 72)
    print("[Step 4] Graph 序列化 (export) ...")
    exported = builder.export_all()
    import json
    print(json.dumps(exported, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print("  ALL PASSED — 动态图 pipeline 工作正常")
    print("=" * 72)


if __name__ == "__main__":
    main()
