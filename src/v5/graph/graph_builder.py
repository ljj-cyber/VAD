"""
Stage 3-B: GraphBuilder — 动态图构建

功能:
  1. 在 Trigger 触发时创建 TemporalNode
  2. 在两个相邻 Node 之间创建 EvolutionEdge
  3. 自动计算 Edge 的时长和物理动能累积
  4. 维护多个实体的 EntityGraph

输入: SemanticVLLM 的推理结果 + EntityTracker 的 trace_log (动能)
输出: dict[entity_id → EntityGraph]
"""

import logging
from typing import Optional

import numpy as np

from ..config import GraphConfig
from ..tracking.entity_tracker import TraceEntry
from .structures import TemporalNode, EvolutionEdge, EntityGraph

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    动态图构建器。

    每次 Trigger 触发时调用 add_semantic_node()，传入 VLLM 推理结果。
    GraphBuilder 自动创建 Node → Edge 链接。
    """

    def __init__(self, cfg: Optional[GraphConfig] = None):
        self.cfg = cfg or GraphConfig()
        self._graphs: dict[int, EntityGraph] = {}
        self._node_counters: dict[int, int] = {}  # entity_id → seq

    def reset(self):
        """重置"""
        self._graphs.clear()
        self._node_counters.clear()

    def add_semantic_node(
        self,
        semantic_result: dict,
        trace_entries: list[TraceEntry],
    ) -> TemporalNode:
        """
        添加一个语义节点。

        Args:
            semantic_result: SemanticVLLM 的推理结果 dict
            trace_entries: 该实体在上一个 Node 到当前 Node 之间的所有 TraceEntry

        Returns:
            新创建的 TemporalNode
        """
        eid = semantic_result["entity_id"]
        frame_idx = semantic_result["frame_idx"]
        timestamp = semantic_result["timestamp"]

        # 确保 EntityGraph 存在
        if eid not in self._graphs:
            self._graphs[eid] = EntityGraph(entity_id=eid)
            self._node_counters[eid] = 0

        graph = self._graphs[eid]
        seq = self._node_counters[eid]
        self._node_counters[eid] = seq + 1

        # 创建 Node
        node = TemporalNode(
            node_id=f"E{eid}_N{seq}",
            entity_id=eid,
            frame_idx=frame_idx,
            timestamp=timestamp,
            action=semantic_result.get("action", "unknown"),
            action_object=semantic_result.get("action_object", "none"),
            posture=semantic_result.get("posture", "unknown"),
            scene_context=semantic_result.get("scene_context", "unknown"),
            is_suspicious=semantic_result.get("is_suspicious", False),
            danger_score=semantic_result.get("danger_score", 0.0),
            anomaly_category_guess=semantic_result.get("anomaly_category_guess", "none"),
            trigger_rule=semantic_result.get("trigger_rule", ""),
            kinetic_energy=self._get_kinetic_from_trace(trace_entries, frame_idx),
        )

        # 如果有对应的 TraceEntry，保留 bbox
        for te in reversed(trace_entries):
            if te.frame_idx == frame_idx and te.entity_id == eid:
                node.bbox = te.bbox
                break

        graph.add_node(node)

        # 如果不是第一个节点，创建 Edge
        if graph.num_nodes >= 2:
            prev_node = graph.nodes[-2]
            edge = self._build_edge(prev_node, node, trace_entries, eid)
            graph.add_edge(edge)

        # 修剪过长的图
        if graph.num_nodes > self.cfg.max_nodes_per_entity:
            self._prune_oldest(graph)

        return node

    def _build_edge(
        self,
        source: TemporalNode,
        target: TemporalNode,
        trace_entries: list[TraceEntry],
        entity_id: int,
    ) -> EvolutionEdge:
        """构建两个节点之间的边"""
        duration = target.timestamp - source.timestamp

        # 计算该时段的动能累积
        kinetic_integral = 0.0
        missing_frames = 0
        relevant = [
            te for te in trace_entries
            if te.entity_id == entity_id
            and source.timestamp <= te.timestamp <= target.timestamp
        ]

        if relevant:
            kinetic_integral = sum(te.kinetic_energy for te in relevant)
            # missing_frames: 中间帧数 - 实际出现帧数
            if len(relevant) >= 2:
                frame_span = relevant[-1].frame_idx - relevant[0].frame_idx
                missing_frames = max(0, frame_span - len(relevant))

        action_transition = f"{source.action} → {target.action}"

        return EvolutionEdge(
            edge_id=f"E{entity_id}_{source.node_id}_{target.node_id}",
            entity_id=entity_id,
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            duration_sec=round(duration, 3),
            kinetic_integral=round(kinetic_integral, 4),
            action_transition=action_transition,
            missing_frames=missing_frames,
        )

    @staticmethod
    def _get_kinetic_from_trace(
        trace_entries: list[TraceEntry],
        frame_idx: int,
    ) -> float:
        """从 trace 中获取指定帧的瞬时动能"""
        for te in reversed(trace_entries):
            if te.frame_idx == frame_idx:
                return te.kinetic_energy
        return 0.0

    def _prune_oldest(self, graph: EntityGraph):
        """移除最早的节点"""
        if graph.num_nodes <= 2:
            return
        graph.nodes.pop(0)
        if graph.edges:
            graph.edges.pop(0)

    @property
    def graphs(self) -> dict[int, EntityGraph]:
        return self._graphs

    def get_graph(self, entity_id: int) -> Optional[EntityGraph]:
        return self._graphs.get(entity_id)

    def get_all_entity_ids(self) -> list[int]:
        return list(self._graphs.keys())

    def get_auditable_entities(self) -> list[int]:
        """返回达到最低审计节点数的实体 ID"""
        return [
            eid for eid, g in self._graphs.items()
            if g.num_nodes >= self.cfg.min_nodes_for_audit
        ]

    def export_all(self) -> dict:
        """导出所有 EntityGraph 为字典"""
        return {eid: g.to_dict() for eid, g in self._graphs.items()}
