"""
Stage 3-A: 图数据结构 — TemporalNode + EvolutionEdge

TemporalNode:
  - 一个语义节点，对应某个实体在某时刻被 VLLM 描述的状态
  - 存 timestamp, semantic_tags (action, posture, etc.)

EvolutionEdge:
  - 连接同一实体的两个相邻 TemporalNode
  - 存 duration (秒), kinetic_integral (动能累积)
  - 编码"这段时间发生了什么物理变化"

EntityGraph:
  - 单实体的完整时间演化图 (链表结构)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TemporalNode:
    """时间语义节点"""

    node_id: str                         # 唯一标识: "E{entity_id}_N{seq}"
    entity_id: int
    frame_idx: int
    timestamp: float                     # 秒

    # 语义标签 (由 VLLM 填充)
    action: str = "unknown"
    action_object: str = "none"
    posture: str = "unknown"
    scene_context: str = "unknown"
    is_suspicious: bool = False
    danger_score: float = 0.0
    anomaly_category_guess: str = "none"

    # 触发规则
    trigger_rule: str = ""               # "birth" / "change_point" / "heartbeat"

    # CLIP embedding (用于 change-point 分析)
    embedding: Optional[list] = None     # 序列化时转 list

    # Bbox
    bbox: tuple = (0, 0, 0, 0)

    # 动能 (该帧的瞬时动能)
    kinetic_energy: float = 0.0


@dataclass
class EvolutionEdge:
    """时间演化边"""

    edge_id: str                          # 唯一标识: "E{eid}_{n1}_{n2}"
    entity_id: int
    source_node_id: str
    target_node_id: str

    # 时间
    duration_sec: float = 0.0            # 两个节点的时间间隔

    # 物理动能累积 (该时段内所有帧的动能之和)
    kinetic_integral: float = 0.0

    # 行为转移描述
    action_transition: str = ""          # "walking → running"

    # 缺失节点数 (该时段内没有被触发的帧数)
    missing_frames: int = 0


@dataclass
class EntityGraph:
    """单实体时间演化图"""

    entity_id: int
    nodes: list[TemporalNode] = field(default_factory=list)
    edges: list[EvolutionEdge] = field(default_factory=list)

    # 统计
    birth_time: float = 0.0
    last_time: float = 0.0
    total_duration: float = 0.0
    total_kinetic_integral: float = 0.0
    max_danger_score: float = 0.0
    has_suspicious: bool = False

    def add_node(self, node: TemporalNode):
        """添加节点并更新统计"""
        self.nodes.append(node)

        if len(self.nodes) == 1:
            self.birth_time = node.timestamp
        self.last_time = node.timestamp
        self.total_duration = self.last_time - self.birth_time

        if node.danger_score > self.max_danger_score:
            self.max_danger_score = node.danger_score
        if node.is_suspicious:
            self.has_suspicious = True

    def add_edge(self, edge: EvolutionEdge):
        """添加边并更新动能统计"""
        self.edges.append(edge)
        self.total_kinetic_integral += edge.kinetic_integral

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_action_sequence(self) -> list[str]:
        """获取动作序列"""
        return [n.action for n in self.nodes]

    def get_danger_timeline(self) -> list[tuple[float, float]]:
        """获取 (timestamp, danger_score) 时间线"""
        return [(n.timestamp, n.danger_score) for n in self.nodes]

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "entity_id": self.entity_id,
            "birth_time": round(self.birth_time, 3),
            "last_time": round(self.last_time, 3),
            "total_duration": round(self.total_duration, 3),
            "total_kinetic_integral": round(self.total_kinetic_integral, 4),
            "max_danger_score": round(self.max_danger_score, 3),
            "has_suspicious": self.has_suspicious,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "timestamp": round(n.timestamp, 3),
                    "action": n.action,
                    "action_object": n.action_object,
                    "posture": n.posture,
                    "scene_context": n.scene_context,
                    "is_suspicious": n.is_suspicious,
                    "danger_score": round(n.danger_score, 3),
                    "trigger_rule": n.trigger_rule,
                    "kinetic_energy": round(n.kinetic_energy, 4),
                    "bbox": list(n.bbox),
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source": e.source_node_id,
                    "target": e.target_node_id,
                    "duration_sec": round(e.duration_sec, 3),
                    "kinetic_integral": round(e.kinetic_integral, 4),
                    "action_transition": e.action_transition,
                    "missing_frames": e.missing_frames,
                }
                for e in self.edges
            ],
        }
