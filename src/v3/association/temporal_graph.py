"""
V4.0 时间关联层 — 时间演化图 (Temporal DiGraph)

V3→V4 变更:
  - 节点新增 visual_ref 字段：存储关键帧路径，供 Decision LLM 溯源
  - 节点新增 is_cinematic / visual_danger_score 字段
  - 边新增 energy_integral (ΣM)：两个采样点之间的积分运动能量
  - 边新增 energy_avg: 平均运动能量
  - 新增 get_entity_narrative_data() 用于叙事引擎

核心思想:
  - 每个节点代表一个实体在某一时刻的状态（快照）。
  - 有向边表示同一实体在时间轴上的状态转移:
    T_n → T_{n+1}
  - 边权由两部分加权:
    w = α * portrait_similarity + β * action_transition_score
    其中 α + β = 1

节点属性:
  - entity_id: 全局实体 ID
  - frame_id: 帧号
  - timestamp: 时间戳
  - portrait: 画像描述
  - action: 原子动作
  - action_object: 动作对象
  - location: 功能区
  - posture: 体态
  - visual_ref: 关键帧路径 (V4.0 新增)
  - is_cinematic: 电影感标记 (V4.0 新增)
  - visual_danger_score: 视觉危险分数 (V4.0 新增)

边属性:
  - weight: 综合权重
  - portrait_sim: 外观相似度
  - action_score: 动作转移合理性分数
  - time_gap: 时间间隔 (秒)
  - energy_integral: 积分运动能量 ΣM (V4.0 新增)
  - energy_avg: 平均运动能量 (V4.0 新增)
"""

import logging
from typing import Optional

import numpy as np
import networkx as nx

from ..config import AssociationConfig
from .entity_pool import EntityPool

logger = logging.getLogger(__name__)

# ── 动作转移合理性矩阵 ──────────────────────────────
# 定义常见动作对之间的转移合理性分数 [0, 1]
# 高分 = 自然转移, 低分 = 异常跳变
ACTION_TRANSITION_SCORES = {
    # 正常购物流程
    ("browsing", "picking up"): 0.9,
    ("picking up", "holding"): 0.95,
    ("holding", "walking"): 0.9,
    ("walking", "handing over"): 0.8,
    ("handing over", "paying"): 0.9,
    ("paying", "leaving"): 0.9,
    ("holding", "putting down"): 0.85,

    # 正常移动
    ("standing still", "walking"): 0.95,
    ("walking", "standing still"): 0.95,
    ("walking", "running"): 0.6,
    ("standing still", "sitting"): 0.85,
    ("sitting", "standing still"): 0.85,

    # 可疑转移
    ("holding", "running"): 0.2,
    ("picking up", "running"): 0.15,
    ("standing still", "running"): 0.3,
    ("walking", "fighting"): 0.1,
    ("standing still", "fighting"): 0.1,
    ("holding", "threatening"): 0.1,
    ("holding", "attacking"): 0.05,

    # 暴力相关
    ("walking", "attacking"): 0.05,
    ("running", "attacking"): 0.1,
    ("threatening", "attacking"): 0.3,
    ("fighting", "running"): 0.3,
    ("fighting", "falling"): 0.4,

    # 对称转移（双向）
    ("walking", "walking"): 0.95,
    ("running", "running"): 0.85,
    ("standing still", "standing still"): 0.98,
    ("sitting", "sitting"): 0.98,
    ("holding", "holding"): 0.9,
}

# 默认转移分数（未列出的动作对）
DEFAULT_TRANSITION_SCORE = 0.5


def get_action_transition_score(action_from: str, action_to: str) -> float:
    """
    查询动作转移合理性分数。

    Args:
        action_from: 起始动作
        action_to: 目标动作

    Returns:
        [0, 1] 之间的合理性分数
    """
    # 标准化动作名
    a1 = action_from.lower().strip()
    a2 = action_to.lower().strip()

    # 精确匹配
    key = (a1, a2)
    if key in ACTION_TRANSITION_SCORES:
        return ACTION_TRANSITION_SCORES[key]

    # 反向匹配
    key_rev = (a2, a1)
    if key_rev in ACTION_TRANSITION_SCORES:
        return ACTION_TRANSITION_SCORES[key_rev]

    # 模糊匹配：检查是否包含关键词
    for (k1, k2), score in ACTION_TRANSITION_SCORES.items():
        if (k1 in a1 or a1 in k1) and (k2 in a2 or a2 in k2):
            return score
        if (k2 in a1 or a1 in k2) and (k1 in a2 or a2 in k1):
            return score

    # 同一动作 → 高分
    if a1 == a2:
        return 0.9

    return DEFAULT_TRANSITION_SCORE


class TemporalGraph:
    """
    V4.0 时间演化有向图。

    节点: (entity_id, frame_id) — 表示实体在特定时刻的状态
    有向边: (entity_id, frame_i) → (entity_id, frame_j) — 状态转移

    V4.0 增强:
      - 节点存储 visual_ref / is_cinematic / visual_danger_score
      - 边存储 energy_integral / energy_avg（积分运动能量）
    """

    def __init__(self, cfg: Optional[AssociationConfig] = None):
        self.cfg = cfg or AssociationConfig()
        self.G = nx.DiGraph()
        self._entity_timelines: dict[int, list[str]] = {}  # entity_id -> [node_id, ...]
        # V4.0: 帧级运动能量缓存 {frame_id: energy}
        self._frame_energies: dict[int, float] = {}

    # ── 运动能量注册 ──────────────────────────────────
    def register_frame_energy(self, frame_id: int, energy: float):
        """注册帧级运动能量 (供建边时计算积分能量)"""
        self._frame_energies[frame_id] = energy

    def register_frame_energies(self, energies: dict[int, float]):
        """批量注册帧级运动能量"""
        self._frame_energies.update(energies)

    def get_frame_energies(self) -> dict[int, float]:
        """获取所有帧级运动能量"""
        return dict(self._frame_energies)

    # ── 节点 ID 生成 ──────────────────────────────────
    @staticmethod
    def _node_id(entity_id: int, frame_id: int) -> str:
        """生成节点唯一标识"""
        return f"e{entity_id}_f{frame_id}"

    # ── 添加节点 (V4.0 增强) ────────────────────────────
    def add_node(
        self,
        entity_id: int,
        frame_id: int,
        timestamp: float,
        portrait: str,
        action: str,
        action_object: str = "",
        location: str = "",
        posture: str = "",
        portrait_embedding: Optional[np.ndarray] = None,
        visual_ref: str = "",
        is_cinematic: bool = False,
        visual_danger_score: float = 0.0,
    ):
        """
        添加一个实体-时刻状态节点。

        如果该实体已有历史节点，自动建立时间边。

        V4.0 新增参数:
          visual_ref: 关键帧缩略图路径
          is_cinematic: 电影感标记
          visual_danger_score: 视觉危险分数
        """
        node_id = self._node_id(entity_id, frame_id)

        self.G.add_node(
            node_id,
            entity_id=entity_id,
            frame_id=frame_id,
            timestamp=timestamp,
            portrait=portrait,
            action=action,
            action_object=action_object,
            location=location,
            posture=posture,
            visual_ref=visual_ref,
            is_cinematic=is_cinematic,
            visual_danger_score=visual_danger_score,
        )

        # 维护时间线
        if entity_id not in self._entity_timelines:
            self._entity_timelines[entity_id] = []

        timeline = self._entity_timelines[entity_id]

        # 如果有前驱节点，建立时间边
        if timeline:
            prev_node_id = timeline[-1]
            prev_data = self.G.nodes[prev_node_id]

            time_gap = timestamp - prev_data["timestamp"]

            # 只在时间间隔合理时建边
            if 0 < time_gap <= self.cfg.max_time_gap:
                self._add_temporal_edge(
                    prev_node_id,
                    node_id,
                    time_gap,
                    portrait_embedding,
                )

        timeline.append(node_id)

    # ── 添加时间边 (V4.0 增强: 积分运动能量) ──────────
    def _add_temporal_edge(
        self,
        src_node_id: str,
        dst_node_id: str,
        time_gap: float,
        portrait_embedding: Optional[np.ndarray] = None,
    ):
        """
        在两个节点之间建立带权有向时间边。

        边权 = α * portrait_sim + β * action_transition_score

        V4.0: 自动计算并存储两个采样点之间的积分运动能量 (ΣM)。
        """
        src = self.G.nodes[src_node_id]
        dst = self.G.nodes[dst_node_id]

        # 1. 动作转移合理性分数
        action_score = get_action_transition_score(
            src["action"], dst["action"]
        )

        # 2. 外观相似度
        portrait_sim = 0.8  # 默认值（同一 entity_id 的外观相似度应该较高）

        # 3. 综合权重
        weight = (
            self.cfg.portrait_weight * portrait_sim
            + self.cfg.action_weight * action_score
        )

        # 4. V4.0: 计算积分运动能量 ΣM
        src_frame = src["frame_id"]
        dst_frame = dst["frame_id"]
        energy_integral, energy_avg = self._compute_energy_between(
            src_frame, dst_frame
        )

        self.G.add_edge(
            src_node_id,
            dst_node_id,
            weight=weight,
            portrait_sim=portrait_sim,
            action_score=action_score,
            time_gap=time_gap,
            action_from=src["action"],
            action_to=dst["action"],
            # V4.0 新增
            energy_integral=energy_integral,
            energy_avg=energy_avg,
        )

        logger.debug(
            f"Edge {src_node_id} -> {dst_node_id}: "
            f"w={weight:.3f} (portrait={portrait_sim:.3f}, "
            f"action={action_score:.3f}, Δt={time_gap:.1f}s, "
            f"ΣM={energy_integral:.3f}, AvgM={energy_avg:.3f})"
        )

    def _compute_energy_between(
        self, src_frame: int, dst_frame: int
    ) -> tuple[float, float]:
        """
        计算两个采样帧之间的积分运动能量和平均运动能量。

        Returns:
            (energy_integral, energy_avg)
        """
        if not self._frame_energies:
            return 0.0, 0.0

        # 收集 [src_frame, dst_frame] 范围内的所有能量值
        energies = []
        for fid in sorted(self._frame_energies.keys()):
            if src_frame <= fid <= dst_frame:
                energies.append(self._frame_energies[fid])

        if not energies:
            # 回退: 使用两端点的能量
            e_src = self._frame_energies.get(src_frame, 0.0)
            e_dst = self._frame_energies.get(dst_frame, 0.0)
            return (e_src + e_dst) / 2.0, (e_src + e_dst) / 2.0

        energy_integral = float(np.sum(energies))
        energy_avg = float(np.mean(energies))
        return energy_integral, energy_avg

    # ── 批量添加帧快照 (V4.0 增强) ──────────────────
    def add_frame_snapshot(
        self,
        matched_entities: list[tuple[dict, int, bool]],
        frame_id: int,
        timestamp: float,
        visual_ref: str = "",
        is_cinematic: bool = False,
        visual_danger_score: float = 0.0,
    ):
        """
        将一帧的所有实体匹配结果添加到图中。

        Args:
            matched_entities: EntityPool.match_entities 的返回值
                [(entity_dict, entity_id, is_new), ...]
            frame_id: 帧号
            timestamp: 时间戳
            visual_ref: 该帧缩略图路径 (V4.0)
            is_cinematic: 电影感标记 (V4.0)
            visual_danger_score: 视觉危险分数 (V4.0)
        """
        for entity_dict, entity_id, is_new in matched_entities:
            self.add_node(
                entity_id=entity_id,
                frame_id=frame_id,
                timestamp=timestamp,
                portrait=entity_dict.get("portrait", "unknown"),
                action=entity_dict.get("action", "unknown"),
                action_object=entity_dict.get("action_object", ""),
                location=entity_dict.get("location", ""),
                posture=entity_dict.get("posture", ""),
                visual_ref=visual_ref,
                is_cinematic=is_cinematic,
                visual_danger_score=visual_danger_score,
            )

    # ── 路径查询 ──────────────────────────────────────
    def get_entity_path(self, entity_id: int) -> list[dict]:
        """
        获取指定实体的完整时间路径。

        Returns:
            按时间排序的节点属性列表
        """
        timeline = self._entity_timelines.get(entity_id, [])
        path = []
        for node_id in timeline:
            if node_id in self.G.nodes:
                data = dict(self.G.nodes[node_id])
                data["node_id"] = node_id
                path.append(data)
        return path

    def get_entity_edges(self, entity_id: int) -> list[dict]:
        """
        获取指定实体所有时间边的信息。
        """
        timeline = self._entity_timelines.get(entity_id, [])
        edges = []
        for i in range(len(timeline) - 1):
            src = timeline[i]
            dst = timeline[i + 1]
            if self.G.has_edge(src, dst):
                edge_data = dict(self.G.edges[src, dst])
                edge_data["src"] = src
                edge_data["dst"] = dst
                edges.append(edge_data)
        return edges

    def get_all_entity_ids(self) -> list[int]:
        """获取图中所有实体 ID"""
        return list(self._entity_timelines.keys())

    def get_entity_action_sequence(self, entity_id: int) -> list[str]:
        """获取实体的动作序列"""
        path = self.get_entity_path(entity_id)
        return [node["action"] for node in path]

    def get_edge_weights_for_entity(self, entity_id: int) -> list[float]:
        """获取实体所有边的权重"""
        edges = self.get_entity_edges(entity_id)
        return [e["weight"] for e in edges]

    # ── V4.0: 叙事引擎数据接口 ───────────────────────
    def get_entity_narrative_data(self, entity_id: int) -> dict:
        """
        获取用于叙事引擎的完整实体数据。

        Returns:
            {
                "entity_id": int,
                "path": list[dict],       # 节点序列
                "edges": list[dict],       # 边序列
                "action_sequence": list[str],
                "timestamps": list[float],
                "total_energy_integral": float,  # 总积分动能
                "has_cinematic_frames": bool,
                "max_danger_score": float,
            }
        """
        path = self.get_entity_path(entity_id)
        edges = self.get_entity_edges(entity_id)
        actions = [n["action"] for n in path]
        timestamps = [n["timestamp"] for n in path]

        total_energy = sum(e.get("energy_integral", 0.0) for e in edges)
        has_cinematic = any(n.get("is_cinematic", False) for n in path)
        max_danger = max(
            (n.get("visual_danger_score", 0.0) for n in path), default=0.0
        )

        return {
            "entity_id": entity_id,
            "path": path,
            "edges": edges,
            "action_sequence": actions,
            "timestamps": timestamps,
            "total_energy_integral": total_energy,
            "has_cinematic_frames": has_cinematic,
            "max_danger_score": max_danger,
        }

    # ── V4.0: 高危信号检测 ───────────────────────────
    def has_danger_signal(self, entity_id: int, threshold: float = 0.6) -> bool:
        """检测实体路径是否触发高危信号"""
        path = self.get_entity_path(entity_id)
        for node in path:
            if node.get("visual_danger_score", 0.0) >= threshold:
                return True
        return False

    def get_path_length_changes(self) -> dict[int, int]:
        """获取所有实体的当前路径长度"""
        return {
            eid: len(timeline)
            for eid, timeline in self._entity_timelines.items()
        }

    # ── 图统计 ────────────────────────────────────────
    def get_stats(self) -> dict:
        """获取图的统计信息"""
        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        n_entities = len(self._entity_timelines)

        path_lengths = [
            len(timeline) for timeline in self._entity_timelines.values()
        ]

        edge_weights = [
            d["weight"] for _, _, d in self.G.edges(data=True)
        ] if n_edges > 0 else []

        # V4.0: 能量统计
        energy_integrals = [
            d.get("energy_integral", 0.0)
            for _, _, d in self.G.edges(data=True)
        ] if n_edges > 0 else []

        return {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "num_entities": n_entities,
            "avg_path_length": float(np.mean(path_lengths)) if path_lengths else 0,
            "max_path_length": max(path_lengths) if path_lengths else 0,
            "avg_edge_weight": float(np.mean(edge_weights)) if edge_weights else 0,
            "min_edge_weight": float(min(edge_weights)) if edge_weights else 0,
            "total_energy_integral": float(np.sum(energy_integrals)) if energy_integrals else 0,
            "avg_energy_per_edge": float(np.mean(energy_integrals)) if energy_integrals else 0,
        }

    def reset(self):
        """重置图"""
        self.G.clear()
        self._entity_timelines.clear()
        self._frame_energies.clear()