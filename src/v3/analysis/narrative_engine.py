"""
V4.0 叙事引擎 — 图拓扑路径→因果叙事文本

职能:
  将时间演化图中的实体路径 (Node A → Node B → ...)
  转化为人类可读的因果描述字符串，供 Decision LLM 审计。

示例输出:
  "实体 #001 在 T=5.0s 表现为'观察'(位于入口区域)，
   在 T=7.0s 突变为'奔跑'(位于过道)，期间积分动能=0.85 (激增)，
   外观保持一致(portrait_sim=0.82)。
   此后在 T=9.0s 表现为'拿取'(位于货架区)，动作转移合理性仅 0.15。"

设计原则:
  1. 支持中文 (zh) 和英文 (en) 两种语言。
  2. 突出异常信号：动作突变、动能激增、低动作合理性边。
  3. 对 Decision LLM 友好的文本格式：简洁、结构化、包含量化信息。
"""

import logging
from typing import Optional

from ..config import NarrativeConfig
from ..association.temporal_graph import TemporalGraph

logger = logging.getLogger(__name__)

# ── 动能等级映射 ──────────────────────────────────────
_ENERGY_LEVELS_ZH = {
    (0.0, 0.05): "极低",
    (0.05, 0.15): "低",
    (0.15, 0.35): "中等",
    (0.35, 0.60): "较高",
    (0.60, 0.85): "高",
    (0.85, 1.01): "极高(激增)",
}
_ENERGY_LEVELS_EN = {
    (0.0, 0.05): "very low",
    (0.05, 0.15): "low",
    (0.15, 0.35): "moderate",
    (0.35, 0.60): "elevated",
    (0.60, 0.85): "high",
    (0.85, 1.01): "very high (surge)",
}


def _energy_label(value: float, lang: str = "zh") -> str:
    """将能量值映射为人类可读等级"""
    levels = _ENERGY_LEVELS_ZH if lang == "zh" else _ENERGY_LEVELS_EN
    for (lo, hi), label in levels.items():
        if lo <= value < hi:
            return label
    return "极高" if lang == "zh" else "extreme"


def _format_time(t: float) -> str:
    """将时间戳格式化为 mm:ss.f"""
    mins = int(t // 60)
    secs = t % 60
    if mins > 0:
        return f"{mins:02d}:{secs:05.2f}"
    return f"{secs:.2f}s"


class NarrativeEngine:
    """
    V4.0 叙事引擎。

    将时间图中实体的拓扑路径转换为 Decision LLM 可理解的
    因果叙事文本。
    """

    def __init__(self, cfg: Optional[NarrativeConfig] = None):
        self.cfg = cfg or NarrativeConfig()

    def path_to_text(
        self,
        entity_id: int,
        graph: TemporalGraph,
        scene_type: str = "",
    ) -> str:
        """
        将指定实体的时间路径转换为因果叙事文本。

        Args:
            entity_id: 实体 ID
            graph: 时间演化图
            scene_type: 场景类型

        Returns:
            人类可读的因果叙事字符串
        """
        data = graph.get_entity_narrative_data(entity_id)
        path = data["path"]
        edges = data["edges"]

        if not path:
            return self._empty_narrative(entity_id)

        if self.cfg.language == "zh":
            text = self._build_zh_narrative(entity_id, path, edges, scene_type, data)
        else:
            text = self._build_en_narrative(entity_id, path, edges, scene_type, data)

        # 截断
        if len(text) > self.cfg.max_narrative_length:
            text = text[: self.cfg.max_narrative_length - 3] + "..."

        return text

    def batch_to_text(
        self,
        graph: TemporalGraph,
        scene_type: str = "",
    ) -> dict[int, str]:
        """
        批量生成所有实体的叙事文本。

        Returns:
            {entity_id: narrative_text}
        """
        narratives = {}
        for eid in graph.get_all_entity_ids():
            narratives[eid] = self.path_to_text(eid, graph, scene_type)
        return narratives

    # ── 中文叙事构建 ──────────────────────────────────
    def _build_zh_narrative(
        self,
        entity_id: int,
        path: list[dict],
        edges: list[dict],
        scene_type: str,
        data: dict,
    ) -> str:
        parts = []

        # 头部摘要
        header = f"【实体 #{entity_id:03d} 行为轨迹】"
        if scene_type:
            header += f"（场景: {scene_type}）"
        header += f"  路径长度: {len(path)} 个采样点"
        if data.get("has_cinematic_frames"):
            header += "  ⚠️ 包含电影感帧"
        if data.get("max_danger_score", 0) >= 0.5:
            header += f"  ⚠️ 最高危险分={data['max_danger_score']:.2f}"
        parts.append(header)
        parts.append("")

        # 逐节点描述
        for i, node in enumerate(path):
            t = node["timestamp"]
            action = node["action"]
            location = node.get("location", "")
            posture = node.get("posture", "")
            danger = node.get("visual_danger_score", 0.0)
            cinematic = node.get("is_cinematic", False)

            line = f"  T={_format_time(t)}  动作='{action}'"
            if location:
                line += f"  位置='{location}'"
            if posture:
                line += f"  姿态='{posture}'"
            if danger >= 0.3:
                line += f"  危险度={danger:.2f}"
            if cinematic:
                line += "  [电影特征]"

            parts.append(line)

            # 如果有对应的边，描述转移
            if i < len(edges):
                edge = edges[i]
                action_score = edge.get("action_score", 0.5)
                time_gap = edge.get("time_gap", 0.0)
                energy_int = edge.get("energy_integral", 0.0)
                energy_avg = edge.get("energy_avg", 0.0)
                next_action = edge.get("action_to", "?")

                # 判断是否为异常转移
                is_suspicious = action_score < 0.3

                if is_suspicious:
                    prefix = "  ⚠️ → "
                else:
                    prefix = "     → "

                trans_line = (
                    f"{prefix}间隔 {time_gap:.1f}s 后转为'{next_action}'  "
                    f"合理性={action_score:.2f}"
                )

                if self.cfg.include_energy and energy_avg > 0:
                    energy_lbl = _energy_label(energy_avg, "zh")
                    trans_line += f"  动能={energy_lbl}(ΣM={energy_int:.3f})"

                if is_suspicious:
                    trans_line += "  [低合理性转移!]"

                parts.append(trans_line)

        # 总结
        parts.append("")
        total_energy = data.get("total_energy_integral", 0.0)
        parts.append(
            f"  总积分动能: {total_energy:.3f}  "
            f"动作序列: {' → '.join(data['action_sequence'][:15])}"
        )

        # 异常线索汇总
        anomaly_hints = self._extract_anomaly_hints_zh(path, edges)
        if anomaly_hints:
            parts.append("")
            parts.append("  异常线索:")
            for hint in anomaly_hints:
                parts.append(f"    - {hint}")

        return "\n".join(parts)

    # ── 英文叙事构建 ──────────────────────────────────
    def _build_en_narrative(
        self,
        entity_id: int,
        path: list[dict],
        edges: list[dict],
        scene_type: str,
        data: dict,
    ) -> str:
        parts = []

        header = f"[Entity #{entity_id:03d} Trajectory]"
        if scene_type:
            header += f" (Scene: {scene_type})"
        header += f"  Path length: {len(path)} sample points"
        if data.get("has_cinematic_frames"):
            header += "  ⚠️ Contains cinematic frames"
        if data.get("max_danger_score", 0) >= 0.5:
            header += f"  ⚠️ Max danger={data['max_danger_score']:.2f}"
        parts.append(header)
        parts.append("")

        for i, node in enumerate(path):
            t = node["timestamp"]
            action = node["action"]
            location = node.get("location", "")
            danger = node.get("visual_danger_score", 0.0)
            cinematic = node.get("is_cinematic", False)

            line = f"  T={_format_time(t)}  action='{action}'"
            if location:
                line += f"  at '{location}'"
            if danger >= 0.3:
                line += f"  danger={danger:.2f}"
            if cinematic:
                line += "  [CINEMATIC]"

            parts.append(line)

            if i < len(edges):
                edge = edges[i]
                action_score = edge.get("action_score", 0.5)
                time_gap = edge.get("time_gap", 0.0)
                energy_avg = edge.get("energy_avg", 0.0)
                energy_int = edge.get("energy_integral", 0.0)
                next_action = edge.get("action_to", "?")

                is_suspicious = action_score < 0.3
                prefix = "  ⚠️ → " if is_suspicious else "     → "

                trans_line = (
                    f"{prefix}after {time_gap:.1f}s → '{next_action}'  "
                    f"reasonableness={action_score:.2f}"
                )

                if self.cfg.include_energy and energy_avg > 0:
                    energy_lbl = _energy_label(energy_avg, "en")
                    trans_line += f"  energy={energy_lbl}(ΣM={energy_int:.3f})"

                if is_suspicious:
                    trans_line += "  [LOW REASONABLENESS!]"

                parts.append(trans_line)

        parts.append("")
        total_energy = data.get("total_energy_integral", 0.0)
        parts.append(
            f"  Total energy integral: {total_energy:.3f}  "
            f"Action seq: {' → '.join(data['action_sequence'][:15])}"
        )

        hints = self._extract_anomaly_hints_en(path, edges)
        if hints:
            parts.append("")
            parts.append("  Anomaly hints:")
            for h in hints:
                parts.append(f"    - {h}")

        return "\n".join(parts)

    # ── 异常线索提取 ──────────────────────────────────
    def _extract_anomaly_hints_zh(
        self, path: list[dict], edges: list[dict]
    ) -> list[str]:
        hints = []

        # 1. 低合理性转移
        for edge in edges:
            if edge.get("action_score", 1.0) < 0.3:
                hints.append(
                    f"在 T={_format_time(edges[0].get('time_gap', 0))} 附近，"
                    f"'{edge.get('action_from', '?')}' → '{edge.get('action_to', '?')}' "
                    f"的转移合理性仅为 {edge.get('action_score', 0):.2f}"
                )

        # 2. 高危险帧
        for node in path:
            if node.get("visual_danger_score", 0) >= 0.6:
                hints.append(
                    f"T={_format_time(node['timestamp'])} 视觉危险度达到 "
                    f"{node['visual_danger_score']:.2f}"
                )

        # 3. 动能激增边
        for edge in edges:
            if edge.get("energy_avg", 0) >= 0.6:
                hints.append(
                    f"两采样点间平均动能={edge['energy_avg']:.3f}，处于高水平"
                )

        # 4. 电影特征帧
        cinematic_count = sum(1 for n in path if n.get("is_cinematic", False))
        if cinematic_count > 0:
            hints.append(f"路径中有 {cinematic_count} 帧被标记为电影/剪辑特征")

        return hints[:5]  # 最多 5 条

    def _extract_anomaly_hints_en(
        self, path: list[dict], edges: list[dict]
    ) -> list[str]:
        hints = []

        for edge in edges:
            if edge.get("action_score", 1.0) < 0.3:
                hints.append(
                    f"Transition '{edge.get('action_from', '?')}' → "
                    f"'{edge.get('action_to', '?')}' has low reasonableness "
                    f"({edge.get('action_score', 0):.2f})"
                )

        for node in path:
            if node.get("visual_danger_score", 0) >= 0.6:
                hints.append(
                    f"T={_format_time(node['timestamp'])} visual danger "
                    f"score={node['visual_danger_score']:.2f}"
                )

        for edge in edges:
            if edge.get("energy_avg", 0) >= 0.6:
                hints.append(
                    f"High avg motion energy={edge['energy_avg']:.3f} between samples"
                )

        cinematic_count = sum(1 for n in path if n.get("is_cinematic", False))
        if cinematic_count > 0:
            hints.append(f"{cinematic_count} frames flagged as cinematic")

        return hints[:5]

    def _empty_narrative(self, entity_id: int) -> str:
        if self.cfg.language == "zh":
            return f"【实体 #{entity_id:03d}】无有效路径数据。"
        return f"[Entity #{entity_id:03d}] No valid path data."
