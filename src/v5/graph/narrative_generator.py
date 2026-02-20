"""
Stage 3-C: NarrativeGenerator — 叙事生成器

遍历 EntityGraph 路径，生成文本叙事:
  "ID_01 在 T=5.0s 蹲下，该状态持续了 40.0s (动能极低=0.0012)，
   随后在 T=45.0s 站起并奔跑 (动能激增=0.85)。"

关键逻辑:
  - 显式写入 duration 和 kinetic_integral
  - 标注 missing_nodes (缺失观测)
  - 标注 danger_score 升高的关键转折
  - 叙事文本将喂给 Decision Prompt 做最终审计
"""

import logging
from typing import Optional, Any

import numpy as np

from ..config import NarrativeConfig
from .structures import EntityGraph, TemporalNode, EvolutionEdge

logger = logging.getLogger(__name__)


def _kinetic_label(value: float) -> str:
    """动能等级标签 — 仅返回数值，不做解释"""
    return ""


def _danger_label(score: float) -> str:
    """危险度标签 — 仅返回数值，不做解释"""
    return ""


class NarrativeGenerator:
    """
    从 EntityGraph 生成结构化叙事文本。
    """

    def __init__(self, cfg: Optional[NarrativeConfig] = None):
        self.cfg = cfg or NarrativeConfig()

    def generate(
        self,
        graph: EntityGraph,
        discordance_alerts: Optional[list] = None,
        drift_info: Optional[dict] = None,
        trace_entries: Optional[list] = None,
    ) -> str:
        """
        生成单个实体的叙事文本，含物理异常预警和物理轨迹摘要。

        Args:
            graph: EntityGraph (已包含 nodes + edges)
            discordance_alerts: DiscordanceAlert 列表（物理-语义矛盾）
            drift_info: {"max_drift": float, "drift_timestamps": [...]}
            trace_entries: 该实体的完整 TraceEntry 列表（零VLLM开销的物理数据）

        Returns:
            叙事文本字符串
        """
        if graph.num_nodes == 0:
            return f"Entity #{graph.entity_id}: no observations."

        parts: list[str] = []
        nodes = graph.nodes
        edges = graph.edges

        # Header
        parts.append(
            f"Entity #{graph.entity_id} | "
            f"Duration: {graph.total_duration:.1f}s | "
            f"Observations: {graph.num_nodes} | "
            f"Scene: {nodes[0].scene_context}"
        )
        parts.append("")

        # ── 物理异常预警段落 ──
        eid = graph.entity_id
        has_physics_warning = False

        if discordance_alerts:
            my_alerts = [a for a in discordance_alerts if a.entity_id == eid or a.entity_id == -1]
            if my_alerts:
                has_physics_warning = True
                parts.append("[Physical Signal]")
                for alert in my_alerts[:3]:
                    parts.append(f"  - {alert.description}")
                    if alert.peak_energy_time > 0:
                        parts.append(
                            f"    Peak kinetic energy at T={alert.peak_energy_time:.2f}s "
                            f"(value={alert.peak_energy_value:.4f}), "
                            f"burst interval=[{alert.burst_start_sec:.2f}s, {alert.burst_end_sec:.2f}s]"
                        )
                parts.append("")

        if drift_info and drift_info.get("max_drift", 0) > 0.15:
            has_physics_warning = True
            parts.append(
                f"[Physical Signal — Scene Change] "
                f"drift={drift_info['max_drift']:.3f}"
            )
            parts.append("")

        # ── 物理轨迹摘要 (来自 trace_log，零 VLLM 开销) ──
        if trace_entries and len(trace_entries) > 1:
            energies = [e.kinetic_energy for e in trace_entries]
            track_duration = trace_entries[-1].timestamp - trace_entries[0].timestamp

            # bbox 位移
            try:
                bx0, by0 = float(trace_entries[0].bbox[0]), float(trace_entries[0].bbox[1])
                bx1, by1 = float(trace_entries[-1].bbox[0]), float(trace_entries[-1].bbox[1])
                displacement = ((bx1 - bx0) ** 2 + (by1 - by0) ** 2) ** 0.5
            except (TypeError, ValueError, IndexError):
                displacement = 0.0

            parts.append("[Physical Trajectory]")
            parts.append(
                f"  Tracked {len(trace_entries)} frames over {track_duration:.1f}s"
            )
            parts.append(
                f"  Kinetic energy: mean={np.mean(energies):.4f}, "
                f"max={np.max(energies):.4f}, "
                f"integral={sum(energies):.4f}"
            )
            if displacement > 20:
                parts.append(f"  Bbox displacement: {displacement:.0f}px")

            # 动能趋势 (上升/下降/平稳)
            if len(energies) >= 4:
                first_half = float(np.mean(energies[: len(energies) // 2]))
                second_half = float(np.mean(energies[len(energies) // 2 :]))
                ratio = second_half / max(first_half, 1e-6)
                if ratio > 1.3:
                    parts.append("  Trend: kinetic energy RISING")
                elif ratio < 0.7:
                    parts.append("  Trend: kinetic energy FALLING")
                else:
                    parts.append("  Trend: kinetic energy STABLE")
            parts.append("")

        # 逐节点 + 逐边描述
        for i, node in enumerate(nodes):
            # Node 描述
            line = f"T={node.timestamp:.1f}s: {node.action}"
            if node.action_object and node.action_object != "none":
                line += f" ({node.action_object})"
            line += f", posture={node.posture}"

            if self.cfg.include_kinetic_integral:
                line += f", kinetic={node.kinetic_energy:.4f} [{_kinetic_label(node.kinetic_energy)}]"

            line += _danger_label(node.danger_score)

            if node.is_suspicious:
                line += " [flagged]"

            line += f"  (trigger: {node.trigger_rule})"
            parts.append(line)

            # Edge 描述 (如果存在)
            if i < len(edges):
                edge = edges[i]
                edge_line = (
                    f"  ↓ {edge.action_transition} | "
                    f"duration={edge.duration_sec:.1f}s | "
                    f"kinetic_integral={edge.kinetic_integral:.4f} "
                    f"[{_kinetic_label(edge.kinetic_integral / max(edge.duration_sec, 0.01))}]"
                )

                if self.cfg.include_missing_nodes and edge.missing_frames > 0:
                    edge_line += f" | missing_frames={edge.missing_frames}"

                parts.append(edge_line)

        # Anomaly category hints (从 VLLM anomaly_category_guess 统计)
        from collections import Counter
        cat_counts = Counter(
            n.anomaly_category_guess for n in nodes
            if hasattr(n, 'anomaly_category_guess')
            and n.anomaly_category_guess not in ("none", "unknown", "")
        )
        if cat_counts:
            parts.append("")
            parts.append(f"Anomaly hints: {dict(cat_counts)}")

        # 全局统计
        parts.append("")
        parts.append(
            f"Summary: total_kinetic={graph.total_kinetic_integral:.4f}, "
            f"max_danger={graph.max_danger_score:.2f}, "
            f"suspicious={graph.has_suspicious}, "
            f"physics_warning={has_physics_warning}"
        )

        text = "\n".join(parts)

        # 长度截断
        if len(text) > self.cfg.max_narrative_length:
            text = text[: self.cfg.max_narrative_length] + "\n...[truncated]"

        return text

    def generate_all(self, graphs: dict[int, EntityGraph]) -> dict[int, str]:
        """批量生成叙事"""
        return {eid: self.generate(g) for eid, g in graphs.items()}

    def generate_video_overview(
        self,
        graphs: dict[int, EntityGraph],
        video_duration: float = 0.0,
    ) -> str:
        """
        生成视频级概览叙事。
        """
        if not graphs:
            return "No entities detected in this video."

        lines = [
            f"Video Overview | Duration: {video_duration:.1f}s | "
            f"Entities: {len(graphs)}",
            "",
        ]

        # 按 max_danger_score 降序
        sorted_eids = sorted(
            graphs.keys(),
            key=lambda eid: graphs[eid].max_danger_score,
            reverse=True,
        )

        for eid in sorted_eids:
            g = graphs[eid]
            actions = g.get_action_sequence()
            action_str = " → ".join(actions[:10])
            if len(actions) > 10:
                action_str += " → ..."

            lines.append(
                f"  Entity #{eid}: {g.total_duration:.1f}s, "
                f"{g.num_nodes} obs, "
                f"max_danger={g.max_danger_score:.2f}, "
                f"actions=[{action_str}]"
                + (" [flagged]" if g.has_suspicious else "")
            )

        return "\n".join(lines)
