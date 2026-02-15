"""
V4.0 异常检测器 — Decision LLM 审计模式 + V3 加权公式 Fallback

V3→V4 核心变更:
  - 主判定路径: Decision LLM 因果审计 (CausalityAuditor)
  - Fallback: V3 多信号加权公式 (路径模板、边权重、语义断裂、运动能量)
  - 新增 V4Result 数据类，包含审计结论、逻辑断裂点、解释文本
  - analyze_entity / analyze_all_entities 自动选择主路径或 fallback

Decision LLM 审计模式:
  1. 叙事引擎将图路径转为因果叙事文本
  2. 决策审计器调用 LLM 进行语义+动能+电影特征三维审计
  3. 返回结构化结论: is_anomaly, confidence, reason, break_timestamp

V3 Fallback 模式 (当 LLM 不可用或路径过短时启用):
  1. 路径模板偏离度
  2. 边权重突降
  3. 运动能量突增
  4. 语义跳变
  → 加权融合 + EMA 平滑
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import AnalysisConfig, DecisionConfig, NarrativeConfig
from ..association.temporal_graph import TemporalGraph
from .path_templates import compute_path_anomaly_score
from .causality_auditor import CausalityAuditor, AuditVerdict, VideoAuditReport

logger = logging.getLogger(__name__)


# ── 异常检测结果数据类 (V4.0 扩展) ──────────────────
@dataclass
class AnomalyResult:
    """
    单个实体的异常检测结果。

    V4.0 新增字段:
      - decision_verdict: Decision LLM 审计结论
      - break_timestamp:  逻辑断裂时间戳
      - reason_zh:        中文解释
      - is_cinematic_false_alarm: 是否电影伪信号
      - mode:             判定模式 ("decision_llm" | "fallback_v3")
    """
    entity_id: int
    anomaly_score: float                       # [0, 1] 综合异常分数
    is_anomaly: bool = False                   # V4.0: 布尔判定

    # V3 信号分 (Fallback 模式使用)
    path_score: float = 0.0
    edge_score: float = 0.0
    breakage_score: float = 0.0
    energy_score: float = 0.0

    # V4.0 审计结论
    break_timestamp: Optional[float] = None    # 逻辑断裂时间
    reason: str = ""                           # 英文原因
    reason_zh: str = ""                        # 中文原因
    is_cinematic_false_alarm: bool = False
    violated_contracts: list[str] = field(default_factory=list)
    mode: str = "fallback_v3"                  # "decision_llm" | "fallback_v3"

    matched_template: Optional[str] = None
    action_sequence: list[str] = field(default_factory=list)


@dataclass
class VideoAnomalyResult:
    """
    视频级异常检测结果 (V4.0)。
    """
    video_anomaly: bool = False
    video_score: float = 0.0
    entity_results: list[AnomalyResult] = field(default_factory=list)
    anomaly_entities: list[int] = field(default_factory=list)
    cinematic_filtered: list[int] = field(default_factory=list)
    scene_type: str = ""
    summary_zh: str = ""
    summary_en: str = ""
    mode: str = "fallback_v3"


class AnomalyDetector:
    """
    V4.0 异常检测器。

    主路径: Decision LLM 因果审计
    Fallback: V3 多信号加权融合

    使用方式:
      detector = AnomalyDetector(analysis_cfg, decision_cfg, narrative_cfg)
      result = detector.analyze_video(graph, scene_type)
    """

    def __init__(
        self,
        cfg: Optional[AnalysisConfig] = None,
        decision_cfg: Optional[DecisionConfig] = None,
        narrative_cfg: Optional[NarrativeConfig] = None,
        vllm_client=None,
    ):
        self.cfg = cfg or AnalysisConfig()
        self.decision_cfg = decision_cfg or DecisionConfig()
        self.narrative_cfg = narrative_cfg
        self._vllm_client = vllm_client

        # 实体级 EMA 状态 (V3 fallback)
        self._ema_scores: dict[int, float] = {}

        # Decision LLM 审计器 (V4.0)
        self._auditor: Optional[CausalityAuditor] = None
        self._prev_path_lengths: dict[int, int] = {}

    # ── V4.0 主路径入口 ──────────────────────────────────
    def analyze_video(
        self,
        graph: TemporalGraph,
        scene_type: str = "",
        frame_energies: Optional[dict[int, float]] = None,
        use_decision_llm: bool = True,
        snapshots: Optional[list] = None,
    ) -> VideoAnomalyResult:
        self._snapshots = snapshots  # 保存供审计层使用
        """
        视频级异常分析 (V4.0)。

        优先使用 Decision LLM 审计; 失败或不可用时回退到 V3 加权公式。

        Args:
            graph: 时间演化图
            scene_type: 场景类型
            frame_energies: {frame_id: motion_energy}
            use_decision_llm: 是否启用 Decision LLM

        Returns:
            VideoAnomalyResult
        """
        entity_ids = graph.get_all_entity_ids()

        if not entity_ids:
            return VideoAnomalyResult(
                summary_zh="无有效实体",
                summary_en="No valid entities",
            )

        # 尝试 Decision LLM 主路径
        if use_decision_llm:
            try:
                return self._analyze_via_decision_llm(
                    graph, scene_type, snapshots=self._snapshots
                )
            except Exception as e:
                logger.warning(
                    f"Decision LLM audit failed, falling back to V3: {e}"
                )

        # Fallback: V3 多信号加权
        return self._analyze_via_v3_fallback(
            graph, scene_type, frame_energies
        )

    # ── Decision LLM 主路径 ──────────────────────────────
    def _analyze_via_decision_llm(
        self,
        graph: TemporalGraph,
        scene_type: str,
        snapshots: Optional[list] = None,
    ) -> VideoAnomalyResult:
        """通过 Decision LLM 进行异常分析"""
        if self._auditor is None:
            self._auditor = CausalityAuditor(
                self.decision_cfg, self.narrative_cfg,
                vllm_client=self._vllm_client,
            )

        # 执行全实体审计 (传入感知快照供规则兜底)
        report: VideoAuditReport = self._auditor.audit_all_entities(
            graph, scene_type, self._prev_path_lengths,
            snapshots=snapshots,
        )

        # 更新路径长度记录
        self._prev_path_lengths = graph.get_path_length_changes()

        # 转换为统一 AnomalyResult 格式
        entity_results = []
        for verdict in report.entity_verdicts:
            result = self._verdict_to_result(verdict)
            entity_results.append(result)

        # 排序
        entity_results.sort(key=lambda r: r.anomaly_score, reverse=True)

        return VideoAnomalyResult(
            video_anomaly=report.video_anomaly,
            video_score=report.video_confidence,
            entity_results=entity_results,
            anomaly_entities=report.anomaly_entities,
            cinematic_filtered=report.cinematic_filtered,
            scene_type=scene_type,
            summary_zh=report.summary_zh,
            summary_en=report.summary_en,
            mode="decision_llm",
        )

    @staticmethod
    def _verdict_to_result(verdict: AuditVerdict) -> AnomalyResult:
        """将 AuditVerdict 转为 AnomalyResult"""
        return AnomalyResult(
            entity_id=verdict.entity_id,
            anomaly_score=verdict.confidence if verdict.is_anomaly else 0.0,
            is_anomaly=verdict.is_anomaly,
            break_timestamp=verdict.break_timestamp,
            reason=verdict.reason_en,
            reason_zh=verdict.reason_zh,
            is_cinematic_false_alarm=verdict.is_cinematic_false_alarm,
            violated_contracts=verdict.violated_contracts,
            mode="decision_llm",
        )

    # ── V3 Fallback ──────────────────────────────────────
    def _analyze_via_v3_fallback(
        self,
        graph: TemporalGraph,
        scene_type: str,
        frame_energies: Optional[dict[int, float]] = None,
    ) -> VideoAnomalyResult:
        """V3 多信号加权融合 (Fallback)"""
        results = self.analyze_all_entities(
            graph, scene_type, frame_energies
        )

        video_score = self.get_video_anomaly_score(results)
        threshold = self.cfg.decision_confidence_threshold

        anomaly_entities = [
            r.entity_id for r in results if r.anomaly_score >= threshold
        ]

        return VideoAnomalyResult(
            video_anomaly=video_score >= threshold,
            video_score=video_score,
            entity_results=results,
            anomaly_entities=anomaly_entities,
            scene_type=scene_type,
            summary_zh=f"V3 Fallback: 视频异常分={video_score:.2f}",
            summary_en=f"V3 Fallback: video anomaly score={video_score:.2f}",
            mode="fallback_v3",
        )

    # ── V3 信号计算 (保留完整) ────────────────────────────

    # 信号 1: 路径模板匹配
    def _compute_path_signal(
        self,
        action_sequence: list[str],
        scene_type: str,
    ) -> tuple[float, str, Optional[str]]:
        """路径模板匹配信号"""
        return compute_path_anomaly_score(
            action_sequence, scene_type, self.cfg
        )

    # 信号 2: 边权重异常
    def _compute_edge_signal(
        self,
        graph: TemporalGraph,
        entity_id: int,
    ) -> float:
        """
        边权重异常信号。
        连续低权重边 → 高异常分。
        """
        weights = graph.get_edge_weights_for_entity(entity_id)
        if not weights:
            return 0.0

        weights = np.array(weights)

        # 1. 平均边权重（越低越异常）
        avg_weight = np.mean(weights)
        avg_anomaly = max(0.0, 1.0 - avg_weight)

        # 2. 最低边权重
        min_weight = np.min(weights)
        min_anomaly = max(0.0, 1.0 - min_weight)

        # 3. 边权重方差（高方差 = 不稳定 = 更可疑）
        if len(weights) > 1:
            var_anomaly = min(1.0, np.std(weights) * 2)
        else:
            var_anomaly = 0.0

        score = 0.4 * avg_anomaly + 0.4 * min_anomaly + 0.2 * var_anomaly
        return float(np.clip(score, 0, 1))

    # 信号 3: 语义断裂检测
    def _compute_breakage_signal(
        self,
        graph: TemporalGraph,
        entity_id: int,
    ) -> float:
        """
        语义断裂信号。
        检测动作/位置的不可解释突变。
        """
        edges = graph.get_entity_edges(entity_id)
        if len(edges) < 2:
            return 0.0

        breakage_scores = []

        for edge in edges:
            action_score = edge.get("action_score", 0.5)
            time_gap = edge.get("time_gap", 1.0)

            # 低动作合理性 + 短时间间隔 = 强烈断裂信号
            if time_gap < 5.0:
                breakage = max(0.0, 1.0 - action_score)
            else:
                decay = min(1.0, time_gap / 30.0)
                breakage = max(0.0, (1.0 - action_score) * (1.0 - decay * 0.5))

            breakage_scores.append(breakage)

        if not breakage_scores:
            return 0.0

        max_break = max(breakage_scores)
        avg_break = np.mean(breakage_scores)

        if max_break + avg_break > 0:
            score = 2 * max_break * avg_break / (max_break + avg_break)
        else:
            score = 0.0

        return float(np.clip(score, 0, 1))

    # 信号 4: 运动能量异常
    def _compute_energy_signal(
        self,
        motion_energies: list[float],
    ) -> float:
        """
        运动能量异常信号。
        持续高能量 → 高异常分。
        """
        if not motion_energies:
            return 0.0

        energies = np.array(motion_energies)

        avg_energy = np.mean(energies)
        max_energy = np.max(energies)

        if len(energies) > 1:
            gradients = np.diff(energies)
            max_gradient = np.max(np.abs(gradients))
        else:
            max_gradient = 0.0

        score = 0.3 * avg_energy + 0.4 * max_energy + 0.3 * min(1.0, max_gradient * 3)
        return float(np.clip(score, 0, 1))

    # ── V3 综合分析 (保留用于 fallback) ──────────────────
    def analyze_entity(
        self,
        graph: TemporalGraph,
        entity_id: int,
        scene_type: str = "",
        motion_energies: Optional[list[float]] = None,
    ) -> AnomalyResult:
        """
        对单个实体进行 V3 多信号融合分析 (Fallback 模式)。
        """
        action_sequence = graph.get_entity_action_sequence(entity_id)

        if len(action_sequence) < self.cfg.min_path_length:
            return AnomalyResult(
                entity_id=entity_id,
                anomaly_score=0.0,
                reason="path too short",
                reason_zh="路径过短",
                matched_template=None,
                action_sequence=action_sequence,
                mode="fallback_v3",
            )

        # 信号 1: 路径模板
        path_score, path_reason, matched_tmpl = self._compute_path_signal(
            action_sequence, scene_type
        )

        # 信号 2: 边权重
        edge_score = self._compute_edge_signal(graph, entity_id)

        # 信号 3: 语义断裂
        breakage_score = self._compute_breakage_signal(graph, entity_id)

        # 信号 4: 运动能量
        energy_score = self._compute_energy_signal(motion_energies or [])

        # 加权融合
        raw_score = (
            0.30 * path_score
            + 0.25 * edge_score
            + 0.30 * breakage_score
            + 0.15 * energy_score
        )

        # EMA 平滑
        if entity_id in self._ema_scores:
            alpha = self.cfg.anomaly_ema_alpha
            smoothed = alpha * raw_score + (1 - alpha) * self._ema_scores[entity_id]
        else:
            smoothed = raw_score

        self._ema_scores[entity_id] = smoothed

        # 综合原因
        signals = []
        if path_score > 0.5:
            signals.append(f"path({path_score:.2f})")
        if edge_score > 0.5:
            signals.append(f"edge({edge_score:.2f})")
        if breakage_score > 0.5:
            signals.append(f"breakage({breakage_score:.2f})")
        if energy_score > 0.5:
            signals.append(f"energy({energy_score:.2f})")

        if signals:
            reason = f"anomaly signals: {', '.join(signals)}; {path_reason}"
            reason_zh = f"异常信号: {', '.join(signals)}"
        else:
            reason = f"normal; {path_reason}"
            reason_zh = "正常"

        threshold = self.cfg.decision_confidence_threshold
        is_anomaly = smoothed >= threshold

        return AnomalyResult(
            entity_id=entity_id,
            anomaly_score=float(np.clip(smoothed, 0, 1)),
            is_anomaly=is_anomaly,
            path_score=path_score,
            edge_score=edge_score,
            breakage_score=breakage_score,
            energy_score=energy_score,
            reason=reason,
            reason_zh=reason_zh,
            matched_template=matched_tmpl,
            action_sequence=action_sequence,
            mode="fallback_v3",
        )

    def analyze_all_entities(
        self,
        graph: TemporalGraph,
        scene_type: str = "",
        frame_energies: Optional[dict[int, float]] = None,
    ) -> list[AnomalyResult]:
        """
        V3 模式: 分析图中所有实体。
        """
        results = []
        frame_energies = frame_energies or {}

        for entity_id in graph.get_all_entity_ids():
            path = graph.get_entity_path(entity_id)
            energies = []
            for node in path:
                fid = node.get("frame_id", -1)
                if fid in frame_energies:
                    energies.append(frame_energies[fid])

            result = self.analyze_entity(
                graph, entity_id, scene_type, energies
            )
            results.append(result)

        results.sort(key=lambda r: r.anomaly_score, reverse=True)
        return results

    def get_video_anomaly_score(
        self,
        results: list[AnomalyResult],
    ) -> float:
        """
        汇总所有实体的异常分数为视频级别 (V3 Fallback)。
        策略: 取最高异常分数。
        """
        if not results:
            return 0.0

        valid = [r for r in results if len(r.action_sequence) >= self.cfg.min_path_length]
        if not valid:
            return 0.0

        return max(r.anomaly_score for r in valid)

    # ── 资源管理 ──────────────────────────────────────────
    def reset(self):
        """重置检测器状态"""
        self._ema_scores.clear()
        self._prev_path_lengths.clear()

    def cleanup(self):
        """释放资源 (包括 Decision LLM)"""
        self.reset()
        if self._auditor is not None:
            self._auditor.cleanup()
            self._auditor = None