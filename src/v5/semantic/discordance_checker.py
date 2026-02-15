"""
Phase 2-A: DiscordanceChecker — 物理-语义矛盾检测器 (v2: 自适应阈值)

核心逻辑:
  动能阈值不再全局固定，而是基于该视频前 N 帧的背景动能 μ+3σ 自适应。
  只有显著超过背景流的动能才标记"矛盾"，避免正常行人运动误触发。

检测:
  - energy_semantic_gap: 动能显著超背景 但 VLLM danger_score 低
  - semantic_drift: 全局 CLIP embedding 突变
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiscordanceAlert:
    """矛盾警报"""
    entity_id: int
    frame_idx: int
    timestamp: float
    alert_type: str            # "energy_semantic_gap" | "semantic_drift"
    motion_energy: float
    danger_score: float
    description: str


class DiscordanceChecker:
    """
    物理-语义矛盾检测器（自适应动能阈值版）。

    使用视频的背景动能统计 (μ + k·σ) 作为阈值，
    而非全局固定常数。
    """

    def __init__(
        self,
        sigma_multiplier: float = 3.0,
        min_energy_threshold: float = 0.05,
        danger_ceiling: float = 0.10,
        drift_threshold: float = 0.20,
    ):
        """
        Args:
            sigma_multiplier: 背景动能 μ + k·σ 中的 k
            min_energy_threshold: 动能阈值的最小值（防止极静场景阈值为0）
            danger_ceiling: 当 VLLM danger_score < 此值时才算"语义保守"
            drift_threshold: CLIP 漂移阈值
        """
        self.sigma_multiplier = sigma_multiplier
        self.min_energy_threshold = min_energy_threshold
        self.danger_ceiling = danger_ceiling
        self.drift_threshold = drift_threshold

        # 运行时计算的自适应阈值
        self._adaptive_threshold: float = min_energy_threshold

    def calibrate(self, background_energies: list[float]):
        """
        用视频前 N 帧的动能校准背景阈值。

        Args:
            background_energies: 前 N 帧每帧的全局动能 (由 MotionExtractor 产生)
        """
        if not background_energies or len(background_energies) < 3:
            self._adaptive_threshold = self.min_energy_threshold
            return

        arr = np.array(background_energies, dtype=np.float64)
        mu = float(arr.mean())
        sigma = float(arr.std())
        adaptive = mu + self.sigma_multiplier * sigma

        # 不低于最小阈值
        self._adaptive_threshold = max(adaptive, self.min_energy_threshold)

        logger.info(
            f"DiscordanceChecker calibrated: μ={mu:.4f}, σ={sigma:.4f}, "
            f"threshold=μ+{self.sigma_multiplier}σ={adaptive:.4f} "
            f"→ effective={self._adaptive_threshold:.4f}"
        )

    @property
    def threshold(self) -> float:
        return self._adaptive_threshold

    def check_entity(
        self,
        semantic_results: list[dict],
        trace_energies: list[float],
    ) -> list[DiscordanceAlert]:
        """
        检查单个实体的语义-物理矛盾。
        """
        alerts = []

        for sr in semantic_results:
            eid = sr.get("entity_id", -1)
            fidx = sr.get("frame_idx", 0)
            ts = sr.get("timestamp", 0.0)
            danger = sr.get("danger_score", 0.0)

            # 该节点附近的动能
            ke = sr.get("kinetic_energy", 0.0)
            if trace_energies:
                ke = max(ke, max(trace_energies) if trace_energies else 0.0)

            # ── 矛盾: 动能显著超背景 + 语义保守 ──
            if ke > self._adaptive_threshold and danger < self.danger_ceiling:
                excess = ke / max(self._adaptive_threshold, 1e-6)
                alerts.append(DiscordanceAlert(
                    entity_id=eid,
                    frame_idx=fidx,
                    timestamp=ts,
                    alert_type="energy_semantic_gap",
                    motion_energy=ke,
                    danger_score=danger,
                    description=(
                        f"Motion energy ({ke:.4f}) exceeds adaptive background "
                        f"threshold ({self._adaptive_threshold:.4f}) by {excess:.1f}x, "
                        f"but semantic danger_score is only {danger:.2f}. "
                        f"Possible perception blind spot — physical change is real "
                        f"but semantic description may be incomplete."
                    ),
                ))

        return alerts

    def check_global_drift(
        self,
        clip_embeddings: list[np.ndarray],
        timestamps: list[float],
    ) -> list[DiscordanceAlert]:
        """检测全局 CLIP Semantic Drift。"""
        alerts = []
        if len(clip_embeddings) < 2:
            return alerts

        for i in range(1, len(clip_embeddings)):
            cos_sim = float(np.dot(clip_embeddings[i], clip_embeddings[i - 1]))
            drift = 1.0 - cos_sim

            if drift > self.drift_threshold:
                alerts.append(DiscordanceAlert(
                    entity_id=-1,
                    frame_idx=-1,
                    timestamp=timestamps[i],
                    alert_type="semantic_drift",
                    motion_energy=0.0,
                    danger_score=0.0,
                    description=(
                        f"Large semantic drift ({drift:.3f}) at T={timestamps[i]:.1f}s. "
                        f"Scene may have changed dramatically (fire, explosion, etc.)."
                    ),
                ))

        return alerts

    def check_video(
        self,
        semantic_results: list[dict],
        entity_trace_energies: dict[int, list[float]],
    ) -> list[DiscordanceAlert]:
        """视频级矛盾检测。"""
        alerts = []

        by_entity: dict[int, list[dict]] = {}
        for sr in semantic_results:
            eid = sr.get("entity_id", -1)
            if eid not in by_entity:
                by_entity[eid] = []
            by_entity[eid].append(sr)

        for eid, srs in by_entity.items():
            energies = entity_trace_energies.get(eid, [])
            entity_alerts = self.check_entity(srs, energies)
            alerts.extend(entity_alerts)

        return alerts
