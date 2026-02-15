"""
Phase 2-A: DiscordanceChecker — 物理-语义矛盾检测器 (v3: 多实体投票 + 高阈值)

核心逻辑:
  动能阈值基于该视频前 N 帧的背景动能 μ+kσ 自适应。
  在此基础上增加三重过滤以抑制误报:
    1. 提高 sigma_multiplier (5.0) 和 min_energy_threshold (0.15)
    2. 要求动能至少超阈值 min_excess_ratio (2.5x) 才触发
    3. 多实体投票: 当 >50% 的活跃实体同时超标时，视为正常场景活跃，抑制 discordance

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

    # 动能峰值定位（由 check_entity 计算）
    peak_energy_time: float = 0.0     # 该实体动能峰值时刻 (秒)
    peak_energy_value: float = 0.0    # 峰值动能
    burst_start_sec: float = 0.0      # 动能突发区间起点
    burst_end_sec: float = 0.0        # 动能突发区间终点


class DiscordanceChecker:
    """
    物理-语义矛盾检测器（v3: 多实体投票 + 高阈值版）。

    改进:
      1. 提高自适应阈值 σ 系数 (3→5) 和最小阈值 (0.05→0.15)
      2. 新增 min_excess_ratio: 动能 / 阈值 必须 ≥ 该倍率
      3. 新增 voting_suppress_ratio: 当超标实体占比 ≥ 该值时，
         视为场景整体活跃（非异常），抑制所有 discordance
    """

    def __init__(
        self,
        sigma_multiplier: float = 5.0,
        min_energy_threshold: float = 0.15,
        danger_ceiling: float = 0.10,
        drift_threshold: float = 0.20,
        min_excess_ratio: float = 2.5,
        voting_suppress_ratio: float = 0.5,
    ):
        """
        Args:
            sigma_multiplier: 背景动能 μ + k·σ 中的 k (从 3→5)
            min_energy_threshold: 动能阈值的最小值 (从 0.05→0.15)
            danger_ceiling: 当 VLLM danger_score < 此值时才算"语义保守"
            drift_threshold: CLIP 漂移阈值
            min_excess_ratio: 动能/阈值 的最小倍率 (新增, ≥2.5x)
            voting_suppress_ratio: 超标实体占活跃实体的比例超此值时
                                   视为正常场景活跃，抑制 discordance (新增, 0.5=50%)
        """
        self.sigma_multiplier = sigma_multiplier
        self.min_energy_threshold = min_energy_threshold
        self.danger_ceiling = danger_ceiling
        self.drift_threshold = drift_threshold
        self.min_excess_ratio = min_excess_ratio
        self.voting_suppress_ratio = voting_suppress_ratio

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

    def _compute_energy_peak_and_burst(
        self,
        trace_time_energy: list[tuple[float, float]],
    ) -> tuple[float, float, float, float]:
        """
        从 (timestamp, kinetic_energy) 序列中计算峰值时间和突发区间。

        突发区间: 峰值附近连续超过自适应阈值的时段，前后各扩展 2 秒余量。

        Returns:
            (peak_time, peak_value, burst_start, burst_end)
        """
        if not trace_time_energy:
            return 0.0, 0.0, 0.0, 0.0

        # 找峰值
        peak_time, peak_value = max(trace_time_energy, key=lambda x: x[1])

        # 找突发区间: 以峰值为中心，向两侧扩展到低于阈值处
        threshold = self._adaptive_threshold
        sorted_te = sorted(trace_time_energy, key=lambda x: x[0])

        # 找到峰值在排序序列中的位置
        peak_idx = 0
        for i, (t, e) in enumerate(sorted_te):
            if abs(t - peak_time) < 1e-6:
                peak_idx = i
                break

        # 向左扩展
        burst_start_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if sorted_te[i][1] >= threshold:
                burst_start_idx = i
            else:
                break

        # 向右扩展
        burst_end_idx = peak_idx
        for i in range(peak_idx + 1, len(sorted_te)):
            if sorted_te[i][1] >= threshold:
                burst_end_idx = i
            else:
                break

        margin = 2.0  # 前后扩展 2 秒
        burst_start = max(0.0, sorted_te[burst_start_idx][0] - margin)
        burst_end = sorted_te[burst_end_idx][0] + margin

        return peak_time, peak_value, burst_start, burst_end

    def check_entity(
        self,
        semantic_results: list[dict],
        trace_energies: list[float],
        trace_time_energy: list[tuple[float, float]] | None = None,
    ) -> list[DiscordanceAlert]:
        """
        检查单个实体的语义-物理矛盾。

        Args:
            semantic_results: 该实体的语义推理结果列表
            trace_energies: 该实体所有帧的动能值列表 (向后兼容)
            trace_time_energy: (timestamp, kinetic_energy) 列表 (可选，用于峰值定位)
        """
        alerts = []

        # 预计算该实体的动能峰值和突发区间
        peak_time, peak_value, burst_start, burst_end = 0.0, 0.0, 0.0, 0.0
        if trace_time_energy:
            peak_time, peak_value, burst_start, burst_end = \
                self._compute_energy_peak_and_burst(trace_time_energy)

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
            # 条件 1: 动能超过自适应阈值
            # 条件 2: 超标倍率 ≥ min_excess_ratio (新增: 防止仅略超阈值即触发)
            # 条件 3: danger_score 低于 danger_ceiling
            if ke > self._adaptive_threshold and danger < self.danger_ceiling:
                excess = ke / max(self._adaptive_threshold, 1e-6)
                if excess < self.min_excess_ratio:
                    # 超标不够显著，跳过
                    continue
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
                    peak_energy_time=peak_time,
                    peak_energy_value=peak_value,
                    burst_start_sec=burst_start,
                    burst_end_sec=burst_end,
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
        entity_trace_time_energy: dict[int, list[tuple[float, float]]] | None = None,
    ) -> list[DiscordanceAlert]:
        """
        视频级矛盾检测 (含多实体投票抑制)。

        Args:
            semantic_results: 所有语义推理结果
            entity_trace_energies: entity_id → [kinetic_energy, ...] (向后兼容)
            entity_trace_time_energy: entity_id → [(timestamp, kinetic_energy), ...]
                用于动能峰值定位
        """
        alerts = []

        by_entity: dict[int, list[dict]] = {}
        for sr in semantic_results:
            eid = sr.get("entity_id", -1)
            if eid not in by_entity:
                by_entity[eid] = []
            by_entity[eid].append(sr)

        for eid, srs in by_entity.items():
            energies = entity_trace_energies.get(eid, [])
            time_energy = (entity_trace_time_energy or {}).get(eid)
            entity_alerts = self.check_entity(
                srs, energies, trace_time_energy=time_energy
            )
            alerts.extend(entity_alerts)

        # ── 多实体投票抑制 ──────────────────────────────
        # 如果超标实体占活跃实体的比例 ≥ voting_suppress_ratio，
        # 说明是整个场景都很活跃（如繁忙路口、行人街道），不是个体异常。
        # 此时抑制所有 energy_semantic_gap 类型的 discordance alert。
        if alerts:
            total_tracked = len(entity_trace_energies)
            alerted_eids = set(a.entity_id for a in alerts if a.alert_type == "energy_semantic_gap")
            n_alerted = len(alerted_eids)

            if total_tracked > 0:
                alert_ratio = n_alerted / total_tracked
                if alert_ratio >= self.voting_suppress_ratio and n_alerted >= 2:
                    logger.info(
                        f"Multi-entity voting suppression: "
                        f"{n_alerted}/{total_tracked} entities ({alert_ratio:.0%}) "
                        f"have energy_semantic_gap alerts — likely normal scene activity. "
                        f"Suppressing all {len(alerts)} discordance alerts."
                    )
                    # 仅保留非 energy_semantic_gap 类型的 alert (如 semantic_drift)
                    alerts = [a for a in alerts if a.alert_type != "energy_semantic_gap"]

        return alerts
