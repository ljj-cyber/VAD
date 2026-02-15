"""
Phase 2-B: GlobalHeartbeat — 低频全图语义心跳 + CLIP 漂移检测

针对 Abuse030 这类低动态异常:
  - 每 heartbeat_sec 秒进行一次强制全图采样（不依赖帧差连通域）
  - 计算全局 CLIP Embedding 的累计偏移量（Semantic Drift）
  - 偏移过大即触发 VLLM 全图扫描

产出:
  - 全图 VLLM 语义节点（补充 entity-based 的盲区）
  - CLIP drift 时间线（供 NarrativeEngine 使用）
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatResult:
    """全图心跳采样结果"""
    frame_idx: int
    timestamp: float
    trigger_reason: str          # "periodic" | "drift"
    clip_embedding: Optional[np.ndarray] = None
    drift_from_baseline: float = 0.0


class GlobalHeartbeat:
    """
    低频全图语义心跳。

    独立于实体追踪运行，定期对全图进行 CLIP 编码和可选的 VLLM 扫描。
    """

    def __init__(
        self,
        heartbeat_sec: float = 2.5,
        drift_threshold: float = 0.18,
        drift_window: int = 3,
    ):
        self.heartbeat_sec = heartbeat_sec
        self.drift_threshold = drift_threshold
        self.drift_window = drift_window

        self._last_heartbeat_time: float = -999.0
        self._baseline_embedding: Optional[np.ndarray] = None
        self._recent_embeddings: list[np.ndarray] = []
        self._recent_timestamps: list[float] = []

        # 收集
        self.heartbeat_results: list[HeartbeatResult] = []
        self.all_embeddings: list[np.ndarray] = []
        self.all_timestamps: list[float] = []

    def reset(self):
        self._last_heartbeat_time = -999.0
        self._baseline_embedding = None
        self._recent_embeddings = []
        self._recent_timestamps = []
        self.heartbeat_results = []
        self.all_embeddings = []
        self.all_timestamps = []

    def should_heartbeat(self, timestamp: float) -> bool:
        """是否应该执行心跳采样"""
        return (timestamp - self._last_heartbeat_time) >= self.heartbeat_sec

    def update(
        self,
        frame_idx: int,
        timestamp: float,
        clip_embedding: np.ndarray,
    ) -> Optional[HeartbeatResult]:
        """
        更新全局 CLIP embedding 并检查是否需要触发心跳。

        Args:
            frame_idx: 帧号
            timestamp: 秒
            clip_embedding: 全图 CLIP embedding (D,)

        Returns:
            HeartbeatResult 如果触发了心跳，否则 None
        """
        self.all_embeddings.append(clip_embedding)
        self.all_timestamps.append(timestamp)

        # 初始化 baseline
        if self._baseline_embedding is None:
            self._baseline_embedding = clip_embedding.copy()

        # 计算漂移
        drift = 1.0 - float(np.dot(clip_embedding, self._baseline_embedding))

        # 更新滑动窗口
        self._recent_embeddings.append(clip_embedding)
        self._recent_timestamps.append(timestamp)
        if len(self._recent_embeddings) > self.drift_window:
            self._recent_embeddings.pop(0)
            self._recent_timestamps.pop(0)

        trigger_reason = None

        # 检查定期心跳
        if self.should_heartbeat(timestamp):
            trigger_reason = "periodic"

        # 检查漂移心跳
        if drift > self.drift_threshold:
            trigger_reason = "drift"
            # 更新 baseline 到当前（漂移已检测到）
            self._baseline_embedding = clip_embedding.copy()

        if trigger_reason:
            self._last_heartbeat_time = timestamp
            result = HeartbeatResult(
                frame_idx=frame_idx,
                timestamp=timestamp,
                trigger_reason=trigger_reason,
                clip_embedding=clip_embedding,
                drift_from_baseline=drift,
            )
            self.heartbeat_results.append(result)
            return result

        return None

    def get_drift_timeline(self) -> list[tuple[float, float]]:
        """获取 (timestamp, drift) 时间线"""
        if len(self.all_embeddings) < 2:
            return []
        baseline = self.all_embeddings[0]
        timeline = []
        for i, (emb, ts) in enumerate(zip(self.all_embeddings, self.all_timestamps)):
            drift = 1.0 - float(np.dot(emb, baseline))
            timeline.append((ts, drift))
        return timeline

    def get_max_drift(self) -> float:
        """获取最大漂移量"""
        timeline = self.get_drift_timeline()
        if not timeline:
            return 0.0
        return max(d for _, d in timeline)
