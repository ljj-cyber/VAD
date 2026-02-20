"""
Stage 2-A: NodeTrigger — 稀疏语义节点触发策略

控制 VLLM 的调用频率，只在"关键时刻"生成语义节点:

  Rule 1 (Birth):       新 ID 出现的第一帧 → Trigger
  Rule 2 (Change Point): 当前 Embedding 与上一次**采样点**(非上一帧)的距离 > τ_jump → Trigger
  Rule 3 (Heartbeat):   距离上一次采样时间 > 3.0s → Trigger

每个触发都会在图中产生一个 TemporalNode，由 VLLM 填充语义标签。
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import NodeTriggerConfig
from ..tracking.entity_tracker import TraceEntry

logger = logging.getLogger(__name__)


@dataclass
class TriggerResult:
    """触发结果"""
    entity_id: int
    frame_idx: int
    timestamp: float
    trigger_rule: str           # "birth" | "change_point" | "heartbeat"
    embedding_distance: float   # 与上次采样点的 1-cosine 距离
    trace_entry: TraceEntry     # 对应的 TraceEntry (含 crop)


class NodeTrigger:
    """
    语义节点触发策略机。

    维护每个实体的"上次采样状态"，判断当前帧是否应该触发 VLLM 语义推理。
    """

    def __init__(self, cfg: Optional[NodeTriggerConfig] = None):
        self.cfg = cfg or NodeTriggerConfig()
        # entity_id → 上次采样信息
        self._last_sample: dict[int, dict] = {}
        # entity_id → 是否已经出现过（用于 birth 检测）
        self._known_entities: set[int] = set()

    def reset(self):
        """重置状态"""
        self._last_sample.clear()
        self._known_entities.clear()

    def check_triggers(
        self,
        entries: list[TraceEntry],
    ) -> list[TriggerResult]:
        """
        检查当前帧的所有 TraceEntry，返回需要触发 VLLM 的列表。

        Args:
            entries: EntityTracker.update() 的返回值

        Returns:
            list[TriggerResult]
        """
        triggers: list[TriggerResult] = []

        for entry in entries:
            eid = entry.entity_id
            trigger = self._check_single(entry)
            if trigger is not None:
                triggers.append(trigger)

        return triggers

    def _check_single(self, entry: TraceEntry) -> Optional[TriggerResult]:
        """检查单个实体是否触发"""
        eid = entry.entity_id

        # ── Rule 1: Birth ──
        if self.cfg.trigger_on_birth and eid not in self._known_entities:
            self._known_entities.add(eid)
            self._update_sample(entry)
            return TriggerResult(
                entity_id=eid,
                frame_idx=entry.frame_idx,
                timestamp=entry.timestamp,
                trigger_rule="birth",
                embedding_distance=1.0,
                trace_entry=entry,
            )

        last = self._last_sample.get(eid)
        if last is None:
            # 首次见到（理论上不会走到这里，但兜底）
            self._update_sample(entry)
            return TriggerResult(
                entity_id=eid,
                frame_idx=entry.frame_idx,
                timestamp=entry.timestamp,
                trigger_rule="birth",
                embedding_distance=1.0,
                trace_entry=entry,
            )

        # 最小间隔保护
        if (entry.frame_idx - last["frame_idx"]) < self.cfg.min_trigger_gap_frames:
            return None

        # ── Rule 2: Change Point ──
        last_emb = last["embedding"]
        cos_sim = float(np.dot(entry.embedding, last_emb))
        distance = 1.0 - cos_sim

        if distance > self.cfg.embedding_jump_threshold:
            self._update_sample(entry)
            return TriggerResult(
                entity_id=eid,
                frame_idx=entry.frame_idx,
                timestamp=entry.timestamp,
                trigger_rule="change_point",
                embedding_distance=round(distance, 4),
                trace_entry=entry,
            )

        # ── Rule 3: Heartbeat ──
        time_since = entry.timestamp - last["timestamp"]
        if time_since >= self.cfg.heartbeat_interval_sec:
            self._update_sample(entry)
            return TriggerResult(
                entity_id=eid,
                frame_idx=entry.frame_idx,
                timestamp=entry.timestamp,
                trigger_rule="heartbeat",
                embedding_distance=round(distance, 4),
                trace_entry=entry,
            )

        return None

    def _update_sample(self, entry: TraceEntry):
        """更新某实体的上次采样记录"""
        self._last_sample[entry.entity_id] = {
            "frame_idx": entry.frame_idx,
            "timestamp": entry.timestamp,
            "embedding": entry.embedding.copy(),
        }

    def check_heartbeat_for_active(
        self,
        active_entity_ids: list[int],
        current_timestamp: float,
        current_frame_idx: int,
        entity_trace_buffer: dict[int, list],
        triggered_eids: set[int] | None = None,
    ) -> list[TriggerResult]:
        """
        对不在当前帧 regions 中的活跃实体，检查是否需要心跳触发。

        当实体不再产生运动区域（停止运动或遮挡），仍可通过心跳
        拿到新的语义节点，使动态图产生 Edge。

        Args:
            active_entity_ids: 当前活跃实体 ID 列表
            current_timestamp: 当前帧时间戳 (秒)
            current_frame_idx: 当前帧序号
            entity_trace_buffer: entity_id → list[TraceEntry]
            triggered_eids: 当前帧已经触发过的实体 ID (跳过)

        Returns:
            list[TriggerResult]
        """
        triggered_eids = triggered_eids or set()
        triggers: list[TriggerResult] = []

        for eid in active_entity_ids:
            if eid in triggered_eids:
                continue

            last = self._last_sample.get(eid)
            if last is None:
                continue

            # 心跳间隔检查
            time_since = current_timestamp - last["timestamp"]
            if time_since < self.cfg.heartbeat_interval_sec:
                continue

            # 用实体最近一条 trace entry 作为代理
            buf = entity_trace_buffer.get(eid, [])
            if not buf:
                continue
            last_entry = buf[-1]

            # 计算 embedding 距离
            cos_sim = float(np.dot(last_entry.embedding, last["embedding"]))
            distance = 1.0 - cos_sim

            # 更新采样点
            self._last_sample[eid] = {
                "frame_idx": current_frame_idx,
                "timestamp": current_timestamp,
                "embedding": last_entry.embedding.copy(),
            }

            triggers.append(TriggerResult(
                entity_id=eid,
                frame_idx=current_frame_idx,
                timestamp=current_timestamp,
                trigger_rule="heartbeat",
                embedding_distance=round(distance, 4),
                trace_entry=last_entry,
            ))

        return triggers

    def get_trigger_stats(self) -> dict:
        """统计信息"""
        return {
            "known_entities": len(self._known_entities),
            "active_samples": len(self._last_sample),
        }
