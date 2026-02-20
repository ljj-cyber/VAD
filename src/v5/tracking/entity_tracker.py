"""
Stage 1-C: EntityTracker — 贪婪匹配 + 实体追踪

核心逻辑:
  1. 计算当前帧 crop embedding 与上一帧活跃 Entity 的余弦相似度
  2. 贪婪匹配:
     - 相似度 ≥ threshold → 沿用旧 ID（同一实体）
     - 相似度 < threshold 且动能大 → 分配新 ID（新实体进场）
  3. 维护活跃实体池，超龄清除

产出: trace_log — frame_id, entity_id, embedding, bbox
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import TrackerConfig
from .motion_extractor import MotionRegion

logger = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    """trace_log 中的一条记录"""
    frame_idx: int
    timestamp: float
    entity_id: int
    bbox: tuple[int, int, int, int]   # (x, y, w, h)
    embedding: np.ndarray              # CLIP feature (D,)
    kinetic_energy: float
    crop_image: Optional[np.ndarray] = None  # BGR crop (可选保留)


@dataclass
class _ActiveEntity:
    """内部活跃实体状态"""
    entity_id: int
    last_embedding: np.ndarray
    last_frame_idx: int
    last_timestamp: float
    last_bbox: tuple[int, int, int, int]
    birth_frame: int
    hit_count: int = 1


class EntityTracker:
    """
    贪婪匹配实体追踪器。

    每帧接收 MotionRegion 列表和对应的 CLIP embeddings，
    输出分配好 entity_id 的 TraceEntry 列表。
    """

    def __init__(self, cfg: Optional[TrackerConfig] = None):
        self.cfg = cfg or TrackerConfig()
        self._active: list[_ActiveEntity] = []
        self._next_id: int = 0
        self._trace_log: list[TraceEntry] = []

    def reset(self):
        """重置追踪器（处理新视频时调用）"""
        self._active = []
        self._next_id = 0
        self._trace_log = []

    @property
    def trace_log(self) -> list[TraceEntry]:
        return self._trace_log

    def update(
        self,
        frame_idx: int,
        timestamp: float,
        regions: list[MotionRegion],
        embeddings: np.ndarray,
        keep_crops: bool = False,
    ) -> list[TraceEntry]:
        """
        处理一帧的动能区域和 CLIP 特征，分配 entity_id。

        Args:
            frame_idx: 帧序号
            timestamp: 秒
            regions: MotionExtractor 输出的连通域列表
            embeddings: shape (N, D)，与 regions 一一对应
            keep_crops: 是否在 TraceEntry 中保留 crop 图像

        Returns:
            当前帧的 TraceEntry 列表
        """
        # 清除过期实体
        self._prune_aged(frame_idx)

        entries: list[TraceEntry] = []

        if len(regions) == 0:
            return entries

        n_regions = len(regions)
        n_active = len(self._active)

        # 贪婪匹配
        matched_active: set[int] = set()  # 已匹配的 active 实体索引
        matched_region: set[int] = set()  # 已匹配的 region 索引

        if n_active > 0 and n_regions > 0:
            # 计算余弦相似度矩阵 (n_regions x n_active)
            active_embs = np.stack(
                [a.last_embedding for a in self._active], axis=0
            )  # (n_active, D)
            # embeddings: (n_regions, D), active_embs: (n_active, D)
            sim_matrix = embeddings @ active_embs.T  # (n_regions, n_active)

            # 贪婪：每次取全局最大值
            for _ in range(min(n_regions, n_active)):
                # 屏蔽已匹配的
                mask = sim_matrix.copy()
                for ri in matched_region:
                    mask[ri, :] = -1.0
                for ai in matched_active:
                    mask[:, ai] = -1.0

                best = np.unravel_index(mask.argmax(), mask.shape)
                ri, ai = int(best[0]), int(best[1])
                best_sim = float(mask[ri, ai])

                if best_sim < self.cfg.similarity_threshold:
                    break

                # 匹配成功：沿用旧 ID
                ent = self._active[ai]
                ent.last_embedding = embeddings[ri]
                ent.last_frame_idx = frame_idx
                ent.last_timestamp = timestamp
                ent.last_bbox = (
                    regions[ri].x, regions[ri].y,
                    regions[ri].w, regions[ri].h,
                )
                ent.hit_count += 1

                entry = TraceEntry(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    entity_id=ent.entity_id,
                    bbox=ent.last_bbox,
                    embedding=embeddings[ri],
                    kinetic_energy=regions[ri].kinetic_energy,
                    crop_image=regions[ri].crop_image if keep_crops else None,
                )
                entries.append(entry)
                self._trace_log.append(entry)

                matched_active.add(ai)
                matched_region.add(ri)

        # 未匹配的 region → 分配新 ID
        # YOLO 检出的异常相关类别（person 等）跳过动能门限；
        # 静态背景（truck/car 等）仍需动能校验，避免噪声实体干扰追踪。
        _YOLO_EXEMPT_CLASSES = {
            "person", "fire", "smoke", "knife", "gun",
            "explosion", "blood", "bat", "hammer", "crowbar",
        }
        for ri in range(n_regions):
            if ri in matched_region:
                continue

            region = regions[ri]
            yolo_exempt = (
                getattr(region, "source", "") in ("yolo", "fused")
                and getattr(region, "class_name", "") in _YOLO_EXEMPT_CLASSES
            )
            if not yolo_exempt and region.kinetic_energy < self.cfg.min_kinetic_for_new:
                continue

            if len(self._active) >= self.cfg.max_active_entities:
                # 淘汰最老的
                self._active.sort(key=lambda a: a.last_frame_idx)
                self._active.pop(0)

            new_id = self._next_id
            self._next_id += 1

            new_ent = _ActiveEntity(
                entity_id=new_id,
                last_embedding=embeddings[ri],
                last_frame_idx=frame_idx,
                last_timestamp=timestamp,
                last_bbox=(region.x, region.y, region.w, region.h),
                birth_frame=frame_idx,
            )
            self._active.append(new_ent)

            entry = TraceEntry(
                frame_idx=frame_idx,
                timestamp=timestamp,
                entity_id=new_id,
                bbox=new_ent.last_bbox,
                embedding=embeddings[ri],
                kinetic_energy=region.kinetic_energy,
                crop_image=region.crop_image if keep_crops else None,
            )
            entries.append(entry)
            self._trace_log.append(entry)

            logger.debug(
                f"New entity #{new_id} at frame {frame_idx}, "
                f"energy={region.kinetic_energy:.3f}"
            )

        return entries

    def _prune_aged(self, current_frame: int):
        """清除超龄实体"""
        max_age = self.cfg.max_age_frames
        self._active = [
            a for a in self._active
            if (current_frame - a.last_frame_idx) <= max_age
        ]

    def get_entity_trace(self, entity_id: int) -> list[TraceEntry]:
        """获取某实体的完整轨迹"""
        return [e for e in self._trace_log if e.entity_id == entity_id]

    def get_active_entity_ids(self) -> list[int]:
        """获取当前活跃实体 ID 列表"""
        return [a.entity_id for a in self._active]

    def get_all_entity_ids(self) -> set[int]:
        """获取所有出现过的实体 ID"""
        return set(e.entity_id for e in self._trace_log)

    def export_trace_log(self) -> list[dict]:
        """导出 trace_log 为可 JSON 序列化的字典列表"""
        out = []
        for e in self._trace_log:
            out.append({
                "frame_idx": e.frame_idx,
                "timestamp": round(e.timestamp, 3),
                "entity_id": e.entity_id,
                "bbox": list(e.bbox),
                "kinetic_energy": round(e.kinetic_energy, 4),
            })
        return out
