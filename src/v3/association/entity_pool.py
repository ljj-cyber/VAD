"""
V3.0 时间关联层 — 语义 Re-ID 实体池

核心思想:
  不使用 YOLO / DeepSort 等视觉追踪器。
  仅通过 VLLM 输出的文字画像 (portrait) 的语义向量相似度
  来判断跨帧的"同一实体"。

流程:
  1. 每帧 VLLM 返回 entities 列表。
  2. 对每个 entity 的 portrait 文本做 Sentence-BERT 编码。
  3. 与实体池中已有实体的 portrait 嵌入计算余弦相似度。
  4. 若最高相似度 >= 阈值，判定为同一实体（Re-ID 命中）。
  5. 否则创建新实体节点。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..config import AssociationConfig, SystemConfig, HF_CACHE_DIR

logger = logging.getLogger(__name__)


@dataclass
class EntityRecord:
    """实体池中的一条记录"""

    entity_id: int  # 全局唯一 ID
    portrait: str  # 最新的画像描述
    embedding: np.ndarray  # 画像的语义嵌入向量
    last_frame_id: int  # 最后出现的帧 ID
    last_timestamp: float  # 最后出现的时间戳
    first_frame_id: int  # 首次出现的帧 ID
    first_timestamp: float  # 首次出现的时间戳
    appearance_count: int = 1  # 出现次数
    portrait_history: list[str] = field(default_factory=list)  # 画像历史

    def update(
        self,
        portrait: str,
        embedding: np.ndarray,
        frame_id: int,
        timestamp: float,
    ):
        """更新实体记录（Re-ID 命中时调用）"""
        self.portrait_history.append(self.portrait)
        self.portrait = portrait
        # 使用指数移动平均更新嵌入，避免画像漂移
        alpha = 0.3  # 新嵌入权重
        self.embedding = alpha * embedding + (1 - alpha) * self.embedding
        # 重新归一化
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm
        self.last_frame_id = frame_id
        self.last_timestamp = timestamp
        self.appearance_count += 1


class EntityPool:
    """
    语义 Re-ID 实体池。

    维护一个"活跃实体"集合，每帧的新实体描述与池中实体
    做语义匹配，实现纯文本驱动的跨帧目标重识别。
    """

    def __init__(self, cfg: Optional[AssociationConfig] = None):
        self.cfg = cfg or AssociationConfig()
        self._pool: dict[int, EntityRecord] = {}  # entity_id -> record
        self._next_id = 0
        self._sbert_model = None
        self._device = SystemConfig.device

    # ── Sentence-BERT 管理（线程安全单例）────────────────
    _sbert_global = None           # 类级全局 SBERT 实例
    _sbert_lock = __import__("threading").Lock()

    def _load_sbert(self):
        """延迟加载 Sentence-BERT 模型 (线程安全，全局单例)"""
        if self._sbert_model is not None:
            return

        with EntityPool._sbert_lock:
            if EntityPool._sbert_global is not None:
                self._sbert_model = EntityPool._sbert_global
                return

            logger.info(f"Loading Sentence-BERT: {self.cfg.sbert_model}")
            try:
                from sentence_transformers import SentenceTransformer

                EntityPool._sbert_global = SentenceTransformer(
                    self.cfg.sbert_model,
                    cache_folder=str(HF_CACHE_DIR / "hub"),
                    device=self._device,
                )
                self._sbert_model = EntityPool._sbert_global
                logger.info("Sentence-BERT loaded successfully.")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise

    def _encode_text(self, text: str) -> np.ndarray:
        """将文本编码为归一化的语义向量"""
        self._load_sbert()
        embedding = self._sbert_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码文本"""
        self._load_sbert()
        embeddings = self._sbert_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.astype(np.float32)

    # ── 核心匹配逻辑 ─────────────────────────────────
    def match_entities(
        self,
        entities: list[dict],
        frame_id: int,
        timestamp: float,
    ) -> list[tuple[dict, int, bool]]:
        """
        将当前帧的实体列表与实体池进行匹配。

        Args:
            entities: VLLM 输出的 entities 列表
            frame_id: 当前帧 ID
            timestamp: 当前帧时间戳

        Returns:
            list of (entity_dict, entity_id, is_new):
              - entity_dict: 原始实体信息
              - entity_id: 分配的全局实体 ID
              - is_new: 是否为新实体
        """
        if not entities:
            return []

        # 批量编码当前帧所有实体的 portrait
        portraits = [e.get("portrait", "unknown") for e in entities]
        new_embeddings = self._encode_batch(portraits)

        # 获取池中活跃实体
        active_records = self._get_active_records(frame_id)

        results = []

        if not active_records:
            # 池为空，全部创建新实体
            for entity, emb in zip(entities, new_embeddings):
                eid = self._create_entity(entity, emb, frame_id, timestamp)
                results.append((entity, eid, True))
            return results

        # 构建池中嵌入矩阵
        pool_ids = list(active_records.keys())
        pool_embeddings = np.stack(
            [active_records[pid].embedding for pid in pool_ids]
        )

        # 计算余弦相似度矩阵: [num_new, num_pool]
        sim_matrix = new_embeddings @ pool_embeddings.T

        # 贪心匹配（防止多个新实体匹配到同一个池实体）
        matched_pool_ids = set()

        for i, (entity, emb) in enumerate(zip(entities, new_embeddings)):
            # 找到最佳匹配
            best_sim = -1.0
            best_pool_idx = -1

            for j, pid in enumerate(pool_ids):
                if pid in matched_pool_ids:
                    continue
                if sim_matrix[i, j] > best_sim:
                    best_sim = sim_matrix[i, j]
                    best_pool_idx = j

            if best_sim >= self.cfg.reid_similarity_threshold and best_pool_idx >= 0:
                # Re-ID 命中
                matched_eid = pool_ids[best_pool_idx]
                matched_pool_ids.add(matched_eid)
                self._pool[matched_eid].update(
                    portraits[i], emb, frame_id, timestamp
                )
                results.append((entity, matched_eid, False))
                logger.debug(
                    f"Re-ID hit: entity '{portraits[i][:50]}' "
                    f"-> pool #{matched_eid} (sim={best_sim:.3f})"
                )
            else:
                # 创建新实体
                eid = self._create_entity(entity, emb, frame_id, timestamp)
                results.append((entity, eid, True))
                logger.debug(
                    f"New entity: '{portraits[i][:50]}' -> #{eid} "
                    f"(best_sim={best_sim:.3f})"
                )

        return results

    # ── 池管理 ────────────────────────────────────────
    def _create_entity(
        self,
        entity: dict,
        embedding: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> int:
        """创建新实体记录"""
        eid = self._next_id
        self._next_id += 1

        record = EntityRecord(
            entity_id=eid,
            portrait=entity.get("portrait", "unknown"),
            embedding=embedding,
            last_frame_id=frame_id,
            last_timestamp=timestamp,
            first_frame_id=frame_id,
            first_timestamp=timestamp,
        )
        self._pool[eid] = record

        # 如果超过最大容量，清理最不活跃的实体
        if len(self._pool) > self.cfg.entity_pool_max_size:
            self._evict_oldest()

        return eid

    def _get_active_records(self, current_frame_id: int) -> dict[int, EntityRecord]:
        """获取活跃的实体记录（排除过期实体）"""
        active = {}
        for eid, record in self._pool.items():
            age = current_frame_id - record.last_frame_id
            if age <= self.cfg.entity_pool_max_age:
                active[eid] = record
        return active

    def _evict_oldest(self):
        """清理最不活跃的实体"""
        if not self._pool:
            return

        # 按 last_frame_id 排序，移除最旧的
        sorted_ids = sorted(
            self._pool.keys(),
            key=lambda eid: self._pool[eid].last_frame_id,
        )

        # 移除最旧的 20%
        n_remove = max(1, len(sorted_ids) // 5)
        for eid in sorted_ids[:n_remove]:
            del self._pool[eid]

        logger.debug(f"Evicted {n_remove} stale entities from pool.")

    # ── 查询接口 ──────────────────────────────────────
    def get_entity(self, entity_id: int) -> Optional[EntityRecord]:
        """获取指定 ID 的实体记录"""
        return self._pool.get(entity_id)

    def get_all_entities(self) -> dict[int, EntityRecord]:
        """获取所有实体记录"""
        return dict(self._pool)

    @property
    def size(self) -> int:
        return len(self._pool)

    @property
    def next_id(self) -> int:
        return self._next_id

    def reset(self):
        """重置实体池"""
        self._pool.clear()
        self._next_id = 0

    def get_stats(self) -> dict:
        """获取实体池统计信息"""
        if not self._pool:
            return {"total_entities": 0, "active_entities": 0}

        ages = [
            max(r.last_frame_id for r in self._pool.values()) - r.last_frame_id
            for r in self._pool.values()
        ]
        return {
            "total_entities": len(self._pool),
            "total_ids_created": self._next_id,
            "avg_appearances": np.mean(
                [r.appearance_count for r in self._pool.values()]
            ),
            "max_appearances": max(
                r.appearance_count for r in self._pool.values()
            ),
            "avg_age": np.mean(ages),
        }
