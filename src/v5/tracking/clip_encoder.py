"""
Stage 1-B: CLIP Encoder — Crop 级别特征提取

只对 crop 区域编码（而非全图），获取 512/768 维 embedding:
  - 速度快：crop 小于全图，batch 推理
  - 语义准：去掉背景噪音，专注运动实体
  - 用途：EntityTracker 的匹配特征 + NodeTrigger 的 change-point 检测
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import CLIPEncoderConfig, HF_CACHE_DIR

logger = logging.getLogger(__name__)

# 全局单例（避免多线程重复加载）
_clip_model = None
_clip_processor = None
_clip_lock = None


def _get_lock():
    global _clip_lock
    if _clip_lock is None:
        import threading
        _clip_lock = threading.Lock()
    return _clip_lock


class CropCLIPEncoder:
    """
    CLIP Encoder — 仅对 crop 图片提取特征。

    特点:
      - 延迟加载模型（首次调用时加载）
      - 线程安全的全局单例
      - 支持 batch 编码
    """

    def __init__(self, cfg: Optional[CLIPEncoderConfig] = None):
        self.cfg = cfg or CLIPEncoderConfig()
        self._ensure_model()

    def _ensure_model(self):
        """确保模型已加载（线程安全单例）"""
        global _clip_model, _clip_processor
        if _clip_model is not None:
            return

        lock = _get_lock()
        with lock:
            if _clip_model is not None:
                return
            logger.info(f"Loading CLIP model: {self.cfg.model_name}")
            from transformers import CLIPProcessor, CLIPModel

            cache = str(HF_CACHE_DIR)
            _clip_processor = CLIPProcessor.from_pretrained(
                self.cfg.model_name, cache_dir=cache
            )
            _clip_model = CLIPModel.from_pretrained(
                self.cfg.model_name, cache_dir=cache
            ).to(self.cfg.device).eval()
            logger.info("CLIP model loaded.")

    @torch.no_grad()
    def encode_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        对一组 crop 图片批量提取 CLIP 特征。

        Args:
            crops: list of BGR numpy arrays

        Returns:
            np.ndarray, shape (N, feature_dim), L2-normalized
        """
        if not crops:
            return np.zeros((0, self.cfg.feature_dim), dtype=np.float32)

        global _clip_model, _clip_processor

        pil_images = [
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops
        ]

        all_feats = []
        bs = self.cfg.batch_size
        for i in range(0, len(pil_images), bs):
            batch = pil_images[i : i + bs]
            inputs = _clip_processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.cfg.device)
            feats = _clip_model.get_image_features(pixel_values=pixel_values)
            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_single(self, crop: np.ndarray) -> np.ndarray:
        """
        对单个 crop 提取特征。

        Returns:
            np.ndarray, shape (feature_dim,)
        """
        result = self.encode_crops([crop])
        return result[0]
