"""
Phase 1-A: VisualPainter — 带高亮框的全局视图

核心思路:
  不再单纯裁剪运动区域，而是在原图上给动能主体画高亮框，
  发给 VLLM 的是这幅"带框的原图"。

功能:
  1. 在原图上用红色/黄色粗框标记 MotionRegion
  2. 在框旁标注 entity_id 和动能等级
  3. Resize 到适合 VLLM 输入的尺寸（保留全图上下文）
  4. 同时输出 crop 用于 CLIP 特征提取（不变）
"""

import cv2
import numpy as np
from typing import Optional
from PIL import Image

from .motion_extractor import MotionRegion


# 颜色方案: 动能越高颜色越暖
_COLORS = {
    "low": (0, 200, 255),      # 黄色 (BGR)
    "medium": (0, 140, 255),   # 橙色
    "high": (0, 0, 255),       # 红色
}


def _energy_color(energy: float) -> tuple:
    """根据动能选颜色"""
    if energy < 0.03:
        return _COLORS["low"]
    elif energy < 0.1:
        return _COLORS["medium"]
    else:
        return _COLORS["high"]


def _energy_label(energy: float) -> str:
    if energy < 0.01:
        return "still"
    elif energy < 0.03:
        return "low"
    elif energy < 0.08:
        return "medium"
    elif energy < 0.2:
        return "high"
    else:
        return "EXTREME"


class VisualPainter:
    """
    在原图上绘制高亮框 + 标注信息，生成"带框全图"。

    用法:
        painter = VisualPainter()
        annotated = painter.paint(frame, regions, entity_ids)
        # annotated: PIL.Image — 发给 VLLM
    """

    def __init__(
        self,
        output_size: tuple = (768, 768),
        box_thickness: int = 3,
        font_scale: float = 0.6,
        spotlight_dim_factor: float = 0.7,
    ):
        self.output_size = output_size
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.spotlight_dim_factor = spotlight_dim_factor

    def paint(
        self,
        frame: np.ndarray,
        regions: list,
        entity_ids: Optional[list[int]] = None,
    ) -> Image.Image:
        """
        在原图上绘制高亮框 + 聚光灯效果（背景亮度下调 30%），返回 PIL Image。

        聚光灯效果：
          1. 将全图亮度降低 30%（形成暗背景）
          2. 在 bbox 区域恢复原始亮度（实体"高亮"）
          3. 绘制红/黄/橙色边界框 + 标注

        Args:
            frame: BGR 原图 (H, W, 3)
            regions: 当前帧的 MotionRegion 或 HybridRegion 列表
            entity_ids: 对应的 entity_id 列表（可选）

        Returns:
            PIL.Image: 带聚光灯效果的标注图（已 resize）
        """
        # 聚光灯：先将全图亮度降低 30%
        dimmed = (frame.astype(np.float32) * self.spotlight_dim_factor).astype(np.uint8)
        canvas = dimmed.copy()

        # 在每个 bbox 区域恢复原始亮度（高亮实体）
        for region in regions:
            x1, y1 = region.x, region.y
            x2, y2 = region.x + region.w, region.y + region.h
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            canvas[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        for i, region in enumerate(regions):
            eid = entity_ids[i] if entity_ids and i < len(entity_ids) else i
            color = _energy_color(region.kinetic_energy)
            cls_tag = ""
            if hasattr(region, "class_name") and region.class_name:
                cls_tag = f" {region.class_name}"
            label = f"E#{eid}{cls_tag} [{_energy_label(region.kinetic_energy)}]"

            x1, y1 = region.x, region.y
            x2, y2 = region.x + region.w, region.y + region.h
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.box_thickness)

            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
            )[0]
            text_y = max(y1 - 8, text_size[1] + 4)
            cv2.rectangle(
                canvas,
                (x1, text_y - text_size[1] - 4),
                (x1 + text_size[0] + 4, text_y + 4),
                color, -1,
            )
            cv2.putText(
                canvas, label,
                (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale, (255, 255, 255), 2,
            )

        resized = cv2.resize(canvas, self.output_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def paint_fullframe_only(
        self,
        frame: np.ndarray,
    ) -> Image.Image:
        """
        不带框的全图（用于全局心跳采样）。
        """
        resized = cv2.resize(frame, self.output_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
