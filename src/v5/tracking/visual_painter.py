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
    ):
        self.output_size = output_size
        self.box_thickness = box_thickness
        self.font_scale = font_scale

    def paint(
        self,
        frame: np.ndarray,
        regions: list,
        entity_ids: Optional[list[int]] = None,
    ) -> Image.Image:
        """
        在原图上绘制高亮框，返回 PIL Image。

        Args:
            frame: BGR 原图 (H, W, 3)
            regions: 当前帧的 MotionRegion 或 HybridRegion 列表
            entity_ids: 对应的 entity_id 列表（可选）

        Returns:
            PIL.Image: 带框的原图（已 resize）
        """
        canvas = frame.copy()

        for i, region in enumerate(regions):
            eid = entity_ids[i] if entity_ids and i < len(entity_ids) else i
            color = _energy_color(region.kinetic_energy)
            # 如果有 YOLO 类别信息，附加到标签
            cls_tag = ""
            if hasattr(region, "class_name") and region.class_name:
                cls_tag = f" {region.class_name}"
            label = f"E#{eid}{cls_tag} [{_energy_label(region.kinetic_energy)}]"

            # 画框
            x1, y1 = region.x, region.y
            x2, y2 = region.x + region.w, region.y + region.h
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.box_thickness)

            # 标注文字 (框上方)
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
            )[0]
            text_y = max(y1 - 8, text_size[1] + 4)
            # 文字背景
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

        # Resize 到 VLLM 输入尺寸
        resized = cv2.resize(canvas, self.output_size, interpolation=cv2.INTER_LINEAR)
        # BGR → RGB → PIL
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
