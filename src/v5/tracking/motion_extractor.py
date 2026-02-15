"""
Stage 1-A: MotionExtractor — 帧差动能图 + Top-K 连通域 Bounding Box

不使用 YOLO，仅凭帧间差分提取画面中"主要动区":
  1. cv2.absdiff → 动能图
  2. 阈值 + 形态学去噪 → 二值掩膜
  3. 连通域分析 → Top-K bbox
  4. Crop 裁剪 + padding

输出: 每帧的 [(x, y, w, h, crop_image, kinetic_energy), ...]
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..config import MotionConfig


@dataclass
class MotionRegion:
    """一个动能连通域"""
    x: int
    y: int
    w: int
    h: int
    crop_image: np.ndarray       # BGR crop
    kinetic_energy: float        # 该区域的平均动能 (0-1)
    area: int                    # 像素面积


class MotionExtractor:
    """
    帧差动能提取器。

    用法:
        extractor = MotionExtractor()
        for frame in video_frames:
            regions = extractor.extract(frame)
            # regions: list[MotionRegion]
    """

    def __init__(self, cfg: Optional[MotionConfig] = None):
        self.cfg = cfg or MotionConfig()
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_idx: int = 0

    def reset(self):
        """重置状态（处理新视频时调用）"""
        self._prev_gray = None
        self._frame_idx = 0

    def extract(self, frame: np.ndarray) -> list[MotionRegion]:
        """
        从当前帧提取 Top-K 动能连通域。

        Args:
            frame: BGR 格式帧 (H, W, 3)

        Returns:
            list[MotionRegion]，按 kinetic_energy 降序排列
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.cfg.blur_kernel_size > 0:
            k = self.cfg.blur_kernel_size
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        self._frame_idx += 1

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        # ── 帧差动能图 ──
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray

        # 全帧动能 (归一化)
        global_energy = float(diff.mean()) / 255.0

        # 阈值化
        _, thresh = cv2.threshold(diff, self.cfg.diff_threshold, 255, cv2.THRESH_BINARY)

        # 形态学闭运算去噪 + 连接碎片
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size),
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # ── 连通域分析 ──
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )

        h_frame, w_frame = frame.shape[:2]
        regions: list[MotionRegion] = []

        for i in range(1, num_labels):  # 跳过背景 label=0
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.cfg.min_region_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # 该区域的局部动能
            mask_region = (labels[y:y+h, x:x+w] == i)
            local_energy = float(diff[y:y+h, x:x+w][mask_region].mean()) / 255.0

            # Padding 外扩
            pad_x = int(w * self.cfg.crop_padding_ratio)
            pad_y = int(h * self.cfg.crop_padding_ratio)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_frame, x + w + pad_x)
            y2 = min(h_frame, y + h + pad_y)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < self.cfg.min_crop_size or crop_h < self.cfg.min_crop_size:
                continue

            crop = frame[y1:y2, x1:x2].copy()

            regions.append(MotionRegion(
                x=x1, y=y1, w=crop_w, h=crop_h,
                crop_image=crop,
                kinetic_energy=local_energy,
                area=area,
            ))

        # 按动能降序排列，取 Top-K
        regions.sort(key=lambda r: r.kinetic_energy, reverse=True)
        regions = regions[: self.cfg.top_k_regions]

        return regions

    def compute_frame_energy(self, frame: np.ndarray) -> float:
        """计算当前帧与上一帧的全局动能 (0-1)，不修改内部状态"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.cfg.blur_kernel_size > 0:
            k = self.cfg.blur_kernel_size
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        if self._prev_gray is None:
            return 0.0
        diff = cv2.absdiff(gray, self._prev_gray)
        return float(diff.mean()) / 255.0
