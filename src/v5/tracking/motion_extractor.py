"""
Stage 1-A: MotionExtractor — 帧差动能图 + Top-K 连通域 Bounding Box

不使用 YOLO，仅凭帧间差分提取画面中"主要动区":
  1. cv2.absdiff → 动能图（支持多帧累积）
  2. 自适应阈值 + 形态学去噪 → 二值掩膜
  3. 连通域分析 → Top-K bbox
  4. Crop 裁剪 + padding

改进:
  - 多帧累积差分: 捕捉慢速运动
  - 自适应阈值回退: 连续无检出时自动降低阈值
  - 面积/尺寸门限下调: 覆盖远景小目标

输出: 每帧的 [(x, y, w, h, crop_image, kinetic_energy), ...]
"""

import cv2
import numpy as np
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..config import MotionConfig

logger = logging.getLogger(__name__)


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
    帧差动能提取器（增强版）。

    增强点:
      1. 多帧累积: 保留最近 N 帧灰度图，用 max-pooled 帧差
         捕捉慢速/渐变运动
      2. 自适应阈值回退: 连续 K 帧检不到区域时，自动降低
         diff_threshold（直到 adaptive_threshold_min），
         检出后逐步恢复
      3. 更灵敏的默认参数: 适配远景监控场景

    用法:
        extractor = MotionExtractor()
        for frame in video_frames:
            regions = extractor.extract(frame)
            # regions: list[MotionRegion]
    """

    def __init__(self, cfg: Optional[MotionConfig] = None):
        self.cfg = cfg or MotionConfig()
        self._prev_gray: Optional[np.ndarray] = None
        self._gray_buffer: deque[np.ndarray] = deque(
            maxlen=max(self.cfg.accumulate_frames, 1)
        )
        self._frame_idx: int = 0
        self._last_global_energy: float = 0.0
        # 自适应阈值状态
        self._current_threshold: int = self.cfg.diff_threshold
        self._empty_streak: int = 0  # 连续无检出帧计数

    def reset(self):
        """重置状态（处理新视频时调用）"""
        self._prev_gray = None
        self._gray_buffer.clear()
        self._frame_idx = 0
        self._last_global_energy = 0.0
        self._current_threshold = self.cfg.diff_threshold
        self._empty_streak = 0

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
            self._gray_buffer.append(gray)
            return []

        # ── 帧差动能图 (多帧累积) ──
        diff = self._compute_diff(gray)

        # 更新缓冲
        self._prev_gray = gray
        self._gray_buffer.append(gray)

        # 全帧动能 (归一化)
        global_energy = float(diff.mean()) / 255.0
        self._last_global_energy = global_energy

        # ── 提取连通域 (带自适应阈值回退) ──
        regions = self._extract_regions(frame, diff, self._current_threshold)

        # 如果没检到区域，尝试低阈值回退
        if not regions and self._current_threshold > self.cfg.adaptive_threshold_min:
            fallback_threshold = max(
                self._current_threshold // 2,
                self.cfg.adaptive_threshold_min,
            )
            regions = self._extract_regions(frame, diff, fallback_threshold)
            if regions:
                logger.debug(
                    f"Frame {self._frame_idx}: fallback threshold "
                    f"{self._current_threshold} → {fallback_threshold}, "
                    f"found {len(regions)} regions"
                )

        # ── 更新自适应阈值 ──
        if regions:
            self._empty_streak = 0
            # 检出后缓慢恢复阈值（每次 +1，不超过原始值）
            if self._current_threshold < self.cfg.diff_threshold:
                self._current_threshold = min(
                    self._current_threshold + 1,
                    self.cfg.diff_threshold,
                )
        else:
            self._empty_streak += 1
            # 连续无检出 → 逐步降低阈值
            if self._empty_streak >= self.cfg.empty_streak_for_fallback:
                new_thresh = max(
                    self._current_threshold - 1,
                    self.cfg.adaptive_threshold_min,
                )
                if new_thresh != self._current_threshold:
                    self._current_threshold = new_thresh
                    logger.debug(
                        f"Adaptive threshold lowered to {self._current_threshold} "
                        f"(empty streak = {self._empty_streak})"
                    )

        return regions

    def _compute_diff(self, gray: np.ndarray) -> np.ndarray:
        """
        计算帧差动能图。

        如果 accumulate_frames > 1，使用 max-pooled 多帧差分，
        否则退化为普通两帧差分。
        """
        # 基础两帧差分
        diff = cv2.absdiff(gray, self._prev_gray)

        # 多帧累积: 与 buffer 中所有帧取最大差分
        if len(self._gray_buffer) > 1:
            for old_gray in self._gray_buffer:
                old_diff = cv2.absdiff(gray, old_gray)
                diff = np.maximum(diff, old_diff)

        return diff

    def _extract_regions(
        self,
        frame: np.ndarray,
        diff: np.ndarray,
        threshold: int,
    ) -> list[MotionRegion]:
        """
        从差分图提取连通域。

        Args:
            frame: 原始 BGR 帧
            diff: 帧差动能图
            threshold: 二值化阈值

        Returns:
            list[MotionRegion]，按 kinetic_energy 降序，Top-K
        """
        # 阈值化
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # 形态学闭运算去噪 + 连接碎片
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size),
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        # 额外膨胀一次，帮助合并邻近碎片
        thresh = cv2.dilate(thresh, kernel, iterations=1)

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

    def compute_frame_energy(self, frame: np.ndarray = None) -> float:
        """
        返回上一次 extract() 计算的全帧动能 (0-1)。

        注意：必须在 extract() 之后调用，返回的是 extract() 时已计算好的值。
        不再重复计算（避免 _prev_gray 已更新导致 diff=0 的 bug）。
        """
        return self._last_global_energy
