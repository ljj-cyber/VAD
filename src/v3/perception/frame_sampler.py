"""
V3.0 语义感知层 — 自适应帧采样器

核心思想:
  - 画面静止时以低频 (0.5 FPS) 采样，节省 VLLM 调用。
  - 画面出现剧烈运动（打斗、奔跑等）时自动提升采样率 (最高 3 FPS)。
  - 运动能量 (motion_energy) 来自两个来源:
    1. 帧间像素差分（实时、不依赖 VLLM）
    2. VLLM 返回的 motion_energy 字段（语义级，作为校准信号）
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ..config import SamplerConfig

logger = logging.getLogger(__name__)


class AdaptiveFrameSampler:
    """
    自适应采样器：根据画面运动能量动态调整采样帧。
    """

    def __init__(self, video_fps: float, total_frames: int, cfg: Optional[SamplerConfig] = None):
        """
        Args:
            video_fps: 视频原始帧率
            total_frames: 视频总帧数
            cfg: 采样器配置
        """
        self.video_fps = video_fps
        self.total_frames = total_frames
        self.cfg = cfg or SamplerConfig()

        # 当前采样状态
        self._current_fps = self.cfg.base_fps
        self._energy_history: list[float] = []
        self._prev_gray: Optional[np.ndarray] = None

        # 预计算帧索引缓冲
        self._sampled_indices: list[int] = []
        self._cursor = 0

        # 初始化：先按 base_fps 生成初始采样点
        self._compute_initial_schedule()

    def _compute_initial_schedule(self):
        """按 base_fps 生成初始均匀采样时间表"""
        interval = max(1, int(self.video_fps / self.cfg.base_fps))
        self._sampled_indices = list(range(0, self.total_frames, interval))
        logger.info(
            f"Initial schedule: {len(self._sampled_indices)} frames "
            f"(interval={interval}, base_fps={self.cfg.base_fps})"
        )

    # ── 运动能量计算 ──────────────────────────────────
    def compute_pixel_energy(self, frame_bgr: np.ndarray) -> float:
        """
        基于帧间差分计算像素级运动能量。
        返回 [0, 1] 之间的归一化值。
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))  # 降采样加速

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        diff = cv2.absdiff(gray, self._prev_gray)
        energy = float(np.mean(diff)) / 255.0  # 归一化到 [0, 1]
        self._prev_gray = gray

        return energy

    def update_energy(self, pixel_energy: float, vllm_energy: Optional[float] = None):
        """
        更新运动能量历史，并据此调整采样率。

        Args:
            pixel_energy: 像素级运动能量 [0, 1]
            vllm_energy: VLLM 返回的语义运动能量 [0, 1]（可选）
        """
        # 融合两个来源的能量
        if vllm_energy is not None and vllm_energy >= 0:
            # 像素能量权重 0.4, VLLM 语义能量权重 0.6
            energy = 0.4 * pixel_energy + 0.6 * vllm_energy
        else:
            energy = pixel_energy

        self._energy_history.append(energy)

        # 只保留最近的窗口
        if len(self._energy_history) > self.cfg.energy_window:
            self._energy_history = self._energy_history[-self.cfg.energy_window:]

        # 计算平均能量
        avg_energy = np.mean(self._energy_history)

        # 根据能量动态调整采样 FPS
        if avg_energy >= self.cfg.energy_threshold_high:
            self._current_fps = self.cfg.max_fps
        elif avg_energy <= self.cfg.energy_threshold_low:
            self._current_fps = self.cfg.base_fps
        else:
            # 线性插值
            ratio = (avg_energy - self.cfg.energy_threshold_low) / (
                self.cfg.energy_threshold_high - self.cfg.energy_threshold_low
            )
            self._current_fps = (
                self.cfg.base_fps + ratio * (self.cfg.max_fps - self.cfg.base_fps)
            )

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @property
    def current_interval(self) -> int:
        """当前采样间隔（帧数）"""
        interval = max(
            self.cfg.min_interval_frames,
            int(self.video_fps / self._current_fps),
        )
        return interval

    # ── 采样调度 ──────────────────────────────────────
    def get_initial_indices(self) -> list[int]:
        """获取初始均匀采样的帧索引列表"""
        return list(self._sampled_indices)

    def should_sample_next(self, current_frame_idx: int, last_sampled_idx: int) -> bool:
        """
        判断是否应该采样当前帧。

        Args:
            current_frame_idx: 当前帧索引
            last_sampled_idx: 上一次采样的帧索引

        Returns:
            True 表示应采样此帧
        """
        gap = current_frame_idx - last_sampled_idx
        return gap >= self.current_interval

    def compute_adaptive_schedule(
        self,
        frames_bgr: list[np.ndarray],
    ) -> list[int]:
        """
        两遍自适应采样策略:
        第一遍: 快速扫描所有帧，计算像素运动能量
        第二遍: 根据能量曲线确定最终采样帧

        Args:
            frames_bgr: 所有帧 (BGR, numpy array)

        Returns:
            需要送入 VLLM 的帧索引列表
        """
        n = len(frames_bgr)
        if n == 0:
            return []

        # 第一遍：快速计算所有帧间差分能量
        logger.info(f"Computing motion energy for {n} frames...")
        energies = np.zeros(n, dtype=np.float32)
        prev_gray = None

        # 每隔几帧采样计算能量（加速）
        energy_sample_step = max(1, int(self.video_fps / 10))  # ~10 FPS 计算能量
        for i in range(0, n, energy_sample_step):
            gray = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                e = float(np.mean(diff)) / 255.0
                # 将能量填充到采样点之间
                start = max(0, i - energy_sample_step)
                energies[start:i + 1] = e

            prev_gray = gray

        # 第二遍：根据能量确定采样点
        sampled = [0]  # 总是采样第一帧
        last_idx = 0

        for i in range(1, n):
            # 获取当前区域的平均能量
            window_start = max(0, i - int(self.cfg.energy_window * self.video_fps))
            avg_energy = float(np.mean(energies[window_start:i + 1]))

            # 计算动态间隔
            if avg_energy >= self.cfg.energy_threshold_high:
                target_fps = self.cfg.max_fps
            elif avg_energy <= self.cfg.energy_threshold_low:
                target_fps = self.cfg.base_fps
            else:
                ratio = (avg_energy - self.cfg.energy_threshold_low) / (
                    self.cfg.energy_threshold_high - self.cfg.energy_threshold_low
                )
                target_fps = self.cfg.base_fps + ratio * (self.cfg.max_fps - self.cfg.base_fps)

            interval = max(self.cfg.min_interval_frames, int(self.video_fps / target_fps))

            if (i - last_idx) >= interval:
                sampled.append(i)
                last_idx = i

        # 总是包含最后一帧
        if sampled[-1] != n - 1:
            sampled.append(n - 1)

        logger.info(
            f"Adaptive sampling: {len(sampled)}/{n} frames selected "
            f"(compression ratio: {len(sampled)/n:.1%})"
        )
        return sampled

    def reset(self):
        """重置采样器状态"""
        self._energy_history.clear()
        self._prev_gray = None
        self._current_fps = self.cfg.base_fps
        self._cursor = 0
