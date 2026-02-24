"""
Multi-Frame Stacking — 聚光灯视频管 (Spotlight Tube) 拼图生成器

构建以触发实体为焦点的时空降采样视频流:
  1. 向历史回溯 τ 秒（默认 τ=3.0s）
  2. 将该时间窗内的视频帧降采样至 4 fps
  3. 从降采样帧中选取 4 帧生成 2×2 拼图

  VLLM 能看到时序上的动作变化（"连环画"效果）:
  [T-τ]  [T-2/3τ]
  [T-1/3τ] [T-0]

用法:
    stacker = MultiFrameStacker(lookback_sec=3.0, target_fps=4.0)
    for frame_idx, frame in video:
        stacker.push(frame_idx, frame)
        if should_trigger:
            grid = stacker.make_grid(current_frame=frame)
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional
from PIL import Image


class MultiFrameStacker:
    """
    聚光灯视频管拼图生成器。

    按照文档规范:
      - 回溯 τ 秒的时间窗
      - 降采样至 target_fps（默认 4fps）
      - 输出 2×2 四宫格或帧序列
    """

    def __init__(
        self,
        buffer_interval_frames: int = 8,
        grid_size: tuple = (768, 768),
        max_buffer: int = 30,
        lookback_sec: float = 3.0,
        target_fps: float = 4.0,
    ):
        self.buffer_interval = buffer_interval_frames
        self.grid_size = grid_size
        self.max_buffer = max_buffer
        self.lookback_sec = lookback_sec
        self.target_fps = target_fps

        self._buffer: deque = deque(maxlen=max_buffer)
        self._last_push_idx: int = -999

    def reset(self):
        self._buffer.clear()
        self._last_push_idx = -999

    def push(self, frame_idx: int, frame: np.ndarray):
        """按间隔将帧推入缓冲区。"""
        if frame_idx - self._last_push_idx >= self.buffer_interval:
            self._buffer.append((frame_idx, frame))
            self._last_push_idx = frame_idx

    def can_make_grid(self) -> bool:
        return len(self._buffer) >= 4

    def make_grid(self, current_frame: Optional[np.ndarray] = None) -> Optional[Image.Image]:
        """
        生成 2×2 聚光灯拼图。

        从缓冲区中按均匀间隔选取 4 帧，模拟 τ 秒回溯 + 4fps 降采样。
        """
        frames = list(self._buffer)

        if current_frame is not None:
            frames.append((-1, current_frame))

        if len(frames) < 2:
            return None

        # 均匀采样 4 帧：从可用帧中等距选取
        n = len(frames)
        if n >= 4:
            indices = [
                0,
                n // 3,
                (2 * n) // 3,
                n - 1,
            ]
            selected = [frames[i] for i in indices]
        else:
            selected = list(frames)
            while len(selected) < 4:
                selected.insert(0, selected[0])

        imgs = [s[1] for s in selected]

        cell_w = self.grid_size[0] // 2
        cell_h = self.grid_size[1] // 2

        resized = [
            cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
            for img in imgs
        ]

        top = np.concatenate([resized[0], resized[1]], axis=1)
        bottom = np.concatenate([resized[2], resized[3]], axis=1)
        grid = np.concatenate([top, bottom], axis=0)

        ts_labels = [str(s[0]) if s[0] >= 0 else "now" for s in selected]
        font = cv2.FONT_HERSHEY_SIMPLEX
        positions = [
            (5, cell_h - 8),
            (cell_w + 5, cell_h - 8),
            (5, self.grid_size[1] - 8),
            (cell_w + 5, self.grid_size[1] - 8),
        ]
        for label, pos in zip(ts_labels, positions):
            cv2.putText(grid, f"F#{label}", pos, font, 0.5, (0, 255, 255), 1)

        rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def make_grid_with_timestamps(
        self,
        fps: float,
        current_frame: Optional[np.ndarray] = None,
    ) -> tuple[Optional[Image.Image], list[float]]:
        """
        生成 4 宫格拼图并返回各帧的时间戳。

        Returns:
            (PIL.Image, [t0, t1, t2, t3]) 或 (None, [])
        """
        frames = list(self._buffer)
        if current_frame is not None:
            frames.append((-1, current_frame))

        if len(frames) < 2:
            return None, []

        n = len(frames)
        if n >= 4:
            indices = [0, n // 3, (2 * n) // 3, n - 1]
            selected = [frames[i] for i in indices]
        else:
            selected = list(frames)
            while len(selected) < 4:
                selected.insert(0, selected[0])

        timestamps = [s[0] / max(fps, 1e-6) if s[0] >= 0 else -1.0 for s in selected]

        grid_img = self.make_grid(current_frame)
        return grid_img, timestamps
