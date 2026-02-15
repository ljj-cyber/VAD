"""
Multi-Frame Stacking — 4 宫格拼图生成器

解决单帧无法识别的异常（如推搡、连续虐待）：
  当 Entity 处于活跃状态时，生成 2×2 拼图：
  [T-6] [T-4]
  [T-2] [T-0]

  VLLM 能看到时序上的动作变化（"连环画"效果）。

用法:
    stacker = MultiFrameStacker(buffer_interval=2, buffer_count=4)
    for frame_idx, frame in video:
        stacker.push(frame_idx, timestamp, frame)
        if should_trigger:
            grid = stacker.make_grid(frame_idx)
            # grid: PIL.Image 2×2 拼图
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional
from PIL import Image


class MultiFrameStacker:
    """
    维护帧缓冲区，按需生成 2×2 拼图。

    缓冲区存最近 N 帧（按 interval 采样），
    生成拼图时取 T, T-interval, T-2*interval, T-3*interval。
    """

    def __init__(
        self,
        buffer_interval_frames: int = 6,  # 每隔 6 帧采一帧进 buffer
        grid_size: tuple = (768, 768),     # 输出拼图总尺寸
        max_buffer: int = 20,              # 最大缓冲帧数
    ):
        self.buffer_interval = buffer_interval_frames
        self.grid_size = grid_size
        self.max_buffer = max_buffer

        # frame_idx → BGR frame
        self._buffer: deque = deque(maxlen=max_buffer)
        self._last_push_idx: int = -999

    def reset(self):
        self._buffer.clear()
        self._last_push_idx = -999

    def push(self, frame_idx: int, frame: np.ndarray):
        """
        按间隔将帧推入缓冲区。

        Args:
            frame_idx: 帧号
            frame: BGR 帧
        """
        if frame_idx - self._last_push_idx >= self.buffer_interval:
            self._buffer.append((frame_idx, frame))
            self._last_push_idx = frame_idx

    def can_make_grid(self) -> bool:
        """是否有足够帧生成 4 宫格"""
        return len(self._buffer) >= 4

    def make_grid(self, current_frame: Optional[np.ndarray] = None) -> Optional[Image.Image]:
        """
        生成 2×2 拼图。

        取缓冲区最新的 4 帧，排列为:
          [最旧] [次旧]
          [次新] [最新]

        如果传入 current_frame 且缓冲区不足 4 帧，用 current_frame 补齐。

        Returns:
            PIL.Image (RGB) 或 None
        """
        frames = list(self._buffer)

        if current_frame is not None:
            # 确保当前帧也在
            frames.append((-1, current_frame))

        if len(frames) < 2:
            return None

        # 取最新的 4 帧（或不足 4 帧时用最新帧重复）
        selected = frames[-4:] if len(frames) >= 4 else frames
        while len(selected) < 4:
            selected.insert(0, selected[0])  # 用最旧帧重复

        # 提取 BGR frames
        imgs = [s[1] for s in selected]

        # 每个子图的尺寸
        cell_w = self.grid_size[0] // 2
        cell_h = self.grid_size[1] // 2

        # Resize 每帧到 cell 大小
        resized = []
        for img in imgs:
            r = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
            resized.append(r)

        # 拼成 2×2
        # [0] [1]
        # [2] [3]
        top = np.concatenate([resized[0], resized[1]], axis=1)
        bottom = np.concatenate([resized[2], resized[3]], axis=1)
        grid = np.concatenate([top, bottom], axis=0)

        # 添加时间标注
        ts_labels = []
        for s in selected:
            fidx = s[0]
            ts_labels.append(str(fidx) if fidx >= 0 else "now")

        font = cv2.FONT_HERSHEY_SIMPLEX
        positions = [
            (5, cell_h - 8),
            (cell_w + 5, cell_h - 8),
            (5, self.grid_size[1] - 8),
            (cell_w + 5, self.grid_size[1] - 8),
        ]
        for label, pos in zip(ts_labels, positions):
            cv2.putText(grid, f"F#{label}", pos, font, 0.5, (0, 255, 255), 1)

        # BGR → RGB → PIL
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

        selected = frames[-4:] if len(frames) >= 4 else frames
        while len(selected) < 4:
            selected.insert(0, selected[0])

        timestamps = [s[0] / max(fps, 1e-6) if s[0] >= 0 else -1.0 for s in selected]

        grid_img = self.make_grid(current_frame)
        return grid_img, timestamps
