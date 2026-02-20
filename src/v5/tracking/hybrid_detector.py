"""
Stage 1-A'': HybridDetector — 帧差 + YOLO-World 融合检测

融合两种互补的检测方式:
  - 帧差法 (MotionExtractor): 捕捉运动变化，对快速刚体运动敏感
  - YOLO-World (YoloDetector): 语义级检测，对火焰/烟雾/武器等静态或渐变目标敏感

融合策略:
  1. 两个检测器各自独立运行
  2. 用 IoU 匹配重叠区域 → 合并为一个（取 YOLO 的语义信息 + 帧差的动能信息）
  3. YOLO 独有的检测（帧差未检出）→ 保留（如静态火焰）
  4. 帧差独有的检测（YOLO 未检出）→ 保留（如快速运动的未知物体）

输出: MotionRegion 列表（与原有接口兼容），附加 class_name 属性
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import MotionConfig, YoloDetectorConfig, HybridDetectorConfig
from .motion_extractor import MotionExtractor, MotionRegion
from .yolo_detector import YoloDetector, YoloRegion

logger = logging.getLogger(__name__)


@dataclass
class HybridRegion:
    """融合后的检测区域 — 兼容 MotionRegion 接口"""
    x: int
    y: int
    w: int
    h: int
    crop_image: np.ndarray       # BGR crop
    kinetic_energy: float        # 动能 (来自帧差; YOLO-only 区域为 0)
    area: int                    # 像素面积
    # 额外语义信息
    class_name: str = ""         # YOLO 类别 (e.g. "fire", "person")
    confidence: float = 0.0      # YOLO 置信度
    source: str = "motion"       # "motion" | "yolo" | "fused"


def _compute_iou(box1: tuple, box2: tuple) -> float:
    """
    计算两个 (x, y, w, h) 格式框的 IoU。
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 转为 xyxy
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    # 交集
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)


class HybridDetector:
    """
    帧差 + YOLO-World 融合检测器（支持 Lazy YOLO Fallback）。

    两种运行模式:
      1. lazy_yolo=True (默认):
         帧差法为主，仅当连续 empty_streak_for_yolo_fallback 帧无检出时
         自动激活 YOLO 进行融合检测。YOLO 激活后持续运行 cooldown 帧，
         避免频繁开关。帧差法全程运行，YOLO 结果始终与帧差融合。

      2. lazy_yolo=False:
         帧差 + YOLO 全程并行（精度优先），等价于原来的 use_yolo 模式。

    用法:
        detector = HybridDetector()
        for frame in video_frames:
            regions = detector.detect(frame)
            # regions: list[HybridRegion]  (兼容 MotionRegion 接口)
    """

    def __init__(
        self,
        motion_cfg: Optional[MotionConfig] = None,
        yolo_cfg: Optional[YoloDetectorConfig] = None,
        hybrid_cfg: Optional[HybridDetectorConfig] = None,
    ):
        self.motion_cfg = motion_cfg or MotionConfig()
        self.yolo_cfg = yolo_cfg or YoloDetectorConfig()
        self.hybrid_cfg = hybrid_cfg or HybridDetectorConfig()

        self.motion_extractor = MotionExtractor(self.motion_cfg)

        # lazy_yolo 模式: YOLO 延迟加载，首次 fallback 触发时才初始化
        lazy_load = self.hybrid_cfg.lazy_yolo
        self.yolo_detector = YoloDetector(self.yolo_cfg, lazy_load=lazy_load)

        self._last_global_energy: float = 0.0

        # ── Lazy YOLO Fallback 状态 ──
        self._empty_streak: int = 0            # 连续帧差空检出计数
        self._yolo_active: bool = not lazy_load  # YOLO 当前是否激活
        self._yolo_cooldown_remaining: int = 0   # YOLO 冷却剩余帧数
        self._yolo_fallback_count: int = 0       # YOLO fallback 触发次数 (统计)
        self._total_yolo_frames: int = 0         # YOLO 实际运行帧数 (统计)

    def reset(self):
        """重置所有状态"""
        self.motion_extractor.reset()
        self.yolo_detector.reset()
        self._last_global_energy = 0.0
        self._empty_streak = 0
        if self.hybrid_cfg.lazy_yolo:
            self._yolo_active = False
        self._yolo_cooldown_remaining = 0

    def detect(self, frame: np.ndarray) -> list[HybridRegion]:
        """
        融合检测：帧差 + YOLO-World（按需激活）。

        Args:
            frame: BGR 格式帧 (H, W, 3)

        Returns:
            list[HybridRegion]，按综合评分降序排列
        """
        # ── 帧差检测 (始终运行) ──
        motion_regions = self.motion_extractor.extract(frame)
        self._last_global_energy = self.motion_extractor.compute_frame_energy()

        # ── 更新空检出计数 ──
        if motion_regions:
            self._empty_streak = 0
        else:
            self._empty_streak += 1

        # ── YOLO 激活逻辑 ──
        yolo_regions = []
        should_run_yolo = False

        if self.hybrid_cfg.lazy_yolo:
            # Lazy 模式: 按需激活 YOLO
            if self._yolo_active:
                # YOLO 已激活 — 持续运行直到冷却结束
                should_run_yolo = True
                self._yolo_cooldown_remaining -= 1
                if self._yolo_cooldown_remaining <= 0 and motion_regions:
                    # 冷却结束 且 帧差已恢复检出 → 退出 YOLO
                    self._yolo_active = False
                    logger.info(
                        f"YOLO fallback deactivated (motion restored, "
                        f"ran {self._yolo_cooldown_remaining + self.hybrid_cfg.yolo_cooldown_frames} frames)"
                    )
            else:
                # YOLO 未激活 — 检查是否需要激活
                if self._empty_streak >= self.hybrid_cfg.empty_streak_for_yolo_fallback:
                    self._yolo_active = True
                    self._yolo_cooldown_remaining = self.hybrid_cfg.yolo_cooldown_frames
                    self._yolo_fallback_count += 1
                    should_run_yolo = True
                    logger.info(
                        f"YOLO fallback ACTIVATED (empty streak = {self._empty_streak}, "
                        f"fallback #{self._yolo_fallback_count})"
                    )
        else:
            # 全程模式: 始终运行 YOLO
            should_run_yolo = True

        if should_run_yolo:
            self._total_yolo_frames += 1
            # 帧差有检出时 force YOLO 以融合; 帧差无检出时也 force 以 fallback
            has_significant_motion = (
                len(motion_regions) > 0
                and self._last_global_energy >= self.hybrid_cfg.motion_only_min_energy
            )
            force_yolo = (
                (has_significant_motion and self.yolo_cfg.force_on_motion)
                or (self._yolo_active and self.hybrid_cfg.lazy_yolo)  # fallback 期间强制
            )
            yolo_regions = self.yolo_detector.detect(frame, force=force_yolo)

        # ── 融合 (帧差 + YOLO, 始终走融合路径) ──
        fused = self._fuse(motion_regions, yolo_regions, frame)

        return fused

    def compute_frame_energy(self) -> float:
        """返回上一次 detect() 的全帧动能"""
        return self._last_global_energy

    @property
    def is_yolo_active(self) -> bool:
        """YOLO 当前是否处于激活状态"""
        return self._yolo_active

    @property
    def yolo_stats(self) -> dict:
        """YOLO fallback 统计信息"""
        return {
            "fallback_count": self._yolo_fallback_count,
            "total_yolo_frames": self._total_yolo_frames,
            "currently_active": self._yolo_active,
            "empty_streak": self._empty_streak,
        }

    def _fuse(
        self,
        motion_regions: list[MotionRegion],
        yolo_regions: list[YoloRegion],
        frame: np.ndarray,
    ) -> list[HybridRegion]:
        """
        融合帧差和 YOLO 检测结果。

        策略:
          1. 对每对 (motion, yolo) 计算 IoU
          2. IoU >= threshold → 合并（fused）
          3. 未匹配的 YOLO → 作为 yolo-only 保留
          4. 未匹配的 Motion → 作为 motion-only 保留
        """
        results: list[HybridRegion] = []

        matched_motion: set[int] = set()
        matched_yolo: set[int] = set()

        # ── 匹配 ──
        if motion_regions and yolo_regions:
            # 计算 IoU 矩阵
            n_m = len(motion_regions)
            n_y = len(yolo_regions)
            iou_matrix = np.zeros((n_m, n_y), dtype=np.float32)

            for mi, mr in enumerate(motion_regions):
                for yi, yr in enumerate(yolo_regions):
                    iou_matrix[mi, yi] = _compute_iou(
                        (mr.x, mr.y, mr.w, mr.h),
                        (yr.x, yr.y, yr.w, yr.h),
                    )

            # 贪婪匹配
            for _ in range(min(n_m, n_y)):
                # 屏蔽已匹配
                mask = iou_matrix.copy()
                for mi in matched_motion:
                    mask[mi, :] = -1.0
                for yi in matched_yolo:
                    mask[:, yi] = -1.0

                best_idx = np.unravel_index(mask.argmax(), mask.shape)
                mi, yi = int(best_idx[0]), int(best_idx[1])
                best_iou = float(mask[mi, yi])

                if best_iou < self.hybrid_cfg.merge_iou_threshold:
                    break

                # 合并: 取 YOLO 的框（更精确）+ 帧差的动能
                mr = motion_regions[mi]
                yr = yolo_regions[yi]

                results.append(HybridRegion(
                    x=yr.x, y=yr.y, w=yr.w, h=yr.h,
                    crop_image=yr.crop_image,
                    kinetic_energy=mr.kinetic_energy,
                    area=yr.area,
                    class_name=yr.class_name,
                    confidence=yr.confidence,
                    source="fused",
                ))

                matched_motion.add(mi)
                matched_yolo.add(yi)

        # ── YOLO-only（帧差未检出）──
        for yi, yr in enumerate(yolo_regions):
            if yi in matched_yolo:
                continue
            if yr.confidence < self.hybrid_cfg.yolo_only_min_conf:
                continue

            results.append(HybridRegion(
                x=yr.x, y=yr.y, w=yr.w, h=yr.h,
                crop_image=yr.crop_image,
                kinetic_energy=0.0,  # 帧差未检出，动能为 0
                area=yr.area,
                class_name=yr.class_name,
                confidence=yr.confidence,
                source="yolo",
            ))

        # ── Motion-only（YOLO 未检出）──
        for mi, mr in enumerate(motion_regions):
            if mi in matched_motion:
                continue
            if mr.kinetic_energy < self.hybrid_cfg.motion_only_min_energy:
                continue

            results.append(HybridRegion(
                x=mr.x, y=mr.y, w=mr.w, h=mr.h,
                crop_image=mr.crop_image,
                kinetic_energy=mr.kinetic_energy,
                area=mr.area,
                class_name="",  # YOLO 未识别
                confidence=0.0,
                source="motion",
            ))

        # ── 排序: fused > yolo > motion, 内部按 confidence/energy 排序 ──
        def _sort_key(r: HybridRegion) -> float:
            source_bonus = {"fused": 1.0, "yolo": 0.5, "motion": 0.0}
            return source_bonus.get(r.source, 0.0) + r.confidence + r.kinetic_energy

        results.sort(key=_sort_key, reverse=True)

        if results:
            summary = {"fused": 0, "yolo": 0, "motion": 0}
            for r in results:
                summary[r.source] = summary.get(r.source, 0) + 1
            classes = [r.class_name for r in results if r.class_name]
            logger.debug(
                f"Hybrid: {len(results)} regions "
                f"(fused={summary['fused']}, yolo={summary['yolo']}, "
                f"motion={summary['motion']}) classes={classes}"
            )

        return results
