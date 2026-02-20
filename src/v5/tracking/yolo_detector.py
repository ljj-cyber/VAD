"""
Stage 1-A': YoloDetector — YOLO-World 开放词汇目标检测

替代/补充帧差法的语义级实体检测:
  - 使用 YOLO-World 进行开放词汇检测
  - 可检测 fire, smoke, person, weapon 等任意文本描述的目标
  - 不依赖运动，静态火焰/烟雾也能检出
  - 输出与 MotionRegion 兼容的 bbox + crop

性能优化:
  - FP16 半精度推理 (CUDA)
  - 批量 GPU→CPU tensor 传输 (避免逐 box 搬运)
  - stream=True 减少内存分配
  - 模型预热消除首帧延迟
  - 智能频率控制 (运动帧也节流)

输出: 每帧的 [(x, y, w, h, crop_image, confidence, class_name), ...]
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import YoloDetectorConfig

logger = logging.getLogger(__name__)

# 全局单例
_yolo_model = None
_yolo_lock = threading.Lock()
_yolo_warmed_up = False


@dataclass
class YoloRegion:
    """YOLO 检测到的一个目标区域"""
    x: int
    y: int
    w: int
    h: int
    crop_image: np.ndarray       # BGR crop
    confidence: float            # 检测置信度 (0-1)
    class_name: str              # 类别名称 (e.g. "fire", "person")
    class_id: int                # 类别 ID
    area: int                    # 像素面积


def _get_yolo_model(cfg: YoloDetectorConfig):
    """懒加载 YOLO-World 模型（线程安全单例），含 FP16 + 预热"""
    global _yolo_model, _yolo_warmed_up

    if _yolo_model is not None:
        return _yolo_model

    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model

        from ultralytics import YOLOWorld

        t0 = time.time()
        logger.info(f"Loading YOLO-World model: {cfg.model_name}")
        _yolo_model = YOLOWorld(cfg.model_name)
        _yolo_model.set_classes(cfg.classes)

        # 注意: 不在此处调用 .model.half()，
        # 因为 ultralytics 的 predict() 内部会先 fuse() 再转半精度，
        # 提前 half() 会导致 fuse_conv_and_bn dtype 不匹配。
        # 只需在 predict() 中传 half=True 即可自动启用 FP16。
        use_half = cfg.half and cfg.device == "cuda"
        if use_half:
            logger.info("YOLO-World will use FP16 via predict(half=True)")

        # 模型预热: 消除首帧 CUDA kernel 编译延迟
        if not _yolo_warmed_up:
            dummy = np.zeros((cfg.imgsz, cfg.imgsz, 3), dtype=np.uint8)
            _yolo_model.predict(
                source=dummy, conf=0.5, imgsz=cfg.imgsz,
                device=cfg.device, verbose=False, half=use_half,
            )
            _yolo_warmed_up = True

        elapsed = time.time() - t0
        logger.info(
            f"YOLO-World loaded + warmed up in {elapsed:.1f}s. "
            f"Classes ({len(cfg.classes)}): {cfg.classes}"
        )
        return _yolo_model


def export_tensorrt(cfg: Optional[YoloDetectorConfig] = None) -> str:
    """
    将 YOLO-World 模型导出为 TensorRT engine (一次性)。

    导出后的 .engine 文件与原 .pt 同目录，后续推理自动使用。
    需要安装 tensorrt: pip install tensorrt

    Returns:
        导出后的 engine 文件路径
    """
    cfg = cfg or YoloDetectorConfig()
    model = _get_yolo_model(cfg)

    logger.info(f"Exporting YOLO-World to TensorRT (imgsz={cfg.imgsz}, half={cfg.half}) ...")
    t0 = time.time()
    engine_path = model.export(
        format="engine",
        imgsz=cfg.imgsz,
        half=cfg.half,
        device=cfg.device,
    )
    logger.info(f"TensorRT export done in {time.time() - t0:.1f}s → {engine_path}")
    return engine_path


class YoloDetector:
    """
    YOLO-World 开放词汇目标检测器 (性能优化版)。

    优化点:
      - FP16 半精度推理 (CUDA 设备)
      - 批量 GPU→CPU tensor 传输
      - stream=True 减少中间对象分配
      - 模型预热消除首帧延迟
      - 智能跳帧: force 模式也服从最小间隔
      - 可选 TensorRT 加速 (需先 export)

    用法:
        detector = YoloDetector()
        for frame in video_frames:
            regions = detector.detect(frame)
            # regions: list[YoloRegion]
    """

    def __init__(self, cfg: Optional[YoloDetectorConfig] = None, lazy_load: bool = False):
        self.cfg = cfg or YoloDetectorConfig()
        self._frame_count: int = 0
        self._last_results: list[YoloRegion] = []
        self._last_detect_frame: int = -999  # 上次实际执行检测的帧号
        self._model_loaded: bool = False
        # lazy_load=True 时延迟到首次 detect() 才加载模型（节省显存）
        if not lazy_load:
            _get_yolo_model(self.cfg)
            self._model_loaded = True

    def detect(
        self,
        frame: np.ndarray,
        force: bool = False,
    ) -> list[YoloRegion]:
        """
        对当前帧进行 YOLO-World 检测。

        Args:
            frame: BGR 格式帧 (H, W, 3)
            force: 强制检测（但仍服从 force_min_gap 最小间隔）

        Returns:
            list[YoloRegion]，按 confidence 降序排列
        """
        self._frame_count += 1

        # ── 智能频率控制 ──
        # 普通模式: 每 N 帧检测一次
        # force 模式: 也需满足最小间隔 (force_min_gap)，防止每帧都调 YOLO
        frames_since_last = self._frame_count - self._last_detect_frame
        if not force:
            if frames_since_last < self.cfg.detect_every_n:
                return self._last_results
        else:
            if frames_since_last < self.cfg.force_min_gap:
                return self._last_results

        model = _get_yolo_model(self.cfg)
        if not self._model_loaded:
            self._model_loaded = True
            logger.info("YOLO model lazy-loaded on first detect()")
        use_half = self.cfg.half and self.cfg.device == "cuda"

        # ── 推理 (stream=True 减少内存开销) ──
        results = model.predict(
            source=frame,
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
            max_det=self.cfg.max_det,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False,
            half=use_half,
            stream=True,  # 流式输出，减少中间对象分配
        )

        regions: list[YoloRegion] = []
        h_frame, w_frame = frame.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # ── 批量 GPU→CPU 传输 (一次性搬运所有 box) ──
            all_xyxy = boxes.xyxy.cpu().numpy().astype(int)    # (N, 4)
            all_conf = boxes.conf.cpu().numpy()                 # (N,)
            all_cls = boxes.cls.cpu().numpy().astype(int)       # (N,)
            n_boxes = len(all_conf)

            for i in range(n_boxes):
                x1, y1, x2, y2 = all_xyxy[i]
                conf = float(all_conf[i])
                cls_id = int(all_cls[i])
                cls_name = (
                    self.cfg.classes[cls_id]
                    if cls_id < len(self.cfg.classes)
                    else "unknown"
                )

                # 边界裁剪
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_frame, x2)
                y2 = min(h_frame, y2)

                w = x2 - x1
                h = y2 - y1
                if w < 10 or h < 10:
                    continue

                crop = frame[y1:y2, x1:x2].copy()
                area = w * h

                regions.append(YoloRegion(
                    x=x1, y=y1, w=w, h=h,
                    crop_image=crop,
                    confidence=conf,
                    class_name=cls_name,
                    class_id=cls_id,
                    area=area,
                ))

        # 按置信度降序
        regions.sort(key=lambda r: r.confidence, reverse=True)

        # 缓存结果 + 记录检测帧号
        self._last_results = regions
        self._last_detect_frame = self._frame_count

        if regions:
            cls_summary = {}
            for r in regions:
                cls_summary[r.class_name] = cls_summary.get(r.class_name, 0) + 1
            logger.debug(
                f"YOLO detected {len(regions)} objects: {cls_summary}"
            )

        return regions

    def reset(self):
        """重置状态（处理新视频时调用）"""
        self._frame_count = 0
        self._last_results = []
        self._last_detect_frame = -999
