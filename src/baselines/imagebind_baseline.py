"""
ImageBind Zero-Shot Video Anomaly Detection Baseline

方法:
  利用 ImageBind 的统一多模态嵌入空间，对视频逐帧提取视觉特征，
  与异常/正常文本 prompt 的特征计算余弦相似度，
  softmax 后得到帧级异常概率。

核心原理:
  score(frame) = softmax( [sim(frame, abnormal_prompts), sim(frame, normal_prompts)] )[0]

与 CLIP baseline 的区别:
  ImageBind 将 6 种模态（图像、文本、音频、深度、热成像、IMU）对齐到同一空间，
  其视觉编码器基于 ViT-H/14（~632M 参数），嵌入维度 1024，特征表示能力更强。

用法:
  python -m baselines.imagebind_baseline --video /path/to/video.mp4
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── 异常/正常文本 Prompt 模板 ──────────────────────────
ANOMALY_PROMPTS = [
    "a violent fight between people",
    "a person being attacked or assaulted",
    "an explosion or fire in the scene",
    "a robbery or theft happening",
    "a shooting or gunfire incident",
    "a person vandalizing property",
    "a car accident or crash",
    "an arson or intentionally set fire",
    "a person shoplifting in a store",
    "a burglary or breaking into a building",
    "a person running away after committing a crime",
    "an abnormal or dangerous situation",
]

NORMAL_PROMPTS = [
    "a normal scene with people walking",
    "people going about their daily activities",
    "a calm and peaceful environment",
    "a regular street scene with traffic",
    "people shopping normally in a store",
    "a person sitting and resting",
    "an empty or quiet environment",
    "normal pedestrian traffic on a sidewalk",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "imagebind" / "imagebind_huge.pth"


@dataclass
class ImageBindBaselineConfig:
    checkpoint_path: str = str(CHECKPOINT_PATH)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    frame_resize: tuple = (224, 224)
    temperature: float = 100.0
    anomaly_prompts: list = field(default_factory=lambda: list(ANOMALY_PROMPTS))
    normal_prompts: list = field(default_factory=lambda: list(NORMAL_PROMPTS))


class ImageBindBaseline:
    """
    ImageBind 零样本视频异常检测。

    对每帧用 ImageBind Vision Encoder 提取特征，
    与异常/正常文本 prompt 集合的平均特征做余弦相似度，
    softmax 得到异常概率。
    """

    def __init__(self, cfg: Optional[ImageBindBaselineConfig] = None):
        self.cfg = cfg or ImageBindBaselineConfig()
        self._model = None
        self._text_features = None
        self._vision_transform = None

    def _ensure_model(self):
        if self._model is not None:
            return

        self._setup_checkpoint_symlink()

        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        from imagebind import data as ib_data

        self._ModalityType = ModalityType
        self._ib_data = ib_data

        logger.info("Loading ImageBind Huge model...")
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval().to(self.cfg.device)
        self._model = model
        logger.info(f"ImageBind model loaded on {self.cfg.device}")

        self._vision_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self._precompute_text_features()

    def _setup_checkpoint_symlink(self):
        """ImageBind 默认从 .checkpoints/ 加载，建立符号链接"""
        os.makedirs(".checkpoints", exist_ok=True)
        link = Path(".checkpoints/imagebind_huge.pth")
        if not link.exists():
            src = Path(self.cfg.checkpoint_path)
            if src.exists():
                os.symlink(src, link)
                logger.info(f"Symlinked checkpoint: {src} -> {link}")
            else:
                raise FileNotFoundError(
                    f"ImageBind checkpoint not found: {src}\n"
                    "Download from: https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
                )

    @torch.no_grad()
    def _precompute_text_features(self):
        """预计算异常/正常文本 prompt 的 ImageBind 特征"""
        ModalityType = self._ModalityType
        ib_data = self._ib_data

        all_prompts = self.cfg.anomaly_prompts + self.cfg.normal_prompts
        text_inputs = {
            ModalityType.TEXT: ib_data.load_and_transform_text(
                all_prompts, self.cfg.device
            )
        }
        embeddings = self._model(text_inputs)
        text_feats = embeddings[ModalityType.TEXT]  # (N, 1024)

        n_anom = len(self.cfg.anomaly_prompts)
        anom_centroid = F.normalize(text_feats[:n_anom].mean(dim=0, keepdim=True), dim=-1)
        norm_centroid = F.normalize(text_feats[n_anom:].mean(dim=0, keepdim=True), dim=-1)

        # (2, D): row 0 = anomaly, row 1 = normal
        self._text_features = torch.cat([anom_centroid, norm_centroid], dim=0)
        logger.info(
            f"Text features precomputed: "
            f"{n_anom} anomaly + {len(self.cfg.normal_prompts)} normal prompts, "
            f"embed_dim={self._text_features.shape[-1]}"
        )

    def _preprocess_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        """BGR numpy frames -> ImageBind vision tensor (B, 3, 224, 224)"""
        images = []
        for f in frames:
            pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            images.append(self._vision_transform(pil))
        return torch.stack(images, dim=0).to(self.cfg.device)

    @torch.no_grad()
    def score_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        计算一批帧的异常分数。

        Args:
            frames: list of BGR numpy arrays (H, W, 3)
        Returns:
            np.ndarray of shape (N,), 每帧异常概率 [0, 1]
        """
        self._ensure_model()
        ModalityType = self._ModalityType

        if not frames:
            return np.array([], dtype=np.float32)

        all_scores = []
        bs = self.cfg.batch_size
        for i in range(0, len(frames), bs):
            batch = frames[i:i + bs]
            vision_tensor = self._preprocess_frames(batch)

            inputs = {ModalityType.VISION: vision_tensor}
            embeddings = self._model(inputs)
            img_feats = embeddings[ModalityType.VISION]  # (B, 1024)

            # (B, 2): similarity to [anomaly, normal]
            sims = img_feats @ self._text_features.T
            probs = F.softmax(sims * self.cfg.temperature, dim=-1)
            anomaly_probs = probs[:, 0].cpu().numpy()
            all_scores.append(anomaly_probs)

        return np.concatenate(all_scores, axis=0).astype(np.float32)

    def analyze_video(
        self,
        video_path: str,
        sample_every_n: int = 1,
        max_frames: int = 0,
    ) -> dict:
        """
        分析单个视频，输出帧级异常分数。

        Returns:
            dict with keys: frame_scores, video_score, total_frames, fps, ...
        """
        self._ensure_model()
        t_start = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"ImageBind Baseline: {video_path} | "
            f"{total_frames} frames, {fps:.1f} fps | sample_every={sample_every_n}"
        )

        frame_buffer = []
        frame_indices = []
        frame_idx = 0
        processed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and processed >= max_frames:
                break
            if frame_idx % sample_every_n != 0:
                frame_idx += 1
                continue

            frame_buffer.append(frame)
            frame_indices.append(frame_idx)
            processed += 1
            frame_idx += 1

        cap.release()

        sampled_scores = self.score_frames(frame_buffer)

        # 插值到全帧
        full_scores = np.zeros(total_frames, dtype=np.float32)
        if len(frame_indices) > 1:
            full_scores = np.interp(
                np.arange(total_frames),
                frame_indices,
                sampled_scores,
            ).astype(np.float32)
        elif len(frame_indices) == 1:
            full_scores[:] = sampled_scores[0]

        # 高斯平滑
        try:
            from scipy.ndimage import gaussian_filter1d
            sigma = max(1, int(1.5 * fps))
            full_scores = gaussian_filter1d(full_scores, sigma=sigma).astype(np.float32)
        except ImportError:
            pass

        full_scores = np.clip(full_scores, 0.0, 1.0)
        video_score = float(full_scores.max())

        elapsed = time.time() - t_start
        logger.info(
            f"ImageBind Baseline done: video_score={video_score:.3f}, "
            f"{processed} frames in {elapsed:.1f}s"
        )

        return {
            "video_path": str(video_path),
            "method": "imagebind_baseline",
            "model": "imagebind_huge",
            "total_frames": total_frames,
            "processed_frames": processed,
            "fps": round(fps, 2),
            "frame_scores": full_scores,
            "video_score": video_score,
            "elapsed_sec": round(elapsed, 2),
        }


# ── CLI 入口 ──────────────────────────────────────────
def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ImageBind Zero-Shot VAD Baseline")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--sample-every", type=int, default=3,
                        help="Process every N-th frame")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = ImageBindBaselineConfig()
    cfg.batch_size = args.batch_size
    cfg.device = args.device if torch.cuda.is_available() else "cpu"

    baseline = ImageBindBaseline(cfg)
    result = baseline.analyze_video(
        video_path=args.video,
        sample_every_n=args.sample_every,
        max_frames=args.max_frames,
    )

    output = {k: v for k, v in result.items() if k != "frame_scores"}
    output["max_frame_score"] = float(result["frame_scores"].max())
    output["mean_frame_score"] = float(result["frame_scores"].mean())

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("output/baselines/imagebind") / Path(args.video).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Result saved to: {out_path}")
    print(f"Video score: {result['video_score']:.3f}")


if __name__ == "__main__":
    main()
