"""
CLIP Zero-Shot Video Anomaly Detection Baseline

方法:
  对视频的每一帧，用 CLIP 计算图像与「正常/异常」文本 prompt 的相似度，
  取 softmax 后的异常概率作为帧级异常分数。

核心原理:
  score(frame) = softmax( [sim(frame, abnormal_prompts), sim(frame, normal_prompts)] )[0]

用法:
  python -m baselines.clip_baseline --video /path/to/video.mp4
  python -m baselines.clip_baseline --video /path/to/video.mp4 --model ViT-L/14
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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


@dataclass
class CLIPBaselineConfig:
    model_name: str = "openai/clip-vit-base-patch16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    frame_resize: tuple = (224, 224)
    temperature: float = 100.0  # logit_scale (CLIP default ~100)
    anomaly_prompts: list = field(default_factory=lambda: list(ANOMALY_PROMPTS))
    normal_prompts: list = field(default_factory=lambda: list(NORMAL_PROMPTS))


class CLIPBaseline:
    """
    CLIP 零样本视频异常检测。

    对每帧计算与异常/正常 prompt 集合的平均相似度，
    用 softmax 得到异常概率。
    """

    def __init__(self, cfg: Optional[CLIPBaselineConfig] = None):
        self.cfg = cfg or CLIPBaselineConfig()
        self._model = None
        self._processor = None
        self._text_features = None  # 缓存文本特征

    def _ensure_model(self):
        if self._model is not None:
            return

        from transformers import CLIPProcessor, CLIPModel

        logger.info(f"Loading CLIP model: {self.cfg.model_name}")
        self._processor = CLIPProcessor.from_pretrained(self.cfg.model_name)
        self._model = CLIPModel.from_pretrained(self.cfg.model_name)
        self._model = self._model.to(self.cfg.device).eval()
        logger.info("CLIP model loaded.")

        self._precompute_text_features()

    @torch.no_grad()
    def _precompute_text_features(self):
        """预计算所有文本 prompt 的 CLIP 特征"""
        all_prompts = self.cfg.anomaly_prompts + self.cfg.normal_prompts
        inputs = self._processor(text=all_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}
        text_feats = self._model.get_text_features(**inputs)
        text_feats = F.normalize(text_feats, dim=-1)

        n_anom = len(self.cfg.anomaly_prompts)
        self._anom_text_feats = text_feats[:n_anom].mean(dim=0, keepdim=True)
        self._anom_text_feats = F.normalize(self._anom_text_feats, dim=-1)
        self._norm_text_feats = text_feats[n_anom:].mean(dim=0, keepdim=True)
        self._norm_text_feats = F.normalize(self._norm_text_feats, dim=-1)

        # (2, D) — row 0 = anomaly centroid, row 1 = normal centroid
        self._text_features = torch.cat(
            [self._anom_text_feats, self._norm_text_feats], dim=0
        )
        logger.info(
            f"Text features precomputed: "
            f"{n_anom} anomaly + {len(self.cfg.normal_prompts)} normal prompts"
        )

    @torch.no_grad()
    def score_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        计算一批帧的异常分数。

        Args:
            frames: list of BGR numpy arrays (H, W, 3)

        Returns:
            np.ndarray of shape (N,), 每帧的异常概率 [0, 1]
        """
        self._ensure_model()

        if not frames:
            return np.array([], dtype=np.float32)

        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

        all_scores = []
        bs = self.cfg.batch_size
        for i in range(0, len(pil_images), bs):
            batch = pil_images[i:i + bs]
            inputs = self._processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.cfg.device)

            img_feats = self._model.get_image_features(pixel_values=pixel_values)
            img_feats = F.normalize(img_feats, dim=-1)

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
            f"CLIP Baseline: {video_path} | "
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

        # 批量推理
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
            f"CLIP Baseline done: video_score={video_score:.3f}, "
            f"{processed} frames in {elapsed:.1f}s"
        )

        return {
            "video_path": str(video_path),
            "method": "clip_baseline",
            "model": self.cfg.model_name,
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

    parser = argparse.ArgumentParser(description="CLIP Zero-Shot VAD Baseline")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--model", default="openai/clip-vit-base-patch16",
                        help="CLIP model name (HuggingFace)")
    parser.add_argument("--sample-every", type=int, default=3,
                        help="Process every N-th frame")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = CLIPBaselineConfig()
    cfg.model_name = args.model
    cfg.batch_size = args.batch_size

    baseline = CLIPBaseline(cfg)
    result = baseline.analyze_video(
        video_path=args.video,
        sample_every_n=args.sample_every,
        max_frames=args.max_frames,
    )

    # frame_scores 不保存到 JSON (太大)
    output = {k: v for k, v in result.items() if k != "frame_scores"}
    output["max_frame_score"] = float(result["frame_scores"].max())
    output["mean_frame_score"] = float(result["frame_scores"].mean())

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("output/baselines/clip") / Path(args.video).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Result saved to: {out_path}")
    print(f"Video score: {result['video_score']:.3f}")


if __name__ == "__main__":
    main()
