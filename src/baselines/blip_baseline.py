"""
BLIP Zero-Shot Video Anomaly Detection Baseline

方法:
  使用 BLIP 的 Image-Text Contrastive (ITC) 头，计算每帧图像与
  「正常/异常」文本 prompt 之间的对比相似度，取 softmax 后的异常概率
  作为帧级异常分数。

  支持两种加载方式:
    1. HuggingFace transformers — BlipForImageTextRetrieval (默认，推荐)
    2. LAVIS 框架 — blip2_image_text_matching (需要 src/LAVIS)

核心原理:
  score(frame) = softmax( [sim(frame, abnormal_centroid), sim(frame, normal_centroid)] )[0]

用法:
  python -m baselines.blip_baseline --video /path/to/video.mp4
  python -m baselines.blip_baseline --video /path/to/video.mp4 --model Salesforce/blip-itm-large-coco
  python -m baselines.blip_baseline --video /path/to/video.mp4 --loader lavis
"""

import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


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
class BLIPBaselineConfig:
    hf_model_name: str = "Salesforce/blip-itm-base-coco"
    lavis_model_name: str = "blip2_image_text_matching"
    lavis_model_type: str = "pretrain"
    loader: str = "transformers"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    temperature: float = 100.0
    anomaly_prompts: list = field(default_factory=lambda: list(ANOMALY_PROMPTS))
    normal_prompts: list = field(default_factory=lambda: list(NORMAL_PROMPTS))


# ── 抽象 Scorer ──────────────────────────────────────
class _BaseScorer(ABC):
    @abstractmethod
    def get_text_centroids(
        self, anomaly_prompts: list[str], normal_prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (anomaly_centroid, normal_centroid)，各 shape (1, D)"""
        ...

    @abstractmethod
    def get_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        """返回 (B, D) 归一化图像特征"""
        ...


# ── HuggingFace Transformers Scorer ──────────────────
class _TransformersScorer(_BaseScorer):
    """使用 BlipForImageTextRetrieval 的 ITC 投影头提取特征"""

    def __init__(self, cfg: BLIPBaselineConfig):
        self.cfg = cfg
        from transformers import BlipProcessor, BlipForImageTextRetrieval

        logger.info(f"Loading BLIP (transformers): {cfg.hf_model_name}")
        self.processor = BlipProcessor.from_pretrained(cfg.hf_model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(
            cfg.hf_model_name, torch_dtype=torch.float16
        ).to(cfg.device).eval()
        logger.info("BLIP model loaded (transformers).")

    @torch.no_grad()
    def get_text_centroids(
        self, anomaly_prompts: list[str], normal_prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_prompts = anomaly_prompts + normal_prompts
        text_feats_list = []
        for text in all_prompts:
            inputs = self.processor(text=text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.cfg.device)
            attn_mask = inputs.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.cfg.device)
            text_out = self.model.text_encoder(
                input_ids=input_ids, attention_mask=attn_mask
            )
            # CLS token → text projection
            cls_emb = text_out.last_hidden_state[:, 0, :]
            text_feat = F.normalize(
                self.model.text_proj(cls_emb).float(), dim=-1
            )
            text_feats_list.append(text_feat)

        text_feats = torch.cat(text_feats_list, dim=0)  # (N, D)
        n_anom = len(anomaly_prompts)
        anom_centroid = F.normalize(text_feats[:n_anom].mean(dim=0, keepdim=True), dim=-1)
        norm_centroid = F.normalize(text_feats[n_anom:].mean(dim=0, keepdim=True), dim=-1)
        return anom_centroid, norm_centroid

    @torch.no_grad()
    def get_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.cfg.device)
        vision_out = self.model.vision_model(pixel_values=pixel_values)
        # CLS token → vision projection
        cls_emb = vision_out.last_hidden_state[:, 0, :]
        img_feats = F.normalize(self.model.vision_proj(cls_emb).float(), dim=-1)
        return img_feats


# ── LAVIS Scorer ─────────────────────────────────────
class _LAVISScorer(_BaseScorer):
    """使用 LAVIS 的 blip2_image_text_matching (ITC mode)"""

    def __init__(self, cfg: BLIPBaselineConfig):
        self.cfg = cfg
        src_dir = Path(__file__).resolve().parents[1]
        lavis_dir = src_dir / "LAVIS"
        if str(lavis_dir) not in sys.path:
            sys.path.insert(0, str(lavis_dir))

        from lavis.models import load_model_and_preprocess

        logger.info(
            f"Loading BLIP-2 (LAVIS): {cfg.lavis_model_name} / {cfg.lavis_model_type}"
        )
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=cfg.lavis_model_name,
            model_type=cfg.lavis_model_type,
            is_eval=True,
            device=cfg.device,
        )
        logger.info("BLIP-2 model loaded (LAVIS).")

    @torch.no_grad()
    def get_text_centroids(
        self, anomaly_prompts: list[str], normal_prompts: list[str]
    ) -> tuple[None, None]:
        # LAVIS 模式不预计算质心，直接在 score_frames 中逐帧计算
        return None, None

    @torch.no_grad()
    def get_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError("LAVIS scorer uses direct similarity computation")

    @torch.no_grad()
    def compute_frame_scores(
        self, images: list[Image.Image],
        anomaly_prompts: list[str], normal_prompts: list[str],
        temperature: float,
    ) -> np.ndarray:
        """LAVIS 模式：逐帧计算异常分数"""
        vis_proc = self.vis_processors["eval"]
        scores = []
        for img in images:
            processed = vis_proc(img).unsqueeze(0).to(self.cfg.device)
            anom_sims = []
            for text in anomaly_prompts:
                sim = self.model(
                    {"image": processed, "text_input": [text]}, match_head="itc"
                )
                anom_sims.append(sim.item())
            norm_sims = []
            for text in normal_prompts:
                sim = self.model(
                    {"image": processed, "text_input": [text]}, match_head="itc"
                )
                norm_sims.append(sim.item())

            anom_avg = np.mean(anom_sims)
            norm_avg = np.mean(norm_sims)
            combined = torch.tensor([[anom_avg, norm_avg]])
            prob = F.softmax(combined * temperature, dim=-1)
            scores.append(prob[0, 0].item())

        return np.array(scores, dtype=np.float32)


class BLIPBaseline:
    """
    BLIP 零样本视频异常检测。

    对每帧计算与异常/正常 prompt 集合的 ITC 相似度，
    用 softmax 得到异常概率。
    """

    def __init__(self, cfg: Optional[BLIPBaselineConfig] = None):
        self.cfg = cfg or BLIPBaselineConfig()
        self._scorer: Optional[_BaseScorer] = None
        self._anom_centroid = None
        self._norm_centroid = None

    def _ensure_model(self):
        if self._scorer is not None:
            return

        if self.cfg.loader == "transformers":
            self._scorer = _TransformersScorer(self.cfg)
        elif self.cfg.loader == "lavis":
            self._scorer = _LAVISScorer(self.cfg)
        else:
            raise ValueError(f"Unknown loader: {self.cfg.loader}")

        anom_c, norm_c = self._scorer.get_text_centroids(
            self.cfg.anomaly_prompts, self.cfg.normal_prompts
        )
        self._anom_centroid = anom_c
        self._norm_centroid = norm_c

        if anom_c is not None:
            logger.info(
                f"Text centroids precomputed: "
                f"{len(self.cfg.anomaly_prompts)} anomaly + "
                f"{len(self.cfg.normal_prompts)} normal"
            )

    @torch.no_grad()
    def score_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        计算一批帧的异常分数。

        Args:
            frames: list of BGR numpy arrays

        Returns:
            np.ndarray of shape (N,), 每帧的异常概率 [0, 1]
        """
        self._ensure_model()

        if not frames:
            return np.array([], dtype=np.float32)

        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

        # LAVIS 模式：直接逐帧计算
        if self.cfg.loader == "lavis":
            assert isinstance(self._scorer, _LAVISScorer)
            return self._scorer.compute_frame_scores(
                pil_images,
                self.cfg.anomaly_prompts, self.cfg.normal_prompts,
                self.cfg.temperature,
            )

        # Transformers 模式：batch 推理
        assert isinstance(self._scorer, _TransformersScorer)
        text_centroids = torch.cat(
            [self._anom_centroid, self._norm_centroid], dim=0
        )  # (2, D)

        all_scores = []
        bs = self.cfg.batch_size
        for i in range(0, len(pil_images), bs):
            batch = pil_images[i:i + bs]
            img_feats = self._scorer.get_image_features(batch)  # (B, D)
            sims = img_feats @ text_centroids.T  # (B, 2)
            probs = F.softmax(sims * self.cfg.temperature, dim=-1)
            all_scores.append(probs[:, 0].cpu().numpy())

        return np.concatenate(all_scores, axis=0).astype(np.float32)

    def analyze_video(
        self,
        video_path: str,
        sample_every_n: int = 3,
        max_frames: int = 0,
    ) -> dict:
        self._ensure_model()
        t_start = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"BLIP Baseline: {video_path} | "
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
            f"BLIP Baseline done: video_score={video_score:.3f}, "
            f"{processed} frames in {elapsed:.1f}s"
        )

        return {
            "video_path": str(video_path),
            "method": "blip_baseline",
            "model": self.cfg.hf_model_name if self.cfg.loader == "transformers"
                     else f"{self.cfg.lavis_model_name}/{self.cfg.lavis_model_type}",
            "loader": self.cfg.loader,
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

    parser = argparse.ArgumentParser(description="BLIP Zero-Shot VAD Baseline")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--model", default="Salesforce/blip-itm-base-coco",
                        help="HuggingFace BLIP model name")
    parser.add_argument("--loader", default="transformers",
                        choices=["transformers", "lavis"])
    parser.add_argument("--lavis-type", default="pretrain",
                        help="LAVIS model type (for --loader lavis)")
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = BLIPBaselineConfig()
    cfg.hf_model_name = args.model
    cfg.loader = args.loader
    cfg.lavis_model_type = args.lavis_type
    cfg.batch_size = args.batch_size

    baseline = BLIPBaseline(cfg)
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
        out_dir = Path("output/baselines/blip") / Path(args.video).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Result saved to: {out_path}")
    print(f"Video score: {result['video_score']:.3f}")


if __name__ == "__main__":
    main()
