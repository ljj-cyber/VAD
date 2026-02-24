"""
Video-LLaVA Zero-Shot Video Anomaly Detection Baseline

方法:
  将视频切分为多个 segment，每个 segment 均匀采样 8 帧送入 Video-LLaVA，
  通过 prompt 让模型给出 0-10 的异常评分，解析后归一化为帧级异常分数。

用法:
  python -m baselines.videollava_baseline --video /path/to/video.mp4
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

ANOMALY_PROMPT = (
    "Watch this surveillance video clip carefully. "
    "Rate the anomaly level from 0 to 10, where:\n"
    "- 0: completely normal, nothing unusual\n"
    "- 3: mildly suspicious but likely harmless\n"
    "- 5: moderately suspicious activity\n"
    "- 7: likely dangerous or violent behavior\n"
    "- 10: clearly dangerous, violent, or criminal activity\n\n"
    "Reply ONLY with: Score: X"
)


@dataclass
class VideoLLaVAConfig:
    model_path: str = "/date/liuzhe/EventVAD/Video-LLaVA/checkpoints/Video-LLaVA-7B"
    device: str = "cuda"
    load_4bit: bool = True
    load_8bit: bool = False
    num_frames: int = 8
    segment_sec: float = 8.0
    overlap_ratio: float = 0.25
    temperature: float = 0.1
    max_new_tokens: int = 64
    default_score_on_parse_fail: float = 0.3


class VideoLLaVABaseline:
    """
    Video-LLaVA 零样本视频异常检测。

    将视频切为固定时长 segment，每个 segment 用 Video-LLaVA 评分，
    聚合为帧级异常分数。
    """

    def __init__(self, cfg: Optional[VideoLLaVAConfig] = None):
        self.cfg = cfg or VideoLLaVAConfig()
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._conv_mode = "llava_v1"

    def _ensure_model(self):
        if self._model is not None:
            return

        os.environ.setdefault("http_proxy", "http://127.0.0.1:7892")
        os.environ.setdefault("https_proxy", "http://127.0.0.1:7892")

        from videollava.model.builder import load_pretrained_model
        from videollava.utils import disable_torch_init
        from videollava.mm_utils import get_model_name_from_path

        disable_torch_init()
        model_name = get_model_name_from_path(self.cfg.model_path)

        logger.info(f"Loading Video-LLaVA: {self.cfg.model_path} (4bit={self.cfg.load_4bit})")
        self._tokenizer, self._model, self._processor, _ = load_pretrained_model(
            self.cfg.model_path, None, model_name,
            self.cfg.load_8bit, self.cfg.load_4bit,
            device=self.cfg.device,
        )
        logger.info("Video-LLaVA loaded.")

    def _sample_frames_from_segment(
        self, cap: cv2.VideoCapture, start_frame: int, end_frame: int
    ) -> list[np.ndarray]:
        """从 [start_frame, end_frame) 区间均匀采样 num_frames 帧"""
        n = self.cfg.num_frames
        total = end_frame - start_frame
        if total <= 0:
            return []

        indices = np.linspace(start_frame, end_frame - 1, n, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1].copy())
        return frames

    @torch.no_grad()
    def _score_segment_frames(self, frames: list[np.ndarray]) -> float:
        """对一组 8 帧用 Video-LLaVA 打分，返回 0~1 异常分数"""
        from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from videollava.conversation import conv_templates, SeparatorStyle
        from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from PIL import Image

        if not frames or len(frames) < 2:
            return 0.0

        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        video_processor = self._processor['video']

        try:
            video_tensor = video_processor.preprocess(pil_images, return_tensors='pt')['pixel_values']
        except Exception:
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensors = [transform(img) for img in pil_images]
            video_tensor = torch.stack(tensors).unsqueeze(0)

        if isinstance(video_tensor, list):
            tensor = [v.to(self._model.device, dtype=torch.float16) for v in video_tensor]
        else:
            tensor = video_tensor.to(self._model.device, dtype=torch.float16)

        n_frames = self._model.get_video_tower().config.num_frames
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * n_frames) + '\n' + ANOMALY_PROMPT

        conv = conv_templates[self._conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self._model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self._tokenizer, input_ids
        )

        output_ids = self._model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=self.cfg.temperature,
            max_new_tokens=self.cfg.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

        response = self._tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        score = self._parse_score(response)
        return score

    def _parse_score(self, response: str) -> float:
        """从模型回复中解析 0-10 分数，归一化到 0-1"""
        patterns = [
            r'[Ss]core:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'(\d+(?:\.\d+)?)\s*out of\s*10',
            r'^(\d+(?:\.\d+)?)\s*$',
        ]
        for pat in patterns:
            m = re.search(pat, response)
            if m:
                val = float(m.group(1))
                return min(val / 10.0, 1.0)

        nums = re.findall(r'\d+(?:\.\d+)?', response)
        if nums:
            val = float(nums[0])
            if val <= 10:
                return val / 10.0

        logger.debug(f"Score parse failed, response: {response!r}")
        return self.cfg.default_score_on_parse_fail

    def analyze_video(
        self,
        video_path: str,
        sample_every_n: int = 1,
        max_segments: int = 0,
    ) -> dict:
        """
        分析单个视频，返回帧级异常分数。

        Returns:
            dict: frame_scores, video_score, total_frames, fps, ...
        """
        self._ensure_model()
        t_start = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        segment_frames = int(self.cfg.segment_sec * fps)
        overlap_frames = int(segment_frames * self.cfg.overlap_ratio)
        step = max(1, segment_frames - overlap_frames)

        segments = []
        pos = 0
        while pos < total_frames:
            end = min(pos + segment_frames, total_frames)
            if end - pos >= self.cfg.num_frames:
                segments.append((pos, end))
            pos += step

        if max_segments > 0 and len(segments) > max_segments:
            indices = np.linspace(0, len(segments) - 1, max_segments, dtype=int)
            segments = [segments[i] for i in indices]

        if sample_every_n > 1 and len(segments) > 1:
            segments = segments[::sample_every_n]

        logger.info(
            f"Video-LLaVA Baseline: {video_path} | "
            f"{total_frames} frames, {fps:.1f} fps | "
            f"{len(segments)} segments (seg={self.cfg.segment_sec}s)"
        )

        segment_scores = []
        segment_ranges = []
        for seg_idx, (sf, ef) in enumerate(segments):
            frames = self._sample_frames_from_segment(cap, sf, ef)
            if not frames:
                segment_scores.append(0.0)
                segment_ranges.append((sf, ef))
                continue

            score = self._score_segment_frames(frames)
            segment_scores.append(score)
            segment_ranges.append((sf, ef))

            logger.debug(
                f"  Segment {seg_idx}: frames [{sf}-{ef}] → score={score:.3f}"
            )

        cap.release()

        full_scores = self._build_frame_scores(
            segment_scores, segment_ranges, total_frames, fps
        )

        video_score = float(full_scores.max()) if len(full_scores) > 0 else 0.0

        elapsed = time.time() - t_start
        logger.info(
            f"Video-LLaVA Baseline done: video_score={video_score:.3f}, "
            f"{len(segments)} segments in {elapsed:.1f}s"
        )

        return {
            "video_path": str(video_path),
            "method": "videollava_baseline",
            "model": self.cfg.model_path,
            "total_frames": total_frames,
            "processed_segments": len(segments),
            "fps": round(fps, 2),
            "frame_scores": full_scores,
            "video_score": video_score,
            "segment_scores": segment_scores,
            "elapsed_sec": round(elapsed, 2),
        }

    def _build_frame_scores(
        self,
        segment_scores: list[float],
        segment_ranges: list[tuple],
        total_frames: int,
        fps: float,
    ) -> np.ndarray:
        """将 segment 分数映射到帧级分数，带高斯平滑"""
        scores = np.zeros(total_frames, dtype=np.float32)
        counts = np.zeros(total_frames, dtype=np.float32)

        for sc, (sf, ef) in zip(segment_scores, segment_ranges):
            scores[sf:ef] += sc
            counts[sf:ef] += 1.0

        mask = counts > 0
        scores[mask] /= counts[mask]

        try:
            from scipy.ndimage import gaussian_filter1d
            sigma = max(1, int(2.0 * fps))
            scores = gaussian_filter1d(scores, sigma=sigma).astype(np.float32)
        except ImportError:
            pass

        return np.clip(scores, 0.0, 1.0)


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Video-LLaVA Zero-Shot VAD Baseline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model-path", default=VideoLLaVAConfig.model_path)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--segment-sec", type=float, default=8.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = VideoLLaVAConfig()
    cfg.model_path = args.model_path
    cfg.segment_sec = args.segment_sec

    baseline = VideoLLaVABaseline(cfg)
    result = baseline.analyze_video(
        video_path=args.video,
        sample_every_n=args.sample_every,
        max_segments=args.max_segments,
    )

    output = {k: v for k, v in result.items() if k != "frame_scores"}
    output["max_frame_score"] = float(result["frame_scores"].max())
    output["mean_frame_score"] = float(result["frame_scores"].mean())

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("output/baselines/videollava") / Path(args.video).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Result: {out_path}")
    print(f"Video score: {result['video_score']:.3f}")


if __name__ == "__main__":
    main()
