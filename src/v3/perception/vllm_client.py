"""
V4.1 语义感知层 — VLLM 推理客户端

支持两种推理后端:
  1. "local"  — 直接加载模型到 GPU (transformers)
  2. "server" — 通过 vLLM OpenAI-compatible API 并行推理 (推荐)

vLLM server 模式优势:
  - Continuous Batching: 自动合并并发请求
  - PagedAttention: 高效显存管理
  - 并行: ThreadPoolExecutor 多线程并发发送请求
  - 吞吐量提升 5-10×
"""

import io
import base64
import time
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from PIL import Image

from ..config import PerceptionConfig, SystemConfig, HF_CACHE_DIR
from ..utils.json_schema import extract_json_from_response, validate_snapshot, sanitize_snapshot
from .prompt_template import SYSTEM_PROMPT, build_user_prompt, RETRY_PROMPT

logger = logging.getLogger(__name__)


def _pil_to_base64(image: Image.Image) -> str:
    """PIL Image → base64 编码字符串"""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VLLMClient:
    """
    视觉语言模型推理客户端。

    支持两种模式:
      - backend="local":  直接加载模型 (适合调试)
      - backend="server": 通过 vLLM API server 并行推理 (推荐生产使用)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        backend: str = "server",
        api_base: str = "http://localhost:8000",
        max_workers: int = 16,
    ):
        self.cfg = PerceptionConfig()
        self.model_name = model_name or self.cfg.model_name
        self.device = device or SystemConfig.device
        self.backend = backend
        self.api_base = api_base.rstrip("/")
        self.max_workers = max_workers

        # local 模式的模型引用
        self.model = None
        self.processor = None
        self._loaded = False

    # ══════════════════════════════════════════════════
    #  vLLM Server 模式 (推荐)
    # ══════════════════════════════════════════════════

    def infer_batch(
        self,
        images: list[Image.Image],
        frame_ids: list[int],
        timestamps: list[float],
        batch_size: int = 8,
    ) -> list[dict]:
        """
        批量推理：根据 backend 选择推理方式。

        server 模式: 并行发送 HTTP 请求，vLLM 自动 continuous batching
        local 模式: 逐帧串行推理
        """
        if self.backend == "server":
            return self._infer_batch_server(images, frame_ids, timestamps)
        else:
            return self._infer_batch_local(images, frame_ids, timestamps, batch_size)

    def _infer_batch_server(
        self,
        images: list[Image.Image],
        frame_ids: list[int],
        timestamps: list[float],
    ) -> list[dict]:
        """通过 vLLM API server 并行推理"""
        import httpx

        total = len(images)
        results = [None] * total

        client = httpx.Client(timeout=90.0)

        def _request_one(idx: int) -> tuple[int, dict]:
            img = images[idx]
            fid = frame_ids[idx]
            ts = timestamps[idx]
            prompt = build_user_prompt(fid, ts)
            b64 = _pil_to_base64(img)

            payload = {
                "model": self.cfg.MODEL_PATHS.get(self.model_name, self.model_name),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                "max_tokens": self.cfg.max_new_tokens,
                "temperature": 0,
            }

            for attempt in range(1, self.cfg.max_retries + 1):
                try:
                    resp = client.post(
                        f"{self.api_base}/v1/chat/completions",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    raw_text = data["choices"][0]["message"]["content"].strip()
                    snapshot = self._parse_response(raw_text, fid, ts)
                    return idx, snapshot

                except Exception as e:
                    if attempt == self.cfg.max_retries:
                        logger.warning(f"Frame {fid} failed after {attempt} retries: {e}")
                        return idx, sanitize_snapshot({
                            "frame_id": fid, "timestamp": ts,
                            "entities": [], "scene": {}, "motion_energy": 0.0,
                        })

        # 并行发送所有请求
        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_request_one, i): i for i in range(total)}

                done_count = 0
                for future in as_completed(futures):
                    idx, snapshot = future.result()
                    results[idx] = snapshot
                    done_count += 1
                    if done_count % 10 == 0 or done_count == total:
                        elapsed = time.time() - t0
                        logger.info(
                            f"  Server inference: {done_count}/{total} frames "
                            f"({elapsed:.1f}s, {done_count / elapsed:.1f} fps)"
                        )
        finally:
            client.close()

        return results

    # ══════════════════════════════════════════════════
    #  Local 模式 (transformers 直接加载)
    # ══════════════════════════════════════════════════

    def load_model(self):
        """延迟加载模型到 GPU (local 模式)"""
        if self._loaded:
            return

        model_path = self.cfg.MODEL_PATHS.get(self.model_name)
        if model_path is None:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Available: {list(self.cfg.MODEL_PATHS.keys())}"
            )

        logger.info(f"Loading VLLM model: {model_path} on {self.device}")
        t0 = time.time()

        if self.model_name == "qwen2-vl-7b":
            self._load_qwen2_vl(model_path)
        elif self.model_name == "moondream2":
            self._load_moondream2(model_path)

        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.1f}s")
        self._loaded = True

    def _load_qwen2_vl(self, model_path: str):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if SystemConfig.fp16_enabled else torch.float32,
            device_map=self.device,
            attn_implementation=self.cfg.attn_implementation,
            cache_dir=str(HF_CACHE_DIR / "hub"),
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, cache_dir=str(HF_CACHE_DIR / "hub"),
        )

    def _load_moondream2(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if SystemConfig.fp16_enabled else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir=str(HF_CACHE_DIR / "hub"),
        ).eval()
        self.processor = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
            cache_dir=str(HF_CACHE_DIR / "hub"),
        )

    @torch.inference_mode()
    def infer_frame(self, image: Image.Image, frame_id: int = 0, timestamp: float = 0.0) -> dict:
        """单帧推理 (local 模式)"""
        if self.backend == "server":
            results = self._infer_batch_server([image], [frame_id], [timestamp])
            return results[0]

        self.load_model()
        user_prompt = build_user_prompt(frame_id, timestamp)

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                raw_text = self._generate_local(image, user_prompt)
                snapshot = self._parse_response(raw_text, frame_id, timestamp)
                if snapshot.get("entities"):
                    return snapshot
                    user_prompt = RETRY_PROMPT
            except Exception as e:
                logger.error(f"[Attempt {attempt}] Inference error: {e}")

        return sanitize_snapshot({
            "frame_id": frame_id, "timestamp": timestamp,
            "entities": [], "scene": {}, "motion_energy": 0.0,
        })

    def _generate_local(self, image: Image.Image, prompt: str) -> str:
        """本地模型推理"""
        if self.model_name == "qwen2-vl-7b":
            from qwen_vl_utils import process_vision_info

        messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
            image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
                text=[text_input], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
        ).to(self.device)
            out = self.model.generate(
                **inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False,
            )
            return self.processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()

        elif self.model_name == "moondream2":
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            return self.model.answer_question(
                self.model.encode_image(image), full_prompt, self.processor,
            ).strip()

        raise ValueError(f"Unsupported model: {self.model_name}")

    def _infer_batch_local(self, images, frame_ids, timestamps, batch_size):
        """本地批量推理（逐帧）"""
        results = []
        for img, fid, ts in zip(images, frame_ids, timestamps):
            results.append(self.infer_frame(img, fid, ts))
        return results

    # ══════════════════════════════════════════════════
    #  通用工具
    # ══════════════════════════════════════════════════

    def _parse_response(self, raw_text: str, frame_id: int, timestamp: float) -> dict:
        """解析 VLLM 响应为快照"""
        parsed = extract_json_from_response(raw_text)
        if parsed is None:
            logger.warning(f"Failed to extract JSON for frame {frame_id}: {raw_text[:150]}")
            return sanitize_snapshot({
                "frame_id": frame_id, "timestamp": timestamp,
                "entities": [], "scene": {}, "motion_energy": 0.0,
            })

        is_valid, errors = validate_snapshot(parsed)
        if not is_valid:
            logger.debug(f"Frame {frame_id} validation: {errors}")

        parsed["frame_id"] = frame_id
        parsed["timestamp"] = timestamp
        return sanitize_snapshot(parsed)

    def unload(self):
        """释放模型显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory freed.")

    def __del__(self):
        try:
        self.unload()
        except Exception:
            pass
