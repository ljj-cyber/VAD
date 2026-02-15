"""
Stage 2-B: Semantic VLLM — Crop 级别语义推理

改造 VLLMClient:
  - 输入: crop_image (裁剪后的实体区域)，而非全图
  - 减少背景干扰，描述更准
  - 输出结构化数据: {action, action_object, posture, scene_context, is_suspicious, danger_score}

支持:
  - vLLM server API (并行，推荐)
  - 本地 transformers (单线程/批量，Qwen2.5-VL-7B-Instruct)
"""

import io
import base64
import time
import json
import logging
import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

from ..config import SemanticVLLMConfig, HF_CACHE_DIR
from .node_trigger import TriggerResult

logger = logging.getLogger(__name__)


# ── 本地模型单例 ──────────────────────────────────────
_local_model = None
_local_processor = None
_local_model_lock = threading.Lock()


def get_local_model(cfg: SemanticVLLMConfig):
    """懒加载本地 Qwen2.5-VL 模型 (单例)"""
    global _local_model, _local_processor

    if _local_model is not None:
        return _local_model, _local_processor

    with _local_model_lock:
        if _local_model is not None:
            return _local_model, _local_processor

        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        model_path = cfg.LOCAL_MODEL_PATHS.get(cfg.model_name, "")
        if not model_path:
            raise ValueError(f"No local model path for {cfg.model_name}")

        logger.info(f"Loading local model from {model_path} ...")
        t0 = time.time()

        _local_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
        )
        _local_processor = AutoProcessor.from_pretrained(model_path)

        logger.info(f"Local model loaded in {time.time() - t0:.1f}s")
        return _local_model, _local_processor


def local_chat_inference(
    model, processor, messages: list[dict],
    max_new_tokens: int = 256, temperature: float = 0.0,
) -> str:
    """用本地模型进行单次推理 (支持图像+文本)"""
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature <= 0.01:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # 只取生成部分
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return result


# ── Prompt ────────────────────────────────────────────

CROP_SYSTEM_PROMPT = """\
You are a surveillance video anomaly detector. You are given a surveillance frame \
with RED/YELLOW bounding boxes highlighting detected moving entities. \
Describe what you see and evaluate whether the behavior is anomalous.

Known anomaly categories to watch for:
  Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, \
  RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism.

Suspicious visual cues include (but are not limited to):
  - Physical contact between people (hitting, grabbing, restraining, pushing)
  - Rapid or chaotic body movements (running, falling, struggling)
  - Fire, smoke, or sudden bright flashes
  - Broken objects, shattered glass, or property damage
  - Weapons or weapon-like objects
  - A person being forced or dragged
  - Unusual postures (lying on the ground, crouching behind objects)
  - Vehicles colliding or driving erratically

Rules:
- "box_action": Describe the visible action of the highlighted entity. Be specific \
about body interactions (e.g. "grabbing another person's arm" not just "standing").
- "context_relation": Describe the visible surroundings and any interaction between entities.
- "scene_type": the overall scene type (e.g. indoor, outdoor, road, parking lot, etc.).
- "is_suspicious": true if you observe ANY of the above cues, false otherwise.
- "danger_score": a float 0.0-1.0. Score ≥0.3 if any suspicious cue is present; \
≥0.6 if the cue is strong (e.g. visible fighting, fire, weapon); ≥0.8 for extreme danger.
- Output ONLY JSON, no extra text.
"""

CROP_USER_PROMPT = """\
Observe this frame at T={timestamp:.2f}s. \
Red/yellow boxes highlight detected motion regions.

Describe what you see. Output ONLY JSON:

{{
  "box_action": "<what is the highlighted entity doing?>",
  "context_relation": "<visible surroundings>",
  "action": "<atomic action verb>",
  "action_object": "<object being interacted with, or none>",
  "posture": "<body posture>",
  "scene_context": "<environment type>",
  "is_suspicious": <true|false>,
  "danger_score": <float 0.0-1.0>
}}
"""

# ── 多帧拼图模式 Prompt ──
GRID_SYSTEM_PROMPT = """\
You are a surveillance video anomaly detector. You are given a 2×2 grid of 4 \
consecutive surveillance frames showing the SAME scene over a short time period. \
The 4 frames are arranged chronologically: top-left is oldest, bottom-right is newest.

Your task: describe the TEMPORAL CHANGE across these 4 frames and detect anomalies.

Known anomaly categories to watch for:
  Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, \
  RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism.

Suspicious temporal cues include (but are not limited to):
  - A person suddenly falls, gets hit, or is restrained across frames
  - Fire or smoke appearing or growing between frames
  - Rapid movement escalation (calm → running/struggling)
  - Objects being broken or thrown
  - Vehicles colliding or swerving
  - A crowd scattering or people fleeing

Rules:
- Describe what CHANGES between frames (not just what's visible in one frame).
- "action": describe the temporal action across frames. Be specific about interactions.
- "is_suspicious": true if you observe ANY of the above cues across the 4 frames.
- "danger_score": based on the SEQUENCE. Score ≥0.3 if suspicious cues emerge; \
≥0.6 for clear anomaly progression; ≥0.8 for extreme danger.
- Output ONLY JSON, no extra text.
"""

GRID_USER_PROMPT = """\
This 2×2 grid shows 4 consecutive frames around T={timestamp:.2f}s. \
Oldest=top-left, Newest=bottom-right.

Describe what happens over these frames. Output ONLY JSON:

{{
  "temporal_action": "<what action unfolds across the 4 frames?>",
  "context_relation": "<visible surroundings>",
  "action": "<atomic action verb for the latest frame>",
  "action_object": "<object or none>",
  "posture": "<body posture in latest frame>",
  "scene_context": "<environment type>",
  "is_suspicious": <true|false>,
  "danger_score": <float 0.0-1.0>
}}
"""


def _pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class SemanticVLLM:
    """
    带全图上下文的语义推理客户端。

    发送"带高亮框的原图"给 VLLM，而非裁剪区域。
    VLLM 同时看到 entity 和周围环境，减少幻觉。
    支持批量并行推理。
    """

    def __init__(self, cfg: Optional[SemanticVLLMConfig] = None):
        self.cfg = cfg or SemanticVLLMConfig()

    def infer_triggers(
        self,
        triggers: list[TriggerResult],
        painted_images: Optional[dict] = None,
        grid_images: Optional[dict] = None,
    ) -> list[dict]:
        """
        对一批触发结果进行语义推理。

        Args:
            triggers: NodeTrigger 输出的触发列表
            painted_images: frame_idx → PIL.Image (带框全图)
            grid_images: frame_idx → PIL.Image (4宫格多帧拼图)

        Returns:
            list[dict]，与 triggers 一一对应
        """
        if not triggers:
            return []

        if self.cfg.backend == "server":
            return self._infer_server(triggers, painted_images, grid_images)
        else:
            return self._infer_local(triggers, painted_images, grid_images)

    def _infer_server(
        self,
        triggers: list[TriggerResult],
        painted_images: Optional[dict] = None,
        grid_images: Optional[dict] = None,
    ) -> list[dict]:
        """通过 vLLM server API 并行推理"""
        import httpx

        model_name = self.cfg.MODEL_PATHS.get(
            self.cfg.model_name, self.cfg.model_name
        )
        results = [None] * len(triggers)

        client = httpx.Client(timeout=90.0)

        def _request_one(idx: int) -> tuple[int, dict]:
            tr = triggers[idx]

            # 选择输入图像和 Prompt:
            # 优先级: grid_images(多帧) > painted_images(带框全图) > crop
            pil_img = None
            use_grid = False

            if grid_images and tr.frame_idx in grid_images:
                pil_img = grid_images[tr.frame_idx]
                use_grid = True
            elif painted_images and tr.frame_idx in painted_images:
                pil_img = painted_images[tr.frame_idx]
            else:
                crop = tr.trace_entry.crop_image
                if crop is None:
                    return idx, self._fallback_result(tr)
                h, w = crop.shape[:2]
                target = self.cfg.crop_resize
                if (w, h) != target:
                    crop = cv2.resize(crop, target, interpolation=cv2.INTER_LINEAR)
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            b64 = _pil_to_base64(pil_img)

            # 根据输入类型选择 Prompt
            if use_grid:
                sys_prompt = GRID_SYSTEM_PROMPT
                user_prompt = GRID_USER_PROMPT.format(timestamp=tr.timestamp)
            else:
                sys_prompt = CROP_SYSTEM_PROMPT
                user_prompt = CROP_USER_PROMPT.format(timestamp=tr.timestamp)

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                "max_tokens": self.cfg.max_new_tokens,
                "temperature": self.cfg.temperature,
            }

            for attempt in range(1, self.cfg.max_retries + 1):
                try:
                    resp = client.post(
                        f"{self.cfg.api_base}/v1/chat/completions",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = data["choices"][0]["message"]["content"].strip()
                    parsed = self._parse_response(raw_text)
                    result = {
                        "entity_id": tr.entity_id,
                        "frame_idx": tr.frame_idx,
                        "timestamp": tr.timestamp,
                        "trigger_rule": tr.trigger_rule,
                        **parsed,
                    }
                    return idx, result
                except Exception as e:
                    if attempt == self.cfg.max_retries:
                        logger.warning(
                            f"Semantic VLLM failed for entity #{tr.entity_id} "
                            f"at frame {tr.frame_idx}: {e}"
                        )
                        return idx, self._fallback_result(tr)

        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
                futures = {executor.submit(_request_one, i): i for i in range(len(triggers))}
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
        finally:
            client.close()

        elapsed = time.time() - t0
        logger.info(
            f"Semantic VLLM: {len(triggers)} crops in {elapsed:.1f}s "
            f"({len(triggers)/max(elapsed, 0.01):.1f} crops/s)"
        )

        return results

    def _infer_local(
        self,
        triggers: list[TriggerResult],
        painted_images: Optional[dict] = None,
        grid_images: Optional[dict] = None,
    ) -> list[dict]:
        """本地 Qwen2.5-VL 推理"""
        if not triggers:
            return []

        model, processor = get_local_model(self.cfg)

        results = []
        t0 = time.time()

        for i, tr in enumerate(triggers):
            # 选择输入图像和 Prompt
            pil_img = None
            use_grid = False

            if grid_images and tr.frame_idx in grid_images:
                pil_img = grid_images[tr.frame_idx]
                use_grid = True
            elif painted_images and tr.frame_idx in painted_images:
                pil_img = painted_images[tr.frame_idx]
            else:
                crop = tr.trace_entry.crop_image
                if crop is None:
                    results.append(self._fallback_result(tr))
                    continue
                h, w = crop.shape[:2]
                target = self.cfg.crop_resize
                if (w, h) != target:
                    crop = cv2.resize(crop, target, interpolation=cv2.INTER_LINEAR)
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            # 根据输入类型选择 Prompt
            if use_grid:
                sys_prompt = GRID_SYSTEM_PROMPT
                user_prompt = GRID_USER_PROMPT.format(timestamp=tr.timestamp)
            else:
                sys_prompt = CROP_SYSTEM_PROMPT
                user_prompt = CROP_USER_PROMPT.format(timestamp=tr.timestamp)

            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            try:
                raw_text = local_chat_inference(
                    model, processor, messages,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                )
                parsed = self._parse_response(raw_text)
                result = {
                    "entity_id": tr.entity_id,
                    "frame_idx": tr.frame_idx,
                    "timestamp": tr.timestamp,
                    "trigger_rule": tr.trigger_rule,
                    **parsed,
                }
                results.append(result)
            except Exception as e:
                logger.warning(
                    f"Local inference failed for entity #{tr.entity_id} "
                    f"at frame {tr.frame_idx}: {e}"
                )
                results.append(self._fallback_result(tr))

        elapsed = time.time() - t0
        logger.info(
            f"Semantic LOCAL: {len(triggers)} crops in {elapsed:.1f}s "
            f"({len(triggers)/max(elapsed, 0.01):.1f} crops/s)"
        )
        return results

    def _parse_response(self, raw_text: str) -> dict:
        """解析 VLLM JSON 响应"""
        import re

        # 尝试直接解析
        try:
            data = json.loads(raw_text)
            return self._sanitize(data)
        except json.JSONDecodeError:
            pass

        # 查找 JSON 块
        match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return self._sanitize(data)
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse semantic response: {raw_text[:200]}")
        return self._default_semantic()

    @staticmethod
    def _sanitize(data: dict) -> dict:
        """清理/标准化语义字段"""
        return {
            "box_action": str(data.get("box_action", data.get("temporal_action", ""))),
            "context_relation": str(data.get("context_relation", "")),
            "action": str(data.get("action", "unknown")),
            "action_object": str(data.get("action_object", "none")),
            "posture": str(data.get("posture", "unknown")),
            "scene_context": str(data.get("scene_context", data.get("scene_type", "unknown"))),
            "is_suspicious": bool(data.get("is_suspicious", False)),
            "danger_score": float(data.get("danger_score", 0.0)),
        }

    @staticmethod
    def _default_semantic() -> dict:
        return {
            "box_action": "",
            "context_relation": "",
            "action": "unknown",
            "action_object": "none",
            "posture": "unknown",
            "scene_context": "unknown",
            "is_suspicious": False,
            "danger_score": 0.0,
        }

    def _fallback_result(self, tr: TriggerResult) -> dict:
        return {
            "entity_id": tr.entity_id,
            "frame_idx": tr.frame_idx,
            "timestamp": tr.timestamp,
            "trigger_rule": tr.trigger_rule,
            **self._default_semantic(),
        }
