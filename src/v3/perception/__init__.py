"""
V3.0 语义感知层

- AdaptiveFrameSampler: 自适应帧采样器
- VLLMClient: VLLM 推理客户端
- prompt_template: 提示词模板
"""

from .frame_sampler import AdaptiveFrameSampler
from .vllm_client import VLLMClient

__all__ = ["AdaptiveFrameSampler", "VLLMClient"]
