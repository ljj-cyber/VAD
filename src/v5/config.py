"""
V5 Tube-Skeleton Pipeline — 全局配置

三阶段架构:
  Stage 1: 物理追踪骨架 (motion + CLIP crop tracking)
  Stage 2: 稀疏语义挂载 (sparse VLLM node generation)
  Stage 3: 动态图审计   (graph + narrative + decision)
"""
import os
import pathlib
import torch

# ── 路径 ─────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # EventVAD/
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "v5"

HF_CACHE_DIR = MODELS_DIR / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))


# ── Stage 1: 物理追踪 ────────────────────────────────
class MotionConfig:
    """帧差动能 + 连通域提取配置"""

    # 帧差阈值 (0-255)，低于此值视为背景
    diff_threshold: int = 25
    # 形态学核大小 (去噪)
    morph_kernel_size: int = 5
    # 高斯模糊核 (预处理)
    blur_kernel_size: int = 5
    # 连通域最小面积 (像素数)，低于此值忽略
    min_region_area: int = 1200
    # 提取 Top-K 动能连通域
    top_k_regions: int = 2
    # Crop padding 比例 (bbox 外扩)
    crop_padding_ratio: float = 0.15
    # 最小 crop 尺寸 (像素)
    min_crop_size: int = 80


class CLIPEncoderConfig:
    """CLIP Crop 特征提取配置"""

    model_name: str = "openai/clip-vit-base-patch32"
    feature_dim: int = 512
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TrackerConfig:
    """Entity Tracker 贪婪匹配配置"""

    # 余弦相似度阈值：高于此值 → 沿用旧 ID
    similarity_threshold: float = 0.82
    # 最大允许的帧间隔（超过则 Entity 死亡）
    max_age_frames: int = 30
    # 最大活跃实体数
    max_active_entities: int = 20
    # 新 Entity 的最小动能 (低于此值不分配新 ID)
    min_kinetic_for_new: float = 0.04


# ── Stage 2: 稀疏语义挂载 ────────────────────────────
class NodeTriggerConfig:
    """语义节点触发策略"""

    # Rule 1: Birth — 新 ID 出现即触发
    trigger_on_birth: bool = True
    # Rule 2: Change Point — 与上次采样的 embedding 距离阈值
    embedding_jump_threshold: float = 0.25  # 1 - cosine_similarity
    # Rule 3: Heartbeat — 距上次采样最大间隔 (秒)
    heartbeat_interval_sec: float = 3.0
    # 最大触发频率保护 (同一实体相邻触发的最小间隔帧数)
    min_trigger_gap_frames: int = 5


class SemanticVLLMConfig:
    """VLLM 语义推理 (仅对 crop 区域)"""

    # VLLM model
    model_name: str = "qwen2-vl-7b"
    MODEL_PATHS: dict = {
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    }

    # API server
    backend: str = "server"
    api_base: str = "http://localhost:8000"
    max_workers: int = 16

    # 推理参数
    max_new_tokens: int = 256
    temperature: float = 0.0
    max_retries: int = 2

    # Crop 尺寸 (resize 后喂给 VLLM)
    crop_resize: tuple = (336, 336)


# ── Stage 3: 动态图审计 ──────────────────────────────
class GraphConfig:
    """时间演化图配置"""

    # 最大图节点数 (每实体)
    max_nodes_per_entity: int = 100
    # 异常判定的最小节点数
    min_nodes_for_audit: int = 2


class NarrativeConfig:
    """叙事生成器配置"""

    language: str = "en"
    include_kinetic_integral: bool = True
    include_duration: bool = True
    include_missing_nodes: bool = True
    max_narrative_length: int = 1500


class DecisionConfig:
    """决策审计配置"""

    # VLLM 决策推理参数
    decision_max_tokens: int = 384
    decision_temperature: float = 0.0
    max_audit_entities: int = 8

    # 业务契约
    business_contracts: dict = {
        "retail store": [
            "拿起物品后，必须在离开前完成结账操作",
            "高速奔跑且携带物品，判定为可疑行为",
        ],
        "supermarket": [
            "拿起物品后，必须在离开前完成结账操作",
            "高速奔跑且携带物品，判定为可疑行为",
        ],
        "street": [
            "突然改变方向并加速奔跑，需要关注",
            "持续追赶他人，判定为攻击性行为",
            "多人围殴单人，判定为暴力行为",
        ],
        "parking lot": [
            "未经开锁直接进入车辆，判定为可疑行为",
            "长时间在多辆车之间徘徊，需要关注",
        ],
        "default": [
            "非休息区蹲下超过 10 秒 = 异常",
            "拿取物品后未结账即离开 = 异常",
            "动作序列出现不合逻辑的突变，需要审查",
            "出现暴力/纵火/持械行为，判定为异常",
        ],
    }

    # 异常置信度阈值
    anomaly_confidence_threshold: float = 0.5


# ── 系统级 ────────────────────────────────────────────
class SystemConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16_enabled: bool = True
    log_level: str = "INFO"
    version: str = "5.0"
