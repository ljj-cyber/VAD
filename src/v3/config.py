"""
V4.0 语义因果决策异常检测系统 — 全局配置

升级要点 (V3→V4):
  - 感知层增加 is_cinematic / visual_danger_score 字段
  - 新增叙事引擎配置 (NarrativeConfig)
  - 新增决策审计层配置 (DecisionConfig)
  - 新增时间定位配置 (LocalizationConfig)
  - AnalysisConfig 扩展为面向 Decision LLM 的审计模式
"""
import os
import pathlib
import torch

# ── 路径 ─────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # EventVAD/
SRC_DIR = PROJECT_ROOT / "src"
V3_DIR = SRC_DIR / "v3"
MODELS_DIR = PROJECT_ROOT / "models"
VIDEOS_DIR = SRC_DIR / "event_seg" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "output" / "v4"

# Hugging Face 缓存目录
HF_CACHE_DIR = MODELS_DIR / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))


class PerceptionConfig:
    """增强感知层配置 (V4.0)"""

    # VLLM 模型选择: "qwen2-vl-7b", "moondream2"
    model_name: str = "qwen2-vl-7b"

    # 模型路径映射
    MODEL_PATHS = {
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "moondream2": "vikhyatk/moondream2",
    }

    # 推理参数
    max_new_tokens: int = 2048
    temperature: float = 0.1
    do_sample: bool = False
    max_retries: int = 3

    # 图片尺寸（喂给 VLLM 的帧大小）
    frame_size: tuple = (448, 448)

    # 注意力实现
    attn_implementation: str = "eager"

    # ── V4.0 新增 ──
    # 视觉锚点: 是否保存关键帧缩略图路径
    save_visual_anchors: bool = True
    visual_anchor_size: tuple = (224, 224)   # 缩略图尺寸

    # 高危险信号阈值 — 超过此值触发 Decision LLM 审计
    danger_trigger_threshold: float = 0.6


class SamplerConfig:
    """自适应采样器配置"""

    base_fps: float = 0.5
    max_fps: float = 3.0
    energy_threshold_low: float = 0.02
    energy_threshold_high: float = 0.15
    energy_window: int = 5
    min_interval_frames: int = 1


class AssociationConfig:
    """时间关联层配置"""

    # 语义 Re-ID
    sbert_model: str = "all-MiniLM-L6-v2"
    reid_similarity_threshold: float = 0.55
    entity_pool_max_age: int = 60
    entity_pool_max_size: int = 50

    # 时间边权重
    portrait_weight: float = 0.4
    action_weight: float = 0.6
    max_time_gap: float = 30.0

    # 图参数
    graph_propagation_iters: int = 2


class NarrativeConfig:
    """叙事引擎配置 (V4.0 新增)"""

    # 叙事文本语言
    language: str = "zh"                      # "zh" | "en"
    # 叙事最大长度 (字符)
    max_narrative_length: int = 2000
    # 是否包含运动能量信息
    include_energy: bool = True
    # 是否包含外观变化信息
    include_portrait_diff: bool = True


class DecisionConfig:
    """决策审计层配置 (V4.0 核心)"""

    # Decision LLM 后端: "qwen-max" | "gpt-4" | "local-qwen"
    decision_backend: str = "local-qwen"

    # API 配置（当使用远程 API 时）
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = "qwen-max"
    api_timeout: int = 60

    # 本地模型配置（当使用本地模型时）
    local_model_path: str = "Qwen/Qwen2-VL-7B-Instruct"

    # 审计触发条件
    trigger_on_path_change: bool = True       # 图路径长度变化时触发
    trigger_on_danger: bool = True            # 感知层高危信号触发
    danger_threshold: float = 0.6             # 高危信号阈值
    min_path_length_for_audit: int = 1        # 最小路径长度才启动审计 (1=审计所有实体)

    # 业务契约 (场景→规则映射)
    # 示例: 超市里 "拿起" 之后必须有 "结账" 才能 "离开"
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
        ],
        "parking lot": [
            "未经开锁直接进入车辆，判定为可疑行为",
            "长时间在多辆车之间徘徊，需要关注",
        ],
        "default": [
            "动作序列出现不合逻辑的突变，需要审查",
        ],
    }

    # Decision LLM 推理参数
    decision_max_tokens: int = 384
    decision_temperature: float = 0.2
    max_audit_entities: int = 8

    # 电影特征判据 — 用于过滤伪信号
    cinematic_filter_enabled: bool = True


class LocalizationConfig:
    """精准片段划分配置 (V4.0 新增)"""

    # 嵌入位移检测
    embedding_shift_threshold: float = 0.4    # 嵌入向量位移阈值
    embedding_shift_window: int = 3           # 回溯搜索窗口 (节点数)

    # 动能微调
    energy_slope_threshold: float = 0.05      # 动能斜率突变阈值
    energy_search_radius_sec: float = 2.0     # 在逻辑起点附近搜索范围 (秒)

    # 片段参数
    min_segment_duration_sec: float = 1.0     # 最短异常片段 (秒)
    max_segment_duration_sec: float = 300.0   # 最长异常片段 (秒), 合并后可能很长
    segment_padding_sec: float = 4.0          # 片段前后填充 (秒), 提升片段覆盖率与IoU

    # 异常片段保存
    save_anomaly_clips: bool = True
    clip_output_dir: str = ""                 # 空则使用 OUTPUT_DIR / video_name / clips


class AnalysisConfig:
    """路径分析与异常判定配置 (V4.0 重构)"""

    # 路径匹配 (仅用于叙事引擎参考，不再作为最终判定)
    path_match_threshold: float = 0.6
    breakage_threshold: float = 0.3

    # V4.0: 异常判定由 Decision LLM 接管
    # 以下为兼容参数，用于 fallback 模式
    anomaly_ema_alpha: float = 0.3
    min_path_length: int = 3

    # V4.0: Decision LLM 置信度阈值
    decision_confidence_threshold: float = 0.7


class SystemConfig:
    """系统级配置"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    fp16_enabled: bool = True
    num_workers: int = 8
    batch_size: int = 4

    # 日志
    log_level: str = "INFO"
    save_intermediate: bool = True

    # V4.0 版本标记
    version: str = "4.0"
