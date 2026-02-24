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
    # 自适应阈值回退：检测不到时自动降低 (最低阈值)
    adaptive_threshold_min: int = 8
    # 形态学核大小 (去噪)
    morph_kernel_size: int = 5
    # 高斯模糊核 (预处理)
    blur_kernel_size: int = 5
    # 连通域最小面积 (像素数)，低于此值忽略
    min_region_area: int = 1500
    # 提取 Top-K 动能连通域 (增大以覆盖更多实体)
    top_k_regions: int = 4
    # Crop padding 比例 (bbox 外扩)
    crop_padding_ratio: float = 0.15
    # 最小 crop 尺寸 (像素)
    min_crop_size: int = 80
    # 多帧累积缓冲区大小 (用于捕捉慢速运动)
    accumulate_frames: int = 3
    # 连续无检出帧数阈值 — 超过后自动降低阈值
    empty_streak_for_fallback: int = 30


class YoloDetectorConfig:
    """YOLO-World 开放词汇目标检测配置 (性能优化版)"""

    # 模型权重路径 (项目内 models/yolo/ 下)
    # 可选: yolov8s-worldv2, yolov8m-worldv2, yolov8l-worldv2
    model_name: str = str(MODELS_DIR / "yolo" / "yolov8l-worldv2.pt")

    # 开放词汇类别 — 监控异常检测场景
    # ★ 优化: 精简冗余同义词 + 隐蔽异常相关目标 (11→16)
    #   - person/people/crowd 合并为 person (YOLO-World 对每个类别都要计算文本嵌入)
    #   - fire/flame 合并为 fire
    #   - car/truck/bus/motorcycle/bicycle/vehicle 精简为 car, truck, motorcycle
    #   - 新增: 破坏工具/痕迹 (vandalism), 斗殴姿态相关
    classes: list = [
        "person",
        "fire", "smoke",
        "car", "truck", "motorcycle",
        "knife", "gun",
        "explosion", "blood",
        "bag",
        # 隐蔽异常相关
        "bat", "hammer", "crowbar",     # 破坏工具
        "broken glass",                  # 破坏痕迹
        "spray can",                     # 涂鸦工具
    ]

    # 推理参数
    confidence_threshold: float = 0.15   # 低阈值，宁可多检不漏
    iou_threshold: float = 0.45          # NMS IoU 阈值
    max_det: int = 10                    # 每帧最大检测数
    imgsz: int = 480                     # ★ 推理输入尺寸 (640→480，提速~40%，监控场景够用)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    half: bool = True                    # ★ FP16 半精度推理 (CUDA 上提速 40-60%)

    # 调用频率: 每 N 帧调用一次 YOLO (节省开销)
    detect_every_n: int = 5              # ★ 3→5, 降低调用密度
    # 当帧差检测到运动时，是否强制调用 YOLO
    force_on_motion: bool = True
    # ★ force 模式的最小帧间隔 (防止每帧都调 YOLO)
    force_min_gap: int = 2


class HybridDetectorConfig:
    """帧差 + YOLO 融合检测配置"""

    # 融合模式: "union" (取并集) | "yolo_primary" (YOLO 为主，帧差补充)
    fusion_mode: str = "union"

    # IoU 阈值: 帧差区域与 YOLO 框重叠超过此值时视为同一目标
    merge_iou_threshold: float = 0.3

    # YOLO 独有检测（帧差未检出）的最低置信度
    yolo_only_min_conf: float = 0.20

    # 帧差独有检测（YOLO 未检出）的最低动能
    motion_only_min_energy: float = 0.04

    # ── Lazy YOLO Fallback（帧差优先 + YOLO 按需激活）──
    # True: 帧差法为主，仅在连续空检出时激活 YOLO (省 GPU，适合日常)
    # False: 帧差 + YOLO 全程并行 (全覆盖，适合精度优先)
    lazy_yolo: bool = True

    # 连续空检出帧数阈值 — 超过后激活 YOLO fallback
    empty_streak_for_yolo_fallback: int = 15

    # YOLO 激活后的冷却帧数 — 激活后持续运行 YOLO N 帧，避免频繁开关
    yolo_cooldown_frames: int = 30


class CLIPEncoderConfig:
    """CLIP Crop 特征提取配置"""

    model_name: str = "openai/clip-vit-base-patch32"
    feature_dim: int = 512
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TrackerConfig:
    """Entity Tracker 贪婪匹配配置"""

    # 余弦相似度阈值：高于此值 → 沿用旧 ID
    # 降低阈值以减少实体碎片化 (0.82 → 0.72)
    similarity_threshold: float = 0.72
    # 最大允许的帧间隔（超过则 Entity 死亡）
    # 增大以让实体存活更久 (30 → 150)
    max_age_frames: int = 150
    # 最大活跃实体数
    max_active_entities: int = 30
    # 新 Entity 的最小动能 (低于此值不分配新 ID)
    # 提高以减少噪声实体 (0.04 → 0.06)
    min_kinetic_for_new: float = 0.06


# ── Stage 2: 稀疏语义挂载 ────────────────────────────
class NodeTriggerConfig:
    """语义节点触发策略"""

    # Rule 1: Birth — 新 ID 出现即触发
    trigger_on_birth: bool = True
    # Rule 2: Change Point — 与上次采样的 embedding 距离阈值
    # 降低以对火焰/烟雾等渐变信号更敏感 (0.30 → 0.22)
    embedding_jump_threshold: float = 0.22  # 1 - cosine_similarity
    # Rule 3: Heartbeat — 距上次采样最大间隔 (秒)
    heartbeat_interval_sec: float = 3.0
    # 最大触发频率保护 (同一实体相邻触发的最小间隔帧数)
    min_trigger_gap_frames: int = 5


class SemanticVLLMConfig:
    """VLLM 语义推理 (仅对 crop 区域)"""

    # VLLM model
    model_name: str = "qwen2.5-vl-32b"
    MODEL_PATHS: dict = {
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2.5-vl-7b": "/data/liuzhe/Qwen2.5-VL-7B-Instruct",
        "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    }

    # 本地模型路径 (用于 backend="local")
    LOCAL_MODEL_PATHS: dict = {
        "qwen2.5-vl-7b": "/data/liuzhe/Qwen2.5-VL-7B-Instruct",
        "qwen2-vl-7b": str(HF_CACHE_DIR / "hub" / "models--Qwen--Qwen2-VL-7B-Instruct"),
        "qwen2.5-vl-32b": str(HF_CACHE_DIR / "hub" / "models--Qwen--Qwen2.5-VL-32B-Instruct"),
    }

    # API server
    backend: str = "server"
    api_base: str = "http://localhost:8000"
    max_workers: int = 16

    # 推理参数
    max_new_tokens: int = 512
    temperature: float = 0.0
    max_retries: int = 2

    # Crop 尺寸 (resize 后喂给 VLLM)
    crop_resize: tuple = (336, 336)


# ── Stage 3: 动态图审计 ──────────────────────────────
class GraphConfig:
    """时间演化图配置"""

    # 最大图节点数 (每实体)
    max_nodes_per_entity: int = 100
    # 异常判定的最小节点数 (降为 1，单节点实体也需审计)
    min_nodes_for_audit: int = 1


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
    # ★ 减少 decision_max_tokens 加速推理 (384 → 256)
    decision_max_tokens: int = 256
    decision_temperature: float = 0.0
    # 增大审计容量以覆盖更多实体 (8 → 12)
    max_audit_entities: int = 12

    # 业务契约 — 场景级先验规则，注入 Decision Prompt
    business_contracts: dict = {
        "default": [],
        "store": [
            "A person picking up merchandise AND THEN either (a) concealing it "
            "(hiding in clothing, bag, or pocket), or (b) bypassing all checkout "
            "counters and moving toward the store exit without paying = Shoplifting.",
            "IMPORTANT: Normal shopping behaviors (examining items, carrying items "
            "between aisles, interacting at checkout counters, using POS terminals, "
            "or working as store staff) are NOT shoplifting. Only flag if there is "
            "clear evidence of concealment OR deliberate exit without payment.",
        ],
        "supermarket": [
            "A person picking up merchandise AND THEN either (a) concealing it "
            "(hiding in clothing, bag, or pocket), or (b) bypassing all checkout "
            "counters and moving toward the store exit without paying = Shoplifting.",
            "IMPORTANT: Normal shopping behaviors (examining items, carrying items "
            "between aisles, interacting at checkout counters, using POS terminals, "
            "or working as store staff) are NOT shoplifting. Only flag if there is "
            "clear evidence of concealment OR deliberate exit without payment.",
        ],
        "shop": [
            "A person picking up merchandise and then either concealing it or "
            "exiting the shop without payment = Shoplifting.",
            "IMPORTANT: Normal shopping behaviors are NOT shoplifting.",
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
