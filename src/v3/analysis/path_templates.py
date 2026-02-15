"""
V3.0 路径分析层 — 正常行为路径模板

系统维护一组"正常行为路径模板"，每个模板是一个有序动作序列。
通过计算实体在时间图中的实际路径与模板的匹配度来判断是否正常。

模板设计原则:
  1. 每个模板代表一种常见的正常行为模式。
  2. 模板使用原子动作粒度，与 VLLM 输出对齐。
  3. 允许模糊匹配（动作语义相似即可）。
  4. 模板可按场景类型索引（零售店、街道、停车场等）。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class PathTemplate:
    """正常行为路径模板"""

    name: str                          # 模板名称
    scene_types: list[str]             # 适用的场景类型
    action_sequence: list[str]         # 有序动作序列
    optional_actions: list[str] = field(default_factory=list)  # 可选动作（允许出现也允许缺失）
    description: str = ""              # 模板描述
    weight: float = 1.0                # 模板重要性权重


# ── 预定义正常行为模板库 ────────────────────────────
NORMAL_PATH_TEMPLATES = [
    # ── 零售场景 ──────────────────────────────────
    PathTemplate(
        name="normal_shopping",
        scene_types=["retail store", "supermarket", "convenience store", "shop"],
        action_sequence=[
            "entering", "walking", "browsing", "picking up",
            "holding", "walking", "handing over", "paying", "leaving",
        ],
        optional_actions=["putting down", "standing still", "looking at"],
        description="Standard shopping: enter → browse → pick up → checkout → leave",
    ),
    PathTemplate(
        name="quick_purchase",
        scene_types=["retail store", "supermarket", "convenience store", "shop"],
        action_sequence=[
            "entering", "walking", "picking up", "holding",
            "paying", "leaving",
        ],
        optional_actions=["standing still"],
        description="Quick buy: enter → grab → pay → leave",
    ),
    PathTemplate(
        name="browsing_only",
        scene_types=["retail store", "supermarket", "convenience store", "shop"],
        action_sequence=[
            "entering", "walking", "browsing", "walking", "leaving",
        ],
        optional_actions=["standing still", "looking at"],
        description="Browse without buying: enter → look around → leave",
    ),

    # ── 街道场景 ──────────────────────────────────
    PathTemplate(
        name="normal_pedestrian",
        scene_types=["street", "sidewalk", "road", "crosswalk"],
        action_sequence=["walking", "walking", "walking"],
        optional_actions=["standing still", "looking at", "crossing"],
        description="Normal pedestrian walking along the street",
    ),
    PathTemplate(
        name="waiting_crossing",
        scene_types=["street", "crosswalk", "intersection"],
        action_sequence=["walking", "standing still", "walking"],
        optional_actions=["looking at"],
        description="Pedestrian waiting at crosswalk then crossing",
    ),

    # ── 停车场场景 ────────────────────────────────
    PathTemplate(
        name="parking_normal",
        scene_types=["parking lot", "garage", "parking"],
        action_sequence=[
            "walking", "approaching vehicle", "opening door",
            "entering vehicle", "driving",
        ],
        optional_actions=["standing still", "holding", "putting down"],
        description="Normal parking lot behavior: walk to car → get in → drive away",
    ),

    # ── 通用场景 ──────────────────────────────────
    PathTemplate(
        name="standing_conversation",
        scene_types=["*"],  # 适用于所有场景
        action_sequence=["standing still", "talking", "standing still"],
        optional_actions=["gesturing", "nodding"],
        description="Two people having a conversation",
    ),
    PathTemplate(
        name="sitting_activity",
        scene_types=["*"],
        action_sequence=["walking", "sitting", "sitting", "standing still", "walking"],
        optional_actions=["eating", "drinking", "reading", "using phone"],
        description="Normal sitting activity",
    ),
]


def _action_similarity(action_a: str, action_b: str) -> float:
    """
    计算两个动作的语义相似度（简单规则匹配）。

    Returns:
        [0, 1] 相似度分数
    """
    a = action_a.lower().strip()
    b = action_b.lower().strip()

    # 完全匹配
    if a == b:
        return 1.0

    # 一个包含另一个
    if a in b or b in a:
        return 0.85

    # 同义词映射
    synonyms = {
        "walking": {"moving", "strolling", "pacing"},
        "running": {"sprinting", "rushing", "dashing", "fleeing"},
        "standing still": {"stationary", "idle", "waiting", "standing"},
        "picking up": {"grabbing", "taking", "grasping", "snatching"},
        "putting down": {"placing", "setting down", "dropping"},
        "holding": {"carrying", "gripping"},
        "handing over": {"giving", "passing", "delivering", "transferring"},
        "paying": {"checkout", "purchasing", "buying"},
        "browsing": {"looking", "examining", "inspecting", "viewing"},
        "entering": {"coming in", "arriving"},
        "leaving": {"exiting", "departing", "going out", "walking away"},
        "sitting": {"seated", "sat down"},
        "fighting": {"attacking", "hitting", "punching", "assaulting"},
        "threatening": {"menacing", "intimidating", "brandishing"},
    }

    a_group = None
    b_group = None
    for canonical, syns in synonyms.items():
        all_forms = syns | {canonical}
        if a in all_forms or any(s in a for s in all_forms):
            a_group = canonical
        if b in all_forms or any(s in b for s in all_forms):
            b_group = canonical

    if a_group and b_group and a_group == b_group:
        return 0.8

    return 0.0


def match_path_to_template(
    action_sequence: list[str],
    template: PathTemplate,
) -> float:
    """
    计算实体动作序列与模板的匹配度。

    使用 Needleman-Wunsch 风格的动态规划序列对齐，
    允许跳过模板中的可选动作。

    Args:
        action_sequence: 实体的实际动作序列
        template: 路径模板

    Returns:
        [0, 1] 匹配度分数
    """
    seq = action_sequence
    tmpl = template.action_sequence
    optional = set(a.lower() for a in template.optional_actions)

    if not seq or not tmpl:
        return 0.0

    n = len(seq)
    m = len(tmpl)

    # DP 对齐
    # dp[i][j] = 将 seq[:i] 与 tmpl[:j] 对齐的最大分数
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)

    # 初始化：跳过模板前缀的代价
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - 0.1  # 小惩罚

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - 0.2  # 实际序列多出的动作
        for j in range(1, m + 1):
            # 选项 1：对齐 seq[i-1] 和 tmpl[j-1]
            sim = _action_similarity(seq[i - 1], tmpl[j - 1])
            match_score = dp[i - 1][j - 1] + sim

            # 选项 2：跳过实际序列中的动作
            skip_actual = dp[i - 1][j] - 0.15
            # 如果是可选动作，惩罚更小
            if seq[i - 1].lower() in optional:
                skip_actual = dp[i - 1][j] - 0.05

            # 选项 3：跳过模板中的动作
            skip_template = dp[i][j - 1] - 0.1

            dp[i][j] = max(match_score, skip_actual, skip_template)

    # 归一化分数到 [0, 1]
    raw_score = dp[n][m]
    max_possible = min(n, m)  # 最多能匹配的数量
    if max_possible == 0:
        return 0.0

    normalized = max(0.0, raw_score / max_possible)
    return min(1.0, normalized)


def find_best_matching_template(
    action_sequence: list[str],
    scene_type: str = "",
    templates: Optional[list[PathTemplate]] = None,
) -> tuple[Optional[PathTemplate], float]:
    """
    在模板库中找到与动作序列最匹配的模板。

    Args:
        action_sequence: 实体的实际动作序列
        scene_type: 当前场景类型（用于过滤模板）
        templates: 自定义模板列表（默认使用预定义模板）

    Returns:
        (best_template, best_score): 最佳模板和匹配度
    """
    if templates is None:
        templates = NORMAL_PATH_TEMPLATES

    best_template = None
    best_score = 0.0

    scene_lower = scene_type.lower().strip()

    for tmpl in templates:
        # 场景过滤
        if tmpl.scene_types != ["*"]:
            scene_match = any(
                st.lower() in scene_lower or scene_lower in st.lower()
                for st in tmpl.scene_types
            )
            if not scene_match and scene_lower:
                continue

        score = match_path_to_template(action_sequence, tmpl)

        if score > best_score:
            best_score = score
            best_template = tmpl

    return best_template, best_score


def compute_path_anomaly_score(
    action_sequence: list[str],
    scene_type: str = "",
    cfg: Optional[AnalysisConfig] = None,
) -> tuple[float, str, Optional[str]]:
    """
    基于路径模板匹配计算异常分数。

    Args:
        action_sequence: 实体的实际动作序列
        scene_type: 场景类型
        cfg: 分析配置

    Returns:
        (anomaly_score, reason, matched_template_name):
          - anomaly_score: [0, 1]，越高越异常
          - reason: 判定原因
          - matched_template_name: 匹配到的模板名称（如果有）
    """
    cfg = cfg or AnalysisConfig()

    if len(action_sequence) < cfg.min_path_length:
        return 0.0, "path too short for analysis", None

    best_template, best_score = find_best_matching_template(
        action_sequence, scene_type
    )

    if best_template is None:
        return 0.5, "no matching template found", None

    # 匹配度越高 → 异常分数越低
    anomaly_score = 1.0 - best_score

    if best_score >= cfg.path_match_threshold:
        reason = (
            f"matches template '{best_template.name}' "
            f"(score={best_score:.3f})"
        )
    else:
        reason = (
            f"poor match to best template '{best_template.name}' "
            f"(score={best_score:.3f} < threshold={cfg.path_match_threshold})"
        )

    return anomaly_score, reason, best_template.name
