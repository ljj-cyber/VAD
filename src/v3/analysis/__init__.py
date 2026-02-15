"""
V4.0 路径分析与异常判定层

- AnomalyDetector: 多信号融合异常检测器 (V3 fallback)
- path_templates: 正常行为路径模板库
- NarrativeEngine: V4.0 叙事引擎 — 图路径转因果叙事文本
- CausalityAuditor: V4.0 决策审计器 — Decision LLM 语义因果审计
- TemporalLocalizer: V4.0 精准片段划分 — 回溯定位+动能微调+片段切分
"""

from .anomaly_detector import AnomalyDetector, AnomalyResult, VideoAnomalyResult
from .path_templates import (
    PathTemplate,
    NORMAL_PATH_TEMPLATES,
    find_best_matching_template,
    compute_path_anomaly_score,
)
from .narrative_engine import NarrativeEngine
from .causality_auditor import CausalityAuditor, AuditVerdict, VideoAuditReport
from .temporal_localizer import TemporalLocalizer, AnomalySegment, LocalizationResult

__all__ = [
    "AnomalyDetector",
    "AnomalyResult",
    "VideoAnomalyResult",
    "PathTemplate",
    "NORMAL_PATH_TEMPLATES",
    "find_best_matching_template",
    "compute_path_anomaly_score",
    "NarrativeEngine",
    "CausalityAuditor",
    "AuditVerdict",
    "VideoAuditReport",
    "TemporalLocalizer",
    "AnomalySegment",
    "LocalizationResult",
]
