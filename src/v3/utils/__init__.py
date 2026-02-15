"""
V4.0 工具模块

- json_schema: JSON 校验与清洗工具
- result_schema: V4.0 输出 Schema 定义与校验
"""

from .json_schema import extract_json_from_response, validate_snapshot, sanitize_snapshot
from .result_schema import (
    format_time,
    parse_time,
    build_segment,
    build_analysis_result,
    validate_result,
    compact_result,
)

__all__ = [
    "extract_json_from_response",
    "validate_snapshot",
    "sanitize_snapshot",
    "format_time",
    "parse_time",
    "build_segment",
    "build_analysis_result",
    "validate_result",
    "compact_result",
]
