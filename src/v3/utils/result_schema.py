"""
V4.0 输出 Schema — analysis_result.json 结构定义与校验

定义:
  - V4.0 最终输出的 JSON 结构
  - 包含 anomaly_segments, anomaly_explanation, cinematic_filter
  - 提供 validate / build / format_time 等辅助函数

与需求文档对齐:
  {
    "status": "Anomaly Detected",
    "anomaly_segments": [
      {
        "start": "00:15.5",
        "end": "00:18.2",
        "confidence": 0.94,
        "reason": "..."
      }
    ],
    "cinematic_filter": "Negative (Real-world Event)"
  }
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── 时间格式化 ────────────────────────────────────────

def format_time(seconds: float) -> str:
    """
    将秒数格式化为 "mm:ss.f" 格式 (与需求文档对齐)。

    Examples:
        0.0    → "00:00.0"
        15.5   → "00:15.5"
        72.3   → "01:12.3"
        185.25 → "03:05.3"
    """
    if seconds < 0:
        seconds = 0.0
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:04.1f}"


def parse_time(time_str: str) -> float:
    """
    将 "mm:ss.f" 格式解析为秒数。

    Examples:
        "00:15.5" → 15.5
        "01:12.3" → 72.3
    """
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            mins = int(parts[0])
            secs = float(parts[1])
            return mins * 60 + secs
        return float(time_str)
    except (ValueError, AttributeError):
        return 0.0


# ── Schema 定义 ───────────────────────────────────────

# V4.0 anomaly_segments[i] 必填字段
SEGMENT_REQUIRED_FIELDS = {"start", "end", "confidence", "reason"}

# V4.0 顶层必填字段
RESULT_REQUIRED_FIELDS = {
    "status",
    "anomaly_segments",
    "cinematic_filter",
}

# 合法 status 值
VALID_STATUSES = {"Anomaly Detected", "Normal", "Error"}


# ── 构建 ─────────────────────────────────────────────

def build_segment(
    entity_id: int,
    start_sec: float,
    end_sec: float,
    confidence: float,
    reason: str,
    clip_path: str = "",
) -> dict:
    """
    构建单个 anomaly_segment 字典 (符合 V4.0 Schema)。
    """
    return {
        "entity_id": entity_id,
        "start": format_time(start_sec),
        "end": format_time(end_sec),
        "start_sec": round(start_sec, 2),
        "end_sec": round(end_sec, 2),
        "confidence": round(confidence, 2),
        "reason": reason,
        "clip_path": clip_path,
    }


def build_analysis_result(
    status: str = "Normal",
    anomaly_segments: Optional[list[dict]] = None,
    cinematic_filter: str = "Negative (Real-world Event)",
    anomaly_explanation: str = "",
    anomaly_explanation_en: str = "",
    video_path: str = "",
    video_name: str = "",
    anomaly_score: float = 0.0,
    duration_sec: float = 0.0,
    total_frames: int = 0,
    sampled_frames: int = 0,
    num_entities: int = 0,
    scene_type: str = "",
    mode: str = "v4",
    entity_results: Optional[list[dict]] = None,
    localization: Optional[dict] = None,
    graph_stats: Optional[dict] = None,
    pool_stats: Optional[dict] = None,
    processing_time_sec: float = 0.0,
) -> dict:
    """
    构建完整的 V4.0 analysis_result.json 字典。
    """
    return {
        # V4.0 核心三字段 (与需求文档对齐)
        "status": status,
        "anomaly_segments": anomaly_segments or [],
        "cinematic_filter": cinematic_filter,

        # 解释文本
        "anomaly_explanation": anomaly_explanation,
        "anomaly_explanation_en": anomaly_explanation_en,

        # 元数据
        "video_path": video_path,
        "video_name": video_name,
        "anomaly_score": round(anomaly_score, 4),
        "duration_sec": round(duration_sec, 2),
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "num_entities_tracked": num_entities,
        "scene_type": scene_type,
        "mode": mode,

        # 详情
        "entity_results": entity_results or [],

        # 定位
        "localization": localization or {
            "total_anomaly_duration_sec": 0.0,
            "num_segments": 0,
            "clips_saved": [],
        },

        # 统计
        "graph_stats": graph_stats or {},
        "pool_stats": pool_stats or {},
        "processing_time_sec": round(processing_time_sec, 2),
    }


# ── 校验 ─────────────────────────────────────────────

def validate_result(result: dict) -> tuple[bool, list[str]]:
    """
    校验 V4.0 输出结构是否完整。

    Returns:
        (is_valid, errors)
    """
    errors = []

    # 顶层必填字段
    for field in RESULT_REQUIRED_FIELDS:
        if field not in result:
            errors.append(f"Missing required field: '{field}'")

    # status 值检查
    status = result.get("status", "")
    if status and status not in VALID_STATUSES:
        errors.append(
            f"Invalid status '{status}'. "
            f"Must be one of: {VALID_STATUSES}"
        )

    # anomaly_segments 检查
    segments = result.get("anomaly_segments", [])
    if not isinstance(segments, list):
        errors.append("'anomaly_segments' must be a list")
    else:
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                errors.append(f"anomaly_segments[{i}] must be a dict")
                continue
            for field in SEGMENT_REQUIRED_FIELDS:
                if field not in seg:
                    errors.append(
                        f"anomaly_segments[{i}] missing field: '{field}'"
                    )

    # 一致性检查
    if status == "Anomaly Detected" and not segments:
        errors.append(
            "'status' is 'Anomaly Detected' but 'anomaly_segments' is empty"
        )

    if status == "Normal" and segments:
        errors.append(
            "'status' is 'Normal' but 'anomaly_segments' is non-empty"
        )

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning(f"Result validation failed: {errors}")

    return is_valid, errors


def compact_result(result: dict) -> dict:
    """
    生成精简版输出 (仅保留需求文档要求的核心三字段)。
    适用于 API 返回或轻量日志。
    """
    return {
        "status": result.get("status", "Error"),
        "anomaly_segments": result.get("anomaly_segments", []),
        "cinematic_filter": result.get("cinematic_filter", "N/A"),
    }
