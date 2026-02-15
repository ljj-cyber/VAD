"""
V4.0 JSON Schema 校验工具 — 验证 VLLM 输出的结构化 JSON 快照

V3→V4 变更:
  - 新增 is_cinematic (bool) 字段
  - 新增 visual_danger_score (float) 字段
  - 清洗逻辑增加对新字段的处理
"""
import json
import re
from typing import Any, Optional


# ── VLLM 输出 JSON Schema 定义 ───────────────────────
ENTITY_SCHEMA = {
    "required": ["portrait", "action"],
    "optional": ["action_object", "location", "posture"],
    "types": {
        "portrait": str,
        "action": str,
        "action_object": str,
        "location": str,
        "posture": str,
    },
}

SCENE_SCHEMA = {
    "required": [],
    "optional": ["type", "zones", "crowd_density"],
    "types": {
        "type": str,
        "zones": list,
        "crowd_density": str,
    },
}

FRAME_SNAPSHOT_SCHEMA = {
    "required": ["entities"],
    "optional": [
        "frame_id", "timestamp", "scene", "motion_energy",
        "is_cinematic", "visual_danger_score",                   # V4.0 新增
    ],
    "types": {
        "frame_id": int,
        "timestamp": (int, float),
        "entities": list,
        "scene": dict,
        "motion_energy": (int, float),
        "is_cinematic": bool,                                    # V4.0 新增
        "visual_danger_score": (int, float),                     # V4.0 新增
    },
}


def validate_entity(entity: dict) -> tuple[bool, list[str]]:
    """
    验证单个实体的 JSON 结构

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    if not isinstance(entity, dict):
        return False, ["entity is not a dict"]

    for field in ENTITY_SCHEMA["required"]:
        if field not in entity:
            errors.append(f"missing required field: {field}")
        elif not isinstance(entity[field], ENTITY_SCHEMA["types"][field]):
            errors.append(
                f"field '{field}' type mismatch: expected {ENTITY_SCHEMA['types'][field].__name__}, "
                f"got {type(entity[field]).__name__}"
            )

    for field in ENTITY_SCHEMA["optional"]:
        if field in entity and entity[field] is not None:
            expected = ENTITY_SCHEMA["types"][field]
            if not isinstance(entity[field], expected):
                errors.append(
                    f"optional field '{field}' type mismatch: expected {expected.__name__}, "
                    f"got {type(entity[field]).__name__}"
                )

    return len(errors) == 0, errors


def validate_snapshot(snapshot: dict) -> tuple[bool, list[str]]:
    """
    验证完整帧快照的 JSON 结构

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    if not isinstance(snapshot, dict):
        return False, ["snapshot is not a dict"]

    # 检查顶层必需字段
    for field in FRAME_SNAPSHOT_SCHEMA["required"]:
        if field not in snapshot:
            errors.append(f"missing required field: {field}")

    # 类型检查
    for field, expected_type in FRAME_SNAPSHOT_SCHEMA["types"].items():
        if field in snapshot and snapshot[field] is not None:
            if not isinstance(snapshot[field], expected_type):
                errors.append(
                    f"field '{field}' type mismatch: expected {expected_type}, "
                    f"got {type(snapshot[field])}"
                )

    # 验证 entities 数组
    if "entities" in snapshot and isinstance(snapshot["entities"], list):
        for i, entity in enumerate(snapshot["entities"]):
            valid, entity_errors = validate_entity(entity)
            if not valid:
                errors.extend([f"entities[{i}]: {e}" for e in entity_errors])

    # 验证 scene (如果存在)
    if "scene" in snapshot and isinstance(snapshot["scene"], dict):
        scene = snapshot["scene"]
        for field, expected_type in SCENE_SCHEMA["types"].items():
            if field in scene and scene[field] is not None:
                if not isinstance(scene[field], expected_type):
                    errors.append(
                        f"scene.{field} type mismatch: expected {expected_type}, "
                        f"got {type(scene[field])}"
                    )

    # V4.0: 验证 is_cinematic
    if "is_cinematic" in snapshot:
        val = snapshot["is_cinematic"]
        if not isinstance(val, bool):
            # 允许 0/1 或 "true"/"false" 的宽松解析
            if val not in (0, 1, "true", "false", "True", "False"):
                errors.append(
                    f"field 'is_cinematic' should be bool, got {type(val).__name__}: {val}"
                )

    # V4.0: 验证 visual_danger_score 范围
    if "visual_danger_score" in snapshot:
        val = snapshot["visual_danger_score"]
        if isinstance(val, (int, float)):
            if not (0.0 <= val <= 1.0):
                errors.append(
                    f"field 'visual_danger_score' out of range [0,1]: {val}"
                )

    return len(errors) == 0, errors


def extract_json_from_response(response: str) -> Optional[dict]:
    """
    从 VLLM 原始文本响应中提取 JSON 对象。
    支持多种格式: 纯 JSON、markdown 代码块、混合文本等。

    Returns:
        解析后的 dict，或 None
    """
    if not response or not isinstance(response, str):
        return None

    response = response.strip()

    # 策略 1: 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 策略 2: 提取 markdown 代码块中的 JSON
    code_block_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL
    )
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 策略 3: 提取最外层的 { ... } 块
    brace_match = re.search(r"\{.*\}", response, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 策略 3b: 修复常见 JSON 问题（尾部逗号、单引号、非法数值）
        fixed = candidate
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # 移除尾部逗号
        fixed = fixed.replace("'", '"')  # 单引号替换
        # 修复 "break_timestamp": 01:01.33 → "break_timestamp": "01:01.33"
        fixed = re.sub(
            r'"break_timestamp"\s*:\s*(\d{2}:\d{2}(?:\.\d+)?)',
            r'"break_timestamp": "\1"',
            fixed,
        )
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # 策略 4: 修复截断的 JSON（token 限制导致输出不完整）
    # 找到以 { 开头的内容，尝试补全缺失的括号
    brace_start = response.find("{")
    if brace_start >= 0:
        truncated = response[brace_start:]
        # 移除最后一个不完整的字符串值（截断处）
        truncated = re.sub(r',\s*"[^"]*":\s*"[^"]*$', '', truncated)  # 截断的 key-value
        truncated = re.sub(r',\s*\{[^}]*$', '', truncated)  # 截断的对象
        truncated = re.sub(r',\s*"[^"]*$', '', truncated)   # 截断的字符串
        truncated = re.sub(r",\s*$", "", truncated)          # 末尾逗号

        # 补全缺失的括号
        open_braces = truncated.count("{") - truncated.count("}")
        open_brackets = truncated.count("[") - truncated.count("]")
        truncated += "]" * max(0, open_brackets)
        truncated += "}" * max(0, open_braces)

        # 再次清理尾部逗号
        truncated = re.sub(r",\s*([}\]])", r"\1", truncated)

        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    return None


def sanitize_snapshot(raw: dict) -> dict:
    """
    清洗和标准化快照数据 (V4.0):
    - 确保 entities 是列表
    - 填充缺失的可选字段为默认值
    - 截断过长的文本
    - 处理 V4.0 新增字段: is_cinematic, visual_danger_score
    """
    snapshot = {}

    snapshot["frame_id"] = raw.get("frame_id", -1)
    snapshot["timestamp"] = raw.get("timestamp", -1.0)

    # 处理 entities
    raw_entities = raw.get("entities", [])
    if not isinstance(raw_entities, list):
        raw_entities = [raw_entities] if isinstance(raw_entities, dict) else []

    entities = []
    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        clean_ent = {
            "portrait": str(ent.get("portrait", "unknown"))[:200],
            "action": str(ent.get("action", "unknown"))[:100],
            "action_object": str(ent.get("action_object", ""))[:100] if ent.get("action_object") else "",
            "location": str(ent.get("location", ""))[:100] if ent.get("location") else "",
            "posture": str(ent.get("posture", ""))[:100] if ent.get("posture") else "",
            "is_suspicious": bool(ent.get("is_suspicious", False)),
            "suspicious_reason": str(ent.get("suspicious_reason", ""))[:200] if ent.get("suspicious_reason") else "",
        }
        entities.append(clean_ent)

    snapshot["entities"] = entities

    # 处理 scene
    raw_scene = raw.get("scene", {})
    if isinstance(raw_scene, dict):
        snapshot["scene"] = {
            "type": str(raw_scene.get("type", ""))[:100],
            "zones": raw_scene.get("zones", []) if isinstance(raw_scene.get("zones"), list) else [],
            "crowd_density": str(raw_scene.get("crowd_density", ""))[:50],
        }
    else:
        snapshot["scene"] = {"type": "", "zones": [], "crowd_density": ""}

    snapshot["motion_energy"] = float(raw.get("motion_energy", 0.0))

    # ── V4.0 新增字段 ──
    # is_cinematic: 宽松解析 bool
    raw_cinematic = raw.get("is_cinematic", False)
    if isinstance(raw_cinematic, bool):
        snapshot["is_cinematic"] = raw_cinematic
    elif isinstance(raw_cinematic, (int, float)):
        snapshot["is_cinematic"] = bool(raw_cinematic)
    elif isinstance(raw_cinematic, str):
        snapshot["is_cinematic"] = raw_cinematic.lower() in ("true", "1", "yes")
    else:
        snapshot["is_cinematic"] = False

    # visual_danger_score: 确保 [0, 1]
    raw_danger = raw.get("visual_danger_score", 0.0)
    try:
        danger_val = float(raw_danger)
        snapshot["visual_danger_score"] = max(0.0, min(1.0, danger_val))
    except (TypeError, ValueError):
        snapshot["visual_danger_score"] = 0.0

    return snapshot
