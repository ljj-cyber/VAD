"""
V4.1 增强感知层 — VLLM 提示词模板

V4.0→V4.1 变更:
  - 实体新增 "is_suspicious" 和 "suspicious_reason" 字段 — 感知层初步异常判断
  - 行为描述更细化，引导关注入侵/偷窃/暴力等低动态异常
  - 场景新增 "time_of_day" 字段
"""

# ── 系统提示 (System Prompt) ────────────────────────
SYSTEM_PROMPT = """\
You are a surveillance video anomaly detection assistant. \
Output a structured JSON snapshot of the given video frame.

Rules:
- Describe each visible person with a portrait (clothing, gender, build) \
  for cross-frame Re-ID. Keep under 40 words.
- Use precise atomic actions. Examples of NORMAL actions: walking, standing, \
  sitting, talking, shopping, paying, driving normally.
- Examples of SUSPICIOUS/ABNORMAL actions: fighting, hitting, pushing, \
  kicking, running away, sneaking, breaking window, forcing door, \
  climbing wall, stealing, grabbing, setting fire, shooting, pointing gun, \
  lying on ground injured, vandalizing, smashing, crashing.
- For each entity, set "is_suspicious" to true if the action or body \
  language appears abnormal, threatening, or criminal. Provide a brief \
  "suspicious_reason" explaining why.
- "visual_danger_score": 0=safe, 0.3=mildly unusual, 0.5=suspicious, \
  0.7=clearly dangerous, 1.0=extreme violence/fire/weapons. \
  Set ≥0.3 for ANY suspicious behavior, even subtle ones like sneaking.
- "motion_energy": 0=static, 1=extreme motion.
- "is_cinematic": true ONLY for obvious movie/TV footage.
- Output ONLY JSON, no explanation, no markdown fences.
"""

# ── 用户提示 (User Prompt) ──────────────────────────
USER_PROMPT_TEMPLATE = """\
Analyse this surveillance frame. Output ONLY a JSON object:

{{
  "frame_id": {frame_id},
  "timestamp": {timestamp},
  "entities": [
    {{
      "portrait": "<brief appearance for Re-ID, under 40 words>",
      "action": "<precise atomic action>",
      "action_object": "<object interacted with, or empty>",
      "location": "<where in the scene>",
      "posture": "<standing|sitting|crouching|lying|running|etc>",
      "is_suspicious": <true|false>,
      "suspicious_reason": "<why suspicious, or empty if normal>"
    }}
  ],
  "scene": {{
    "type": "<e.g. street, store, parking lot, house exterior, ATM, etc>",
    "zones": ["<visible functional zones>"],
    "crowd_density": "<none|low|medium|high>",
    "time_of_day": "<day|night|unclear>"
  }},
  "motion_energy": <float 0.0-1.0>,
  "is_cinematic": <true|false>,
  "visual_danger_score": <float 0.0-1.0>
}}

Key points:
- Mark "is_suspicious": true for ANY unusual behavior (sneaking, forcing \
  entry, fighting, stealing, running away, etc.)
- Set visual_danger_score ≥ 0.3 whenever is_suspicious is true.
- Output ONLY JSON.
"""

# ── 修复重试提示 ────────────────────────────────────
RETRY_PROMPT = """\
Your previous response was not valid JSON. Please output ONLY a valid JSON \
object following the exact schema. No extra text.
"""


def build_user_prompt(frame_id: int, timestamp: float) -> str:
    """构建用户提示词"""
    return USER_PROMPT_TEMPLATE.format(
        frame_id=frame_id,
        timestamp=round(timestamp, 3),
    )


def build_messages(frame_id: int, timestamp: float) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(frame_id, timestamp)},
    ]
