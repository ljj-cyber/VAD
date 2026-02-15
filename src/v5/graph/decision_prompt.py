"""
Stage 3-D: Decision Prompt — 决策审计 + 契约注入

核心功能:
  1. 接收叙事文本 (NarrativeGenerator 输出)
  2. 注入"工业/社会契约" (场景规则)
  3. 调用 Decision LLM 进行最终异常判定
  4. 返回结构化审计结论

契约示例:
  "非休息区蹲下 >10s = 异常"
  "拿取物品 -> 离开 (无结账) = 异常"
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import DecisionConfig, SemanticVLLMConfig
from .structures import EntityGraph
from .narrative_generator import NarrativeGenerator

logger = logging.getLogger(__name__)


# ── 审计结论 ──────────────────────────────────────────
@dataclass
class AuditVerdict:
    """Decision LLM 审计结论"""
    entity_id: int
    is_anomaly: bool
    confidence: float
    break_timestamp: Optional[float] = None
    reason: str = ""
    is_cinematic_false_alarm: bool = False
    raw_response: str = ""

    # 异常区间估计 (秒)
    anomaly_start_sec: float = 0.0
    anomaly_end_sec: float = 0.0


@dataclass
class VideoVerdict:
    """视频级审计结论"""
    is_anomaly: bool
    confidence: float
    entity_verdicts: list[AuditVerdict] = field(default_factory=list)
    anomaly_entity_ids: list[int] = field(default_factory=list)
    scene_type: str = ""
    summary: str = ""


# ── Prompt Templates ──────────────────────────────────

DECISION_SYSTEM_PROMPT = """\
You are a Decision Audit Expert for a video anomaly detection system.

Your task: Given an entity's behavior trajectory narrative, determine if the \
behavior is anomalous by analyzing:

1. **Semantic Logic**: Does the action sequence follow normal causal relations?
   e.g., In a store: "pick up item" should be followed by "checkout" before "leave".
2. **Kinetic Physics**: Does the kinetic energy match the action semantics?
   e.g., "standing still" suddenly becoming "sprinting" with energy surge is suspicious.
3. **Temporal Pattern**: Are there abnormally long pauses or sudden accelerations?
4. **Cinematic Filter**: If footage appears to be from a movie/TV (cinematic), \
   mark as false alarm.

Output ONLY valid JSON. No extra text.
"""

DECISION_USER_TEMPLATE = """\
## Scenario: {scene_type}
{scene_priors}

## Business Contracts (Rules for this scenario):
{contracts}

## Entity Behavior Narrative:
{narrative}

---

IMPORTANT — How to interpret evidence:

[Physical Signal] = camera detected significant pixel changes (motion energy). \
  This ONLY means the image changed, NOT that behavior is illegal. \
  Normal causes: person walking, car passing, door opening, wind, lighting change. \
  Abnormal causes: explosion, fire spreading, sudden violent movement.

[Semantic Observation] = what the entity appears to be DOING based on visual analysis.

DECISION RULE: \
  You must INDEPENDENTLY evaluate the [Semantic Observation] first. \
  Only if the ACTION LOGIC is abnormal (e.g., fighting, forced entry, fire, \
  stealing without checkout) should you consider the physical signal as \
  supporting evidence. \
  DO NOT flag as anomaly simply because motion energy is high — \
  most high-energy motion is normal (walking, running for exercise, etc.).

Analyze this entity's behavior. Output ONLY JSON:

{{
  "is_anomaly": <true|false>,
  "confidence": <float 0.0-1.0>,
  "break_timestamp": <float seconds where anomaly starts, or null>,
  "reason": "<one-sentence explanation>",
  "is_cinematic_false_alarm": <true|false>
}}

- If is_anomaly=false, break_timestamp must be null.
- confidence should reflect how certain you are.
"""

# 场景先验知识 — 让 LLM 自动激活领域知识
SCENE_PRIORS = {
    "gas station": "Prior: Flammable environment. Any fire, smoke, or spark is CRITICAL.",
    "parking lot": "Prior: Vehicle theft and break-ins are common. Watch for forced entry or loitering.",
    "store": "Prior: Shoplifting pattern = pick up item → leave without checkout.",
    "supermarket": "Prior: Shoplifting pattern = pick up item → leave without checkout.",
    "street": "Prior: Watch for fighting, chasing, hit-and-run, or mob violence.",
    "corridor": "Prior: Restricted access. Unauthorized entry or loitering is suspicious.",
    "bank": "Prior: Any threatening gesture, weapon, or forced entry is CRITICAL.",
    "atm": "Prior: Robbery risk. Watch for threatening behavior or forced card usage.",
    "road": "Prior: Watch for car accidents, hit-and-run, reckless driving.",
    "house": "Prior: Break-in risk. Watch for forced entry through windows or doors.",
}


class DecisionAuditor:
    """
    决策审计器。

    对 EntityGraph 中的实体进行 Decision LLM 审计。
    支持并行审计多个实体。
    """

    def __init__(
        self,
        decision_cfg: Optional[DecisionConfig] = None,
        vllm_cfg: Optional[SemanticVLLMConfig] = None,
    ):
        self.cfg = decision_cfg or DecisionConfig()
        self.vllm_cfg = vllm_cfg or SemanticVLLMConfig()
        self.narrative_gen = NarrativeGenerator()

    def audit_video(
        self,
        graphs: dict[int, EntityGraph],
        scene_type: str = "",
        discordance_alerts: Optional[list] = None,
        drift_info: Optional[dict] = None,
    ) -> VideoVerdict:
        """
        审计视频中所有实体。

        Args:
            graphs: entity_id → EntityGraph
            scene_type: 场景类型 (用于匹配契约)
            discordance_alerts: 物理-语义矛盾警报
            drift_info: CLIP 漂移信息

        Returns:
            VideoVerdict
        """
        if not graphs:
            return VideoVerdict(
                is_anomaly=False, confidence=0.0,
                summary="No entities to audit."
            )

        # ── 软过滤：优先审计有可疑信号的实体，但矛盾实体也要审计 ──
        discordance_eids = set()
        if discordance_alerts:
            for alert in discordance_alerts:
                if alert.entity_id >= 0:
                    discordance_eids.add(alert.entity_id)

        # 有漂移信号时，所有实体都需审计
        force_audit_all = False
        if drift_info and drift_info.get("max_drift", 0) > 0.15:
            force_audit_all = True
            logger.info(f"Scene drift detected (max={drift_info['max_drift']:.3f}), force auditing all entities")

        audit_graphs = {}
        for eid, g in graphs.items():
            # 三种情况进入审计：
            # 1. 语义层标记可疑
            # 2. 物理-语义矛盾
            # 3. 场景漂移触发全审计
            if (g.has_suspicious or g.max_danger_score >= 0.2
                    or eid in discordance_eids
                    or force_audit_all):
                audit_graphs[eid] = g

        if not audit_graphs:
            logger.info(
                f"No entities need audit among {len(graphs)} "
                f"(no suspicious, no discordance). Verdict: NORMAL"
            )
            return VideoVerdict(
                is_anomaly=False, confidence=0.0,
                summary=f"No suspicious entities among {len(graphs)} tracked."
            )

        # 选择需要审计的实体 (按 max_danger_score + num_nodes + discordance 排序)
        candidates = sorted(
            audit_graphs.items(),
            key=lambda item: (
                item[1].max_danger_score * 2.0
                + item[1].num_nodes * 0.1
                + (0.5 if item[0] in discordance_eids else 0.0)
            ),
            reverse=True,
        )
        audit_list = candidates[: self.cfg.max_audit_entities]

        logger.info(f"Auditing {len(audit_list)} entities (of {len(graphs)} total)")

        # 并行审计
        verdicts: list[AuditVerdict] = []

        t0 = time.time()

        with ThreadPoolExecutor(max_workers=min(8, len(audit_list))) as executor:
            future_to_eid = {
                executor.submit(
                    self._audit_single, eid, graph, scene_type,
                    discordance_alerts, drift_info,
                ): eid
                for eid, graph in audit_list
            }
            for future in as_completed(future_to_eid):
                eid = future_to_eid[future]
                try:
                    verdict = future.result()
                    verdicts.append(verdict)
                except Exception as e:
                    logger.error(f"Audit entity #{eid} failed: {e}")

        elapsed = time.time() - t0
        logger.info(f"Audit completed: {len(verdicts)} entities in {elapsed:.1f}s")

        # 聚合视频级结论
        anomaly_eids = [
            v.entity_id for v in verdicts
            if v.is_anomaly and not v.is_cinematic_false_alarm
        ]
        is_anomaly = len(anomaly_eids) > 0
        confidence = 0.0
        if anomaly_eids:
            confidence = max(
                v.confidence for v in verdicts
                if v.entity_id in anomaly_eids
            )

        summary_parts = []
        for v in verdicts:
            if v.is_anomaly and not v.is_cinematic_false_alarm:
                summary_parts.append(
                    f"Entity #{v.entity_id}: {v.reason} "
                    f"(conf={v.confidence:.2f}, T={v.break_timestamp})"
                )

        return VideoVerdict(
            is_anomaly=is_anomaly,
            confidence=confidence,
            entity_verdicts=verdicts,
            anomaly_entity_ids=anomaly_eids,
            scene_type=scene_type,
            summary="; ".join(summary_parts) if summary_parts else "No anomaly detected.",
        )

    def _audit_single(
        self,
        entity_id: int,
        graph: EntityGraph,
        scene_type: str,
        discordance_alerts: Optional[list] = None,
        drift_info: Optional[dict] = None,
    ) -> AuditVerdict:
        """审计单个实体"""
        # 1. 生成叙事（含物理异常预警）
        narrative = self.narrative_gen.generate(
            graph,
            discordance_alerts=discordance_alerts,
            drift_info=drift_info,
        )

        # 2. 获取契约
        contracts = self._get_contracts(scene_type)

        # 3. 获取场景先验知识
        scene_lower = scene_type.lower().strip()
        scene_priors = ""
        for key, prior in SCENE_PRIORS.items():
            if key in scene_lower or scene_lower in key:
                scene_priors = prior
                break
        if not scene_priors:
            scene_priors = "Prior: General surveillance. Watch for violence, theft, fire, or unauthorized access."

        # 4. 构建 prompt
        prompt = DECISION_USER_TEMPLATE.format(
            scene_type=scene_type or "unknown",
            scene_priors=scene_priors,
            contracts=contracts,
            narrative=narrative,
        )

        # 4. 调用 Decision LLM
        raw_response = self._call_llm(prompt)

        # 5. 解析
        verdict = self._parse_response(entity_id, raw_response, graph)

        logger.info(
            f"Entity #{entity_id}: anomaly={verdict.is_anomaly}, "
            f"conf={verdict.confidence:.2f}, reason={verdict.reason[:60]}"
        )

        return verdict

    def _call_llm(self, prompt: str) -> str:
        """通过 vLLM server 调用 Decision LLM"""
        import httpx

        model_name = self.vllm_cfg.MODEL_PATHS.get(
            self.vllm_cfg.model_name, self.vllm_cfg.model_name
        )

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.cfg.decision_max_tokens,
            "temperature": self.cfg.decision_temperature,
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{self.vllm_cfg.api_base}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Decision LLM call failed: {e}")
            return self._rule_based_fallback(prompt)

    def _rule_based_fallback(self, prompt: str) -> str:
        """规则兜底"""
        text = prompt.lower()
        is_anomaly = False
        reasons = []
        conf = 0.3

        danger_kw = [
            "suspicious", "dangerous", "extreme danger",
            "fighting", "stealing", "shooting", "fire",
            "running away", "climbing", "vandalizing",
        ]
        for kw in danger_kw:
            if kw in text:
                is_anomaly = True
                reasons.append(kw)
                conf += 0.1

        conf = min(conf, 0.85)
        return json.dumps({
            "is_anomaly": is_anomaly,
            "confidence": round(conf, 2),
            "break_timestamp": None,
            "reason": f"Rule-based: {', '.join(reasons)}" if reasons else "No anomaly",
            "is_cinematic_false_alarm": False,
        })

    def _parse_response(
        self,
        entity_id: int,
        raw_response: str,
        graph: EntityGraph,
    ) -> AuditVerdict:
        """解析 Decision LLM 响应"""
        import re

        parsed = None
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            logger.warning(
                f"Parse failed for entity #{entity_id}: {raw_response[:200]}"
            )
            return AuditVerdict(
                entity_id=entity_id,
                is_anomaly=False,
                confidence=0.0,
                reason="Failed to parse LLM response",
                raw_response=raw_response,
            )

        is_anomaly = bool(parsed.get("is_anomaly", False))
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        reason = str(parsed.get("reason", ""))
        is_cinematic = bool(parsed.get("is_cinematic_false_alarm", False))

        break_ts = parsed.get("break_timestamp")
        if break_ts is not None:
            try:
                break_ts = float(break_ts)
            except (TypeError, ValueError):
                break_ts = None

        # ── 精确估计异常区间：广播到实体的完整活跃时间段 ──
        anomaly_start = 0.0
        anomaly_end = 0.0
        if is_anomaly:
            # 策略：从 break_timestamp 开始到实体最后出现时间
            # 如果没有 break_ts，用实体 birth_time 到 last_time
            if break_ts is not None:
                anomaly_start = max(0.0, break_ts - 1.0)
            else:
                anomaly_start = graph.birth_time

            anomaly_end = graph.last_time

            # 额外扩展：向前回溯到最早的可疑节点
            for node in graph.nodes:
                if node.is_suspicious or node.danger_score >= 0.2:
                    anomaly_start = min(anomaly_start, node.timestamp)

            # 向后扩展：如果最后一个节点就是可疑的，向后多延伸 5 秒
            if graph.nodes and (graph.nodes[-1].is_suspicious or graph.nodes[-1].danger_score >= 0.2):
                anomaly_end = graph.last_time + 5.0

            # 确保 start < end
            if anomaly_end <= anomaly_start:
                anomaly_end = anomaly_start + 3.0

        return AuditVerdict(
            entity_id=entity_id,
            is_anomaly=is_anomaly,
            confidence=confidence,
            break_timestamp=break_ts,
            reason=reason,
            is_cinematic_false_alarm=is_cinematic,
            raw_response=raw_response,
            anomaly_start_sec=anomaly_start,
            anomaly_end_sec=anomaly_end,
        )

    def _get_contracts(self, scene_type: str) -> str:
        """获取场景业务契约"""
        contracts = self.cfg.business_contracts
        scene_lower = scene_type.lower().strip()

        matched = []
        for key, rules in contracts.items():
            if key == "default":
                continue
            if key in scene_lower or scene_lower in key:
                matched.extend(rules)

        if not matched:
            matched = contracts.get("default", [])

        if not matched:
            return "  (No specific contracts)"

        return "\n".join(f"  {i}. {r}" for i, r in enumerate(matched, 1))
