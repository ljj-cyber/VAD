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
You are a decision module for a surveillance video anomaly detection system.

Your task: Given an entity's behavior trajectory narrative, determine whether \
the behavior constitutes a REAL-WORLD ANOMALY that would require security attention.

## What counts as ANOMALOUS (flag is_anomaly=true):

### Overt anomalies (clearly visible):
- Violence: fighting, assault, pushing, kicking, restraining someone
- Fire / arson: burning, smoke, explosion, fire spreading
- Accidents: vehicle collision, person falling and not getting up, hit-and-run
- Weapons: brandishing a gun, knife, or weapon
- Trespassing / intrusion: climbing fences, forced entry

### Covert anomalies (subtle but equally important):
- Shoplifting: picking up items then concealing them; browsing unusually long near \
merchandise then leaving without checkout; handling items while repeatedly looking around
- Stealing: accessing or tampering with property that doesn't belong to the person \
(e.g. reaching into a car, opening someone else's bag); taking items from an \
unattended location; removing objects without a visible transaction
- Abuse: one person physically dominating another — pushing, shoving, slapping, \
cornering, forcing someone to the ground; one person cowering/crouching while \
another looms over or strikes them; aggressive close-proximity interactions
- Vandalism: kicking, hitting, or throwing objects at property; spray-painting \
walls or surfaces; smashing windows, signs, or fixtures; deliberately damaging \
vehicles or public infrastructure

## What is NORMAL (flag is_anomaly=false):
- Everyday activities: walking, standing, sitting, reading, working, talking
- Normal transitions: standing→sitting, walking→stopping, driving→parking
- Posture changes: bending, stretching, leaning, crouching briefly to pick something up
- Routine work: typing, operating machines, stocking shelves, cleaning
- Normal movement patterns: entering/exiting, browsing in a store, waiting in line
- Retail/store-specific normal behaviors: picking up items to examine then \
replacing them, walking between aisles while carrying items, standing near or \
at checkout counters, using self-checkout terminals, interacting with cashiers, \
paying for goods. Store employees approaching customers, operating POS terminals, \
restocking shelves, and handling merchandise are all NORMAL work activities — \
do NOT flag these as shoplifting.
- Idle or stationary behavior, even for prolonged periods
- Kinetic energy fluctuations alone do NOT indicate anomaly unless paired with \
clearly dangerous actions

## CRITICAL — High Kinetic Energy ≠ Anomaly:
- Many NORMAL activities produce high kinetic energy: sports (basketball, soccer, \
martial arts), dancing, running/jogging, military parades, action movie scenes, \
fast-paced driving, crowd celebrations, concerts, and physical exercise.
- Movie/film clips often contain dramatic scenes (chase sequences, stunts, \
choreographed fights, explosions) that are FICTIONAL — these are NOT real anomalies.
- Do NOT flag high kinetic energy as anomalous unless the narrative explicitly \
describes REAL violence, property damage, or criminal behavior with clear victims.
- "Perception blind spot" or "kinetic energy exceeding threshold" alone is \
NEVER sufficient evidence for anomaly — you MUST see specific dangerous actions \
described in the narrative text.

## Key principles:
- Flag behavior as anomalous ONLY if the ACTIONS described in the narrative \
clearly match a specific overt OR covert anomaly category above.
- Do NOT flag behavior merely because it involves transitions, posture changes, \
or kinetic energy variations.
- High kinetic energy, "perception blind spot", or "energy exceeding threshold" \
are NOT anomaly evidence by themselves. You MUST identify SPECIFIC dangerous \
actions (hitting, shooting, stealing, burning, etc.) in the narrative.
- Pay special attention to person-object and person-person interactions described \
in the narrative — these are critical for detecting covert anomalies.
- When the narrative describes someone handling objects near others' property, \
or aggressive physical contact between people, treat these as strong anomaly signals.
- In retail settings, most people are legitimate customers or employees. \
Shoplifting requires CLEAR evidence of concealment or deliberately avoiding \
checkout — merely picking up items, walking through aisles, or interacting \
near checkout counters is NORMAL shopping behavior.
- When is_cinematic_false_alarm applies: if the scene appears to be from a \
movie, TV show, sports broadcast, or staged performance, set \
is_cinematic_false_alarm=true even if the depicted actions look violent.

Output ONLY valid JSON. No extra text.
"""

DECISION_USER_TEMPLATE = """\
## Scenario: {scene_type}
{contracts_section}
## Entity Behavior Narrative:
{narrative}

---

Determine whether this entity's behavior is anomalous (matches violence, theft, \
fire, accident, weapons, vandalism, or intrusion). Normal daily activities and \
routine transitions should NOT be flagged.
{contracts_reminder}
Output ONLY JSON:

{{
  "is_anomaly": <true|false>,
  "confidence": <float 0.0-1.0>,
  "anomaly_start_sec": <float — earliest second the anomaly begins, or 0.0 if normal>,
  "anomaly_end_sec": <float — latest second the anomaly ends, or 0.0 if normal>,
  "reason": "<one-sentence explanation>",
  "is_cinematic_false_alarm": <true|false>
}}

- If is_anomaly=false, anomaly_start_sec and anomaly_end_sec must both be 0.0.
- anomaly_start_sec / anomaly_end_sec should tightly cover the anomalous behavior \
observed in the narrative (use the T=... timestamps as reference).
- confidence should reflect how certain you are.
"""

# 场景先验知识 — 已合并到 business_contracts，不再单独注入
# (避免与 contracts_section 三重注入导致 FP)
SCENE_PRIORS = {}


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
        entity_trace_buffer: Optional[dict] = None,
        is_cinematic: bool = False,
    ) -> VideoVerdict:
        """
        审计视频中所有实体。

        Args:
            graphs: entity_id → EntityGraph
            scene_type: 场景类型 (用于匹配契约)
            discordance_alerts: 物理-语义矛盾警报
            drift_info: CLIP 漂移信息
            entity_trace_buffer: entity_id → list[TraceEntry] (物理轨迹)
            is_cinematic: 是否检测到电影/影视场景

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
        if drift_info and drift_info.get("max_drift", 0) > 0.25:
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

        _trace_buf = entity_trace_buffer or {}

        with ThreadPoolExecutor(max_workers=min(8, len(audit_list))) as executor:
            future_to_eid = {
                executor.submit(
                    self._audit_single, eid, graph, scene_type,
                    discordance_alerts, drift_info,
                    _trace_buf.get(eid, []),
                    is_cinematic,
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

        # ── Cinematic post-filter ──
        # The prompt injection (Level 1) asks the LLM to set
        # is_cinematic_false_alarm=true for non-violent movie scenes.
        # Level 2 (here): for entities where the LLM flags anomaly but
        # NOT cinematic, we trust the LLM's judgment (it found strong
        # evidence of real violence). Only apply a mild attenuation (×0.7)
        # as a safety net; the prompt is the primary mechanism.
        if is_cinematic:
            n_cinematic_flagged = sum(
                1 for v in verdicts if v.is_cinematic_false_alarm
            )
            n_attenuated = 0
            for v in verdicts:
                if v.is_anomaly and not v.is_cinematic_false_alarm:
                    old_conf = v.confidence
                    v.confidence *= 0.7
                    n_attenuated += 1
                    logger.debug(
                        f"Cinematic attenuation: Entity #{v.entity_id} "
                        f"conf {old_conf:.2f} → {v.confidence:.2f}"
                    )
            if n_cinematic_flagged or n_attenuated:
                logger.info(
                    f"Cinematic filter: {n_cinematic_flagged} entities flagged "
                    f"cinematic by LLM, {n_attenuated} attenuated ×0.7"
                )

        # 聚合视频级结论
        anomaly_eids = [
            v.entity_id for v in verdicts
            if v.is_anomaly
            and not v.is_cinematic_false_alarm
            and v.confidence >= self.cfg.anomaly_confidence_threshold
        ]
        # 日志: 被置信度阈值过滤的实体
        n_pre_filter = sum(
            1 for v in verdicts
            if v.is_anomaly and not v.is_cinematic_false_alarm
        )
        if n_pre_filter > len(anomaly_eids):
            logger.info(
                f"Confidence threshold ({self.cfg.anomaly_confidence_threshold}) "
                f"filtered {n_pre_filter - len(anomaly_eids)} low-confidence anomaly verdicts"
            )

        # ── Multi-entity saturation suppression ──
        anomaly_eids = self._suppress_saturation(
            anomaly_eids, verdicts, len(audit_list), scene_type
        )

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
                    f"(conf={v.confidence:.2f}, "
                    f"interval=[{v.anomaly_start_sec:.1f}s, {v.anomaly_end_sec:.1f}s])"
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
        trace_entries: Optional[list] = None,
        is_cinematic: bool = False,
    ) -> AuditVerdict:
        """审计单个实体"""
        # 1. 生成叙事（含物理异常预警 + 物理轨迹摘要）
        narrative = self.narrative_gen.generate(
            graph,
            discordance_alerts=discordance_alerts,
            drift_info=drift_info,
            trace_entries=trace_entries,
        )

        # 2. 构建 prompt（注入场景先验知识 + 业务契约）
        contracts_text = self._get_contracts(scene_type)
        scene_prior = self._get_scene_prior(scene_type)

        contracts_section = ""
        if contracts_text != "  (No specific contracts)":
            contracts_section = (
                f"\n## Scene Rules (prior knowledge):\n{contracts_text}\n"
            )

        contracts_reminder = ""
        if contracts_section:
            contracts_reminder = (
                "\nEvaluate whether the behavior described above matches "
                "any of the Scene Rules. Only flag as anomalous if the "
                "evidence is clear and specific — do NOT flag routine "
                "shopping or work activities.\n"
            )

        prompt = DECISION_USER_TEMPLATE.format(
            scene_type=scene_type or "unknown",
            narrative=narrative,
            contracts_section=contracts_section,
            contracts_reminder=contracts_reminder,
        )

        # 2.5 Cinematic context injection
        if is_cinematic:
            prompt += (
                "\n⚠ IMPORTANT — CINEMATIC / ENTERTAINMENT CONTENT DETECTED: "
                "This video is likely from a MOVIE, TV SHOW, or BROADCAST. "
                "Be VERY skeptical about declaring anomaly. Specifically:\n"
                "  - Dramatic tension, fearful expressions, suspense, tense "
                "postures = NORMAL in movies. NOT anomaly.\n"
                "  - Choreographed fight sequences, martial arts, stunt "
                "scenes = FICTIONAL. NOT real violence.\n"
                "  - Characters aiming weapons, car chases, explosions in "
                "movie context = SCRIPTED. NOT real danger.\n"
                "  - Sports (basketball falls, tackles, wrestling) = "
                "COMPETITIVE PLAY. NOT abuse.\n"
                "  - Injuries shown in post-action movie scenes (bandaging, "
                "aftermath) = FICTIONAL. NOT real assault.\n"
                "For cinematic content, set is_cinematic_false_alarm=true "
                "UNLESS the narrative describes EXTREME EXPLICIT violence "
                "with clear real-world victims (mass shooting, actual arson "
                "with victims, brutal assault with visible injury). "
                "When in doubt, default to is_cinematic_false_alarm=true.\n"
            )

        # 3. 收集该实体的 discordance 峰值信息
        entity_discordance = None
        if discordance_alerts:
            my_alerts = [
                a for a in discordance_alerts
                if a.entity_id == entity_id and a.alert_type == "energy_semantic_gap"
            ]
            if my_alerts:
                entity_discordance = my_alerts
                # 在 prompt 中提示 discordance 信号（保持中立，不诱导）
                prompt += (
                    "\n⚠ NOTE: This entity has elevated physical motion energy compared "
                    "to what the semantic description suggests. However, high kinetic energy "
                    "is VERY COMMON in normal activities such as sports, dancing, action "
                    "movie scenes, military parades, fast driving, and physical exercise. "
                    "You MUST check whether the narrative describes SPECIFIC real-world "
                    "anomaly behaviors (actual violence with victims, actual property damage, "
                    "actual theft, actual fire). High kinetic energy ALONE — from any form "
                    "of vigorous but non-criminal activity — is NOT evidence of anomaly. "
                    "If the only evidence is 'kinetic energy exceeding threshold', "
                    "you MUST set is_anomaly=false.\n"
                )

        # 4. 调用 Decision LLM
        raw_response = self._call_llm(prompt)

        # 5. 解析（传入 discordance 峰值信息用于区间定位）
        verdict = self._parse_response(
            entity_id, raw_response, graph,
            discordance_alerts=entity_discordance,
        )

        logger.info(
            f"Entity #{entity_id}: anomaly={verdict.is_anomaly}, "
            f"conf={verdict.confidence:.2f}, reason={verdict.reason[:60]}"
        )

        return verdict

    def _call_llm(self, prompt: str) -> str:
        """调用 Decision LLM (支持 server / local 两种后端)"""
        if self.vllm_cfg.backend == "local":
            return self._call_llm_local(prompt)
        return self._call_llm_server(prompt)

    def _call_llm_server(self, prompt: str) -> str:
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

    def _call_llm_local(self, prompt: str) -> str:
        """通过本地 Qwen2.5-VL 模型调用 Decision LLM (纯文本)"""
        try:
            from ..semantic.vllm_semantic import get_local_model, local_chat_inference

            model, processor = get_local_model(self.vllm_cfg)

            messages = [
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            result = local_chat_inference(
                model, processor, messages,
                max_new_tokens=self.cfg.decision_max_tokens,
                temperature=self.cfg.decision_temperature,
            )
            return result
        except Exception as e:
            logger.error(f"Local Decision LLM call failed: {e}")
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
            "anomaly_start_sec": 0.0,
            "anomaly_end_sec": 0.0,
            "reason": f"Rule-based: {', '.join(reasons)}" if reasons else "No anomaly",
            "is_cinematic_false_alarm": False,
        })

    def _parse_response(
        self,
        entity_id: int,
        raw_response: str,
        graph: EntityGraph,
        discordance_alerts: Optional[list] = None,
    ) -> AuditVerdict:
        """
        解析 Decision LLM 响应。

        对于 discordance 检出的异常，用动能峰值时间定位区间，
        而非完全依赖 LLM 给出的（可能不准确的）时间。
        """
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

        # ── 兼容: 仍然解析 break_timestamp (如果 LLM 仍然输出) ──
        break_ts = parsed.get("break_timestamp")
        if break_ts is not None:
            try:
                break_ts = float(break_ts)
            except (TypeError, ValueError):
                break_ts = None

        # ── 直接从 LLM 响应解析异常区间 ──
        anomaly_start = 0.0
        anomaly_end = 0.0

        if is_anomaly:
            # ══════════════════════════════════════════════════
            # 优先级 1: discordance 动能峰值定位
            # 当该实体有 discordance alert (物理-语义矛盾) 时,
            # 用动能峰值时间和突发区间来锚定异常区间,
            # 而非依赖 LLM 给出的时间(LLM 在 discordance 场景
            # 下缺乏精确时间信息)
            # ══════════════════════════════════════════════════
            used_discordance_peak = False
            if discordance_alerts:
                # 取动能最大的 alert 作为主锚点
                best_alert = max(
                    discordance_alerts, key=lambda a: a.peak_energy_value
                )
                if best_alert.burst_end_sec > best_alert.burst_start_sec > 0:
                    anomaly_start = best_alert.burst_start_sec
                    anomaly_end = best_alert.burst_end_sec
                    used_discordance_peak = True
                    logger.info(
                        f"Entity #{entity_id}: discordance peak-anchored interval "
                        f"[{anomaly_start:.2f}, {anomaly_end:.2f}] "
                        f"(peak={best_alert.peak_energy_time:.2f}s, "
                        f"KE={best_alert.peak_energy_value:.4f})"
                    )
                elif best_alert.peak_energy_time > 0:
                    # burst 区间无效，但有峰值时间 → 用峰值 ± 5s
                    anomaly_start = max(0.0, best_alert.peak_energy_time - 5.0)
                    anomaly_end = best_alert.peak_energy_time + 5.0
                    used_discordance_peak = True
                    logger.info(
                        f"Entity #{entity_id}: discordance peak ± 5s interval "
                        f"[{anomaly_start:.2f}, {anomaly_end:.2f}] "
                        f"(peak={best_alert.peak_energy_time:.2f}s)"
                    )

            # ══════════════════════════════════════════════════
            # 优先级 2: LLM 直接输出的区间 (非 discordance 时)
            # ══════════════════════════════════════════════════
            if not used_discordance_peak:
                raw_start = parsed.get("anomaly_start_sec")
                raw_end = parsed.get("anomaly_end_sec")

                llm_start = None
                llm_end = None
                if raw_start is not None:
                    try:
                        llm_start = float(raw_start)
                    except (TypeError, ValueError):
                        pass
                if raw_end is not None:
                    try:
                        llm_end = float(raw_end)
                    except (TypeError, ValueError):
                        pass

                if llm_start is not None and llm_end is not None and llm_end > llm_start:
                    anomaly_start = max(0.0, llm_start)
                    anomaly_end = llm_end
                    logger.debug(
                        f"Entity #{entity_id}: LLM interval "
                        f"[{anomaly_start:.2f}, {anomaly_end:.2f}]"
                    )
                else:
                    # Fallback: 用实体可疑节点时间范围
                    suspicious_times = [
                        node.timestamp for node in graph.nodes
                        if node.is_suspicious or node.danger_score >= 0.3
                    ]
                    if suspicious_times:
                        anomaly_start = max(0.0, min(suspicious_times) - 2.0)
                        anomaly_end = max(suspicious_times) + 5.0
                    else:
                        anomaly_start = graph.birth_time
                        anomaly_end = graph.last_time + 3.0

                    logger.debug(
                        f"Entity #{entity_id}: fallback interval "
                        f"[{anomaly_start:.2f}, {anomaly_end:.2f}]"
                    )

            # 安全校验
            anomaly_start = max(0.0, anomaly_start)
            if anomaly_end <= anomaly_start:
                anomaly_end = anomaly_start + 6.0

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

    @staticmethod
    def _get_scene_prior(scene_type: str) -> str:
        """获取场景先验知识提示"""
        scene_lower = scene_type.lower().strip()
        for key, prior in SCENE_PRIORS.items():
            if key in scene_lower or scene_lower in key:
                return prior
        return ""

    @staticmethod
    def _suppress_saturation(
        anomaly_eids: list[int],
        verdicts: list["AuditVerdict"],
        total_audited: int,
        scene_type: str,
    ) -> list[int]:
        """
        多实体饱和抑制：当 >50% 审计实体被标记且理由收敛于同一关键词时，
        提高有效置信度阈值以抑制 prompt 偏置导致的 FP。

        逻辑参考 discordance_checker.py 的 voting suppression。
        """
        if total_audited < 3 or len(anomaly_eids) < 2:
            return anomaly_eids

        ratio = len(anomaly_eids) / total_audited
        if ratio < 0.5:
            return anomaly_eids

        # 检查是否所有异常理由收敛于同一关键词模式
        BIAS_KEYWORDS = {
            "shoplifting": ["shoplifting", "shoplift", "checkout", "without paying"],
            "kinetic_energy": ["kinetic energy", "energy exceeding", "perception blind spot", "energy threshold"],
            "sports_activity": ["basketball", "soccer", "football", "gameplay", "player", "match"],
        }

        anomaly_verdicts = [
            v for v in verdicts if v.entity_id in anomaly_eids
        ]
        reasons_text = " ".join(v.reason.lower() for v in anomaly_verdicts)

        for bias_name, keywords in BIAS_KEYWORDS.items():
            keyword_hits = sum(1 for kw in keywords if kw in reasons_text)
            if keyword_hits >= 2:
                # 饱和检出 — 提高阈值至 0.95
                elevated_threshold = 0.95
                filtered = [
                    eid for eid in anomaly_eids
                    if any(
                        v.entity_id == eid and v.confidence >= elevated_threshold
                        for v in verdicts
                    )
                ]
                logger.info(
                    f"Saturation suppression: {len(anomaly_eids)}/{total_audited} "
                    f"entities flagged as '{bias_name}' (ratio={ratio:.0%}). "
                    f"Raising threshold to {elevated_threshold}. "
                    f"Kept {len(filtered)}/{len(anomaly_eids)} entities."
                )
                return filtered

        return anomaly_eids
