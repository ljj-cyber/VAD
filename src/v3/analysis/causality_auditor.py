"""
V4.0 决策审计层 — Decision LLM 语义因果审计 (Causality Auditor)

核心重构: V3.0 的多信号线性加权被替换为 Decision LLM 因果审计。

工作流:
  1. 接收叙事引擎生成的因果叙事文本
  2. 注入场景业务契约 (Business Contracts)
  3. 调用 Decision LLM 进行语义+动能双向验证
  4. 返回结构化审计结论: 是否异常、逻辑断裂点、置信度、解释文本

触发条件 (异步审计):
  - 图路径长度变化
  - 感知层触发高危信号 (visual_danger_score ≥ threshold)

Decision LLM 同时审计:
  - 语义逻辑: 动作序列是否违背因果律
  - 物理动能: 动作转移时的运动能量是否合理
  - 电影特征: 是否为电影/剪辑产生的伪信号
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..config import DecisionConfig, NarrativeConfig
from ..association.temporal_graph import TemporalGraph
from .narrative_engine import NarrativeEngine

logger = logging.getLogger(__name__)


# ── 审计结论数据类 ────────────────────────────────────
@dataclass
class AuditVerdict:
    """Decision LLM 返回的审计结论"""

    entity_id: int
    is_anomaly: bool                        # 是否异常
    confidence: float                       # 置信度 [0, 1]
    break_node_id: Optional[str] = None     # 逻辑断裂节点 (V_break)
    break_timestamp: Optional[float] = None # 断裂时间戳
    reason_zh: str = ""                     # 中文审计结论
    reason_en: str = ""                     # 英文审计结论
    is_cinematic_false_alarm: bool = False  # 是否电影伪信号
    violated_contracts: list[str] = field(default_factory=list)  # 违反的业务契约
    raw_llm_response: str = ""             # LLM 原始返回（调试用）


@dataclass
class VideoAuditReport:
    """视频级审计报告"""

    video_anomaly: bool                         # 视频是否存在异常
    video_confidence: float                     # 视频级置信度
    entity_verdicts: list[AuditVerdict]         # 各实体审计结论
    anomaly_entities: list[int]                 # 异常实体 ID 列表
    cinematic_filtered: list[int]               # 被电影特征过滤的实体
    scene_type: str = ""
    summary_zh: str = ""                        # 视频级中文摘要
    summary_en: str = ""                        # 视频级英文摘要


# ── Decision Prompt 模板 ─────────────────────────────
DECISION_SYSTEM_PROMPT = """\
你是一个工业视频异常检测系统的 **决策审计专家**。

你的任务是根据提供的 **实体行为轨迹叙事文本**，综合以下三个维度进行判定：

1. **语义逻辑审计**：动作序列是否符合场景下的正常因果关系。
   例如：超市中"拿起物品"之后应有"结账"才能"离开"。
2. **物理动能审计**：动作转移时的运动能量是否与动作语义匹配。
   例如："静止"突然变为"奔跑"，且动能激增，需高度关注。
3. **电影特征过滤**：如果多帧被标记为电影/剪辑特征 (is_cinematic=true)，
   异常可能是来自影视片段而非真实事件，应标记为伪信号。

请你根据以下信息进行审计并输出 **严格 JSON** 结果。
"""

DECISION_USER_PROMPT_TEMPLATE = """\
## Scene: {scene_type}
## Rules:
{business_contracts}

## Entity behavior trajectory:
{narrative_text}

---

Analyze the entity's behavior. Output ONLY JSON:

{{
  "is_anomaly": <true|false>,
  "confidence": <float 0.0-1.0>,
  "break_timestamp": <float seconds|null>,
  "reason": "<one-sentence explanation>",
  "is_cinematic_false_alarm": <true|false>
}}

- Output ONLY JSON, no extra text.
- If is_anomaly=false, break_timestamp must be null.
"""


class CausalityAuditor:
    """
    V4.0 决策审计器。

    异步触发 Decision LLM 对实体行为进行因果审计。
    """

    def __init__(
        self,
        decision_cfg: Optional[DecisionConfig] = None,
        narrative_cfg: Optional[NarrativeConfig] = None,
        vllm_client=None,
    ):
        self.cfg = decision_cfg or DecisionConfig()
        self.narrative_engine = NarrativeEngine(narrative_cfg)
        self._decision_client = None   # 延迟初始化
        self._vllm_client = vllm_client  # 复用感知层模型

    # ── 审计触发判断 ──────────────────────────────────
    def should_trigger_audit(
        self,
        graph: TemporalGraph,
        entity_id: int,
        prev_path_lengths: Optional[dict[int, int]] = None,
    ) -> bool:
        """
        判断是否应触发 Decision LLM 审计。

        触发条件:
          1. 图路径长度发生变化（有新节点加入）
          2. 感知层触发高危信号 (visual_danger_score ≥ threshold)
          3. 路径长度达到最小审计长度
        """
        path = graph.get_entity_path(entity_id)
        if len(path) < self.cfg.min_path_length_for_audit:
            return False

        # 条件 1: 路径长度变化
        if self.cfg.trigger_on_path_change and prev_path_lengths:
            prev_len = prev_path_lengths.get(entity_id, 0)
            curr_len = len(path)
            if curr_len > prev_len:
                return True

        # 条件 2: 高危信号
        if self.cfg.trigger_on_danger:
            if graph.has_danger_signal(entity_id, self.cfg.danger_threshold):
                return True

        return False

    # ── 单实体审计 ────────────────────────────────────
    def audit_entity(
        self,
        entity_id: int,
        graph: TemporalGraph,
        scene_type: str = "",
    ) -> AuditVerdict:
        """
        对单个实体执行 Decision LLM 审计。

        Args:
            entity_id: 实体 ID
            graph: 时间演化图
            scene_type: 场景类型

        Returns:
            AuditVerdict
        """
        # 1. 生成叙事文本
        narrative = self.narrative_engine.path_to_text(
            entity_id, graph, scene_type
        )

        # 2. 获取业务契约
        contracts = self._get_contracts(scene_type)

        # 3. 构建 Decision Prompt
        prompt = DECISION_USER_PROMPT_TEMPLATE.format(
            scene_type=scene_type or "未知场景",
            business_contracts=contracts,
            narrative_text=narrative,
        )

        # 4. 调用 Decision LLM
        raw_response = self._call_decision_llm(prompt)

        # 5. 解析结论
        verdict = self._parse_verdict(entity_id, raw_response, graph)

        logger.info(
            f"Audit Entity #{entity_id}: "
            f"anomaly={verdict.is_anomaly}, "
            f"confidence={verdict.confidence:.2f}, "
            f"cinematic_filter={verdict.is_cinematic_false_alarm}"
        )

        return verdict

    # ── 批量审计 ──────────────────────────────────────
    def audit_all_entities(
        self,
        graph: TemporalGraph,
        scene_type: str = "",
        prev_path_lengths: Optional[dict[int, int]] = None,
        snapshots: Optional[list[dict]] = None,
    ) -> VideoAuditReport:
        """
        审计图中所有符合条件的实体，生成视频级审计报告。
        使用并行请求加速 (vLLM server continuous batching)。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time

        # 筛选需要审计的实体 — 按路径长度排序，优先审计出现次数多的
        max_audit_entities = max(1, int(getattr(self.cfg, "max_audit_entities", 8)))
        candidates = []
        for eid in graph.get_all_entity_ids():
            path = graph.get_entity_path(eid)
            if len(path) >= self.cfg.min_path_length_for_audit:
                candidates.append((eid, len(path)))

        # 按路径长度降序排序，取 top-N
        candidates.sort(key=lambda x: x[1], reverse=True)
        audit_eids = [eid for eid, _ in candidates[:max_audit_entities]]

        logger.info(f"Auditing {len(audit_eids)} entities (parallel)...")
        t0 = _time.time()

        # 并行审计所有实体
        verdicts = []
        anomaly_entities = []
        cinematic_filtered = []

        max_workers = 8
        if self._vllm_client and hasattr(self._vllm_client, 'max_workers'):
            max_workers = self._vllm_client.max_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_eid = {
                executor.submit(self.audit_entity, eid, graph, scene_type): eid
                for eid in audit_eids
            }
            for future in as_completed(future_to_eid):
                try:
                    verdict = future.result()
            verdicts.append(verdict)
            if verdict.is_anomaly and not verdict.is_cinematic_false_alarm:
                        anomaly_entities.append(verdict.entity_id)
            elif verdict.is_cinematic_false_alarm:
                        cinematic_filtered.append(verdict.entity_id)
                except Exception as e:
                    eid = future_to_eid.get(future, "?")
                    logger.error(f"Audit entity #{eid} failed: {e}")

        elapsed = _time.time() - t0
        logger.info(f"Audit done: {len(verdicts)} entities in {elapsed:.1f}s")

        # ── 规则兜底: 用感知层信号补捕 LLM 漏检的异常 ──
        if not anomaly_entities and snapshots:
            rule_result = self._rule_based_perception_check(snapshots)
            if rule_result is not None:
                anomaly_entities.append(-1)  # 标记为规则检出
                verdicts.append(rule_result)
                logger.info(
                    f"Rule-based fallback triggered: "
                    f"danger={rule_result.confidence:.2f}, "
                    f"reason={rule_result.reason_zh[:80]}"
                )

        # 视频级判定
        video_anomaly = len(anomaly_entities) > 0
        video_confidence = 0.0
        if verdicts:
            valid_verdicts = [
                v for v in verdicts if not v.is_cinematic_false_alarm
            ]
            if valid_verdicts:
                video_confidence = max(
                    v.confidence for v in valid_verdicts
                    if v.is_anomaly
                ) if any(v.is_anomaly for v in valid_verdicts) else 0.0

        # 生成摘要
        summary_zh, summary_en = self._build_summary(
            verdicts, anomaly_entities, cinematic_filtered, scene_type
        )

        report = VideoAuditReport(
            video_anomaly=video_anomaly,
            video_confidence=video_confidence,
            entity_verdicts=verdicts,
            anomaly_entities=anomaly_entities,
            cinematic_filtered=cinematic_filtered,
            scene_type=scene_type,
            summary_zh=summary_zh,
            summary_en=summary_en,
        )

        logger.info(
            f"Video Audit: anomaly={video_anomaly}, "
            f"confidence={video_confidence:.2f}, "
            f"anomaly_entities={anomaly_entities}, "
            f"cinematic_filtered={cinematic_filtered}"
        )

        return report

    # ── Decision LLM 调用 ─────────────────────────────
    def _call_decision_llm(self, user_prompt: str) -> str:
        """
        调用 Decision LLM。

        支持三种后端:
          - local-qwen: 复用本地 Qwen2-VL
          - qwen-max:   通过 API 调用阿里云 Qwen-Max
          - gpt-4:      通过 API 调用 OpenAI GPT-4
        """
        backend = self.cfg.decision_backend

        if backend == "local-qwen":
            return self._call_local_qwen(user_prompt)
        elif backend in ("qwen-max", "gpt-4"):
            return self._call_remote_api(user_prompt)
        else:
            logger.warning(
                f"Unknown decision backend: {backend}, using rule-based fallback"
            )
            return self._rule_based_fallback(user_prompt)

    def _call_local_qwen(self, user_prompt: str) -> str:
        """
        使用 Qwen 模型进行决策推理（纯文本，无需视觉输入）。

        优先级:
          1. vLLM server API（如果感知层使用 server 模式）
          2. 复用本地已加载的模型
          3. 规则兜底
        """
        try:
            # 方式 1: 通过 vLLM server API 调用（server 模式）
            if (self._vllm_client is not None
                    and self._vllm_client.backend == "server"):
                return self._call_via_vllm_server(user_prompt)

            # 方式 2: 复用本地已加载的模型
            if (self._vllm_client is not None
                    and self._vllm_client._loaded):
                return self._generate_with_vllm_client(user_prompt)

            # 方式 3: 规则兜底
            logger.info("No VLLM backend available, using rule-based fallback")
            return self._rule_based_fallback(user_prompt)

        except Exception as e:
            logger.error(f"Decision LLM failed: {e}")
            return self._rule_based_fallback(user_prompt)

    def _call_via_vllm_server(self, user_prompt: str) -> str:
        """通过 vLLM server API 进行纯文本决策推理"""
        import httpx

        model_name = self._vllm_client.cfg.MODEL_PATHS.get(
            self._vllm_client.model_name, self._vllm_client.model_name
        )
        api_base = self._vllm_client.api_base

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.cfg.decision_max_tokens,
            "temperature": 0,
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{api_base}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"].strip()

    def _generate_with_vllm_client(self, user_prompt: str) -> str:
        """复用感知层 Qwen2-VL 进行纯文本推理"""
        import torch

        model = self._vllm_client.model
        processor = self._vllm_client.processor

            messages = [
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

        text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.decision_max_tokens,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1]
        response = processor.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
            return response.strip()

    def _call_remote_api(self, user_prompt: str) -> str:
        """通过 HTTP API 调用远程 Decision LLM"""
        try:
            import requests

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            }

            payload = {
                "model": self.cfg.api_model,
                "messages": [
                    {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.cfg.decision_temperature,
                "max_tokens": self.cfg.decision_max_tokens,
            }

            response = requests.post(
                f"{self.cfg.api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.cfg.api_timeout,
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Remote Decision LLM API failed: {e}")
            return self._rule_based_fallback(user_prompt)

    def _rule_based_fallback(self, user_prompt: str) -> str:
        """
        规则兜底: 当 LLM 不可用时，基于叙事文本中的关键词进行简单判断。
        返回模拟的 JSON 格式。
        """
        text = user_prompt.lower()

        # 简单关键词检测
        is_anomaly = False
        reasons = []
        confidence = 0.3

        danger_keywords = [
            "低合理性", "low reasonableness", "激增", "surge",
            "危险度", "danger", "⚠️", "断裂", "breakage",
        ]
        for kw in danger_keywords:
            if kw in text:
                is_anomaly = True
                reasons.append(kw)
                confidence += 0.1

        cinematic = "电影特征" in text or "cinematic" in text
        confidence = min(confidence, 0.85)

        result = {
            "is_anomaly": is_anomaly,
            "confidence": round(confidence, 2),
            "break_timestamp": None,
            "reason_zh": f"规则兜底判定: {'发现异常信号 (' + ', '.join(reasons) + ')' if is_anomaly else '未发现明显异常'}",
            "reason_en": f"Rule-based fallback: {'anomaly signals detected' if is_anomaly else 'no clear anomaly'}",
            "is_cinematic_false_alarm": cinematic and is_anomaly,
            "violated_contracts": [],
        }
        return json.dumps(result, ensure_ascii=False)

    # ── 结论解析 ──────────────────────────────────────
    def _parse_verdict(
        self,
        entity_id: int,
        raw_response: str,
        graph: TemporalGraph,
    ) -> AuditVerdict:
        """解析 Decision LLM 的 JSON 返回"""
        from ..utils.json_schema import extract_json_from_response

        parsed = extract_json_from_response(raw_response)

        if parsed is None:
            logger.warning(
                f"Failed to parse Decision LLM response for entity #{entity_id}: "
                f"{raw_response[:200]}"
            )
            return AuditVerdict(
                entity_id=entity_id,
                is_anomaly=False,
                confidence=0.0,
                reason_zh="Decision LLM 返回解析失败",
                reason_en="Failed to parse Decision LLM response",
                raw_llm_response=raw_response,
            )

        # 提取字段
        is_anomaly = bool(parsed.get("is_anomaly", False))
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        break_ts = parsed.get("break_timestamp")
        if break_ts is not None:
            try:
                break_ts = float(break_ts)
            except (TypeError, ValueError):
                # 支持 "01:01.33" 或 "MM:SS.SS" 格式
                try:
                    ts_str = str(break_ts)
                    if ":" in ts_str:
                        parts = ts_str.split(":")
                        break_ts = float(parts[0]) * 60 + float(parts[1])
                    else:
                        break_ts = None
                except Exception:
                break_ts = None

        # 查找断裂节点
        break_node = None
        if break_ts is not None:
            path = graph.get_entity_path(entity_id)
            closest_node = None
            min_diff = float("inf")
            for node in path:
                diff = abs(node["timestamp"] - break_ts)
                if diff < min_diff:
                    min_diff = diff
                    closest_node = node
            if closest_node:
                break_node = closest_node.get("node_id")

        is_cinematic = bool(parsed.get("is_cinematic_false_alarm", False))
        violated = parsed.get("violated_contracts", parsed.get("violated", []))
        if not isinstance(violated, list):
            violated = [str(violated)] if violated else []

        return AuditVerdict(
            entity_id=entity_id,
            is_anomaly=is_anomaly,
            confidence=confidence,
            break_node_id=break_node,
            break_timestamp=break_ts,
            reason_zh=str(parsed.get("reason", parsed.get("reason_zh", ""))),
            reason_en=str(parsed.get("reason", parsed.get("reason_en", ""))),
            is_cinematic_false_alarm=is_cinematic,
            violated_contracts=violated,
            raw_llm_response=raw_response,
        )

    # ── 业务契约 ──────────────────────────────────────
    def _get_contracts(self, scene_type: str) -> str:
        """获取场景对应的业务契约文本"""
        contracts = self.cfg.business_contracts

        scene_lower = scene_type.lower().strip()
        matched = []

        # 匹配场景
        for scene_key, rules in contracts.items():
            if scene_key == "default":
                continue
            if scene_key in scene_lower or scene_lower in scene_key:
                matched.extend(rules)

        # 如果没有匹配到场景，使用 default
        if not matched:
            matched = contracts.get("default", [])

        if not matched:
            return "  （无特定业务契约）"

        lines = []
        for i, rule in enumerate(matched, 1):
            lines.append(f"  {i}. {rule}")
        return "\n".join(lines)

    # ── 摘要生成 ──────────────────────────────────────
    @staticmethod
    def _build_summary(
        verdicts: list[AuditVerdict],
        anomaly_entities: list[int],
        cinematic_filtered: list[int],
        scene_type: str,
    ) -> tuple[str, str]:
        """生成视频级摘要"""
        if not verdicts:
            return "无有效实体可审计。", "No valid entities to audit."

        n_total = len(verdicts)
        n_anomaly = len(anomaly_entities)
        n_filtered = len(cinematic_filtered)

        if n_anomaly == 0:
            zh = f"场景: {scene_type or '未知'}。审计 {n_total} 个实体，未发现异常行为。"
            en = f"Scene: {scene_type or 'unknown'}. Audited {n_total} entities, no anomalies found."
            if n_filtered > 0:
                zh += f"（{n_filtered} 个疑似电影特征已过滤）"
                en += f" ({n_filtered} cinematic false alarms filtered)"
        else:
            anomaly_reasons = []
            for v in verdicts:
                if v.is_anomaly and not v.is_cinematic_false_alarm:
                    anomaly_reasons.append(
                        f"实体#{v.entity_id}: {v.reason_zh[:80]}"
                    )

            zh = (
                f"场景: {scene_type or '未知'}。审计 {n_total} 个实体，"
                f"发现 {n_anomaly} 个异常实体。\n"
                + "\n".join(anomaly_reasons[:3])
            )
            en = (
                f"Scene: {scene_type or 'unknown'}. Audited {n_total} entities, "
                f"found {n_anomaly} anomalous entities."
            )

        return zh, en

    # ── 规则兜底: 感知层信号补捕 ─────────────────────
    @staticmethod
    def _rule_based_perception_check(
        snapshots: list[dict],
    ) -> Optional[AuditVerdict]:
        """
        当 LLM 审计全判 Normal 时，用感知层信号进行规则兜底。

        触发条件 (满足任一即判异常):
          1. 任何帧的 visual_danger_score >= 0.45
          2. 任何实体的 is_suspicious == True
          3. 平均 visual_danger_score >= 0.22 (持续的低级别异常)
        """
        max_danger = 0.0
        suspicious_entities = []
        danger_frames = []
        all_dangers = []

        for snap in snapshots:
            danger = snap.get("visual_danger_score", 0.0)
            all_dangers.append(danger)
            ts = snap.get("timestamp", 0)

            if danger >= 0.45:
                danger_frames.append((ts, danger))

            for ent in snap.get("entities", []):
                if ent.get("is_suspicious"):
                    suspicious_entities.append({
                        "timestamp": ts,
                        "action": ent.get("action", ""),
                        "reason": ent.get("suspicious_reason", ""),
                        "portrait": ent.get("portrait", "")[:50],
                    })

            max_danger = max(max_danger, danger)

        avg_danger = sum(all_dangers) / len(all_dangers) if all_dangers else 0.0

        # 判定逻辑
        is_anomaly = False
        confidence = 0.0
        reasons = []

        if max_danger >= 0.45:
            is_anomaly = True
            confidence = max(confidence, min(max_danger + 0.2, 0.9))
            reasons.append(f"高危帧检测: max_danger={max_danger:.2f}")

        if suspicious_entities:
            is_anomaly = True
            n_sus = len(suspicious_entities)
            confidence = max(confidence, min(0.5 + n_sus * 0.1, 0.9))
            top = suspicious_entities[0]
            reasons.append(
                f"感知层标记可疑({n_sus}处): {top['action']} ({top['reason']})"
            )

        if avg_danger >= 0.22 and not is_anomaly:
            is_anomaly = True
            confidence = max(confidence, 0.5)
            reasons.append(f"持续异常信号: avg_danger={avg_danger:.2f}")

        if not is_anomaly:
            return None

        # ── 全局动能信号: 用“高危时间簇”代替全局 min~max ──
        # 避免把零散可疑帧连成过长区间，导致 IoU / 帧级 AUC 偏低
        snap_scores = []
        for snap in snapshots:
            ts = float(snap.get("timestamp", 0.0))
            danger = float(snap.get("visual_danger_score", 0.0))
            sus_count = sum(1 for ent in snap.get("entities", []) if ent.get("is_suspicious"))
            if danger >= 0.45 or sus_count > 0:
                score = danger + min(0.3, 0.08 * sus_count)
                snap_scores.append((ts, score))

        anomaly_start = 0.0
        anomaly_end = 0.0
        anomaly_intervals: list[tuple[float, float]] = []
        break_ts = None
        if snap_scores:
            snap_scores.sort(key=lambda x: x[0])
            clusters = []
            curr = [snap_scores[0]]
            max_gap_sec = 3.0
            for item in snap_scores[1:]:
                if item[0] - curr[-1][0] <= max_gap_sec:
                    curr.append(item)
                else:
                    clusters.append(curr)
                    curr = [item]
            clusters.append(curr)

            # 选 top-k 高危簇，避免单段过长拖低 IoU
            ranked = sorted(
                clusters,
                key=lambda c: (sum(x[1] for x in c) / max(1, len(c))) * len(c),
                reverse=True,
            )
            top_k = ranked[: min(3, len(ranked))]

            for c in top_k:
                s = max(0.0, c[0][0] - 1.0)
                e = c[-1][0] + 1.0
                anomaly_intervals.append((s, e))

            # 默认 break_timestamp 取最强簇中的峰值
            best = top_k[0]
            break_ts = max(best, key=lambda x: x[1])[0]
            anomaly_start, anomaly_end = anomaly_intervals[0]

        reason_zh = "规则兜底: " + "; ".join(reasons)

        verdict = AuditVerdict(
            entity_id=-1,
            is_anomaly=True,
            confidence=round(confidence, 2),
            break_timestamp=break_ts,
            reason_zh=reason_zh,
            reason_en=f"Rule-based fallback: {'; '.join(reasons)}",
        )
        # 附加全局异常区间信息 (供定位器使用)
        verdict._anomaly_start = anomaly_start
        verdict._anomaly_end = anomaly_end
        verdict._anomaly_intervals = anomaly_intervals
        return verdict

    # ── 资源释放 ──────────────────────────────────────
    def cleanup(self):
        """释放 Decision LLM 资源"""
        if self._decision_client is not None:
            model, tokenizer = self._decision_client
            del model
            del tokenizer
            self._decision_client = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Decision LLM resources released.")
