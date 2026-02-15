"""
V5 Bad Case 自动分析脚本

读取评估结果 JSON，自动生成详细的 bad case 分析报告，包括：
  - 与上一次运行的对比
  - FN / FP 逐例根因分析
  - TP 质量分析 (IoU)
  - 类别级分析
  - 改进建议优先级排序

用法:
  python -m v5.analyze_bad_cases \
      --current /path/to/current/results_v5.json \
      --previous /path/to/previous/results_v5.json \
      --output /path/to/bad_case_analysis.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_root_cause(detail: dict) -> tuple[str, str]:
    """
    根据结果详情自动分类根因。
    返回 (root_cause_code, explanation)
    """
    entities = detail.get("stats", {}).get("entities", 0)
    triggers = detail.get("stats", {}).get("triggers", 0)
    entity_verdicts = detail.get("entity_verdicts", [])
    pred_anomaly = detail.get("pred_anomaly", False)
    gt_anomaly = detail.get("gt_anomaly", False)

    # ── FN 分类 ──
    if gt_anomaly and not pred_anomaly:
        if entities == 0:
            return "ZERO_ENTITY_DETECTION", (
                f"Motion Extractor 未检测到任何运动区域 (0 entities, 0 triggers)。"
                f"Pipeline 直接输出 NORMAL，无法挽回。"
            )
        if triggers == 0:
            return "ZERO_TRIGGERS", (
                f"检测到 {entities} 个实体但 0 个触发器。NodeTrigger 未被激活。"
            )

        # 有实体和触发，但所有实体判正常
        all_normal = all(not v.get("is_anomaly", False) for v in entity_verdicts)
        if all_normal:
            # 检查是否所有 danger_score 都很低
            max_danger = 0.0
            for d in detail.get("entity_verdicts", []):
                # entity_verdicts 不直接含 danger_score，需从 verdict 推断
                pass

            return "VLLM_UNDER_DESCRIPTION", (
                f"检测到 {entities} 个实体、{triggers} 个触发器，"
                f"但所有实体均被判为正常。VLLM 语义描述未识别异常行为。"
            )

        return "DECISION_ERROR", (
            f"检测到异常实体但最终决策未输出异常。"
        )

    # ── FP 分类 ──
    if not gt_anomaly and pred_anomaly:
        anomaly_entities = [v for v in entity_verdicts if v.get("is_anomaly", False)]
        reasons = [v.get("reason", "") for v in anomaly_entities]
        reason_text = " | ".join(reasons)

        has_discordance = any(
            "motion energy" in r.lower() or "discordance" in r.lower()
            or "blind spot" in r.lower() or "threshold" in r.lower()
            for r in reasons
        )
        has_semantic = any(
            "fire" in r.lower() or "weapon" in r.lower()
            or "aggressive" in r.lower() or "pushing" in r.lower()
            or "fighting" in r.lower() or "confrontational" in r.lower()
            or "cash register" in r.lower() or "interaction" in r.lower()
            for r in reasons
        )

        if has_discordance and not has_semantic:
            return "DISCORDANCE_FALSE_ALARM", (
                f"Discordance 机制误触发。"
                f"{len(anomaly_entities)} 个实体因物理动能超标被判异常，"
                f"但 VLLM 语义正确识别为正常活动。"
            )
        elif has_semantic:
            return "VLLM_HALLUCINATION", (
                f"VLLM 语义描述产生幻觉或过度解读。"
                f"{len(anomaly_entities)} 个实体基于语义内容被判异常。"
                f"理由: {reason_text[:200]}"
            )
        else:
            return "DECISION_FALSE_ALARM", (
                f"{len(anomaly_entities)} 个实体被判异常，导致正常视频误报。"
                f"理由: {reason_text[:200]}"
            )

    return "OK", ""


def analyze_single_video(detail: dict) -> dict:
    """分析单个视频的评估结果"""
    filename = detail.get("filename", "unknown")
    category = detail.get("category", "Unknown")
    gt_anomaly = detail.get("gt_anomaly", False)
    pred_anomaly = detail.get("pred_anomaly", False)
    pred_score = detail.get("pred_score", 0.0)
    entity_verdicts = detail.get("entity_verdicts", [])
    stats = detail.get("stats", {})
    iou_soft = detail.get("iou_soft")
    iou_hyst = detail.get("iou_hysteresis")
    time_sec = detail.get("time_sec", 0)
    total_frames = detail.get("total_frames", 0)
    fps = detail.get("fps", 30.0)

    # 分类结果
    if gt_anomaly and pred_anomaly:
        result_type = "TP"
    elif gt_anomaly and not pred_anomaly:
        result_type = "FN"
    elif not gt_anomaly and pred_anomaly:
        result_type = "FP"
    else:
        result_type = "TN"

    root_cause, explanation = classify_root_cause(detail)

    anomaly_entities = [v for v in entity_verdicts if v.get("is_anomaly", False)]
    normal_entities = [v for v in entity_verdicts if not v.get("is_anomaly", False)]

    result = {
        "filename": filename,
        "category": category,
        "result_type": result_type,
        "gt_anomaly": gt_anomaly,
        "pred_anomaly": pred_anomaly,
        "pred_score": pred_score,
        "entities": stats.get("entities", 0),
        "triggers": stats.get("triggers", 0),
        "nodes": stats.get("nodes", 0),
        "edges": stats.get("edges", 0),
        "total_frames": total_frames,
        "duration_sec": round(total_frames / max(fps, 1), 1),
        "processing_time_sec": time_sec,
    }

    if iou_soft is not None:
        result["iou_soft"] = round(iou_soft, 4)
    if iou_hyst is not None:
        result["iou_hysteresis"] = round(iou_hyst, 4)

    if result_type in ("FN", "FP"):
        result["root_cause"] = root_cause
        result["detail"] = explanation
        result["anomaly_entities"] = [
            {
                "entity_id": v.get("entity_id"),
                "is_anomaly": v.get("is_anomaly"),
                "confidence": v.get("confidence"),
                "reason": v.get("reason", "")[:200],
                "anomaly_start_sec": v.get("anomaly_start_sec", 0),
                "anomaly_end_sec": v.get("anomaly_end_sec", 0),
            }
            for v in anomaly_entities
        ]
        result["normal_entities_summary"] = [
            {
                "entity_id": v.get("entity_id"),
                "confidence": v.get("confidence"),
                "reason": v.get("reason", "")[:100],
            }
            for v in normal_entities[:5]  # 只取前5个节省空间
        ]

    if result_type == "TP":
        result["anomaly_entities"] = [
            {
                "entity_id": v.get("entity_id"),
                "confidence": v.get("confidence"),
                "reason": v.get("reason", "")[:200],
                "anomaly_start_sec": v.get("anomaly_start_sec", 0),
                "anomaly_end_sec": v.get("anomaly_end_sec", 0),
            }
            for v in anomaly_entities
        ]

    return result


def compare_runs(current_details: list, previous_details: list) -> dict:
    """对比两次运行结果"""
    prev_map = {d["filename"]: d for d in previous_details}
    curr_map = {d["filename"]: d for d in current_details}

    flips = []
    iou_changes = []

    for fname in curr_map:
        if fname not in prev_map:
            continue

        curr = curr_map[fname]
        prev = prev_map[fname]

        curr_correct = curr.get("pred_anomaly") == curr.get("gt_anomaly")
        prev_correct = prev.get("pred_anomaly") == prev.get("gt_anomaly")

        # 分类变化
        def _classify(d):
            if d["gt_anomaly"] and d["pred_anomaly"]:
                return "TP"
            elif d["gt_anomaly"] and not d["pred_anomaly"]:
                return "FN"
            elif not d["gt_anomaly"] and d["pred_anomaly"]:
                return "FP"
            else:
                return "TN"

        prev_type = _classify(prev)
        curr_type = _classify(curr)

        if prev_type != curr_type:
            improved = curr_correct and not prev_correct
            regressed = not curr_correct and prev_correct

            # 获取当前 entity_verdicts 的简要理由
            curr_reasons = []
            for v in curr.get("entity_verdicts", []):
                if v.get("is_anomaly"):
                    curr_reasons.append(
                        f"Entity #{v.get('entity_id')}: {v.get('reason', '')[:80]}"
                    )

            flips.append({
                "filename": fname,
                "change": f"{prev_type} → {curr_type}" + (" ✅" if improved else " ⚠️" if regressed else ""),
                "improved": improved,
                "regressed": regressed,
                "entity_reasons": curr_reasons[:3],
            })

        # IoU 变化
        curr_iou_s = curr.get("iou_soft")
        prev_iou_s = prev.get("iou_soft")
        if curr_iou_s is not None and prev_iou_s is not None:
            delta = curr_iou_s - prev_iou_s
            if abs(delta) > 0.02:
                iou_changes.append({
                    "filename": fname,
                    "old_iou": round(prev_iou_s, 4),
                    "new_iou": round(curr_iou_s, 4),
                    "delta": round(delta, 4),
                    "tag": "✅" if delta > 0 else "⚠️",
                })

    iou_changes.sort(key=lambda x: -abs(x["delta"]))

    return {
        "video_level_flips": flips,
        "iou_changes": iou_changes[:15],  # Top 15 biggest changes
    }


def build_root_cause_summary(analyses: list) -> dict:
    """汇总根因分布"""
    rc_counter = Counter()
    rc_videos = defaultdict(list)

    for a in analyses:
        if a["result_type"] in ("FN", "FP"):
            rc = a.get("root_cause", "UNKNOWN")
            rc_counter[rc] += 1
            rc_videos[rc].append(a["filename"])

    summary = {}
    descriptions = {
        "ZERO_ENTITY_DETECTION": {
            "description": "Motion Extractor 完全未检测到运动区域 (0 entities)。Pipeline 直接输出 NORMAL。",
            "fix_suggestions": [
                "降低 diff_threshold (当前25) → 对低对比度场景使用自适应阈值",
                "降低 min_region_area (当前1500) → 允许更小的运动区域",
                "添加基于光流的运动检测作为帧差法的补充",
                "添加 fallback 机制：当 0 entities 时用全图进行全局语义分析",
            ],
        },
        "ZERO_TRIGGERS": {
            "description": "有实体但无触发器。NodeTrigger 条件未满足。",
            "fix_suggestions": [
                "降低 embedding_jump_threshold",
                "缩短 heartbeat_interval_sec",
            ],
        },
        "VLLM_UNDER_DESCRIPTION": {
            "description": "VLLM 语义描述不足，未识别关键异常行为 (fire/fight/intrusion 等)。",
            "fix_suggestions": [
                "优化 VLLM prompt，强制要求描述: 是否有火焰/烟雾/打斗/入侵行为",
                "在 prompt 中加入 'Look carefully for fire, flames, smoke, fighting, trespassing'",
                "添加 CLIP zero-shot fire/smoke/fight 专用检测器作为辅助信号",
                "增大 crop 区域或使用多尺度输入",
            ],
        },
        "DISCORDANCE_FALSE_ALARM": {
            "description": "Discordance 机制 (高物理动能 + 低语义danger) 在正常高运动场景误触发。",
            "fix_suggestions": [
                "添加多实体一致性投票: 当 >70% 实体判正常时，单个 discordance 不翻转整体",
                "提高 discordance 的动能阈值 (μ+3σ → μ+4σ)",
                "降低 discordance 判定的 confidence (0.90→0.60)",
                "结合场景类型: indoor/retail 场景正常动能范围更大",
            ],
        },
        "VLLM_HALLUCINATION": {
            "description": "VLLM 语义描述产生幻觉或过度解读，将正常行为误判为异常。",
            "fix_suggestions": [
                "在 prompt 中强调 'Only flag truly dangerous or illegal behavior'",
                "添加多轮验证: 对 suspicious=True 的结果进行二次确认",
                "降低单次语义判定的权重",
            ],
        },
        "DECISION_FALSE_ALARM": {
            "description": "Decision 层将正常视频误判为异常。",
            "fix_suggestions": [
                "提高 anomaly_confidence_threshold",
                "添加多实体投票机制",
            ],
        },
        "DECISION_ERROR": {
            "description": "检测到异常信号但最终决策未输出异常。",
            "fix_suggestions": [
                "检查 Decision Auditor 逻辑",
                "降低 anomaly_confidence_threshold",
            ],
        },
    }

    for rc, count in rc_counter.most_common():
        info = descriptions.get(rc, {"description": rc, "fix_suggestions": []})
        summary[rc] = {
            "count": count,
            "affected_videos": rc_videos[rc],
            "description": info["description"],
            "fix_suggestions": info["fix_suggestions"],
        }

    return summary


def build_category_analysis(analyses: list) -> dict:
    """按类别汇总分析"""
    cats = defaultdict(lambda: {
        "total": 0, "correct": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0,
        "ious_soft": [], "ious_hyst": [],
        "fn_videos": [], "fp_videos": [],
    })

    for a in analyses:
        cat = a["category"]
        cats[cat]["total"] += 1
        is_correct = a["result_type"] in ("TP", "TN")
        if is_correct:
            cats[cat]["correct"] += 1
        cats[cat][a["result_type"].lower()] += 1

        if a.get("iou_soft") is not None and a["result_type"] == "TP":
            cats[cat]["ious_soft"].append(a["iou_soft"])
        if a.get("iou_hysteresis") is not None and a["result_type"] == "TP":
            cats[cat]["ious_hyst"].append(a["iou_hysteresis"])

        if a["result_type"] == "FN":
            cats[cat]["fn_videos"].append(a["filename"])
        if a["result_type"] == "FP":
            cats[cat]["fp_videos"].append(a["filename"])

    result = {}
    for cat in sorted(cats.keys()):
        s = cats[cat]
        result[cat] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0,
            "tp": s["tp"], "fn": s["fn"], "fp": s["fp"], "tn": s["tn"],
            "mean_iou_soft": round(float(np.mean(s["ious_soft"])), 4) if s["ious_soft"] else None,
            "mean_iou_hyst": round(float(np.mean(s["ious_hyst"])), 4) if s["ious_hyst"] else None,
            "fn_videos": s["fn_videos"],
            "fp_videos": s["fp_videos"],
        }

    return result


def build_iou_analysis(analyses: list) -> dict:
    """IoU 详细分析"""
    tp_analyses = [a for a in analyses if a["result_type"] == "TP"]

    excellent = []
    moderate = []
    zero = []

    for a in tp_analyses:
        iou_s = a.get("iou_soft", 0)
        iou_h = a.get("iou_hysteresis", 0)
        best_iou = max(iou_s or 0, iou_h or 0)

        entry = {
            "filename": a["filename"],
            "category": a["category"],
            "iou_soft": iou_s,
            "iou_hysteresis": iou_h,
            "pred_score": a["pred_score"],
            "entities": a["entities"],
        }

        if best_iou >= 0.5:
            excellent.append(entry)
        elif best_iou > 0.05:
            moderate.append(entry)
        else:
            zero.append(entry)

    all_ious_soft = [a.get("iou_soft", 0) for a in tp_analyses if a.get("iou_soft") is not None]
    all_ious_hyst = [a.get("iou_hysteresis", 0) for a in tp_analyses if a.get("iou_hysteresis") is not None]

    return {
        "overall_mean_iou_soft": round(float(np.mean(all_ious_soft)), 4) if all_ious_soft else 0,
        "overall_mean_iou_hyst": round(float(np.mean(all_ious_hyst)), 4) if all_ious_hyst else 0,
        "excellent_iou_above_0.5": sorted(excellent, key=lambda x: -(x.get("iou_hysteresis") or 0)),
        "moderate_iou_0.05_to_0.5": sorted(moderate, key=lambda x: -(x.get("iou_hysteresis") or 0)),
        "near_zero_iou": zero,
        "total_tp": len(tp_analyses),
        "count_excellent": len(excellent),
        "count_moderate": len(moderate),
        "count_zero": len(zero),
    }


def build_priority_actions(
    root_causes: dict,
    category_analysis: dict,
    iou_analysis: dict,
    metrics: dict,
) -> list:
    """基于分析结果自动生成优先级行动建议"""
    actions = []

    # P0: 0-entity 问题
    if "ZERO_ENTITY_DETECTION" in root_causes:
        rc = root_causes["ZERO_ENTITY_DETECTION"]
        actions.append({
            "priority": "P0",
            "action": "修复 0-entity 检测的 fallback 机制",
            "reason": f"{rc['count']} 个视频因 0 entities 必然漏判: {', '.join(rc['affected_videos'])}",
            "expected_impact": f"消除 {rc['count']} 个 FN, recall 提升约 +{rc['count']*5}%",
            "suggested_approach": rc["fix_suggestions"],
        })

    # P0: VLLM 描述不足
    if "VLLM_UNDER_DESCRIPTION" in root_causes:
        rc = root_causes["VLLM_UNDER_DESCRIPTION"]
        actions.append({
            "priority": "P0",
            "action": "增强 VLLM 对异常行为的识别能力",
            "reason": f"{rc['count']} 个视频因 VLLM 语义描述不足而漏判: {', '.join(rc['affected_videos'])}",
            "expected_impact": f"消除 {rc['count']}+ 个 FN",
            "suggested_approach": rc["fix_suggestions"],
        })

    # P0: Discordance 误报
    if "DISCORDANCE_FALSE_ALARM" in root_causes:
        rc = root_causes["DISCORDANCE_FALSE_ALARM"]
        actions.append({
            "priority": "P0",
            "action": "优化 discordance 机制，减少正常场景误报",
            "reason": f"{rc['count']} 个正常视频因 discordance 机制误触发: {', '.join(rc['affected_videos'])}",
            "expected_impact": f"FP 减少 {rc['count']}, Precision 提升",
            "suggested_approach": rc["fix_suggestions"],
        })

    # P0: VLLM 幻觉
    if "VLLM_HALLUCINATION" in root_causes:
        rc = root_causes["VLLM_HALLUCINATION"]
        actions.append({
            "priority": "P0",
            "action": "抑制 VLLM 语义幻觉",
            "reason": f"{rc['count']} 个视频因 VLLM 幻觉导致 FP: {', '.join(rc['affected_videos'])}",
            "expected_impact": f"FP 减少 {rc['count']}",
            "suggested_approach": rc["fix_suggestions"],
        })

    # P1: IoU 改善
    zero_iou_count = iou_analysis.get("count_zero", 0)
    if zero_iou_count > 0:
        actions.append({
            "priority": "P1",
            "action": "改善异常区间定位精度 (IoU)",
            "reason": f"{zero_iou_count} 个 TP 视频 IoU 接近 0，视频级正确但时间定位完全偏移",
            "expected_impact": "Frame AUC 提升, Mean IoU 提升",
            "suggested_approach": [
                "增大帧级分数的时间扩散半径 (σ=3s → 5s)",
                "使用更精确的 entity_verdict anomaly_start/end_sec",
                "对 discordance 检出的异常，根据动能峰值定位时间区间",
            ],
        })

    # P1: 弱类别
    for cat, info in category_analysis.items():
        if cat == "Normal":
            continue
        if info["accuracy"] < 0.5 and info["total"] > 0:
            actions.append({
                "priority": "P1",
                "action": f"增强 {cat} 类别检测能力",
                "reason": f"{cat} accuracy={info['accuracy']:.2f} ({info['correct']}/{info['total']}), FN: {', '.join(info['fn_videos'])}",
                "expected_impact": f"{cat} recall 提升",
                "suggested_approach": [
                    f"分析 {cat} FN 的具体原因并针对性优化",
                    "在 prompt 中增加针对性的检测要求",
                ],
            })

    return actions


def generate_analysis(
    current_path: str,
    previous_path: str = None,
    output_path: str = None,
) -> dict:
    """生成完整的 bad case 分析报告"""
    current = load_results(current_path)
    current_metrics = current.get("metrics", {})
    current_details = current.get("details", [])

    # ── 逐视频分析 ──
    analyses = [analyze_single_video(d) for d in current_details]

    # 分类
    fn_list = [a for a in analyses if a["result_type"] == "FN"]
    fp_list = [a for a in analyses if a["result_type"] == "FP"]
    tp_list = [a for a in analyses if a["result_type"] == "TP"]
    tn_list = [a for a in analyses if a["result_type"] == "TN"]

    # ── 与上次对比 ──
    comparison = {}
    if previous_path:
        previous = load_results(previous_path)
        prev_metrics = previous.get("metrics", {})
        prev_details = previous.get("details", [])

        metric_keys = ["accuracy", "precision", "recall", "f1", "frame_auc",
                        "video_auc", "mean_iou_soft", "mean_iou_hysteresis"]
        metric_changes = {}
        for key in metric_keys:
            old_val = prev_metrics.get(key, 0)
            new_val = current_metrics.get(key, 0)
            delta = new_val - old_val
            tag = "✅" if delta > 0.005 else "⚠️" if delta < -0.005 else "━"
            metric_changes[key] = {
                "old": round(old_val, 4),
                "new": round(new_val, 4),
                "delta": f"{delta:+.4f} {tag}",
            }

        cm_keys = ["tp", "fn", "fp", "tn"]
        cm_changes = {}
        for key in cm_keys:
            old_val = prev_metrics.get(key, 0)
            new_val = current_metrics.get(key, 0)
            delta = new_val - old_val
            tag = ""
            if key in ("tp", "tn"):
                tag = "✅" if delta > 0 else "⚠️" if delta < 0 else ""
            else:
                tag = "✅" if delta < 0 else "⚠️" if delta > 0 else ""
            cm_changes[key] = {"old": old_val, "new": new_val, "delta": f"{delta:+d} {tag}".strip()}

        run_comparison = compare_runs(current_details, prev_details)

        comparison = {
            "previous_run": previous_path,
            "current_run": current_path,
            "metric_changes": metric_changes,
            "confusion_matrix_changes": cm_changes,
            **run_comparison,
        }

    # ── 汇总 ──
    root_causes = build_root_cause_summary(analyses)
    category_analysis = build_category_analysis(analyses)
    iou_analysis = build_iou_analysis(analyses)
    priority_actions = build_priority_actions(
        root_causes, category_analysis, iou_analysis, current_metrics
    )

    # ── 组装报告 ──
    report = {
        "eval_summary": {
            "version": "v5",
            "run_dir": current_metrics.get("run_dir", ""),
            "run_timestamp": current_metrics.get("run_timestamp", ""),
            "total_videos": current_metrics.get("total", 0),
            "accuracy": current_metrics.get("accuracy", 0),
            "precision": current_metrics.get("precision", 0),
            "recall": current_metrics.get("recall", 0),
            "f1": current_metrics.get("f1", 0),
            "frame_auc": current_metrics.get("frame_auc", 0),
            "video_auc": current_metrics.get("video_auc", 0),
            "mean_iou_soft": current_metrics.get("mean_iou_soft", 0),
            "mean_iou_hysteresis": current_metrics.get("mean_iou_hysteresis", 0),
            "confusion_matrix": {
                "TP": current_metrics.get("tp", 0),
                "FN": current_metrics.get("fn", 0),
                "FP": current_metrics.get("fp", 0),
                "TN": current_metrics.get("tn", 0),
            },
        },
    }

    if comparison:
        report["comparison_with_previous_run"] = comparison

    report["false_negatives"] = sorted(fn_list, key=lambda x: x["category"])
    report["false_positives"] = fp_list
    report["true_positives_analysis"] = {
        "count": len(tp_list),
        "details": sorted(tp_list, key=lambda x: -(x.get("iou_hysteresis") or 0)),
    }
    report["true_negatives_count"] = len(tn_list)
    report["root_cause_summary"] = root_causes
    report["category_analysis"] = category_analysis
    report["iou_analysis"] = iou_analysis
    report["priority_action_items"] = priority_actions

    # ── 总体评估 ──
    recall = current_metrics.get("recall", 0)
    precision = current_metrics.get("precision", 0)
    strengths = []
    weaknesses = []

    for cat, info in category_analysis.items():
        if cat == "Normal":
            if info["accuracy"] < 0.9:
                weaknesses.append(f"Normal 准确率仅 {info['accuracy']:.2f} ({info['correct']}/{info['total']}), FP: {', '.join(info['fp_videos'])}")
            else:
                strengths.append(f"Normal 准确率 {info['accuracy']:.2f} ({info['correct']}/{info['total']})")
        else:
            if info["accuracy"] >= 0.9:
                strengths.append(f"{cat} recall 满分 ({info['correct']}/{info['total']})")
            elif info["accuracy"] >= 0.7:
                strengths.append(f"{cat} recall {info['accuracy']:.2f} ({info['correct']}/{info['total']})")
            elif info["accuracy"] < 0.5 and info["total"] > 0:
                weaknesses.append(f"{cat} 检测困难 ({info['correct']}/{info['total']}), FN: {', '.join(info['fn_videos'])}")

    if iou_analysis.get("count_zero", 0) > len(tp_list) * 0.3:
        weaknesses.append(f"{iou_analysis['count_zero']}/{len(tp_list)} 个 TP 视频 IoU≈0，时间定位差")

    report["overall_assessment"] = {
        "current_performance": (
            f"Accuracy={current_metrics.get('accuracy', 0):.4f}, "
            f"F1={current_metrics.get('f1', 0):.4f}, "
            f"Recall={recall:.4f}, "
            f"Precision={precision:.4f}, "
            f"Frame AUC={current_metrics.get('frame_auc', 0):.4f}, "
            f"Video AUC={current_metrics.get('video_auc', 0):.4f}"
        ),
        "key_strengths": strengths,
        "key_weaknesses": weaknesses,
    }

    # ── 保存 ──
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Bad case analysis saved to {out}")

    return report


def main():
    parser = argparse.ArgumentParser(description="V5 Bad Case Analysis")
    parser.add_argument("--current", required=True, help="当前评估结果 JSON 路径")
    parser.add_argument("--previous", default=None, help="上一次评估结果 JSON 路径 (可选，用于对比)")
    parser.add_argument("--output", default=None, help="输出分析报告路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if not args.output:
        # 默认输出到当前结果同目录
        args.output = str(Path(args.current).parent / "bad_case_analysis.json")

    report = generate_analysis(
        current_path=args.current,
        previous_path=args.previous,
        output_path=args.output,
    )

    # 打印摘要
    summary = report["eval_summary"]
    print(f"\n{'='*70}")
    print(f"  V5 Bad Case Analysis Report")
    print(f"{'='*70}")
    print(f"  Accuracy:  {summary['accuracy']:.4f}")
    print(f"  Precision: {summary['precision']:.4f}")
    print(f"  Recall:    {summary['recall']:.4f}")
    print(f"  F1:        {summary['f1']:.4f}")
    print(f"  Frame AUC: {summary['frame_auc']:.4f}")
    print(f"  Video AUC: {summary['video_auc']:.4f}")
    print(f"  TP={summary['confusion_matrix']['TP']} "
          f"FN={summary['confusion_matrix']['FN']} "
          f"FP={summary['confusion_matrix']['FP']} "
          f"TN={summary['confusion_matrix']['TN']}")
    print()

    # FN 摘要
    fn = report.get("false_negatives", [])
    if fn:
        print(f"  ❌ False Negatives ({len(fn)}):")
        for item in fn:
            rc = item.get("root_cause", "UNKNOWN")
            print(f"    - {item['filename']} [{item['category']}] → {rc}")
            print(f"      entities={item['entities']} triggers={item['triggers']}")
        print()

    # FP 摘要
    fp = report.get("false_positives", [])
    if fp:
        print(f"  ⚠️  False Positives ({len(fp)}):")
        for item in fp:
            rc = item.get("root_cause", "UNKNOWN")
            print(f"    - {item['filename']} [{item['category']}] → {rc}")
            print(f"      score={item['pred_score']:.2f} entities={item['entities']}")
        print()

    # 对比摘要
    comp = report.get("comparison_with_previous_run", {})
    if comp:
        flips = comp.get("video_level_flips", [])
        if flips:
            print(f"  🔄 Video-level Flips ({len(flips)}):")
            for flip in flips:
                print(f"    - {flip['filename']}: {flip['change']}")
            print()

    # 优先行动
    actions = report.get("priority_action_items", [])
    if actions:
        print(f"  📋 Priority Actions:")
        for a in actions:
            print(f"    [{a['priority']}] {a['action']}")
            print(f"         → {a['reason'][:100]}")
        print()

    print(f"  Report saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
