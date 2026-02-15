"""
V4.0 消融实验 — 对比不同策略组合的检测效果

消融维度:
  A. v4 完整 (Decision LLM + 业务契约)
  B. v4 无契约 (Decision LLM, 业务契约置空)
  C. v3 fallback (纯加权公式, 无 LLM)

测试视频:
  1. Robbery137_x264.mp4 (抢劫, 应检测为异常)
  2. Normal_Videos_943_x264.mp4 (正常, 应检测为正常)
"""

import sys
import json
import time
import logging
import copy
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,  # 减少输出
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 只对关键模块显示 INFO
for name in ["__main__", "v3.perception.vllm_client"]:
    logging.getLogger(name).setLevel(logging.INFO)

logger = logging.getLogger(__name__)

from v3.pipeline import VideoAnomalyPipeline
from v3.config import DecisionConfig, OUTPUT_DIR


def run_ablation(
    video_path: str,
    mode: str,
    label: str,
    backend: str = "server",
    api_base: str = "http://localhost:8000",
    clear_contracts: bool = False,
):
    """运行单次消融实验"""
    logger.info(f"\n{'='*60}")
    logger.info(f"  [{label}] mode={mode}, contracts={'OFF' if clear_contracts else 'ON'}")
    logger.info(f"  video: {Path(video_path).name}")
    logger.info(f"{'='*60}")

    pipeline = VideoAnomalyPipeline(
        model_name="qwen2-vl-7b",
        mode=mode,
        backend=backend,
        api_base=api_base,
        max_workers=16,
        save_intermediate=False,
    )

    # 消融: 清空业务契约
    if clear_contracts:
        pipeline.decision_cfg.business_contracts = {"default": []}

    t0 = time.time()
    result = pipeline.process_video(video_path)
    elapsed = time.time() - t0

    pipeline.cleanup()

    return {
        "label": label,
        "video": Path(video_path).name,
        "mode": mode,
        "contracts": not clear_contracts,
        "status": result.get("status", "?"),
        "score": result.get("anomaly_score", 0),
        "num_anomaly_entities": len([
            e for e in result.get("entity_results", []) if e.get("is_anomaly")
        ]),
        "num_entities": result.get("num_entities_tracked", 0),
        "num_segments": len(result.get("anomaly_segments", [])),
        "scene": result.get("scene_type", ""),
        "time_sec": round(elapsed, 1),
        "explanation": result.get("anomaly_explanation", "")[:120],
    }


def main():
    videos = {
        "异常(抢劫)": "/data/liuzhe/EventVAD/src/event_seg/videos/Robbery137_x264.mp4",
        "正常(监控)": "/data/liuzhe/EventVAD/src/event_seg/videos/ucf_crime/Testing_Normal_Videos_Anomaly/Normal_Videos_943_x264.mp4",
    }

    ablations = [
        ("A: v4完整(LLM+契约)", "v4", False),
        ("B: v4无契约(仅LLM)",  "v4", True),
        ("C: v3纯加权(无LLM)",  "v3", False),
    ]

    all_results = []

    for video_label, video_path in videos.items():
        if not Path(video_path).exists():
            logger.warning(f"跳过: {video_path} 不存在")
            continue

        for abl_label, mode, clear_contracts in ablations:
            try:
                result = run_ablation(
                    video_path, mode, f"{video_label} | {abl_label}",
                    clear_contracts=clear_contracts,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"实验失败: {abl_label} on {video_label}: {e}")
                all_results.append({
                    "label": f"{video_label} | {abl_label}",
                    "status": f"ERROR: {e}",
                })

    # ── 打印消融结果对比表 ──
    print(f"\n{'='*100}")
    print(f"  消融实验结果对比")
    print(f"{'='*100}")
    print(f"{'实验配置':<35} {'视频':<18} {'状态':<10} {'分数':>6} {'异常实体':>8} {'片段数':>6} {'耗时':>6}")
    print(f"{'-'*100}")

    for r in all_results:
        if "score" not in r:
            print(f"{r.get('label', '?'):<35} {r.get('status', '?')}")
            continue
        print(
            f"{r['label']:<35} "
            f"{r['video']:<18} "
            f"{r['status']:<10} "
            f"{r['score']:>6.4f} "
            f"{r['num_anomaly_entities']:>8} "
            f"{r['num_segments']:>6} "
            f"{r['time_sec']:>5.1f}s"
        )

    print(f"{'='*100}")

    # 保存详细结果
    out_path = OUTPUT_DIR / "ablation_results.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
