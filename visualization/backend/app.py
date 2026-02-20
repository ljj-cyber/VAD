"""
EventVAD V5 可视化系统 — FastAPI 后端

提供 REST API:
  - /api/videos               列出所有已分析的视频
  - /api/video/{name}/metadata 视频元数据 + 全局判定
  - /api/video/{name}/entities 所有实体的演化图数据
  - /api/entities/{name}/{id}/trajectory 单实体动能轨迹
  - /api/audit/{name}/narrative 审计叙事
  - /api/frames/{name}/{frame_idx} 关键帧图像
  - /api/video/{name}/stream   视频流帧抽样
"""

import json
import math
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EventVAD V5 Visualization API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路径配置 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # EventVAD/
OUTPUT_DIR = PROJECT_ROOT / "output" / "v5"
DEMO_DIR = OUTPUT_DIR / "demo_vlm_input"
EVAL_DIR = OUTPUT_DIR / "eval_ucf_crime"
VIDEO_BASE = PROJECT_ROOT / "src" / "event_seg" / "videos"


# ── 数据缓存 ──────────────────────────────────────────
_cache: dict[str, dict] = {}


def _load_result(video_name: str) -> dict:
    """加载单个视频的 result.json"""
    if video_name in _cache:
        return _cache[video_name]

    # 优先查找独立 result.json
    result_path = OUTPUT_DIR / video_name / "result.json"
    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _cache[video_name] = data
        return data

    # 从 eval 汇总文件中查找
    for run_dir in sorted(EVAL_DIR.glob("run_*"), reverse=True):
        rp = run_dir / "results_v5.json"
        if rp.exists():
            with open(rp, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            for detail in eval_data.get("details", []):
                fn = detail.get("filename", "")
                name = fn.replace(".mp4", "")
                if name == video_name:
                    # 构造兼容格式
                    data = _convert_eval_detail(detail)
                    _cache[video_name] = data
                    return data

    # 从 demo summary 构造最小化结果（用于 Assault001 等只有 demo 的视频）
    demo = _load_demo_summary(video_name)
    if demo:
        data = _convert_demo_summary(demo)
        _cache[video_name] = data
        return data

    raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found")


def _convert_eval_detail(detail: dict) -> dict:
    """将 eval 的 detail 条目转为标准 result 格式"""
    entity_verdicts = detail.get("entity_verdicts", [])
    anomaly_eids = [v["entity_id"] for v in entity_verdicts if v.get("is_anomaly")]

    summary_parts = []
    for v in entity_verdicts:
        if v.get("is_anomaly"):
            summary_parts.append(
                f"Entity #{v['entity_id']}: {v.get('reason', '')} "
                f"(conf={v.get('confidence', 0):.2f}, "
                f"interval=[{v.get('anomaly_start_sec', 0):.1f}s, "
                f"{v.get('anomaly_end_sec', 0):.1f}s])"
            )

    return {
        "video_path": "",
        "video_duration_sec": round(detail.get("total_frames", 0) / max(detail.get("fps", 30), 1), 2),
        "total_frames": detail.get("total_frames", 0),
        "processed_frames": 0,
        "fps": detail.get("fps", 30.0),
        "verdict": {
            "is_anomaly": detail.get("pred_anomaly", False),
            "confidence": detail.get("pred_score", 0.0),
            "anomaly_entity_ids": anomaly_eids,
            "scene_type": detail.get("category", "unknown"),
            "summary": "; ".join(summary_parts) if summary_parts else "No anomaly detected.",
            "entity_verdicts": entity_verdicts,
        },
        "timing": detail.get("timing", {}),
        "stats": detail.get("stats", {}),
        "graphs": detail.get("graphs", {}),
        "trace_log": detail.get("trace_log", []),
    }


def _convert_demo_summary(demo: dict) -> dict:
    """将 demo summary 转为标准 result 格式"""
    fps = demo.get("fps", 30.0)
    total_frames = demo.get("total_frames", 0)
    trigger_log = demo.get("trigger_log", [])

    # 从 trigger_log 构建 graphs
    entity_triggers: dict[int, list] = {}
    for t in trigger_log:
        eid = t.get("entity_id", 0)
        if eid not in entity_triggers:
            entity_triggers[eid] = []
        entity_triggers[eid].append(t)

    graphs = {}
    for eid, trigs in entity_triggers.items():
        trigs.sort(key=lambda x: x.get("timestamp", 0))
        nodes = []
        edges = []
        for i, t in enumerate(trigs):
            nodes.append({
                "node_id": f"E{eid}_N{i}",
                "timestamp": t.get("timestamp", 0),
                "action": "unknown",
                "action_object": "none",
                "posture": "unknown",
                "scene_context": "unknown",
                "is_suspicious": False,
                "danger_score": 0,
                "trigger_rule": t.get("trigger_rule", ""),
                "kinetic_energy": t.get("kinetic_energy", 0),
                "bbox": [],
                "frame_idx": t.get("frame_idx", 0),
            })
            if i > 0:
                edges.append({
                    "edge_id": f"E{eid}_N{i-1}_N{i}",
                    "source": f"E{eid}_N{i-1}",
                    "target": f"E{eid}_N{i}",
                    "duration_sec": round(t["timestamp"] - trigs[i-1]["timestamp"], 3),
                    "kinetic_integral": round(
                        sum(tr.get("kinetic_energy", 0) for tr in trigs[i-1:i+1]) / 2, 4
                    ),
                    "action_transition": "unknown → unknown",
                    "missing_frames": 0,
                })

        graphs[str(eid)] = {
            "entity_id": eid,
            "birth_time": trigs[0].get("timestamp", 0),
            "last_time": trigs[-1].get("timestamp", 0),
            "total_duration": round(trigs[-1].get("timestamp", 0) - trigs[0].get("timestamp", 0), 3),
            "total_kinetic_integral": round(sum(t.get("kinetic_energy", 0) for t in trigs), 4),
            "max_danger_score": 0,
            "has_suspicious": False,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    # 构建 trace_log (简单版)
    trace_log = []
    for t in trigger_log:
        trace_log.append({
            "frame_idx": t.get("frame_idx", 0),
            "timestamp": t.get("timestamp", 0),
            "entity_id": t.get("entity_id", 0),
            "bbox": [],
            "kinetic_energy": t.get("kinetic_energy", 0),
        })

    return {
        "video_path": demo.get("video", ""),
        "video_duration_sec": round(total_frames / max(fps, 1), 2),
        "total_frames": total_frames,
        "processed_frames": demo.get("processed_frames", 0),
        "fps": fps,
        "verdict": {
            "is_anomaly": False,
            "confidence": 0.0,
            "anomaly_entity_ids": [],
            "scene_type": "unknown",
            "summary": f"Demo data: {demo.get('total_entities', 0)} entities, {demo.get('total_triggers', 0)} triggers",
            "entity_verdicts": [],
        },
        "timing": {},
        "stats": {
            "entities": demo.get("total_entities", 0),
            "triggers": demo.get("total_triggers", 0),
        },
        "graphs": graphs,
        "trace_log": trace_log,
    }


def _load_demo_summary(video_name: str) -> Optional[dict]:
    """加载 demo_vlm_input 下的 summary.json"""
    sp = DEMO_DIR / video_name / "summary.json"
    if sp.exists():
        with open(sp, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _find_video_file(video_name: str) -> Optional[Path]:
    """在已知目录中搜索视频文件"""
    # 从 result.json 中获取路径
    try:
        result = _load_result(video_name)
        vp = result.get("video_path", "")
        if vp and Path(vp).exists():
            return Path(vp)
    except Exception:
        pass

    # 递归搜索 VIDEO_BASE
    for mp4 in VIDEO_BASE.rglob(f"{video_name}.mp4"):
        return mp4
    return None


# ── API: 视频列表 ─────────────────────────────────────
@app.get("/api/videos")
def list_videos():
    """列出所有可用的已分析视频"""
    videos = set()

    # 从独立 result.json
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and (d / "result.json").exists():
            videos.add(d.name)

    # 从 demo_vlm_input
    if DEMO_DIR.exists():
        for d in DEMO_DIR.iterdir():
            if d.is_dir() and (d / "summary.json").exists():
                videos.add(d.name)

    # 从最新 eval run
    latest_runs = sorted(EVAL_DIR.glob("run_*"), reverse=True)
    if latest_runs:
        rp = latest_runs[0] / "results_v5.json"
        if rp.exists():
            with open(rp, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            for detail in eval_data.get("details", []):
                fn = detail.get("filename", "")
                name = fn.replace(".mp4", "")
                if name:
                    videos.add(name)

    result = []
    for name in sorted(videos):
        info = {"name": name, "has_result": False, "has_demo": False}
        if (OUTPUT_DIR / name / "result.json").exists():
            info["has_result"] = True
        if (DEMO_DIR / name / "summary.json").exists():
            info["has_demo"] = True
        result.append(info)

    return {"videos": result, "total": len(result)}


# ── API: 视频元数据 ───────────────────────────────────
@app.get("/api/video/{video_name}/metadata")
def get_video_metadata(video_name: str):
    """返回视频元数据和全局判定"""
    result = _load_result(video_name)
    verdict = result.get("verdict", {})

    return {
        "video_name": video_name,
        "is_anomaly": verdict.get("is_anomaly", False),
        "confidence": verdict.get("confidence", 0.0),
        "scene_type": verdict.get("scene_type", "unknown"),
        "summary": verdict.get("summary", ""),
        "anomaly_entity_ids": verdict.get("anomaly_entity_ids", []),
        "total_frames": result.get("total_frames", 0),
        "processed_frames": result.get("processed_frames", 0),
        "fps": result.get("fps", 30.0),
        "video_duration_sec": result.get("video_duration_sec", 0),
        "timing": result.get("timing", {}),
        "stats": result.get("stats", {}),
    }


# ── API: 实体演化图 ───────────────────────────────────
@app.get("/api/video/{video_name}/entities")
def get_entities(video_name: str):
    """返回所有实体的演化图数据（节点+边）"""
    result = _load_result(video_name)
    graphs = result.get("graphs", {})
    verdict = result.get("verdict", {})
    entity_verdicts = {
        v["entity_id"]: v for v in verdict.get("entity_verdicts", [])
    }

    entities = []
    for eid_str, graph in graphs.items():
        eid = int(eid_str) if isinstance(eid_str, str) else eid_str
        ev = entity_verdicts.get(eid, {})
        entities.append({
            "entity_id": eid,
            "is_anomaly": ev.get("is_anomaly", False),
            "confidence": ev.get("confidence", 0.0),
            "reason": ev.get("reason", ""),
            "anomaly_start_sec": ev.get("anomaly_start_sec", 0.0),
            "anomaly_end_sec": ev.get("anomaly_end_sec", 0.0),
            "graph": graph,
        })

    return {"video_name": video_name, "entities": entities}


# ── API: 实体轨迹 (动能序列) ──────────────────────────
@app.get("/api/entities/{video_name}/{entity_id}/trajectory")
def get_entity_trajectory(video_name: str, entity_id: int):
    """返回指定实体的动能序列（用于折线图）和异常区间"""
    result = _load_result(video_name)
    graphs = result.get("graphs", {})
    trace_log = result.get("trace_log", [])
    verdict = result.get("verdict", {})

    # 从 trace_log 提取该实体的动能序列
    trajectory = []
    for entry in trace_log:
        if entry.get("entity_id") == entity_id:
            trajectory.append({
                "frame_idx": entry["frame_idx"],
                "timestamp": entry.get("timestamp", 0),
                "kinetic_energy": entry.get("kinetic_energy", 0),
                "bbox": entry.get("bbox", []),
            })

    # 从 graph 补充节点级别数据（包含语义信息）
    graph = graphs.get(str(entity_id), {})
    nodes = graph.get("nodes", [])
    node_data = []
    for n in nodes:
        node_data.append({
            "node_id": n.get("node_id", ""),
            "frame_idx": 0,
            "timestamp": n.get("timestamp", 0),
            "action": n.get("action", "unknown"),
            "is_suspicious": n.get("is_suspicious", False),
            "danger_score": n.get("danger_score", 0),
            "kinetic_energy": n.get("kinetic_energy", 0),
            "trigger_rule": n.get("trigger_rule", ""),
        })

    # 异常区间
    ev = {}
    for v in verdict.get("entity_verdicts", []):
        if v["entity_id"] == entity_id:
            ev = v
            break

    return {
        "entity_id": entity_id,
        "trajectory": trajectory,
        "nodes": node_data,
        "is_anomaly": ev.get("is_anomaly", False),
        "anomaly_start_sec": ev.get("anomaly_start_sec", 0),
        "anomaly_end_sec": ev.get("anomaly_end_sec", 0),
        "reason": ev.get("reason", ""),
    }


# ── API: 审计叙事 ─────────────────────────────────────
@app.get("/api/audit/{video_name}/narrative")
def get_audit_narrative(video_name: str):
    """返回审计叙事和结论"""
    result = _load_result(video_name)
    verdict = result.get("verdict", {})

    # 构造结构化叙事
    narratives = []
    graphs = result.get("graphs", {})
    entity_verdicts = {
        v["entity_id"]: v for v in verdict.get("entity_verdicts", [])
    }

    for eid_str, graph in graphs.items():
        eid = int(eid_str) if isinstance(eid_str, str) else eid_str
        ev = entity_verdicts.get(eid, {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # 构建动作序列叙事
        action_seq = []
        for n in nodes:
            action_seq.append({
                "timestamp": n.get("timestamp", 0),
                "action": n.get("action", "unknown"),
                "action_object": n.get("action_object", "none"),
                "posture": n.get("posture", "unknown"),
                "is_suspicious": n.get("is_suspicious", False),
                "danger_score": n.get("danger_score", 0),
                "trigger_rule": n.get("trigger_rule", ""),
                "kinetic_energy": n.get("kinetic_energy", 0),
            })

        transitions = []
        for e in edges:
            transitions.append({
                "action_transition": e.get("action_transition", ""),
                "duration_sec": e.get("duration_sec", 0),
                "kinetic_integral": e.get("kinetic_integral", 0),
            })

        narratives.append({
            "entity_id": eid,
            "is_anomaly": ev.get("is_anomaly", False),
            "confidence": ev.get("confidence", 0.0),
            "reason": ev.get("reason", ""),
            "anomaly_start_sec": ev.get("anomaly_start_sec", 0),
            "anomaly_end_sec": ev.get("anomaly_end_sec", 0),
            "action_sequence": action_seq,
            "transitions": transitions,
            "birth_time": graph.get("birth_time", 0),
            "last_time": graph.get("last_time", 0),
            "total_duration": graph.get("total_duration", 0),
            "max_danger_score": graph.get("max_danger_score", 0),
            "has_suspicious": graph.get("has_suspicious", False),
        })

    return {
        "video_name": video_name,
        "is_anomaly": verdict.get("is_anomaly", False),
        "confidence": verdict.get("confidence", 0.0),
        "summary": verdict.get("summary", ""),
        "scene_type": verdict.get("scene_type", "unknown"),
        "narratives": narratives,
    }


# ── API: 触发事件时间线 ───────────────────────────────
@app.get("/api/video/{video_name}/triggers")
def get_triggers(video_name: str):
    """返回所有 VLM 触发事件（用于演化图节点）"""
    # 优先使用 demo summary
    demo = _load_demo_summary(video_name)
    if demo:
        return {
            "video_name": video_name,
            "total_triggers": demo.get("total_triggers", 0),
            "trigger_stats": demo.get("trigger_stats", {}),
            "triggers": demo.get("trigger_log", []),
        }

    # 从 result.json graphs 中提取
    result = _load_result(video_name)
    graphs = result.get("graphs", {})
    triggers = []
    for eid_str, graph in graphs.items():
        for node in graph.get("nodes", []):
            triggers.append({
                "trigger_id": len(triggers) + 1,
                "frame_idx": 0,
                "timestamp": node.get("timestamp", 0),
                "entity_id": int(eid_str) if isinstance(eid_str, str) else eid_str,
                "trigger_rule": node.get("trigger_rule", ""),
                "kinetic_energy": node.get("kinetic_energy", 0),
                "action": node.get("action", "unknown"),
                "danger_score": node.get("danger_score", 0),
                "is_suspicious": node.get("is_suspicious", False),
            })

    triggers.sort(key=lambda x: x["timestamp"])
    return {
        "video_name": video_name,
        "total_triggers": len(triggers),
        "triggers": triggers,
    }


# ── API: 关键帧图像服务 ───────────────────────────────
@app.get("/api/frames/{video_name}/{frame_idx}")
def get_frame_image(
    video_name: str,
    frame_idx: int,
    type: str = Query("raw", description="grid|crop|painted|raw"),
):
    """返回关键帧图像"""
    # 1. 尝试从 demo_vlm_input 获取预生成图
    if DEMO_DIR.exists():
        demo_dir = DEMO_DIR / video_name
        if demo_dir.exists():
            # 查找匹配的 trigger 目录
            for trigger_dir in demo_dir.iterdir():
                if not trigger_dir.is_dir():
                    continue
                # trigger_012_birth_E2_F548
                parts = trigger_dir.name.split("_")
                try:
                    fidx = int(parts[-1].replace("F", ""))
                    if fidx == frame_idx:
                        fname_map = {
                            "grid": "vlm_input_grid.jpg",
                            "crop": "crop_entity.jpg",
                            "painted": "painted_fullframe.jpg",
                            "raw": "raw_frame.jpg",
                        }
                        fname = fname_map.get(type, "raw_frame.jpg")
                        img_path = trigger_dir / fname
                        if img_path.exists():
                            return StreamingResponse(
                                open(img_path, "rb"),
                                media_type="image/jpeg",
                            )
                except (ValueError, IndexError):
                    pass

    # 2. 从视频文件中直接抽帧
    video_path = _find_video_file(video_name)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video file not found")

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise HTTPException(status_code=404, detail=f"Cannot read frame {frame_idx}")

    # 如果是 painted 类型，尝试在帧上画 bbox
    if type == "painted":
        result = _load_result(video_name)
        trace_log = result.get("trace_log", [])
        for entry in trace_log:
            if entry["frame_idx"] == frame_idx:
                bbox = entry.get("bbox", [])
                eid = entry.get("entity_id", 0)
                if len(bbox) == 4:
                    x, y, w, h = [int(b) for b in bbox]
                    color = (0, 0, 255) if eid % 2 == 0 else (0, 255, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"E{eid}", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


# ── API: 视频流帧服务 ─────────────────────────────────
@app.get("/api/video/{video_name}/frame")
def get_video_frame(
    video_name: str,
    t: float = Query(0, description="时间戳(秒)"),
):
    """根据时间戳返回视频帧"""
    video_path = _find_video_file(video_name)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video file not found")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise HTTPException(status_code=404, detail=f"Cannot read frame at t={t}")

    # 缩小以加速传输
    h, w = frame.shape[:2]
    if w > 960:
        scale = 960 / w
        frame = cv2.resize(frame, (960, int(h * scale)))

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


# ── API: 视频文件直传 (用于前端 video 标签) ────────────
@app.get("/api/video/{video_name}/file")
def get_video_file(video_name: str):
    """返回视频文件流"""
    video_path = _find_video_file(video_name)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video file not found")

    def iterfile():
        with open(video_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


# ── API: 进度条热图数据 ───────────────────────────────
@app.get("/api/video/{video_name}/heatmap")
def get_heatmap(video_name: str, bins: int = Query(200, description="热图分辨率")):
    """生成视频进度条热图数据（异常区间标记）"""
    result = _load_result(video_name)
    verdict = result.get("verdict", {})
    duration = result.get("video_duration_sec", 0)
    fps = result.get("fps", 30.0)

    if duration <= 0:
        duration = result.get("total_frames", 1) / max(fps, 1)

    bin_width = duration / bins
    heatmap = []

    for i in range(bins):
        t_start = i * bin_width
        t_end = (i + 1) * bin_width
        level = "normal"  # normal / warning / danger

        for ev in verdict.get("entity_verdicts", []):
            if not ev.get("is_anomaly"):
                continue
            a_start = ev.get("anomaly_start_sec", 0)
            a_end = ev.get("anomaly_end_sec", 0)
            # 检查区间重叠
            if a_start < t_end and a_end > t_start:
                conf = ev.get("confidence", 0)
                if conf >= 0.7:
                    level = "danger"
                else:
                    level = "warning"
                break

        heatmap.append({
            "bin": i,
            "t_start": round(t_start, 2),
            "t_end": round(t_end, 2),
            "level": level,
        })

    return {"video_name": video_name, "duration": duration, "bins": bins, "heatmap": heatmap}


# ── API: BBox 叠加数据 ────────────────────────────────
@app.get("/api/video/{video_name}/bboxes")
def get_bboxes(video_name: str, t: float = Query(0)):
    """返回指定时间戳附近的所有实体 BBox"""
    result = _load_result(video_name)
    trace_log = result.get("trace_log", [])
    fps = result.get("fps", 30.0)

    target_frame = int(t * fps)
    tolerance = max(int(fps * 0.2), 2)  # ±0.2s

    bboxes = []
    seen_eids = set()
    for entry in trace_log:
        if abs(entry["frame_idx"] - target_frame) <= tolerance:
            eid = entry["entity_id"]
            if eid not in seen_eids:
                seen_eids.add(eid)
                # 获取语义标签
                graphs = result.get("graphs", {})
                graph = graphs.get(str(eid), {})
                action = "unknown"
                for node in graph.get("nodes", []):
                    if abs(node.get("timestamp", 0) - t) < 2.0:
                        action = node.get("action", "unknown")
                        break

                bboxes.append({
                    "entity_id": eid,
                    "bbox": entry.get("bbox", []),
                    "kinetic_energy": entry.get("kinetic_energy", 0),
                    "action": action,
                })

    return {"time": t, "frame_idx": target_frame, "bboxes": bboxes}


# ── 静态前端文件服务 ──────────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    @app.get("/")
    def serve_frontend():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")
