# EventVAD V5 功能文档

> 版本：V5（Tube-Skeleton Pipeline）  
> 日期：2026-02-19  
> 适用范围：`src/v5/` 视频异常检测系统（UCF-Crime / XD-Violence 评估）

---

## 1. 系统概述

### 1.1 项目简介

EventVAD V5 是一个**免训练（Training-Free）**的视频异常检测系统，采用 **Tube-Skeleton（管状骨架）** 三阶段架构。系统能够在无需端到端重训练的前提下，仅凭视觉语言模型（VLM）完成监控视频中的异常识别，同时兼顾**视频级判定**与**帧级时间定位**，输出可解释的检测结果。

### 1.2 核心设计理念

- **物理先验 + 语义理解双通道**：帧差动能提供物理信号，VLM 提供语义理解，两者互补
- **稀疏调用策略**：仅在关键时刻触发 VLM 推理，大幅降低计算开销
- **矛盾检测兜底**：当语义盲区出现时，物理信号自动补偿
- **零先验判断**：不注入场景先验知识，让模型自主判断

### 1.3 三阶段架构总览

| 阶段 | 名称 | 核心功能 | 模块 |
|------|------|---------|------|
| **Stage 1** | 物理追踪骨架 | 帧差运动检测 + CLIP Crop 特征 + 实体追踪 | `tracking/` |
| **Stage 2** | 稀疏语义挂载 | 关键帧触发 + VLM 语义推理 + 矛盾检测 + 全局心跳 | `semantic/` |
| **Stage 3** | 动态图审计 | 图构建 + 叙事生成 + Decision LLM 审计 | `graph/` |

---

## 2. 模块详解

### 2.1 Stage 1：物理追踪骨架 (`src/v5/tracking/`)

#### 2.1.1 MotionExtractor — 帧差动能提取器

**文件**：`tracking/motion_extractor.py`

**功能**：从视频帧中通过帧间差分提取画面中"主要运动区域"（MotionRegion），不依赖任何预训练检测模型。

**核心流程**：
1. `cv2.absdiff` → 帧差动能图（支持多帧累积）
2. 自适应阈值 + 形态学去噪 → 二值掩膜
3. 连通域分析 → Top-K bbox
4. Crop 裁剪 + padding

**增强特性**：
- **多帧累积差分**：保留最近 N 帧灰度图，使用 max-pooled 帧差捕捉慢速/渐变运动
- **自适应阈值回退**：连续无检出时自动降低阈值（从 `diff_threshold=25` 逐步降至 `adaptive_threshold_min=8`），检出后缓慢恢复
- **灵敏默认参数**：适配远景监控场景（最小面积 1500px，最小 Crop 80px）

**输出数据结构 `MotionRegion`**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `x, y, w, h` | int | Bounding box 坐标 |
| `crop_image` | np.ndarray | BGR 裁剪图像 |
| `kinetic_energy` | float | 该区域平均动能 (0-1) |
| `area` | int | 像素面积 |

**关键配置 (`MotionConfig`)**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `diff_threshold` | 25 | 帧差二值化阈值 (0-255) |
| `adaptive_threshold_min` | 8 | 自适应回退最低阈值 |
| `morph_kernel_size` | 5 | 形态学核大小 |
| `min_region_area` | 1500 | 连通域最小面积 (px) |
| `top_k_regions` | 4 | 每帧最大提取区域数 |
| `crop_padding_ratio` | 0.15 | Crop 外扩比例 |
| `accumulate_frames` | 3 | 多帧累积缓冲区大小 |
| `empty_streak_for_fallback` | 30 | 连续无检出帧数触发降阈值 |

---

#### 2.1.2 YoloDetector — YOLO-World 开放词汇目标检测

**文件**：`tracking/yolo_detector.py`

**功能**：基于 YOLO-World 的语义级目标检测，可识别预定义的开放词汇类别，补充帧差法在静态/渐变目标上的盲区。

**检测类别**（11类）：
```
person, fire, smoke, car, truck, motorcycle, knife, gun, explosion, blood, bag
```

**性能优化**：
- FP16 半精度推理（CUDA 上提速 40-60%）
- 批量 GPU→CPU tensor 传输
- `stream=True` 减少内存分配
- 模型预热消除首帧延迟
- 智能频率控制：每 5 帧检测一次，force 模式也需满足最小间隔（2 帧）

**输出数据结构 `YoloRegion`**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `x, y, w, h` | int | Bounding box 坐标 |
| `crop_image` | np.ndarray | BGR 裁剪图像 |
| `confidence` | float | 检测置信度 (0-1) |
| `class_name` | str | 类别名称 |
| `class_id` | int | 类别 ID |
| `area` | int | 像素面积 |

**关键配置 (`YoloDetectorConfig`)**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | yolov8l-worldv2.pt | YOLO-World 模型 |
| `confidence_threshold` | 0.15 | 低阈值宁可多检不漏 |
| `iou_threshold` | 0.45 | NMS IoU |
| `imgsz` | 480 | 推理输入尺寸 |
| `half` | True | FP16 半精度 |
| `detect_every_n` | 5 | 普通模式调用间隔 |
| `force_min_gap` | 2 | force 模式最小帧间隔 |

---

#### 2.1.3 HybridDetector — 帧差 + YOLO 融合检测器

**文件**：`tracking/hybrid_detector.py`

**功能**：融合帧差法和 YOLO-World 的互补优势——帧差捕捉快速运动，YOLO 检测静态/语义目标（如火焰、烟雾）。

**融合策略**：
1. 两个检测器各自独立运行
2. 通过 IoU 匹配重叠区域 → 合并（取 YOLO 的框 + 帧差的动能）
3. YOLO 独有检测（帧差未检出）→ 保留（如静态火焰）
4. 帧差独有检测（YOLO 未检出）→ 保留（如快速运动的未知物体）

**输出**: `HybridRegion` 列表（兼容 `MotionRegion` 接口），附加 `class_name`、`confidence`、`source` 字段。

**排序优先级**：`fused > yolo > motion`，内部按 `confidence + kinetic_energy` 排序。

> ⚠️ YOLO 模式为可选功能，通过 `--yolo` 命令行参数启用。默认仅使用帧差法。

---

#### 2.1.4 CropCLIPEncoder — CLIP 特征提取器

**文件**：`tracking/clip_encoder.py`

**功能**：对 Crop 区域（而非全图）提取 CLIP 视觉特征，用于实体匹配和变化点检测。

**技术特点**：
- 模型：`openai/clip-vit-base-patch32`（512 维）
- 延迟加载 + 线程安全全局单例
- 支持 batch 编码
- L2 归一化输出

**用途**：
- `EntityTracker`：跨帧实体匹配的余弦相似度计算
- `NodeTrigger`：Change-point 检测的嵌入距离度量
- `GlobalHeartbeat`：全图 CLIP 漂移检测

---

#### 2.1.5 EntityTracker — 贪婪匹配实体追踪器

**文件**：`tracking/entity_tracker.py`

**功能**：基于 CLIP 嵌入的贪婪匹配实体追踪，为每个运动区域分配稳定的 `entity_id`。

**核心逻辑**：
1. 计算当前帧 crop embedding 与上一帧活跃 Entity 的余弦相似度矩阵
2. 贪婪匹配：相似度 ≥ threshold → 沿用旧 ID
3. 相似度 < threshold 且动能大 → 分配新 ID
4. 维护活跃实体池，超龄清除

**输出数据结构 `TraceEntry`**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_idx` | int | 帧序号 |
| `timestamp` | float | 秒 |
| `entity_id` | int | 实体 ID |
| `bbox` | tuple | (x, y, w, h) |
| `embedding` | np.ndarray | CLIP 特征 (512,) |
| `kinetic_energy` | float | 瞬时动能 |
| `crop_image` | np.ndarray | BGR 裁剪（可选保留） |

**关键配置 (`TrackerConfig`)**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `similarity_threshold` | 0.72 | 余弦相似度匹配阈值 |
| `max_age_frames` | 150 | 最大帧间隔（超龄清除） |
| `max_active_entities` | 30 | 最大活跃实体数 |
| `min_kinetic_for_new` | 0.06 | 新实体最低动能门限 |

---

#### 2.1.6 VisualPainter — 可视化标注器

**文件**：`tracking/visual_painter.py`

**功能**：在原图上绘制高亮框 + 标注信息，生成"带框全图"作为 VLM 的输入。

**标注内容**：
- 红色/黄色/橙色框（动能越高颜色越暖）
- 实体 ID + YOLO 类别（如有）
- 动能等级标签（still/low/medium/high/EXTREME）
- 输出 resize 至 768×768 的 PIL Image

---

#### 2.1.7 MultiFrameStacker — 多帧拼图生成器

**文件**：`tracking/multi_frame_stacker.py`

**功能**：生成 2×2 的四宫格拼图，让 VLM 看到时序上的动作变化（"连环画"效果），解决单帧无法识别的异常（如推搡、连续虐待）。

**排列方式**：
```
[T-6] [T-4]
[T-2] [T-0]   (最旧→最新，按时序排列)
```

**配置**：
- 默认每隔 6 帧采一帧进缓冲区
- 输出总尺寸 768×768
- 最大缓冲 20 帧

---

### 2.2 Stage 2：稀疏语义挂载 (`src/v5/semantic/`)

#### 2.2.1 NodeTrigger — 语义节点触发策略

**文件**：`semantic/node_trigger.py`

**功能**：控制 VLM 的调用频率，只在"关键时刻"触发语义推理，生成语义节点。

**三条触发规则**：

| 规则 | 名称 | 条件 | 说明 |
|------|------|------|------|
| Rule 1 | **Birth** | 新 entity_id 首次出现 | 新实体入场即触发 |
| Rule 2 | **Change Point** | `1 - cos_sim(当前, 上次采样)` > 0.22 | CLIP embedding 突变 |
| Rule 3 | **Heartbeat** | 距上次采样 > 3.0s | 周期性强制采样 |

**保护机制**：
- 最小触发间隔保护：同一实体相邻触发最少间隔 5 帧
- 心跳扫描：对未产生 MotionRegion 的活跃实体也进行心跳检查

**输出数据结构 `TriggerResult`**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `entity_id` | int | 实体 ID |
| `frame_idx` | int | 帧号 |
| `timestamp` | float | 秒 |
| `trigger_rule` | str | "birth" / "change_point" / "heartbeat" |
| `embedding_distance` | float | 与上次采样的 1-cosine 距离 |
| `trace_entry` | TraceEntry | 对应的追踪记录 |

---

#### 2.2.2 SemanticVLLM — VLM 语义推理客户端

**文件**：`semantic/vllm_semantic.py`

**功能**：对触发的关键帧调用视觉语言模型（Qwen2.5-VL-7B），进行 Crop 级别语义理解和异常判定。

**双模式输入**：
| 输入类型 | 优先级 | 说明 |
|---------|--------|------|
| **4宫格拼图** (grid) | 最高 | 多帧时序上下文 |
| **带框全图** (painted) | 次之 | 原图 + 高亮框，保留环境上下文 |
| **Crop 裁剪** | 兜底 | 仅运动区域（resize 到 336×336） |

**Prompt 设计**：
- **单帧 Prompt**：描述带框全图中高亮实体的行为、姿态、场景
- **多帧 Prompt**：描述 4 帧间的时序变化（如"人突然倒地"、"火焰蔓延"）
- 监控异常类别提示：Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
- 可疑视觉线索提示：身体接触、快速运动、火焰烟雾、武器、车辆碰撞等

**输出结构化 JSON**：
```json
{
  "box_action": "<高亮实体的动作描述>",
  "context_relation": "<周围环境>",
  "action": "<原子动作动词>",
  "action_object": "<交互对象>",
  "posture": "<身体姿态>",
  "scene_context": "<场景类型>",
  "is_suspicious": true/false,
  "danger_score": 0.0-1.0
}
```

**推理后端**：
| 后端 | 说明 | 推荐场景 |
|------|------|---------|
| `server` | 通过 vLLM OpenAI 兼容 API 并行调用 | 生产/评估（推荐） |
| `local` | 本地加载 Qwen2.5-VL-7B-Instruct | 调试/离线 |

**并行特性**：server 模式支持 `ThreadPoolExecutor`，最大 `max_workers=16` 并行请求。

---

#### 2.2.3 DiscordanceChecker — 物理-语义矛盾检测器

**文件**：`semantic/discordance_checker.py`

**功能**：检测**物理信号**与**语义描述**之间的矛盾——当运动动能显著超出背景水平，但 VLM 语义描述却报告"正常"时，发出矛盾警报。

**核心逻辑**：

1. **自适应阈值校准**：用视频前 30 帧的动能计算背景基准 `μ + 5σ`
2. **矛盾检测条件**（三重过滤）：
   - 条件 1：动能超过自适应阈值
   - 条件 2：超标倍率 ≥ 2.5x
   - 条件 3：`danger_score` < 0.10
3. **多实体投票抑制**：当超标实体占活跃实体 ≥ 50% 时，视为正常场景活跃，抑制所有 discordance alert

**矛盾类型**：
| 类型 | 含义 |
|------|------|
| `energy_semantic_gap` | 高物理动能 + 低语义危险度 |
| `semantic_drift` | 全局 CLIP embedding 突变 |

**输出数据结构 `DiscordanceAlert`**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `entity_id` | int | 实体 ID |
| `alert_type` | str | 矛盾类型 |
| `motion_energy` | float | 物理动能 |
| `danger_score` | float | 语义危险度 |
| `peak_energy_time` | float | 动能峰值时刻 (秒) |
| `burst_start_sec` / `burst_end_sec` | float | 动能突发区间 |

> ⚠️ DiscordanceChecker 可通过 `--no-discordance` 参数禁用（消融实验模式）。

---

#### 2.2.4 GlobalHeartbeat — 全局心跳 + CLIP 漂移检测

**文件**：`semantic/global_heartbeat.py`

**功能**：独立于实体追踪运行的低频全图语义心跳，针对低动态异常场景（如 Abuse 类）。

**运行机制**：
1. 每 2.5 秒进行一次强制全图 CLIP 编码（不依赖帧差连通域）
2. 计算全图 CLIP Embedding 的累计偏移量（Semantic Drift）
3. 偏移 > 0.18 即触发额外 VLM 全图扫描

**触发类型**：
| 类型 | 条件 |
|------|------|
| `periodic` | 定时心跳（每 2.5s） |
| `drift` | CLIP embedding 偏移 > 0.18 |

---

### 2.3 Stage 3：动态图审计 (`src/v5/graph/`)

#### 2.3.1 图数据结构

**文件**：`graph/structures.py`

**三层数据结构**：

**TemporalNode（时间语义节点）**：
- 对应某个实体在某时刻被 VLM 描述的状态快照
- 包含：timestamp, action, action_object, posture, scene_context, is_suspicious, danger_score, trigger_rule, kinetic_energy

**EvolutionEdge（时间演化边）**：
- 连接同一实体的两个相邻 TemporalNode
- 包含：duration_sec, kinetic_integral（动能累积）, action_transition, missing_frames

**EntityGraph（实体时间演化图）**：
- 单实体的完整时间演化链表
- 自动维护统计信息：total_duration, total_kinetic_integral, max_danger_score, has_suspicious

---

#### 2.3.2 GraphBuilder — 动态图构建器

**文件**：`graph/graph_builder.py`

**功能**：
1. 每次 Trigger 触发时创建 TemporalNode
2. 在两个相邻 Node 之间创建 EvolutionEdge
3. 自动计算 Edge 的时长和物理动能累积
4. 维护多个实体的 EntityGraph（最大 100 节点/实体）

---

#### 2.3.3 NarrativeGenerator — 叙事生成器

**文件**：`graph/narrative_generator.py`

**功能**：遍历 EntityGraph 路径，生成供 Decision LLM 审计的结构化文本叙事。

**叙事内容包含**：
1. **Header**：实体 ID、持续时间、观测数、场景类型
2. **[Physical Signal]**：物理-语义矛盾预警（来自 DiscordanceChecker）
3. **[Physical Signal — Scene Change]**：CLIP 漂移预警
4. **[Physical Trajectory]**：物理轨迹摘要（帧数、动能统计、位移、趋势）
5. **逐节点+逐边描述**：时间线上的动作序列 + 持续时间 + 动能累积
6. **Summary**：总动能、最大危险度、是否可疑、是否有物理预警

**示例叙事**：
```
Entity #3 | Duration: 25.0s | Observations: 8 | Scene: outdoor

[Physical Signal]
  - Motion energy (0.3500) exceeds adaptive background threshold (0.0512) by 6.8x, ...
    Peak kinetic energy at T=12.50s (value=0.3500), burst interval=[10.50s, 16.50s]

[Physical Trajectory]
  Tracked 120 frames over 24.0s
  Kinetic energy: mean=0.0850, max=0.3500, integral=10.2000
  Trend: kinetic energy RISING

T=5.0s: walking, posture=upright, kinetic=0.0200  (trigger: birth)
  ↓ walking → running | duration=5.0s | kinetic_integral=0.6500
T=10.0s: running (toward person), posture=leaning forward, kinetic=0.1500 [flagged]  (trigger: change_point)
...

Summary: total_kinetic=10.2000, max_danger=0.75, suspicious=True, physics_warning=True
```

---

#### 2.3.4 DecisionAuditor — 决策审计器

**文件**：`graph/decision_prompt.py`

**功能**：对 EntityGraph 中的实体进行 Decision LLM 最终异常判定。

**审计流程**：
1. **实体筛选**：优先审计有可疑信号/矛盾警报/场景漂移的实体
2. **叙事生成**：调用 NarrativeGenerator 生成含物理预警的叙事
3. **LLM 调用**：将叙事文本发送给 Decision LLM
4. **响应解析**：提取 is_anomaly, confidence, anomaly_start_sec, anomaly_end_sec, reason

**异常区间定位优先级**：
1. **Discordance 动能峰值锚定**：用动能峰值时间和突发区间定位（最准确）
2. **LLM 直接输出的区间**：Decision LLM 给出的 anomaly_start/end_sec
3. **Fallback 可疑节点时间范围**：从 EntityGraph 中可疑节点的时间戳推算

**Decision Prompt 设计原则**：
- **异常标准**：暴力、盗窃、纵火、事故、武器、破坏、入侵
- **正常标准**：日常活动、正常转换、姿态变化、例行工作
- **核心原则**：仅当动作明确匹配异常类别时才标记异常；动能波动本身不构成异常

**输出数据结构**：

`AuditVerdict`（实体级）：
| 字段 | 类型 | 说明 |
|------|------|------|
| `entity_id` | int | 实体 ID |
| `is_anomaly` | bool | 是否异常 |
| `confidence` | float | 置信度 (0-1) |
| `reason` | str | 判定理由 |
| `anomaly_start_sec` | float | 异常起始时间 |
| `anomaly_end_sec` | float | 异常结束时间 |
| `is_cinematic_false_alarm` | bool | 是否影视误报 |

`VideoVerdict`（视频级）：
| 字段 | 类型 | 说明 |
|------|------|------|
| `is_anomaly` | bool | 视频是否包含异常 |
| `confidence` | float | 最大异常置信度 |
| `entity_verdicts` | list | 所有实体的审计结论 |
| `anomaly_entity_ids` | list | 异常实体 ID 列表 |
| `scene_type` | str | 场景类型 |
| `summary` | str | 文本摘要 |

**规则兜底**：当 LLM 调用失败时，使用关键词匹配规则（suspicious, fighting, fire 等）作为 fallback。

---

## 3. 主流程（Pipeline）

**文件**：`src/v5/pipeline.py`  
**入口类**：`TubeSkeletonPipeline`

### 3.1 输入/输出

| 项目 | 说明 |
|------|------|
| **输入** | 视频文件路径（mp4/avi/mov 等） |
| **输出** | JSON 结构化结果（verdict + graphs + trace_log + timing） |

### 3.2 完整处理流程

```
视频输入
    ↓
[Stage 1+2: 逐帧处理循环]
    ├─ 帧差/混合检测 → MotionRegion / HybridRegion
    ├─ CLIP 特征提取 → embeddings (N, 512)
    ├─ 实体追踪 → TraceEntry (entity_id 分配)
    ├─ NodeTrigger 检查 → TriggerResult (关键帧判定)
    ├─ 全图标注 → painted_images (带框全图)
    ├─ 多帧拼图 → grid_images (4宫格)
    ├─ 实体心跳扫描 → 补充 Trigger
    ├─ 全局心跳 → CLIP 漂移检测
    └─ 背景动能校准 (前30帧)
    ↓
[Stage 2: 批量 VLM 语义推理]
    ├─ SemanticVLLM.infer_triggers → semantic_results
    └─ DiscordanceChecker.check_video → discordance_alerts
    ↓
[Stage 3: 动态图审计]
    ├─ GraphBuilder.add_semantic_node → EntityGraph 构建
    ├─ NarrativeGenerator.generate → 叙事文本
    └─ DecisionAuditor.audit_video → VideoVerdict
    ↓
JSON 结果输出
```

### 3.3 输出 JSON 结构

```json
{
  "video_path": "...",
  "video_duration_sec": 120.0,
  "total_frames": 3600,
  "processed_frames": 1800,
  "fps": 30.0,
  "verdict": {
    "is_anomaly": true,
    "confidence": 0.85,
    "anomaly_entity_ids": [3, 7],
    "scene_type": "outdoor",
    "summary": "Entity #3: fighting detected (conf=0.85, interval=[10.0s, 25.0s])",
    "entity_verdicts": [
      {
        "entity_id": 3,
        "is_anomaly": true,
        "confidence": 0.85,
        "break_timestamp": 12.5,
        "reason": "Two persons engaged in physical fight",
        "anomaly_start_sec": 10.0,
        "anomaly_end_sec": 25.0
      }
    ]
  },
  "timing": {
    "tracking_sec": 15.2,
    "semantic_sec": 8.3,
    "decision_sec": 2.1,
    "total_sec": 25.6
  },
  "stats": {
    "entities": 12,
    "triggers": 45,
    "nodes": 45,
    "edges": 33
  },
  "graphs": { ... },
  "trace_log": [ ... ]
}
```

---

## 4. 评估系统

### 4.1 UCF-Crime 评估

**文件**：`src/v5/eval_ucf_crime.py`

**评估指标**：
| 指标 | 说明 |
|------|------|
| **Frame-level AUC-ROC** | 核心指标，帧级异常分数的 ROC 曲线下面积 |
| **Video-level AUC-ROC** | 视频级分数的 AUC |
| **Accuracy / Precision / Recall / F1** | 视频级分类指标 |
| **Mean Soft IoU** | 无阈值的软 IoU (Σmin(gt, pred) / Σmax(gt, pred)) |
| **Mean Hysteresis IoU** | 滞回阈值二值化后的硬 IoU |

**帧级分数生成策略**：
1. 将 `anomaly_start_sec / anomaly_end_sec` 区间广播 confidence 值
2. 前后加 5 秒 ramp 渐变
3. 高斯平滑 (σ = 2 秒)
4. 如果视频判异常但分数全为 0，给全视频低底分

**运行特性**：
- 支持并行视频处理（`--parallel N`）
- 分层采样：每个异常类别至少 1 个视频
- 每次运行自动创建带时间戳的日志目录
- 自动创建 `latest` 软链接

**使用示例**：
```bash
cd src
python -m v5.eval_ucf_crime \
    --max-videos 40 \
    --sample-every 2 \
    --api-base http://localhost:8000 \
    --parallel 6 \
    --max-workers 48
```

### 4.2 Bad Case 分析

**文件**：`src/v5/analyze_bad_cases.py`

**功能**：自动化 bad case 根因分析与改进建议。

**分析维度**：
1. **与上一次运行对比**：指标变化、混淆矩阵变化、视频级 flip（TP↔FN 等）
2. **FN/FP 自动根因分类**：

| 根因代码 | 含义 | 典型场景 |
|---------|------|---------|
| `ZERO_ENTITY_DETECTION` | 帧差完全未检出 | 低对比度/慢速运动 |
| `ZERO_TRIGGERS` | 有实体但无触发器 | NodeTrigger 条件未满足 |
| `VLLM_UNDER_DESCRIPTION` | VLM 语义描述不足 | 火焰/烟雾未被识别 |
| `DISCORDANCE_FALSE_ALARM` | 矛盾检测误触发 | 繁忙路口的正常运动 |
| `VLLM_HALLUCINATION` | VLM 语义幻觉 | 正常行为被误解为异常 |
| `DECISION_FALSE_ALARM` | 决策层误报 | 综合判定错误 |

3. **TP IoU 质量分析**：excellent (≥0.5) / moderate (0.05-0.5) / near-zero (<0.05)
4. **类别级分析**：按异常类别统计 accuracy、IoU
5. **优先级行动建议**：自动生成 P0/P1 改进项

**使用示例**：
```bash
cd src
python -m v5.analyze_bad_cases \
    --current output/v5/eval_ucf_crime/run_xxx/results_v5.json \
    --previous output/v5/eval_ucf_crime/run_yyy/results_v5.json
```

---

## 5. 配置系统

**文件**：`src/v5/config.py`

### 5.1 全局配置

| 配置类 | 阶段 | 说明 |
|--------|------|------|
| `MotionConfig` | Stage 1 | 帧差动能检测参数 |
| `YoloDetectorConfig` | Stage 1 | YOLO-World 检测参数 |
| `HybridDetectorConfig` | Stage 1 | 帧差+YOLO 融合策略 |
| `CLIPEncoderConfig` | Stage 1 | CLIP 特征提取参数 |
| `TrackerConfig` | Stage 1 | 实体追踪匹配参数 |
| `NodeTriggerConfig` | Stage 2 | 语义节点触发策略 |
| `SemanticVLLMConfig` | Stage 2 | VLM 语义推理参数 |
| `GraphConfig` | Stage 3 | 时间演化图参数 |
| `NarrativeConfig` | Stage 3 | 叙事生成参数 |
| `DecisionConfig` | Stage 3 | 决策审计参数 |
| `SystemConfig` | 全局 | 设备/精度/日志 |

### 5.2 路径配置

| 变量 | 值 | 说明 |
|------|------|------|
| `PROJECT_ROOT` | `EventVAD/` | 项目根目录 |
| `SRC_DIR` | `EventVAD/src/` | 源码目录 |
| `MODELS_DIR` | `EventVAD/models/` | 模型目录 |
| `OUTPUT_DIR` | `EventVAD/output/v5/` | V5 输出目录 |
| `HF_CACHE_DIR` | `EventVAD/models/huggingface/` | HuggingFace 缓存 |

### 5.3 系统信息

| 项目 | 值 |
|------|------|
| 版本号 | 5.0 |
| Python | 3.10 |
| VLM 模型 | Qwen2.5-VL-7B-Instruct |
| CLIP 模型 | openai/clip-vit-base-patch32 |
| YOLO 模型 | yolov8l-worldv2 (可选) |
| 设备 | CUDA (自动检测) |
| 精度 | FP16 |

---

## 6. 使用方法

### 6.1 单视频分析

```bash
cd src
python -m v5.pipeline \
    --video /path/to/video.mp4 \
    --api-base http://localhost:8000 \
    --backend server \
    --sample-every 2 \
    --max-workers 48
```

**可选参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | (必填) | 视频文件路径 |
| `--api-base` | http://localhost:8000 | vLLM API 地址 |
| `--backend` | server | 推理后端 (server/local) |
| `--sample-every` | 2 | 每 N 帧处理一帧 |
| `--max-frames` | 0 | 最大处理帧数 (0=全部) |
| `--max-workers` | 48 | VLM 并行请求数 |
| `--output` | 自动生成 | 输出 JSON 路径 |
| `--no-discordance` | False | 禁用矛盾检测 (消融) |
| `--yolo` | False | 启用 YOLO 混合检测 |

### 6.2 UCF-Crime 批量评估

```bash
cd src
python -m v5.eval_ucf_crime \
    --max-videos 40 \
    --sample-every 2 \
    --api-base http://localhost:8000 \
    --parallel 6 \
    --max-workers 48
```

### 6.3 Bad Case 分析

```bash
cd src
python -m v5.analyze_bad_cases \
    --current /path/to/current/results_v5.json \
    --previous /path/to/previous/results_v5.json
```

### 6.4 启动 vLLM Server（前置依赖）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --served-model-name qwen2.5-vl-7b \
    --port 8000 \
    --tensor-parallel-size 1
```

---

## 7. 与 V4 的主要变化

| 维度 | V4 | V5 |
|------|------|------|
| **实体检测** | 全图语义 | 帧差连通域 + 可选 YOLO |
| **实体匹配** | Sentence-BERT 文本匹配 | CLIP 视觉嵌入贪婪匹配 |
| **VLM 输入** | 全图 | 带框全图 / 4宫格拼图 |
| **调用策略** | 固定采样 | 三规则稀疏触发 (Birth/Change/Heartbeat) |
| **矛盾检测** | 无 | DiscordanceChecker (物理-语义矛盾) |
| **全局监控** | 无 | GlobalHeartbeat (CLIP漂移) |
| **图结构** | 简单链表 | TemporalNode + EvolutionEdge + kinetic_integral |
| **叙事** | 简单描述 | 含物理预警 + 轨迹摘要的结构化叙事 |
| **区间定位** | LLM 直接输出 | 动能峰值锚定 + LLM + Fallback 三级 |
| **场景先验** | 业务契约注入 | 零先验判断 |
| **评估** | 基础 AUC | Soft IoU + Hysteresis IoU + 分层采样 + 自动 Bad Case 分析 |
| **YOLO 支持** | 无 | 可选 YOLO-World 混合检测 |
| **消融实验** | 无 | 支持 --no-discordance 消融 |

---

## 8. 项目文件结构

```
EventVAD/
├── src/
│   ├── v5/                          # V5 主系统
│   │   ├── __init__.py
│   │   ├── config.py                # 全局配置
│   │   ├── pipeline.py              # 主流程管线
│   │   ├── eval_ucf_crime.py        # UCF-Crime 评估脚本
│   │   ├── analyze_bad_cases.py     # Bad Case 分析工具
│   │   ├── tracking/                # Stage 1: 物理追踪
│   │   │   ├── motion_extractor.py  # 帧差动能提取
│   │   │   ├── yolo_detector.py     # YOLO-World 检测
│   │   │   ├── hybrid_detector.py   # 帧差+YOLO 融合
│   │   │   ├── clip_encoder.py      # CLIP 特征编码
│   │   │   ├── entity_tracker.py    # 实体贪婪追踪
│   │   │   ├── visual_painter.py    # 可视化标注
│   │   │   └── multi_frame_stacker.py # 多帧拼图
│   │   ├── semantic/                # Stage 2: 语义挂载
│   │   │   ├── node_trigger.py      # 节点触发策略
│   │   │   ├── vllm_semantic.py     # VLM 语义推理
│   │   │   ├── discordance_checker.py # 物理-语义矛盾检测
│   │   │   └── global_heartbeat.py  # 全局心跳+漂移检测
│   │   └── graph/                   # Stage 3: 动态图审计
│   │       ├── structures.py        # 图数据结构
│   │       ├── graph_builder.py     # 图构建器
│   │       ├── narrative_generator.py # 叙事生成器
│   │       └── decision_prompt.py   # 决策审计器
│   ├── event_seg/                   # 事件分割模块 (V1-V3 遗留)
│   ├── score/                       # 评分模块 (V1-V3 遗留)
│   └── evaluate.py                  # 全局评估入口
├── models/                          # 模型权重目录
├── output/                          # 输出结果
│   └── v5/                          # V5 输出
├── docs/                            # 文档
├── scripts/                         # 脚本
└── README.md                        # 项目说明
```

---

## 9. 依赖模型

| 模型 | 用途 | 大小 | 加载方式 |
|------|------|------|---------|
| Qwen2.5-VL-7B-Instruct | 语义推理 + 决策审计 | ~15GB | vLLM Server / 本地加载 |
| CLIP ViT-B/32 | Crop 特征编码 + 漂移检测 | ~600MB | HuggingFace 自动下载 |
| YOLO-World v2 Large | 开放词汇检测 (可选) | ~200MB | 本地权重文件 |

---

## 10. 关键设计决策与原理

### 10.1 为什么用帧差而非 YOLO 作为默认检测？
- 帧差法无需预训练，对任意场景通用
- 帧差直接捕捉"变化"，比静态检测更适合异常检测
- YOLO 作为可选增强，补充火焰/烟雾等静态目标

### 10.2 为什么用 CLIP 而非 Sentence-BERT 做实体匹配？
- CLIP 直接在视觉空间匹配，无需先做文本描述再匹配
- 避免了 VLM 描述不一致导致的匹配错误
- 计算效率高（512 维向量的余弦相似度）

### 10.3 为什么需要 DiscordanceChecker？
- VLM 存在"感知盲区"：对某些异常（如远景打斗）描述为"正常"
- 物理信号（帧差动能）不会说谎，提供了客观的交叉验证
- 多实体投票机制防止繁忙场景的误报

### 10.4 为什么需要多帧拼图？
- 单帧无法判断是否存在"动作序列"（如推搡→倒地→站起→再推）
- 4 宫格拼图给 VLM 提供了时序上下文
- 对 Abuse、Assault 等需要"过程判断"的异常尤为重要

### 10.5 为什么用零先验而非业务契约？
- V4 的业务契约（如"蹲下>10s=异常"）在不同场景下容易产生偏见
- V5 让 Decision LLM 基于叙事文本自主判断，泛化性更好
- 物理预警信息已注入叙事中，LLM 有足够信息做判断
