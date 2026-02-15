# 工业视频异常检测原型系统 V3.0 — 技术文档

> 纯语义时间图版 · Semantic Temporal Graph Anomaly Detection

---

## 一、系统概述

V3.0 系统是一套**不依赖像素坐标或欧氏距离**的视频异常检测系统。核心理念是通过视觉语言模型 (VLLM) 提取视频中每一帧的**实体画像**与**原子动作**，在时间轴上构建一张**有向语义演化图**，然后通过路径模板匹配与多信号融合来判定异常。

### 核心设计思想

| 理念 | 说明 |
|------|------|
| **语义解耦** | 不依赖像素坐标或欧氏距离，仅通过 VLLM 提取的"动作语义"和"实体属性"进行推理 |
| **逻辑连贯性** | 异常定义为"时间轴上的语义路径断裂"或"非法的状态转移" |
| **自适应采样** | 通过画面动能反馈控制 VLLM 的调用频率，确保关键动作不遗漏 |

### 数据流

```
视频 → 自适应采样 → VLLM 感知 → 语义 Re-ID → 时间图 → 路径分析 → 异常分数
```

---

## 二、系统架构

```
src/v3/
├── __init__.py
├── config.py                         # 全局配置中心
├── pipeline.py                       # 主管线入口
├── perception/                       # 第一层: 语义感知层
│   ├── __init__.py
│   ├── prompt_template.py            # VLLM 提示词模板
│   ├── vllm_client.py                # Qwen2-VL / Moondream2 推理客户端
│   └── frame_sampler.py              # 自适应帧采样器
├── association/                      # 第二层: 时间语义关联层
│   ├── __init__.py
│   ├── entity_pool.py                # Sentence-BERT 语义 Re-ID 实体池
│   └── temporal_graph.py             # 时间演化有向图 (NetworkX DiGraph)
├── analysis/                         # 第三层: 路径分析与异常判定层
│   ├── __init__.py
│   ├── path_templates.py             # 正常行为路径模板 + NW 对齐算法
│   └── anomaly_detector.py           # 多信号融合异常检测器
└── utils/                            # 工具层
    ├── __init__.py
    └── json_schema.py                # JSON Schema 校验与清洗
```

---

## 三、各模块详细说明

### 3.1 全局配置 (`config.py`)

统一管理所有模块的超参数，按职责划分为五个配置类：

| 配置类 | 职责 | 关键参数 |
|--------|------|----------|
| `PerceptionConfig` | 语义感知层 | `model_name` (qwen2-vl-7b / moondream2)、`max_new_tokens=1024`、`temperature=0.1`、`frame_size=(448,448)` |
| `SamplerConfig` | 自适应采样器 | `base_fps=0.5`、`max_fps=3.0`、`energy_threshold_low=0.02`、`energy_threshold_high=0.15` |
| `AssociationConfig` | 时间关联层 | `sbert_model=all-MiniLM-L6-v2`、`reid_similarity_threshold=0.70`、`portrait_weight(α)=0.4`、`action_weight(β)=0.6` |
| `AnalysisConfig` | 异常判定层 | `path_match_threshold=0.6`、`anomaly_ema_alpha=0.3`、`min_path_length=3` |
| `SystemConfig` | 系统级 | `device=cuda/cpu`、`fp16_enabled=True`、`num_workers=8` |

HuggingFace 模型缓存统一存放在项目目录 `models/huggingface/` 下，避免使用 `~/.cache`。

---

### 3.2 语义感知层 (Perception Layer)

#### 3.2.1 VLLM 推理客户端 (`vllm_client.py`)

**功能：** 将视频帧送入视觉语言模型，输出结构化 JSON 快照。

**支持的模型：**

| 模型 | HuggingFace 路径 | 说明 |
|------|-------------------|------|
| Qwen2-VL-7B | `Qwen/Qwen2-VL-7B-Instruct` | 高精度，7B 参数，支持 FP16 |
| Moondream2 | `vikhyatk/moondream2` | 轻量级，速度快 |

**核心流程：**
1. 延迟加载模型到 GPU（首次调用时加载）
2. 接收 PIL Image + Prompt → 模型推理 → 原始文本
3. 从文本中提取 JSON（支持纯 JSON、Markdown 代码块、混合文本等多种格式）
4. JSON Schema 校验 + 清洗修复
5. 解析失败时自动重试（最多 3 次），重试使用修复提示词

**输出 JSON Schema：**

```json
{
  "frame_id": 120,
  "timestamp": 4.0,
  "entities": [
    {
      "portrait": "man in blue plaid shirt, approximately 30 years old, wearing glasses",
      "action": "picking up",
      "action_object": "knife",
      "location": "near shelf area",
      "posture": "standing"
    }
  ],
  "scene": {
    "type": "retail store",
    "zones": ["shelf area", "checkout counter"],
    "crowd_density": "low"
  },
  "motion_energy": 0.3
}
```

**关键设计：**
- `portrait` 字段必须足够详细以支持跨帧纯文本 Re-ID
- `action` 采用原子粒度（"拿起""放下""行走"），不使用复合活动
- `motion_energy` 为 [0, 1] 的浮点数，0 = 完全静止，1 = 极端运动
- `temperature=0.1`、`do_sample=False` 以保证 JSON 输出稳定性

#### 3.2.2 提示词模板 (`prompt_template.py`)

精心设计的 System + User 两级提示词：

- **System Prompt**：定义角色（精确的视觉观察助手）、输出规则（严格 JSON、原子动作、详细画像、运动能量估计）
- **User Prompt**：包含完整的 JSON Schema 示例，注入当前帧号和时间戳
- **Retry Prompt**：JSON 解析失败时的修复提示

#### 3.2.3 自适应帧采样器 (`frame_sampler.py`)

**核心思想：** 画面静止时低频采样（节省 VLLM 调用），剧烈运动时自动提升采样率。

**两遍自适应策略：**
1. **第一遍（快速扫描）**：以 ~10 FPS 遍历所有帧，计算帧间差分的像素级运动能量
2. **第二遍（动态决策）**：根据能量曲线动态确定最终采样帧

**采样率调整逻辑：**

```
avg_energy <= 0.02 (低能量)  → base_fps = 0.5 FPS
avg_energy >= 0.15 (高能量)  → max_fps  = 3.0 FPS
中间值                        → 线性插值
```

**运动能量融合：**
- 像素级能量（帧间差分，实时、不依赖 VLLM）权重 0.4
- VLLM 返回的 `motion_energy`（语义级校准信号）权重 0.6

---

### 3.3 时间语义关联层 (Association Layer)

#### 3.3.1 语义 Re-ID 实体池 (`entity_pool.py`)

**核心思想：** 不使用 YOLO / DeepSort 等视觉追踪器，仅通过 VLLM 输出的文字画像 (portrait) 的**语义向量相似度**来判断跨帧的"同一实体"。

**流程：**
1. 每帧 VLLM 返回 `entities` 列表
2. 对每个 entity 的 `portrait` 文本做 **Sentence-BERT** 编码（模型：`all-MiniLM-L6-v2`）
3. 与实体池中已有实体的 portrait 嵌入计算**余弦相似度**
4. 若最高相似度 ≥ 阈值（0.70），判定为**同一实体**（Re-ID 命中）
5. 否则创建**新实体**节点

**关键特性：**

| 特性 | 实现方式 |
|------|----------|
| 批量编码 | 使用 `encode(batch_size=32)` 批量处理当前帧所有实体 |
| 嵌入更新 | Re-ID 命中时使用 EMA (`α=0.3`) 更新嵌入，防止画像漂移 |
| 贪心匹配 | 防止多个新实体匹配到同一个池实体 |
| 池容量管理 | 最大 50 个实体，超容时清理最不活跃的 20% |
| 超时淘汰 | 超过 60 帧不出现的实体视为过期 |

#### 3.3.2 时间演化有向图 (`temporal_graph.py`)

**数据结构：** 基于 NetworkX 的有向图 `DiGraph`

**节点：** `(entity_id, frame_id)` — 表示实体在特定时刻的状态
- 属性：`entity_id`, `frame_id`, `timestamp`, `portrait`, `action`, `action_object`, `location`, `posture`

**有向边：** `(entity_id, frame_i) → (entity_id, frame_j)` — 同一实体的状态转移
- 边权计算公式：

$$w = \alpha \times \text{portrait\_sim} + \beta \times \text{action\_transition\_score}$$

其中 $\alpha = 0.4$, $\beta = 0.6$

**动作转移合理性矩阵（部分示例）：**

| 动作对 | 分数 | 含义 |
|--------|------|------|
| browsing → picking up | 0.90 | 正常购物 |
| picking up → holding | 0.95 | 自然转移 |
| holding → running | 0.20 | 可疑 |
| standing still → fighting | 0.10 | 异常 |
| holding → attacking | 0.05 | 高度异常 |
| walking → walking | 0.95 | 正常持续 |

**自动建边逻辑：** 添加新节点时，若该实体已有历史节点且时间间隔 ≤ 30s，自动建立时间边。

---

### 3.4 路径分析与异常判定层 (Analysis Layer)

#### 3.4.1 正常行为路径模板 (`path_templates.py`)

系统预定义了 **8 套正常行为路径模板**，涵盖以下场景：

| 模板名称 | 适用场景 | 动作序列 |
|----------|----------|----------|
| `normal_shopping` | 零售店 / 超市 | entering → walking → browsing → picking up → holding → walking → handing over → paying → leaving |
| `quick_purchase` | 零售店 / 超市 | entering → walking → picking up → holding → paying → leaving |
| `browsing_only` | 零售店 / 超市 | entering → walking → browsing → walking → leaving |
| `normal_pedestrian` | 街道 / 人行道 | walking → walking → walking |
| `waiting_crossing` | 十字路口 | walking → standing still → walking |
| `parking_normal` | 停车场 | walking → approaching vehicle → opening door → entering vehicle → driving |
| `standing_conversation` | 通用（*） | standing still → talking → standing still |
| `sitting_activity` | 通用（*） | walking → sitting → sitting → standing still → walking |

**路径匹配算法：Needleman-Wunsch 风格动态规划**

- 对齐实体的实际动作序列与模板的动作序列
- 允许跳过模板中的**可选动作**（惩罚更小）
- 支持**同义词模糊匹配**（如 "picking up" ≈ "grabbing" ≈ "snatching"）

**同义词映射（部分）：**

| 规范动作 | 同义词 |
|----------|--------|
| walking | moving, strolling, pacing |
| running | sprinting, rushing, dashing, fleeing |
| picking up | grabbing, taking, grasping, snatching |
| holding | carrying, gripping |
| leaving | exiting, departing, going out, walking away |
| fighting | attacking, hitting, punching, assaulting |

#### 3.4.2 多信号融合异常检测器 (`anomaly_detector.py`)

**四路检测信号：**

| 信号 | 权重 | 计算方式 |
|------|------|----------|
| **路径模板偏离** | 0.30 | `1 - match_score`，匹配度越高异常分越低 |
| **边权重异常** | 0.25 | 平均边权(0.4) + 最低边权(0.4) + 方差(0.2) |
| **语义断裂** | 0.30 | 低动作合理性 + 短时间间隔 = 强断裂信号，使用调和平均 |
| **运动能量** | 0.15 | 平均能量(0.3) + 峰值(0.4) + 能量突增梯度(0.3) |

**融合公式：**

$$\text{raw} = 0.30 \times S_{\text{path}} + 0.25 \times S_{\text{edge}} + 0.30 \times S_{\text{breakage}} + 0.15 \times S_{\text{energy}}$$

**EMA 平滑：**

$$\text{smoothed}_t = \alpha \times \text{raw}_t + (1 - \alpha) \times \text{smoothed}_{t-1}, \quad \alpha = 0.3$$

**视频级异常分数：** 取所有实体中**最高的异常分数**（最可疑实体决定视频异常性）。

---

### 3.5 工具层 (Utils)

#### 3.5.1 JSON Schema 校验 (`json_schema.py`)

- **Schema 定义：** 为 `entity`、`scene`、`frame_snapshot` 三级结构定义了必需和可选字段
- **JSON 提取：** 支持四种格式的 VLLM 输出解析：
  1. 纯 JSON 直接解析
  2. Markdown 代码块 (\`\`\`json ... \`\`\`)
  3. 从混合文本中提取最外层 `{...}` 块
  4. 自动修复尾部逗号、单引号等常见问题
- **清洗标准化：** 填充缺失字段、截断过长文本（portrait ≤ 200 字符、action ≤ 100 字符）

---

### 3.6 主管线 (`pipeline.py`)

#### `VideoAnomalyPipeline` 类

端到端处理单个视频的完整流程：

```
1. 打开视频 → 获取帧率、总帧数、时长
2. 读取所有帧 (BGR)
3. 自适应采样 → 选出关键帧
4. VLLM 感知 → 逐帧推理，生成语义快照
5. 语义 Re-ID + 时间图构建 → EntityPool 匹配 + TemporalGraph 建图
6. 异常检测 → AnomalyDetector 多信号融合
7. 汇总结果 → JSON 输出
```

**输出结果结构：**

```json
{
  "video_path": "path/to/video.mp4",
  "video_name": "video",
  "anomaly_score": 0.5861,
  "duration_sec": 12.5,
  "total_frames": 375,
  "sampled_frames": 25,
  "num_entities_tracked": 3,
  "scene_type": "retail store",
  "entity_results": [
    {
      "entity_id": 1,
      "anomaly_score": 0.5861,
      "path_score": 0.7200,
      "edge_score": 0.5000,
      "breakage_score": 0.8200,
      "energy_score": 0.0000,
      "reason": "anomaly signals: path(0.72), edge(0.50), breakage(0.82); ...",
      "matched_template": "normal_shopping",
      "action_sequence": ["standing still", "picking up", "running", "attacking"]
    }
  ],
  "graph_stats": { "num_nodes": 9, "num_edges": 7, "num_entities": 2 },
  "pool_stats": { "total_entities": 2, "avg_appearances": 4.5 },
  "processing_time_sec": 45.2
}
```

#### 批量处理 (`process_video_directory`)

- 递归扫描目录中的 `.mp4`, `.avi`, `.mkv`, `.mov` 文件
- 模型仅加载一次，复用推理所有视频
- 输出 `batch_summary.json`，包含所有视频的异常分数排名及 Top-5 最异常视频

#### 中间结果持久化

每个视频生成三个文件：
- `analysis_result.json` — 完整分析结果
- `frame_snapshots.json` — 所有帧的 VLLM 语义快照
- `entity_timelines.json` — 每个实体的完整时间路径和边信息

输出目录：`output/v3/<video_name>/`

---

## 四、使用方式

### 命令行

```bash
# 单视频分析
conda run -n eventvad_vllm python -m v3.pipeline \
    --video path/to/video.mp4

# 指定模型
conda run -n eventvad_vllm python -m v3.pipeline \
    --video path/to/video.mp4 \
    --model moondream2

# 指定 GPU
conda run -n eventvad_vllm python -m v3.pipeline \
    --video path/to/video.mp4 \
    --device cuda:1

# 批量处理整个目录
conda run -n eventvad_vllm python -m v3.pipeline \
    --video_dir path/to/videos/

# 调试模式
conda run -n eventvad_vllm python -m v3.pipeline \
    --video path/to/video.mp4 \
    --log-level DEBUG

# 不保存中间结果
conda run -n eventvad_vllm python -m v3.pipeline \
    --video path/to/video.mp4 \
    --no-save
```

### Python API

```python
from v3.pipeline import VideoAnomalyPipeline, process_video_directory

# 单视频
pipeline = VideoAnomalyPipeline(model_name="qwen2-vl-7b")
result = pipeline.process_video("video.mp4")
print(f"异常分数: {result['anomaly_score']:.4f}")
pipeline.cleanup()

# 批量
results = process_video_directory("videos/", model_name="qwen2-vl-7b")
```

---

## 五、业务场景示例

### 场景：购刀流程判定

| 时间点 | VLLM 识别 | 图节点 | 动作转移分数 |
|--------|-----------|--------|-------------|
| T1 (选取) | "man picking up knife near shelf" | $N_1$(picking up) | — |
| T2 (移动) | "man holding knife, walking calmly" | $N_2$(holding) | picking up → holding = **0.95** |
| T3 (交付) | "man handing over knife to cashier" | $N_3$(handing over) | holding → handing over ≈ walking → handing over = **0.80** |

**路径评估：** `picking up → holding → handing over` 与 `normal_shopping` 模板高度匹配 → 异常分数**低位**。

### 对比：抢劫

| 时间点 | VLLM 识别 | 图节点 | 动作转移分数 |
|--------|-----------|--------|-------------|
| T1 | "man picking up knife" | $N_1$(picking up) | — |
| T2 | "man running with knife" | $N_2$(running) | picking up → running = **0.15** ⚠️ |
| T3 | "man attacking with knife" | $N_3$(attacking) | running → attacking = **0.10** ⚠️ |

**路径评估：** `picking up → running → attacking` 与所有模板偏离严重，边权暴跌 → 异常分数**瞬间爆发**。

---

## 六、功能验证结果

通过模拟数据的端到端测试，系统正确区分了正常行为与异常行为：

### 动作转移合理性

| 动作对 | 分数 | 判定 |
|--------|------|------|
| browsing → picking up | 0.90 | ✅ 正常 |
| holding → running | 0.20 | ⚠️ 可疑 |
| walking → walking | 0.95 | ✅ 正常 |
| standing still → fighting | 0.10 | ❌ 异常 |

### 路径模板匹配

| 动作序列 | 最佳模板 | 匹配度 |
|----------|----------|--------|
| entering → walking → browsing → picking up → holding → paying → leaving | normal_shopping | **0.804** |
| entering → picking up → running → running | normal_shopping | **0.279** |

### 综合异常检测

| 实体 | 行为描述 | 异常分数 | 各信号 |
|------|----------|----------|--------|
| Entity #0 (正常购物) | browsing → picking up → holding → paying → leaving | **0.1740** | path=0.08, edge=0.27, break=0.27 |
| Entity #1 (抢夺跑路) | standing still → picking up → running → attacking | **0.5861** | path=0.72, edge=0.50, break=0.82 |

**视频级异常分数：0.5861** — 由最高异常实体决定。

---

## 七、依赖项

```
# 核心依赖
torch >= 2.0
transformers >= 4.37
sentence-transformers
networkx
numpy
opencv-python
Pillow

# VLLM 模型（按需选择）
# Qwen2-VL: qwen-vl-utils
# Moondream2: trust_remote_code=True

# 运行环境
conda 环境: eventvad_vllm
```

---

## 八、配置调优指南

| 场景 | 建议调整 |
|------|----------|
| VLLM 输出不稳定 | 降低 `temperature` 至 0.05，增加 `max_retries` |
| Re-ID 匹配不准 | 降低 `reid_similarity_threshold` 至 0.60 |
| 误报率高 | 提高 `path_match_threshold`，增大 `min_path_length` |
| 漏报率高 | 降低 `breakage_threshold`，增大断裂信号权重 |
| 采样太稀疏 | 提高 `base_fps` 至 1.0，降低 `energy_threshold_high` |
| 显存不足 | 切换至 `moondream2` 模型，或减小 `frame_size` |

---

*文档生成日期: 2026-02-13*
*系统版本: V3.0 Phase 1*
