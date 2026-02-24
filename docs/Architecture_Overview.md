# EventVAD 项目全局架构总览

> Training-Free Event-Aware Video Anomaly Detection · ACM Multimedia 2025

---

## 一、项目定位

EventVAD 是一套**无需训练**的视频异常检测系统，核心思想是将长时未剪辑的监控视频首先切分为语义连贯的**事件片段**，再利用视觉语言模型 (VLLM / LMM) 对每个片段进行异常评分，最终输出帧级异常分数并用 AUC 评估。

项目目前包含**三套异常检测路径**：

| 路径 | 代号 | 方法论 | 状态 |
|------|------|--------|------|
| **事件分割 + VLLM 评分** | V1/V2 Pipeline | CLIP + 光流 → 图传播 → 边界检测 → VideoLLaMA2 评分 | ✅ 已上线 |
| **纯语义时间图** | V3 System | VLLM 感知 → 语义 Re-ID → 时间图 → 路径模板匹配 | ✅ 原型完成 |
| **管状骨架 (Tube-Skeleton)** | V5 System | 物理感知追踪 → 聚光灯流 + LMM 视觉语义 → 动态图 + LMM 文本审计 | ✅ 主力系统 |

---

## 二、顶层数据流

```
                    ┌───────────────────────────────────────────────────────┐
                    │                    EventVAD System                    │
                    └───────────────────────────────────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
  ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
  │   V1/V2 Pipeline     │   │     V3 System        │   │   V5 System (主力)   │
  │  (Event Segmentation │   │ (Semantic Temporal   │   │ (Tube-Skeleton       │
  │   + VLLM Scoring)    │   │     Graph)           │   │  Pipeline)           │
  └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
              │                            │                            │
              ▼                            ▼                            ▼
  ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────────┐
  │ CLIP+光流→分割   │       │ VLLM→语义Re-ID   │       │ Stage1: 物理感知追踪 │
  │ → VideoLLaMA2    │       │ → 时间图→路径匹配│       │ Stage2: 聚光灯流+LMM │
  │ → AUC评估        │       │ → 异常检测       │       │ Stage3: 动态图+LMM审计│
  └──────────────────┘       └──────────────────┘       └──────────────────────┘
              │                            │                            │
              ▼                            ▼                            ▼
        帧级异常分数               视频级异常分析              Verdict + 帧级定位
        + AUC 指标                + 实体级分析              + 卷宗叙事 + 逻辑依据
```

---

## 三、目录结构

```
EventVAD/
├── README.md                          # 项目主页（安装 / 使用 / 引用）
├── docs/
│   ├── Architecture_Overview.md       # 本文档 — 全局架构总览
│   ├── EventSeg_Pipeline_Documentation.md  # V1/V2 事件分割管线技术文档
│   └── V3_System_Documentation.md     # V3 纯语义时间图技术文档
│
├── scripts/
│   ├── run_pipeline.sh                # 一键执行完整流程（分割→评分→评估）
│   └── setup_envs.sh                  # Conda 环境自动配置脚本
│
├── src/
│   ├── event_seg/                     # ─── V1/V2 事件分割管线 ───
│   │   ├── config.py                  #   全局配置（特征/图/边界参数）
│   │   ├── main.py                    #   主入口（5种并行模式）
│   │   ├── video_utils.py             #   视频帧读取（多线程/GPU解码）
│   │   ├── feature_extractor.py       #   CLIP + 光流特征提取（多GPU）
│   │   ├── uniseg_processor.py        #   UniSeg 核心（图构建+分割）
│   │   ├── graph_operations.py        #   GAT 风格图传播
│   │   ├── boundary_detection.py      #   事件边界检测（MAD 阈值）
│   │   ├── batch_processor.py         #   批量视频处理器
│   │   ├── video_processing.py        #   视频处理与保存
│   │   ├── benchmark.py               #   性能基准测试
│   │   ├── ucf_crime_*.py             #   UCF-Crime 数据集工具
│   │   ├── videos/                    #   输入视频（xdviolence/ ucf_crime/）
│   │   └── requirements.txt           #   依赖清单
│   │
│   ├── score/                         # ─── VLLM 异常评分 ───
│   │   ├── event_score.py             #   VideoLLaMA2 评分器
│   │   ├── src/videollama2/           #   VideoLLaMA2 模型源码
│   │   └── requirements.txt           #   依赖清单
│   │
│   ├── evaluate.py                    # ─── AUC 评估器 ───
│   │
│   ├── v3/                            # ─── V3 纯语义时间图系统 ───
│   │   ├── config.py                  #   全局配置（5个配置类）
│   │   ├── pipeline.py                #   主管线入口
│   │   ├── perception/                #   第一层: 语义感知
│   │   │   ├── vllm_client.py         #     VLLM 推理客户端
│   │   │   ├── frame_sampler.py       #     自适应帧采样器
│   │   │   └── prompt_template.py     #     提示词模板
│   │   ├── association/               #   第二层: 时间关联
│   │   │   ├── entity_pool.py         #     语义 Re-ID 实体池
│   │   │   └── temporal_graph.py      #     时间演化有向图
│   │   ├── analysis/                  #   第三层: 异常判定
│   │   │   ├── path_templates.py      #     正常行为路径模板
│   │   │   └── anomaly_detector.py    #     多信号融合检测器
│   │   └── utils/
│   │       └── json_schema.py         #     JSON Schema 校验
│   │
│   ├── v5/                            # ─── V5 管状骨架系统 (主力) ───
│   │   ├── config.py                  #   全局配置（11个配置类）
│   │   ├── pipeline.py                #   主管线入口
│   │   ├── tracking/                  #   Stage 1: 物理感知与追踪
│   │   │   ├── motion_extractor.py    #     动能提取（帧差法）
│   │   │   ├── yolo_detector.py       #     开放词汇检测（YOLO-World）
│   │   │   ├── hybrid_detector.py     #     帧差+YOLO 融合
│   │   │   ├── clip_encoder.py        #     CLIP 特征编码
│   │   │   ├── entity_tracker.py      #     实体贪婪追踪
│   │   │   ├── visual_painter.py      #     聚光灯视觉提示
│   │   │   └── multi_frame_stacker.py #     多帧时序构建
│   │   ├── semantic/                  #   Stage 2: 稀疏触发+视觉语义
│   │   │   ├── node_trigger.py        #     稀疏触发网关
│   │   │   ├── vllm_semantic.py       #     LMM 视觉感知调用
│   │   │   ├── discordance_checker.py #     矛盾校验
│   │   │   └── global_heartbeat.py    #     全局心跳+漂移检测
│   │   └── graph/                     #   Stage 3: 动态图组装+审计
│   │       ├── structures.py          #     图数据结构 G^(e)
│   │       ├── graph_builder.py       #     动态图组装
│   │       ├── narrative_generator.py #     卷宗叙事生成
│   │       └── decision_prompt.py     #     LMM 文本审计
│   │
│   ├── LAVIS/                         # ─── LAVIS 框架（CLIP 模型） ───
│   └── RAFT/                          # ─── RAFT 光流模型 ───
│
├── models/
│   └── huggingface/                   # HuggingFace 模型缓存
│
├── output/                            # 输出目录
│   ├── xdviolence/                    #   XD-Violence 结果
│   ├── ucf_crime/                     #   UCF-Crime 结果
│   └── v3/                            #   V3 分析结果
│
└── assets/
    ├── performance(1).png             # UCF-Crime 性能表
    └── performance(2).png             # XD-Violence 性能表
```

---

## 四、两条异常检测路径对比

### 4.1 V1/V2 事件分割 + VLLM 评分

```
视频 → 帧读取(多线程) → CLIP(512d) + 光流(128d) → 640d 特征
     → 动态图构建(GPU) → GAT 图传播 → 边界检测(MAD)
     → 视频分段 + manifest
     → VideoLLaMA2 逐段评分 → 帧级异常分数
     → AUC 评估
```

**优势：**
- 可处理超大规模数据集（4000+ 视频）
- 高吞吐（~150 帧/秒），5 种并行模式
- 分段后的评分可替换为任意 VLLM

**关键模型：**
| 模型 | 用途 | 环境 |
|------|------|------|
| CLIP ViT-B/16 (LAVIS) | 视觉语义特征 | `eventvad_lavis` |
| RAFT / RAFT-Small | 光流运动特征 | `eventvad_lavis` |
| VideoLLaMA2 | 视频片段异常评分 | `eventvad_vllm` |

### 4.2 V3 纯语义时间图

```
视频 → 自适应采样(运动能量) → VLLM 感知(Qwen2-VL / Moondream2)
     → 结构化 JSON 快照(实体·动作·画像·场景)
     → Sentence-BERT 语义 Re-ID
     → 时间演化有向图(NetworkX DiGraph)
     → 路径模板匹配(Needleman-Wunsch DP)
     → 多信号融合(path + edge + breakage + energy)
     → 异常分数(EMA 平滑)
```

**优势：**
- 不依赖像素坐标或欧氏距离，纯语义推理
- 可解释性强：输出具体的异常原因和动作序列
- 适合精细分析单个视频

**关键模型：**
| 模型 | 用途 | 环境 |
|------|------|------|
| Qwen2-VL-7B / Moondream2 | 帧级语义感知 | `eventvad_vllm` |
| Sentence-BERT (all-MiniLM-L6-v2) | 文本画像嵌入 / Re-ID | `eventvad_vllm` |

### 4.3 V5 管状骨架 (Tube-Skeleton Pipeline)

```
原始视频流 (1080P/720P, 常规帧率)
     → Stage 1: 物理感知与追踪
         帧差动能提取 + YOLO-World 开放词汇检测
         → CLIP 视觉特征 z_i → 贪婪匹配 → 实体追踪 (Tracklets)
     → Stage 2: 稀疏触发与视觉语义抽取
         三规则触发 (Birth/Change/Heartbeat)
         → 聚光灯流构建 (回溯τs + 4fps降采样 + 红框视觉提示)
         → LMM 视觉感知 (Qwen2.5-VL / LLaVA-OneVision)
         → 结构化 JSON 语义标签 s_i
     → Stage 3: 动态图组装与文本审计
         → 有向图 G^(e) 构建 + 矛盾校验 (DiscordanceCheck)
         → 卷宗叙事生成 + LMM 文本审计
         → 最终裁决 Verdict (is_anomaly, confidence, 异常区间, reason)
```

**优势：**
- 物理先验 + 语义理解双通道互补
- 稀疏触发策略大幅降低 LMM 调用开销
- 聚光灯流提供时序上下文，解决单帧盲区
- 矛盾校验兜底语义盲区
- 零先验判断，泛化性强
- 可解释性：卷宗叙事 + 逻辑依据

**关键模型：**
| 模型 | 用途 | 环境 |
|------|------|------|
| Qwen2.5-VL-7B / LLaVA-OneVision | LMM 视觉感知 + 文本审计 | `eventvad_vllm` |
| CLIP ViT-B/32 | 实体特征编码 + 漂移检测 | `eventvad_vllm` |
| YOLO-World v2 Large | 开放词汇检测 (可选) | `eventvad_vllm` |

### 4.4 设计哲学差异

| 维度 | V1/V2 Pipeline | V3 System | V5 System |
|------|----------------|-----------|-----------|
| **异常判定点** | 段级（VideoLLaMA2 对每个片段评分） | 实体级（路径演化偏离模板） | 实体级（动态图 + LMM 文本审计） |
| **特征空间** | 视觉特征（CLIP + 光流） | 语义特征（动作 + 画像文本） | 物理动能 + CLIP 视觉 + LMM 语义 |
| **图结构** | 无向图（帧 ↔ 帧，边权 = 语义+运动） | 有向图（实体状态 → 实体状态） | 有向图 G^(e)（TemporalNode + EvolutionEdge） |
| **边界意义** | 事件切换点 | 行为状态转移 | 稀疏触发点 (Birth/Change/Heartbeat) |
| **可解释性** | 低（VLLM 黑盒评分） | 高（可追溯异常路径和断裂点） | 高（卷宗叙事 + 物理预警 + 逻辑依据） |
| **适用规模** | 大规模批量（数千视频） | 精细分析（单视频 / 小批量） | 中大规模（稀疏触发降低开销） |
| **计算开销** | 中（CLIP + 光流 + VideoLLaMA2） | 高（逐帧 VLLM 推理） | 中（稀疏触发 + 并行 LMM） |
| **矛盾检测** | 无 | 无 | DiscordanceChecker + CLIP 漂移 |
| **LMM 调用模式** | 段级评分 | 逐帧感知 | Vision-Mode (Stage 2) + Text-Mode (Stage 3) |

---

## 五、Conda 环境矩阵

项目使用两个独立的 Conda 环境隔离依赖冲突：

```
┌─────────────────┐      ┌─────────────────┐
│  eventvad_lavis  │      │  eventvad_vllm   │
│  Python 3.10     │      │  Python 3.10     │
├─────────────────┤      ├─────────────────┤
│ PyTorch 2.2+cu121│      │ PyTorch 2.2+cu121│
│ LAVIS (CLIP)     │      │ Transformers 4.40│
│ RAFT             │      │ VideoLLaMA2      │
│ OpenCV           │      │ Qwen2-VL / Moon. │
│ NetworkX         │      │ Sentence-BERT    │
│ SciPy            │      │ NetworkX         │
│ scikit-learn     │      │ qwen-vl-utils    │
└─────────────────┘      └─────────────────┘
     ↓ 用于                    ↓ 用于
 事件分割(V1/V2)          VLLM 评分 + V3 系统
 AUC 评估                 端到端异常检测
```

环境自动配置：`bash scripts/setup_envs.sh`

---

## 六、完整执行流程

### 6.1 V1/V2 Pipeline（推荐：大规模数据集）

```bash
# 一键执行（推荐）
MODE=turbo BATCH_SIZE=8 GPUS=0,1 bash scripts/run_pipeline.sh

# 或分步执行：
# Step 1: 事件分割
conda activate eventvad_lavis
cd src/event_seg
python main.py --input ./videos/xdviolence --output ../../output/xdviolence/segments --gpus 0 --turbo --batch_size 4

# Step 2: 异常评分
conda activate eventvad_vllm
python src/score/event_score.py --input_csv output/xdviolence/segments/segment_manifest.txt --output_csv output/xdviolence/scores.txt --gpus 0,1

# Step 3: AUC 评估
conda activate eventvad_lavis
python src/evaluate.py --model_output output/xdviolence/scores.txt --auc_output output/xdviolence/auc.txt
```

### 6.2 V5 System（推荐：主力异常检测）

```bash
conda activate eventvad_vllm
cd src

# 单视频分析
python -m v5.pipeline \
    --video /path/to/video.mp4 \
    --api-base http://localhost:8000 \
    --sample-every 2 --max-workers 48

# UCF-Crime 批量评估
python -m v5.eval_ucf_crime \
    --max-videos 40 --sample-every 2 \
    --api-base http://localhost:8000 \
    --parallel 6 --max-workers 48

# Bad Case 分析
python -m v5.analyze_bad_cases \
    --current output/v5/eval_ucf_crime/run_xxx/results_v5.json \
    --previous output/v5/eval_ucf_crime/run_yyy/results_v5.json
```

### 6.3 V3 System（精细分析）

```bash
conda activate eventvad_vllm
cd src

# 单视频分析
python -m v3.pipeline --video event_seg/videos/xdviolence/video.mp4

# 指定轻量模型
python -m v3.pipeline --video video.mp4 --model moondream2

# 批量分析目录
python -m v3.pipeline --video_dir event_seg/videos/xdviolence/

# 调试模式
python -m v3.pipeline --video video.mp4 --log-level DEBUG
```

---

## 七、关键算法概览

### 7.1 事件分割（V1/V2）

| 阶段 | 算法 | 公式 / 说明 |
|------|------|-------------|
| 特征提取 | CLIP ViT-B/16 + 帧差/RAFT | 640d = CLIP(512d) ⊕ Flow(128d) |
| 图构建 | 加权无向图 | $w_{ij} = \frac{\alpha \cdot \text{CLIP\_sim} + (1-\alpha) \cdot e^{-\text{flow\_dist}}}{1 + \delta \cdot \|t_i - t_j\|}$ |
| 图传播 | GAT 消息传递 | $\mathbf{h}_i^{(l+1)} = 0.5 \cdot \mathbf{h}_i^{(l)} + 0.5 \cdot \sum_{j} \frac{w_{ij}}{d_i} \mathbf{h}_j^{(l)}$ |
| 边界检测 | Savgol + EMA + MAD | $T = \text{median}(s) + 3.0 \times \text{MAD}(s)$ |

### 7.2 语义异常检测（V3）

| 阶段 | 算法 | 公式 / 说明 |
|------|------|-------------|
| 自适应采样 | 运动能量反馈 | 像素能量(0.4) + VLLM 能量(0.6) → 线性插值采样率 |
| 语义 Re-ID | Sentence-BERT 余弦相似度 | $\text{sim}(\mathbf{e}_i, \mathbf{e}_j) \geq 0.70$ → 同一实体 |
| 时间图边权 | 画像相似度 + 动作转移分数 | $w = 0.4 \times \text{portrait\_sim} + 0.6 \times \text{action\_score}$ |
| 路径匹配 | Needleman-Wunsch DP | 对齐实体动作序列与正常模板，支持同义词模糊匹配 |
| 异常融合 | 四路信号加权 + EMA | $S = 0.30 \cdot S_{\text{path}} + 0.25 \cdot S_{\text{edge}} + 0.30 \cdot S_{\text{break}} + 0.15 \cdot S_{\text{energy}}$ |

---

## 八、GPU 加速策略

### 8.1 V1/V2 性能优化总表

| 优化点 | 技术手段 | 效果 |
|--------|----------|------|
| 帧读取 | 多线程(16) + 帧采样(1/3) + 降分辨率 | I/O 非瓶颈 |
| CLIP 推理 | batch=128 + 多GPU + 异步传输 + 动态batch | GPU 利用率 80%+ |
| 光流 | 帧差法替代 RAFT（可选） | 20min → 0.1s/视频 |
| 图构建 | GPU 矩阵运算 + 分块(500) | ~100× 加速 |
| 图传播 | scatter_add_ GPU 聚合 | 毫秒级 |
| OOM 防护 | 动态 batch + 自动降级 + 定期 cache 清理 | 稳定运行 |

### 8.2 五种并行模式

| 模式 | 命令标志 | GPU | 适用场景 |
|------|----------|-----|----------|
| `serial` | 默认 | 单卡 | 调试、小数据 |
| `pipeline` | `--pipeline` | 单卡 | CPU/GPU 重叠 |
| `batch` | `--batch` | 单卡 | 视频短、I/O 瓶颈 |
| `parallel` | `--parallel` | 多卡 | 线性扩展 |
| **`turbo`** | `--turbo` | 单卡 | **推荐**，最高吞吐 |

---

## 九、输出结构

### V1/V2 Pipeline 输出

```
output/
├── xdviolence/
│   ├── segments/                    # 分割的视频片段
│   │   ├── <video>_segments/
│   │   │   ├── segment_0000.mp4
│   │   │   ├── segment_0001.mp4
│   │   │   └── ...
│   │   └── segment_manifest.txt     # 片段清单（路径 起始帧 结束帧）
│   ├── scores.txt                   # VideoLLaMA2 异常评分
│   └── auc.txt                      # AUC 评估结果
└── ucf_crime/
    └── (同上)
```

### V3 System 输出

```
output/v3/
└── <video_name>/
    ├── analysis_result.json          # 完整分析结果（异常分数 + 实体详情）
    ├── frame_snapshots.json          # VLLM 语义快照（每帧）
    └── entity_timelines.json         # 实体时间路径 + 边权信息
```

### V5 System 输出

```
output/v5/
├── <video_name>/
│   └── result.json                   # 完整结果（verdict + graphs + trace_log + timing）
└── eval_ucf_crime/
    └── run_<timestamp>/
        ├── results_v5.json           # 批量评估结果
        └── logs/                     # 逐视频详细日志
```

---

## 十、支持的数据集

| 数据集 | 视频数量 | 异常类型 | 论文 |
|--------|----------|----------|------|
| **XD-Violence** | 4,754 | 暴力 (6 类): 打架、枪击、爆炸、纵火、虐待、交通事故 | ECCV 2020 |
| **UCF-Crime** | 1,900 | 犯罪 (13 类): 纵火、殴打、入室盗窃、爆炸等 | CVPR 2018 |

视频存放位置：
- `src/event_seg/videos/xdviolence/`
- `src/event_seg/videos/ucf_crime/`
- 标注文件：`src/event_seg/videos/annotations.txt`

---

## 十一、模块依赖关系图

```
┌─────────────────────────────────────────────────────────┐
│                    scripts/run_pipeline.sh               │
│               (一键执行 + 环境切换 + 参数配置)              │
└────────┬──────────────────┬──────────────────┬──────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│  event_seg/    │ │   score/       │ │   evaluate.py  │
│  main.py       │ │   event_score  │ │                │
│  (5 种模式)    │ │   .py          │ │  AUC 计算      │
└────┬───────────┘ └───┬────────────┘ └───┬────────────┘
     │                 │                  │
     ▼                 │                  │
┌────────────────┐     │                  │
│ video_utils    │     │                  │
│ feature_extract│     │                  │
│ uniseg_process │     │                  │
│ graph_ops      │     │                  │
│ boundary_detect│     │                  │
│ batch_processor│     │                  │
│ video_process  │     │                  │
└────┬──────┬────┘     │                  │
     │      │          │                  │
     ▼      ▼          ▼                  │
┌────────┐ ┌────┐ ┌──────────┐            │
│ LAVIS  │ │RAFT│ │VideoLLaMA│            │
│ (CLIP) │ │    │ │    2     │            │
└────────┘ └────┘ └──────────┘            │
                                          │
                  ┌───────────────────────-┘
                  ▼
           sklearn.metrics
           (roc_auc_score)


┌─────────────────────────────────────────────────────────┐
│                  v3/pipeline.py                          │
│              (V3 独立端到端管线)                            │
└────┬──────────────────┬──────────────────┬──────────────┘
     ▼                  ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌───────────────┐
│  perception/ │ │ association/ │ │   analysis/   │
│  vllm_client │ │ entity_pool  │ │ path_templates│
│  frame_samp. │ │ temporal_gr. │ │ anomaly_det.  │
└──────┬───────┘ └──────┬───────┘ └───────────────┘
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────┐
│ Qwen2-VL /   │ │Sentence-BERT│
│ Moondream2   │ │(MiniLM-L6)  │
└──────────────┘ └──────────────┘
```

---

## 十二、错误处理与鲁棒性

| 问题 | 应对策略 | 涉及模块 |
|------|----------|----------|
| RAFT shape mismatch | 不完整批次自动填充 + 推理后截断 | `feature_extractor.py` |
| GPU OOM | 动态 batch_size + RAFT→帧差法自动降级 + 定期 `empty_cache()` | `feature_extractor.py` |
| 视频损坏 | OpenCV → FFMPEG 后端回退 + 跳过并记录日志 | `video_utils.py` |
| 编码器不兼容 | mp4v → avc1 → XVID → PNG 序列 | `video_processing.py` |
| VLLM 输出不稳定 | JSON Schema 校验 + 清洗修复 + 重试(3次) | `vllm_client.py`, `json_schema.py` |
| Re-ID 漂移 | 嵌入 EMA 更新 + 实体池容量/超时淘汰 | `entity_pool.py` |

---

## 十三、性能基线

### V1/V2 Pipeline

| 指标 | 数值 |
|------|------|
| 单视频处理速度 | ~150 帧/秒 (~4.3s/视频) |
| 4000 视频 (turbo 单卡) | ~2.5 小时 |
| 4000 视频 (parallel 双卡) | ~1.3 小时 |
| GPU 显存利用率 | 70-85% |
| 最大支持分辨率 | 1920×1080 (自动降采样) |

### V3 System

| 指标 | 数值 |
|------|------|
| 单视频分析 (Qwen2-VL-7B) | ~45s/视频 (10s 视频) |
| 单视频分析 (Moondream2) | ~15s/视频 |
| 自适应采样节省 | VLLM 调用减少 40-60% |
| 实体 Re-ID 准确率 | ~85% (portrait sim ≥ 0.70) |

---

## 十四、快速上手

```bash
# 1. 克隆项目
git clone https://github.com/YihuaJerry/EventVAD.git
cd EventVAD

# 2. 配置环境
bash scripts/setup_envs.sh

# 3. 放置数据集
# 将 XD-Violence 视频放入 src/event_seg/videos/xdviolence/
# 将 UCF-Crime 视频放入 src/event_seg/videos/ucf_crime/

# 4. 运行 V1/V2 Pipeline（大规模评测）
MODE=turbo BATCH_SIZE=8 bash scripts/run_pipeline.sh

# 5. 运行 V3 System（精细分析）
conda activate eventvad_vllm
cd src
python -m v3.pipeline --video event_seg/videos/xdviolence/sample.mp4
```

---

## 十五、引用

```bibtex
@article{shao2025eventvad,
  title={Eventvad: Training-free event-aware video anomaly detection},
  author={Shao, Yihua and He, Haojin and Li, Sijie and Chen, Siyu and Long, Xinwei and Zeng, Fanhu and Fan, Yuxuan and Zhang, Muyang and Yan, Ziyang and Ma, Ao and others},
  journal={arXiv preprint arXiv:2504.13092},
  year={2025}
}
```

---

## 附录 A：配置速查

### V1/V2 事件分割配置 (`src/event_seg/config.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `clip_batch_size` | 128 | CLIP 推理批量大小 |
| `flow_mode` | `"fast"` | 光流模式: fast / sparse / raft |
| `frame_sample_rate` | 3 | 帧采样率 (每N帧取1帧) |
| `max_resolution` | (640, 360) | 最大帧分辨率 |
| `clip_weight` | 0.8 | 图边权中 CLIP 权重 (α) |
| `time_decay` | 0.05 | 时间衰减率 (δ) |
| `mad_multiplier` | 3.0 | MAD 阈值倍率 |
| `min_segment_gap` | 2.0 | 最小分段间隔 (秒) |
| `fp16_enabled` | True | FP16 混合精度 |
| `use_multi_gpu` | True | 多 GPU CLIP |

### V3 系统配置 (`src/v3/config.py`)

| 配置类 | 关键参数 | 默认值 |
|--------|----------|--------|
| `PerceptionConfig` | `model_name` | `"qwen2-vl-7b"` |
| | `temperature` | `0.1` |
| | `max_retries` | `3` |
| `SamplerConfig` | `base_fps` | `0.5` |
| | `max_fps` | `3.0` |
| `AssociationConfig` | `reid_similarity_threshold` | `0.70` |
| | `portrait_weight (α)` | `0.4` |
| | `action_weight (β)` | `0.6` |
| `AnalysisConfig` | `path_match_threshold` | `0.6` |
| | `anomaly_ema_alpha` | `0.3` |
| | `min_path_length` | `3` |

---

## 附录 B：详细文档索引

| 文档 | 路径 | 内容 |
|------|------|------|
| **本文档** | `docs/Architecture_Overview.md` | 项目全局架构、三条路径对比、目录结构、执行流程 |
| **V5 功能文档** | `docs/EventVAD_V5_Function_Documentation.md` | V5 管状骨架系统三阶段详细说明（物理感知→聚光灯流→动态图审计） |
| **V1/V2 技术文档** | `docs/EventSeg_Pipeline_Documentation.md` | 事件分割管线每个模块的详细说明、优化策略、配置调优 |
| **V3 技术文档** | `docs/V3_System_Documentation.md` | 纯语义时间图系统的完整设计、算法公式、业务示例 |
| **项目 README** | `README.md` | 安装说明、数据集准备、基础用法、论文引用 |
| **UCF-Crime 说明** | `src/event_seg/UCF_CRIME_README.md` | UCF-Crime 数据集专用工具说明 |

---

*文档更新日期: 2026-02-23*
*系统版本: EventVAD V5 (ACM Multimedia 2025)*
