# EventVAD 事件分割流水线 — 高性能优化技术文档

> Event Segmentation Pipeline · GPU-Accelerated Video Processing

---

## 一、系统概述

事件分割流水线是 EventVAD 系统的**第一阶段处理器**，负责将长时未剪辑的监控视频自动切分为语义连贯的**事件片段 (Event Segments)**。每个片段代表一个独立的"事件"，后续交由 VLLM 评分模块进行异常打分。

### 核心处理流程

```
视频输入 → 帧读取(多线程) → CLIP特征 + 光流特征 → 动态图构建(GPU) → GAT图传播 → 边界检测 → 分段输出
```

### 关键性能指标

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 单视频处理时间 | ~20 分钟 | ~4.3 秒 |
| 4000 视频总耗时 | ~55 天 | ~2.5 小时 |
| GPU 显存利用率 | ~10% | ~70-85% |
| 处理吞吐量 | ~2 帧/秒 | ~150 帧/秒 |

---

## 二、系统架构

```
src/event_seg/
├── config.py                # 全局配置中心
├── main.py                  # 主入口（5种并行模式）
├── video_utils.py           # 视频帧读取（多线程/GPU 加速）
├── feature_extractor.py     # CLIP + 光流 特征提取器（多GPU）
├── uniseg_processor.py      # UniSeg 处理器（图构建 + 分割）
├── graph_operations.py      # 图传播算法（GAT 风格）
├── boundary_detection.py    # 事件边界检测
├── batch_processor.py       # 批量视频处理器
├── video_processing.py      # 视频处理与保存
├── benchmark.py             # 性能基准测试
├── ucf_crime_*.py           # UCF-Crime 数据集专用工具
└── requirements.txt         # Python 依赖
```

---

## 三、各模块详细说明

### 3.1 全局配置 (`config.py`)

所有超参数集中管理在 `Config` 类中：

```python
class Config:
    device = "cuda"
    fp16_enabled = True        # FP16 混合精度

    # 特征提取
    clip_model_name = "clip"
    feature_dim = 640          # CLIP(512) + Flow(128)
    clip_batch_size = 128      # CLIP 批量大小
    flow_batch_size = 8        # 光流批量大小上限

    # RAFT 光流
    raft_small = True          # RAFT-Small（更快）
    raft_iters = 6             # 迭代次数
    flow_mode = "fast"         # fast / sparse / raft
    flow_sample_rate = 5       # sparse 模式采样率

    # 帧读取
    frame_sample_rate = 3      # 每3帧取1帧
    max_resolution = (640, 360)
    num_workers = 16           # 多线程数

    # 动态图
    time_decay = 0.05
    clip_weight = 0.8
    graph_block_size = 500
    gat_iters = 1

    # 边界检测
    ema_window = 2.0
    mad_multiplier = 3.0
    min_segment_gap = 2.0

    # 显存
    gpu_memory_reserve = 0.3
    use_multi_gpu = True
```

---

### 3.2 视频帧读取 (`video_utils.py`)

提供三种帧读取方式，按性能递增：

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| `video_to_frames()` | 标准 OpenCV + 多线程 resize | 通用 |
| `video_to_frames_fast()` | 限制最大帧数 + 动态采样 | 大规模批处理 |
| `video_to_frames_decord()` | GPU 硬件解码 (decord) | 高吞吐场景 |

**关键优化：**

1. **帧采样**：按 `frame_sample_rate` 间隔采样，减少处理帧数
2. **多线程 resize**：使用 `ThreadPoolExecutor` 并行处理帧缩放和色彩空间转换
3. **尺寸对齐**：输出帧尺寸对齐到 8 的倍数（RAFT 要求）
4. **动态采样**：对超长视频自动增大采样间隔，确保帧数不超过阈值

```
原始尺寸(1920×1080) → 降采样(640×360) → 8倍对齐(640×360) → RGB
```

---

### 3.3 特征提取器 (`feature_extractor.py`)

每一帧提取 **640 维**特征向量 = CLIP(512 维) ⊕ Flow(128 维)

#### 3.3.1 CLIP 特征提取

**模型**：LAVIS 框架下的 CLIP ViT-B/16

**单卡模式 (`FeatureExtractor`)：**
1. 多线程预处理（PIL Image → CLIP transforms）
2. 批量 stack 到 CPU tensor
3. 异步传输到 GPU（`non_blocking=True`）
4. 批量推理（batch_size=128）
5. 定期清理显存（每 10 批次）

**多卡模式 (`MultiGPUClipExtractor`)：**
1. 在每张 GPU 上加载独立的 CLIP 模型副本
2. 按帧数均匀分配到各 GPU
3. 多线程并行推理
4. 合并结果

**显存保护机制 (`_adjust_batch_size`)：**
- 实时查询 GPU 可用显存
- 估算单帧显存占用（像素 + ViT 中间变量 ~50MB/帧）
- 动态下调 batch_size 防止 OOM

#### 3.3.2 光流特征提取

提供三种光流模式，按精度 / 速度递增排列：

| 模式 | 方法 | 速度 | 精度 | 显存占用 |
|------|------|------|------|----------|
| `raft` | 完整 RAFT 模型 | 慢 | 高 | 高 |
| `sparse` | RAFT 稀疏采样 + 线性插值 | 中 | 中 | 中 |
| `fast` | 帧差法近似 | **极快** | 低 | **极低** |

**`fast` 模式（帧差法，默认推荐）：**

```
帧序列 → RGB转灰度(GPU) → 相邻帧差分 → 水平/垂直梯度 → 2D向量 → 正交投影(128维)
```

- 分块处理（每次 500 帧），防止大视频爆显存
- 跨块边界帧使用 CPU 回退计算
- 正交投影矩阵预初始化（固定种子 42）

**`raft` 模式（完整 RAFT）：**

- 使用 RAFT-Small 模型（更快）
- 6 次迭代（标准 12 次的一半）
- FP16 混合精度加速
- **不完整批次填充**：尾部不足 batch_size 的批次用最后一帧重复填充，推理后截断

**`sparse` 模式（稀疏 RAFT）：**

- 每 `flow_sample_rate` 帧调用一次 RAFT
- 中间帧使用线性插值
- 速度 ≈ RAFT 的 1/sample_rate

**OOM 容错：**
```python
try:
    flow_feats = extract_flow_features_raft(frames)
except RuntimeError("out of memory"):
    flow_feats = extract_flow_features_fast(frames)  # 自动降级
```

---

### 3.4 UniSeg 处理器 (`uniseg_processor.py`)

整合特征提取、图构建、图传播和边界检测的**核心处理器**。

#### 3.4.1 动态图构建

**数据结构**：NetworkX 无向图 `Graph`

**节点**：每一帧 → 一个节点，属性为该帧的 640 维特征向量

**边权**：综合 CLIP 语义相似度、光流运动距离和时间衰减

$$w_{ij} = \frac{\alpha \cdot \text{CLIP\_sim}(i,j) + (1 - \alpha) \cdot e^{-\text{flow\_dist}(i,j)}}{1 + \delta \cdot |t_i - t_j|}$$

其中 $\alpha = 0.8$（CLIP 权重），$\delta = 0.05$（时间衰减率）

**GPU 加速策略：**
1. 全部特征转到 GPU
2. 分块计算相似度矩阵（block_size=500，防止 $O(n^2)$ 爆显存）
3. GPU 上做 top-k 选择（每节点取 3-5 条最强边）
4. 批量转回 CPU 构建 NetworkX 图

**快速图构建（Turbo 模式）：**
- 仅连接局部窗口（前后 10 帧）
- CPU numpy 向量化，无需 GPU
- 适合短视频批量处理

---

### 3.5 图传播算法 (`graph_operations.py`)

**算法**：GAT（Graph Attention Network）风格的消息传递

$$\mathbf{h}_i^{(l+1)} = 0.5 \cdot \mathbf{h}_i^{(l)} + 0.5 \cdot \sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{d_i} \mathbf{h}_j^{(l)}$$

**实现细节：**

1. 构建稀疏邻接矩阵的边列表和权重（双向）
2. 度归一化权重
3. 使用 `scatter_add_` 高效聚合邻居特征（GPU 加速）
4. 残差连接（0.5 权重）
5. 迭代传播 `gat_iters` 次（默认 1 次）

**两种实现：**

| 实现 | 依赖 | 说明 |
|------|------|------|
| `graph_propagation()` | PyTorch scatter_add_ | 默认，兼容性好 |
| `graph_propagation_sparse()` | torch_sparse (可选) | 超大图更高效 |

---

### 3.6 事件边界检测 (`boundary_detection.py`)

从传播后的节点特征中检测**事件切换时刻**。

**流程：**

```
相邻帧差异 → 余弦不相似度 → Savgol滤波 → EMA移动平均 → 信号/均值比 → MAD阈值 → 合并相近边界 → 时间边界
```

**详细步骤：**

1. **差异计算**（GPU / CPU 自动选择）：
   - 欧式距离差异：$s_i^{L2} = \|\mathbf{h}_i - \mathbf{h}_{i+1}\|^2$
   - 余弦不相似度：$s_i^{cos} = 1 - \frac{\mathbf{h}_i \cdot \mathbf{h}_{i+1}}{\|\mathbf{h}_i\| \|\mathbf{h}_{i+1}\|}$
   - 综合分数：$s_i = s_i^{L2} + s_i^{cos}$

2. **双重平滑**：
   - Savitzky-Golay 滤波（窗口 = `fps × ema_window`，多项式阶 2）
   - EMA 移动平均

3. **阈值检测**：
   - 计算信号 / EMA 比值
   - **MAD（Median Absolute Deviation）阈值**：$T = \text{median} + 3.0 \times \text{MAD}$
   - 超过阈值的位置即为候选边界

4. **后处理**：
   - 合并间隔 < `min_segment_gap × fps` 的相近边界
   - 转换为 `(start_sec, end_sec)` 时间元组

---

### 3.7 批量视频处理器 (`batch_processor.py`)

**核心思想**：将多个视频的帧合并为一个大批次，统一提取 CLIP 特征，最大化 GPU 利用率。

```
[视频A 100帧] + [视频B 200帧] + [视频C 150帧]
           ↓ 合并
    [450帧统一 CLIP 推理]
           ↓ 拆分
    [各视频独立后处理]
```

**四步流水线：**

| 步骤 | 操作 | 并行方式 |
|------|------|----------|
| 1. 帧读取 | 并行读取多个视频 | `ThreadPoolExecutor` |
| 2. CLIP 特征 | 合并帧批量推理 | GPU batch |
| 3. 图处理 | 逐视频图构建 + 边界检测 | 串行 (CPU + GPU) |
| 4. 视频保存 | 并行写入分段视频 | `ThreadPoolExecutor` |

---

## 四、五种并行处理模式

`main.py` 提供了五种处理模式，适配不同硬件和场景：

### 4.1 串行模式 (`serial`)

```bash
python main.py --input ./videos --output ./output --gpus 0
```

- 逐视频串行处理
- 模型只初始化一次，复用处理所有视频
- 适合调试和小规模数据

### 4.2 流水线模式 (`pipeline`)

```bash
python main.py --input ./videos --output ./output --gpus 0 --pipeline --prefetch 2 --save_workers 2
```

- **三阶段流水线**：预读取(CPU) → GPU推理 → 保存(CPU)
- 预取队列控制内存占用
- CPU 和 GPU 操作重叠

```
时间线:
CPU: [读取V1] [读取V2] [读取V3] ...
GPU:          [推理V1] [推理V2] [推理V3] ...
CPU:                   [保存V1] [保存V2] ...
```

### 4.3 批量模式 (`batch`)

```bash
python main.py --input ./videos --output ./output --gpus 0 --batch --batch_size 4
```

- 每次并行读取 N 个视频的帧到内存
- 逐个视频在 GPU 上处理
- 适合视频较短、I/O 瓶颈明显的场景

### 4.4 多 GPU 并行 (`parallel`)

```bash
python main.py --input ./videos --output ./output --gpus 0,1 --parallel
```

- 每张 GPU 启动独立工作进程（`multiprocessing`）
- 视频队列均匀分配
- 结果通过进程间队列汇总
- 线性扩展：2 GPU ≈ 2× 速度

### 4.5 高吞吐模式 (`turbo`) — **推荐**

```bash
python main.py --input ./videos --output ./output --gpus 0 --turbo --batch_size 4
```

- 使用 `BatchVideoProcessor`
- 多视频帧合并统一 CLIP 推理
- 快速光流（帧差法）
- 快速图构建（局部窗口）
- 并行保存

---

## 五、事件分割执行脚本 (`scripts/run_pipeline.sh`)

### 完整三步流程

```
Step 1: 事件分割（event_seg）→ 视频片段 + manifest
Step 2: 异常评分（score）→ 每片段的异常分数
Step 3: 评估（evaluate）→ AUC 指标
```

### 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GPUS` | `0,1` | GPU ID，逗号分隔 |
| `MODE` | `turbo` | 处理模式：serial/pipeline/batch/parallel/turbo |
| `BATCH_SIZE` | `4` | batch/turbo 模式的批量大小 |
| `PREFETCH` | `2` | pipeline 模式的预取队列 |
| `SAVE_WORKERS` | `2` | pipeline 模式的保存线程 |
| `DATASET` | `all` | 数据集：xdviolence/ucf_crime/all |

### 使用示例

```bash
# 快速处理（推荐）
MODE=turbo BATCH_SIZE=8 bash scripts/run_pipeline.sh

# 双卡并行
MODE=parallel GPUS=0,1 bash scripts/run_pipeline.sh

# 只处理 XD-Violence
DATASET=xdviolence bash scripts/run_pipeline.sh

# 串行调试
MODE=serial bash scripts/run_pipeline.sh
```

### 输出目录结构

```
output/
├── xdviolence/
│   ├── segments/              # 分割的视频片段
│   │   ├── video_name_segments/
│   │   │   ├── segment_0000.mp4
│   │   │   ├── segment_0001.mp4
│   │   │   └── ...
│   │   └── segment_manifest.txt   # 片段清单
│   ├── scores.txt             # 异常评分
│   └── auc.txt                # AUC 评估结果
└── ucf_crime/
    ├── segments/
    ├── scores.txt
    └── auc.txt
```

### Manifest 文件格式

每行一个片段：`<绝对路径> <起始帧> <结束帧>`

```
/data/liuzhe/EventVAD/output/xdviolence/segments/video_segments/segment_0000.mp4 0 150
/data/liuzhe/EventVAD/output/xdviolence/segments/video_segments/segment_0001.mp4 150 320
```

---

## 六、性能优化详解

### 6.1 帧读取优化

| 优化手段 | 效果 |
|----------|------|
| 帧采样 1/3 | 处理帧数减少 67% |
| 分辨率降到 640×360 | 单帧内存减少 ~75% |
| 多线程 resize (16 线程) | resize 耗时减少 ~10× |
| decord GPU 解码 | I/O 瓶颈消除 |

### 6.2 CLIP 特征优化

| 优化手段 | 效果 |
|----------|------|
| batch_size=128 | GPU 利用率 10% → 80% |
| 多线程预处理 (16 线程) | 预处理从瓶颈变为非瓶颈 |
| 预 stack + 异步传输 | 数据传输与推理重叠 |
| 多 GPU 并行 | 线性加速 |
| 动态 batch_size | 防止 OOM |

### 6.3 光流特征优化

| 优化手段 | 效果 |
|----------|------|
| 帧差法替代 RAFT | 20分钟/视频 → 0.1秒/视频 |
| RAFT-Small + 6 iters | RAFT 速度 ↑ ~50% |
| FP16 混合精度 | 显存 ↓ 50%，速度 ↑ ~30% |
| 不完整批次填充 | 消除 RAFT shape mismatch 错误 |
| OOM 自动降级 | 稳定性保障 |

### 6.4 图构建优化

| 优化手段 | 效果 |
|----------|------|
| GPU 矩阵乘法 | 相似度计算加速 ~100× |
| 分块计算 (block=500) | 避免 $O(n^2)$ 显存爆炸 |
| GPU top-k 选择 | 减少 CPU-GPU 数据传输 |
| 局部窗口图（turbo 模式） | 边数大幅减少 |

### 6.5 图传播优化

| 优化手段 | 效果 |
|----------|------|
| scatter_add_ 聚合 | 避免稀疏矩阵构造 |
| 仅 1 次迭代 | 速度 ↑ 与精度平衡 |
| torch_sparse 可选 | 超大图进一步加速 |

### 6.6 边界检测优化

| 优化手段 | 效果 |
|----------|------|
| GPU 向量化差异计算 | 大幅加速 |
| Savgol + EMA 双重平滑 | 减少误检 |
| MAD 自适应阈值 | 无需手动调参 |

---

## 七、错误处理与容错机制

### 7.1 RAFT Shape Mismatch

**问题**：不完整批次导致 `grid_sampler()` 输入维度不匹配。

**解决**：

```python
if remainder < batch_size:
    pad_size = batch_size - remainder
    prev_batch = torch.cat([prev_batch, prev_batch[-1:].repeat(pad_size,1,1,1)])
    curr_batch = torch.cat([curr_batch, curr_batch[-1:].repeat(pad_size,1,1,1)])

# 推理后截断
pooled = pooled[:remainder]
```

### 7.2 显存溢出 (OOM)

**三级容错：**

1. **预防**：动态调整 batch_size（`_adjust_batch_size`）
2. **降级**：RAFT OOM → 自动回退到帧差法
3. **清理**：每 10 批次主动 `torch.cuda.empty_cache()`

### 7.3 视频文件损坏

- OpenCV 打开失败 → 尝试 FFMPEG 后端
- 空视频 → 记录到 `empty_videos.log`
- 帧读取异常 → 跳过该视频，继续处理队列

### 7.4 编码器兼容性

按优先级尝试 `mp4v` → `avc1` → `XVID`，全部失败则回退到 PNG 序列保存。

---

## 八、配置调优指南

### 速度优先（推荐）

```python
flow_mode = "fast"           # 帧差近似
frame_sample_rate = 3        # 每3帧取1帧
max_resolution = (640, 360)  # 低分辨率
clip_batch_size = 128        # 大批量
num_workers = 16             # 多线程
```

**预估速度**：~150 帧/秒，4000 视频 ~2.5 小时

### 精度优先

```python
flow_mode = "raft"           # 完整 RAFT
frame_sample_rate = 1        # 不采样
max_resolution = (1280, 720) # 高分辨率
clip_batch_size = 32         # 小批量（防 OOM）
raft_iters = 12              # 标准迭代
```

### 显存受限（< 16GB）

```python
flow_mode = "fast"
clip_batch_size = 32
frame_sample_rate = 5
max_resolution = (480, 270)
gpu_memory_reserve = 0.4     # 预留更多
```

### 多 GPU 场景

```python
use_multi_gpu = True
# 启动命令
python main.py --gpus 0,1,2,3 --parallel
# 或
python main.py --gpus 0 --turbo --batch_size 8  # 单卡最大吞吐
```

---

## 九、支持的数据集

| 数据集 | 视频数 | 异常类型 | 输入目录 |
|--------|--------|----------|----------|
| **XD-Violence** | 4754 | 暴力 (6 类) | `src/event_seg/videos/xdviolence/` |
| **UCF-Crime** | 1900 | 犯罪 (13 类) | `src/event_seg/videos/ucf_crime/` |

---

## 十、依赖项

```
# 核心
torch >= 2.0 (with CUDA)
numpy
opencv-python
networkx
scipy
tqdm
Pillow

# CLIP 模型（LAVIS）
salesforce-lavis

# 光流（可选）
RAFT (src/RAFT/)

# GPU 解码（可选）
decord

# 高效稀疏运算（可选）
torch-sparse

# 运行环境
conda 环境: eventvad_lavis
```

---

## 十一、Conda 环境说明

| 环境名 | 用途 | 主要依赖 |
|--------|------|----------|
| `eventvad_lavis` | 事件分割 + 评估 | PyTorch, LAVIS, RAFT |
| `eventvad_vllm` | V3 系统 + VLLM 评分 | Transformers, Sentence-BERT |
| `score` | 传统评分 | 参见 `src/score/requirements.txt` |

---

*文档生成日期: 2026-02-13*
*系统版本: EventVAD Event Segmentation Pipeline v2.0 (GPU-Optimized)*
