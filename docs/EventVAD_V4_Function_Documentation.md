# EventVAD V4 功能文档

> 版本：V4（Decision LLM + Temporal Graph）  
> 适用范围：`src/v3/` 主线异常检测系统（UCF-Crime 评估）

---

## 1. 系统目标

EventVAD V4 是一个面向工业监控视频的异常检测系统，目标是：

- 在无需端到端重训练的前提下，基于视觉语言模型完成异常识别；
- 兼顾视频级判定与时间段定位，输出可解释结果；
- 支持本地推理和 vLLM API Server 并行推理；
- 支持 UCF-Crime 数据集的标准化评估（Accuracy / F1 / Frame AUC / IoU）。

---

## 2. 核心功能总览

### 2.1 端到端异常检测

- 输入：单个视频文件（mp4/avi/mkv/mov）。
- 输出：
  - 视频级结果：`Anomaly Detected` / `Normal`；
  - 异常分数：`anomaly_score`；
  - 异常时间段：`anomaly_segments`；
  - 实体级判定：`entity_results`；
  - 可解释文本：`anomaly_explanation`。

### 2.2 双后端感知推理

- `backend=server`：通过 vLLM OpenAI 兼容接口并行调用（推荐）。
- `backend=local`：直接加载 HuggingFace 模型本地推理（调试/离线）。

### 2.3 自适应采样

- 根据画面运动能量动态调整采样密度；
- 低动能降低采样率以加速，高动能提升采样率以减少漏检。

### 2.4 语义 Re-ID 与时间图构建

- 使用 Sentence-BERT 对实体文本画像进行跨帧匹配；
- 建立实体时间路径与关系边，注入帧级能量信号；
- 支持后续因果审计与异常时间定位。

### 2.5 决策审计（Decision LLM）

- 对候选实体进行语义因果审计，输出结构化结论：
  - `is_anomaly`
  - `confidence`
  - `break_timestamp`
  - `reason`
  - `is_cinematic_false_alarm`

### 2.6 规则兜底（Fallback）

- 当 LLM 未检出异常时，使用感知层信号兜底：
  - `visual_danger_score`
  - `is_suspicious`
- 兜底会返回异常区间与原因，用于补抓低动态异常。

### 2.7 时间段精准定位

- 对异常实体进行时间回溯与动能微调；
- 输出 `[start, end]` 异常区间；
- 支持区间合并，减少碎片化段落。

### 2.8 数据集评估能力

- 提供 UCF-Crime 评估脚本；
- 输出指标：
  - Video-level Accuracy / Precision / Recall / F1
  - Frame-level AUC-ROC
  - Video-level AUC-ROC
  - Mean IoU
- 输出明细日志和结果 JSON。

---

## 3. 模块划分与职责

目录：`src/v3/`

- `pipeline.py`
  - 主流程编排：读取视频、采样、感知、关联、审计、定位、结果封装。
- `config.py`
  - 配置中心（感知、采样、关联、审计、定位、系统级参数）。
- `perception/vllm_client.py`
  - 感知层推理客户端，支持 server/local 模式。
- `perception/frame_sampler.py`
  - 自适应采样器。
- `perception/prompt_template.py`
  - 感知提示词模板与输出 schema 约束。
- `association/entity_pool.py`
  - 实体池与语义 Re-ID（Sentence-BERT）。
- `association/temporal_graph.py`
  - 时间图结构与能量信号注入。
- `analysis/causality_auditor.py`
  - 决策审计器（LLM 判定 + 规则兜底）。
- `analysis/anomaly_detector.py`
  - 视频级异常聚合与结果标准化。
- `analysis/temporal_localizer.py`
  - 异常时间区间定位与合并。
- `eval_ucf_crime.py`
  - UCF-Crime 评估入口脚本。
- `utils/json_schema.py`
  - 感知/审计结果 JSON 提取、校验、清洗。

---

## 4. 处理流程（V4）

```text
视频输入
  -> 自适应采样
  -> VLLM感知（结构化快照）
  -> 语义Re-ID（实体对齐）
  -> 构建时间图（含动能）
  -> Decision LLM审计
  -> 规则兜底（必要时）
  -> 时间段定位与合并
  -> 输出结构化结果
```

---

## 5. 输入与输出

### 5.1 输入

- 单视频模式：`--video /path/to/video.mp4`
- 评估模式：UCF-Crime 标注文件 + 视频目录

### 5.2 输出（单视频）

结果字段（核心）：

- `status`: `"Anomaly Detected"` / `"Normal"`
- `anomaly_score`: 视频异常置信度
- `anomaly_segments`: 异常时间段数组
- `anomaly_explanation`: 中文解释
- `entity_results`: 实体级判定列表
- `processing_time_sec`: 处理时长

### 5.3 输出（评估）

- `output/v4/eval_ucf_crime/results_*.json`
  - `metrics`：总体指标
  - `details`：逐视频明细
- `output/v4/eval_ucf_crime/eval_detailed.log`
  - 实时处理日志（逐视频 TP/TN/FP/FN 与 IoU）

---

## 6. 运行方式

### 6.1 单视频检测

```bash
cd /data/liuzhe/EventVAD/src
python -m v3.pipeline \
  --video /data/liuzhe/EventVAD/src/event_seg/videos/xxx.mp4 \
  --mode v4 \
  --backend server \
  --api-base http://localhost:8000 \
  --max-workers 16
```

### 6.2 UCF-Crime 评估

```bash
cd /data/liuzhe/EventVAD/src
python -m v3.eval_ucf_crime \
  --mode v4 \
  --no-contracts \
  --backend server \
  --api-base http://localhost:8000 \
  --max-workers 16 \
  --parallel 2
```

### 6.3 小规模快速验证（建议）

```bash
cd /data/liuzhe/EventVAD/src
python -m v3.eval_ucf_crime \
  --max-videos 20 \
  --mode v4 \
  --no-contracts \
  --backend server \
  --api-base http://localhost:8000 \
  --max-workers 16 \
  --parallel 2
```

---

## 7. 关键配置说明

主要配置位于 `src/v3/config.py`：

- `PerceptionConfig`
  - `model_name`: 感知模型类型
  - `frame_size`: 帧缩放尺寸
  - `max_new_tokens`: 感知输出长度
- `SamplerConfig`
  - `base_fps` / `max_fps`: 采样密度上下限
- `AssociationConfig`
  - `reid_similarity_threshold`: 实体关联阈值
- `DecisionConfig`
  - `decision_max_tokens`: 决策输出长度
  - `min_path_length_for_audit`: 最小审计路径长度
  - `max_audit_entities`: 每视频最大审计实体数
- `LocalizationConfig`
  - `segment_padding_sec`: 时间段前后扩展
  - `max_segment_duration_sec`: 最长异常段长度

---

## 8. 性能与精度优化策略（实践）

### 8.1 性能优化

- 优先使用 `backend=server`（vLLM Continuous Batching）；
- 调整采样密度（`base_fps`）平衡速度与召回；
- 降低 `max_new_tokens`（感知与决策侧）减少延迟；
- 控制 `max_audit_entities`，减少单视频审计成本；
- 评估时关闭 clip 导出，减少磁盘 I/O 开销。

### 8.2 精度优化（尤其帧级 AUC / IoU）

- 优化规则兜底区间：避免“全局 min-max”过宽带来的 IoU 下降；
- 使用高危时间簇而非离散点并集，提升时间定位精度；
- 合理设置 `segment_padding_sec` 和区间合并间隔；
- 对 `visual_danger_score` 阈值进行正常样本校准，抑制误报。

---

## 9. 评估口径说明

- 帧级 AUC 依赖“帧级 GT mask vs 帧级预测分数”；
- IoU 反映异常时间段重叠质量；
- 视频级 Accuracy/F1 不能替代时序定位质量；
- 当测试子集仅包含异常视频时，`video_auc` 可能退化（缺少负类）。

---

## 10. 已知限制

- 对低动态异常（如轻微徘徊、缓慢入侵）仍较敏感于阈值配置；
- 规则兜底若区间过宽，会显著拉低帧级 AUC/IoU；
- LLM 输出格式偶发异常，依赖 JSON 清洗与重试机制；
- 在高并发和长视频场景下，感知层仍是主要耗时阶段。

---

## 11. 推荐交付清单

建议随实验一并归档：

- `results_v4_no_contracts.json`（完整指标 + 明细）
- `eval_detailed.log`（逐视频分析日志）
- `eval_full_stdout.log`（全流程运行日志）
- 本文档（功能说明）

---

## 12. 文档维护建议

- 每次参数策略变更后，更新“关键配置说明”与“评估口径说明”；
- 每次全量评估后，补充“性能/精度表现”快照；
- 若新增模型或后端，优先补充“运行方式”和“已知限制”章节。

