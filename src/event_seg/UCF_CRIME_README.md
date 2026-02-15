# UCF Crime 数据集处理指南 - EventVAD 适配

本文档说明如何使用 EventVAD 处理 UCF Crime 数据集进行视频异常检测的事件分割。

## 数据集概述

**UCF Crime** 是一个大规模的真实世界监控视频异常检测数据集，包含：
- **13种异常类别**：Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
- **正常视频**：Normal
- **视频帧率**：30 fps
- **总视频数**：约1900个视频

## 处理流程

### 步骤 1：准备数据集

首先运行数据准备脚本，解压并组织 UCF Crime 数据集：

```bash
cd /data/liuzhe/EventVAD/src/event_seg

# 解压并组织数据
python ucf_crime_prepare.py \
    --ucf_dir ./videos/ucf_crime \
    --extract_dir ./videos/ucf_crime_extracted \
    --organized_dir ./videos/ucf_crime_organized
```

**参数说明**：
- `--ucf_dir`：原始数据目录（包含zip文件）
- `--extract_dir`：解压后的临时目录
- `--organized_dir`：组织后的最终目录
- `--skip_extract`：跳过解压步骤（如已解压）
- `--skip_organize`：跳过组织步骤

**输出结构**：
```
ucf_crime_organized/
├── train/
│   ├── anomaly/
│   │   ├── Abuse/
│   │   ├── Arrest/
│   │   └── ...
│   └── normal/
├── test/
│   ├── anomaly/
│   │   ├── Abuse/
│   │   └── ...
│   └── normal/
└── video_list.txt
```

### 步骤 2：运行 EventVAD 处理

使用 EventVAD 对视频进行事件分割：

```bash
# 处理所有视频
python ucf_crime_process.py \
    --input ./videos/ucf_crime_organized \
    --output ./output/ucf_crime \
    --annotation ./videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
    --gpu 0

# 只处理测试集
python ucf_crime_process.py \
    --input ./videos/ucf_crime_organized \
    --output ./output/ucf_crime \
    --subset test \
    --gpu 0

# 只处理特定类别
python ucf_crime_process.py \
    --input ./videos/ucf_crime_organized \
    --output ./output/ucf_crime \
    --category Robbery \
    --gpu 0
```

**参数说明**：
- `--input`：组织后的视频目录
- `--output`：输出目录
- `--annotation`：时间标注文件
- `--subset`：数据子集 (all/train/test/anomaly/normal)
- `--category`：指定类别
- `--gpu`：GPU编号

**输出文件**：
- `segment_manifest.txt`：分割清单（TSV格式）
- `segment_manifest_annotated.json`：带完整标注的JSON清单
- `processing_stats.json`：处理统计信息
- `empty_videos.log`：空结果视频记录

### 步骤 3：评估分割效果

评估 EventVAD 的分割与 ground truth 的对齐情况：

```bash
python ucf_crime_evaluate.py \
    --manifest ./output/ucf_crime/segment_manifest.txt \
    --annotation ./videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
    --output ./output/ucf_crime/evaluation_results.json
```

**评估指标**：
1. **边界检测指标**
   - 平均边界误差（秒）
   - 1秒内边界检测率
   - 起始/结束边界检测率

2. **覆盖度指标**
   - 平均覆盖率
   - 完全覆盖率
   - 碎片化程度

3. **片段统计**
   - 各类别片段数
   - 异常片段比例
   - 平均重叠度

## 输出格式

### segment_manifest.txt

```
segment_path	start_frame	end_frame	category	contains_anomaly	overlap_ratio
/path/to/segment_0000.mp4	0	150	Robbery	0	0.0000
/path/to/segment_0001.mp4	150	450	Robbery	1	0.8500
```

### segment_manifest_annotated.json

```json
{
  "processed_videos": ["video1.mp4", "video2.mp4"],
  "segments": [
    {
      "segment_path": "/path/to/segment_0000.mp4",
      "start_frame": 0,
      "end_frame": 150,
      "start_time": 0.0,
      "end_time": 5.0,
      "contains_anomaly": false,
      "anomaly_overlap_ratio": 0.0,
      "source_video": "/path/to/video.mp4",
      "category": "Robbery"
    }
  ]
}
```

## 标注格式说明

原始 UCF Crime 时间标注格式：
```
视频名  类别  起始帧1  结束帧1  起始帧2  结束帧2
```

示例：
```
Robbery137_x264.mp4  Robbery  135  1950  -1  -1
Shooting046_x264.mp4  Shooting  4005  4230  4760  5088
```

- 帧号 `-1` 表示无对应的异常区间
- 部分视频有两个异常区间
- 帧率固定为 30 fps

## 注意事项

1. **磁盘空间**：完整处理需要约 100GB 磁盘空间
2. **处理时间**：单个视频约 1-5 分钟（取决于视频长度和 GPU）
3. **断点续传**：脚本支持中断后继续处理，已处理的视频会自动跳过
4. **GPU 内存**：建议使用至少 8GB 显存的 GPU

## 引用

如果使用此数据集，请引用：

```bibtex
@inproceedings{sultani2018real,
  title={Real-world anomaly detection in surveillance videos},
  author={Sultani, Waqas and Chen, Chen and Shah, Mubarak},
  booktitle={CVPR},
  year={2018}
}
```
