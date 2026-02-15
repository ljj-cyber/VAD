#!/bin/bash

# EventVAD 完整处理流程脚本
# 包含：事件分割 → 异常评分 → 评估
# 支持多种并行模式

set -e

PROJECT_ROOT="/data/liuzhe/EventVAD"
VIDEOS_DIR="$PROJECT_ROOT/src/event_seg/videos"
OUTPUT_DIR="$PROJECT_ROOT/output"

# =====================================
# 配置参数
# =====================================
# GPU 配置（多个 GPU 用逗号分隔）
GPUS="${GPUS:-0,1}"

# 处理模式：serial, pipeline, batch, parallel, turbo
# - serial:   串行处理（调试用）
# - pipeline: 流水线模式（CPU/GPU 重叠）
# - batch:    批量模式（同时读取多个视频）
# - parallel: 多 GPU 并行（需要多张 GPU）
# - turbo:    高吞吐模式（推荐，最快）
MODE="${MODE:-turbo}"

# 批量大小（batch 模式）
BATCH_SIZE="${BATCH_SIZE:-4}"

# 预取队列大小（pipeline 模式）
PREFETCH="${PREFETCH:-2}"

# 保存线程数（pipeline 模式）
SAVE_WORKERS="${SAVE_WORKERS:-2}"

# 要处理的数据集（可选：xdviolence, ucf_crime, all）
DATASET="${DATASET:-all}"

# =====================================
# 辅助函数
# =====================================
activate_env() {
    local env_name="$1"
    source ~/miniconda3/bin/activate "$env_name"
    echo "  [环境] 已激活: $env_name"
}

check_gpu() {
    echo "  [GPU] 检测可用 GPU..."
    python -c "import torch; print(f'    可用 GPU: {torch.cuda.device_count()} 张')"
    python -c "import torch; [print(f'    GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
}

get_mode_args() {
    case "$MODE" in
        serial)
            echo "--gpus ${GPUS%%,*}"
            ;;
        pipeline)
            echo "--gpus ${GPUS%%,*} --pipeline --prefetch $PREFETCH --save_workers $SAVE_WORKERS"
            ;;
        batch)
            echo "--gpus ${GPUS%%,*} --batch --batch_size $BATCH_SIZE"
            ;;
        parallel)
            echo "--gpus $GPUS --parallel"
            ;;
        turbo)
            echo "--gpus ${GPUS%%,*} --turbo --batch_size $BATCH_SIZE"
            ;;
        *)
            echo "--gpus ${GPUS%%,*} --turbo --batch_size $BATCH_SIZE"
            ;;
    esac
}

run_segmentation() {
    local dataset="$1"
    local input_dir="$2"
    local output_dir="$3"
    
    echo ""
    echo "[分割] $dataset"
    echo "  输入: $input_dir"
    echo "  输出: $output_dir"
    echo "  模式: $MODE"
    
    if [ ! -d "$input_dir" ]; then
        echo "  [跳过] 输入目录不存在"
        return
    fi
    
    mkdir -p "$output_dir"
    
    activate_env eventvad_lavis
    
    local mode_args=$(get_mode_args)
    echo "  [参数] $mode_args"
    
    python "$PROJECT_ROOT/src/event_seg/main.py" \
        --input "$input_dir" \
        --output "$output_dir" \
        $mode_args
    
    echo "  [完成] 分割结果: $output_dir/segment_manifest.txt"
}

run_scoring() {
    local dataset="$1"
    local manifest="$2"
    local output="$3"
    
    echo ""
    echo "[评分] $dataset"
    echo "  输入: $manifest"
    echo "  输出: $output"
    
    if [ ! -f "$manifest" ]; then
        echo "  [跳过] manifest 文件不存在"
        return
    fi
    
    if [ ! -s "$manifest" ]; then
        echo "  [跳过] manifest 文件为空"
        return
    fi
    
    activate_env eventvad_vllm
    
    echo "  [模式] 多 GPU 推理 ($GPUS)"
    python "$PROJECT_ROOT/src/score/event_score.py" \
        --input_csv "$manifest" \
        --output_csv "$output" \
        --gpus "$GPUS"
    
    echo "  [完成] 评分结果: $output"
}

run_evaluation() {
    local dataset="$1"
    local scores="$2"
    local auc_output="$3"
    
    echo ""
    echo "[评估] $dataset"
    echo "  输入: $scores"
    echo "  输出: $auc_output"
    
    if [ ! -f "$scores" ]; then
        echo "  [跳过] scores 文件不存在"
        return
    fi
    
    activate_env eventvad_lavis
    
    python "$PROJECT_ROOT/src/evaluate.py" \
        --model_output "$scores" \
        --auc_output "$auc_output"
    
    echo "  [完成] AUC 结果: $auc_output"
}

# =====================================
# 主流程
# =====================================
echo ""
echo "========================================"
echo "  EventVAD Pipeline"
echo "========================================"
echo "  项目目录: $PROJECT_ROOT"
echo "  视频目录: $VIDEOS_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  GPU:      $GPUS"
echo "  模式:     $MODE"
echo "  数据集:   $DATASET"
case "$MODE" in
    batch)
        echo "  批量大小: $BATCH_SIZE"
        ;;
    pipeline)
        echo "  预取队列: $PREFETCH"
        echo "  保存线程: $SAVE_WORKERS"
        ;;
esac
echo "========================================"

# 检查 GPU
source ~/miniconda3/bin/activate eventvad_lavis 2>/dev/null || true
check_gpu

# =====================================
# Step 1: 事件分割
# =====================================
echo ""
echo "========================================"
echo "  Step 1: 事件分割"
echo "========================================"

if [ "$DATASET" = "all" ] || [ "$DATASET" = "xdviolence" ]; then
    run_segmentation "XD-Violence" \
        "$VIDEOS_DIR/xdviolence" \
        "$OUTPUT_DIR/xdviolence/segments"
fi

if [ "$DATASET" = "all" ] || [ "$DATASET" = "ucf_crime" ]; then
    run_segmentation "UCF-Crime" \
        "$VIDEOS_DIR/ucf_crime" \
        "$OUTPUT_DIR/ucf_crime/segments"
fi

# =====================================
# Step 2: 异常评分
# =====================================
echo ""
echo "========================================"
echo "  Step 2: 异常评分"
echo "========================================"

if [ "$DATASET" = "all" ] || [ "$DATASET" = "xdviolence" ]; then
    run_scoring "XD-Violence" \
        "$OUTPUT_DIR/xdviolence/segments/segment_manifest.txt" \
        "$OUTPUT_DIR/xdviolence/scores.txt"
fi

if [ "$DATASET" = "all" ] || [ "$DATASET" = "ucf_crime" ]; then
    run_scoring "UCF-Crime" \
        "$OUTPUT_DIR/ucf_crime/segments/segment_manifest.txt" \
        "$OUTPUT_DIR/ucf_crime/scores.txt"
fi

# =====================================
# Step 3: 评估
# =====================================
echo ""
echo "========================================"
echo "  Step 3: 评估"
echo "========================================"

if [ "$DATASET" = "all" ] || [ "$DATASET" = "xdviolence" ]; then
    run_evaluation "XD-Violence" \
        "$OUTPUT_DIR/xdviolence/scores.txt" \
        "$OUTPUT_DIR/xdviolence/auc.txt"
fi

if [ "$DATASET" = "all" ] || [ "$DATASET" = "ucf_crime" ]; then
    run_evaluation "UCF-Crime" \
        "$OUTPUT_DIR/ucf_crime/scores.txt" \
        "$OUTPUT_DIR/ucf_crime/auc.txt"
fi

# =====================================
# 结果汇总
# =====================================
echo ""
echo "========================================"
echo "  处理完成！"
echo "========================================"
echo ""
echo "输出目录结构："
echo "  $OUTPUT_DIR/"
echo "  ├── xdviolence/"
echo "  │   ├── segments/          # 分割的视频片段"
echo "  │   │   └── segment_manifest.txt"
echo "  │   ├── scores.txt         # 异常评分"
echo "  │   └── auc.txt            # AUC 评估结果"
echo "  └── ucf_crime/"
echo "      ├── segments/"
echo "      ├── scores.txt"
echo "      └── auc.txt"
echo ""

# 显示结果
echo "========================================"
echo "  评估结果"
echo "========================================"

if [ -f "$OUTPUT_DIR/xdviolence/auc.txt" ]; then
    echo ""
    echo "XD-Violence:"
    cat "$OUTPUT_DIR/xdviolence/auc.txt"
fi

if [ -f "$OUTPUT_DIR/ucf_crime/auc.txt" ]; then
    echo ""
    echo "UCF-Crime:"
    cat "$OUTPUT_DIR/ucf_crime/auc.txt"
fi

echo ""
echo "========================================"
echo "  使用方法"
echo "========================================"
echo ""
echo "# 快速处理（推荐，4000视频约5小时）"
echo "MODE=batch BATCH_SIZE=8 bash scripts/run_pipeline.sh"
echo ""
echo "# 双卡并行（4000视频约2.5小时）"
echo "MODE=parallel GPUS=0,1 bash scripts/run_pipeline.sh"
echo ""
echo "# 只处理 XD-Violence"
echo "DATASET=xdviolence bash scripts/run_pipeline.sh"
echo ""
echo "# 串行模式（调试用）"
echo "MODE=serial bash scripts/run_pipeline.sh"
echo ""
echo "========================================"
echo "  速度优化说明"
echo "========================================"
echo ""
echo "当前配置（高吞吐模式）:"
echo "  - 分辨率: 640x360"
echo "  - 帧采样: 1/3"
echo "  - CLIP batch: 256"
echo "  - 预处理线程: 16"
echo "  - 光流: fast（帧差近似）"
echo "  - 预估速度: ~150 帧/秒, ~2.5小时/4000视频"
echo ""
echo "如需更高精度，修改 config.py:"
echo "  - flow_mode = 'raft'"
echo "  - frame_sample_rate = 1"
echo "  - max_resolution = (1280, 720)"
echo ""
