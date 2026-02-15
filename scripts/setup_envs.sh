#!/bin/bash

set -e

echo "=== 创建并配置两个独立环境 ==="
echo "1) eventvad_lavis  (分割 + LAVIS)"
echo "2) eventvad_vllm   (评分 + VideoLLaMA2)"
echo ""

source ~/miniconda3/bin/activate

echo "[Step 1] 创建环境..."
conda create -y -n eventvad_lavis python=3.10
conda create -y -n eventvad_vllm python=3.10

echo ""
echo "[Step 2] 配置 eventvad_lavis..."
source ~/miniconda3/bin/activate eventvad_lavis

# PyTorch (CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121

# LAVIS 依赖
pip install numpy==1.26.4 pillow==10.4.0 filelock==3.20.3
pip install opencv-python==4.7.0.72 opencv-python-headless==4.5.5.64
pip install transformers==4.25.1 timm==0.4.12 tokenizers==0.13.3
pip install huggingface_hub==0.25.2 sentencepiece==0.2.0

# 安装 LAVIS (本地源码)
pip install -e /data/liuzhe/EventVAD/src/LAVIS

echo ""
echo "[Step 3] 配置 eventvad_vllm..."
source ~/miniconda3/bin/activate eventvad_vllm

# PyTorch (CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121

# VideoLLaMA2 依赖
pip install numpy==1.24.4 pillow==10.4.0 filelock==3.20.3
pip install opencv-python==4.6.0.66 opencv-python-headless==4.5.5.64
pip install transformers==4.40.0 timm==1.0.3 tokenizers==0.19.1
pip install huggingface_hub==0.23.4 sentencepiece==0.2.0

# 安装 VideoLLaMA2 (本地源码)
pip install -e /data/liuzhe/EventVAD/src/score/src/videollama2

echo ""
echo "=== 完成 ==="
echo "接下来可运行："
echo "  bash /data/liuzhe/EventVAD/scripts/run_pipeline.sh"
