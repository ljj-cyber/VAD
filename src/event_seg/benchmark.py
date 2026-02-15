#!/usr/bin/env python3
"""
性能基准测试脚本
"""
import sys
import pathlib
import time
import numpy as np
import torch

_current_dir = pathlib.Path(__file__).parent.resolve()
_src_dir = _current_dir.parent.resolve()

sys.path.insert(0, str(_current_dir))
sys.path.insert(0, str(_src_dir))
sys.path.insert(0, str(_src_dir / "LAVIS"))
sys.path.insert(0, str(_src_dir / "RAFT" / "core"))

from config import Config

def benchmark_frame_processing():
    """测试帧处理速度"""
    from video_utils import video_to_frames
    import cv2
    
    print("\n" + "="*50)
    print("帧处理性能测试")
    print("="*50)
    
    # 生成测试数据（模拟 1000 帧视频）
    n_frames = 1000
    h, w = 720, 1280
    
    print(f"测试数据: {n_frames} 帧, {w}x{h}")
    
    # 测试多线程 resize
    frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(100)]
    
    from concurrent.futures import ThreadPoolExecutor
    
    def process_frame(frame):
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 360))
    
    # 单线程
    start = time.time()
    for f in frames:
        process_frame(f)
    single_time = time.time() - start
    
    # 多线程
    start = time.time()
    with ThreadPoolExecutor(max_workers=Config.num_workers) as executor:
        list(executor.map(process_frame, frames))
    multi_time = time.time() - start
    
    print(f"单线程: {single_time:.2f}s")
    print(f"多线程 ({Config.num_workers} workers): {multi_time:.2f}s")
    print(f"加速比: {single_time/multi_time:.2f}x")


def benchmark_gpu_operations():
    """测试 GPU 计算速度"""
    print("\n" + "="*50)
    print("GPU 矩阵运算性能测试")
    print("="*50)
    
    n = 5000  # 模拟 5000 帧
    feat_dim = 512
    
    # 生成测试数据
    features_np = np.random.randn(n, feat_dim).astype(np.float32)
    
    # CPU numpy
    start = time.time()
    for _ in range(3):
        sim_cpu = np.dot(features_np, features_np.T)
    cpu_time = (time.time() - start) / 3
    
    # GPU torch
    features_gpu = torch.from_numpy(features_np).cuda()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(3):
        sim_gpu = torch.mm(features_gpu, features_gpu.t())
        torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 3
    
    print(f"矩阵大小: {n}x{feat_dim}")
    print(f"CPU (numpy): {cpu_time*1000:.1f}ms")
    print(f"GPU (torch): {gpu_time*1000:.1f}ms")
    print(f"加速比: {cpu_time/gpu_time:.1f}x")
    
    del features_gpu, sim_gpu
    torch.cuda.empty_cache()


def benchmark_flow_batching():
    """测试光流批量 vs 逐帧处理"""
    print("\n" + "="*50)
    print("光流特征提取模拟测试")
    print("="*50)
    
    n_frames = 100
    h, w = 360, 640
    
    # 模拟帧数据
    frames = torch.randn(n_frames, 3, h, w, device='cuda')
    
    # 模拟逐帧处理
    start = time.time()
    for i in range(n_frames - 1):
        prev = frames[i:i+1]
        curr = frames[i+1:i+2]
        # 模拟一些计算
        diff = (curr - prev).mean()
    torch.cuda.synchronize()
    single_time = time.time() - start
    
    # 模拟批量处理
    batch_size = Config.flow_batch_size
    start = time.time()
    for i in range(0, n_frames - 1, batch_size):
        end = min(i + batch_size, n_frames - 1)
        prev_batch = frames[i:end]
        curr_batch = frames[i+1:end+1]
        diff = (curr_batch - prev_batch).mean()
    torch.cuda.synchronize()
    batch_time = time.time() - start
    
    print(f"帧数: {n_frames}")
    print(f"逐帧处理: {single_time*1000:.1f}ms")
    print(f"批量处理 (batch={batch_size}): {batch_time*1000:.1f}ms")
    print(f"加速比: {single_time/batch_time:.1f}x")
    
    del frames
    torch.cuda.empty_cache()


def benchmark_graph_building():
    """测试图构建 CPU vs GPU"""
    print("\n" + "="*50)
    print("动态图构建性能测试")
    print("="*50)
    
    n = 2000  # 模拟 2000 帧
    feat_dim = 640
    
    features = np.random.randn(n, feat_dim).astype(np.float32)
    clip_feats = features[:, :512]
    flow_feats = features[:, 512:]
    
    block_size = 500
    
    # CPU 版本
    start = time.time()
    clip_norms = np.linalg.norm(clip_feats, axis=1, keepdims=True)
    clip_feats_norm = clip_feats / (clip_norms + 1e-6)
    
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            sim = np.dot(clip_feats_norm[i:i_end], clip_feats_norm[j:j_end].T)
    cpu_time = time.time() - start
    
    # GPU 版本
    clip_feats_t = torch.from_numpy(clip_feats).cuda()
    clip_norms_t = torch.norm(clip_feats_t, dim=1, keepdim=True)
    clip_feats_t = clip_feats_t / (clip_norms_t + 1e-6)
    torch.cuda.synchronize()
    
    start = time.time()
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            sim = torch.mm(clip_feats_t[i:i_end], clip_feats_t[j:j_end].t())
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"节点数: {n}")
    print(f"CPU (numpy): {cpu_time*1000:.1f}ms")
    print(f"GPU (torch): {gpu_time*1000:.1f}ms")
    print(f"加速比: {cpu_time/gpu_time:.1f}x")
    
    del clip_feats_t
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("="*50)
    print("EventVAD 性能基准测试")
    print("="*50)
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"FP16 启用: {Config.fp16_enabled}")
    print(f"CLIP 批量大小: {Config.clip_batch_size}")
    print(f"光流批量大小: {Config.flow_batch_size}")
    
    benchmark_frame_processing()
    benchmark_gpu_operations()
    benchmark_flow_batching()
    benchmark_graph_building()
    
    print("\n" + "="*50)
    print("基准测试完成！")
    print("="*50)
