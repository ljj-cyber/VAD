import torch

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 启用 FP16 混合精度加速
    fp16_enabled = True
    
    # 特征提取
    clip_model_name = "clip"
    feature_dim = 640
    
    # 批量处理大小（48GB GPU 优化）
    clip_batch_size = 128     # CLIP 批量大小（增大到 128）
    flow_batch_size = 8       # 光流批量大小上限
    
    # RAFT 设置
    raft_small = True         # 使用 RAFT-Small（更快）
    raft_iters = 6            # 减少迭代次数
    
    # 动态图参数
    time_decay = 0.05
    clip_weight = 0.8
    
    # 边界检测参数
    ema_window = 2.0
    
    # 系统
    max_resolution = (640, 360)  # 降低分辨率加速
    chunk_size = 500
    graph_block_size = 500
    ortho_dim = 64
    gat_iters = 1
    mad_multiplier = 3.0
    min_segment_gap = 2.0
    
    # 多线程
    num_workers = 16          # 增大线程数
    
    # 显存管理
    gpu_memory_reserve = 0.3  # 预留 30% 显存
    
    # 光流模式: "fast", "sparse", "raft"
    flow_mode = "fast"        # 帧差近似（最快）
    flow_sample_rate = 5      # sparse 模式采样率
    
    # 帧采样率: 1 表示不采样，N 表示每 N 帧取 1 帧
    frame_sample_rate = 3
    
    # 多 GPU
    use_multi_gpu = True      # 启用多 GPU CLIP
