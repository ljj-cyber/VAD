import numpy as np
import torch
from scipy.signal import savgol_filter
from config import Config


def detect_boundaries(G, fps):
    """
    优化的边界检测：使用向量化计算 + GPU 加速
    """
    nodes = sorted(G.nodes())
    features = np.array([G.nodes[i]['feature'] for i in nodes])
    n = len(features)
    
    if n < 2:
        return []
    
    # 使用 GPU 计算（如果可用）
    if torch.cuda.is_available():
        return _detect_boundaries_gpu(features, fps)
    else:
        return _detect_boundaries_cpu(features, fps)


def _detect_boundaries_gpu(features, fps):
    """
    GPU 加速的边界检测
    """
    device = Config.device
    features_t = torch.from_numpy(features).float().to(device)
    n = features_t.size(0)
    
    # 计算相邻帧差异
    diffs = features_t[1:] - features_t[:-1]
    s = torch.sum(diffs ** 2, dim=1)
    
    # 计算余弦相似度（向量化）
    norms = torch.norm(features_t, dim=1)
    dot_products = torch.sum(features_t[:-1] * features_t[1:], dim=1)
    cos_sim = dot_products / (norms[:-1] * norms[1:] + 1e-6)
    s_cos = 1 - cos_sim
    
    # 综合分数
    s_combined = (s + s_cos).cpu().numpy()
    
    del features_t, diffs, s, cos_sim, s_cos
    torch.cuda.empty_cache()
    
    return _postprocess_boundaries(s_combined, fps, n)


def _detect_boundaries_cpu(features, fps):
    """
    CPU 向量化版本
    """
    n = len(features)
    
    # 向量化计算差异
    diffs = features[1:] - features[:-1]
    s = np.sum(diffs ** 2, axis=1)
    
    # 向量化计算余弦相似度
    norms = np.linalg.norm(features, axis=1)
    dot_products = np.sum(features[:-1] * features[1:], axis=1)
    cos_sim = dot_products / (norms[:-1] * norms[1:] + 1e-6)
    s_cos = 1 - cos_sim
    
    s_combined = s + s_cos
    
    return _postprocess_boundaries(s_combined, fps, n)


def _postprocess_boundaries(s_combined, fps, n):
    """
    后处理：平滑、阈值检测、合并
    """
    window_size = max(3, int(fps * Config.ema_window))
    
    # 确保窗口大小是奇数
    if window_size % 2 == 0:
        window_size += 1
    
    if len(s_combined) < window_size * 2:
        return []
    
    # Savgol 平滑
    try:
    s_smoothed = savgol_filter(s_combined, window_length=window_size, polyorder=2)
    except ValueError:
        # 如果数据太短，使用简单平滑
        s_smoothed = np.convolve(s_combined, np.ones(3)/3, mode='same')
    
    # EMA 平滑
    ema = np.convolve(s_smoothed, np.ones(window_size)/window_size, mode='valid')
    
    if len(ema) == 0:
        return []
    
    # 计算比值
    s_ratio = s_smoothed[window_size-1:len(ema)+window_size-1] / (ema + 1e-6)
    
    if len(s_ratio) == 0:
        return []
    
    # MAD 阈值
    median = np.median(s_ratio)
    mad = np.median(np.abs(s_ratio - median))
    threshold = median + Config.mad_multiplier * mad
    
    # 检测边界
    boundaries = np.where(s_ratio > threshold)[0] + window_size // 2
    
    if len(boundaries) == 0:
        return []
    
    # 合并相近边界
    merged = []
    prev = boundaries[0]
    min_gap = int(Config.min_segment_gap * fps)
    
    for b in boundaries[1:]:
        if b - prev < min_gap:
            prev = b  # 保留后一个
        else:
            merged.append(prev)
            prev = b
    merged.append(prev)
    
    # 生成时间边界
    time_boundaries = []
    total_time = n / fps
    
    for i in range(len(merged)):
        start = merged[i] / fps
        if i + 1 < len(merged):
            end = merged[i + 1] / fps
        else:
            end = min(merged[i] / fps + Config.min_segment_gap, total_time)
        
        start = max(0, start)
        end = min(end, total_time)
        
        if end > start:
            time_boundaries.append((start, end))
    
    return time_boundaries
