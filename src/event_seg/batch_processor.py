"""
高吞吐批量处理器
同时处理多个视频，最大化 GPU 利用率
"""
import os
import sys
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc

from config import Config
from video_utils import video_to_frames
from feature_extractor import FeatureExtractor
from uniseg_processor import UniSegProcessor
from video_processing import _save_segments


class BatchVideoProcessor:
    """
    批量视频处理器：
    1. 并行读取多个视频到内存
    2. 合并所有帧进行批量 CLIP 推理
    3. 并行保存结果
    """
    
    def __init__(self):
        print("初始化批量处理器...")
        self.extractor = FeatureExtractor(use_raft=False)  # fast 模式不需要 RAFT
        print(f"批量处理器就绪 (CLIP batch={Config.clip_batch_size})")
    
    def process_video_batch(self, video_infos):
        """
        批量处理多个视频
        
        Args:
            video_infos: [(video_path, output_dir), ...]
        
        Returns:
            results: [(video_path, output_dir, boundaries), ...]
        """
        if not video_infos:
            return []
        
        n_videos = len(video_infos)
        print(f"\n处理 {n_videos} 个视频...")
        
        # Step 1: 并行读取所有视频帧
        print("  [1/4] 读取视频帧...")
        video_data = self._parallel_load_videos(video_infos)
        
        valid_videos = [(vp, od, frames, fps, size) 
                        for vp, od, frames, fps, size in video_data 
                        if frames is not None and len(frames) > 0]
        
        if not valid_videos:
            return []
        
        # Step 2: 合并帧，批量提取 CLIP 特征
        print("  [2/4] 批量提取 CLIP 特征...")
        all_frames, frame_indices = self._merge_frames(valid_videos)
        
        if len(all_frames) == 0:
            return []
        
        clip_features = self._batch_clip_extract(all_frames)
        
        # Step 3: 分离特征，处理每个视频
        print("  [3/4] 处理分割...")
        results = []
        
        for i, (video_path, output_dir, frames, fps, new_size) in enumerate(valid_videos):
            start_idx, end_idx = frame_indices[i]
            video_clip_feats = clip_features[start_idx:end_idx]
            
            # 快速光流
            flow_feats = self.extractor.extract_flow_features_fast(frames)
            
            # 合并特征
            features = np.concatenate([
                Config.clip_weight * video_clip_feats,
                (1 - Config.clip_weight) * flow_feats
            ], axis=1).astype(np.float32)
            
            # 构建图 + 边界检测
            boundaries = self._process_single_video(features, fps)
            
            results.append((video_path, output_dir, frames, fps, new_size, boundaries))
        
        del all_frames, clip_features
        torch.cuda.empty_cache()
        gc.collect()
        
        # Step 4: 并行保存
        print("  [4/4] 保存分段...")
        final_results = self._parallel_save(results)
        
        return final_results
    
    def _parallel_load_videos(self, video_infos):
        """并行加载视频"""
        def load_one(info):
            video_path, output_dir = info
            try:
                frames, fps, size = video_to_frames(video_path)
                return (video_path, output_dir, frames, fps, size)
            except Exception as e:
                print(f"    读取失败: {os.path.basename(video_path)} - {e}")
                return (video_path, output_dir, None, None, None)
        
        with ThreadPoolExecutor(max_workers=Config.num_workers) as executor:
            results = list(executor.map(load_one, video_infos))
        
        return results
    
    def _merge_frames(self, valid_videos):
        """合并所有视频的帧"""
        all_frames = []
        frame_indices = []
        current_idx = 0
        
        for video_path, output_dir, frames, fps, size in valid_videos:
            n_frames = len(frames)
            all_frames.extend(frames)
            frame_indices.append((current_idx, current_idx + n_frames))
            current_idx += n_frames
        
        return all_frames, frame_indices
    
    @torch.no_grad()
    def _batch_clip_extract(self, all_frames):
        """批量提取 CLIP 特征"""
        from PIL import Image
        
        n_frames = len(all_frames)
        batch_size = Config.clip_batch_size
        features = []
        
        for i in tqdm(range(0, n_frames, batch_size), desc="    CLIP", leave=False):
            chunk = all_frames[i:i+batch_size]
            
            processed = torch.stack([
                self.extractor.vis_processors["eval"](Image.fromarray(f)) 
                for f in chunk
            ]).to(Config.device).float()
            
            chunk_feats = self.extractor.clip_model.encode_image(processed)
            features.append(chunk_feats.cpu().numpy())
            
            del processed, chunk_feats
        
        torch.cuda.empty_cache()
        return np.concatenate(features)
    
    def _process_single_video(self, features, fps):
        """处理单个视频的图构建和边界检测"""
        from graph_operations import graph_propagation
        from boundary_detection import detect_boundaries
        import networkx as nx
        
        n = len(features)
        if n < 10:
            return []
        
        # 简化的图构建（快速版本）
        G = self._build_graph_fast(features)
        G = graph_propagation(G)
        boundaries = detect_boundaries(G, fps)
        
        return boundaries
    
    def _build_graph_fast(self, features):
        """快速图构建"""
        import networkx as nx
        
        n = len(features)
        G = nx.Graph()
        
        # 添加节点
        for i in range(n):
            G.add_node(i, feature=features[i])
        
        # 只连接相邻帧和局部窗口（大幅加速）
        window = 10  # 只看前后10帧
        
        # 归一化特征
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features_norm = features / (norms + 1e-6)
        
        for i in range(n):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            
            # 计算相似度
            sims = np.dot(features_norm[i], features_norm[start:end].T)
            
            # 添加 top-k 边
            k = min(5, end - start - 1)
            if k > 0:
                top_k = np.argsort(-sims)[:k+1]  # +1 因为包含自己
                for j in top_k:
                    global_j = start + j
                    if global_j != i and sims[j] > 0:
                        G.add_edge(i, global_j, weight=float(sims[j]))
        
        return G
    
    def _parallel_save(self, results):
        """并行保存结果"""
        def save_one(item):
            video_path, output_dir, frames, fps, new_size, boundaries = item
            try:
                os.makedirs(output_dir, exist_ok=True)
                frame_boundaries = _save_segments(
                    frames, fps, new_size, boundaries, len(frames), output_dir
                )
                return (video_path, output_dir, frame_boundaries, None)
            except Exception as e:
                return (video_path, output_dir, None, str(e))
        
        with ThreadPoolExecutor(max_workers=Config.num_workers) as executor:
            final_results = list(executor.map(save_one, results))
        
        return final_results


def process_videos_batch(video_list, output_base, input_base, batch_size=None):
    """
    批量处理视频列表
    
    Args:
        video_list: [(video_path, output_dir), ...]
        output_base: 输出根目录
        input_base: 输入根目录
        batch_size: 每批处理的视频数
    """
    if batch_size is None:
        batch_size = Config.video_batch_size
    
    processor = BatchVideoProcessor()
    
    all_results = []
    n_videos = len(video_list)
    
    for batch_start in range(0, n_videos, batch_size):
        batch_end = min(batch_start + batch_size, n_videos)
        batch = video_list[batch_start:batch_end]
        
        print(f"\n{'='*50}")
        print(f"批次 {batch_start//batch_size + 1}/{(n_videos + batch_size - 1)//batch_size}")
        print(f"{'='*50}")
        
        results = processor.process_video_batch(batch)
        all_results.extend(results)
        
        # 打印进度
        success = sum(1 for r in results if r[2] is not None)
        print(f"  完成: {success}/{len(batch)}")
        
        # 清理
        torch.cuda.empty_cache()
        gc.collect()
    
    return all_results
