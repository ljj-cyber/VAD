import sys
import pathlib
import gc
import os

# 动态获取路径
_current_dir = pathlib.Path(__file__).parent.resolve()
_src_dir = _current_dir.parent.resolve()
_raft_core_dir = _src_dir / "RAFT" / "core"

if str(_raft_core_dir) not in sys.path:
    sys.path.insert(0, str(_raft_core_dir))
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from argparse import Namespace
from config import Config
from lavis.models import load_model_and_preprocess

# RAFT 导入
from raft import RAFT

# 启用性能优化
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends, 'cuda'):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_gpu_memory_info(device_id=0):
    """获取指定 GPU 显存信息"""
    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        with torch.cuda.device(device_id):
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            free = total - reserved
            return allocated, reserved, total, free
    return 0, 0, 0, 0


def get_available_gpus():
    """获取可用 GPU 列表"""
    if not torch.cuda.is_available():
        return []
    
    # 检查环境变量
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible:
        return list(range(len(visible.split(','))))
    
    return list(range(torch.cuda.device_count()))


class MultiGPUClipExtractor:
    """
    多 GPU CLIP 特征提取器
    将帧分配到多个 GPU 并行处理
    """
    
    def __init__(self, gpu_ids=None):
        """
        初始化多 GPU 提取器
        Args:
            gpu_ids: GPU ID 列表，如 [0, 1]
        """
        if gpu_ids is None:
            gpu_ids = get_available_gpus()
        
        self.gpu_ids = gpu_ids if len(gpu_ids) > 0 else [0]
        self.n_gpus = len(self.gpu_ids)
        
        print(f"初始化多 GPU CLIP 提取器 (GPUs: {self.gpu_ids})...")
        
        # 在每个 GPU 上加载模型
        self.models = []
        self.processors = None
        
        for i, gpu_id in enumerate(self.gpu_ids):
            device = f"cuda:{gpu_id}"
            model_info = load_model_and_preprocess(
                name=Config.clip_model_name,
                model_type="ViT-B-16",
                is_eval=True,
                device=device
            )
            self.models.append(model_info[0].float())
            if self.processors is None:
                self.processors = model_info[1]
            print(f"  GPU {gpu_id}: CLIP 模型加载完成")
        
        print(f"多 GPU CLIP 初始化完成 ({self.n_gpus} 卡)")
    
    @torch.no_grad()
    def extract(self, frames, batch_size=None):
        """
        多 GPU 并行提取 CLIP 特征
        """
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        if batch_size is None:
            batch_size = Config.clip_batch_size
        
        n_frames = len(frames)
        
        if isinstance(frames, np.ndarray):
            frames = [frames[i] for i in range(len(frames))]
        
        # 多线程预处理
        def preprocess_frame(frame):
            return self.processors["eval"](Image.fromarray(frame))
        
        with ThreadPoolExecutor(max_workers=Config.num_workers) as executor:
            all_processed = list(executor.map(preprocess_frame, frames))
        
        if self.n_gpus == 1:
            # 单卡模式
            return self._extract_single_gpu(all_processed, batch_size, 0)
        
        # 多卡并行
        # 将帧分配到各 GPU
        frames_per_gpu = (n_frames + self.n_gpus - 1) // self.n_gpus
        
        results = [None] * self.n_gpus
        threads = []
        
        def extract_on_gpu(gpu_idx, start_idx, end_idx):
            chunk = all_processed[start_idx:end_idx]
            if len(chunk) > 0:
                results[gpu_idx] = self._extract_single_gpu(chunk, batch_size, gpu_idx)
        
        for gpu_idx in range(self.n_gpus):
            start_idx = gpu_idx * frames_per_gpu
            end_idx = min(start_idx + frames_per_gpu, n_frames)
            
            t = threading.Thread(target=extract_on_gpu, args=(gpu_idx, start_idx, end_idx))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        # 合并结果
        valid_results = [r for r in results if r is not None]
        
        del all_processed
        gc.collect()
        
        return np.concatenate(valid_results) if valid_results else np.array([])
    
    def _extract_single_gpu(self, processed_frames, batch_size, gpu_idx):
        """单 GPU 提取"""
        model = self.models[gpu_idx]
        device = f"cuda:{self.gpu_ids[gpu_idx]}"
        
        n_frames = len(processed_frames)
        features = []
        
        # 预先 stack 批次
        batches = []
        for i in range(0, n_frames, batch_size):
            chunk = processed_frames[i:i+batch_size]
            batches.append(torch.stack(chunk))
        
        # GPU 推理
        with torch.no_grad():
            for batch_tensor in batches:
                batch_gpu = batch_tensor.to(device, non_blocking=True).float()
                chunk_feats = model.encode_image(batch_gpu)
                features.append(chunk_feats.detach().cpu().numpy())
                del batch_gpu, chunk_feats
        
        torch.cuda.empty_cache()
        return np.concatenate(features)


class FeatureExtractor:  
    """
    特征提取器（支持单卡/多卡）
    """
    
    def __init__(self, use_raft=True, gpu_ids=None):
        """
        初始化特征提取器
        Args:
            use_raft: 是否使用 RAFT
            gpu_ids: GPU ID 列表
        """
        print("初始化特征提取器...")
        
        # 获取 GPU 配置
        if gpu_ids is None:
            gpu_ids = get_available_gpus()
        
        self.gpu_ids = gpu_ids if len(gpu_ids) > 0 else [0]
        self.primary_device = f"cuda:{self.gpu_ids[0]}"
        
        # 多 GPU CLIP
        if len(self.gpu_ids) > 1:
            self.clip_extractor = MultiGPUClipExtractor(self.gpu_ids)
            self.clip_model = None
            self.vis_processors = self.clip_extractor.processors
        else:
            # 单卡模式
            self.clip_extractor = None
        model_info = load_model_and_preprocess(
            name=Config.clip_model_name,
            model_type="ViT-B-16",
            is_eval=True,
                device=self.primary_device
        )
        self.clip_model = model_info[0].float()
        self.vis_processors = model_info[1]
        
        self.use_raft = use_raft
        self.raft_model = None
        
        if use_raft:
        raft_args = Namespace(
            model='/path/raft-things.pth',
                small=Config.raft_small,
                mixed_precision=True,
            alternate_corr=False,
            dropout=0.0
        )
            self.raft_model = RAFT(raft_args).to(self.primary_device).eval()
            
            if Config.fp16_enabled:
                self.raft_model = self.raft_model.half()
        
        self.flow_proj = self._init_random_ortho(2, 128)
        
        mode = "多GPU" if len(self.gpu_ids) > 1 else "单GPU"
        print(f"特征提取器初始化完成 ({mode}: {self.gpu_ids}, RAFT: {use_raft})")

    def _init_random_ortho(self, in_dim, out_dim):
        np.random.seed(42)
        q, _ = np.linalg.qr(np.random.randn(max(in_dim, out_dim), max(in_dim, out_dim)))
        return q[:in_dim, :out_dim].astype(np.float32)

    @torch.no_grad()
    def extract_clip_features(self, frames, batch_size=None):
        """提取 CLIP 特征"""
        if batch_size is None:
            batch_size = Config.clip_batch_size
        
        # 动态调整 batch_size 防止爆显存
        batch_size = self._adjust_batch_size(frames, batch_size)
        
        if self.clip_extractor is not None:
            # 多 GPU
            return self.clip_extractor.extract(frames, batch_size)
        
        # 单 GPU
        from concurrent.futures import ThreadPoolExecutor
        
        n_frames = len(frames)
        
        if isinstance(frames, np.ndarray):
            frames = [frames[i] for i in range(len(frames))]
        
        def preprocess_frame(frame):
            return self.vis_processors["eval"](Image.fromarray(frame))
        
        with ThreadPoolExecutor(max_workers=Config.num_workers) as executor:
            all_processed = list(executor.map(preprocess_frame, frames))
        
        # 分批 stack 和推理
        features = []
        for i in tqdm(range(0, n_frames, batch_size), desc="CLIP特征", leave=False):
            chunk = all_processed[i:i+batch_size]
            batch_tensor = torch.stack(chunk).to(self.primary_device, non_blocking=True).float()
            chunk_feats = self.clip_model.encode_image(batch_tensor)
            features.append(chunk_feats.detach().cpu().numpy())
            del batch_tensor, chunk_feats
            
            # 定期清理显存
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
        
        del all_processed
            torch.cuda.empty_cache()
        return np.concatenate(features)

    def _adjust_batch_size(self, frames, batch_size):
        """根据显存动态调整 batch size"""
        n_frames = len(frames)
        h, w = frames[0].shape[:2] if n_frames > 0 else (360, 640)
        
        # 估算单帧显存（MB）
        frame_memory_mb = (h * w * 3 * 4) / 1024 / 1024  # float32
        
        # 获取可用显存
        _, _, total, free = get_gpu_memory_info(self.gpu_ids[0])
        available_mb = free * 1024 * 0.7  # 使用 70%
        
        # 估算安全的 batch size
        # CLIP ViT 需要额外 ~50MB/帧 中间变量
        per_frame_total = frame_memory_mb + 50
        safe_batch = max(1, int(available_mb / per_frame_total))
        
        adjusted = min(batch_size, safe_batch, n_frames)
        
        if adjusted < batch_size:
            print(f"  [显存保护] batch_size: {batch_size} -> {adjusted}")
        
        return adjusted

    @torch.no_grad()
    def extract_flow_features_fast(self, frames):
        """快速光流近似（帧差法）"""
        n_frames = len(frames)
        if n_frames < 2:
            return np.zeros((n_frames, 128), dtype=np.float32)
        
        if isinstance(frames, list):
            frames = np.array(frames)
        
        # 分块处理防止爆显存
        chunk_size = 500  # 每次处理 500 帧
        all_flow_features = []
        
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            chunk = frames[start:end]
            
            frames_tensor = torch.from_numpy(chunk).float().to(self.primary_device)
            
            # RGB 转灰度
            gray = 0.299 * frames_tensor[:, :, :, 0] + \
                   0.587 * frames_tensor[:, :, :, 1] + \
                   0.114 * frames_tensor[:, :, :, 2]
            
            if len(gray) < 2:
                all_flow_features.append(np.zeros((len(gray), 2), dtype=np.float32))
            else:
                diff = gray[1:] - gray[:-1]
                
                # 水平和垂直梯度
                dx = torch.mean(diff[:, :, 1:] - diff[:, :, :-1], dim=[1, 2])
                dy = torch.mean(diff[:, 1:, :] - diff[:, :-1, :], dim=[1, 2])
                
                flow_chunk = torch.stack([dx, dy], dim=1).cpu().numpy()
                
                # 第一帧补零
                if start == 0:
                    flow_chunk = np.vstack([np.zeros((1, 2)), flow_chunk])
                else:
                    # 计算与前一块最后一帧的差异
                    prev_gray = 0.299 * frames[start-1][:, :, 0] + \
                               0.587 * frames[start-1][:, :, 1] + \
                               0.114 * frames[start-1][:, :, 2]
                    curr_gray = gray[0].cpu().numpy()
                    diff0 = curr_gray - prev_gray
                    dx0 = np.mean(diff0[:, 1:] - diff0[:, :-1])
                    dy0 = np.mean(diff0[1:, :] - diff0[:-1, :])
                    flow_chunk = np.vstack([[dx0, dy0], flow_chunk])
                
                all_flow_features.append(flow_chunk)
            
            del frames_tensor, gray
            torch.cuda.empty_cache()
        
        flow_features = np.concatenate(all_flow_features, axis=0)
        
        # 投影到 128 维
        projected = np.dot(flow_features, self.flow_proj)
        return projected.astype(np.float32)

    @torch.no_grad()
    def extract_flow_features_sparse(self, frames, sample_rate=None):
        """稀疏采样光流"""
        if sample_rate is None:
            sample_rate = Config.flow_sample_rate
        
        n_frames = len(frames)
        if n_frames < 2:
            return np.zeros((n_frames, 128), dtype=np.float32)
        
        if not self.use_raft or self.raft_model is None:
            return self.extract_flow_features_fast(frames)
        
        key_indices = list(range(0, n_frames - 1, sample_rate))
        if key_indices[-1] != n_frames - 2:
            key_indices.append(n_frames - 2)
        
        n_keys = len(key_indices)
        batch_size = min(Config.flow_batch_size, n_keys)
        
        key_features = []
        
        for i in tqdm(range(0, n_keys, batch_size), desc=f"光流(1/{sample_rate}采样)", leave=False):
            batch_indices = key_indices[i:i+batch_size]
            
            prev_frames = np.stack([frames[idx] for idx in batch_indices])
            curr_frames = np.stack([frames[idx + 1] for idx in batch_indices])
            
            prev_batch = torch.from_numpy(prev_frames).permute(0, 3, 1, 2).float().to(self.primary_device)
            curr_batch = torch.from_numpy(curr_frames).permute(0, 3, 1, 2).float().to(self.primary_device)
            
            if Config.fp16_enabled:
                prev_batch = prev_batch.half()
                curr_batch = curr_batch.half()
            
            with torch.cuda.amp.autocast(enabled=Config.fp16_enabled):
                flow = self.raft_model(prev_batch, curr_batch, iters=Config.raft_iters)[-1]
            
            pooled = torch.mean(flow, dim=[2, 3]).cpu().numpy()
            key_features.append(pooled)
            
            del prev_batch, curr_batch, flow
        
        torch.cuda.empty_cache()
        key_features = np.concatenate(key_features, axis=0)
        
        # 插值
        all_features = np.zeros((n_frames - 1, 2), dtype=np.float32)
        
        for i, (start_idx, end_idx) in enumerate(zip(key_indices[:-1], key_indices[1:])):
            start_feat = key_features[i]
            end_feat = key_features[i + 1]
            
            for j in range(start_idx, end_idx):
                alpha = (j - start_idx) / (end_idx - start_idx)
                all_features[j] = (1 - alpha) * start_feat + alpha * end_feat
        
        all_features[key_indices[-1]:] = key_features[-1]
        all_features = np.vstack([np.zeros((1, 2)), all_features])
        
        projected = np.dot(all_features, self.flow_proj)
        return projected.astype(np.float32)

    @torch.no_grad()
    def extract_flow_features_raft(self, frames, batch_size=None):
        """完整 RAFT 光流"""
        if batch_size is None:
            batch_size = Config.flow_batch_size
        
        n_frames = len(frames)
        if n_frames < 2:
            return np.zeros((n_frames, 128), dtype=np.float32)
        
        if not self.use_raft or self.raft_model is None:
            return self.extract_flow_features_fast(frames)
        
        n_pairs = n_frames - 1
        flow_features = []
        
        n_full_batches = n_pairs // batch_size
        remainder = n_pairs % batch_size
        
        for batch_idx in tqdm(range(n_full_batches), desc="光流(RAFT)", leave=False):
            start = batch_idx * batch_size
            end = start + batch_size
            
            prev_frames = frames[start:end]
            curr_frames = frames[start+1:end+1]
            
            prev_batch = torch.from_numpy(np.stack(prev_frames)).permute(0, 3, 1, 2).float().to(self.primary_device)
            curr_batch = torch.from_numpy(np.stack(curr_frames)).permute(0, 3, 1, 2).float().to(self.primary_device)
            
            if Config.fp16_enabled:
                prev_batch = prev_batch.half()
                curr_batch = curr_batch.half()
            
            with torch.cuda.amp.autocast(enabled=Config.fp16_enabled):
                flow = self.raft_model(prev_batch, curr_batch, iters=Config.raft_iters)[-1]
            
            pooled = torch.mean(flow, dim=[2, 3]).cpu().numpy()
            flow_features.append(pooled)
            
            del prev_batch, curr_batch, flow
            
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        if remainder > 0:
            start = n_full_batches * batch_size
            
            prev_frames = frames[start:start+remainder]
            curr_frames = frames[start+1:start+1+remainder]
            
            prev_batch = torch.from_numpy(np.stack(prev_frames)).permute(0, 3, 1, 2).float().to(self.primary_device)
            curr_batch = torch.from_numpy(np.stack(curr_frames)).permute(0, 3, 1, 2).float().to(self.primary_device)
            
            if Config.fp16_enabled:
                prev_batch = prev_batch.half()
                curr_batch = curr_batch.half()
            
            if remainder < batch_size and batch_size > 1:
                pad_size = batch_size - remainder
                prev_batch = torch.cat([prev_batch, prev_batch[-1:].repeat(pad_size, 1, 1, 1)], dim=0)
                curr_batch = torch.cat([curr_batch, curr_batch[-1:].repeat(pad_size, 1, 1, 1)], dim=0)
            
            with torch.cuda.amp.autocast(enabled=Config.fp16_enabled):
                flow = self.raft_model(prev_batch, curr_batch, iters=Config.raft_iters)[-1]
            
            pooled = torch.mean(flow, dim=[2, 3]).cpu().numpy()[:remainder]
            flow_features.append(pooled)
            
            del prev_batch, curr_batch, flow
        
        torch.cuda.empty_cache()
        
        flow_features = np.concatenate(flow_features, axis=0)
        flow_features = np.vstack([np.zeros((1, 2)), flow_features])
        
        projected = np.dot(flow_features, self.flow_proj)
        return projected.astype(np.float32)

    def extract_features(self, frames):
        """提取 CLIP 和光流特征"""
        n_frames = len(frames)
        h, w = frames[0].shape[:2] if n_frames > 0 else (0, 0)
        
        allocated, _, total, free = get_gpu_memory_info(self.gpu_ids[0])
        print(f"处理 {n_frames} 帧 ({w}x{h}), GPU: {allocated:.1f}/{total:.1f}GB (空闲 {free:.1f}GB)")
        
        # CLIP 特征
        clip_feats = self.extract_clip_features(frames)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # 光流特征
        flow_mode = Config.flow_mode
        
        try:
            if flow_mode == "fast":
                flow_feats = self.extract_flow_features_fast(frames)
            elif flow_mode == "sparse":
                flow_feats = self.extract_flow_features_sparse(frames)
            elif flow_mode == "raft":
                flow_feats = self.extract_flow_features_raft(frames)
            else:
                flow_feats = self.extract_flow_features_fast(frames)
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] 回退到快速方法...")
                torch.cuda.empty_cache()
                gc.collect()
                flow_feats = self.extract_flow_features_fast(frames)
            else:
                raise e
        
        assert clip_feats.shape[0] == flow_feats.shape[0], \
            f"特征维度不匹配: CLIP {clip_feats.shape[0]} vs Flow {flow_feats.shape[0]}"
        
        return np.concatenate([
            Config.clip_weight * clip_feats,
            (1-Config.clip_weight) * flow_feats
        ], axis=1).astype(np.float32)
