import numpy as np
import torch
import networkx as nx
import math
from tqdm import tqdm
from config import Config
from feature_extractor import FeatureExtractor
from graph_operations import graph_propagation
from boundary_detection import detect_boundaries

# 启用 GPU 加速
torch.backends.cudnn.benchmark = True


class UniSegProcessor:
    def __init__(self, use_raft=None):
        # 根据配置决定是否使用 RAFT
        if use_raft is None:
            use_raft = Config.flow_mode in ["raft", "sparse"]
        self.extractor = FeatureExtractor(use_raft=use_raft)
        
    def build_dynamic_graph_gpu(self, features, fps):
        """
        使用 GPU 加速的动态图构建
        """
        n = len(features)
        device = Config.device
        
        print(f"构建动态图 ({n} 节点)...")
        
        # 转到 GPU
        features_t = torch.from_numpy(features).float().to(device)
        
        clip_feats = features_t[:, :512]
        flow_feats = features_t[:, 512:]
        
        # 归一化 CLIP 特征
        clip_norms = torch.norm(clip_feats, dim=1, keepdim=True)
        clip_feats = clip_feats / (clip_norms + 1e-6)
        
        # 分块计算相似度矩阵（避免显存溢出）
        block_size = min(Config.graph_block_size, n)
        init_k = 5
        
        # 收集边
        edges = []
        
        total_blocks = math.ceil(n / block_size) ** 2
        with tqdm(total=total_blocks, desc="构建动态图") as pbar:
            for i in range(0, n, block_size):
                i_end = min(i + block_size, n)
                clip_block_i = clip_feats[i:i_end]
                flow_block_i = flow_feats[i:i_end]
                
                for j in range(0, n, block_size):
                    j_end = min(j + block_size, n)
                    clip_block_j = clip_feats[j:j_end]
                    flow_block_j = flow_feats[j:j_end]
                    
                    # GPU 矩阵乘法计算 CLIP 相似度
                    clip_sim = torch.mm(clip_block_i, clip_block_j.t())
                    
                    # GPU 计算光流距离
                    flow_diff = flow_block_i.unsqueeze(1) - flow_block_j.unsqueeze(0)
                    flow_dist = torch.sqrt(torch.sum(flow_diff ** 2, dim=2))
                    
                    # 时间惩罚
                    time_i = torch.arange(i, i_end, device=device).unsqueeze(1)
                    time_j = torch.arange(j, j_end, device=device).unsqueeze(0)
                    time_diff = torch.abs(time_i - time_j).float()
                    time_penalty = 1 + Config.time_decay * time_diff
                    
                    # 综合相似度
                    combined_sim = (Config.clip_weight * clip_sim + 
                                   (1 - Config.clip_weight) * torch.exp(-flow_dist)) / time_penalty
                    
                    # 提取 top-k 边
                    block_size_i = i_end - i
                    block_size_j = j_end - j
                    dynamic_k = max(3, init_k - (i // (max(n // 10, 1))))
                    valid_k = min(dynamic_k, block_size_j)
                    
                    if valid_k > 0:
                        # GPU 上选择 top-k
                        topk_vals, topk_idx = torch.topk(combined_sim, k=valid_k, dim=1)
                        
                        # 转到 CPU 构建边列表
                        topk_idx_cpu = topk_idx.cpu().numpy()
                        topk_vals_cpu = topk_vals.cpu().numpy()
                        
                        for local_i in range(block_size_i):
                            global_i = i + local_i
                            for k_idx in range(valid_k):
                                local_j = topk_idx_cpu[local_i, k_idx]
                                global_j = j + local_j
                                weight = topk_vals_cpu[local_i, k_idx]
                                if global_i != global_j and weight > 0:
                                    edges.append((global_i, global_j, weight))
                    
                    pbar.update(1)
        
        # 清理 GPU 内存
        del features_t, clip_feats, flow_feats
        torch.cuda.empty_cache()
        
        # 构建 NetworkX 图
        G = nx.Graph()
        for i in range(n):
            G.add_node(i, feature=features[i])
        
        for u, v, w in edges:
            if G.has_edge(u, v):
                # 取较大的权重
                if G[u][v]['weight'] < w:
                    G[u][v]['weight'] = w
            else:
                G.add_edge(u, v, weight=w)
        
        print(f"动态图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G
        
    def build_dynamic_graph(self, features, fps):
        """
        兼容接口，自动选择 GPU 或 CPU 版本
        """
        if torch.cuda.is_available():
            return self.build_dynamic_graph_gpu(features, fps)
        else:
            return self._build_dynamic_graph_cpu(features, fps)
    
    def _build_dynamic_graph_cpu(self, features, fps):
        """
        原始 CPU 版本（回退用）
        """
        n = len(features)
        G = nx.Graph()
        init_k = 5
        clip_feats = features[:, :512].astype(np.float32)
        flow_feats = features[:, 512:].astype(np.float32)
        clip_norms = np.linalg.norm(clip_feats, axis=1, keepdims=True)
        clip_feats = clip_feats / (clip_norms + 1e-6)
        
        total_blocks = math.ceil(n / Config.graph_block_size)
        with tqdm(total=total_blocks**2, desc="构建动态图(CPU)") as pbar:
            for i in range(0, n, Config.graph_block_size):
                i_end = min(i + Config.graph_block_size, n)
                block_size_i = i_end - i
                clip_block = clip_feats[i:i_end]
                for j in range(0, n, Config.graph_block_size):
                    j_end = min(j + Config.graph_block_size, n)
                    block_size_j = j_end - j
                    clip_sim_block = np.dot(clip_block, clip_feats[j:j_end].T)
                    flow_block = flow_feats[j:j_end]
                    flow_dist_block = np.sqrt(
                        np.sum((flow_feats[i:i_end, np.newaxis] - flow_block)**2, axis=2))
                    time_diff = np.abs(np.arange(i, i_end)[:, None] - np.arange(j, j_end))
                    time_penalty = 1 + Config.time_decay * time_diff
                    combined_sim = (Config.clip_weight * clip_sim_block + 
                                   (1-Config.clip_weight) * np.exp(-flow_dist_block)) / time_penalty
                    dynamic_k = max(3, init_k - (i//(n//10)))
                    valid_k = min(dynamic_k, block_size_j)
                    for local_i in range(block_size_i):
                        global_i = i + local_i
                        if valid_k <= 0: continue
                        kth = min(valid_k - 1, block_size_j - 1)
                        top_k = np.argpartition(-combined_sim[local_i], kth)[:valid_k]
                        for local_j in top_k:
                            global_j = j + local_j
                            if global_i != global_j and combined_sim[local_i, local_j] > 0:
                                G.add_edge(global_i, global_j, weight=combined_sim[local_i, local_j])
                    pbar.update(1)
        for i in range(n):
            G.add_node(i, feature=features[i])
        return G

    def process(self, frames, fps):
        features = self.extractor.extract_features(frames)
        G = self.build_dynamic_graph(features, fps)
        G = graph_propagation(G)
        return detect_boundaries(G, fps)
