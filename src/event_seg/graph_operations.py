import numpy as np
import torch
from config import Config


def graph_propagation(G):
    """
    GPU 加速的图传播算法：使用 GAT 风格的注意力机制更新节点特征
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return G
    
    device = Config.device if torch.cuda.is_available() else 'cpu'
    
    # 获取特征矩阵
    features = np.array([G.nodes[i]['feature'] for i in nodes])
    features_t = torch.from_numpy(features).float().to(device)
    
    # 构建稀疏邻接矩阵的边列表
    edge_list = []
    edge_weights = []
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        edge_list.append([u, v])
        edge_list.append([v, u])  # 无向图
        edge_weights.append(w)
        edge_weights.append(w)
    
    if len(edge_list) == 0:
        return G
    
    # 转为 tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=device)
    
    # 计算度归一化
    row = edge_index[0]
    deg = torch.zeros(n, device=device)
    deg.scatter_add_(0, row, edge_weight)
    deg_inv = 1.0 / (deg + 1e-6)
    
    # 归一化权重
    norm_weight = edge_weight * deg_inv[row]
    
    # 迭代传播
    for _ in range(Config.gat_iters):
        # 稀疏矩阵乘法
        col = edge_index[1]
        
        # 聚合邻居特征
        new_features = torch.zeros_like(features_t)

        # 使用 scatter_add 进行高效聚合
        weighted_features = features_t[col] * norm_weight.unsqueeze(1)
        new_features.scatter_add_(0, row.unsqueeze(1).expand(-1, features_t.size(1)), weighted_features)
        
        # 残差连接
        features_t = 0.5 * features_t + 0.5 * new_features
    
    # 转回 numpy
    features = features_t.cpu().numpy()
    
    # 更新图中的特征
    for i, node in enumerate(nodes):
        G.nodes[node]['feature'] = features[i]
    
    del features_t
    torch.cuda.empty_cache()
    
    return G


def graph_propagation_sparse(G):
    """
    使用 PyTorch Sparse 的更高效版本（适合超大图）
    """
    try:
        import torch_sparse
        # 如果安装了 torch_sparse，可以用更高效的实现
        return _graph_propagation_torch_sparse(G)
    except ImportError:
        return graph_propagation(G)


def _graph_propagation_torch_sparse(G):
    """
    使用 torch_sparse 的实现
    """
    import torch_sparse
    
    nodes = sorted(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return G
    
    device = Config.device if torch.cuda.is_available() else 'cpu'
    
    features = np.array([G.nodes[i]['feature'] for i in nodes])
    features_t = torch.from_numpy(features).float().to(device)
    
    edge_list = []
    edge_weights = []
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        edge_list.append([u, v])
        edge_list.append([v, u])
        edge_weights.append(w)
        edge_weights.append(w)
    
    if len(edge_list) == 0:
        return G
    
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=device)
    
    # 使用 torch_sparse 进行高效稀疏矩阵运算
    for _ in range(Config.gat_iters):
        # 稀疏矩阵乘法
        new_features = torch_sparse.spmm(
            edge_index, edge_weight, n, n, features_t
        )
        
        # 度归一化
        deg = torch_sparse.spmm(
            edge_index, edge_weight, n, n, 
            torch.ones(n, 1, device=device)
        ).squeeze()
        new_features = new_features / (deg.unsqueeze(1) + 1e-6)
        
        # 残差连接
        features_t = 0.5 * features_t + 0.5 * new_features
    
    features = features_t.cpu().numpy()
    
    for i, node in enumerate(nodes):
        G.nodes[node]['feature'] = features[i]
    
    del features_t
    torch.cuda.empty_cache()
    
    return G
