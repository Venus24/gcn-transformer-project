import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix

characters_df = pd.read_excel("data.xlsx", sheet_name="characters")
relations_df = pd.read_excel("data.xlsx", sheet_name="relations")

def build_graph_and_features(characters_df, relations_df):
    name_to_index = {name: idx for idx, name in enumerate(characters_df["谥号"])}
    num_characters = len(characters_df)
    adj_matrix = np.zeros((num_characters, num_characters))

    # 初始化边类型列表
    edge_types = []

    # 构建邻接矩阵并获取边类型
    for _, row in relations_df.iterrows():
        char1 = row["人物1"]
        char2 = row["人物2"]
        rel_type = row["关系"]
        if char1 in name_to_index and char2 in name_to_index:
            idx1 = name_to_index[char1]
            idx2 = name_to_index[char2]
            adj_matrix[idx1, idx2] = 1
            adj_matrix[idx2, idx1] = 1
            # 添加边类型到 edge_types 列表中
            edge_types.append(rel_type)

    # 转换为稀疏矩阵并生成边索引
    adj_csr = csr_matrix(adj_matrix)
    edge_index, _ = from_scipy_sparse_matrix(adj_csr)

    # 创建特征矩阵
    country_one_hot = pd.get_dummies(characters_df["国籍"], prefix='国籍')
    workplace_one_hot = pd.get_dummies(characters_df["工作单位"], prefix='工作单位')
    
    # 计算特征的数量
    num_features = country_one_hot.shape[1] + workplace_one_hot.shape[1]
    
    # 构建特征矩阵
    features = np.zeros((num_characters, num_features))
    features[:, :country_one_hot.shape[1]] = country_one_hot.values
    features[:, country_one_hot.shape[1]:] = workplace_one_hot.values

    # 将边类型转换为整数类型的列表
    unique_edge_types = list(set(edge_types))
    edge_type_to_index = {edge_type: idx for idx, edge_type in enumerate(unique_edge_types)}
    edge_types_int = [edge_type_to_index[edge_type] for edge_type in edge_types]
    
    # 如果边类型数量不足，则补全为默认类型（假设默认类型为0）
    if len(edge_types_int) < edge_index.shape[1]:
        default_type = edge_type_to_index[unique_edge_types[0]]  # 选择第一个边类型作为默认类型
        num_missing_types = edge_index.shape[1] - len(edge_types_int)
        edge_types_int.extend([default_type] * num_missing_types)
        print(f"补全了 {num_missing_types} 条边类型为默认类型 {unique_edge_types[0]}")

    edge_types_tensor = th.tensor(edge_types_int, dtype=th.int64)

    print("Number of edges:", edge_index.shape[1])
    print("Number of edge types:", len(edge_types))
    
    return edge_index, features, edge_types_tensor, unique_edge_types

# 调试语句
edge_index, features, edge_types_tensor, unique_edge_types = build_graph_and_features(characters_df, relations_df)
print("Edge index shape:", edge_index.shape)
print("Features shape:", features.shape)
print("Edge types tensor length:", len(edge_types_tensor))
print("Number of unique edge types:", len(unique_edge_types))

# 定义GCN层
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return F.relu(x)

# 定义GCN网络
class GCNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNNetwork, self).__init__()
        self.layer1 = GCNLayer(input_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        return x

# 定义Transformer层
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.layer_norm(x)
        ff_output = self.feedforward(x)
        x = x + ff_output
        return self.layer_norm(x)

# 集成模型的示例
def integrate_models(edge_index, features, support_set, query_set):
    input_dim = features.shape[1]
    hidden_dim = 64
    output_dim = 32
    transformer_dim = 32
    nhead = 4

    # 实例化GCN
    gcn = GCNNetwork(input_dim, hidden_dim, output_dim)

    # 应用GCN
    gcn_output = gcn(th.tensor(features, dtype=th.float32), edge_index)

    # 实例化并应用Transformer
    transformer = TransformerLayer(transformer_dim, nhead)
    transformer_output = transformer(gcn_output.unsqueeze(0)).squeeze(0)

    # 计算支持集和查询集之间的余弦相似度
    support_embeddings = transformer_output[support_set]
    query_embeddings = transformer_output[query_set]

    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarities = cos_sim(support_embeddings.unsqueeze(1), query_embeddings.unsqueeze(0))

    return similarities

# 示例调用 integrate_models
support_set = [0, 1, 2]  # 假设支持集的节点索引
query_set = [3, 4, 5]    # 假设查询集的节点索引
similarities = integrate_models(edge_index, features, support_set, query_set)
print("Similarities:", similarities)
