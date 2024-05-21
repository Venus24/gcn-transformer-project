import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from data import load_data
from model import GCNNetwork, TransformerLayer, KnowledgeGraphEmbedding
from utils import train, test, train_link_prediction, test_link_prediction

# 加载数据
file_path = 'D:/desktop/JNU/M2/知识图谱/解决知识图谱中的实体表示学习/data/data.xlsx'
g, features_tensor, edge_types_tensor, unique_edge_types, edge_index = load_data(file_path)

# 定义训练、验证和测试掩码（示例：随机划分）
num_nodes = g.num_nodes()
train_mask = th.zeros(num_nodes, dtype=th.bool)
val_mask = th.zeros(num_nodes, dtype=th.bool)
test_mask = th.zeros(num_nodes, dtype=th.bool)

indices = np.arange(num_nodes)
np.random.shuffle(indices)
train_split = int(num_nodes * 0.6)
val_split = int(num_nodes * 0.2)
train_mask[indices[:train_split]] = True
val_mask[indices[train_split:train_split + val_split]] = True
test_mask[indices[train_split + val_split:]] = True

data = type('', (), {})()  # 创建一个空对象以存储数据
data.x = features_tensor
data.edge_index = edge_index
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
data.y = torch.randint(0, 2, (num_nodes,))  # 用随机标签填充y（根据需要修改）
data.g = g  # 添加图对象

# 实例化模型和优化器
input_dim = features_tensor.shape[1]
hidden_dim = 64
output_dim = 64  # 确保Transformer和GCN的维度一致

gcn_model = GCNNetwork(input_dim, hidden_dim, output_dim)
optimizer = th.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练和评估GCN模型
for epoch in range(100):
    loss = train(gcn_model, optimizer, data)
    val_acc = test(gcn_model, data, data.val_mask)
    if epoch % 10 == 0:
        print(f'Epoch For GCN: {epoch}, Loss: {loss}, Val Acc: {val_acc}')

test_acc = test(gcn_model, data, data.test_mask)
print(f'Test Accuracy: {test_acc}')

# 链路预测模型
rgcn_model = KnowledgeGraphEmbedding(input_dim, hidden_dim, output_dim, num_rels=len(unique_edge_types))
optimizer_lp = th.optim.Adam(rgcn_model.parameters(), lr=0.01, weight_decay=5e-4)
neg_edge_index = negative_sampling(data.edge_index, num_nodes=g.num_nodes(), num_neg_samples=data.edge_index.size(1))

# 训练和评估RGCN模型
for epoch in range(100):
    loss = train_link_prediction(rgcn_model, optimizer_lp, data, data.edge_index, neg_edge_index, edge_types_tensor)
    if epoch % 10 == 0:
        print(f'Epoch For RGCN: {epoch}, Loss: {loss}')

roc_auc = test_link_prediction(rgcn_model, data, data.edge_index, neg_edge_index, edge_types_tensor, roc_auc_score)
print(f'ROC AUC: {roc_auc}')

# 计算和打印相似性
def integrate_models_gcn(edge_index, features, support_set, query_set):
    transformer_dim = 64  # 使Transformer的维度与GCN输出一致
    nhead = 4

    gcn_output = gcn_model(features, edge_index)
    print(f"GCN Output: {gcn_output}")  # 添加调试信息

    transformer = TransformerLayer(transformer_dim, nhead)
    transformer_output = transformer(gcn_output.unsqueeze(0)).squeeze(0)
    print(f"Transformer Output: {transformer_output}")  # 添加调试信息

    support_embeddings = transformer_output[support_set]
    query_embeddings = transformer_output[query_set]

    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarities = cos_sim(support_embeddings.unsqueeze(1), query_embeddings.unsqueeze(0))
    print(f"GCN Similarities: {similarities}")  # 添加调试信息

    return similarities

def integrate_models_rgcn(g, features, etypes, support_set_indices, query_set_indices):
    support_set = features[support_set_indices]
    query_set = features[query_set_indices]
    node_emb = rgcn_model(g, features, etypes, support_set)
    print(f'Node embeddings: {node_emb}')  # 添加调试信息
    support_mean = node_emb[support_set_indices].mean(dim=0)
    print(f'Support mean: {support_mean}')  # 添加调试信息
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarities = cos_sim(node_emb[query_set_indices], support_mean.unsqueeze(0).expand(query_set.size(0), -1))
    print(f'RGCN Similarities: {similarities}')  # 添加调试信息
    return similarities

# 支持集和查询集的索引
support_set_indices = [0, 1, 2]
query_set_indices = [3, 4, 5]

# 计算并打印GCN模型的相似性
similarities_gcn = integrate_models_gcn(edge_index, features_tensor, support_set_indices, query_set_indices)
print("GCN Similarities:", similarities_gcn)

# 计算并打印RGCN模型的相似性
similarities_rgcn = integrate_models_rgcn(g, features_tensor, edge_types_tensor, support_set_indices, query_set_indices)
print("RGCN Similarities:", similarities_rgcn)
