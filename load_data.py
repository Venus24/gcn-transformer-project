import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch as th
from torch_geometric.utils import from_scipy_sparse_matrix
import dgl

def load_data(file_path):
    # 加载 Excel 文件中的两个 sheet
    characters_df = pd.read_excel(file_path, sheet_name="characters")
    relations_df = pd.read_excel(file_path, sheet_name="relations")

    name_to_index = {name: idx for idx, name in enumerate(characters_df["谥号"])}
    num_characters = len(characters_df)
    adj_matrix = np.zeros((num_characters, num_characters))
    edge_types = []

    for _, row in relations_df.iterrows():
        char1 = row["人物1"]
        char2 = row["人物2"]
        rel_type = row["关系"]
        if char1 in name_to_index and char2 in name_to_index:
            idx1 = name_to_index[char1]
            idx2 = name_to_index[char2]
            adj_matrix[idx1, idx2] = 1
            adj_matrix[idx2, idx1] = 1
            edge_types.append(rel_type)

    adj_csr = csr_matrix(adj_matrix)
    edge_index, _ = from_scipy_sparse_matrix(adj_csr)
    g = dgl.from_scipy(adj_csr)

    country_one_hot = pd.get_dummies(characters_df["国籍"], prefix='国籍')
    workplace_one_hot = pd.get_dummies(characters_df["工作单位"], prefix='工作单位')
    
    num_features = country_one_hot.shape[1] + workplace_one_hot.shape[1]
    features = np.zeros((num_characters, num_features))
    features[:, :country_one_hot.shape[1]] = country_one_hot.values
    features[:, country_one_hot.shape[1]:] = workplace_one_hot.values

    unique_edge_types = list(set(edge_types))
    edge_type_to_index = {edge_type: idx for idx, edge_type in enumerate(unique_edge_types)}
    edge_types_int = [edge_type_to_index[edge_type] for edge_type in edge_types]

    if len(edge_types_int) < g.number_of_edges():
        default_type = edge_type_to_index[unique_edge_types[0]]
        num_missing_types = g.number_of_edges() - len(edge_types_int)
        edge_types_int.extend([default_type] * num_missing_types)
        print(f"补全了 {num_missing_types} 条边类型为默认类型 {unique_edge_types[0]}")

    edge_types_tensor = th.tensor(edge_types_int, dtype=th.int64)
    features_tensor = th.tensor(features, dtype=th.float32)

    return g, features_tensor, edge_types_tensor, unique_edge_types, edge_index
