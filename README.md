# 集成GCN-Transformer模型用于实体表示学习

# 概述
该项目集成了图卷积网络（GCN）和Transformer模型，在知识图谱上进行实体表示学习。数据集包含人物及其关系，模型旨在生成这些实体的有意义的嵌入。
# 文件
data.xlsx: 包含人物信息及其关系的数据集。

main.py: 主脚本，用于加载数据、构建图和特征、定义GCN和Transformer模型并计算相似度。

README.md: 本文件，提供项目概述和运行说明。
# 数据集包含两个工作表：
characters: 包含人物属性，如“name”，“nationality”，“work unit”和“谥号”。

relations: 包含人物之间的关系，列有“人物1”和“人物2”。

# 安装所需包
pip install torch torch_geometric pandas numpy scipy scikit-learn

