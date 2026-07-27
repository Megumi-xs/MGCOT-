MGCOT：多图协同训练用于基于会话推荐

本项目是对论文《多图协同训练用于基于会话推荐的用户意图捕捉》(MGCOT) 的非官方实现与复现。

模型从当前视角 (Current View)、局部视角 (Local View) 和全局视角 (Global View) 三个角度捕捉用户意图，并通过多头注意力与对比学习增强会话表征。


环境依赖

Python 3.6+
PyTorch (>=1.0)
numpy
scipy
networkx
numba
entmax

安装依赖：
pip install torch numpy scipy networkx numba entmax


数据集

本项目支持以下数据集：
- diginetica (默认)
- Tmall
- RetailRocket


请将预处理好的数据集放置于 datasets/<数据集名称>/ 目录下，所需文件包括：
- train.txt  训练会话及标签（pickle 格式）
- test.txt   测试会话及标签（pickle 格式）
- adj_global.npz  预计算的全局物品共现邻接矩阵（scipy 稀疏矩阵）

目录结构示例：
datasets/
  diginetica/
    train.txt
    test.txt
    adj_global.npz


训练

运行主脚本并指定数据集名称：
python main.py --dataset diginetica

常用超参数：
--batchSize        批次大小（默认：512）
--hiddenSize       隐藏层维度（默认：100）
--epoch            最大训练轮数（默认：30）
--lr               初始学习率（默认：0.001）
--patience         早停耐心值（默认：3）
--contrastive_weight 对比损失权重（默认：1.0）

训练好的模型会保存在 ./final_model/<数据集名称>_model.pkl。


测试

训练完成后，运行测试脚本计算 P@20 和 MRR@20：
python test.py --dataset diginetica

确保模型文件已存在于 ./final_model/ 目录下。


复现结果（Diginetica 数据集）

指标      最佳轮次    数值
P@20       6         67.79%
MRR@20     6         29.08%


引用

如果使用本代码，请引用原论文及开源仓库：
https://github.com/liang-tian-tian/MGCOT


本项目仅供研究与学习使用，具体许可请参考原仓库。