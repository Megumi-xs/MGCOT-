#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
独立测试脚本：加载训练好的 MGCOT 模型，在测试集上计算 P@20 和 MRR@20。
用法：
    python test.py --dataset diginetica [--batchSize 512]
"""

import argparse
import pickle
import time
import numpy as np
import torch
from utils import Data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica',
                    help='数据集名称: Tmall/Nowplaying/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=512, help='测试批次大小')
args = parser.parse_args()

def evaluate(model, test_data, batch_size, device):
    model.eval()
    hits, mrrs = [], []
    slices = test_data.generate_batch(batch_size)

    with torch.no_grad():
        for i in slices:
            # 1. 获取一个 batch 的数据（与训练时 forward 函数完全一致）
            global_inputs, global_items, alias_inputs, A_list, items, mask, targets = test_data.get_slice(i)

            # 2. 构建局部会话图邻接矩阵和度矩阵逆（使用去重物品集 items）
            A_hat, D_hat = test_data.get_overlap(items.tolist() if isinstance(items, np.ndarray) else items)
            A_hat = torch.FloatTensor(A_hat).to(device)
            D_hat = torch.FloatTensor(D_hat).to(device)

            # 3. 数据转为张量并移至设备（类型与训练时保持一致）
            global_inputs = torch.LongTensor(global_inputs).to(device)
            global_items = torch.LongTensor(global_items).to(device)
            alias_inputs = torch.LongTensor(alias_inputs).to(device)
            A = torch.FloatTensor(np.array(A_list)).to(device)      # 避免逐元素转换的性能警告
            items = torch.LongTensor(items).to(device)
            mask = torch.LongTensor(mask).to(device)                # 训练时 mask 转为 long
            targets = torch.LongTensor(targets).to(device)

            # 4. 模型前向传播（与 recommender.py 中 forward 的调用一致）
            seq_hidden_gnn, target_emb, x_n, relation_emb, global_emb = model(
                items, A, alias_inputs, A_hat, D_hat, mask, global_inputs, global_items
            )

            # 5. 使用 compute_scores 获得最终预测分数（训练时完全相同的方式）
            #    compute_scores 参数: (hidden, mask, target_emb, att_hidden, relation_emb, global_emb)
            scores, _ = model.compute_scores(seq_hidden_gnn, mask, target_emb, x_n, relation_emb, global_emb)
            # scores 形状: (batch_size, num_items-1)  ，第 0 列对应物品 ID 1，以此类推

            # 6. 取 Top-20 索引（0-based）
            _, top20_idx = torch.topk(scores, 20, dim=1)  # (batch_size, 20)

            # 7. 计算命中率与 MRR
            for idx, target in enumerate(targets):
                target_idx = target.item() - 1  # 转为 0‑based 索引（对应 scores 的列）
                if target_idx < 0:              # 标签不应为 0，安全跳过
                    continue
                top_list = top20_idx[idx].cpu().numpy()
                if target_idx in top_list:
                    hits.append(1)
                    rank = np.where(top_list == target_idx)[0][0] + 1  # 排名（1-based）
                    mrrs.append(1.0 / rank)
                else:
                    hits.append(0)
                    mrrs.append(0)

    hit = np.mean(hits) * 100
    mrr = np.mean(mrrs) * 100
    return hit, mrr

def main():
    # 加载测试数据
    test_data_path = f'./datasets/{args.dataset}/test.txt'
    test_data = pickle.load(open(test_data_path, 'rb'))
    test_data = Data(test_data, shuffle=False)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载训练好的模型
    model_path = f"./final_model/{args.dataset}_model.pkl"
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # 计时评估
    start = time.time()
    hit, mrr = evaluate(model, test_data, args.batchSize, device)
    end = time.time()

    # 输出结果（与原注释格式一致）
    print('Result:')
    print('\tP@20:\t%.4f\tMRR@20:\t%.4f' % (hit, mrr))
    print('-------------------------------------------------------')
    print("Test time: %f s" % (end - start))

if __name__ == '__main__':
    main()