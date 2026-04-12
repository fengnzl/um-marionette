from recbole.quick_start import run_recbole, run_recboles
from recbole.config import Config
import argparse
from ast import arg
from pathlib import Path
import os

def run_NexLoc_task(dataset_path, model, dataset, setting, cuda):
    """
    【下一个地点预测考验】：
    这也是利用 RecBole 平台。不同于上一个一般推荐不考虑时间，这个叫 `NexLoc`，看重大数据里的先后顺序。
    我们会传入 timestamp，考核推荐算法能不能发现时序规律（比如用户总是先去沙滩，然后顺路去旁边买冰淇淋）。
    """
    parameter_dict = {
        'data_path': dataset_path,              
        'metrics': ['MRR', 'NDCG', 'Hit'],      
        'valid_metric': 'MRR@10',               
        'topk': [5, 10],                        
        # 【重点】：序列推荐不仅要知道 是谁(user) 在 哪个店(item)，还要知道它的 发生时间段(timestamp) 以确立先后排序。
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'epochs': 50,                           
        'train_neg_sample_args': None if setting == '1' else {'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0},  
        'train_batch_size': 512,
        'eval_batch_size': 512, 
        'gpu_id': cuda
    }

    run_recbole(model=model, dataset=dataset, config_dict=parameter_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savepath", type=str, default='', help="name of savepath")
    parser.add_argument("--dataset_name", type=str, default='', help="name of dataset")
    parser.add_argument("--model_name", type=str, default='FPMC', help="name of model")
    parser.add_argument("--change_setting", type=str, default='0', help="parameter changing")
    parser.add_argument("--cuda", type=str, default='0', help="index of cuda")
    args, _ = parser.parse_known_args()
    run_NexLoc_task(args.savepath, args.model_name, args.dataset_name, args.change_setting, args.cuda)

"""
========================================================================
【给外行新手的通俗讲解】 -- run_NexLoc.py (下一个去哪考卷脚本)
========================================================================
跟 `run_LocRec`（一般推荐大锅乱炖）非常相似，都是借助伯乐测评系统 (RecBole)。
但这个脚本的考点是【时序规律能力】。

打个比方：`run_LocRec` 考验的是 AI 懂不懂“小明是个吃货，所以总爱去饭店”。
而这个脚本考验的则是 “小明【吃完饭】下一步必去【消食买奶茶】”。

如果 AI 造出来的数据全是乱随机的，吃完饭下一步就闪现去了医院，再下一步闪现去五金店。
那它在这里得出的分数肯定就是0鸭蛋，因为经典的考官算法 (如 FPMC) 根本从这些乱数据里找不出任何连贯的逻辑！
========================================================================
"""