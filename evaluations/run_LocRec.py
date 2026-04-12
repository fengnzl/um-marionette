from recbole.quick_start import run_recbole, run_recboles
from recbole.config import Config
import argparse
from ast import arg
from pathlib import Path
import os

def run_LocRec_task(dataset_path, model, dataset, setting, cuda):
    """
    【通用地点推荐评测员】：
    这个脚本会呼叫当今业内极度出名的推荐算法测评框架库：RecBole。
    也就是：把我们的路线放进去，让考官用经典的 BPR 等算法蒙住一只眼睛，看它能不能根据前面走过的路线，准确推荐 / 猜对下一步用户会去哪个地点。
    """
    parameter_dict = {
        'data_path': dataset_path,              # 找刚才预处理生成的 CSV 大表格
        'metrics': ['MRR', 'NDCG', 'Hit'],      # 三个经典考试项目：命中率、倒数排名折损等（如果第一个就是正解的分数最高）
        'valid_metric': 'MRR@10',               # 拿首选前十名的准确结果来代表其水平
        'topk': [5, 10],                        # 给算法每次蒙五次机会和蒙十次机会
        'load_col': {'inter': ['user_id', 'item_id', 'rating']}, # 【重点】：一般推荐只认是谁(user)对什么(item)去过几次(rating)
        'epochs': 50,                           # 补习 50 轮
        'train_neg_sample_args': None if setting == '1' else {'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0},  
        'train_batch_size': 512,
        'eval_batch_size': 512, 
        'gpu_id': cuda                          # 在哪个显卡上干活
    }

    # 一把梭子：给库扔配置跑分，最后 RecBole 会在终端自己啪啪啪给你打出及格成绩！
    run_recbole(model=model, dataset=dataset, config_dict=parameter_dict)


if __name__ == "__main__":
    # 接收从外面命令行敲进来的指令和参数要求
    parser = argparse.ArgumentParser()
    parser.add_argument("--savepath", type=str, default='', help="name of savepath")
    parser.add_argument("--dataset_name", type=str, default='', help="name of dataset")
    parser.add_argument("--model_name", type=str, default='BPR', help="name of model")
    parser.add_argument("--change_setting", type=str, default='0', help="parameter changing")
    parser.add_argument("--cuda", type=str, default='0', help="index of cuda")
    args, _ = parser.parse_known_args()
    
    # 执行评测
    run_LocRec_task(args.savepath, args.model_name, args.dataset_name, args.change_setting, args.cuda)

"""
========================================================================
【给外行新手的通俗讲解】 -- run_LocRec.py (一般预测地点考卷脚本)
========================================================================
这就像是一张“去外校参加竞赛联考”的准考证打印脚本。
当我们模型在家里被关着小黑屋苦训，它产生了一大批乱七八糟它自以为非常真实的路线。

现在，我们要把它丢进一个公正严明的第三方机器法庭：`RecBole （伯乐推荐算法库）`。
`run_LocRec` 就是告诉老法官：
- “你给我考查一下这个学生(AI造出的数据集)，你看它里面的虚假游客，平常喜欢去星巴克和火锅店，你能用最老土经典的 `BPR推荐算法模型`，预测出他明天还会去哪吗？”长久来看，他这种乱造的死数据能推荐出结果吗？

如果AI造假得太垃圾没有规律逻辑，那么这个测评环节的分数(`MRR@10` 等) 会低的惨不忍睹！相反，分数越高，说明 AI 自己造的人拥有类似普通人类一样的稳定爱好和规律！
========================================================================
"""