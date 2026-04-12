import torch
import numpy as np
import pandas as pd
import os

def preprocessing_data(dicts, source_file, mode):  
    """
    【格式红娘】：负责把我们 AI 生成的时间地点轨迹，翻译转化成下游的推荐算法测试库（RecBole）所能读懂的特定格式文件。
    """
    # 比如截取 'generated_data' 作为名称前缀
    target_file_prefix = source_file.split('.')[0]
    
    # 1. 拆开快递箱：从 .pkl 档案解压出用户的生平轨迹字典
    data = torch.load(dicts + '/' + source_file)
    seqs = data.get('sequences')
    
    user_ids = []
    checkin_times = []
    checkins = []
    
    # 2. 梳理户口本：遍历每个用户，把他们零散的日记整理成规整的一行行大表格
    for i, item in enumerate(seqs):
        # 如果这个人好歹去过哪怕一个以上的地方，我们才要研究他
        if len(item['checkins']) > 1:
            uid = [i] * len(item['checkins'])
            user_ids.extend(uid)
            checkin_times.extend(item['arrival_times'])
            checkins.extend(item['checkins'])
            
    # 用 Pandas 做出大表格 (类似 Excel)
    data_transformed = pd.DataFrame()
    data_transformed['user_id'] = user_ids
    data_transformed['checkin_times'] = checkin_times
    data_transformed['checkins'] = checkins

    # 3. 按照要求，分发两种定制化的表格给下游不同的考场使用：
    if mode:
        # 【模式 一】：专供“序列推荐 (Sequential Recommendation)”考场。讲究的是时间发生的先后顺序。
        if not os.path.exists(f'{dicts}/{target_file_prefix}_for_sequential'):
            os.makedirs(f'{dicts}/{target_file_prefix}_for_sequential')
        
        # 裁剪出需要的列，严格按照 RecBole 要求的特殊原子命名法则（user_id:token 等）
        data_4_seqRec = data_transformed.loc[:, ['user_id', 'checkins', 'checkin_times']]
        data_4_seqRec.columns = ['user_id:token', 'item_id:token', 'timestamp:float']
        
        # 存成一个带有制表符分隔的 .inter 白板文件
        data_4_seqRec.to_csv(f'{dicts}/{target_file_prefix}_for_sequential/{target_file_prefix}_for_sequential.inter', index=False, sep='\t')
    else:
        # 【模式 二】：专供“一般推荐 (General Recommendation)”考场。不看时间，只看总体去了几次。
        if not os.path.exists(f'{dicts}/{target_file_prefix}_for_general'):
            os.makedirs(f'{dicts}/{target_file_prefix}_for_general')
            
        data_4_locRec = data_transformed.loc[:, ['user_id', 'checkins']]
        data_4_locRec.columns = ['user_id:token', 'item_id:token']
        # 统计这个人去过同一个地方分别吃了几次？（计数作为喜欢程度 rating）
        data_4_locRec = data_4_locRec.groupby(['user_id:token', 'item_id:token']).size().reset_index(name='count')
        data_4_locRec.columns = ['user_id:token', 'item_id:token', 'rating:float']
        
        data_4_locRec.to_csv(f'{dicts}/{target_file_prefix}_for_general/{target_file_prefix}_for_general.inter', index=False, sep='\t')

"""
========================================================================
【给外行新手的通俗讲解】 -- preprocessing.py (黑客帝国转码器)
========================================================================
假设我们造出了一个极其聪明的写字机器人（目前的扩散模型），它写出了千万篇人类的《去哪儿玩日记》。
但我们要评判它写的日记像不像真人，还得交给市面上的“考官”（推荐系统算法评测库）。

然而，这考官是个老古董，他不认识我们这个机器人的文字，他只认以 `.inter` 为后缀，而且列名叫 `user_id:token` 这种死板格式的报表。

这也就是这个代码的作用：
它充当一个海关转译员，把我们 AI 生成的漂亮数据，粗暴地揉捏清洗，转换成下游评估包（如著名的算法库 RecBole）规定死认的数据表格，并生成特定的文件夹和 CSV 文件供它们读取评分！
========================================================================
"""