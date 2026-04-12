import numpy as np

def distance(lat1, lon1, lat2, lon2):
    """
    【球面距离计算】：
    因为地球是个圆球（近似），不能用纯纯直角坐标系来算直线距离。
    使用半正矢公式 (Haversine Formula) 计算真实地表行走经纬度的圆弧距离（以 km 公里为单位）。
    """
    # 将经纬度度数转换为弧度制
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # 经典的半正矢核心算式
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # 地球平均半径，单位：公里
    return c * r

def travel_distance(geo):
    """
    一个人从早到晚跑了这么多个连续地点，
    他今天走过的“总里程”是多少。(把前一个点到后一个点的跳跃距离加在一起)
    """
    return np.sum(distance(geo[:-1, 0], geo[:-1, 1], geo[1:, 0], geo[1:, 1]))

def radius(geo):
    """
    他今天活动的“离家半径/生活圈子”有多大。
    (算出他的所有活动中心原点，然后取各个位置到它中心的均方根距离)
    """
    center = np.mean(geo, axis=0)
    return np.sqrt(np.mean(distance(geo[:, 0], geo[:, 1], center[0], center[1])))

def JSD(P_A, P_B):
    """
    【JS 散度 Jensen-Shannon Divergence】：
    一种基于信息熵来衡量“两群人的长相脾气（两个概率分布）有多相似”的方法。
    用来判断机器造出的假数据分布是否完美贴合真人的分布。
    """
    epsilon = 1e-14
    # 为防对数 0 崩溃，加一个极小的扰动量，并在后面平移均一化成正儿八经的百分数分布。
    P_A = (P_A / P_A.sum() + epsilon)
    P_B = (P_B / P_B.sum() + epsilon)
    # 取两者的中和折中
    P_merged = 0.5 * (P_A + P_B)
    
    # 算 KL 这个纯种散度
    kl_PA_PM = np.sum(P_A * np.log(P_A / P_merged))
    kl_PB_PM = np.sum(P_B * np.log(P_B / P_merged))
    
    jsd = 0.5 * (kl_PA_PM + kl_PB_PM)
    return jsd

def arr_to_distribution(arr, min, max, bins):
    """用直方图网格强行把无规则的连续数字分桶（归入各柱子内），借此测定分布热图"""
    distribution, base = np.histogram(
        arr, np.arange(min, max, float(max - min) / bins))
    return distribution

def compute_probability_distribution(data):
    """把零散的代号频率转换为占比率"""
    unique_elements, counts = np.unique(data, return_counts=True)
    total_counts = np.sum(counts)
    probabilities = counts / total_counts
    return unique_elements, probabilities

def category_jsd(generated_category, real_category):
    """
    专门评定【行为类别分布】的差距：
    它造出来假的人去公园玩和去餐厅的比重，是否和真人一样？
    """
    gen_category, prob_gen = compute_probability_distribution(generated_category)
    real_category, prob_real = compute_probability_distribution(real_category)

    p, q = (list(zip(gen_category, prob_gen)), list(zip(real_category, prob_real)))
    p, q = np.asarray(p), np.asarray(q)

    all_elements = set(p[:, 0]).union(set(q[:, 0]))
    p_probs = {element: 0.0 for element in all_elements}
    q_probs = {element: 0.0 for element in all_elements}
    
    for element, prob in p: p_probs[element] = prob
    for element, prob in q: q_probs[element] = prob

    return JSD(np.array(list(p_probs.values())), np.array(list(q_probs.values())))

def grank_jsd(generated_category, real_category, top=1000):
    """
    【热门事件/大网红地标 G-RANK】：
    专门揪出最火爆的热门地点头头们，比较 AI 造的假网红地和现实的真网红地排行出入程度。
    """
    gen_category, prob_gen = compute_probability_distribution(generated_category)
    real_category, prob_real = compute_probability_distribution(real_category)
    
    # 排行榜打榜排序
    sorted_indices = np.argsort(-prob_gen)
    gen_category, prob_gen = gen_category[sorted_indices], prob_gen[sorted_indices]
    
    sorted_indices = np.argsort(-prob_real)
    real_category, prob_real = real_category[sorted_indices], prob_real[sorted_indices]
    
    # 只取最靠前的明星头部来比对
    tt = top
    gen_category, prob_gen = gen_category[:tt], prob_gen[:tt]
    real_category, prob_real = real_category[:tt], prob_real[:tt]
    p, q = (list(zip(gen_category, prob_gen)), list(zip(real_category, prob_real)))

    p, q = np.asarray(p), np.asarray(q)

    all_elements = set(p[:, 0]).union(set(q[:, 0]))
    p_probs = {element: 0.0 for element in all_elements}
    q_probs = {element: 0.0 for element in all_elements}
    
    for element, prob in p: p_probs[element] = prob
    for element, prob in q: q_probs[element] = prob

    return JSD(np.array(list(p_probs.values())), np.array(list(q_probs.values())))

def evaluation(generated, original):
    """通用的 JS 散度测量法"""
    generated = np.array(generated)
    original = np.array(original)
    assert len(generated) > 0 and len(original) > 0
    max_val = np.max(generated) if np.max(generated) > np.max(original) else np.max(original)
    p_gen = arr_to_distribution(generated, 0, max_val, 100)
    p_real = arr_to_distribution(original, 0, max_val, 100)
    return JSD(p_gen, p_real)

def get_visits(trajs, max_locs):
    visits = np.zeros(shape=(max_locs), dtype=float)
    for t in trajs: visits[t] += 1
    return visits / np.sum(visits)

def get_topk_visits(visits, K):
    locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
    locs_visits.sort(reverse=True, key=lambda d: d[1])
    topk_locs = [locs_visits[i][0] for i in range(K)]
    topk_probs = [locs_visits[i][1] for i in range(K)]
    return np.array(topk_probs), topk_locs

def Get_Statistical_Metrics(real_data, generated_data, min_seq_len=1, top=1000):
    """
    【总统计结算面板】：
    大杂烩流水线，把真数据和假数据拿进来。然后一一测出六大维度的特征全给汇算一个大评分出具发票单子！
    """
    Real_Statistics = {'Distance': [], 'Radius': [], 'DailyLoc': [], 'Interval': [], 'Category':[], 'G-RANK':[]}
    Generated_Statistics = {'Distance': [], 'Radius': [], 'DailyLoc': [], 'Interval': [], 'Category':[], 'G-RANK':[]}
    JSD = {'Distance': 1.0, 'Radius': 1.0, 'DailyLoc': 1.0, 'Interval': 1.0, 'Category':1.0, 'G-RANK':1.0, 'totalJSD': 6.0}

    data = [generated_data, real_data]
    if len(generated_data) == 0 and len(real_data) == 0:
        return JSD
    assert len(generated_data) > 0 and len(real_data) > 0

    metrics_dicts = [Generated_Statistics, Real_Statistics]
    
    # 抽取他们两者在各种维度上的表现
    for idx, seqs in enumerate(data):
        for seq in seqs:
            if len(seq['gps']) > min_seq_len:
                gps = np.array(seq['gps'])
                metrics_dicts[idx]['Distance'].append(travel_distance(gps))
                metrics_dicts[idx]['Radius'].append(radius(gps))
                metrics_dicts[idx]['DailyLoc'].append(len(set(seq['checkins']))) # 一天去过几个非重复特殊门店
                metrics_dicts[idx]['Interval'].extend(np.ediff1d(np.concatenate([[0], seq["arrival_times"]])).tolist())
                metrics_dicts[idx]['Category'].extend(seq['marks'])
                metrics_dicts[idx]['G-RANK'].extend(seq['checkins'])

    # 拉出来溜溜，对每一类别挨个判给相似度得分（JSD）
    for metric in JSD.keys():
        if metric == 'Category':
            JSD[metric] = category_jsd(Generated_Statistics[metric], Real_Statistics[metric])
        elif metric == 'G-RANK':
            JSD[metric] = grank_jsd(Generated_Statistics[metric], Real_Statistics[metric], top)
        elif metric != 'totalJSD':
            JSD[metric] = evaluation(Generated_Statistics[metric], Real_Statistics[metric])
        else:
            break
        JSD['totalJSD'] = sum([JSD[metric] for metric in JSD.keys() if metric != 'totalJSD'])
    return JSD

"""
========================================================================
【给外行新手的通俗讲解】 -- statistical_metrics.py (统计学相似法官)
========================================================================
当我们要验证机器假扮人类厉不厉害时。我们不能光靠肉眼看，我们需要铁证如山的数学指标。
这个文件就是请来的六位非常严苛挑剔的“统计学老法官”：

1. Distance法官：看他一天走的总路程（是不是都在家蹲着，或者满地球乱蹿？）
2. Radius法官：看他一天打转转的生活活动圈子（生活半径）是不是跟凡人一样？
3. DailyLoc法官：看他一个人一天去换着逛的特殊店址数量。
4. Interval法官：看这哥们两件事中间休息磨蹭的空档期是多久。
5. Category法则：它今天喜欢去餐馆、超市、或者是理发店的频率，跟正常市民对不上对的上？
6. G-RANK法则：考察这家伙到底懂得不懂去凑真正的网红店地标的热闹。

最后的 JS 散度 (JSD) ，就像是警察拿两个人的心电图重叠比对，重叠得越精准散度越小（越接近0，造的数据越完美）。这给后面的开发者提供了“机器到底是哪一点漏出了不像人类的破绽”的抓手！
========================================================================
"""