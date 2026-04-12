import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

def calculate_seq_per_poi(dataset, poi_label_dict, t_end, t_granularity):
    """
    【抽取 POI 门面热度日历】：
    分析每个具体的商铺店面（POI），在这一周的每一小时里分别迎来了多少人流量。
    比如查出“某酒吧”总是在半夜人流暴增，而“某早餐店”总是在早六点人流暴增。
    """
    checkin_seq_per_poi = {}
    for uid, seq in enumerate(dataset):
        # 把大段时间转换成星期几、几点这种直白的小时坐标(24格装)
        checkin_hour = (np.array(seq['arrival_times']) * (seq['condition1_indicator'][0]-24)).astype(int) 
        
        for pid, poi in enumerate(seq['checkins']):
            if poi not in checkin_seq_per_poi.keys():       
                # 建立一个记录该店面 7*24 小时的查账长条阵列
                checkin_seq_per_poi[poi] = {'chekcin_array': np.zeros((1, t_end * t_granularity)), 'total_check_in': 0, 'label': poi_label_dict[poi]}
            
            # 记录那家店在那个小时进客人了
            checkin_seq_per_poi[poi]['chekcin_array'][0, checkin_hour[pid]] += 1
            checkin_seq_per_poi[poi]['total_check_in'] += 1
    return checkin_seq_per_poi

def generate_training_data(checkin_seq_per_poi, limit=1):
    """【过滤清洗工】：把那些只来过不到 limit 个客人、根本看不出经营规律的死店给剔除扔掉"""
    X = []
    y = []
    for key, values in checkin_seq_per_poi.items():
        if values['total_check_in'] >= limit:
            X.append(values['chekcin_array'])
            y.append(values['label']) # label 就是这家店到底是干啥的（比如标签：餐饮，酒店等）
    if len(X) == 0:
        return np.array([]), np.array([])
    X = np.concatenate(X)
    y = np.array(y)
    return X, y

def run_SemLoc_task(test_data, generated_data, poi_label_dict, t_day=7, t_granularity=24):
    """
    【语义推理交叉统考】：
    先拿真人的店面热度推测这店是啥类型；再拿 AI 生成的店面热度去推测类型。看两者差别大不大。
    """
    data_transformed_list = []
    datalist = [test_data, generated_data]
    for idx, data in enumerate(datalist):
        checkin_seq_per_poi = calculate_seq_per_poi(data, poi_label_dict, t_day, t_granularity)
        data_transformed_list.append(checkin_seq_per_poi)

    all_result = []
    # 请来 sklearn 家族里的 5 位常见看相大师（分类预测器）：决策树、朴素贝叶斯、近邻、支持向量机、逻辑回归。
    models = [tree.DecisionTreeClassifier(), GaussianNB(), sklearn.neighbors.KNeighborsClassifier(), sklearn.svm.LinearSVC(), sklearn.linear_model.LogisticRegression(multi_class="multinomial")]
    model_names = ['tree.DecisionTreeClassifier', 'naive_bayes.GaussianNB', 'neighbors.KNeighborsClassifier', 'svm.LinearSVC', 'linear_model.LogisticRegression']
    metrics_names = ['accuracy', 'f1_micro', 'f1_macro']
    
    for idx, data in enumerate(data_transformed_list):
        results = pd.DataFrame(columns=model_names)
        X, y = generate_training_data(data, 2)
        
        for modelid, model in enumerate(models):
            res = []
            for metricid, metric in enumerate(metrics_names):
                # 交叉十折验证：把数据切十块，轮流拿九块当教材训练大师，一块当考卷算分
                cv_scores = cross_val_score(model, X, y, cv=10, scoring=metric)
                res.append(np.mean(cv_scores))
            results[model_names[modelid]] = res
        all_result.append(results)
        
    real_res = all_result[0]
    gene_res = all_result[1]
    
    # 算出他们通过“伪造数据”学到的水平，离“通过人类真实数据”学到的水平差了多少百分比误差！
    absolute_differences = (gene_res - real_res).abs()
    relative_differences = absolute_differences / real_res
    MAPE = relative_differences.mean().mean()
    relative_differences_2 = np.square(relative_differences)
    MSPE = relative_differences_2.mean().mean()
    
    return MAPE, MSPE

"""
========================================================================
【给外行新手的通俗讲解】 -- run_SemLoc.py (店面属性逆推验证器)
========================================================================
这在测试 AI 生成的人群有没有逻辑常识。

现实世界里有一个**“时间-空间潜规则”**：
如果一个店（POI）总是在半夜人声鼎沸狂欢，那么这个店大概率是个**酒吧**或**夜总会**；
如果在早上7点客流爆满，那他大概率是**早餐店**。

这个脚本就是通过这种“根据每天的经营热度时间表，猜测店面是做什么行业（标签）的”。
我们让经典的五大机器学习模型分别在【原版纯人类的足迹】里看几遍找找经验。然后再跑到【AI瞎编造出来的足迹】里找找经验。如果 AI 瞎编的人群里大半夜全挤在早餐店喝粥，那机器学习到的预测特征就全给带歪了拉垮了。
最后结算对比他俩的预测准确率绝对误差 (MAPE / MSPE)。误差越小，说明 AI 学习领悟的社会常识越完美贴切近人类真实生活法则。
========================================================================
"""
