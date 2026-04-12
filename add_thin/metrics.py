"""
这些评估指标函数取自 TriTPP 的官方实现：
https://github.com/shchur/triangular-tpp/

如需进一步了解详情，请查阅 TriTPP 论文的附录 E.2。
在这个文件中定义了各种距离和指标（Metric），用于衡量模型预测的时序和真实发生的数据之间到底有多“像”。
"""

from typing import List

import numpy as np
# 导入 scipy 的 wasserstein_distance 函数。外行俗称“推土机距离”，就是把一种土堆（概率分布）推成另一种土堆所需的最小工作量。
from scipy.stats import wasserstein_distance


def forecast_wasserstein(
    X: List,                # 模型生成的随机序列（预测值）
    Y: List,                # 真实的历史事件序列（真实值）
    t_max: float,           # 序列发生的最大时间跨度
):
    """
    计算两组点过程序列之间的 Wasserstein 平均真实距离。
    外行理解：衡量两个小球时间序列序列在整体感觉上差多少。
    """
    # 步骤1：形状对齐（保证两个序列列表可以一一对应和作比较）
    X, Y = match_shapes(X, Y, t_max)
    
    # 步骤2：时间归一化，把所有发生的时间点按比例缩小到 0~1 的范围内
    X = X / t_max
    Y = Y / t_max
    t_max = 1

    distance = []
    # zip 是把预测的一条线 X 和真实的一条线 Y 绑在一起对比
    for x, y in zip(X, Y):
        # 步骤3：计算每对时间序列在计数上的距离（counting_distance），存入列表
        distance.append(counting_distance(x, y.reshape(1, -1), t_max=t_max))
    
    # 步骤4：将所有的距离拼接成一个长数组并求平均值（mean），得出一个代表整体“不像度”的得分
    return np.concatenate(distance).mean()


def counting_distance(x: np.ndarray, Y: np.ndarray, t_max: float):
    """
    计算两个序列计数过程（Counting Process）之间的面积距离。
    基于论文：https://arxiv.org/abs/1705.08051 提出的理念。
    
    外行理解：你可以在纸上画一个楼梯图，每发生一件事楼梯就向上走一格。
    这个函数计算的是预测的楼梯和真实的楼梯之间围成的“阴影面积”有多大。阴影面积越小，说明预测越准。
    """
    # 为了不在原数据上直接修改，拷贝一份副本
    x, Y = x.copy(), Y.copy()
    
    # 对 x 进行扩展（重复拷贝），使其行数和 Y 的行数（批量大小）保持一致以便做矩阵减法
    x = x[None].repeat(Y.shape[0], 0)
    
    # 算一下在这个最大时间内，实际上各发生了多少次事件（把不足 t_max 的小颗粒挑出来数一数）
    x_len = (x < t_max).sum(-1)
    y_len = (Y < t_max).sum(-1)
    
    # 判断是谁发生的次数比较多，找出需要交互位置的情况
    to_swap = x_len > y_len
    
    # 修复了原来开源代码里的小 bug（当 x 事件多于 Y 时作一下位置对调交换）
    x[to_swap], Y[to_swap] = Y[to_swap].copy(), x[to_swap].copy()
    
    # 获取有效事件发生的范围（抛弃掉用来占位填补的无效数据点）
    mask_x = x < t_max
    mask_y = Y < t_max
    
    # 第一段计算：两个发生事件时间点的直接绝对值误差（也就是水平距离的误差的总和）
    result = (np.abs(x - Y) * mask_x).sum(-1)
    # 第二段计算：对于真实数据有而预测数据没发生的那些多余部分，计算它们距离 t_max 的截断误差
    result += ((t_max - Y) * (~mask_x & mask_y)).sum(-1)
    
    return result


def gaussian_kernel(x: np.ndarray, sigma2: float = 1):
    """
    高斯核函数。
    外行理解：这就像一个以自己为中心的圆晕，距离中心越近权重越大，越远越小。用于下面计算 MMD 时衡量两个事物的连续相近程度。
    """
    return np.exp(-x / (2 * sigma2))


def match_shapes(X: List, Y: List, t_max: float):
    """
    将两个不同长度的时间序列强行补齐到相同长度，好让矩阵能相减。
    外行理解：找队伍里最高的那个人（最多发生了多少件事情），其余没达到这个次数的人，统统在这个次数的位置后面用 t_max 垫脚填充补齐。
    """
    # 找出列表X所有序列中发生的最多事件次数
    max_x = max([(x < t_max).sum() for x in X])
    # 找出Y里面的最多事件次数
    max_y = max([(y < t_max).sum() for y in Y])
    # 两人比一比，取绝对的最大次数
    max_size = max(max_x, max_y)
    
    # 创建全部用 t_max 填充好的空盘子
    new_X = np.ones((len(X), max_size)) * t_max
    new_Y = np.ones((len(Y), max_size)) * t_max
    
    # 把真正的有效时间点（小于 t_max 的点）逐个塞还到新盘子里
    for i, x in enumerate(X):
        x = x[x < t_max]
        new_X[i, : len(x)] = x
    for i, y in enumerate(Y):
        y = y[y < t_max]
        new_Y[i, : len(y)] = y
        
    # 返回这两盘已经被对齐好的“规则矩阵小方块”
    return new_X, new_Y


def MMD(
    X: List, Y: List, t_max: float, sample_size: int = None, sigma: float = None
):
    """
    计算预测样本 X 和真实样本 Y 的最大均值差异（Maximum Mean Discrepancy）。
    这其实来源于生硬粗糙统计学上的假设检验：MMD = E[k(x,x)] - 2E[k(x,y)] + E[k(y,y)]。
    
    外行理解：用来在“群体层面（分布层面）”看这两组人（预测数据和真实数据）像不像。
    如果这两组人几乎是一模一样的来源，那么 MMD 会趋近于 0。它的考量比直接算平均数要高级和敏感得多。
    """
    # 步骤1：队伍看齐
    X, Y = match_shapes(X, Y, t_max)

    # 如果启用了局部采样（为了加速计算而牺牲一点精度），随机抽几个人出来做代表
    if sample_size is not None:
        X = [X[i] for i in np.random.choice(len(X), sample_size)]
        Y = [Y[i] for i in np.random.choice(len(Y), sample_size)]
        
    # 时间归一化到 0-1 之间
    X = X / t_max
    Y = Y / t_max
    t_max = 1

    # （1）计算 X 组内人员互相之间的楼梯距离集合
    x_x_d = []
    for i, x1 in enumerate(X):
        x_x_d.append(counting_distance(x1, X, t_max=t_max))
    x_x_d = np.concatenate(x_x_d)

    # （2）计算 X 组人员与 Y 组人员互相交错纠缠的距离集合
    x_y_d = []
    for x in X:
        x_y_d.append(counting_distance(x, Y, t_max=t_max))
    x_y_d = np.concatenate(x_y_d)

    # （3）计算 Y 组内部的自嗨距离集合
    y_y_d = []
    for i, y1 in enumerate(Y):
        y_y_d.append(counting_distance(y1, Y, t_max=t_max))
    y_y_d = np.concatenate(y_y_d)

    # 如果没提供高斯算盘的“扩散半径sigma”，就拿这堆距离的中位数作为一个相对客观的代表值
    if sigma is None:
        sigma = np.median(np.concatenate([x_x_d, x_y_d, y_y_d]))
    sigma2 = sigma**2
    
    # 经过高斯核的透镜后，求各自的期望值（均值）
    E_x_x = np.mean(gaussian_kernel(x_x_d, sigma2))
    E_x_y = np.mean(gaussian_kernel(x_y_d, sigma2))
    E_y_y = np.mean(gaussian_kernel(y_y_d, sigma2))

    # 返还最终 MMD 高级数学指标，并且返回刚用的平滑参数 sigma
    return np.sqrt(E_x_x - 2 * E_x_y + E_y_y), sigma


def lengths_distribution_wasserstein_distance(
    X: List, Y: List, t_max: float, mean_number_items: float
):
    """
    单纯只考虑“序列长度分布（发生的总次数分布）”这一项维度，计算两者的 Wasserstein（推土机）距离。
    
    外行理解：上面一大堆函数都是在对比“时间点精准么”。这里直接换个思路抛弃时间戳：
    真实情况是你在这十天出门了 5 次，预测说你出门了 7 次。
    把所有人的这个“次数差”做一个宏观拉偏比较。它只验证“频次”上的预测能力。
    """
    # 数一数 X（预测）里每个序列有效发生了多少次
    X_lengths = np.array([(s < t_max).sum().item() for s in X])
    # 数一数 Y（真实）里每个序列有效发生了多少次
    Y_lengths = np.array([(s < t_max).sum().item() for s in Y])
    
    # 用总平均发生次数去除一下，作归一化，使得结果在 0-1 类似范畴更客观，然后调 SciPy 包计算出两座沙堡被推平需要的力气
    return wasserstein_distance(
        X_lengths / mean_number_items, Y_lengths / mean_number_items
    )

"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/metrics.py
========================================================================

1. 该文件用来干嘛？
   - 它是项目的“AI考核阅卷系统”。
   - 模型好不容易输出了预测的时间序列。但由于这不是个简单的选择题，所以不能用简单的“对与错”(Accuracy) 或“差值大小”(MSE) 来打分。
   - 这点在离散点过程预测中尤其重要，例如模型预测了早上9点和10点会有事件，可真实数据是发生在9点半和10点半。传统对冲差值计算不了。
   
2. 里面这堆复杂的数学是什么意思？
   - MMD：最大均值差异。想象有两个画师画了一群鸟（整体事件流），MMD 是在考官不看落款的情况下，辨别出是不是同一个人画的手法区别。如果MMD很小，说明AI“仿写”时间流的功力已经到了真假难辨的地步。
   - Wasserstein（推土机距离）：它计算把“机器猜的模型”揉碎成真正的现实发生状况，需要多少“搬砖的能量”。需要搬的砖越少，这得分就表明机器的预测越省力、越精准。
   
3. 为什么要写这么长的 `counting_distance`（计数距离）计算法？
   - 因为在多重时序预测中，两组序列长短不一、时间不齐。这就把时序抽象成“随时间往上走的爬楼梯折线图”。然后这个函数就在算两个楼梯投影重叠剩下的空白阴影面积误差。
========================================================================
"""
