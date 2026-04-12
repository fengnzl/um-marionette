import torch
import math
import torch.nn as nn

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    计算给定函数 (alpha_bar) 所对应的 Beta 衰减时间表。

    在扩散模型中，我们需要知道每一步注入多少噪声。这个参数表就是用来控制从第一步完全清晰
    到最后一步完全混浊（随机噪声）的变质过程的平滑程度。
    
    外行理解：这里使用了基于余弦平滑（Cosine schedule）的算法，
    这是著名论文 https://arxiv.org/abs/2112.10741 里提出的一种让加噪过程变得更平滑的技术，
    能防止破坏图像（或序列）时破坏得太突然。

    Parameters
    ----------
    num_diffusion_timesteps : int
        我们要走的总扩散步数（比如 100 步）
    alpha_bar : callable
        这是一个传进来的数学函数表达式，它代表着在这 100 步里，信息的保留率（累积乘积）。
        例如计算在 t=[0,1] 百分比进度下还剩多少原本的信息。
    max_beta : float
        单次加噪的最大限制值（默认 0.999，不允许一步把信息全抹除干净变成 1）

    Returns
    -------
    betas : torch.Tensor
        返回一个列表，每一项就是对应那一步该加多少比例的噪声（变质率）。
    """
    betas = []
    # 循环走完所有的扩散步数
    for i in range(num_diffusion_timesteps):
        # t1：上一步所在的百分比进度
        t1 = i / num_diffusion_timesteps
        # t2：当前步所在的百分比进度
        t2 = (i + 1) / num_diffusion_timesteps
        
        # 通过代入 alpha_bar 函数，算出这两步之间信息丢失的比例，反推求出需要加多少噪音 beta。
        # 且设定上限为 max_beta。
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        
    # 打包成 PyTorch 能算的大张量返回
    return torch.tensor(betas)

"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/diffusion/utils.py
========================================================================

1. 这个工具管什么用？
   - 它是制造“时序沙漏”的刻度尺。
   - 扩散模型就是把一个好好的沙堡（事件序列序列）慢慢推倒变成一盘散沙（噪声），然后再学会怎么把散沙重新推回沙堡。
   - 这段代码回答了：在推倒它的几百步里，每一步该推掉多少沙子？（即 `betas` 时间表）。

2. 为什么不用最简单的“每次都倒掉 1%”？
   - 那种叫线性 schedule。但研究界发现如果按线性推倒，开头两下就把信息破坏得面目全非了，导致AI很难学好收尾修补工作。
   - 这里实现的叫 "Cosine Beta Schedule"，就像汽车平稳加速和减速一样，它在推倒沙堡的过程里显得更为细腻连贯。
========================================================================
"""
