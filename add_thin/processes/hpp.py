# ==========================================
# 导入所需的库
# ==========================================
# 从 typing 引入 Union，用于表示可以是多种类型之一（例如：可以是张量，也可以是空值）
from typing import Union

# 导入深度学习核心库 PyTorch
import torch
# 导入用于张量类型检查和断言的工具（便于开发时查错）
from torchtyping import TensorType, patch_typeguard
# 导入 typeguard 用于自动检查函数参数是否符合指定的类型
from typeguard import typechecked

# 引入我们自定义的数据模块 Batch（表示一个批次的训练数据）
from datamodule import Batch

# 开启类型检查补丁，必须在 @typechecked 装饰器使用前调用
patch_typeguard()  

# ==========================================
# 核心函数定义
# ==========================================
# @typechecked 会在此函数被调用时自动检查参数类型是否跟定义的一样
@typechecked
def generate_hpp(
    tmax: TensorType,             # 时间的最大跨度，即观察窗口的长度，例如[0, T]中的T
    n_sequences: int,             # 需要生成的序列数量（批次中的样本数）
    x_n: Batch,                   # 获取参考数据批次，主要为了拿它里面的离散条件特征（如天气、星期几）
    time_segments : int,          # 时间切片的数量，将一整段[0, tmax]切分成多少段
    intensity: Union[TensorType, None] = None,  # 强度参数𝜆，表示平均单位时间发生的事件数量。如果不传则默认为1
) -> Batch:
    """
    Generate a batch of sequences from a homogeneous Poisson process on [0,T].
    生成一批在[0, T]时间区间上的齐次泊松过程（HPP）事件序列。
    
    外行理解：就像模拟随机客流。假如我们知道一家店平均每天来10个客人，
    这个函数的作用就是“随机掷骰子”，为你生成10天里，客人们走进店里的具体随机时间点。

    Returns
    -------
    Batch
        返回包含生成的随机事件时间和对应条件的 Batch 对象
    """
    # 拿到 tmax 所在的设备（比如是在 CPU 还是在显卡 GPU 运算，后续数据要对齐）
    device = tmax.device
    
    # 如果没有指定强度强度参数（intensity），则默认每条序列的强度都是 1
    if intensity is None:
        intensity = torch.ones(n_sequences, device=device)

    # 1. 确定每个序列里到底发生多少次事件
    # 根据泊松分布采样，获取每个序列最终会发生的事件总数（n_samples）
    # 期望值 = 时间跨度(tmax) * 发生强度(intensity)
    n_samples = torch.poisson(tmax * intensity)
    
    # 找出这一批次里面事件数量最多的那个，并加 1 留做余地，作为矩阵的最大列数
    max_samples = int(torch.max(n_samples).item()) + 1

    # 2. 随机生成事件发生的具体时间点
    # 在 0 到 1 之间随机采取 max_samples 个点，然后乘以 tmax 扩展到全局时间范围
    # 注意：均匀泊松过程特性允许我们在时间轴上直接进行均匀分布(rand)采样
    times = torch.rand((n_sequences, max_samples), device=device) * tmax
    
    # 3. 创建掩码（Mask）从而忽略掉填充多余的时间点
    # 假装最大长度有10次，但这行序列实际只采出了3次，所以前3次记为True，其余记为False
    mask = (
        torch.arange(0, max_samples, device=device)[None, :]
        < n_samples[:, None]
    )
    # 将被忽略位置（False）的事件时间置为0
    times = times * mask

    # 4. 把原参考数据(x_n)中的额外条件信息强行拷贝对齐给新生成的时间点
    # 外行理解：原数据记录了"早上天气晴"，生成了早上发生的时间点后，要把"天气晴"的标签复制过去。
    
    # 处理第 1 种外部条件
    condition1 = torch.ones_like(times) # 先创建一个全是1的矩阵备用
    for index in range(1, time_segments + 1): # 遍历每一个时间切片
        cond_window1 = times >= index - 1     # 找出落在这个切片左边界的事件
        cond_window2 = times < index          # 找出落在这个切片右边界的事件
        combined_condition = cond_window1 & cond_window2 # 取交集，也就是找出现处于该时间段的所有事件
        # 把处于该时间段的所有事件，赋上参考数据在这个时间段的条件值
        condition1 = torch.where(combined_condition, x_n.condition1_indicator[:, [index - 1]], condition1)
    # 将填充无效位置的条件置为0，同时将数据类型转换为整数
    condition1 = (condition1 * mask).to(torch.int64)

    # 处理第 2 种外部条件（逻辑跟第1种完全一样）
    condition2 = torch.ones_like(times)
    for index in range(1, time_segments + 1):
        cond_window1 = times >= index - 1
        cond_window2 = times < index 
        combined_condition = cond_window1 & cond_window2
        condition2 = torch.where(combined_condition, x_n.condition2_indicator[:, [index - 1]], condition2)
    condition2 = (condition2 * mask).to(torch.int64)

    # 处理第 3 种外部条件
    condition3 = torch.ones_like(times)
    for index in range(1, time_segments + 1):
        cond_window1 = times >= index - 1
        cond_window2 = times < index 
        combined_condition = cond_window1 & cond_window2
        condition3 = torch.where(combined_condition, x_n.condition3_indicator[:, [index - 1]], condition3)
    condition3 = (condition3 * mask).to(torch.int64)

    # 处理第 4 种外部条件
    condition4 = torch.ones_like(times)
    for index in range(1, time_segments + 1):
        cond_window1 = times >= index - 1
        cond_window2 = times < index 
        combined_condition = cond_window1 & cond_window2
        condition4 = torch.where(combined_condition, x_n.condition4_indicator[:, [index - 1]], condition4)
    condition4 = (condition4 * mask).to(torch.int64)

    # 处理第 5 种外部条件
    condition5 = torch.ones_like(times)
    for index in range(1, time_segments + 1):
        cond_window1 = times >= index - 1
        cond_window2 = times < index 
        combined_condition = cond_window1 & cond_window2
        condition5 = torch.where(combined_condition, x_n.condition5_indicator[:, [index - 1]], condition5)
    condition5 = (condition5 * mask).to(torch.int64)

    # 处理第 6 种外部条件
    condition6 = torch.ones_like(times)
    for index in range(1, time_segments + 1):
        cond_window1 = times >= index - 1
        cond_window2 = times < index 
        combined_condition = cond_window1 & cond_window2
        condition6 = torch.where(combined_condition, x_n.condition6_indicator[:, [index - 1]], condition6)
    condition6 = (condition6 * mask).to(torch.int64)

    # 断言（检查站）：确保根据逻辑挑选出来的有效事件数目能跟最开始随机抽样出的总数完美匹配，不对的话就报错
    assert (mask.sum(-1) == n_samples).all(), "wrong number of samples"
    
    # 5. 返回封装好的新型数据批次结构
    # 利用原生的数据格式构造函数封装所有信息，同时调用方法清理掉可能多余占位的无效数据列
    return Batch.remove_unnescessary_padding(
        time=times,                      # 随机出来的事件序列时间点
        condition1=condition1,           # 对齐好的各类外部条件组合...
        condition2=condition2,
        condition3=condition3,
        condition4=condition4,
        condition5=condition5,
        condition6=condition6,
        condition1_indicator=x_n.condition1_indicator, # 携带最原始的参考条件段
        condition2_indicator=x_n.condition2_indicator,
        condition3_indicator=x_n.condition3_indicator,
        condition4_indicator=x_n.condition4_indicator,
        condition5_indicator=x_n.condition5_indicator,
        condition6_indicator=x_n.condition6_indicator,
        mask=mask,                       # 区分数据里哪些是真人真事哪些是填补的假占位
        tmax=tmax,                       # 最大时间约束参数
        kept=None                        # 不需要保留其它额外参数
    )

"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/processes/hpp.py
========================================================================

1. 这个文件在做什么？
   - 它的主功能是模拟和生成基础的 **“随机事件流”**。
   - 机器在学习规律之前，往往需要拿一种“没有任何规律、彻底瞎猜”生成的基准数据去作为对比（或者作为扩散模型的起点）。
   - 在统计学中，这种最简单、彻底随大流的生成方式叫 **齐次泊松过程（HPP）**。

2. 怎么理解泊松过程（Poisson Process）？
   - 想想公交站等车。如果公交车时刻表错乱，全天彻底随机派车，但平均每天发100班车。
   - 我们这个算法就是第一步先掷一个复杂的色子抛出结论（比如今天实际上来了105班车 `n_samples`）。
   - 第二步在24小时里随意点 105 个点（`times`），就模拟出了这个随机流。

3. 为什么代码里还要费尽周折处理 condition1~6？
   - 真实业务里总是带有附加信息的。可能上午那段时间正好是“雨天”，下午是“晴天”。
   - 当程序在上午时间里随机洒下了几个事件时间点时，它必须同时给这些时间点“贴上”对应的“雨天”标签（这个动作对应代码里那6个看着像复制粘贴一样的 `condition = torch.where(...)` 循环块）。
   
4. 这在这个项目中起什么作用？
   - 基于扩散模型的策略，通常需要把真实、极度复杂的人类行为数据“模糊、退化”为这种全随机的HPP噪声。然后再倒过来训练模型如何从这团随机流里“恢复出”原本的精密人类行为轨迹。所以它是构建噪声世界的造雪机器。
========================================================================
"""
