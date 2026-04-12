import random
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(config: DictConfig) -> np.random.Generator:
    """
    为 PyTorch、NumPy 和 Python 内置的 random 模块统一设置随机种子。
    
    细节：由于 wandb (一个深度学习看板) 会把很大的整数种子强行转换成浮点数，
    导致精度丢失、种子被破坏。所以我们在配置字典 (config) 里把大整数种子存成字符串。
    """
    # 尝试从配置中读出种子并转换成整型数字
    big_seed = int(config.seed) if config.seed is not None else None
    
    # 调用下面写的 manual_seed 函数去播撒种子，并拿回由系统生成的绝对根种子以及 NumPy 生成器
    big_seed, rng = manual_seed(big_seed)
    
    # 强制把它转换回字符串再还回 config 里，防止被 wandb 破坏
    config.seed = str(big_seed)
    
    # 扔回这个随机数生成器提供给后续代码使用
    return rng


def manual_seed(seed: Optional[int]):
    """
    手动为所有能产生随机数的引擎发牌（给 RNGs 设定种子），
    而且聪明地避免了不让所有的引擎使用同一个死板的相同初始种子。
    """
    # ====== 第一阶段：创建种子序列 ======
    # 采用 NumPy 的 SeedSequence 功能，它就像是随机数的“总代理”。可以由一个主种子向下派生出无数个互不干扰的子种子。
    # 这样做的好处是避免了不同库之间（比如 Python 自带的和 NumPy 的）因为用了同一个初始种子而产生诡异的状态重合。
    # 如果没传种子进来（None），它就会向操作系统要一些高质量的绝对随机噪音（entropy）当老祖宗。
    # SeedSequence(seed) = 造一个总种子
    # .spawn(N) = 从总种子生 N 个独立子种子

    root_ss = np.random.SeedSequence(seed)

    # 算一下我们需要发多少套牌（多少个生成器）。基础需要4套。
    num_rngs = 4
    # 如果电脑有显卡计算 (CUDA)，那么有几张显卡就得多加几套给显卡用
    if torch.cuda.is_available():
        num_rngs += torch.cuda.device_count()
        
    # 从 root_ss 这个总代理那，切分出对应数量互相独立的子序列（SeedSequence）
    # std_ss：用于 Python 自带的 random 模块
    # np_ss：相当于 NumPy 内部新的局部随机引擎
    # npg_ss：用于 NumPy 最老旧全局共享的引擎
    # pt_ss：用于 PyTorch CPU 的引擎
    # cuda_ss：这会是一个列表，为所有的 PyTorch GPU 引擎发牌
    std_ss, np_ss, npg_ss, pt_ss, *cuda_ss = root_ss.spawn(num_rngs)

    # ====== 第二阶段：为 Python 内置打基础 ======
    # Python 的 random 模块底层使用的是“梅森旋转算法（Mersenne Twister）”。
    # 它需要极长的状态才能完美启动（需要 624 个 32位 整数的状态字）。
    # 所以我们从 std_ss 里抠出 624 个状态字，变成字节流喂倒给 random.seed。
    random.seed(std_ss.generate_state(624).tobytes()) 

    # ====== 第三阶段：为 NumPy 阵营打基础 ======
    # 虽然现在写代码不推荐使用带全局副作用的 np.random.seed，
    # 但保不齐底下哪个第三方老旧库会用到。为了防止它们不随机，我们还是得给全局种子赋个值（安全兜底fallback）。
    np.random.seed(int(npg_ss.generate_state(1, np.uint32)))

    # ====== 第四阶段：为 PyTorch 阵营打基础 ======
    # 有显卡的时候...
    if torch.cuda.is_available():

        # 先包一个延迟执行的函数...
        def lazy_seed_cuda():
            # 循环遍历你的每个显卡（0号、1号...），分别用上面扣出的那一小段 cuda_ss 给它发专用的种子牌
            for i in range(torch.cuda.device_count()):
                device_seed = int(cuda_ss[i].generate_state(1, np.uint64))
                torch.cuda.default_generators[i].manual_seed(device_seed)

        # 接下来立刻给 PyTorch 的 CPU 全局生成器喂一颗 64位 的种子
        torch.random.default_generator.manual_seed(
            int(pt_ss.generate_state(1, np.uint64))
        )
        
        # 将上面包好的那个专门打理 GPU 种子的函数，放入 PyTorch 的冷藏器中（lazy call 等需要用显卡运算了它才开机）
        torch.cuda._lazy_call(lazy_seed_cuda)

    # ====== 第五阶段：安全构建现代局部随机引擎 ======
    # 新时代的 NumPy 推荐使用自己单独随身携带的局部生成器 rng。咱们也构建一个。
    rng = np.random.default_rng(np_ss)

    # 把刚才老祖宗决定用的最终真理种子数字，和最新锐的局部生成器一起返回给上家。
    seed = root_ss.entropy
    return seed, rng


"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/utils/seed.py
========================================================================

1. 这个文件的根本目的是什么？
   - 保证 AI 训练的“可重复性”和真正的“随机公正性”。
   - 在机器学习里，“随机”分为两种：如果完全不可控，别人就无法复现你的实验结果；但如果各个库凑巧用了同一个种子，生成的随机数一模一样，模型学出的就全是重复的偏见。
   
2. 它是怎么操作的？
   - 这个代码像是一个严谨的发牌荷官。它有一个最大的“主控制数字”（也就是配置文件里指定的种子值）。
   - 它用极度科学的 NumPy SeedSequence，为 Python的基础库、NumPy的全局、PyTorch的CPU、甚至机箱里的每一张独立显卡，各自分发一套截然不同且不可互相推测的“副种子”。
   - 这样既保证了所有组件一开始的起步是一起联动的（你给个主种子我就能每次一样），又能保证它们各司其职计算时产生的随机数绝不串号冲突。
========================================================================
"""
