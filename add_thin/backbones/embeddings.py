import numpy as np
import torch
import torch.nn as nn


class NyquistFrequencyEmbedding(nn.Module):
    """
    一个很神仙的“时间流”向“高维电波频率”转换加工器。
    它采用正弦余弦编码来表示抽象的时标度（timesteps）。
    """
    def __init__(self, dim: int, timesteps: int | float) -> None:
        super().__init__()

        # 要求嵌入维度(特征)必须是个能被 2 整除的偶数，因为正余弦刚好是一对儿。
        assert dim % 2 == 0

        # T 是总长观察范围的最大尽头
        T = timesteps
        # k 是要凑出几组波纹（每一组出两个正副信号，合起来就是总维度数）
        k = dim // 2

        # 下面几句纯属是科学大佬们做实验尝试出的骚操作（结合信号学Nyquist频带和数学黄金分割）：
        # 根据系统学 Nyquist（奈奎斯特）采样定律频带频率折半极限
        nyquist_frequency = T / 2

        # 引入数学上很玄幻的黄金分割率（1.618...）
        # 用于特意保证创造出“无理数频率波动”，确保各个时间点编码不会呈现机械式的规律重合
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # 在对数空间均匀撒下我们想要利用的多种频段宽广波纹
        frequencies = np.geomspace(
            1 / 8, nyquist_frequency / (2 * golden_ratio), num=k
        )

        # 把这组拨出来的神仙频段，一份用来算 Sin，另一份悄悄平移（+ pi/2，就变成了 Cos）
        scale = np.repeat(2 * np.pi * frequencies / timesteps, 2)
        bias = np.tile(np.array([0, np.pi / 2]), k)

        # 把它以系统死参数（不跑模型训练修改它）的身份注册固定在网络里作为刻度尺的基底
        self.register_buffer(
            "scale",
            torch.from_numpy(scale.astype(np.float32)),
            persistent=False, # 意思是它仅仅存在运行期，它不用像模型的脑细胞一样随着存盘被持久化刻下来
        )
        self.register_buffer(
            "bias", torch.from_numpy(bias.astype(np.float32)), persistent=False
        )

    def forward(self, t) -> torch.Tensor:
        # 当某一个具体的事件时间 ｔ（例如发生这事是早上 7:34分 用小数标榜）来了。
        # 就用 t 乘上各种频率的神仙波长基石再加上偏移最后取 sin 正弦波值输出。
        # 这一秒，普通的“时间标”就被奇妙地变成了“交响乐一样有粗有细多种波纹振幅”的维度。
        return torch.addcmul(self.bias, self.scale, t[..., None]).sin()

"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/backbones/embeddings.py
========================================================================

1. 什么是 Embedding (嵌入)？
   - 机器是不懂自然界里的“时间”这是啥意思的。你给个数字 10(点钟)，它也就是个普通大小等于 10 的死板数字。
   - 但是时间的内涵可太丰富了！比如时间是周期性的呀：过了24小时就是第二天同一时刻了！所以 1 点钟 和 25 点钟在人类意义和行为规律上很相近，如果直接丢数字给模型它就觉得 25 比 1 大好多。
   - Embedding 即：找个方法，把这些单一的数字强行翻译为有着复杂内涵和周期性格的一串高维度多数字组成的阵列特征密码本。让机器能感受时间里的波峰、波谷等意境。

2. 为什么要搞傅里叶、Nyquist和正弦波？
   - 这个代码灵感来自于大名鼎鼎的 Transformer (里面用的就是三角函数做位置编码器)。
   - 因为人类的发生行为高度契合正弦曲线波动：你看，人经常“每天早上都起床洗漱”、“每周一次周末下馆子”。这种非常强烈的周期重复感。
   - 通过这段算法的强行渲染转化，原本只是普通光点的时间值，摇身变成了如同几根不同大小弹簧拧在一起的复杂形状，极其完美契合模型用这些“发条波形”去解开人类起承转合的行为密码。
========================================================================
"""
