import torch
import torch.nn as nn
import warnings

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

@typechecked
class CNNSeqEmb(nn.Module):
    """
    负责提取和理解长序列上下文大局观的神经网络 (Dilated CNN - 扩张卷积网络)
    而且它使用了“环形填充(circular padding)”。
    """

    def __init__(
        self,
        emb_layer: int,       # 要堆几层剥片理解机（看多深）
        input_dim: int,       # 输入来的特征有多宽
        emb_dims: int,        # 自己想要输出的特征有多宽
        kernel_size: int = 16, # CNN的“眼睛”有多长：一次看并融合周边 16 个点
    ) -> None:
        super().__init__()
        
        # 巧妙之举：所谓扩张 (dilation) 参数
        # 意味着我们在长序列里看的时候，第一层是挨个紧挨着看，
        # 第二层是每隔 4 个跳着看，第三层是每隔 8 个跳着捞取。
        # 这也是为什么模型能不用超长计算就能快速顾及开头和结尾（扩大了感知视野）。
        dilation = [1, 4, 8, 16, 32, 64]

        layers = []
        for i in range(emb_layer):
            # 判断如果是第一层处理，就接外界的宽度输入，否则就接上层的宽度输出
            input_dim = input_dim if i == 0 else emb_dims
            
            # 使用列表拼装一层的内部构造，加入序列流水带
            layers.append(
                nn.Sequential(
                    *[
                        # 核心组件：一维卷积网络
                        nn.Conv1d(
                            input_dim,
                            emb_dims,
                            kernel_size,
                            padding="same",          # 强行保留原序列长度，不让卷积削短两头
                            padding_mode="circular", # “环形借用”：比如看开头的点，旁边缺少邻居，就偷偷跑到序列最尾巴把最后一件事搬过来假装邻居（这假设了长期规律可能带有周期旋转性）
                            dilation=dilation[i],    # 把上面定的扩张倍率设定进去
                        ),
                        # 分组归一化组件：把跑偏的太离谱的数值给勒回来，让模型稳定容易收敛学习
                        nn.GroupNorm(8, emb_dims), 
                    ]
                )
            )
            
        # 激励函数
        self.activation = nn.ReLU(inplace=True)
        # 最后画龙点睛的转化收尾处理层
        self.linear = nn.Linear(emb_dims, emb_dims)
        
        # 把攒好的列表用 PyTorch 官方专用的 ModuleList 保存挂载到自身。
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, x: TensorType[float, "batch", "sequence", "embedding"]
    ) -> TensorType[float, "batch", "sequence", "embedding"]:
        """运行处理这段系列"""
        
        # 如果喂进来的事件短于 30 个点（为了容错应对有些卷积由于扩张太大需要垫脚情况）
        if x.shape[1] < 30:
            x_before = x.shape[1]
            # 用全 0 的空白板强行把短数据续长到 30 大小
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        (x.shape[0], 30 - x.shape[1], x.shape[2]),
                        device=x.device,
                    ),
                ],
                dim=1,
            )
        else:
            x_before = None

        # 因为 PyTorch 的 Conv1d 标准脾气是喜欢把特征厚度放在第二维。
        # 所以必须 swapaxes(1, 2) 把本来在第二维的序列长度和第三位的特征互换排位。
        x = x.swapaxes(1, 2) 

        # 残差结构流转：进入刚刚装好的好几层网络去跑
        for l in self.layers:
            # 所谓残差（x = x + 处理后(x)），就是保留本来的自己加上新学到的东西。
            # 这种结构让特别深度的网络不会在层层叠叠中把自己一开始是谁给彻底遗忘。
            x = x + self.activation(l(x)) 
                                          
        # 算完了，把尺寸排位赶紧又倒退换回来
        x = x.swapaxes(1, 2) 

        # 如果刚才用 0 硬生生拉长凑数的，这会儿要无情地从原来接缝的位置“咔嚓”切掉不管。
        if x_before is not None:
            x = x[:, :x_before]
            
        # 走最后一道画龙点睛提取加工程序就送出厂
        return self.linear(x)

"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/backbones/cnn.py
========================================================================

1. CNN 不是处理图像的吗，这怎么在处理事件时间？
   - CNN 的本质原理是拿着一个放大镜（卷积核 kernel），每次罩住身边邻近小范围的要素去提炼出这堆东西表达的中心含义。
   - 在图里，放大镜罩住上下左右看（二维）。但在人类行为的序列流里（一维），放大镜沿着时间线“顺藤摸瓜”看：它把当前办的事、跟上个小时前两件办的事强行合在一起看，试图发现这里面是不是藏着“连桥规律”（比如买了爆米花，下一个事往往大概率就是看电影）。

2. Dilation (空洞扩张) 的魔幻之处？
   - 如果一件事对另一件事的影响不仅在它旁边，还在 3 天前，你怎么能看到3天前？
   - 笨办法：把放大镜买得无限巨大。但这会导致计算慢死，参数多到背不动。
   - 聪明办法（Dilation）：第一层，我们连着紧密看3眼。到了第四层，我们像睁开天眼一样，每隔 16 步盯上一眼。这种跳步法，用极少的计算量就能兼顾“洞察过去极其深远的影响规律”与“近在咫尺的相邻规律”。这非常具有创造性。
========================================================================
"""
