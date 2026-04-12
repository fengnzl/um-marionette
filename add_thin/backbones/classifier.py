import torch
import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

# 为 Python 的动态类型检查库（typeguard）打补丁， 
# 让它能够正确理解深度学习矩阵（Tensor）的复杂结构要求（比如尺寸形状的限制）
patch_typeguard()  

@typechecked
class PointClassifier(nn.Module):
    """
    点分类器。
    在扩散模型把数据变乱的过程中，它的任务是：看着一团夹杂了真事件和假造事件的序列（x_n），
    负责把真正曾发生过的事件（属于 x_0 的部分）给“挑出来、指认出来”。

    Parameters:
    ----------
    hidden_dims : int
        大脑隐含层神经元的肥胖程度（处理复杂度） 即模型的脑力大小 越大模型学得越细，能处理更复杂的任务（图像、语音、预测、分类等）
    layer : int
        大脑的思考深度（神经网络层数）层数多 = 思考深 = 会推理、会抽象
        以上两者参数都是越大，越聪明，越吃算力
    """
    def __init__(
        self,
        hidden_dims: int,
        layer: int,
    ) -> None:
        super().__init__()
        
        # 1. 计算输入的特征包含多长：
        # 根据我们前面看到的代码，输入包含了：扩散进度时间（1）+ 发生时间（1）+ 事件类型特征（1）+ 条件特征组（6种）。
        # 所以进入分类器之前，有 9 股特征汇聚，总长度就是 9 倍的 hidden_dims
        input_dim = 9 * hidden_dims 

        # 2. 搭建判决神经网络 (MLP - 多层感知机) 维度从 input_dim 变为 hidden_dims
        # 第一层：把超长的外部特征压缩进入隐含层思考空间，并加一个 ReLU 激活函数（让模型懂得“转弯”不去钻牛角尖死算直线规律）
        layers = [nn.Linear(input_dim, hidden_dims), nn.ReLU()]
        
        # 中间的层：不断地把思考结果加深提炼
        for _ in range(layer - 1):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.ReLU())
            
        # 最后一层：不管你想了多少，最后只许给出一个“打分值（logit）”出来，代表你觉得这个事件有多大的概率是真的。
        # 在 PyTorch 中，把一个 3D 张量扔进线性层，它只会“原封不动地保留前两维”，
        # 只拿最后一维（宽）去做矩阵乘积。所以长宽高里的“高和长 (批次和L)”都保留了下来，只有“宽”被压缩成了 1。
        layers.append(nn.Linear(hidden_dims, 1))
        
        # 把这些层按顺序排好，包装成一个整体的工作流水线
        self.model = nn.Sequential(*layers)

    def forward(
        self,
        dif_time_emb: TensorType[float, "batch", "embedding"], # 代表扩散进度，比如现在是毁到了第 50 步
        time_emb: TensorType[float, "batch", "sequence", "time_emb"], # 事件发声的时间自身特征
        event_emb: TensorType["batch", "sequence", "embedding"], # 提取过的事件上下文大局观特征
        cond_emb: TensorType["batch", "sequence", "event_cond_embedding"], # 外界条件特征（地点天气等）
    ) -> TensorType[float, "batch", "sequence"]:
        """
        前向计算开始：就是模型真正在干活时的全流程。
        """
        # 取出时间特征的三维尺码 (批次大小， 序列里面有几个事件 L，特征大小)
        _, L, _ = time_emb.shape
        
        # 将各路神仙特征强行拼凑在一起：
        # 注意 dif_time_emb 这个是指扩散第几步，它对目前这一整条序列 L 里的所有事件都是共用的。
        # 因此，要用 repeat(1, L, 1) 给每个事件都发一份同样的“当前身处第几步”的情报。
        # nn.Linear 属于单线程瞎子，它在同时给这 L 个事件打分时是完全彼此孤立的。
        # 如果不把属于全局大环境的“当前是第 50 步”强行拷贝并塞进每一个孤立事件的口袋里，
        # 那这 L 个事件自己是根本无从得知现在的外部进展的
        x = torch.cat(
            [
                time_emb,                              
                event_emb,                             
                dif_time_emb.unsqueeze(1).repeat(1, L, 1), 
                cond_emb,                              
            ],
            dim=-1,                                    # 沿着最后一个特征维度进行拼接缝合
        )

        # 把这长长一串特征送进刚才搭好的神经网络流水线里面算分。
        # 得出的结果本来是 [批次, L, 1] 形态，但尾巴那个 1 有点多余，用 squeeze(-1) 挤掉它。
        # 对单独的事件点进行打分，而不是给整个时间线进行打分
        logits = self.model(x).squeeze(-1)
        
        # 返回每个点最后得出的“它是真事件的自信得分”
        return logits

"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/backbones/classifier.py
========================================================================

1. 什么是 Backbone (主干网络)？什么又是 Classifier？
   - 主干网络就相当于是人类的“视神经”。专门负责把看到的东西抽炼出关键信息。
   - 这个 Classifier 就像是海关里的查验员。
   - 在我们模型推演的“神仙打架”里，这名查验员拿着各种护照、证件材料（拼接特征 `x`），职责就是在一大群夹带着假乱入的旅客（污染序列 `x_n`）里，凭借自己的神经网络经验，揪出那些真的原住民（真事件 `x_0`）。

2. 它是怎么运算的？
   - 它就是一个非常朴素经典的 MLP (多层全链接层神经网络)。
   - 原理就像在层层递进的长廊里有好多道门。
   - 外边的线索挤进门廊，通过不同的加权转化和过滤碰撞（ReLU 非线性），最好凝结成一个纯粹的想法：这个时间点发生的事，我打分 8.5（极高概率是真的）。
========================================================================
"""
