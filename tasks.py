import pytorch_lightning as pl
import torch
import torch.nn as nn
from pathlib import Path
import wandb
import os 

import numpy as np
import matplotlib.pyplot as plt

from datamodule import Batch
from add_thin.metrics import (
    MMD,
    lengths_distribution_wasserstein_distance,
)
from evaluations.statistical_metrics import Get_Statistical_Metrics


class Tasks(pl.LightningModule):
    """
    【双主干结合大总管】：
    因为我们这个项目其实揉了两个截然不同的模型：
    1. TPP时间推演模型 (add_thin) - 处理几点几分做什么。
    2. 离散空间扩散模型 (discrete_diffusion) - 处理去哪玩、分类是啥。
    这类把两尊大佛请到一个供桌上，并且协调它们各自拿各自的总分、甚至分别配置怎么考试怎么学习的业务，都交给他。
    """
    def __init__(
        self,
        tpp_model,              # 第一尊大佛：连续时间扩散器
        discrete_diffusion,     # 第二尊大佛：离散事件变压器
        learning_rate1,         # 时间模型的领悟速度（学习率）
        learning_rate2,         # 空间模型的领悟速度
        weight_decay1: float = 0.0,
        weight_decay2: float = 0.0,
    ):
        super().__init__()
        # 把大佛装进名册，且忽略掉那些不该保存为超参数的神像本体
        self.save_hyperparameters(ignore=("tpp_model","discrete_diffusion"))

        self.weight_decay1 = weight_decay1
        self.weight_decay2 = weight_decay2
        self.learning_rate1 = learning_rate1
        self.learning_rate2 = learning_rate2
        self.tpp_model = tpp_model
        self.discrete_diffusion = discrete_diffusion
        
        # 准备一个常用的“交叉熵打分器”（看预测和事实对不对得上）供时间推演分支纠正存留点时使用
        self.classification_loss_func = nn.BCEWithLogitsLoss(reduction="none")

    @property
    def automatic_optimization(self) -> bool:
        """
        因为我们有两尊大佛，不能让外面的傻瓜循环一通乱改参数！
        所以强行关闭自动优化，宣告：“我要手工作业，两边分开各自优化”。
        """
        return False     

    def classification_loss(self, x_n_int_x_0, x_n: Batch):
        """
        为时间线推演算【分类保留损失 BCE】：
        这部分负责评价“这个被噪音毁掉的连续时间点，老子到底要不要把它拔除还是存留？”的辨别精准度。
        外行理解：考的是在一堆混淆视听的假时间点里，认出真兄弟的能力。
        """
        # 压扁阵列并挑出重点蒙版的关注区
        x_n_int_x_0 = x_n_int_x_0.flatten()[x_n.mask.flatten()]
        target = x_n.kept.flatten()[x_n.mask.flatten()]
        # 用 二元交叉熵 核对答案给分
        loss = self.classification_loss_func(x_n_int_x_0, target.float())
        # 求出平均分
        loss = (loss).sum() / len(x_n)
        return loss

    def intensity_loss(self, log_prob_x_0):
        """
        为时间线推演算【强度负对数似然损失】：
        外行理解：考的是大环境概率地形预测得浪不浪漫（它预言的高峰是不是真实事件的爆发区）。取负号是因为我们希望损失越低越好，也就意味着这个事件越有大概率发生。
        """
        return -log_prob_x_0.mean()

    def get_loss(self, log_prob_x_0, x_n_int_x_0, x_n):
        """
        核算【时间推演大佛】的总成绩（分为判断力成绩和强度力成绩）
        """
        # 给巨大数值除以一下最大容量均摊
        intensity = self.intensity_loss(log_prob_x_0) / self.tpp_model.n_max
        classification = (
            self.classification_loss(x_n_int_x_0, x_n) / self.tpp_model.n_max
        )
        # 把两项罚单加一起变成总损失罚单
        loss = classification + intensity
        return loss, classification, intensity

    def step(self, batch, name):
        """
        单次统整算账流程（无论训练还是测试，都得拉出来走一转评出分）
        """
        # 第一步：把题卷(batch)递给时间模型大佛，它算出来一系列强度预言和辩别记号
        x_n_int_x_0, log_prob_x_0, x_n = self.tpp_model.forward(batch)

        # 第二步：把题卷递给空间离散模型大佛，让它算出变质还原的差异损失
        spatial_loss = self.discrete_diffusion.training_losses(batch).mean()

        # 第三步：让旁边的主考官拿第一步的结果去评定时间推演考差了多少分
        temporal_loss, classification, intensity = self.get_loss(
            log_prob_x_0, x_n_int_x_0, x_n
        )

        # 第四步：好家伙，双杀，加一块就是大统考总罚分数！
        total_loss = spatial_loss + temporal_loss
        
        # 第五步：挨个给日志报点（把各种偏科项的详细失分记录用喇叭喊出去：Wandb 记录）
        self.log(
            f"{name}/total_loss",
            total_loss.detach().item(),
            batch_size=batch.batch_size,
        )
        self.log(
            f"{name}/spatial_loss",
            spatial_loss.detach().item(),
            batch_size=batch.batch_size,
        )

        self.log(
            f"{name}/temporal_loss",
            temporal_loss.detach().item(),
            batch_size=batch.batch_size,
        )

        self.log(
            f"{name}/log-likelihood",
            intensity.detach().item(),
            batch_size=batch.batch_size,
        )
        if classification is not None:
            self.log(
                f"{name}/BCE",
                classification.detach().item(),
                batch_size=batch.batch_size,
            )
        return temporal_loss,spatial_loss,total_loss

    def configure_optimizers(self):
        """
        准备【改正错题套路和教训领悟方案】。
        这就定义了拿什么类型的铅笔擦去修正各自的脑子，以及进步不了了之后该怎么调整。
        """
        # 第一尊大佛吃 Adam 这个优化器丹药体系
        optimizer1 = torch.optim.Adam(self.tpp_model.parameters(), lr=self.learning_rate1, weight_decay=self.weight_decay1,)
        
        # 第二尊空间大佛吃 AdamW（带权衰减更好的版本） 丹药体系
        optimizer2 = torch.optim.AdamW(self.discrete_diffusion.parameters(), lr=self.learning_rate2, weight_decay=self.weight_decay2)

        # 这下面是【降维打击机制】：如果考了 1000 次了分数再也没降下来，
        # 就意味着他碰到了认知瓶颈，我们马上削减 5% 的学习速率（factor=0.95），让他把步子迈小点扣细节。
        lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer1, factor=0.95, patience=1000, verbose=True
        )

        lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2, factor=0.95, patience=1000, verbose=True
        )
        
        # 将这两套体系分别打包好交付出去上路
        return ({"optimizer": optimizer1,"lr_scheduler": {"scheduler": lr_scheduler1}},
                {"optimizer": optimizer2,"lr_scheduler": {"scheduler": lr_scheduler2}})


class DensityEstimation(Tasks):
    """
    真正干起苦力训练的跑腿执行专员（继承了上面那位配置统筹大总管的基础功能）
    这里明确描述了从拿卷子、考试算分、拍脑袋反省、更正大脑思想的全流程环。
    """
    def __init__(
        self, tpp_model, discrete_diffusion, learning_rate1, learning_rate2, weight_decay1, weight_decay2
    ):
        super().__init__(
            tpp_model, discrete_diffusion, learning_rate1, learning_rate2, weight_decay1, weight_decay2
        )

    def training_step(self, batch, batch_idx):
        """【一轮闭环的惩戒反省室】"""
        # 第一步：进屋子拿卷子，挨一顿批，算出失分明细表
        loss_temporal, loss_spatial, loss_all = self.step(batch, "train")
        
        # 第二步：把两位专属改错辅导员（优化器）叫进来
        opt1, opt2 = self.optimizers()

        # 第三步：专门拿时间大佛的错题，清空他原有的抱怨，沿着神经往回抽骨髓查错（反向传播），强行逼着他吃药更正思维网络！
        opt1.zero_grad()              
        self.manual_backward(loss_temporal)  
        opt1.step()                  

        # 第四步：再单独拿空间离散大佛的失分表，单独狠狠打他一巴掌并逼他反省更新自己的大脑突触！
        opt2.zero_grad()              
        self.manual_backward(loss_spatial)    
        opt2.step()                  

        # 第五步：把这事发给统筹看，看看要不要削减他们接下来的骄傲和步子跨度（降学习率）
        sch1, sch2 = self.lr_schedulers()
        sch1.step(loss_temporal)
        sch2.step(loss_spatial)

        # 第六步：把两人的凄惨失分榜单贴在控制台显耀的大屏幕上供人类参考！
        self.log_dict({"t_loss": loss_temporal, "s_loss": loss_spatial}, prog_bar=True)


    def test_step(self, batch, batch_idx):
        # 预留空位：真实测试评测等跑代码的占位。通常有别的大批量外挂接管测试指标
        pass

"""
========================================================================
【给外行新手的通俗讲解】 -- tasks.py (神经枢纽协调总管)
========================================================================

1. 这个文件到底是干嘛的？它和 train.py 有什么区别？
   - `train.py` 像是“校长”，它决定在哪天军训、用什么配置本、记录到云端。
   - `tasks.py` 才是真正的“班主任和教务处”。
   - 这个项目比较奇葩的一点是它有两个并行的重型大脑（一文一武）：
      - 武将：时间极值大佛 `add_thin`（负责连续时间）
      - 文将：离散空间大佛 `discrete_diffusion`（负责在哪里、什么事类别）
   - 他们两者没法混为一谈算一个分，甚至用的考试方法、错题反思路线（优化器、反倒速度）完全不同！所以这个 `Tasks` 类就是专干这苦力活的。
   
2. 它是怎么教导这两个大佛变聪明的？（请看 `training_step`）
   - 首先扔进去一张答卷(batch)。
   - 它在 `step()` 分离让两人各自写各自负责的一半题目，并利用 `get_loss` 算出各自极其惨烈的丢分(比如时间分类错了罚5分、被污染的地方选错罚8分)。
   - 随后进入玄妙的 `manual_backward()` 倒打钉耙：
     - 把时间模型摁住，清空它脑子里的浆糊(`opt1.zero_grad()`)，顺着它做错时间的神经线通上高压电溯源找是哪个脑细胞犯傻了(`manual_backward`)，强制调整这个小缺心眼神经 (`opt1.step()`)。
     - 然后再对离散模型如法炮制。
   - 两者相互平级、独立改造，确保你管时间我管空间，两者同步朝着完美逼近！
========================================================================
"""
