from typing import Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import warnings
from torch.distributions import MixtureSameFamily
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from datamodule import Batch
# 导入刚刚我们讲过的，特制的斩去了下半身的单向正态分布
from add_thin.distributions.densities import DISTRIBUTIONS

patch_typeguard()

# TPP 专门管：连续时间上随机突发的离散事件；
# λ(t) 看完所有历史、结合当前空窗等待时长；
# 它就是：此时此刻，下一秒马上发生新事件的活跃强度 / 危险冲动
@typechecked
class MixtureIntensity(nn.Module):
    """
    负责计算“强度（Intensity）”的核心模型。（相当于负责大局估计的右脑）
    外行理解：所谓混合强度（Mixture），
    就是拿 10 个（默认 n_components=10）不同的钟形曲线给它们配上不同的高低权重，
    然后混叠叠加在一起。这种混音器一样的做法，理论上能极其完美地逼近并描绘出现实中任何奇形怪状的人类复杂行为概率密度。
    """

    def __init__(
        self,
        n_components: int = 10,     # 混合的组数（拿几条基础的曲线去堆叠混合）
        embedding_size: int = 128,  # 每条进来的特征线宽度
        distribution: str = "normal", # 底层小零件采用什么曲线（我们这里用特制的正态）
        time_segments: int = 24,    # 一天被切成 24 块
    ) -> None:
        super().__init__()

        # 保险起见，确定请求的小零件确实在我们提供的可选花名册里
        assert (
            distribution in DISTRIBUTIONS.keys()
        ), f"{distribution} not in {DISTRIBUTIONS.keys()}"
        
        # Softplus 是种曲线平滑激活器，确保输出绝对为正（因为曲线混合时的权重不可能占个负数比例）
        self.w_activation = torch.nn.Softplus()
        # 挂载基础发球机零件
        self.distribution = DISTRIBUTIONS[distribution]

        self.n_components = n_components
        
        # 构建它的专属思考器官（MLP）：
        # 它拿 3 倍的特征段进去，最后精准吐出 3倍的 n_components。
        # 因为这吐出来的结果会被等分劈成三份，分别去给 10根线设定：
        # 中心点定在哪(mu)、线条多胖多瘦(sigma)、这根线多重要(weight)。
        self.mlp = nn.Sequential(
            nn.Linear(3 * embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 3 * n_components),
        )
        
        # 拒绝采样法（用来推演真实发生点的土方法）内部初始试错倍率。
        self.rejections_sample_multiple = 2
        self.time_segments = time_segments 

    def get_intensity_parameters(
        self,
        x_n: Batch,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        seq_cond_emb: TensorType[float, "batch", "seq_cond_embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
    ) -> Tuple[TensorType, TensorType, TensorType]:
        """
        经过深思熟虑，输出那 10 条小曲线的详细物理搭建参数图纸。
        """

        # 计算到底目前有效发生了多少次事件
        n_events = x_n.mask.sum(-1)
        # 求特征均值。把这整个事件序列上的所有事压缩成“一个浓缩的精华大特征”代表整个序列。
        # sum(1) 把整条包含了 L 个事件的历史记录揉碎压缩成了一团精华（原本的 L 维度被抹平）。 
        # 因此，MLP 吐出来的 10 条线图纸（如 location），形状都是紧凑的 [批次, 10]。 
        # 也就是只有一套全局总纲。
        seq_emb = event_emb.sum(1) / torch.clamp(n_events[..., None], min=1)
        seq_cond_emb = seq_cond_emb.reshape(seq_emb.shape[0], -1)

        # 送进大脑计算区，得出 30 个长数值（假设默认 10 条）
        parameters = self.mlp(torch.cat([seq_emb, dif_time_emb, seq_cond_emb], dim=-1))
        
        # 用屠龙刀把长数据切分成均等的三份，各领着 10个数值：每条线的位置、宽窄、比重。
        return torch.split(
            parameters,
            [self.n_components, self.n_components, self.n_components],
            dim=-1,
        )

    def get_distribution(
        self,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        seq_cond_emb: TensorType[float, "batch", "seq_cond_embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
        x_n: Batch,
        L,
    ):
        """
        用刚才造好的那些物理参数，生生把这个包含了 10 条线的【终极概率混合大怪兽】分布函数给具现化构建出来。
        返回的这个分布对象，以后用来算概率、抽样，就像调系统内置的工具一样丝滑。
        """
        location, scale, weight = self.get_intensity_parameters(
            x_n=x_n,
            event_emb=event_emb,
            seq_cond_emb=seq_cond_emb,
            dif_time_emb=dif_time_emb,
        ) 

        # 对权重数值使用 Softplus 平滑处理，让它们统统变为正数且有连贯性
        weight = self.w_activation(weight)  
        
        # 顺手把所有线权重积加起来，再乘以此前实际发生过的次数，这叫“累积大强度 （CIF）”。
        cumulative_intensity = (weight).sum(-1) * (x_n.mask.sum(-1) + 1) 
        
        # 构建混合骨架分布（决定投哪颗骰子）
        mixture_dist = D.Categorical(probs=weight.unsqueeze(1).repeat(1, L, 1))

        # 构建底层的具体每一根正态分布实体线
        component_dist = self.distribution(
            location.unsqueeze(1).repeat(1, L, 1),
            scale.unsqueeze(1).repeat(1, L, 1), 
        )
        # 将骨架和实体两块泥巴揉成一块，诞生【终极概率混合机 (MixtureSameFamily)】
        return (
            MixtureSameFamily(mixture_dist, component_dist),
            cumulative_intensity,
        )

    def log_likelihood(
        self,
        x_0: Batch,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        seq_cond_emb: TensorType[float, "batch", "seq_cond_embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
        x_n: Batch,
    ) -> TensorType[float, "batch"]:
        """
        评估能力大考场。
        算出当机器费劲心思搞出了上述那个概率函数地形图之后。用它来复原当初那些被故意切掉弄没的真事件(x_0)时，发生的概率密度究竟有多高（似然评估）？
        这个分越高，证明机器推演出来的地形图越符合自然界中那些真实的事情发生规律。
        """
        # 第一步：老规矩，先铺开生成那个包含了10条曲线重叠的地形图（密度）和总体数。
        density, cif = self.get_distribution(
            event_emb=event_emb,
            seq_cond_emb=seq_cond_emb,
            dif_time_emb=dif_time_emb,
            x_n=x_n, 
            L=x_0.seq_len, 
        )

        # 把丢失的那些真记录发生的时间给放缩到 0-1 之间。
        x = x_0.time / x_0.tmax 

        # 计算落在我们这套牛逼地形图上，那些点对应的对数概率海拔高度（Log-intensity）
        log_intensity = (
            (density.log_prob(x) + torch.log(cif)[..., None]) * x_0.mask
        ).sum(-1) 

        # 通过积分原理算分母用于归一化
        cdf = density.cdf(torch.ones_like(x)).mean(1) 
        cif = cif * cdf  

        # 返回积分后的终极可能性得分
        return log_intensity - cif

    def sample(
        self,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        seq_cond_emb: TensorType[float, "batch", "seq_cond_embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
        n_samples: int,
        x_n: Batch,
    ) -> Batch:
        """
        【拒绝采样法（Rejection Sampling）】专门用来无中生有的。
        现在地形图有了。我要求机器现在立刻给我按这个规律自己凭空洒一批新点下来生造几条新事，怎么搞？
        """
        tmax = x_n.tmax
        # 拿图纸
        density, cif = self.get_distribution(
            event_emb=event_emb,
            seq_cond_emb=seq_cond_emb,
            dif_time_emb=dif_time_emb,
            x_n=x_n,
            L=1,
        )

        # 第一步：先凭直觉大概定一下接下来我要造的事情，大概平均得撒出几颗来比较合理。
        count_distribution = D.Poisson(
            rate=cif
            * density.cdf(
                torch.ones(n_samples, 1, device=event_emb.device)
            ).squeeze()
        )
        sequence_len = (
            count_distribution.sample((n_samples,)).squeeze()
        ).long()

        max_seq_len = sequence_len.max()

        # 第二步：土办法开启！狂撒。
        # 什么叫拒绝采样外行化解释：你想在一张复杂的中国地图的不规则多边形湖面上撒网捞鱼，你没办法精准计算这个湖在哪，那你就简单粗暴先把整个中国国土撒满大网，然后把落在没湖地方的网统统抛弃（或者重撒），剩下的就是你要的！
        while True:
            # 拿到机器撒的点
            times = (
                density.sample(
                    ((max_seq_len + 1) * self.rejections_sample_multiple,) 
                )
                .squeeze(-1)
                .T
                * tmax
            )

            # 过滤1：这破点落在合理时间区间 [0, tmax] 里面没？出轨的不要，留下正经的（inside）。
            inside = torch.logical_and(times <= tmax, times >= 0)
            
            # 把正经的挑出来排前面
            sort_idx = torch.argsort(
                inside.int(), stable=True, descending=True, dim=-1
            )
            inside = torch.take_along_dim(inside, sort_idx, dim=-1)[
                :, :max_seq_len
            ]
            times = torch.take_along_dim(times, sort_idx, dim=-1)[
                :, :max_seq_len
            ]

            # 过滤2：强行裁剪掉多余撒过头的点
            mask = (
                torch.arange(0, times.shape[-1], device=times.device)[None, :]
                < sequence_len[:, None]
            )
            mask = mask * inside

            # 如果这批最终剩下来的确实凑够了刚才定下来的撒料数目，那大功告成跳出黑洞循环。
            if (mask.sum(-1) == sequence_len).all():
                break
            else:
                # 倒霉，符合要求的太少了没够指标。那就把网做大一倍（试错倍率+1），重新扔一遍继续等。
                self.rejections_sample_multiple += 1
                warnings.warn(
                    f"""
拒绝采样试错倍率已经翻倍到了 {self.rejections_sample_multiple}。因为有效数据点太少了。
""".strip()
                )
                
        # 第三阶段：下面这一长串的复制粘贴，依然是我们之前的老戏码：
        # 根据我们盲洒下去的点此时对应的钟表时间刻度，给它强行把天气和环境等情况赋值绑定上。
        condition1 = torch.ones_like(times)
        for index in range(1,self.time_segments+1):
            cond_window1 = times>= index-1
            cond_window2 = times< index 
            combined_condition = cond_window1 & cond_window2
            condition1 = torch.where(combined_condition ,x_n.condition1_indicator[:,[index-1]],condition1)
        condition1 = (condition1 * mask).to(torch.int64)

        condition2 = torch.ones_like(times)
        for index in range(1,self.time_segments+1):
            cond_window1 = times>= index-1
            cond_window2 = times< index 
            combined_condition = cond_window1 & cond_window2
            condition2 = torch.where(combined_condition ,x_n.condition2_indicator[:,[index-1]],condition2)
        condition2 = (condition2 * mask).to(torch.int64)

        condition3 = torch.ones_like(times)
        for index in range(1,self.time_segments+1):
            cond_window1 = times>= index-1
            cond_window2 = times< index 
            combined_condition = cond_window1 & cond_window2
            condition3 = torch.where(combined_condition ,x_n.condition3_indicator[:,[index-1]],condition3)
        condition3 = (condition3 * mask).to(torch.int64)

        condition4 = torch.ones_like(times)
        for index in range(1,self.time_segments+1):
            cond_window1 = times>= index-1
            cond_window2 = times< index 
            combined_condition = cond_window1 & cond_window2
            condition4 = torch.where(combined_condition ,x_n.condition4_indicator[:,[index-1]],condition4)
        condition4 = (condition4 * mask).to(torch.int64)

        condition5 = torch.ones_like(times)
        for index in range(1,self.time_segments+1):
            cond_window1 = times>= index-1
            cond_window2 = times< index 
            combined_condition = cond_window1 & cond_window2
            condition5 = torch.where(combined_condition ,x_n.condition5_indicator[:,[index-1]],condition5)
        condition5 = (condition5 * mask).to(torch.int64)

        condition6 = torch.ones_like(times)
        for index in range(1,self.time_segments+1):
            cond_window1 = times>= index-1
            cond_window2 = times< index 
            combined_condition = cond_window1 & cond_window2
            condition6 = torch.where(combined_condition ,x_n.condition6_indicator[:,[index-1]],condition6)
        condition6 = (condition6 * mask).to(torch.int64)

        # 把空白处的污点用0清理赶紧
        times = times * mask
        
        # 将我们历经生造出来的，非常完美的遵循高深人类统计学概率密度的一条全新序列返回发出去！
        return Batch.remove_unnescessary_padding(
            time=times,
            condition1=condition1, 
            condition2=condition2,
            condition3=condition3,
            condition4=condition4,
            condition5=condition5,
            condition6=condition6,
            condition1_indicator=x_n.condition1_indicator,
            condition2_indicator=x_n.condition2_indicator,
            condition3_indicator=x_n.condition3_indicator,
            condition4_indicator=x_n.condition4_indicator,
            condition5_indicator=x_n.condition5_indicator,
            condition6_indicator=x_n.condition6_indicator,
            mask=mask, 
            tmax=tmax, 
            kept=None
        )

"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/distributions/intensities.py
========================================================================

1. 什么是 混合分布 和 强度（Intensity） ？
   - “时间点过程”就像是预测心率图上的心跳声何时发出。
   - 强度就是概率密码。如果 8 点这个时间点的在图表海平面上代表高度为 50，那就意味着人在此时出发作案的概率超高。
   - 问题是一整个周期有高高低低极其复杂的多个波峰波谷组合在一起，你拿单纯一个曲线方程写不出来。那怎么办？
   - 设计出 `n_components` 比如 10 个独立的小人（10条独立小正态分布）。通过神经网络指挥它们有的长得胖，有的长得尖锐，有的挤在上半夜有的在下半夜。这就成功拼装出了一张任何形状都能拟合的变态大海拔图。这就是“混合模型”。
   
2. Log-likelihood 在干啥？
   - 相当于一个量天尺。模型建好以后。如果把历史真数据的时间丢进这个海面方程里核对。
   - 如果发生过的事，正好处于方程中的高海拔（说明本就高概率发生）。那总得分就飙高。
   - 但若系统给低谷的位置发生了真事，得分就垮。通过它机器就能知道怎样调整自己那十个小人去逼近真理。
   
3. Rejection Sampling （拒绝抽样法）为什么看着这么低效暴力？
   - 因为从这种用各种小人硬堆叠出来的奇形怪状的地形表面上，想算出精准的发生位置是不可能的（没有积分反解）。那怎么办？
   - 把大网洒满全图，然后根据这儿的海拔筛下去。不在范围里就直接一脚踹走抛掉重来。这个听起来虽然暴力，但在高级的蒙特卡洛算法范畴，这是能够极度精密重现统计学特征的神级技术！
========================================================================
"""
