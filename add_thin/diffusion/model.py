import math
import torch
import torch.nn as nn

from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from datamodule import Batch
from add_thin.backbones.cnn import CNNSeqEmb
from add_thin.backbones.embeddings import NyquistFrequencyEmbedding
from torch.nn import TransformerEncoderLayer, TransformerEncoder 
from add_thin.processes.hpp import generate_hpp
from add_thin.diffusion.utils import betas_for_alpha_bar

patch_typeguard()

class ConditionEmbeddingModel(nn.Module):
    """
    负责处理整个“序列级别”的外部条件的嵌入转化模型（比如将星期几、什么天气转化成神经网络能看懂的向量）。
    """
    def __init__(
        self,
        cond_token_num = 200,    # 系统所允许的所有类型的条件字典大小（比如有200个天气词汇）
        emb_dims = 256,          # 转化成电脑语言后这个语言包的长度大小
        num_condition_types = 6, # 有几种不同的条件类型（例如：季节、天气、周末与否...）
        max_position_embeddings = 3000 # 最大位置容纳量，序列最长不会超过三千步长
    ):
        super().__init__()
        self.token_num = cond_token_num
        self.emb_dims = emb_dims
        self.num_condition_types = num_condition_types
        self.max_position_embeddings = max_position_embeddings
        
        # Encoder负责：查字典，把数字编号的条件（id）替换成连续向量
        self.encoder = nn.Embedding(self.token_num, self.emb_dims)
        
        # 负责把好几个零散的条件向量揉在了一起，压缩梳理一番
        self.input_up_proj =  nn.Sequential(
            nn.Linear(self.num_condition_types * self.emb_dims, self.emb_dims),
            nn.ReLU(),
            nn.Linear(self.emb_dims, self.emb_dims)
        )
        
        # 位置编码（告诉模型这些事件的先后排序感念）
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.emb_dims)
        self.register_buffer("position_ids", torch.arange(self.max_position_embeddings).expand((1, -1)))
        
        # 使用 Transformer 注意力机制层来深度混合所有环境条件之间的关系
        encoder_layer = TransformerEncoderLayer(d_model=self.emb_dims, nhead=4, batch_first=True)
        self.condition_transformers = TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, batch):
        # 1. 拿到了真实的队列长度
        seq_length = batch.condition1_indicator.size(1)
        # 2. 从仓库里切下正好等身长的一段序列排序编号（0,1,2,3...）
        position_ids = self.position_ids[:, : seq_length ]
        
        # 3. 查字典，把这六种不同的离散条件，各自变成多维度向量
        condition1_embeddings = self.encoder(batch.condition1_indicator)
        condition2_embeddings = self.encoder(batch.condition2_indicator)
        condition3_embeddings = self.encoder(batch.condition3_indicator)
        condition4_embeddings = self.encoder(batch.condition4_indicator)
        condition5_embeddings = self.encoder(batch.condition5_indicator)
        condition6_embeddings = self.encoder(batch.condition6_indicator)

        # 4. 把它们全部并排粘在一起，再送进揉面机（input_up_proj）进行压缩特征提取
        condition_embeddings = self.input_up_proj(torch.cat([condition1_embeddings,condition2_embeddings,\
            condition3_embeddings,condition4_embeddings,condition5_embeddings,condition6_embeddings],dim=-1))

        # 5. 上面揉好的面团 加上 当前排位的位置信息，融合为带时空感的特征
        condition_embeddings = self.position_embeddings(position_ids) + condition_embeddings

        # 6. 送进 Transformer 经过 3 层的注意力深加工学习
        encoded_conditions = self.condition_transformers(condition_embeddings)
        
        return encoded_conditions


@typechecked
class DiffusionModel(nn.Module):
    """
    扩散模型的核心基座。不管之后是做时间预测还是类别预测，计算加噪/去噪过程的表格核心参数都在这里。

    Parameters
    ----------
    steps : int, optional
        模型执行扩散的总步数（默认 1000 步）
    """
    def __init__(self, steps: int = 1000) -> None:
        super().__init__()
        self.steps = steps

        # 1. 算出按照“余弦算法”变质进度表，获得每一步需要加多少噪 Beta
        beta = betas_for_alpha_bar(
            steps,
            lambda n: math.cos((n + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

        # 2. 算出一个核心变量 α（阿尔法）= 1 - beta
        alpha = 1 - beta
        
        # 以及累计连乘积 α拔 (alpha_cumprod)，代表从最初干净状态走到现在还能保留了多少真正的特征
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        
        # 存到模型身上（随着模型存盘一起保存，但不需要模型梯度去学习）
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

        # 3. 为点过程扩散中特有的 AddThin（加减事件）技术算专属变质比例
        # 分别计算各种复杂的后验概率
        # add_remove：即加事件又减事件时的比例
        add_remove = (1 - self.alpha_cumprod)[:-1] * beta[1:]
        # alpha_x0_kept：计算在完全干净的样本里，保留下的样本所占概率
        alpha_x0_kept = (self.alpha_cumprod[:-1] - self.alpha_cumprod[1:]) / (
            1 - self.alpha_cumprod[1:]
        )
        # alpha_xn_kept：在带噪的阶段 xn 被保留下的概率
        alpha_xn_kept = (
            (self.alpha - self.alpha_cumprod) / (1 - self.alpha_cumprod)
        )[1:]

        # 把这些复杂的预备数据公式结果全都登记在案备用
        self.register_buffer("alpha_x0_kept", alpha_x0_kept)
        self.register_buffer("alpha_xn_kept", alpha_xn_kept)
        self.register_buffer("add_remove", add_remove)


@typechecked
class AddThin(DiffusionModel):
    """
    实现 AddThin 算法（全称：“Add and Thin”针对时间点过程专门设计的新型扩散方式）。
    就是这个！最核心的时间序列推演大脑。
    """
    def __init__(
        self,
        classifier_model,     # 负责预测要哪些点(分类器，左脑)
        intensity_model,      # 负责估计在这个环境下本来该发生多强烈事件(强度器，右脑)
        max_time: float,      # 观察这帮人活动序列的最广时间界限
        n_max: int = 100,     # 一段活动流序列最多容纳的事件上限
        steps: int = 100,     # 扩散洗牌的步数
        hidden_dims: int = 128,   # 特征维度容量
        emb_dim: int = 32,    # 特征编码容量
        encoder_layer: int = 4,   # 堆叠层数的深度
        kernel_size: int = 16, # CNN的核大小，也就是看的窗口跨度
        num_condition_types: int = 6, # 条件标签有几种
        time_segments: int = 24, # 时间一天切片数量（如24小时）
    ) -> None:
        super().__init__(steps)
        self.classifier_model = classifier_model
        self.intensity_model = intensity_model

        self.n_max = n_max
        self.cond_emb_size = hidden_dims
        self.num_condition_types = num_condition_types
        # 给单纯“具体发生那一下”时环境挂带条件用的查字典编解码器
        self.event_condition_encoder = nn.Embedding(200, self.cond_emb_size)
        self.time_segments = time_segments
        
        # 初始化处理纯时间序列用到的工具（往下看 set_encoders）
        self.set_encoders(
            hidden_dims=hidden_dims,
            max_time=max_time,
            emb_dim=emb_dim,
            encoder_layer=encoder_layer,
            kernel_size=kernel_size,
            steps=steps,
        )

        # 给整段序列的宏观条件做上下文编码
        self.seq_condition_encoder = ConditionEmbeddingModel(num_condition_types = self.num_condition_types,emb_dims=self.cond_emb_size) 

    def set_encoders(
        self,
        hidden_dims: int,
        max_time: float,
        emb_dim: int,
        encoder_layer: int,
        kernel_size: int,
        steps: int,
    ) -> None:
        """配置并启动所需要的子加工编码车间"""
        
        # 1. 把事件发生的时间点翻译成机器懂的多维波动。利用傅里叶Nyquist感官编码。（对应信号处理）
        position_emb = NyquistFrequencyEmbedding(  
            dim=emb_dim // 2, timesteps=max_time
        )
        self.time_encoder = nn.Sequential(position_emb) 

        # 2. 把当前模型走到了第几个扩散步骤也翻译成为机器懂的波动信号（防迷路）
        position_emb = NyquistFrequencyEmbedding(dim=emb_dim, timesteps=steps)
        self.diffusion_time_encoder = nn.Sequential(
            position_emb,
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # 3. 将整套时间流进行图像级别的一维卷叠提取规律
        self.sequence_encoder = CNNSeqEmb(
            emb_layer=encoder_layer,
            input_dim=hidden_dims,
            emb_dims=hidden_dims,
            kernel_size=kernel_size,
        )

    def set_condition(
            self, batch
    ) -> TensorType["batch", "seq_level_cond_embedding"]:
        # 加工出能代表过去整段时间状态的特征精华，取 Transformer 输出在最尾端的序列结果
        seq_level_cond_emb = self.seq_condition_encoder(batch)
        return seq_level_cond_emb[:, -1, :]


    def compute_emb(
        self, n: TensorType[torch.long, "batch"], x_n: Batch
    ):
        """
        外行理解这步：就是把外面的“大白话（事件时间、处在扩散第几步、所携带条件）”
        翻译成大脑皮层用来放电沟通的高维度“向量串”。
        """
        B, L = x_n.batch_size, x_n.seq_len

        # 翻译步骤进度 n 
        dif_time_emb = self.diffusion_time_encoder(n) 

        # 翻译这件事是几点发生的，以及它离上件事隔了多久(tau)
        time_emb = self.time_encoder(
            torch.cat([x_n.time.unsqueeze(-1), x_n.tau.unsqueeze(-1)], dim=-1)
        ).reshape(B, L, -1) 

        # 提取整条序列时间的顺滑上下文影响规律（CNN）
        event_emb = self.sequence_encoder(time_emb) 
        # 把填来凑数位的空白数据打上遮罩不理会
        event_emb = event_emb * x_n.mask[..., None]

        # 提取各个条件字典的单点映射内涵
        condition1_cond_emb = self.event_condition_encoder(x_n.condition1)
        condition2_cond_emb = self.event_condition_encoder(x_n.condition2)
        condition3_cond_emb = self.event_condition_encoder(x_n.condition3)
        condition4_cond_emb = self.event_condition_encoder(x_n.condition4)
        condition5_cond_emb = self.event_condition_encoder(x_n.condition5)
        condition6_cond_emb = self.event_condition_encoder(x_n.condition6)

        # 强行打成合集包
        event_level_cond_emb = torch.cat([condition1_cond_emb,condition2_cond_emb,condition3_cond_emb,condition4_cond_emb,condition5_cond_emb,condition6_cond_emb],dim=-1)

        # 再取一下前面计算出来的全局宏观状态
        seq_level_cond_emb = self.set_condition(x_n) 
        
        # 将翻译好的这全部 5 种东西发给接下来用于运算
        return (
            dif_time_emb, 
            time_emb, 
            event_emb, 
            event_level_cond_emb,
            seq_level_cond_emb, 
        )


    def get_n(self, shape, device, min=None, max=None) -> TensorType[int]: 
        """
        这很简单：为了全方位训练 AI 恢复数据的能力。闭上眼在这 100 步或者 1000 步过程里，
        随机点名抽个具体的步数（n步）。考考你现在这步退化成了什么样，且怎么复原。
        """
        if min is None or max is None:
            min = 0
            max = self.steps
        return torch.randint(
            min,
            max,
            size=shape,
            device=device,
            dtype=torch.long,
        )

    def noise(
        self, x_0: Batch, n: TensorType[torch.long, "batch"]
    ) -> Tuple[Batch, Batch]:
        """
        【弄脏数据的过程 - 也是训练的第一步】
        外行理解：拿到完美的数据(x_0)后，咱们按部就班把它通过 `Thin`和 `Add HPP`技术把它变得越来越乱、越来越充满噪音。
        让 AI 一会看着被整得乱七八糟的 `x_n`，被逼想办法把它整理还原。
        """
        # 第一招：随机“漏听”点（Thin out x_0），根据当前的衰退进度表 alpha，直接删减掉部分真实的事件点
        x_0_kept, x_0_thinned = x_0.thin(alpha=self.alpha_cumprod[n])

        # 第二招：塞入“胡编乱造”的随机点，拿 HPP 机器凭空凭概率给你瞎造几个假事件叠加上去
        hpp = generate_hpp(
            tmax=x_0.tmax,
            x_n=x_0,
            n_sequences=len(x_0),
            time_segments = self.time_segments,
            intensity=1 - self.alpha_cumprod[n],
        )
        # 最后，剩下的少数真话跟大多数胡编的假话掺和一块，这就是今天的考题 x_n ！
        x_n = x_0_kept.add_events(hpp) 

        return x_n, x_0_thinned 

    def forward(
        self, x_0: Batch
    ):
        """
        【模型训练的主入口】：当启动一次训练轮回，就会进入这套动作。
        在知道 x_0 (现实里明确发生了什么) 的情况下，考模型怎么从迷雾中分辨现实。
        """
        # 1. 裁判随机抽签决定今天考把数据弄脏到什么地步（抽第 n 步阶段）
        n = self.get_n(
            min=0,
            max=self.steps,
            shape=(len(x_0),),
            device=x_0.time.device,
        ) 
        
        # 2. 从真实的干净数据开始加噪变乱作伪装（获得考卷 x_n ）和被删减的记录 x_0_thin
        x_n, x_0_thin = self.noise(x_0=x_0, n=n)

        # 3. 读考卷：把它经过大量复杂的向量特征编码，搞清每段细节含义
        (dif_time_emb, time_emb, event_emb, event_level_cond_emb, seq_level_cond_emb) = self.compute_emb(n=n, x_n=x_n) 

        # 4. 做大题：让模型里专门搞分类的同学出马去挑：从这混杂的考卷 x_n 里，把真正的 x_0(真事件) 分辨挑选回来。
        x_n_and_x_0_logits = self.classifier_model(
            dif_time_emb=dif_time_emb,
            time_emb=time_emb,
            event_emb=event_emb,
            cond_emb=event_level_cond_emb,
        ) 

        # 5. 做测算题：让专门算强度的同学出马测绘：去估测那些我们在第一段 `Thin` 阶段故意遗漏的空白角落里，本来该发生多大规模的事？（似然求导）
        log_like_x_0 = self.intensity_model.log_likelihood(
            event_emb=event_emb,
            seq_cond_emb=seq_level_cond_emb,
            dif_time_emb=dif_time_emb,
            x_0=x_0_thin,
            x_n=x_n,
        )

        return x_n_and_x_0_logits, log_like_x_0, x_n

    def sample(self, n_samples: int, x_n: Batch, tmax) -> Batch:
        """
        【神笔马良环节：无中生有】这模型学成出师后，你让它在空旷的时间里给个推演结果。
        它就是从纯粹的一盘散沙逐渐倒推理出规律的行为。
        """
        # 一开始，就起手捏一团毫无逻辑可言的完全瞎写泊松点 x_N 当初稿
        x_N = generate_hpp(tmax=tmax, n_sequences=n_samples, x_n=x_n, time_segments = self.time_segments,) 
        x_n_1 = x_N

        # 有了那团瞎字后，模型像名医号脉：一步一步倒着解！第99步、第98步...一直倒退到第一手
        for n_int in range(self.steps - 1, 0, -1):
            n = torch.full(
                (n_samples,), n_int, device=tmax.device, dtype=torch.long
            )
            # 持续召唤下面的去噪还原函数帮忙打扫清理这一步假点，添上推演出的真点
            x_n_1 = self.sample_posterior(x_n=x_n_1, n=n)
        
        # 退无可退时，最后发出了干净终极版本 x_0！
        n = torch.full(
            (n_samples,), n_int - 1, device=tmax.device, dtype=torch.long
        )
        x_0, _, _, _ = self.sample_x_0(n=n, x_n=x_n_1) 

        return x_0

    def sample_x_0(
        self, n: TensorType[int], x_n: Batch
    ):
        """
        生成过程的核心拆解机。给模型一批带病数据模型x_n，试图找回最原汁原味的x_0。
        外行理解：这里使用了双模型共同会诊制。
        """
        # 先翻译出所有病情细节指标
        (
            dif_time_emb,
            time_emb,
            event_emb,
            event_level_cond_emb,
            seq_level_cond_emb,
        ) = self.compute_emb(n=n, x_n=x_n)

        # 会诊医生 A (负责强度测算的家伙)：去凭空算算有哪些应该发生但是这次没列进病历簿的事情
        sampled_x_0 = self.intensity_model.sample(
            event_emb=event_emb,
            seq_cond_emb=seq_level_cond_emb,
            dif_time_emb=dif_time_emb,
            n_samples=1,
            x_n=x_n,
        ) 

        # 会诊医生 B (负责挑细节挑毛病的家伙)：从现有乱糟糟的清单中，挨个指认真伪挑出来（打 Logit 模型分然后概率裁剪Thin）。
        x_n_and_x_0_logits = self.classifier_model(
            dif_time_emb=dif_time_emb, time_emb=time_emb, event_emb=event_emb, cond_emb=event_level_cond_emb
        )
        # 用挑选机器裁剪出了医生B挑选出的真的（x_0_kept）和剔除出来的伪造病（x_0_not_kept）
        classified_x_0, classified_not_x_0 = x_n.thin(
            alpha=x_n_and_x_0_logits.sigmoid()
        )
        
        # 把两名医生各自的努力和结果叠加起来反馈（被挑剩下的真加上被凭空猜的真，这就是总还原的真）。
        return (
            classified_x_0.add_events(sampled_x_0),
            classified_x_0, 
            sampled_x_0, 
            classified_not_x_0,
        )

    def sample_posterior(self, x_n: Batch, n: TensorType[int]) -> Batch:
        """
        为了在推理回退时非常严格严谨地做到步步为营，不能直接从全乱猜到全干净，
        我们需要数学公式（后验概率分布）来帮我们倒退一小步回到 x_n-1。
        这牵扯到极多的严丝合缝的概率学操作。
        """
        # 第一发，用上面两个医生拼了老命先往极端猜测出终极真解是啥样（但这个猜测现在可能不稳拿）
        _, classified_x_0, sampled_x_0, classified_not_x_0 = self.sample_x_0(
            n=n, x_n=x_n,
        ) 

        # 接下来是一套极其讲究的神奇混调过程：
        # C项操作：把医生A凭空猜出来的真事情概率再稍作压缩和精简
        x_0_kept, _ = sampled_x_0.thin(alpha=self.alpha_x0_kept[n - 1])

        # D项操作：重新引入微量的一些随机干扰假数据（为了过程足够物理足够平滑）
        hpp = generate_hpp(
            tmax=x_n.tmax,
            x_n=x_n,
            n_sequences=x_n.batch_size,
            time_segments = self.time_segments,
            intensity=self.add_remove[n - 1],
        )

        # E项操作：从原本医生认为已经不是真实成分的那部分废料里，按后验倒推表捞点回来。
        x_n_kept, _ = classified_not_x_0.thin(alpha=self.alpha_xn_kept[n - 1])

        # 最后，把各大家：医生挑的真料、再揉杂出来的废料、重新引入的微干扰素等强行融合。
        # 所得结果就是刚好比这步强一点，离完美还差一点的 x_n-1 (走完一步了)！
        x_n_1 = (
            classified_x_0.add_events(hpp)
            .add_events(x_n_kept)
            .add_events(x_0_kept)
        )
        return x_n_1

"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/diffusion/model.py
========================================================================

1. 这个大几百行的代码到底是干嘛的？
   - 它是整个项目的【中心核电站】或者叫【推演大本营】。
   - 所有在 `configs.py` 里设定的所谓“高深模型”，落足点就是这个名叫 `AddThin` 的巨大类。它掌控了把人的行为时间用魔法预测出来的最高权限系统。

2. 时间点扩散是怎么运行的原理？（大白话拆解）：
   - 想象我们研究犯罪记录（什么时间发案）。这就像画卷上有几个散落精准标定的案发红点。
   - 【前向破坏(Forward)】：模型在训练阶段拿到案发记录。每一步都利用它特殊的魔法，随机擦掉你红点中的一部分（这叫 Thin），再同时随机在画卷别的地上乱洒几滴没有逻辑的苍蝇屎墨水红光点（这叫 Add）。这个加减折腾100次，案发现卷变成满屏乱溅没任何线索的全域马赛克（纯噪声）。
   - 【模型学习(Model)】：模型在这一百次毁画卷过程中偷窥，去训练它左脑和右脑（分类器：看穿假点；强度器：猜出由于毁坏导致我们遗漏真点的地方哪里该补）。
   - 【后项推演(Sample)】：等模型训练好去考场时，只给它发一张全马赛克的乱飞溅白纸。它就反复驱动那一双左右手回退一步、调和一步、挑选一步、补齐一步...像神笔马良一样倒回一百次，最终居然能洗出像清明上河图一样有着优美起伏作案规律周期的案件作案图。这就是 Diffusion 的终极魅力！
========================================================================
"""
