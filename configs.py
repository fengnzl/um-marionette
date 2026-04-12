# 导入内置库：处理文件和目录路径
from pathlib import Path
# 导入 omegaconf 库：用于处理和解析层级结构的配置参数（通常结合 Hydra 使用）
from omegaconf import DictConfig

# 导入项目中自定义的模块
# 导入数据模块（DataModule），用于统一管理数据集的加载、预处理和划分（如训练集、验证集）
from datamodule import DataModule
# 导入基于扩散模型（Diffusion）的时间点过程模型框架 AddThin
from add_thin.diffusion.model import AddThin
# 导入用于处理事件点特征的分类器（神经网络的主干结构之一）
from add_thin.backbones.classifier import PointClassifier
# 导入混合强度模型（MixtureIntensity），用于建模事件发生的时间概率分布（强度函数）
from add_thin.distributions.intensities import MixtureIntensity
# 导入任务模块（DensityEstimation），通常是一个对模型和训练流程进行包装的类（例如基于 PyTorch Lightning）
from tasks import DensityEstimation
# 导入离散扩散模型（DiffusionTransformer），专门用于处理离散的空间/类别特征
from discrete_diffusion.diffusion_transformer import DiffusionTransformer

def instantiate_datamodule(config: DictConfig):
    """
    实例化数据模块的工厂函数。
    作用：根据传入的配置对象（config）创建并返回一个 DataModule 实例，
    外行理解：就像是准备做菜前，根据"菜谱(config)"去买好并切好所有的"食材(数据)"。
    """
    return DataModule(
        config.root,                 # 数据集所在的根目录路径
        config.name,                 # 数据集的具体名称（例如：某城市的交通数据集或签到数据集）
        batch_size=config.batch_size, # 批次大小：每次喂给模型进行训练的数据量
    )

def instantiate_model(config: DictConfig, datamodule) -> AddThin:
    """
    实例化主模型的工厂函数。
    作用：初始化项目中的两个核心子模型：时间模型(TPP)和空间/类别模型(Discrete Diffusion)
    外行理解：这是在组装我们的人工智能大脑，分为处理"时间"的左脑和处理"空间位置"的右脑。
    """
    # ---------------- 1. 创建分类器（用于时间模型） ----------------
    # 本质是一个记忆压缩器 作用：提取和处理历史事件的特征，判断下一个时间点发生事件的概率走势。
    # 深度神经网络极其讨厌处理“长短不一”的东西。所以
    #  PointClassifier 的核心职责就是：用循环神经网络或者注意力机制（Transformer）
    # ，把这个用户乱七八糟的历史长度全部“压缩”成一个固定大小的纯数字向量包
    classifier = PointClassifier(
        hidden_dims=config.temporal_hidden_dims, # 神经网络的隐藏层维度（神经元的数量，控制模型的计算容量）
        layer=config.temporal_classifier_layer,  # 分类器的层数（网络有多深）
    )
    
    # ---------------- 2. 创建强度函数（用于时间模型） ----------------
    # 作用：强度函数(Intensity)是时间点过程的核心，预测未来某个时刻发生事件的具体强度。
    intensity = MixtureIntensity(
        n_components=config.temporal_mix_components, # 混合分布中包含的子成分数量（例如用几条正态曲线来拟合复杂的真实分布）
        embedding_size=config.temporal_hidden_dims,  # 将特征映射到的嵌入向量大小
        distribution="normal",                       # 基础概率分布类型：这里使用的是正态分布（Gaussian/Normal）
        time_segments=config.temporal_time_segments,  # 时间段的划分数量（例如将一天切分成几个小时段）
    )
    
    # ---------------- 3. 创建 TPP 模型（时间点过程模型） ----------------
    # 作用：组装前面创建的分类器和强度函数，构建出一个完整的基于连续时间和扩散模型的时间预测引擎。
    tpp_model = AddThin(
        classifier_model=classifier,                 # 传入步骤1建好的分类器
        intensity_model=intensity,                   # 传入步骤2建好的强度模型
        max_time=datamodule.train_data.tmax.item(),  # 时间序列的最大时间跨度（从训练数据中自动获取）
        steps=config.temporal_steps,                 # 时间扩散模型的扩散步数（类似做画时从模糊到清晰的涂抹次数）
        hidden_dims=config.temporal_hidden_dims,     # 隐藏层维度
        emb_dim=config.temporal_hidden_dims,         # 特征嵌入的学习维度
        encoder_layer=config.temporal_encoder_layer, # 编码器（Encoder）的层数
        n_max=datamodule.n_max,                      # 序列的最长事件数量（防止数据过长导致内存溢出）
        kernel_size=config.temporal_kernel_size,     # 卷积核或者时间窗口的感受野大小
        num_condition_types=config.num_condition_types, # 条件约束的种类数量（例如天气、星期几等外部条件）
        time_segments=config.temporal_time_segments,  # 时间分段数量与前面保持一致
    )
    
    # ---------------- 4. 创建离散扩散模型（空间/类别模型） ----------------
    # 作用：时间预测出来后，负责预测这个时间点发生的事情"在哪儿(POI)"或者"是什么类别(Category)"。
    discrete_diffusion = DiffusionTransformer(
        diffusion_step=config.spatial_hidden_dims,   # 这里命名可能有点绕，借用了配置里的维度作为扩散参数
        alpha_init_type='alpha1',                    # 扩散过程的噪声增加策略（alpha参数的初始化方案）
        type_classes=datamodule.num_category,        # 预测结果有多少种大类别（从数据模块动态获取）
        poi_classes=datamodule.num_poi,              # 预测结果有多少个具体的POI(兴趣点/地点)选项
        num_condition_types=config.num_condition_types, # 模型所接收的外部条件类型数
    )
    
    # 返回组合好的"时间模型"和"空间预测模型"
    return tpp_model, discrete_diffusion


def instantiate_task(config: DictConfig, tpp_model, discrete_diffusion):
    """
    实例化训练任务的工厂函数。
    作用：将模型、学习率等训练参数封装进一个统一的 Task 类里，方便给后续的训练框架（如 Pytorch Lightning）去统筹调用。
    外行理解：这就像拟定了一份"训练计划书"，告诉教练（训练框架）该怎么训练这两个大脑。
    """
    return DensityEstimation(
        tpp_model,                  # 传入时间模型 (TPP)
        discrete_diffusion,         # 传入空间离散扩散模型 (Diffusion)
        config.learning_rate1,      # 指定 TPP 模型的学习率（即时间模型每次试错后修正的步子有多大）
        config.learning_rate2,      # 指定 Diffusion 模型的学习率（即空间模型修正的步子有多大）
        config.weight_decay1,       # 指定 TPP 模型的权重衰减（一种防止模型"死记硬背"或过拟合的正则化手段）
        config.weight_decay2,       # 指定 Diffusion 模型的权重衰减
    )

