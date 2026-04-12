#!/usr/bin/env python
import faulthandler
import logging
import warnings

import hydra
import torch
import wandb

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from configs import (
    instantiate_datamodule,
    instantiate_model,
    instantiate_task,
)
from add_thin.utils import (
    WandbModelCheckpoint,
    WandbSummaries,
    filter_device_available,
    get_logger,
    print_config,
    print_exceptions,
    set_seed,
)

def get_callbacks(config):
    """
    配置训练过程中的“贴身护卫”（Callbacks）。
    外行理解：就像是在马拉松比赛中，设定好哪些时候要拍照留念（保存模型）、
    哪些时候要把成绩写到榜单上（Wandb日志）、甚至如果跑不动了直接叫停比赛（EarlyStopping）。
    """
    # 监控标准：监控什么？默认找损失最小的 (mode="min")
    monitor = {"monitor": None, "mode": "min"}
    callbacks = [
        # 负责在每次记录数据时，自动提取关键的总结性信息（比如 Loss 最低多少），并上传到云端 Wandb 看板
        WandbSummaries(**monitor),
        
        # 负责在训练过程中自动保存模型权重。
        # save_last=True：永远保存最后一刻的模型。
        # filename="{epoch}"：保存的文件名会带上这是第几个纪元。
        WandbModelCheckpoint(
            save_last=True,
            save_top_k=0,
            filename="{epoch}",
            **monitor,
        ),
        
        # 进度条展示。负责在终端控制台里画出一个漂亮的 TQDM 进度条。
        # refresh_rate=1：表示每次迭代都刷新视图，让你看着爽。
        TQDMProgressBar(refresh_rate=1),
    ]
    
    # 【早停机制】：如果模型在连续 patience 个纪元里一直没有进步（考分不涨），
    # 直接暴力打断并结束训练，避免模型“走火入魔”死记硬背，也省电费和显卡算力时间。
    if config.early_stopping is not None:
        stopper = EarlyStopping(
            patience=int(config.early_stopping), # 容忍它多少次不进步
            min_delta=0,                         # 至少进步多少才算数
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    return callbacks


# 开启内存崩溃追踪错误打印，防止死得不明不白
faulthandler.enable(all_threads=False)

# 过滤掉一些无关紧要的、我们已经知道的警告信息，防止刷屏
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)
warnings.filterwarnings(
    "ignore",
    "The dataloader, [^,]+, does not have many workers",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    filter_device_available
)
log = get_logger()

# 利用 Hydra 这个配置大管家，从 config/train.yaml 里把厚厚的配置字典抽调出来
@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    # 【固定随机种子】：为了让实验每次跑出来的结果都是一样的，方便排错
    rng = set_seed(config)
    
    # 强制要求 PyTorch 使用确定性（Deterministic）算法计算，杜绝底层硬件的算数随机性
    # 保证同一代码、同一输入，每次运行结果完全一模一样。
    torch.use_deterministic_algorithms(True)

    # 解决 Hydra 字典配置里的插值问题（比如有个字段引用了另一个字段，这里给它算清楚绝对值）
    OmegaConf.resolve(config)

    # 把这巨长的配置表打印在屏幕上
    print_config(config)
    
    # 【连接到云端看板】：初始化 Wandb 实验追踪系统
    wandb.init(
      entity=config.entity,       # 云端账户名
      project=config.project,     # 项目名称，比如 "Marionette"
      group=config.group,         # 这是哪一组实验
      name=config.name,           # 实验名称
      resume="allow",             # 允许断点续连
      id=config.id,               # 运行ID号
      mode=config.mode,           # 在线/离线模式
      dir=config.run_dir,         # 本地存日志的文件夹
      anonymous="must",           
    )
    
    # 把当前确定的这套配置单持久化存下来
    OmegaConf.save(config, wandb.run.dir + "/config_hydra.yaml")
    log.info(wandb.run.dir)
    log.info("Loading data")

    # 【1. 拉取数据】：根据配置实例化数据列车（DataModule）并准备好打包数据
    datamodule = instantiate_datamodule(config.data)
    datamodule.prepare_data()

    log.info(config.data.name)

    log.info("Instantiating model")
    
    # 【2. 实例化双头怪兽】：从总部把“时间推演(AddThin)”和“空间离散扩散(DiffusionTransformer)”两尊大佛请出来
    tpp_model, discrete_diffusion = instantiate_model(config.model, datamodule)
    
    # 【3. 绑定训练任务】：用 Task 模块把这两尊大佛捏在一块，方便一起考试算总分
    task = instantiate_task(config.task, tpp_model, discrete_diffusion)

    # 声明用 Wandb 记录考试分数曲线
    logger = WandbLogger()

    log.info("Loading checkpoint")
    # 安装好那些早停、保存权重的“贴身护卫”
    callbacks = get_callbacks(config)

    log.info("Instantiating trainer")
    # 【4. 实例化大教官（Trainer）】：由 PyTorch-Lightning 提供的全自动包办教官，
    # 把它跟日志本、护卫队、硬件配置全部交给他。
    trainer: Trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log.info("Starting training!")
    # 【5. 一声令下，开始军训！】：教官拿着数据不断地喂给 task 考试、打分、优化。
    trainer.fit(task, datamodule=datamodule)

    # 如果配置要求顺带在测试集上测一波：
    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    # 收工，给云端看板说再见
    wandb.finish()


if __name__ == "__main__":
    main()

"""
========================================================================
【给外行新手的通俗讲解】 -- train.py (训练主引擎)
========================================================================

1. 这文件是个什么角色？
   - 它是整个 Marionette 项目的绝对总司令部（司令台）。我们之前写的各种复杂的物理模型、变压器、损失函数，全都被它统筹起来安排进行流水线式的批处理“学习”。
   
2. 里面有什么高级机制吗？为什么不手写 while 循环拿资料学习？
   - 这使用了 `PyTorch Lightning` 和 `Hydra` 两把大杀器。
   - `Hydra` 是个点餐大管家，它能把极其复杂的菜单（学习率、用哪种数据集、跑几次）从 `yaml` 写好的文件里读取并组合。
   - `Pytorch Lightning (Trainer)` 就是这个铁血大教官。有了它，你就不需要像写底层代码一样自己控制“数据放显卡”、“求导”、“清除临时记忆”。你只需要说：`trainer.fit(模型, 数据)`，它内部就会全自动安排这数万次的疯狂考试循环！
   
3. 万一机器学走火入魔了（即过拟合）或者停电了咋办？
   - 这里加了 `Callbacks` (类似贴身护卫保姆)。
   - **EarlyStopping (早停法)**: 如果它考了100次试，发现成绩死活上不去了，它不仅不会死撑，还会立刻鸣金收兵打断训练。
   - **WandbModelCheckpoint**: 每次考完试，它都会偷偷把模型此时此刻大脑脑电波的完美记忆存成一个大文件以备不测。
========================================================================
"""
