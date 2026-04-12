from copy import deepcopy
from pathlib import Path

# 导入 pytorch_lightning 框架（我们用来训练大脑的高级管家）
import pytorch_lightning as pl
import torch
# 导入 wandb 库（Weights & Biases），这是行内最火的一款机器学习训练在线监控面板工具
import wandb
# 导入模型检查点（ModelCheckpoint）插件工具，用于在训练时自动保存模型状态
from pytorch_lightning.callbacks import ModelCheckpoint


class WandbModelCheckpoint(ModelCheckpoint):
    """
    插件工具 1：在 Weights & Biases(wandb) 的运行目录里一并保存模型检查点。
    外行理解：这是一个自动“游戏存档”机。由于我们使用了在线监控仪表盘，我们希望每次游戏存档（权重模型保存）时，
    不仅仅存在本地，还能直接自动同步传到云端面板上与这次训练关联起来。
    """

    def __init__(self, **kwargs):
        # 找到正在运行的 wandb 专属云端挂载目录的路径
        run_dir = Path(wandb.run.dir)
        # 在那里新建一个叫做 "checkpoints" 的文件夹
        cp_dir = run_dir / "checkpoints"

        # 继续运行自带保存功能的传统启动器，只不过把保存的地址（dirpath）偷偷换成了刚刚新建的给云端同步用的文件夹
        super().__init__(**kwargs, dirpath=str(cp_dir))


class WandbSummaries(pl.Callback):
    """
    插件工具 2：对模型输出指标进行最好的摘要抓取。
    外行理解：如果模型训练过程像跑马拉松，它就是一个拿秒表的记录员。
    他会一直盯着哪次表现最好（比如准确率最高或者损耗最低），然后在比赛结束时，把那个最辉煌瞬间的一全套参数给大声喊出来广播到大屏上。
    """

    def __init__(self, monitor, mode):
        # 初始化这个记录员
        super().__init__()

        # 告诉记录员，你需要紧盯着哪个指标？（例如 loss 损失率、或 acc 准确率）
        self.monitor = monitor
        # 告诉记录员，是越大越好（"max"）还是越小越好（"min"）？
        self.mode = mode

        # 分别准备小本子，记录目前见过最牛的历史最好成绩
        self.best_metric = None
        # 以及取得那个成绩瞬间时，身上所带的所有其它附带指标大礼包
        self.best_metrics = None

        # 一个门栓，标记现在是不是正式的训练期
        self.ready = True

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Lightning 在正式训练前往往会跑几步用假数据做的“健康体检”（Sanity check），
        # 此时记录员应当闭上眼睛（ready设为False），不要把这个时候闹着玩出来的成绩当成好成绩。
        self.ready = False

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # 体检结束了，准备开始干活
        self.ready = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # 在走完一套验证集（期中考试）之后...
        if not self.ready:
            return

        # 收集系统所有已经上报了的考试成绩（metrics）
        metrics = trainer.logged_metrics
        # 如果老师交上来的成绩单里有我们要找的那个关键分（monitor）
        if self.monitor in metrics:
            metric = metrics[self.monitor]
            
            # 如果成绩是带着各种计算图的张量结构，把它变成纯数字
            if torch.is_tensor(metric):
                metric = metric.item()

            # 呼叫内部的 _better 函数来比一比，这成绩比我本子上的曾经最好成绩还要好吗？
            if self._better(metric):
                # 破纪录了！赶紧擦掉旧记录，写下新记录
                self.best_metric = metric
                # 把这一整套破纪录瞬间的所有分数合集都完好复印一份封存起来
                self.best_metrics = deepcopy(metrics)

        # 把这事报告给大屏
        self._update_summaries()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # 当全部好几十个世代（epoch）训练彻底打完收工时，最后再确认更新一次
        self._update_summaries()

    def state_dict(self):
        # 如果需要临时存放在硬盘上，打包他的记忆本子
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_metric": self.best_metric,
            "best_metrics": self.best_metrics,
        }

    def load_state_dict(self, state_dict):
        # 如果从云端或者硬盘读取旧进度，把记忆重新塞回他的头脑
        self.monitor = state_dict["monitor"]
        self.mode = state_dict["mode"]
        self.best_metric = state_dict["best_metric"]
        self.best_metrics = state_dict["best_metrics"]

    def _better(self, metric):
        # 内置判断成绩好坏的小逻辑
        if self.best_metric is None:
            # 如果这是跑出来第一笔成绩，那哪怕再差也是目前天下第一
            return True
        elif self.mode == "min" and metric < self.best_metric:
            # 如果要求越小越好（例如失误率），且今天的新失误率数字破了之前最低失误率的新低记录
            return True
        elif self.mode == "max" and metric > self.best_metric:
            # 如果要求越大越好（例如得分），且新成绩数字盖过了之前的老大
            return True
        else:
            # 除此以外就是没破记录
            return False

    def _update_summaries(self):
        # 这是一个强行写进 WandB 大屏的黑科技修正方法
        # 官方机制有时候会自作聪明停止刷新最后那一下，此处的代码强制覆盖其内部参数，确保最终大屏显示的是破纪录那一刻的截图。
        if self.best_metrics is not None:
            wandb.summary.update(self.best_metrics)


"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/utils/callbacks.py
========================================================================

1. 什么是 Callback (回调函数) 机制?
   - 把跑程序的 AI 主结构比作正在舞台上表演话剧的演员。
   - 回调函数（Callback），就是躲在幕布后面、不参与演出的小助手。
   - 当导演在预设的特殊节点吹哨子（例如：一幕戏开始了、一幕戏结束了、验证成绩出了），
   这帮助手就会快速跳出来，做做记录、发发邮件或者是把演员当时的着装和站位拍照存个档（即模型的 Checkpoint），然后又赶紧躲回帷幕后面去。

2. 本文件里这两个跟 wandb 有关的回调有啥用？
   - `WandbModelCheckpoint` 保证了模型的最新“游戏进度”不再只是埋单机硬盘的角落里，而是会被实时打包装入能联云的文件夹中；万一服务器断电，可以在网上找回进度。
   - `WandbSummaries` 则是个计分器。因为训练要跑千千万万次，有时候训练跑到了第 50 圈是巅峰表现，但跑到 100 圈反而能力倒退了。它会自动锁定住那个“最强状态下的全套指标准确画面”，方便开发者在一眼就能看出这次实验最高战力在哪儿。
========================================================================
"""
