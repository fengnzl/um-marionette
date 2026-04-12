import inspect                # 系统自带库：用于获取检查代码执行时的信息（如谁调用了谁）
import logging                # 系统自带库：用于记录程序运行时的日志记录

import pytorch_lightning as pl # 第三方库：对 PyTorch 的高级封装，简化训练代码
import rich                   # 第三方库：让在终端打印的文字变得有颜色、更好看
import torch.nn as nn         # 导入 PyTorch 的神经网络模块
from omegaconf import DictConfig, OmegaConf  # 用于解析 YAML 配置文件的工具
from pytorch_lightning.utilities import rank_zero_only # 装饰器：只在主进程（排名为0的GPU）中执行
from rich.syntax import Syntax # 用于格式化并高亮显示输出

def get_logger():
    """
    获取一个日志记录器。
    巧妙之处：它能够自动识别是哪个 Python 文件在调用它，并将日志名设为那个文件的名字。
    """
    # inspect.stack() 会获取到 Python 运行时的“调用栈（Call Stack）”。
    # [0] 代表当前正在执行的函数自己。
    # [1] 代表上一层，也就是获取“谁”调用了 get_logger() 这个函数。
    caller = inspect.stack()[1]
    
    # 找出这个调用者究竟身处在哪个 Python 模块里面
    module = inspect.getmodule(caller.frame)
    logger_name = None
    
    # 如果成功找到了调用者所在的模块
    if module is not None:
        # module.__name__ 拿到完整的模块路径名（比如 foo.bar.test）
        # .split(".")[-1] 切割并拿到最后一个词（比如 test）作为日志收集器的名称
        logger_name = module.__name__.split(".")[-1]
    
    # 返回构建好的专属日志记录器
    return logging.getLogger(logger_name)

# @rank_zero_only：在多显卡（多进程）一起训练大模型时，这个装饰器保证下面的函数只在第一张卡上执行一次。
# 否则如果有8张显卡，屏幕上会重复打印8遍一样的配置。
@rank_zero_only
def print_config(config: DictConfig) -> None:
    """
    将当前模型的运行配置漂亮地打印到控制台上。
    """
    # 将内部的配置参数字典，转换回容易阅读的 YAML 文本格式
    content = OmegaConf.to_yaml(config, resolve=True)
    # 利用 rich 库的语法高亮功能，在黑框框终端里打出五颜六色的 YAML 排版
    rich.print(Syntax(content, "yaml"))

def count_params(model: nn.Module):
    """
    计算并统计神经网络模型里面参数的个数（模型的大小）。
    参数总量越多，模型越”聪明“但越吃显存。
    """
    return {
        # params-total：所有参数的总量（不论能不能训练） -> numel() 表示提取出张量中元素的个数
        "params-total": sum(p.numel() for p in model.parameters()),
        # params-trainable：可被训练修改的参数数量（带有 requires_grad=True 的参数）
        "params-trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        # params-not-trainable：被冻结不可训练的死参数数量
        "params-not-trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
    }

"""
========================================================================
【给外行新手的通俗讲解】 -- add_thin/utils/logging.py
========================================================================

1. 这个文件在做什么？
   - 这是一个“打杂的大内总管”，专门提供日志打印、参数统计这类系统服务功能。
   
2. 核心的三个功能：
   - `get_logger()`：给其他每个代码文件发一个专属的笔记本（日志器）。它很聪明，可以自动看身份证（调用堆栈）知道是谁在找它要笔记本，然后在笔记本封面上写上那个文件的名字。这样后续找茬排错就知道是哪里出的问题。
   - `print_config()`：在屏幕上用五颜六色的字体，好看地列出当前这套AI训练用的全套设定参数，而且很贴心地保证在多机连结时只打印一次。
   - `count_params()`：算一算这个建好的 AI 模型到底有多大，包含多少万个“脑细胞”（神经元参数）。这用来评估你的显卡能不能跑得动。
========================================================================
"""
