from .statistical_metrics import Get_Statistical_Metrics
from . import preprocessing
from .run_SemLoc import run_SemLoc_task
from .run_EpiSim import run_EpiSim_task

"""
========================================================================
【给外行新手的通俗讲解】 -- __init__.py (包声明文件)
========================================================================
这是一个 Python 模块的“迎宾前台”。
当我们想在别的代码夹里（比如项目根目录）去使用 evaluations 目录下的功能时，
这个前台把四个最核心的主打产品摆在了橱窗里：
1. 统计评测器指标 (Get_Statistical_Metrics)
2. 数据预处理工具 (preprocessing)
3. 语义地点推荐测试运行器 (run_SemLoc_task)
4. 传染病流行病学模拟测试运行器 (run_EpiSim_task)

这样外面调用时只需 `from evaluations import Get_Statistical_Metrics` 即可，
简单整洁，不需要知道背后的复杂路径！
========================================================================
"""