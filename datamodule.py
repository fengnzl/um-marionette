import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()
# 设置全局默认数据类型为单精度浮点，为了兼顾精度与计算速度。
torch.set_default_dtype(torch.float32)


@typechecked
class Sequence:
    """
    【数据收纳单元】：
    这相当于现实中“一个人的一份完整行动生平档案夹”。
    里面事无巨细地记录了他在一天中：多会儿干了啥（time）、去了哪的具体坐标（checkins）、这是属于哪个大类属（category）、以及那一刻的周边情况（condition1到6分别代表各种气温、拥堵、天气等等）。
    这也是我们系统进行处理的“最小基础粒子”。
    """
    def __init__(
        self,
        time: np.ndarray | TensorType[float, "events"],               # 事件发生的时间（例如 8.5 表示早上8点半）
        checkins: np.ndarray | TensorType[int, "events"],             # 具体去了哪个商户/地标号
        category: np.ndarray | TensorType[int, "events"],             # 去了干嘛的类别（吃饭1、看电影2等）
        condition1: np.ndarray | TensorType[int, "events"],           # 六管外部客观环境变量
        condition2: np.ndarray | TensorType[int, "events"],
        condition3: np.ndarray | TensorType[int, "events"],
        condition4: np.ndarray | TensorType[int, "events"],
        condition5: np.ndarray | TensorType[int, "events"],
        condition6: np.ndarray | TensorType[int, "events"],
        condition1_indicator: np.ndarray | TensorType[int, "granularity"],  # 环境标志物的辅助指示器
        condition2_indicator: np.ndarray | TensorType[int, "granularity"],
        condition3_indicator: np.ndarray | TensorType[int, "granularity"],
        condition4_indicator: np.ndarray | TensorType[int, "granularity"],
        condition5_indicator: np.ndarray | TensorType[int, "granularity"],
        condition6_indicator: np.ndarray | TensorType[int, "granularity"],
        tmax: Union[np.ndarray, TensorType[float], float],            # 这个人档案记录的最大限度时间跨度
        device: Union[torch.device, str] = "cpu",                     # 默认存放在CPU内存里
        kept_points: Union[np.ndarray, TensorType, None] = None,      # 筛选保留的点标志
    ) -> None:
        super().__init__()
        
        # --- 数据格式防呆转换区 ---
        # 如果外面传进来的不是 PyTorch 的原生张量(Tensor)，就强行给它包装转换成 Tensor。
        if not isinstance(time, torch.Tensor): time = torch.as_tensor(time)
        if not isinstance(checkins, torch.Tensor): checkins = torch.as_tensor(checkins)
        if not isinstance(category, torch.Tensor): category = torch.as_tensor(category)

        if not isinstance(condition1, torch.Tensor): condition1 = torch.as_tensor(condition1)
        if not isinstance(condition2, torch.Tensor): condition2 = torch.as_tensor(condition2)
        if not isinstance(condition3, torch.Tensor): condition3 = torch.as_tensor(condition3)
        if not isinstance(condition4, torch.Tensor): condition4 = torch.as_tensor(condition4)
        if not isinstance(condition5, torch.Tensor): condition5 = torch.as_tensor(condition5)
        if not isinstance(condition6, torch.Tensor): condition6 = torch.as_tensor(condition6)

        if not isinstance(condition1_indicator, torch.Tensor): condition1_indicator = torch.as_tensor(condition1_indicator)
        if not isinstance(condition2_indicator, torch.Tensor): condition2_indicator = torch.as_tensor(condition2_indicator)
        if not isinstance(condition3_indicator, torch.Tensor): condition3_indicator = torch.as_tensor(condition3_indicator)
        if not isinstance(condition4_indicator, torch.Tensor): condition4_indicator = torch.as_tensor(condition4_indicator)
        if not isinstance(condition5_indicator, torch.Tensor): condition5_indicator = torch.as_tensor(condition5_indicator)
        if not isinstance(condition6_indicator, torch.Tensor): condition6_indicator = torch.as_tensor(condition6_indicator)

        if tmax is not None and not isinstance(tmax, torch.Tensor):
            tmax = torch.as_tensor(tmax)

        if kept_points is not None and not isinstance(kept_points, torch.Tensor):
            kept_points = torch.as_tensor(kept_points)

        # 档案录入建档
        self.time = time
        self.checkins = checkins
        self.category = category

        self.condition1 = condition1
        self.condition2 = condition2
        self.condition3 = condition3
        self.condition4 = condition4
        self.condition5 = condition5
        self.condition6 = condition6

        self.condition1_indicator = condition1_indicator
        self.condition2_indicator = condition2_indicator
        self.condition3_indicator = condition3_indicator
        self.condition4_indicator = condition4_indicator
        self.condition5_indicator = condition5_indicator
        self.condition6_indicator = condition6_indicator

        self.tmax = tmax
        self.kept_points = kept_points
        self.device = device
        
        # 一次性将包裹打包移交目标设备
        self.to(device)
        
        # 【算出时间的步态差】：这人每次行动之间的空档期(tau)是多少？
        # 比如：8点行动，10点行动。那么 tau = (0->8=8, 8->10=2, 10->tmax)
        tau = torch.diff(
            self.time,
            prepend=torch.as_tensor([0.0], device=device),
            append=torch.as_tensor([self.tmax], device=device),
        )
        self.tau = tau

    def __len__(self) -> int:
        # 获取这本档案里总共发生了几次行动
        return len(self.time)

    def __getitem__(self, key: str):
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def keys(self) -> List[str]:
        # 最终返回仅包含当前正在起作用的属性名称列表
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def to(self, device: Union[str, torch.device]) -> "Sequence":
        # 数据移卡转移大法：习惯了 tensor.to('cuda') 的人也可以对单个包装类无痛转换
        self.device = device
        for key in self.keys():
            if key != "device":
                self[key] = self[key].to(device)
        return self


@typechecked
class Batch:
    """
    【批量包裹封装战车】：
    光有几万个单独的文件袋(Sequence)显卡这大家伙是吞不下的。
    它需要把这文件袋一打一打摞整齐，把里面短的档案后面盖上补齐空白章(Mask/Padding)。
    用一个极其庞大正规整齐的矩阵方阵送给军方发货(Batch)。
    """
    def __init__(
        self,
        mask: TensorType[bool, "batch", "sequence"],          # 掩码戳印：标明矩阵哪里是真的，哪里是拿0滥竽充数的空白
        time: TensorType[float, "batch", "sequence"],
        condition1: TensorType[int, "batch", "sequence"],
        condition2: TensorType[int, "batch", "sequence"],
        condition3: TensorType[int, "batch", "sequence"],
        condition4: TensorType[int, "batch", "sequence"],
        condition5: TensorType[int, "batch", "sequence"],
        condition6: TensorType[int, "batch", "sequence"],
        condition1_indicator: TensorType[int, "batch", "granularity"],
        condition2_indicator: TensorType[int, "batch", "granularity"],
        condition3_indicator: TensorType[int, "batch", "granularity"],
        condition4_indicator: TensorType[int, "batch", "granularity"],
        condition5_indicator: TensorType[int, "batch", "granularity"],
        condition6_indicator: TensorType[int, "batch", "granularity"],
        tau: TensorType[float, "batch", "sequence"],
        tmax: TensorType[float],
        unpadded_length: TensorType[int, "batch"],            # 真实有效记录到底有多长（不含水分装样子的位数）
        kept: Union[TensorType, None] = None,
        checkin_sequences: Union[TensorType, None] = None,    # 合成后的综合轨迹字符串序列（离散空间用的题库）
        category_mask: Union[TensorType, None] = None,
        poi_mask: Union[TensorType, None] = None,
    ):
        super().__init__()
        self.time = time
        self.condition1 = condition1
        self.condition2 = condition2
        self.condition3 = condition3
        self.condition4 = condition4
        self.condition5 = condition5
        self.condition6 = condition6

        self.condition1_indicator = condition1_indicator
        self.condition2_indicator = condition2_indicator
        self.condition3_indicator = condition3_indicator
        self.condition4_indicator = condition4_indicator
        self.condition5_indicator = condition5_indicator
        self.condition6_indicator = condition6_indicator

        self.tau = tau
        self.tmax = tmax
        self.kept = kept

        self.unpadded_length = unpadded_length
        self.mask = mask

        self.checkin_sequences = checkin_sequences
        self.category_mask = category_mask
        self.poi_mask = poi_mask
        self._validate()

    @property
    def batch_size(self) -> int:
        return self.time.shape[0]

    @property
    def seq_len(self) -> int:
        return self.time.shape[1]

    @property
    def content_len(self) -> int:
        # 离散变压器专门需要知道混杂组合拼接以后的“乱炖连体句子”总长度
        if self.checkin_sequences is not None:
            assert (
                self.checkin_sequences.shape[1] == max(self.unpadded_length) * 2 + 3
            ), "wrong content_len"
            return self.checkin_sequences.shape[1]
        return (
            (max(self.unpadded_length) * 2 + 3).item()
            if isinstance(max(self.unpadded_length) * 2 + 3, torch.Tensor)
            else max(self.unpadded_length) * 2 + 3
        )

    def __len__(self):
        return self.batch_size

    def __getitem__(self, key: str):
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def keys(self) -> List[str]:
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def to(self, device: Union[str, torch.device]) -> "Batch":
        self.device = device
        for key in self.keys():
            if key != "device":
                self[key] = self[key].to(device)
        return self

    @staticmethod
    def from_sequence_list(sequences: List[Sequence]) -> "Batch":
        """
        【极其核心的发卡工厂流水线】：
        它怎么把一堆长短不一的个人档案（Sequence），强行压成规整的长方形装箱（Batch）？
        答：找到最高的那个人，别人不够高就底下塞木板补齐（Pad padding_value = 0 或 3）。
        同时，在发放试卷给《离散扩散模型》时，它用极其粗暴但巧妙的拼接法：把一个人的 category 和 poi 交错拼接。
        做成一条 "0号头, 类别1, 类别2, 类别3, 1号中枢, POI1, POI2, POI3, 2号尾巴" 这种人首蛇身串作为它专属的离散语言。
        """
        # 取出来最大的极限事件长
        tmax = torch.cat(
            [sequence.tmax.unsqueeze(dim=0) for sequence in sequences], dim=0
        ).max()
        # 把每个人的步伐间距空隙拉成方阵补齐
        tau = pad([sequence.tau for sequence in sequences])
        time = pad([sequence.time for sequence in sequences], length=tau.shape[-1])

        # 环境指标拉方阵...
        condition1 = pad([sequence.condition1 for sequence in sequences], length=tau.shape[-1])
        condition2 = pad([sequence.condition2 for sequence in sequences], length=tau.shape[-1])
        condition3 = pad([sequence.condition3 for sequence in sequences], length=tau.shape[-1])
        condition4 = pad([sequence.condition4 for sequence in sequences], length=tau.shape[-1])
        condition5 = pad([sequence.condition5 for sequence in sequences], length=tau.shape[-1])
        condition6 = pad([sequence.condition6 for sequence in sequences], length=tau.shape[-1])

        condition1_indicator = torch.stack([sequence.condition1_indicator for sequence in sequences])
        condition2_indicator = torch.stack([sequence.condition2_indicator for sequence in sequences])
        condition3_indicator = torch.stack([sequence.condition3_indicator for sequence in sequences])
        condition4_indicator = torch.stack([sequence.condition4_indicator for sequence in sequences])
        condition5_indicator = torch.stack([sequence.condition5_indicator for sequence in sequences])
        condition6_indicator = torch.stack([sequence.condition6_indicator for sequence in sequences])

        device = tau.device

        # 记录下每本档案在没垫木板前有多高
        sequence_length = torch.tensor([len(sequence) for sequence in sequences], device=device)

        if sequences[0].kept_points != None:
            kept_points = pad([sequence.kept_points for sequence in sequences],length=tau.shape[-1])
        else:
            kept_points = None

        # 发放识别垫木板的水印（比如长2的小明，在5上限的方阵里。就是 True True False False False）
        mask = (torch.arange(0, tau.shape[-1], device=device)[None, :] < sequence_length[:, None])

        # 【重点】：给那个离散事件模型发混杂串卷子 (0开始, 1分隔, 2收尾, 3垫厚度)
        if sequences[0].category is not None:
            checkin_sequences = pad(
                [
                    torch.cat(
                        (
                            torch.tensor([0], device=device),          # 头衔标志 0
                            sequences[idx].category,                   # 中间夹杂该人的类别历程
                            torch.tensor([1], device=device),          # 类别结束地点开始了的分割标志 1
                            sequences[idx].checkins,                   # 事件具体地图 POI 历程
                            torch.tensor([2], device=device),          # 说走完停局的标志 2
                        ),
                        dim=0,
                    )
                    for idx in range(len(sequences))
                ],
                value=3, # 剩余不够的用废话乱码 “3” 垫到齐为止
            )

        # 做出两幅遮光镜子：一副专门在混杂串里只透视类别词的。
        category_mask = pad(
            [
                torch.cat(
                    (
                        torch.tensor([0], device=device, dtype=torch.int64),
                        torch.tensor([1] * len(sequences[idx].time), device=device, dtype=torch.int64),
                        torch.tensor([0] * (len(sequences[idx].time) + 2), device=device, dtype=torch.int64),
                    ),
                    dim=0,
                )
                for idx in range(len(sequences))
            ]
        )
        # 另一副是只能看见 POI 地图实体的。
        poi_mask = pad(
            [
                torch.cat(
                    (
                        torch.tensor([0] * (len(sequences[idx].time) + 2), device=device, dtype=torch.int64),
                        torch.tensor([1] * len(sequences[idx].time), device=device, dtype=torch.int64),
                        torch.tensor([0], device=device, dtype=torch.int64),
                    ),
                    dim=0,
                )
                for idx in range(len(sequences))
            ]
        )

        batch = Batch(
            mask=mask,
            time=time,
            checkin_sequences=checkin_sequences,
            category_mask=category_mask,
            poi_mask=poi_mask,
            condition1=condition1,
            condition2=condition2,
            condition3=condition3,
            condition4=condition4,
            condition5=condition5,
            condition6=condition6,
            condition1_indicator=condition1_indicator,
            condition2_indicator=condition2_indicator,
            condition3_indicator=condition3_indicator,
            condition4_indicator=condition4_indicator,
            condition5_indicator=condition5_indicator,
            condition6_indicator=condition6_indicator,
            tau=tau,
            tmax=tmax,
            unpadded_length=sequence_length,
            kept=kept_points,
        )
        return batch

    def add_events(self, other: "Batch") -> "Batch":
        """强行向这一批车厢塞入额外乱七八糟的事，常被用来作为干扰测试或者拼接时间段"""
        assert len(other) == len(self), "The number of sequences to add does not match the number of sequences in the batch."
        other = other.to(self.time.device)
        tmax = max(self.tmax, other.tmax)

        if self.kept is None:
            kept = torch.cat([torch.ones_like(self.time, dtype=bool),torch.zeros_like(other.time, dtype=bool)],dim=-1,)
        else:
            kept = torch.cat([self.kept, torch.zeros_like(other.time, dtype=bool)], dim=-1)

        return self.remove_unnescessary_padding(
            time=torch.cat([self.time, other.time], dim=-1),
            condition1=torch.cat([self.condition1, other.condition1], dim=-1),
            condition2=torch.cat([self.condition2, other.condition2], dim=-1),
            condition3=torch.cat([self.condition3, other.condition3], dim=-1),
            condition4=torch.cat([self.condition4, other.condition4], dim=-1),
            condition5=torch.cat([self.condition5, other.condition5], dim=-1),
            condition6=torch.cat([self.condition6, other.condition6], dim=-1),
            condition1_indicator=self.condition1_indicator,
            condition2_indicator=self.condition2_indicator,
            condition3_indicator=self.condition3_indicator,
            condition4_indicator=self.condition4_indicator,
            condition5_indicator=self.condition5_indicator,
            condition6_indicator=self.condition6_indicator,
            mask=torch.cat([self.mask, other.mask], dim=-1),
            kept=kept,
            tmax=tmax,
        )

    def to_time_list(self):
        time = []
        for i in range(len(self)):
            # 去除水分，纯把干货时间轴抽出来塞进列表。
            time.append(self.time[i][self.mask[i]].detach().cpu().numpy())
        return time

    def to_seq_list(self, gps_dict):
        """
        把高维度神乎其神的机器懂的批次模型（Batch），反着扒皮解肉还原成凡人前端/画图脚本能看懂的：
        一个包含【精准到现实地球经纬度的去向列表字典】。
        """
        seqs_new = []
        for i in range(len(self)):
            gps = []
            index = []
            arrival_times_gen = self.time[i][self.mask[i]].detach().cpu().numpy()
            marks_gen = (self.checkin_sequences[i][self.category_mask[i].bool()].detach().cpu().numpy())
            condition1_gen = self.condition1[i][self.mask[i]].detach().cpu().numpy()
            condition2_gen = self.condition2[i][self.mask[i]].detach().cpu().numpy()
            condition3_gen = self.condition3[i][self.mask[i]].detach().cpu().numpy()
            condition4_gen = self.condition4[i][self.mask[i]].detach().cpu().numpy()
            condition5_gen = self.condition5[i][self.mask[i]].detach().cpu().numpy()
            condition6_gen = self.condition6[i][self.mask[i]].detach().cpu().numpy()
            checkins_gen = (self.checkin_sequences[i][self.poi_mask[i].bool()].detach().cpu().numpy())

            for idx in range(len(checkins_gen)):
                if checkins_gen[idx] not in gps_dict.keys():
                    index.append(idx)
                    continue
                # 把抽象的 POI 代号变成了真经纬度
                gps_str = gps_dict[checkins_gen[idx]].split(",")
                gps.append([float(gps_str[0]), float(gps_str[1])])

            # 剔除那些在字典里找不着废弃越界的怪点。并清洗。
            arrival_times_clean = np.delete(arrival_times_gen, index)
            marks_clean = np.delete(marks_gen, index)
            checkins_clean = np.delete(checkins_gen, index)
            condition1_clean = np.delete(condition1_gen, index)
            condition2_clean = np.delete(condition2_gen, index)
            condition3_clean = np.delete(condition3_gen, index)
            condition4_clean = np.delete(condition4_gen, index)
            condition5_clean = np.delete(condition5_gen, index)
            condition6_clean = np.delete(condition6_gen, index)

            seqs_new.append(
                {
                    "arrival_times": arrival_times_clean,
                    "marks": marks_clean,
                    "checkins": checkins_clean,
                    "gps": gps,
                    "condition1": condition1_clean,
                    "condition2": condition2_clean,
                    "condition3": condition3_clean,
                    "condition4": condition4_clean,
                    "condition5": condition5_clean,
                    "condition6": condition6_clean,
                    "condition1_indicator": self.condition1_indicator[i].detach().cpu().numpy(),
                    "condition2_indicator": self.condition2_indicator[i].detach().cpu().numpy(),
                    "condition3_indicator": self.condition3_indicator[i].detach().cpu().numpy(),
                    "condition4_indicator": self.condition4_indicator[i].detach().cpu().numpy(),
                    "condition5_indicator": self.condition5_indicator[i].detach().cpu().numpy(),
                    "condition6_indicator": self.condition6_indicator[i].detach().cpu().numpy(),
                }
            )
        return seqs_new

    def mask_check(self):
        # 万一缺失了就重新生配那两把照看类目和事件的蒙头纱布
        if self.checkin_sequences is None:
            self.category_mask = pad(
                [
                    torch.cat(
                        (
                            torch.tensor([0], device=self.time.device, dtype=torch.int64),
                            torch.tensor([1] * length, device=self.time.device, dtype=torch.int64),
                            torch.tensor([0] * (length + 2),device=self.time.device,dtype=torch.int64,),
                        ), dim=0,) for length in self.unpadded_length
                ]
            )
            self.poi_mask = pad(
                [
                    torch.cat(
                        (
                            torch.tensor([0] * (length + 2),device=self.time.device,dtype=torch.int64,),
                            torch.tensor([1] * length, device=self.time.device, dtype=torch.int64),
                            torch.tensor([0], device=self.time.device, dtype=torch.int64),
                        ), dim=0,) for length in self.unpadded_length
                ]
            )
        self.device = self.time.device
        return self

    @staticmethod
    def sort_time(
        time: TensorType[float, "batch", "sequence"],
        condition1: TensorType[int, "batch", "sequence"],
        condition2: TensorType[int, "batch", "sequence"],
        condition3: TensorType[int, "batch", "sequence"],
        condition4: TensorType[int, "batch", "sequence"],
        condition5: TensorType[int, "batch", "sequence"],
        condition6: TensorType[int, "batch", "sequence"],
        mask: TensorType[bool, "batch", "sequence"],
        kept,
        tmax: TensorType[float],
    ):
        """重新整理档案室的乱章，按照时间的流逝顺序把事一件件顺回来排齐。"""
        # 未发生的烂占位时间直接丢到无穷远大后边 (2*tmax)
        time[~mask] = 2 * tmax
        # 生成基于时间的新秩序位号排序卡
        sort_idx = torch.argsort(time, dim=-1)
        # 用新身份卡去替换时间、遮罩、环境以及留存点
        mask = torch.take_along_dim(mask, sort_idx, dim=-1)
        time = torch.take_along_dim(time, sort_idx, dim=-1)
        condition1 = torch.take_along_dim(condition1, sort_idx, dim=-1)
        condition2 = torch.take_along_dim(condition2, sort_idx, dim=-1)
        condition3 = torch.take_along_dim(condition3, sort_idx, dim=-1)
        condition4 = torch.take_along_dim(condition4, sort_idx, dim=-1)
        condition5 = torch.take_along_dim(condition5, sort_idx, dim=-1)
        condition6 = torch.take_along_dim(condition6, sort_idx, dim=-1)
        if kept is not None:
            kept = torch.take_along_dim(kept, sort_idx, dim=-1)
        else:
            kept = None
            
        time = time * mask
        condition1 = condition1 * mask
        condition2 = condition2 * mask
        condition3 = condition3 * mask
        condition4 = condition4 * mask
        condition5 = condition5 * mask
        condition6 = condition6 * mask
        return (time, condition1, condition2, condition3, condition4, condition5, condition6, mask, kept,)

    @staticmethod
    def remove_unnescessary_padding(
        time: TensorType[float, "batch", "sequence"],
        condition1: TensorType[int, "batch", "sequence"],
        condition2: TensorType[int, "batch", "sequence"],
        condition3: TensorType[int, "batch", "sequence"],
        condition4: TensorType[int, "batch", "sequence"],
        condition5: TensorType[int, "batch", "sequence"],
        condition6: TensorType[int, "batch", "sequence"],
        condition1_indicator: TensorType[int, "batch", "granularity"],
        condition2_indicator: TensorType[int, "batch", "granularity"],
        condition3_indicator: TensorType[int, "batch", "granularity"],
        condition4_indicator: TensorType[int, "batch", "granularity"],
        condition5_indicator: TensorType[int, "batch", "granularity"],
        condition6_indicator: TensorType[int, "batch", "granularity"],
        mask: TensorType[bool, "batch", "sequence"],
        kept,
        tmax,
    ):
        """减肥操：把这方阵后边跟着的那长长的一截没用的 0 垫子给连骨剁掉，节省卡显存"""
        (time, condition1, condition2, condition3, condition4, condition5, condition6, mask, kept,) = Batch.sort_time(
            time, condition1, condition2, condition3, condition4, condition5, condition6, mask, kept, tmax=tmax,
        )

        # 找到这批队伍里站得最考后的（最大实际有用长度）
        max_length = max(mask.sum(-1)).int()
        # 咔嚓，一刀截断后边的空壳队伍。
        mask = mask[:, : max_length + 1]
        time = time[:, : max_length + 1]
        condition1 = condition1[:, : max_length + 1]
        condition2 = condition2[:, : max_length + 1]
        condition3 = condition3[:, : max_length + 1]
        condition4 = condition4[:, : max_length + 1]
        condition5 = condition5[:, : max_length + 1]
        condition6 = condition6[:, : max_length + 1]
        if kept is not None:
            kept = kept[:, : max_length + 1]

        # 重新再核算一遍步频距离差
        time_tau = torch.where(mask, time, tmax)
        tau = torch.diff(time_tau, prepend=torch.zeros_like(time_tau)[:, :1], dim=-1)
        tau = tau * mask

        return Batch(
            mask=mask, time=time, condition1=condition1, condition2=condition2, condition3=condition3,
            condition4=condition4, condition5=condition5, condition6=condition6, condition1_indicator=condition1_indicator,
            condition2_indicator=condition2_indicator, condition3_indicator=condition3_indicator, condition4_indicator=condition4_indicator,
            condition5_indicator=condition5_indicator, condition6_indicator=condition6_indicator, tau=tau, tmax=tmax, unpadded_length=mask.sum(-1).long(), kept=kept,
        )

    def thin(self, alpha: TensorType[float]) -> Tuple["Batch", "Batch"]:
        """
        这个就是前边扩散用的毒药：凭概率（这就像是抓阄）扔骰子去随机保留或剔除某个事。（Thinning算法的立足之基）
        """
        if alpha.dim() == 1:
            keep = torch.bernoulli(alpha.unsqueeze(1).repeat(1, self.seq_len)).bool()
        elif alpha.dim() == 2:
            keep = torch.bernoulli(alpha).bool()
        else:
            raise Warning("alpha has too many dimensions")

        # 这个是抽签留下来的天选保命区
        keep_mask = self.mask * keep
        # 这个是惨遭不幸废掉扔垃圾桶的点
        rem_mask = self.mask * ~keep

        return self.remove_unnescessary_padding(
            time=self.time * keep_mask, condition1=self.condition1 * keep_mask, condition2=self.condition2 * keep_mask,
            condition3=self.condition3 * keep_mask, condition4=self.condition4 * keep_mask, condition5=self.condition5 * keep_mask,
            condition6=self.condition6 * keep_mask, condition1_indicator=self.condition1_indicator, condition2_indicator=self.condition2_indicator,
            condition3_indicator=self.condition3_indicator, condition4_indicator=self.condition4_indicator, condition5_indicator=self.condition5_indicator,
            condition6_indicator=self.condition6_indicator, mask=keep_mask, kept=self.kept * keep_mask if self.kept is not None else self.kept, tmax=self.tmax,
        ), self.remove_unnescessary_padding(
            time=self.time * rem_mask, condition1=self.condition1 * rem_mask, condition2=self.condition2 * rem_mask,
            condition3=self.condition3 * rem_mask, condition4=self.condition4 * rem_mask, condition5=self.condition5 * rem_mask,
            condition6=self.condition6 * rem_mask, condition1_indicator=self.condition1_indicator, condition2_indicator=self.condition2_indicator,
            condition3_indicator=self.condition3_indicator, condition4_indicator=self.condition4_indicator, condition5_indicator=self.condition5_indicator,
            condition6_indicator=self.condition6_indicator, mask=rem_mask, kept=self.kept * rem_mask if self.kept is not None else self.kept, tmax=self.tmax,
        )

    def _validate(self):
        """验货部门：确保装车打包出来的货物队伍长度不差、防伪标识不错位。"""
        assert (self.mask.sum(-1) == self.unpadded_length).all(), "wrong mask"
        assert (self.time * self.mask == self.time).all(), "wrong mask"
        assert torch.allclose(self.tau.cumsum(-1) * self.mask, self.time * self.mask, atol=1e-5), "wrong tau"
        assert self.tau.shape == (self.batch_size, self.seq_len,), f"tau has wrong shape {self.tau.shape}, expected {(self.batch_size, self.seq_len)}"
        if self.checkin_sequences != None:
            assert (self.category_mask.sum(-1) == self.unpadded_length).all(), "wrong mask"
            assert (self.poi_mask.sum(-1) == self.unpadded_length).all(), "wrong mask"
            assert (self.checkin_sequences.shape == self.category_mask.shape == self.poi_mask.shape), "wrong mask"

@typechecked
def pad(sequences, length: Union[int, None] = None, value: float = 0):
    """垫板脚：把不齐的边全都塞满 0 或 3 ，凑成方方正正的好发车。"""
    if length:
        device = sequences[0].device
        dtype = sequences[0].dtype
        tensor_length = sequences[0].size(0)
        intial_pad = torch.empty(torch.Size([length]) + sequences[0].shape[1:], dtype=dtype, device=device,).fill_(value)
        intial_pad[:tensor_length, ...] = sequences[0]
        sequences[0] = intial_pad
    sequences = pad_sequence(sequences, batch_first=True, padding_value=value)  
    return sequences


@typechecked
class SequenceDataset(torch.utils.data.Dataset):
    """
    【包裹暂存处】：用来托管存放大量的上述的个体“文件袋（Sequence）”以方便 PyTorch 的流水大货车快速发单拉取。
    """
    def __init__(self, sequences: List[Sequence],):
        self.sequences = sequences
        self.tmax = sequences[0].tmax

    def __getitem__(self, idx: int):
        return self.sequences[idx]

    def __len__(self) -> int:
        return len(self.sequences)

    def to(self, device: Union[torch.device, str]):
        for sequence in self.sequences:
            sequence.to(device)


@typechecked
class DataModule(pl.LightningDataModule):
    """
    【总管数据流水列车的车长】：这是 PyTorch Lightning 特有的一种专门独立分离“数据读取”的官方框架设计。
    避免你到处去找数据存放的代码。全由它一人掌管。
    """
    def __init__(self, root: str, name: str, batch_size: int = 32,) -> None:
        super().__init__()
        self.root = root               # 去哪里捞原始包裹库
        self.batch_size = batch_size   # 我们每次打包装几本生平字典为一手交货
        self.name = name

        # 延迟装货不爆显存：一开始先占坑不塞内容。
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        """从后勤仓库读取文件，剥开壳子塞进 Sequence 并装载进暂存处 Dataset """
        time_sequences_train, num_category, num_poi, gps_dict = load_sequences(Path(self.root + f"/{self.name}"), self.name + "_train")
        time_sequences_test, _, _, _ = load_sequences(Path(self.root + f"/{self.name}"), self.name + "_test")

        self.train_data = SequenceDataset(sequences=time_sequences_train)
        self.test_data = SequenceDataset(sequences=time_sequences_test)

        self.tmax = self.train_data.tmax
        self.num_category = num_category
        self.num_poi = num_poi
        self.gps_dict = gps_dict

        self.get_statistics()
        
    def get_statistics(self):
        """【排查极高个】：算出这里头最啰嗦的人一天干了多少事，定下了最大的容量尺寸防止爆掉。"""
        seq_lengths = []
        for i in range(len(self.train_data)):
            seq_lengths.append(len(self.train_data[i]))
        self.n_max = max(seq_lengths)

    def setup(self, stage=None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        """把暂存处交接给专门发火车的搬运工，每次拉 32 本卷宗组成大纸箱（即前文写过的 Batch.from_sequence_list）走发"""
        return DataLoader(
            self.train_data, batch_size=self.batch_size, collate_fn=Batch.from_sequence_list, num_workers=0, shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, batch_size=self.batch_size, collate_fn=Batch.from_sequence_list, num_workers=0, drop_last=False,
        )

def load_sequences(root, name: str) -> List[Sequence]:
    """读取生硬死板的 .pkl 二进制数据文档，把它拆壳分类映射成我们精心准备好的包装大结构（Sequence）。"""
    path = os.path.join(root, f"{name}.pkl")
    # map_location 把硬盘数据只抽调在内存，不抢占可贵的显存！
    # loader = torch.load(path, map_location=torch.device("cpu"))
    loader = torch.load(path, map_location=torch.device("cpu"), weights_only=False)

    sequences = loader["sequences"]
    tmax = loader["t_max"]
    num_category = loader["num_marks"]
    num_poi = loader["num_pois"]
    gps_dict = loader["poi_gps"]

    time_sequences = [
        Sequence(
            time=seq["arrival_times"], condition1=seq["condition1"], condition2=seq["condition2"],
            condition3=seq["condition3"], condition4=seq["condition4"], condition5=seq["condition5"],
            condition6=seq["condition6"], condition1_indicator=seq["condition1_indicator"],
            condition2_indicator=seq["condition2_indicator"], condition3_indicator=seq["condition3_indicator"],
            condition4_indicator=seq["condition4_indicator"], condition5_indicator=seq["condition5_indicator"],
            condition6_indicator=seq["condition6_indicator"], tmax=tmax, checkins=seq["checkins"], category=seq["marks"],
        )
        for seq in sequences
    ]
    return time_sequences, num_category, num_poi, gps_dict

"""
========================================================================
【给外行新手的通俗讲解】 -- datamodule.py (后勤数据调度大仓)
========================================================================

1. 为什么一个简简单单的“读数据”搞得代码如此冗长惊人（800多行）？
   - 因为这个模型的胃口特别挑剔而且胃肠道很容易因为长短不一的“骨头”卡崩溃。
   - 我们的个人活动足迹是一个非常**混乱、长短不一**的数据（有人出门一趟只有2个点位，有个人出门一趟去 15 个地方）。加上有海量的天气环境（condition1-6）掺杂在一起。如果你随意粗暴的扔给显卡做矩阵同乘，显存一秒内就会崩溃报错。

2. 为了保证不出乱子，这里组建了四个后勤科室来包办：
   - 【第一科室 `Sequence` 档案袋】：负责专门规范化每个单独个人的履历表，不准漏项，缺的地方自动转为标准墨水（张量 tensor）。
   - 【第二科室 `Batch` 超级打包车间】：由于模型算一万个人是把他们组成一个矩阵并行的，必须找个最长的补齐（pad）。并且这里有个人工智能鬼才设计 —— 把抽象孤立的一个个分类标签变成了字符串一样的拼句（`0头, 事件一, 分隔符1, POI二...`）。交给后面的 Transformer 去读！
   - 【第三科室 `thin` 减重/随机污染】：这里随时听候调用，对某个人的时间线执行抛硬币拔取丢弃，模拟扩散模型中噪音破坏事实！
   - 【第四科室 `DataModule` 列车发发发】：向外暴露傻瓜接口。不管底下读文件、组方块、防崩内存多费劲。交给上头的长官 `trainer.fit` 一人，他什么心都不操，只会说“开下一枪”。
========================================================================
"""
