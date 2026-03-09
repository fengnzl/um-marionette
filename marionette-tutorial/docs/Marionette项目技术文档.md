# Marionette 项目技术文档

## 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [核心模块详解](#核心模块详解)
4. [数据流程](#数据流程)
5. [模型架构](#模型架构)
6. [API 参考](#api-参考)
7. [部署指南](#部署指南)

---

## 项目概述

### 基本信息

- **项目名称**: Marionette
- **论文来源**: ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'25)
- **研究目标**: 细粒度可控生成人类轨迹数据
- **技术栈**: Python 3.10, PyTorch 2.0, PyTorch Lightning, Hydra, W&B

### 核心特性

1. **混合扩散模型架构**
   - 时间维度：Add-THIN 扩散模型
   - 空间维度：离散扩散模型

2. **条件生成**
   - 支持 6 种条件类型
   - 星期几、斋月、假期等

3. **多任务评估**
   - 位置推荐 (LocRec)
   - 下一个位置预测 (NexLoc)
   - 语义位置 (SemLoc)
   - 疫情模拟 (EpiSim)
   - 统计指标 (Stat)

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        Marionette                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  train.py   │───▶│ configs.py  │───▶│ tasks.py    │      │
│  │  sample.py  │    │             │    │             │      │
│  │ evaluation  │    └─────────────┘    └─────────────┘      │
│  └─────────────┘           │                   │            │
│                            ▼                   ▼            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    datamodule.py                     │   │
│  │  (Sequence 类, Batch 类, DataModule 类)             │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                   │            │
│                            ▼                   ▼            │
│  ┌─────────────────────┐  ┌─────────────────────┐         │
│  │    add_thin/        │  │ discrete_diffusion/ │         │
│  │  (时间扩散模型)      │  │  (空间扩散模型)      │         │
│  │                     │  │                     │         │
│  │  - AddThin          │  │  - DiffusionTrans-  │         │
│  │  - PointClassifier  │  │    former           │         │
│  │  - MixtureIntensity │  │  - Transformer      │         │
│  └─────────────────────┘  └─────────────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 目录结构

```
Marionette/
├── add_thin/                    # 时间扩散模型
│   ├── diffusion/               #   核心扩散实现
│   │   └── model.py            #     AddThin 类
│   ├── backbones/               #   骨干网络
│   │   ├── classifier.py       #     PointClassifier
│   │   ├── cnn.py              #     CNNSeqEmb
│   │   └── embeddings.py       #     NyquistFrequencyEmbedding
│   ├── distributions/           #   概率分布
│   │   └── intensities.py      #     MixtureIntensity
│   ├── processes/               #   过程定义
│   │   └── hpp.py              #     齐次泊松过程
│   └── metrics.py               #   评估指标
│
├── discrete_diffusion/          # 空间扩散模型
│   ├── conditional_attention.py #   条件注意力
│   └── diffusion_transformer.py #   DiffusionTransformer
│
├── datamodule.py               # 数据管理
├── configs.py                  # 配置实例化
├── train.py                    # 训练入口
├── sample.py                   # 采样入口
├── evaluation.py               # 评估入口
│
├── tasks.py                    # 任务定义
│   ├── Tasks (基类)
│   └── DensityEstimation
│
├── config/                     # 配置文件
│   ├── train.yaml              #   训练配置
│   ├── model/
│   │   └── Marionette.yaml     #   模型配置
│   ├── task/
│   │   └── density.yaml        #   任务配置
│   └── data/
│       └── data_set.yaml       #   数据配置
│
└── evaluations/                # 评估任务
    ├── run_LocRec.py           #   位置推荐
    ├── run_NexLoc.py           #   下一个位置预测
    ├── run_SemLoc.py           #   语义位置
    ├── run_EpiSim.py           #   疫情模拟
    └── statistical_metrics.py  #   统计指标
```

---

## 核心模块详解

### datamodule.py - 数据管理

#### Sequence 类

**作用**: 表示单个用户轨迹序列

**主要字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| `time` | Tensor | 到达时间戳 [seq_len] |
| `tau` | Tensor | 时间间隔 [seq_len+1] |
| `condition1-6` | Tensor | 条件变量 [seq_len] |
| `condition1-6_indicator` | Tensor | 条件指示器 [granularity] |
| `checkins` | Tensor | POI ID [seq_len] |
| `category` | Tensor | POI 类别 [seq_len] |
| `tmax` | float | 最大时间 |

**关键方法**:
```python
class Sequence:
    def __init__(self, time, condition1, ..., tmax, checkins=None, category=None):
        # 存储所有字段
        # 计算 tau（时间间隔）

    def to(self, device):
        # 将所有数据移动到指定设备
```

#### Batch 类

**作用**: 批次数据封装，处理变长序列

**关键方法**:
```python
class Batch:
    @staticmethod
    def from_sequence_list(sequences):
        # 从 Sequence 列表创建 Batch
        # 自动 padding 和创建 mask
```

**处理流程**:
1. 填充 tau 到相同长度
2. 填充 time 和所有条件
3. 计算序列长度
4. 创建布尔掩码

---

### add_thin/diffusion/model.py - 时间扩散模型

#### AddThin 类

**作用**: 实现 Add-THIN 扩散算法

**核心思想**:
- **Thin**: 删除事件
- **Add**: 添加 HPP 事件

**关键方法**:

| 方法 | 输入 | 输出 | 作用 |
|------|------|------|------|
| `noise()` | x0, n | xn, x0_thin | 前向扩散 |
| `forward()` | x0 | logits, log_like, xn | 训练前向传播 |
| `sample()` | n_samples, x_n, tmax | x0 | 生成新序列 |
| `sample_posterior()` | xn, n | xn-1 | 反向采样一步 |

**前向扩散流程**:
```python
def noise(self, x0, n):
    # 1. Thin 操作：保留部分事件
    x0_kept, x0_thinned = x0.thin(alpha=self.alpha_cumprod[n])

    # 2. Add 操作：添加 HPP 事件
    hpp = generate_hpp(tmax=x0.tmax, n_sequences=x0.batch_size)
    xn = x0_kept.add_events(hpp)

    return xn, x0_thinned
```

**反向采样流程**:
```python
def sample(self, n_samples, x_n, tmax):
    # 1. 初始化 x_N（从 HPP 采样）
    x_N = generate_hpp(tmax=tmax, n_sequences=n_samples)
    x_n_1 = x_N

    # 2. 反向采样：x_N → x_0
    for n_int in range(self.steps - 1, 0, -1):
        n = torch.full((n_samples,), n_int, ...)
        x_n_1 = self.sample_posterior(x_n=x_n_1, n=n)

    # 3. 采样 x_0
    x_0 = self.sample_x_0(n=n, x_n=x_n_1)

    return x_0
```

---

### discrete_diffusion/diffusion_transformer.py - 空间扩散模型

#### DiffusionTransformer 类

**作用**: 处理 POI 序列的离散扩散

**关键技术**: Gumbel Softmax 采样

**关键方法**:

| 方法 | 输入 | 输出 | 作用 |
|------|------|------|------|
| `q_sample()` | log_x_start, t, batch | log_xt | 前向扩散 |
| `predict_start()` | log_x_t, cond_emb, t | log_pred | 预测原始序列 |
| `p_sample()` | log_x, cond_emb, t | log_sample | 反向采样 |
| `sample_fast()` | batch | Batch | 快速生成 |

**前向扩散**:
```python
def q_sample(self, log_x_start, t, batch):
    # 将原始分布逐步混合为均匀分布
    log_xt = (1 - alpha_t) * log_x_start + alpha_t * log_uniform
    return log_xt
```

**反向采样**:
```python
def p_sample(self, log_x, cond_emb, t, batch):
    # 1. 使用 Transformer 预测原始分布
    log_x0_recon = self.predict_start(log_x, cond_emb, t, batch)

    # 2. Gumbel Softmax 采样
    sample = gumbel_softmax_sample(log_x0_recon, temperature=self.temperature)

    return sample
```

---

### tasks.py - 训练任务

#### DensityEstimation 类

**作用**: 协调时间模型和空间模型的训练

**训练流程**:
```python
def training_step(self, batch, batch_idx):
    # 1. 时间维度训练
    x_n_int_x_0, log_prob_x_0, x_n = self.tpp_model.forward(batch)
    temporal_loss = self.get_loss(log_prob_x_0, x_n_int_x_0, x_n)

    # 2. 空间维度训练
    spatial_loss = self.discrete_diffusion.training_losses(batch)

    # 3. 总损失
    total_loss = temporal_loss + spatial_loss

    # 4. 记录到 W&B
    self.log('train_loss', total_loss)
    self.log('temporal_loss', temporal_loss)
    self.log('spatial_loss', spatial_loss)

    return total_loss
```

**优化器配置**:
```python
def configure_optimizers(self):
    opt1 = Adam(self.tpp_model.parameters(), lr=1e-3)
    opt2 = Adam(self.discrete_diffusion.parameters(), lr=1e-3)
    return [opt1, opt2]
```

---

## 数据流程

### 数据加载流程

```
原始数据 (.pkl)
    ↓
load_sequences()
    ↓
Sequence 对象列表
    ↓
Batch.from_sequence_list()
    ↓
Batch 对象 (padded + masked)
    ↓
DataModule.train_dataloader()
    ↓
训练批次
```

### 数据格式转换

**原始格式** → **Sequence**:
```python
# 从 .pkl 文件加载
loader = torch.load("Istanbul.pkl")
sequences = loader["sequences"]

# 转换为 Sequence 对象
for seq_data in sequences:
    seq = Sequence(
        time=torch.tensor(seq_data["arrival_times"]),
        condition1=torch.tensor(seq_data["condition1"]),
        # ...
        checkins=torch.tensor(seq_data["checkins"]),
        category=torch.tensor(seq_data["marks"])
    )
```

**Sequence** → **Batch**:
```python
# 创建批次
batch = Batch.from_sequence_list(sequences)

# Batch 包含:
# - tau: [batch_size, max_len]
# - time: [batch_size, max_len]
# - condition1-6: [batch_size, max_len]
# - mask: [batch_size, max_len] (布尔掩码)
```

---

## 模型架构

### 混合架构设计

**为什么分离时间和空间？**

1. **数据特性不同**
   - 时间：连续值，有顺序依赖
   - 空间：离散值，类别依赖

2. **扩散模型不同**
   - 时间：Add-THIN（事件级别的扩散）
   - 空间：离散扩散（token级别的扩散）

3. **训练效率**
   - 可以使用不同的学习率
   - 可以分别优化

### 模型参数

**时间模型 (AddThin)**:
```yaml
temporal:
  steps: 100              # 扩散步数
  hidden_dim: 32          # 隐藏维度
  time_segments: 24       # 时间分段
  mix_components: 10      # 混合成分数
```

**空间模型 (DiffusionTransformer)**:
```yaml
spatial:
  num_layers: 6           # Transformer 层数
  hidden_dim: 256         # 隐藏维度
  num_heads: 8            # 注意力头数
  diffusion_steps: 200    # 扩散步数
```

---

## API 参考

### train.py

**功能**: 训练入口

**用法**:
```bash
python train.py --config-name train
```

**主要流程**:
1. 设置随机种子
2. 初始化 W&B
3. 实例化数据模块
4. 实例化模型
5. 实例化任务
6. 创建 Trainer
7. 开始训练

### sample.py

**功能**: 采样入口

**用法**:
```bash
python sample.py --run_id <wandb_run_id>
```

**主要流程**:
1. 从 W&B 获取模型
2. 加载测试数据
3. 生成时间样本
4. 生成 POI 序列
5. 保存结果

### evaluation.py

**功能**: 评估入口

**用法**:
```bash
python evaluation.py --task LocRec --datasets Istanbul
python evaluation.py --task NexLoc --datasets Istanbul
python evaluation.py --task SemLoc --datasets Istanbul
python evaluation.py --task EpiSim --datasets Istanbul
python evaluation.py --task Stat --datasets Istanbul
```

---

## 部署指南

### 环境要求

- Python 3.10+
- CUDA 11.7+ (如果使用 GPU)
- 8GB+ RAM
- 10GB+ 磁盘空间

### 安装步骤

1. **克隆项目**:
```bash
git clone https://github.com/your-repo/Marionette.git
cd Marionette
```

2. **创建虚拟环境**:
```bash
python -m venv marionette_env
source marionette_env/bin/activate  # Linux/Mac
# marionette_env\\Scripts\\activate  # Windows
```

3. **安装依赖**:
```bash
pip install torch==2.0.0 pytorch-lightning==1.9.5
pip install -r requirements.txt
```

### 训练模型

1. **准备数据**:
```bash
# 将数据集放在 data/ 目录下
data/Istanbul.pkl
```

2. **开始训练**:
```bash
python train.py
```

3. **监控训练**:
- 访问 W&B: https://wandb.ai/
- 查看损失曲线
- 监控 GPU 使用

### 生成轨迹

1. **加载模型**:
```bash
python sample.py --run_id <your_run_id>
```

2. **结果保存在**:
```
outputs/
└── generated_sequences.pkl
```

---

## 常见问题

### Q1: 训练时显存不足怎么办？

**A**: 减小 batch_size 或使用梯度累积：
```python
# config/train.yaml
trainer:
  accumulate_grad_batches: 4  # 累积 4 步再更新
data:
  batch_size: 128  # 减小批次大小
```

### Q2: 如何调整生成速度？

**A**: 调整扩散步数：
```python
# config/model/Marionette.yaml
temporal:
  steps: 50  # 减少步数加快生成
spatial:
  diffusion_steps: 100
```

### Q3: 如何适配新数据集？

**A**: 参考 Day 3 的 checklist:
1. 转换数据格式
2. 调整条件字段数量
3. 更新模型配置
4. 重新训练

---

## 参考资料

### 论文
- Marionette (KDD'25)

### 工具文档
- PyTorch: https://pytorch.org/docs/
- PyTorch Lightning: https://pytorch-lightning.readthedocs.io/
- Hydra: https://hydra.cc/
- W&B: https://docs.wandb.ai/

### 代码资源
- GitHub: [项目仓库]
- 示例 Notebook: `marionette-tutorial/notebooks/`

---

*文档版本: 1.0*
*最后更新: 2024-03-06*
