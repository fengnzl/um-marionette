# Dropout 详解

> 深度学习中防止过拟合的正则化技术

---

## 📚 目录

1. [Dropout 是什么？](#dropout-是什么)
2. [作用原理](#作用原理)
3. [在 FeedForward 中的应用](#在-feedforward-中的应用)
4. [训练 vs 测试](#训练-vs-测试)
5. [数学原理](#数学原理)
6. [代码示例](#代码示例)
7. [总结](#总结)

---

## Dropout 是什么？

**Dropout** 是深度学习中一种**防止过拟合**的正则化技术。

### 核心思想：训练时"随机扔掉"一部分神经元

```
没有 Dropout          有 Dropout (训练时)
   [神经元1]            [神经元1] ✓
   [神经元2]            [神经元2] ✗ (被扔掉)
   [神经元3]            [神经元3] ✓
   [神经元4]            [神经元4] ✗ (被扔掉)
   [神经元5]            [神经元5] ✓
```

---

## 作用原理

### 1️⃣ 防止过拟合（Overfitting）

**过拟合问题**：
- 网络太聪明，"死记硬背"训练数据
- 到了测试数据就"傻眼"了

**Dropout 解决方案**：
- 每次训练随机"关掉"一些神经元
- 强迫网络**不能依赖某个特定神经元**
- 就像考试不能只靠一个学霸，全班都得学会

### 2️⃣ 模型集成效果（Model Ensemble）

```
训练时，Dropout 创造了"无数个不同的子网络":

第 1 次: [✓ ✗ ✓ ✓ ✗]  ← 子网络 A
第 2 次: [✗ ✓ ✓ ✗ ✓]  ← 子网络 B
第 3 次: [✓ ✓ ✗ ✓ ✓]  ← 子网络 C
...

测试时: 所有子网络"投票" → 更鲁棒的预测！
```

---

## 在 FeedForward 中的应用

### 代码结构

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)  # ← 定义 Dropout 层
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        # 流程：线性 → ReLU → Dropout → 线性
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
```

### 数据流图

```
FeedForward 层中的 Dropout:

输入 [4维]
   │
   ▼
┌─────────────┐
│  Linear 1   │  扩展到 8 维
└─────────────┘
   │
   ▼
┌─────────────┐
│    ReLU     │  激活函数
└─────────────┘
   │
   ▼
┌─────────────┐
│   Dropout   │  ← 扔掉 10% 的神经元！
└─────────────┘     [■ □ ■ ■ □ ■ ■ ■]  □=被扔掉
   │
   ▼
┌─────────────┐
│  Linear 2   │  压缩回 4 维
└─────────────┘
   │
   ▼
输出 [4维]
```

### 逐层执行过程

假设输入 `x` 形状为 `[batch_size, d_model] = [2, 4]`：

```python
# 假设输入
x = [
    [0.5, 1.2, -0.3, 0.8],   # 样本 1
    [1.1, -0.7, 0.4, 0.2]    # 样本 2
]

# 第 1 步：self.linear1(x) - 扩展维度
# 假设 dim_feedforward = 8
x = [
    [0.3, 1.5, -0.2, 0.9, 1.1, -0.4, 0.6, 0.7],
    [0.8, -0.3, 1.2, 0.4, -0.1, 0.9, 0.5, 1.3]
]  # 形状: [2, 8]

# 第 2 步：torch.relu(x) - ReLU 激活
x = [
    [0.3, 1.5, 0.0, 0.9, 1.1, 0.0, 0.6, 0.7],   # 负数变 0
    [0.8, 0.0, 1.2, 0.4, 0.0, 0.9, 0.5, 1.3]
]

# 第 3 步：self.dropout(x) - Dropout (训练时！)
# dropout = 0.1 意味着每个元素有 10% 概率被置零
# 假设这次随机扔掉了这些位置：
mask = [
    [1,   1,   0,   1,   1,   1,   0,   1],   # 0 表示被扔掉
    [1,   1,   1,   0,   1,   1,   1,   1]
]

x = [
    [0.3, 1.5, 0.0, 0.9, 1.1, 0.0, 0.0, 0.7],   # 第 2、6 个被置零
    [0.8, 0.0, 1.2, 0.0, 0.0, 0.9, 0.5, 1.3]    # 第 3 个被置零
]

# ⚠️ 重要：剩余元素会放大 1/(1-p) = 1/0.9 ≈ 1.11 倍
# 保持期望值不变！

# 第 4 步：self.linear2(x) - 压缩回原维度
output = [
    [..., ..., ..., ...],   # 压缩回 d_model=4
    [..., ..., ..., ...]
]
```

---

## 训练 vs 测试

### 行为差异表

| 阶段 | Dropout 行为 |
|------|-------------|
| **训练** | 随机扔掉神经元（按 dropout 概率） |
| **测试** | 不扔掉任何神经元，所有神经元参与 |

```python
# 训练时
model.train()  # 启用 Dropout
output = model(x)  # 随机扔掉一些神经元

# 测试时
model.eval()   # 禁用 Dropout
output = model(x)  # 所有神经元都参与
```

### 直观解释

**类比：考试作弊 vs 真正学习**

**没有 Dropout**（作弊）：
```
学生（网络）只记住答案 → 考试（测试）换题就傻眼 ❌
```

**有 Dropout**（真正学习）：
```
每次练习随机捂住一部分笔记
→ 强迫学生理解知识而不是死记
→ 考试换题也能应对 ✓
```

---

## 数学原理

### 为什么要放大剩余元素？

```python
# 假设 dropout = 0.1 (扔掉 10%)

# 训练时：
# 只有 90% 的神经元活跃
# 期望输出 = 0.9 × x

# 测试时：
# 所有 100% 的神经元都参与
# 期望输出 = 1.0 × x

# ⚠️ 如果不放大，训练和测试的输出规模不一致！
# 解决方案：训练时将剩余元素 × 1/(1-0.1) ≈ 1.11
# 这样期望输出 = 0.9 × 1.11 × x ≈ 1.0 × x ✓
```

### Dropout 参数含义

```python
nn.Dropout(dropout=0.1)
```

| dropout 值 | 含义 | 保留比例 |
|-----------|------|---------|
| `0.0` | 不扔掉任何神经元 | 100% |
| `0.1` | 扔掉 10% | 90% |
| `0.5` | 扔掉 50% | 50% |
| `0.9` | 扔掉 90% | 10% |

**Transformer 中常用 `dropout=0.1`**（相对温和）

---

## 代码示例

### 基础示例

```python
import torch
import torch.nn as nn

# 创建 Dropout 层
dropout = nn.Dropout(p=0.5)  # 50% 概率置零

# 测试数据
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# 训练模式：随机置零
dropout.train()
print("训练模式（多次运行结果不同）:")
for i in range(3):
    print(f"  第 {i+1} 次: {dropout(x)}")

# 测试模式：不置零
dropout.eval()
print(f"\n测试模式: {dropout(x)}")
# 输出: tensor([1., 2., 3., 4., 5.])
```

### 在网络中使用

```python
import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(0.1)  # 10% dropout
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 在激活函数后应用
        x = self.fc2(x)
        return x

# 使用
model = MyNetwork()

# 训练阶段
model.train()
output_train = model(input_data)  # 应用 dropout

# 测试阶段
model.eval()
output_test = model(input_data)   # 不应用 dropout
```

---

## 总结

### 核心要点

| 问题 | 答案 |
|------|------|
| **Dropout 是什么？** | 训练时随机"关掉"一部分神经元 |
| **为什么要用？** | 防止过拟合，提高泛化能力 |
| **训练时怎么工作？** | 按概率随机置零，剩余元素放大 |
| **测试时怎么工作？** | 不置零，所有神经元参与 |
| **在 FeedForward 的位置？** | ReLU 之后，第二层线性之前 |
| **常用值？** | Transformer 中 0.1 |

### 为什么 Dropout 放在 ReLU 之后？

```python
self.dropout(torch.relu(self.linear1(x)))
```

1. 先用 ReLU 把负数过滤掉
2. 再在正数上应用 Dropout
3. 避免浪费"扔掉 0"的操作

### 与其他正则化技术对比

| 技术 | 作用 | 使用场景 |
|------|------|---------|
| **Dropout** | 随机扔神经元 | 全连接层、Transformer |
| **BatchNorm** | 归一化特征 | CNN、全连接层 |
| **L1/L2 正则** | 惩罚大权重 | 线性模型、神经网络 |
| **数据增强** | 增加训练数据 | 图像、文本 |

---

## 参考代码位置

在 Marionette 项目中：
- 文件：`discrete_diffusion/conditional_attention.py`
- 类：`FeedForward`
- 行：第 66-81 行

---

**核心思想**：让网络"不依赖任何一个神经元"，强迫学习更鲁棒的特征！
