# Transformer 代码详解

> Marionette 项目中的核心 Transformer 模型完整讲解

---

## 📚 目录

1. [Transformer 类概览](#transformer-类概览)
2. [初始化方法详解](#1-__init__-方法初始化模型)
3. [核心组件详解](#核心组件详解)
4. [前向传播详解](#2-forward-方法前向传播)
5. [完整数据流](#完整数据流图)
6. [总结](#总结)

---

## Transformer 类概览

```python
class Transformer(nn.Module):
    """
    为了满足"离散扩散"定制的全套 Transformer 调度站。
    不仅接收被污染马赛克打乱的序列，还接收扩散进行到了哪一步、
    以及外部环境影响等极度繁杂的信号，全部揉进黑匣子里翻译。
    """
```

**作用**：Marionette 项目的核心模型，处理离散扩散的空间部分（事件类别和 POI）

**特点**：
- 结合了词嵌入、位置编码、时间步嵌入、token 类型嵌入
- 使用 Transformer Decoder 架构
- 专门为离散扩散任务设计

---

## 1. `__init__` 方法：初始化模型

### 方法签名

```python
def __init__(
    self,
    tgt_vocab_size,           # 目标词汇表大小（所有可能的类别+POI总数）
    num_spectial,             # 特殊标记数量（如 PAD、MASK 等）
    type_classes,             # 事件类别数量（如 9 大类）
    poi_classes,              # POI 类别数量（如 3477 个地点）
    src_vocab_size=100,       # （未使用的参数）
    d_model=256,              # 模型维度
    num_layers=4,             # Decoder 层数
    num_heads=4,              # 多头注意力的头数
    dim_feedforward=1024,     # Feed Forward 的隐藏层维度
    dropout=0.1,              # Dropout 比例
    max_len=3000              # 最大序列长度
):
```

### 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `tgt_vocab_size` | 目标词汇表大小 | 3477 + 9 + 4 = 3490 |
| `num_spectial` | 特殊标记数量 | 4 (SOS, PAD, MASK_CAT, MASK_POI) |
| `type_classes` | 事件类别数量 | 9 |
| `poi_classes` | POI 类别数量 | 3477 |
| `d_model` | 模型维度 | 256 |
| `num_layers` | Decoder 层数 | 4 |
| `num_heads` | 多头注意力头数 | 4 |
| `dim_feedforward` | FFN 隐藏层维度 | 1024 |
| `dropout` | Dropout 比例 | 0.1 |
| `max_len` | 最大序列长度 | 3000 |

---

## 核心组件详解

### 组件 1：词嵌入层

```python
self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
```

**作用**：将离散的类别 ID 转换成连续的向量

**示例**：
```python
# 输入：类别 ID
input_id = 42

# 词嵌入
embedding = self.tgt_embedding(input_id)
# 输出：[256] 维向量
```

---

### 组件 2：位置编码

```python
self.positional_encoding = nn.Embedding(max_len, d_model)
self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))
```

**作用**：给序列中的每个位置添加位置信息

**示例**：
```python
# 位置 ID
position_ids = [0, 1, 2, 3, 4]  # 序列中的位置编号

# 位置嵌入
pos_emb = self.positional_encoding(position_ids)
# 输出：[seq_len, 256]
```

---

### 组件 3：Decoder 核心

```python
self.decoder = Decoder(num_layers, d_model, num_heads, dim_feedforward, dropout)
```

**作用**：多层 Transformer Decoder，负责序列生成

**架构**：
```
Decoder (4 层)
  ├─ Decoder Layer 1
  │   ├─ Self-Attention
  │   ├─ Cross-Attention (关注条件)
  │   └─ Feed Forward
  ├─ Decoder Layer 2
  ├─ Decoder Layer 3
  └─ Decoder Layer 4
```

---

### 组件 4：输出层

```python
self.output_layer = nn.Linear(d_model, tgt_vocab_size - 2)
```

**作用**：将模型输出转换成词汇表大小的 logits

**为什么减 2？**
```python
tgt_vocab_size 包含了 2 个特殊标记：
- MASK_CAT (num_classes - 2)
- MASK_POI (num_classes - 1)

输出时不需要预测这两个标记，所以减去 2
```

---

### 组件 5：时间步嵌入

```python
self.time_embed = nn.Sequential(
    nn.Linear(self.d_model, self.d_model * 4),
    SiLU(),
    nn.Linear(self.d_model * 4, self.d_model),
)
```

**作用**：将时间步编码成高维特征

**处理流程**：
```
时间步 t (如 50)
    ↓
timestep_embedding(t, 256)  # 正弦波编码
    ↓
[1, 256]
    ↓
Linear(256, 1024) + SiLU + Linear(1024, 256)  # MLP 处理
    ↓
[1, 256]  # 高级时间特征
```

---

### 组件 6：Token 类型嵌入

```python
self.token_type_layer = nn.Embedding(3, self.d_model)
self.input_projection = nn.Linear(self.d_model * 2, self.d_model)
```

**作用**：区分不同类型的 token（类别 vs POI）

**token_type 的含义**：

| category_mask | poi_mask | token_type | 含义 |
|---------------|----------|------------|------|
| 0 | 0 | 0 | 特殊标记（SOS、PAD） |
| 1 | 0 | 1 | 类别 token |
| 0 | 1 | 2 | POI token |

**示例**：
```python
# 序列中的 token 类型
token_type = [0, 1, 1, 2, 2]
#             ↑  ↑  ↑  ↑  ↑
#           SOS 类 类 POI POI

# 获取类型嵌入
type_emb = self.token_type_layer(token_type)
# 输出：[5, 256]
```

---

## 2. `forward` 方法：前向传播

### 方法签名

```python
def forward(self, x, cond_emb, t, batch):
    """主入口调度"""
```

### 参数说明

| 参数 | 形状 | 作用 |
|------|------|------|
| `x` | `[batch, seq_len]` | 输入序列（类别 ID） |
| `cond_emb` | `[batch, seq_len, d_model]` | 条件嵌入（外部环境信息） |
| `t` | `[batch]` | 扩散时间步 |
| `batch` | - | 包含 mask 等信息的 Batch 对象 |

---

### 步骤 1：获取时间步嵌入

```python
diffusion_step_emb = self.time_embed(timestep_embedding(t, self.d_model))
```

**处理流程**：
```python
t = [50, 100]  # 两个样本，分别在 50 步和 100 步
      ↓
timestep_embedding(t, 256)  # 正弦波编码
      ↓
[2, 256]
      ↓
self.time_embed(...)  # MLP 处理
      ↓
[2, 256]  # 输出时间嵌入
```

---

### 步骤 2：获取位置编码

```python
seq_length = x.size(1)
position_ids = self.position_ids[:, :seq_length]
```

**示例**：
```python
x.shape = [2, 5]  # batch=2, seq_len=5
seq_length = 5
position_ids = [[0, 1, 2, 3, 4]]  # 只取前 5 个位置
```

---

### 步骤 3：核心拼装 1 - 融合多种信息

```python
x = self.positional_encoding(position_ids) + \
    self.tgt_embedding(x) + \
    diffusion_step_emb.unsqueeze(1).expand(-1, seq_length, -1)
```

**详细拆解**：

#### 3.1 词嵌入
```python
self.tgt_embedding(x)
# [batch, seq_len] → [batch, seq_len, d_model]
# [2, 5] → [2, 5, 256]
```

#### 3.2 位置编码
```python
self.positional_encoding(position_ids)
# [1, seq_len] → [1, seq_len, d_model]
# [1, 5] → [1, 5, 256]（广播到 [2, 5, 256]）
```

#### 3.3 时间步嵌入
```python
diffusion_step_emb.unsqueeze(1).expand(-1, seq_length, -1)
# [batch, d_model] → [batch, 1, d_model] → [batch, seq_len, d_model]
# [2, 256] → [2, 1, 256] → [2, 5, 256]
```

#### 3.4 三者相加
```python
x = 词嵌入 + 位置编码 + 时间嵌入
# [2, 5, 256] + [2, 5, 256] + [2, 5, 256]
# = [2, 5, 256]
```

**可视化**：
```
每个位置的特征 = 词义 + 位置 + 时间

位置 0: [词嵌入0] + [位置0编码] + [时间嵌入]
位置 1: [词嵌入1] + [位置1编码] + [时间嵌入]
位置 2: [词嵌入2] + [位置2编码] + [时间嵌入]
位置 3: [词嵌入3] + [位置3编码] + [时间嵌入]
位置 4: [词嵌入4] + [位置4编码] + [时间嵌入]
```

---

### 步骤 4：生成 mask

```python
tgt_mask = (batch.category_mask + batch.poi_mask).bool()
```

**作用**：标记哪些位置是有效的（非 padding）

**示例**：
```python
batch.category_mask = [[1, 1, 0],  # 样本 0：前 2 个位置是类别
                       [1, 1, 1]]  # 样本 1：3 个位置都是类别

batch.poi_mask = [[0, 0, 1],      # 样本 0：第 3 个位置是 POI
                   [0, 0, 1]]      # 样本 1：第 3 个位置是 POI

tgt_mask = [[1, 1, 1],  # 样本 0：所有位置都有效
            [1, 1, 1]]  # 样本 1：所有位置都有效
```

---

### 步骤 5：核心拼装 2 - 添加 token 类型嵌入

```python
token_type = batch.category_mask + batch.poi_mask * 2
token_type_emb = self.token_type_layer(token_type)
```

**token_type 计算规则**：

| category_mask | poi_mask | token_type | 含义 |
|---------------|----------|------------|------|
| 0 | 0 | 0 | 特殊标记 |
| 1 | 0 | 1 | 类别 token |
| 0 | 1 | 2 | POI token |

**示例**：
```python
category_mask = [0, 1, 1, 0, 0]
poi_mask = [0, 0, 0, 1, 1]

token_type = [0, 1, 1, 2, 2]
#             ↑  ↑  ↑  ↑  ↑
#           特殊 类 类 POI POI
```

---

### 步骤 6：拼接和投影

```python
x = self.input_projection(torch.cat([x, token_type_emb], dim=-1))
```

**处理流程**：
```python
# 拼接
torch.cat([x, token_type_emb], dim=-1)
# [batch, seq_len, d_model] + [batch, seq_len, d_model]
# [2, 5, 256] + [2, 5, 256]
# = [2, 5, 512]  # 维度翻倍！

# 投影压缩
self.input_projection(...)  # Linear(512, 256)
# [2, 5, 512] → [2, 5, 256]
```

---

### 步骤 7：Decoder 处理

```python
output = self.decoder(tgt=x, cond=cond_emb, tgt_mask=tgt_mask, cond_mask=batch.mask)
```

**Decoder 的工作**：
1. **Self-Attention**：让序列内的元素互相交互
2. **Cross-Attention**：让序列关注外部条件（cond_emb）

**示例**：
```python
# 输入
x = [2, 5, 256]        # 目标序列
cond_emb = [2, 5, 256]  # 条件嵌入

# 输出
output = [2, 5, 256]  # 经过多层处理后的特征
```

---

### 步骤 8：输出层

```python
output = self.output_layer(output)
```

**处理流程**：
```python
# Decoder 输出
output = [2, 5, 256]

# 输出层
self.output_layer(output)  # Linear(256, vocab_size - 2)
# [2, 5, 256] → [2, vocab_size - 2, 5]
```

---

### 步骤 9：调整维度顺序

```python
output = rearrange(output, 'b l v -> b v l')
```

**处理流程**：
```python
# rearrange 前
output.shape = [batch, seq_len, vocab_size]
               [2, 5, 3477]

# rearrange 后
output.shape = [batch, vocab_size, seq_len]
               [2, 3477, 5]
```

**为什么要调整？**
- PyTorch 的 `CrossEntropyLoss` 期望输入是 `[batch, num_classes, seq_len]`

---

## 完整数据流图

```
输入 x [batch, seq_len]
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
tgt_embedding(x)              position_ids
    │                                 │
    ▼                                 ▼
[batch, seq, 256]           positional_encoding()
                                       │
                                       ▼
                                 [1, seq, 256]
    │                                 │
    │                ┌────────────────┘
    │                │
    ▼                ▼
timestep_embedding(t)    (时间步扩展到每个位置)
    │
    ▼
time_embed()
    │
    ▼
[batch, 256]
    │
    └──────────┬──────────────────────┐
               │                      │
               ▼                      ▼
          求和融合              token_type_emb
          [batch, seq, 256]            │
               │                      │
               │                      ▼
               │              [batch, seq, 256]
               │                      │
               └──────────┬───────────┘
                          ▼
                    torch.cat (拼接)
                          │
                          ▼
                   [batch, seq, 512]
                          │
                          ▼
                  input_projection
                          │
                          ▼
                   [batch, seq, 256]
                          │
                          ▼
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
         Decoder(x,              cond_emb
              │                           │
              └─────────────┬─────────────┘
                            ▼
                      [batch, seq, 256]
                            │
                            ▼
                      output_layer
                            │
                            ▼
                   [batch, vocab, seq]
                            │
                            ▼
                       rearrange
                            │
                            ▼
                   [batch, vocab, seq]
```

---

## 总结

### 组件功能表

| 组件 | 作用 | 输入 → 输出 |
|------|------|-------------|
| **tgt_embedding** | 词嵌入 | `[batch, seq] → [batch, seq, 256]` |
| **positional_encoding** | 位置编码 | `[seq] → [seq, 256]` |
| **time_embed** | 时间步嵌入 | `[batch] → [batch, 256]` |
| **token_type_layer** | Token 类型 | `[batch, seq] → [batch, seq, 256]` |
| **input_projection** | 特征融合 | `[batch, seq, 512] → [batch, seq, 256]` |
| **decoder** | 核心解码 | `[batch, seq, 256] → [batch, seq, 256]` |
| **output_layer** | 输出层 | `[batch, seq, 256] → [batch, vocab, seq]` |

### 核心思想

1. **多信息融合**：
   - 词嵌入（内容）
   - 位置编码（顺序）
   - 时间步嵌入（扩散阶段）
   - Token 类型嵌入（类别/POI）

2. **Transformer Decoder**：
   - Self-Attention：序列内元素交互
   - Cross-Attention：结合外部条件

3. **灵活输出**：
   - 预测每个位置的事件类别和 POI
   - 支持变长序列（通过 mask）

### 使用场景

```python
# 训练时
# 输入：被污染的序列
output = model(x_noisy, cond_emb, t, batch)
loss = cross_entropy(output, x_clean)

# 采样时
# 输入：纯噪声序列
for t in range(T, 0, -1):
    output = model(x_noisy, cond_emb, t, batch)
    x_noisy = sample(output)
```

---

**文件位置**：`discrete_diffusion/conditional_attention.py` (第 215-283 行)
