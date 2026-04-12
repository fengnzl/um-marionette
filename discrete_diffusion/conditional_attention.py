import torch
import torch.nn as nn
import math
from einops import rearrange

'''
把一个数字（比如：现在是扩散的第 50 步）转换成一串数字密码（比如：256 个数字组成的一维向量）。

  为什么要这么做？
  - 神经网络对单个数字不敏感（50 和 51 差别太小）
  - 但对"波纹形状"很敏感（不同频率正弦波的组合可以创造独特的"指纹"）
  - 这样网络就能准确知道"现在到了哪一步"，从而决定该用多大的力气去修复噪声
'''
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建基于正弦波的“时间步（第几步扩散）”编码。
    外行理解：类似于我们把时间转化为波浪，让神经网络能通过“波峰波谷”感知这是扩散破坏模型的第几步。
    - timesteps：时间步张量，比如 tensor([50]) 表示第 50 步
    - dim：输出向量的维度，比如 256 表示输出 256 个数字
    - max_period=10000：最大周期（控制最低频率），默认 10000
    :return: 一个 [N x dim] 的信号矩阵。
    """
    # 除以 2 是因为每个频率需要 2 个数字来存储（一个 cos，一个 sin）
    half = dim // 2
    # 生成一组从高频到低频的频率值（从快到慢）
    '''
     第 1 步：torch.arange(start=0, end=half, dtype=torch.float32)
    - 生成 [0, 1, 2, ..., half-1]
    - 例如 half=4 时：[0, 1, 2, 3]

    第 2 步：/ half
    - 归一化到 [0, 1)
    - 例如：[0, 0.25, 0.5, 0.75]

    第 3 步：-math.log(max_period) * ...
    - 乘以 ln(10000) ≈ 9.21 的负数
    - 结果：[0, -2.3, -4.6, -6.9]（从 0 到负数）

    第 4 步：torch.exp(...)
    - 取指数，得到频率值
    - exp(0)=1.0（高频）, exp(-6.9)≈0.001（低频）
    - 最终频率序列：从快到慢 [1.0, 0.1, 0.01, 0.001, ...]
    '''
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    # 把具体走到第几步和各个频率相乘得出角度
    '''
    timesteps[:, None]
    - 增加一个维度，形状从 [N] 变成 [N, 1]
    - 例如 tensor([50]) → tensor([[50]])

    .float()
    - 转换为浮点数类型

    freqs[None]
    - 也增加维度，形状从 [half] 变成 [1, half]
    - 例如 [1.0, 0.1, 0.01] → [[1.0, 0.1, 0.01]]

    乘法广播
    - [N, 1] × [1, half] → [N, half]
    - 每个时间步与每个频率相乘，得到角度（弧度）
    
    时间步 = 50，频率 = [1.0, 0.1, 0.01]
    角度 = [50×1.0, 50×0.1, 50×0.01] = [50, 5, 0.5] 弧度
    '''
    args = timesteps[:, None].float() * freqs[None]
    # 一半算余弦，一半算正弦，合并成密码本 dim = -1 沿最后一个维度进行拼接
    # 因为单独一个 sin 或 cos 无法区分正负方向（例如 sin(30°) = sin(150°)）
    # 但 sin+cos 组合可以在 2D 平面上唯一确定一个角度位置
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # 如果要求的维度恰好是奇数，为了凑数就在屁股后面补个零
    # embedding[:, :1]) 取所有行，第零列
    '''
        embedding = torch.tensor([
            [0.5, 0.3, -0.2, 0.8],   # 第 1 个样本
            [0.1, -0.5, 0.9, 0.2],   # 第 2 个样本
            [-0.3, 0.7, 0.4, -0.1]   # 第 3 个样本]
        ])  # 形状: [3, 4]
        tensor([
            [0.5],    # 只取每行的第 0 列
            [0.1],
            [-0.3]
        ])  # 形状: [3, 1] ← 注意是二维的！
    '''
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SiLU(nn.Module):
    """
    SiLU 激活函数（也叫 Swish）。它跟 ReLU 类似也是用来“转弯”的，但它的线条更平滑。
    公式就是 x 乘以 x的Sigmoid概率。
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    """
    经典 Transformer 模型中自带的【位置编码系统】。
    因为 Transformer 是个“近视眼”，它一次性看所有的词，缺乏“前后顺序”的概念。
    所以我们要用这套机器给每个排位的词烙印上一个“坐标波纹”。
    """
    def __init__(self, d_model, max_len=5000):
        '''
        作用：初始化位置编码
        - d_model：编码维度（比如 256）
        - max_len=5000：最大支持序列长度（可以处理 5000 个位置的编码）
        '''
        super(PositionalEncoding, self).__init__()
        # 创建一个空的大画布，容纳可能的最长句子 torch.Size([5000, 256])
        pe = torch.zeros(max_len, d_model)
        # 生成位置编号 [0, 1, 2, ..., max_len-1]  形状 [5000] .unsqueeze(1) 变成 [5000, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 用指数和对数生成频率缩放因子，div_term 平分 d_model， 
        # 从而对偶数位置和奇数位置可以分别设置 cos 和 sin 数据
        '''
        第 1 步：torch.arange(0, d_model, 2)
        - 生成偶数索引 [0, 2, 4, 6, ...]
        - 假设 d_model=256：[0, 2, 4, ..., 254]，共 128 个

        第 2 步：.float() * (-math.log(10000.0) / d_model)
        - 计算 -ln(10000) / 256 ≈ -9.21 / 256 ≈ -0.036
        - 每个偶数索引乘以这个值：[0, -0.072, -0.144, ...]

        第 3 步：torch.exp(...)
        - 取指数，得到频率：[1.0, 0.93, 0.87, ...]
        - 从快到慢的频率序列
        '''
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数位置上填入正弦波纹
        '''
        pe[:, 0::2] = torch.sin(position * div_term)
        在偶数列（0, 2, 4, ...）填入正弦波
        '''
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数列位置上填入余弦波纹
        pe[:, 1::2] = torch.cos(position * div_term)
        # 调整形状以适应后续使用 
        # 原始 [max_len, d_model] = [5000, 256]
        # unsequeeze(0) 在第 0 维增加维度 [1, 5000, 256]
        # transpose(0, 1) 交换第 0 维和第 1 维 [5000, 1, 256]
        # 后续使用时，输入 x 的形状通常是 [seq_len, batch, d_model]
        # - 调整后 pe 的形状 [seq_len, 1, d_model] 可以广播相加
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 把这段神仙波纹密码注册为底层的固定参考坐标系（不参与训练修改）
        '''
        作用：将 pe 注册为模型的"缓冲区"（buffer）
        什么是 buffer？
        - 它是模型的一部分（会被保存到模型文件）
        - 但不是参数（不会被梯度更新）
        - 会在设备迁移时自动移动（GPU ↔ CPU）

        为什么是 buffer 而不是参数？
        - 位置编码是固定的，不需要训练
        - 只需要预先计算好，然后复用
        '''
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 真正的运行时：把本来单纯的输入 x 和刚才计算的对应长度的位置坐标波纹强行加起来混合
        '''
        x.size(0)  输入序列长度（比如 10）
        self.pe[:x.size(0), :] 取前 10 行的位置编码
        x + ...    相加（广播机制） 
        '''
        return x + self.pe[:x.size(0), :]

class FeedForward(nn.Module):
    """
    Transformer 每思考完一次注意力的牵连之后，都会走到这间【深度反思房】（前馈层）。
    也就是一个简单朴素的两层全连接 MLP，用来消化和提炼刚才注意力的信息。
    """
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        # 先把思考维度暴涨（比如从 256 撑到 1024）寻找复杂规律
        # 1. 你看到一个信息，先展开写下很多想法和联想（膨胀到 1024 维） 膨胀阶段：有更多空间探索复杂的非线性关系
        # 2. 然后你总结提炼，把最重要的内容浓缩回去（压缩到 256 维） 压缩阶段：只保留最重要的信息，丢弃噪声
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        # 寻找完之后再压缩回正常的 256 维度
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        # 过一层线性 -> ReLU激活 -> 随机忘掉一部分防死记硬背(dropout) -> 压回原维
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    """
    多头注意力上帝（Multi-Head Attention）本尊。
    也就是著名论文"Attention is all you need"里的核心：它帮网络搞清楚“整句话里面，谁该重点观察谁”。
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 总维数必须能被脑袋个数整除，方便给每个脑袋平分工作量
        # d_model: 每个词的向量维度（比如 256）
        # num_heads: 有几个"注意力头"（比如 4 个）
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        # 切分给每个小脑袋（head）分配多少维度
        self.d_k = d_model // num_heads  

        # 准备著名的 Q(查询), K(钥匙), V(价值) 三把刷子
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        # 最后一个总结收尾转换
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        #假设  query: [batch_size, seq_len, d_model] = [2, 10, 256]
        batch_size = query.size(0) 
        
        # 把传进来的线索生成出各自的 Q, K, V
        # 并且运用 reshape(view) 和 transpose 把大工程拆散给好几个 num_heads 小分队同步干活
        #  query 从 [batch, seq_len, d_model] -》 [batch, seq_len, num_heads, d_K] 最后转换成 [batch, num_heads, seq_len, d_k]
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 核心机密计算：Q和K握手(内积)，除以个根号缩放因子防止数太大，得出来的就是它们俩直接相互的“在意程度分数”
        # key.transpose(-2, -1) — 转置 Key
        # torch.matmul(query, key_T) — 矩阵乘法
        # / math.sqrt(self.d_k) — 缩放
        # A × B 要求 A 的最后一维 = B 的倒数第二维
        # 计算每个 Query 和每个 Key 的点积（相似度）这个 Query 和这个 Key 有多相关
        # d_k 很大时，点积会变得很大，导致 softmax 饱和度会消失，从而需要进行缩放 更小更稳定
        # scores = query @ key.T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 遮罩(Mask)：有些没发生的事情（凑数的0），我们强行把它们的在意分数设为“负无穷”阻断视线
        # 便于计算：统一序列长度，可以批处理
        # 忽略噪声：-inf → softmax → 0，模型完全忽略填充位置
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # 对齐下四维形态 (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 用 SoftMax 强行把所有在意分数变成总和为 1 的“概率百分比”
        # 每个 Query 应该关注每个 Key 的程度
        attn = torch.softmax(scores, dim=-1)
        # 用概率加权求和 V 用这个在意百分比，去把所有包含实质信息的 V(价值) 给它搅拌浓缩成一段精华 (context)
        # 加权求和 每个 Query 应该关注每个 Key 的程度
        context = torch.matmul(attn, value)
        
        # 小分队汇总：把所有头的精华拼回原来的老样子
        # 因为之前 query 使用 transpose 交换位置，因此现在需要交换位置
        # contiguous 确保内存连续 创建一个内存连续的副本
        # 通过 view 将 shape 变成与 query 一致
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 做一下收尾汇报
        return self.out_linear(context)


class EncoderLayer(nn.Module):
    """
    编码器组建（通常包含：自己和自己交流的注意力层 + 深度反思房）
    """
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # 一套自说自话的注意力体系
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        # 每做完一步都需要一次“深呼吸（归一化）”，平复激动异常的数字
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 只关注自己
        # 1. 拿自己（src）当Q、当K、当V去注意自己
        src2 = self.self_attn(src, src, src, src_mask)
        # 融入原来的记忆残差并深呼吸平复
        src = self.norm1(src + self.dropout(src2))

        # 2. 去反思房提炼
        src2 = self.feed_forward(src)
        src = self.norm2(src + self.dropout(src2))
        return src

class DecoderLayer(nn.Module):
    """
    解码器（比起上面的兄弟，它多了一步：不仅注意自己，还要分出目光去注意上司（条件）交代的任务）
    """
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # 多了这一层：交互注意力机制（一半放自己，一半放外部线索）
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, cond, tgt_mask=None, cond_mask=None):
        # 关注自己 + 外部条件
        # tgt：被污染的序列（需要被修复）
        # cond: 外部条件（时间、用户信息等）

        # 第一步：自己整理自己的队伍情况 先整理自己（自注意力）
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # 第二步：自己的疑惑做Q去查询，把外部环境条件(cond)当成图书馆的 K和V 去借阅融合力量！
        # 再去查询外部条件（交叉注意力）
        tgt2 = self.cross_attn(tgt, cond, cond, cond_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # 第三步：去深度反思房进一步浓缩消化
        tgt2 = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt

# 很多个 Encoder 叠在一起组成了个楼层
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cond, cond_mask=None):
        # 爬楼：一层接一层淬炼
        for layer in self.layers:
            cond = layer(cond, cond_mask)
        return self.norm(cond)

# 很多个 Decoder 叠在一起
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, cond, tgt_mask=None, cond_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, cond, tgt_mask, cond_mask)
        return self.norm(tgt)


class Transformer(nn.Module):
    """
    为了满足“离散扩散(Discrete Diffusion)”定制的全套 Transformer 调度站。
    不仅接收被污染马赛克打乱的序列，还接收扩散进行到了哪一步、以及外部环境影响等极度繁杂的信号，全部揉进黑匣子里翻译。
    """
    def __init__(self, tgt_vocab_size,num_spectial,type_classes,poi_classes, src_vocab_size=100, d_model=256, num_layers=4, num_heads=4, dim_feedforward=1024, dropout=0.1, max_len=3000):
        super(Transformer, self).__init__()
        # 准备事件类别本身的字典查找器
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # 准备座位上的序列座号编号记录
        self.positional_encoding = nn.Embedding(max_len, d_model)
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))
        
        # 挂起核心解码大楼主楼
        self.decoder = Decoder(num_layers, d_model, num_heads, dim_feedforward, dropout)
        
        # 最后一层：把大脑思考好的脑电波重新翻译回外界能看懂的“各大词汇打分（Logit分类输出）”
        self.output_layer = nn.Linear(d_model, tgt_vocab_size-2) # 抛弃掉专门做掩码的两个特殊词标志
        self.d_model = d_model
        
        # 处理扩散步骤的小加工厂（比如把 50步 的微弱信号转化为极其复杂的多维感官）
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*4),
            SiLU(),
            nn.Linear(self.d_model*4, self.d_model),
        )
        
        # 为事件打上它是个粗心词汇、地理大坐标词汇等“词性烙印”的分配器
        self.token_type_layer = nn.Embedding(3,self.d_model)
        # 用来把合并肥胖的特征重新压榨出精华尺寸的榨汁机
        self.input_projection = nn.Linear(self.d_model*2, self.d_model)
        
        self.num_spectial = num_spectial
        self.type_classes = type_classes
        self.poi_classes = poi_classes

    def forward(self, x, cond_emb, t, batch):
        # 输入：
        # x: 被污染的序列索引
        # cond_emb: 外部条件的编码
        # t: 当前是第几步扩散
        # batch: 其他信息（mask 等）
        """主入口调度"""

        # 第一步：时间步编码：把"第50步"转换成向量
        diffusion_step_emb = self.time_embed(timestep_embedding(t, self.d_model))
        
        # 获取序列座号牌
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]

        # 词嵌入 + 位置编码 + 时间步编码，三者相加
        x = self.positional_encoding(position_ids) + self.tgt_embedding(x) + diffusion_step_emb.unsqueeze(1).expand(-1, seq_length, -1)

        # 把分类和地理遮罩提取出来备用
        tgt_mask = (batch.category_mask + batch.poi_mask).bool()

        # 核心拼装2：搞清楚每个字到底代表着地理还是类别词性，然后把它对应的词性特殊光环加上去
        # 词性编码：区分是"类别词"还是"地点词"
        token_type = batch.category_mask + batch.poi_mask * 2
        token_type_emb = self.token_type_layer(token_type)

        # 把刚才的混合特征与词性强行缝在一起然后用推土机强压扁回标准体型
        x = self.input_projection(torch.cat([x,token_type_emb],dim=-1))

        # 送进核心大楼疯狂解码运转，不仅注意自己(x)，还得跟过去的外部环境条件(cond)去比对参照
        # 送入 Decoder（包含 self-attn + cross-attn + feedforward）
        output = self.decoder(tgt=x, cond=cond_emb, tgt_mask=tgt_mask, cond_mask=batch.mask)
        
        # 用最后一层线性皮条把复杂向量转行为“大字典里所有可能的事件哪一个应该发生的百分比分数”
        # 转换成每个类别的概率
        output = self.output_layer(output)

        # 将最后的数组形状为了符合外界库规范拧一下顺序
        output = rearrange(output, 'b l v -> b v l')
        
        return output

"""
========================================================================
【给外行新手的通俗讲解】 -- discrete_diffusion/conditional_attention.py
========================================================================

1. 这里面这一大坨长篇累牍全都是在干嘛？
   - 这里在手工从底层搭建大名鼎鼎的【Transformer 模型】也就是 ChatGPT 的老祖宗核引擎。
   - 跟前面不同的是，这里不用 PyTorch 自带省事的，而是从零（Multi-Head多头机制一步一个部件）人工搭建。
   - 目的只有一个：原厂自带的是为了翻译英语做准备的。
   - 而我们这里研究的是：时间点事件、地点事件、这事被噪声破坏到了第 50 步、之前这人有没有去过电影院等乱七八糟混杂的“外星人乱炖英语”。
   - 手填底层让我们有权利做手势：把外界的环境魔法(Condition)、破坏程度时间魔法(t)、甚至分门别类的掩码全部注入它的血液。

2. Attention 机制是怎么发挥魔力的？
   - Transformer 抛弃了所谓流水线的逐步理解。它是用一种全局眼光去看的：不管这个事件多长，我在第 3 步去吃肯德基，第 8 步去看电影。这种事它怎么看出规律？
   - Attention 用数学办法实现了：让它拿着“看电影”这件事为查询词，跑到过往所有事（比如吃饭、喝水、坐公交前），给每一个元素当场发一份百分制问卷调查（这就是算 Attention 分数）。大家发现“吃肯德基”和“看电影”在历史记录里有神秘联系，两人就会擦出火花握手交融，从而“醍醐灌顶”得出这步该做些啥！这正是人类思考所谓“触景生情和因果牵连”的数学展现！
========================================================================
"""