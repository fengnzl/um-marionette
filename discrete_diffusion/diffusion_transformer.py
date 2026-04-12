#update matrix for larger dataset
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from torch.nn import TransformerEncoderLayer, TransformerEncoder 
from discrete_diffusion.conditional_attention import Transformer
from datamodule import Batch
from einops import rearrange

# 极小值保护伞防崩溃
eps = 1e-8

def sum_except_batch(x, num_dims=1):
    # 除了批次维度，把其它的全部挤压平铺然后求和
    '''
        x = torch.tensor([
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3]
            ],
            [
                [4, 4, 4, 4],
                [5, 5, 5, 5],
                [6, 6, 6, 6]
            ]
        ])
        x.shape = torch.Size([2, 3, 4])
        x.shape[:1] = (2,)
        *x.shape[:1] = 2
        y = x.reshape(2, -1)
        y.shape = torch.Size([2, 3 * 4]) = torch.Size(2, 12)
        y = tensor([
            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]
        ])
        sum(-1) 沿最后一个维度求和
        y.sum(-1) = tensor([24, 60])
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    # 假设 p 为概率。a 一般为 log(p). a.exp() = e^(log(p)) = p
    # a 被理解为 exp(input) 的结果
    # 数学小手段：当你想算 log(1 - a) 时，因为直接算容易出现下溢出的数值黑洞崩溃。
    # 加 1e-40 位了避免 a.exp()为1的情况，从而导致 torch.log(0) -inf（负无穷大），会导致计算崩溃
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    # 想算 log(exp(a) + exp(b)) 怎么防爆?
    # 假如 a 为 1000 b 为 999
    # result = torch.log(torch.exp(torch.tensor(1000.0)) + torch.exp(torch.tensor(999.0)))
    # exp(1000) = 无穷大！💥
    # 结果：NaN 或 inf
    # log(exp(a) + exp(b)) = log(exp(max) * (exp(a - max) + exp(b - max)))
    # = log(exp(max)) + log(exp(a - max) + exp(b - max)) = max + log(exp(a - max) + exp(b - max))
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    # 一种神奇的钳子。从配置字典表（a）里，根据当前的扩散时机（t 比如抽第5步），准确地钳出第 5 步的数据。
    b, *_ = t.shape
    out = a.to(t.device).gather(-1, t)
    # 最后把它做成能够和别的阵列随便拼接缝合的薄片形状
    # 假设 x_shape torch.Size([3, 4, 4, 4]) len(x_shape) - 1 = 4 - 1 = 3
    # (1, ) * 3 = (1, 1, 1)
    # *(1, 1, 1) = 1 1 1
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    # 给定预测概率分布，发生真实标签这个事件的对数概率
    # 1. log_x_start.exp() → 把对数 one-hot 转回普通 one-hot（如 [1,0,0]）
    # 2. 乘以 log_prob → 只保留真实类别位置的对数概率
    # 3. 求和 → 提取出真实类别的对数概率值
    # 假设我们有3个类别（看电影、吃饭、唱歌），现在已知：类别 0（看电影）→ one_hot = [1, 0, 0]
    # [log(1), log(很小的数), log(很小的数)] 如 log_x_start ≈ [0, -70, -70]（代表"看电影"）
    # log_prob = [-2.3, -1.6, -3.1]（模型预测的对数概率）

    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    """
    【类别独热打码化法】：给个编号 x=3（比如发生类别为看电影）。
    将它转换成一排 0 和 1 (0 0 0 1 0)，然后立刻拿去对数黑洞漂白一下防爆。
    """
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    # 1. 变成 one-hot 编码
    x_onehot = F.one_hot(x, num_classes)
    # 2. 调整维度顺序
    # 假设 x.size 为 torch.Size([3,4]) len(x.size()) = 2，tuple(range(1, len(x.size()) = (1,)
    # permute_order = (0, -1, 1)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    # # 3. 取对数 而且防备极端 0 崩溃情况 clamp 住
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    # 反向过程：看到哪一个位面分最大，直接挑最大概率认祖归宗还原成数字编号。
    # dim 1 代表是事件分类
    return log_x.argmax(1)


def alpha_schedule(
    time_step,
    att_1 = 0.99999,
    att_T = 0.000009,
    ctt_1 = 0.000009,
    ctt_T = 0.99999,
    type_classes=9,
    poi_classes=381
):
    """
    【离散空间毁坏参数时间表】：这是空间模型的心血管。
    它极为变态的定义了：“在推演损坏时，每一步把某件本来明确的事，瞎改成其他不同事情的概率”。
    外行理解：你第一步把“看电影”弄脏，你大概有 0.999 几率还是保留它是“看电影”(att)，
    但是有极小的几率它突然变成了无中生有的乱码(ctt)或者变成其他的干扰项(btt)。随着步数越走越高，
    它突变成乱码其他东西的比例急剧放大成主导。
    整个函数里布满了极复杂的人肉人工干预设计的渐变区间和拼接操作，确保破坏从头到脚是非常连贯有逻辑的。
    """
    # 前 4/5：缓慢损坏 后 1/5：快速崩坏
    sep=5
    sep_1=sep-1
    # 保真率 构造保留自身真理信息的衰减几率 att（随着步数越长，下降越猛直到完全保不住） 构造保真率 att（从高 → 低）
    # np.arange(n) / (n-1) * (end - start) + start 生成从 start 到 end，共 n 个数的等差数列
    # time_step*sep_1//sep 是前一部分的长度 相当于 前 4/5 生成 从 0.99999 到 0.0001 的等差数列
    # time_step-time_step*sep_1//sep 是后一部分的长度 后 1/5 生成 从 0.00009 到 0.000009 的等差数列
    att= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.0001 - 0.99999) + 0.99999,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.000009- 0.00009) + 0.00009))
    # 初始 α_bar_0 = 1
    att = np.concatenate(([1], att))
    # 计算 α_t = α_bar_t / α_bar_{t-1}
    at = att[1:]/att[:-1]

    # 一些次要信息的独立衰减序列 att1  构造另一组保真率 att1（给 type_classes 用）
    att1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.9999 - 0.99999) + 0.99999,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.000009- 0.9999) + 0.9999))
    att1 = np.concatenate(([1], att1))
    at1 = att1[1:]/att1[:-1]

    # 毁容率 构造无中生有乱搞信息的上升速率 ctt （起初微小，后期霸榜）从低 → 高
    # 前 4/5：几乎为 0 后 1/5：直接飙升到 0.9999
    ctt= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.00009 - 0.000009) + 0.000009,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.9999- 0.0001) + 0.0001))
    ctt = np.concatenate(([0], ctt))

    one_minus_ctt = 1 - ctt # （1）没毁容的累积概率
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1] # （2）没毁容的瞬时概率
    ct = 1-one_minus_ct # （3）毁容的瞬时概率

    ctt1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.00009 - 0.000009) + 0.000009,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.9998- 0.00009) + 0.00009))
    ctt1 = np.concatenate(([0], ctt1)) 
    one_minus_ctt1 = 1 - ctt1 
    one_minus_ct1 = one_minus_ctt1[1:] / one_minus_ctt1[:-1]
    ct1 = 1-one_minus_ct1 

    # 将前后边界生生固定收拢进数组
    # 保真率 att 最后一步 = 1
    # 毁容率 ctt 最后一步 = 0 让整个扩散的概率分布
    # 开头干净、中间损坏、结尾收敛稳定不爆炸
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    att1 = np.concatenate((att1[1:], [1]))
    ctt1 = np.concatenate((ctt1[1:], [0]))
    
    # 因为既然是变质，变成别的选项，那总类变出多少总有个比例，所以拿抛给类别的比重算 btt
    btt1 = (1-att1-ctt1) / type_classes
    # 变节率 变节概率 = 剩下的所有概率 = 1 - 保真 - 毁容
    btt2 = (1-att-ctt)

    bt1 = (1-at1-ct1) / type_classes 
    btt2 = np.concatenate(([0], btt2))
    one_minus_btt2 = 1 - btt2
    one_minus_bt = one_minus_btt2[1:] / one_minus_btt2[:-1]
    bt = 1-one_minus_bt
    btt2 = (1-att-ctt)/poi_classes

    # 把各种离散分段人工拼接粘在一起，并确保不能掉进 0 以下
    # 前 4/5：用 at, bt, ct 规则 后 1/5：用 at1, bt1, ct1 规则
    bt=np.concatenate((bt[:time_step*sep_1//sep],at1[time_step*sep_1//sep:]/poi_classes))
    at=np.concatenate((at[:time_step*sep_1//sep],(1-ct-bt*poi_classes)[time_step*sep_1//sep:])).clip(min=1e-30)
    ct=np.concatenate(((1-at-bt)[:time_step*sep_1//sep],ct[time_step*sep_1//sep:])).clip(min=1e-30)

    # 就这样，产出了 12 道各种概率走势的表头清单下发使用。
    return at,at1, bt,bt1, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 


class ConditionEmbeddingModel(nn.Module):
    """
    负责翻译环境外部条件的中间层机构。
    像前面所讲，帮预测的时候多开一个天神视野。
    """
    def __init__(
        self,
        cond_token_num = 200, 
        emb_dims = 256,
        num_condition_types = 6,
        max_position_embeddings = 3000
    ):
        super().__init__()
        self.token_num = cond_token_num
        self.emb_dims = emb_dims
        self.num_condition_types = num_condition_types + 1 # 把一天里面的小时这种时间刻度单拉出来单独做额外考量条件加进去
        self.max_position_embeddings = max_position_embeddings
        self.encoder = nn.Embedding(self.token_num, self.emb_dims)
        self.input_up_proj =  nn.Sequential(
            nn.Linear(self.num_condition_types * self.emb_dims, self.emb_dims),
            nn.ReLU(),
            nn.Linear(self.emb_dims, self.emb_dims)
        )
        # 位置编码：记录序列中每个位置
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.emb_dims)
        # 预先生成位置ID [0, 1, ..., 2999] expand 缓冲区 【1， 3000】
        self.register_buffer("position_ids", torch.arange(self.max_position_embeddings).expand((1, -1)))
        # Transformer：深度处理这些条件
        '''
        Transformer 的作用：
        1. 关联不同条件：学习"早上8点"更可能和"咖啡店"关联，"晚上8点"更可能和"电影院"关联
        2. 提取高层特征：从原始条件中提取更有用的抽象表示
        3. 序列建模：考虑整个序列的上下文，而不是孤立地看每个位置
        '''
        encoder_layer = TransformerEncoderLayer(d_model=self.emb_dims, nhead=4, batch_first=True)
        self.condition_transformers = TransformerEncoder(encoder_layer, num_layers=3)


    def forward(self, batch):
        # 截取恰当顺位的标号
        #  - batch.time：时间信息（比如几点钟）
        #  - batch.condition1 到 condition6：6种其他条件（可能是用户ID、星期几、天气等）
        # 编码后的条件向量 [batch, seq_len, 256]，
        # 这些向量会被传给 Transformer，帮助它更准确地预测真实的类别信息。
        seq_length = batch.time.size(1)
        # 取前 seq_length 个位置
        position_ids = self.position_ids[:, : seq_length ]
        
        # 将一大堆时间标和六等不同维度条件字典翻译转换
        time_embeddings = self.encoder(batch.time.long()+1)
        condition1_embeddings = self.encoder(batch.condition1)
        condition2_embeddings = self.encoder(batch.condition2)
        condition3_embeddings = self.encoder(batch.condition3)
        condition4_embeddings = self.encoder(batch.condition4)
        condition5_embeddings = self.encoder(batch.condition5)
        condition6_embeddings = self.encoder(batch.condition6)

        # 拼接 → [batch, seq_len, 7×256] = [batch, seq_len, 1792]
        # 投影压缩 → [batch, seq_len, 256]（降低计算量）
        condition_embeddings = self.input_up_proj(torch.cat([time_embeddings,condition1_embeddings,condition2_embeddings,\
            condition3_embeddings,condition4_embeddings,condition5_embeddings,condition6_embeddings],dim=-1))
            
        # 加上位置编码 → 记录每个事件在序列中的位置
        # 打上排队烙印位号座次，并且送入 3 层的变压大脑思考最终抽象
        condition_embeddings = self.position_embeddings(position_ids) + condition_embeddings
        # Transformer深度处理 → 提取条件之间的高层关联
        # 输出：[batch, seq_len, 256] 编码条件向量
        encoded_conditions = self.condition_transformers(condition_embeddings)
        return encoded_conditions


class DiffusionTransformer(nn.Module):
    """
    重头戏大本营：针对空间分类（干了什么？在哪干的？）研制的特种【离散数据空间扩散核弹】。
    
    外行理解：刚才的 add_thin 负责了时间（几点钟，是连续数值的推演）。
    这里的机制管什么？“看电影、吃饭、唱歌”（这些是断层无序的类别）。
    扩散如何把“看电影”这三个字慢慢扭曲变成毫无线索的乱码？
    且如何用 Transformer 重新在乱码堆里反查它是哪类词组。都靠他。
    """
    def __init__(
        self,
        *,
        diffusion_step=200,   # 破坏多少层到达终极马赛克
        alpha_init_type='alpha1',
        num_condition_types=6,
        type_classes=9,      # 事件的九大系统类别
        poi_classes=3477,    # 具体的空间定位（商铺/地点分类，共三千多）
        num_spectial=4,      # 占道防空号和掩码的四个特指
        num_classes=None,

    ):
        super().__init__()  

        self.schedule_type=alpha_init_type
        self.amp = False
        self.num_condition_types = num_condition_types

        # 把这花里胡哨的各大阵列全部统加在字典大库大小里面（总共就这么多可能了）
        # +2 推理的时候是因为我们需要用特殊占位符表示当前位置是空的还是未知的
        # mask_cat (num_classes - 2) 标记"事件类型"位置未知   另一种 mask_poi (num_classes - 1) 标记"地点"位置未知  
        self.num_classes = type_classes+poi_classes+num_spectial+2 
        self.type_classes = type_classes 
        self.num_spectial = num_spectial 
        self.poi_classes = poi_classes 
        
        # 分配上一步讲解里重金手工构建的终极 Transformer 主解密机
        self.transformer = Transformer(tgt_vocab_size=self.num_classes,num_spectial=self.num_spectial,type_classes=self.type_classes,poi_classes=self.poi_classes)
        self.loss_type = 'vb_stochastic'
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'

        # ===== 以下海量代码全都是在对我们前面的 “破坏图表” 进行系统刻录上盘注册 ======
        # 让它们能够长存为计算参数基底陪着跑。全都会被限制在 -70 起底防下溢。
        at,at1, bt,bt1, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 = alpha_schedule(self.num_timesteps, type_classes=self.type_classes, poi_classes = self.poi_classes)
 
        at1 = torch.tensor(at1.astype('float64'))
        bt1 = torch.tensor(bt1.astype('float64'))
        ct1 = torch.tensor(ct1.astype('float64'))
        log_at1 = torch.log(at1).clamp(-70,0)
        log_bt1 = torch.log(bt1).clamp(-70,0)
        log_ct1 = torch.log(ct1).clamp(-70,0)

        att1 = torch.tensor(att1.astype('float64'))
        btt1 = torch.tensor(btt1.astype('float64'))
        ctt1 = torch.tensor(ctt1.astype('float64'))
        log_cumprod_at1 = torch.log(att1).clamp(-70,0)
        log_cumprod_bt1 = torch.log(btt1).clamp(-70,0)
        log_cumprod_ct1 = torch.log(ctt1).clamp(-70,0) 

        log_1_min_ct1 = log_1_min_a(log_ct1) 
        log_1_min_cumprod_ct1 = log_1_min_a(log_cumprod_ct1)
        assert log_add_exp(log_ct1, log_1_min_ct1).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct1, log_1_min_cumprod_ct1).abs().sum().item() < 1.e-5
        
        # 疯狂挂载到模型主身上备案...
        self.register_buffer('log_ct1', log_ct1.float())
        self.register_buffer('log_bt1', log_bt1.float())
        self.register_buffer('log_at1', log_at1.float())
        self.register_buffer('log_cumprod_at1', log_cumprod_at1.float())
        self.register_buffer('log_cumprod_bt1', log_cumprod_bt1.float())
        self.register_buffer('log_cumprod_ct1', log_cumprod_ct1.float())
        self.register_buffer('log_1_min_ct1', log_1_min_ct1.float())
        self.register_buffer('log_1_min_cumprod_ct1', log_1_min_cumprod_ct1.float())

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at).clamp(-70,0)
        log_bt = torch.log(bt).clamp(-70,0)
        log_ct = torch.log(ct).clamp(-70,0)
        att = torch.tensor(att.astype('float64'))
        btt2 = torch.tensor(btt2.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att).clamp(-70,0)
        log_cumprod_bt = torch.log(btt2).clamp(-70,0)
        log_cumprod_ct = torch.log(ctt).clamp(-70,0)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())
        
        # 其他一些算步数权重留作备用档案的本子
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        # ===== 前方破坏图表登记完毕 ======
        
        self.zero_vector = None

        self.condition_encoder = ConditionEmbeddingModel(num_condition_types=self.num_condition_types)

    def multinomial_kl(self, log_prob1, log_prob2):   
        # 计算两个离散事件分部图之间的散度差（KL Loss）。看看现在差的远不远  是衡量两个概率分布之间差异的指标。
        '''
            KL 散度的数学公式：
                KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
                           = Σ P(x) × (log(P(x)) - log(Q(x)))
            指导模型学习方向，差异越大损失越大
        '''
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t, batch):         
        """
        前向破坏器（只走一步）：q(xt|xt_1)
        已知昨天的污蔑样子 xt-1 ，按大图表查一下今天应该进一步恶化到啥配方地步 xt，并在对数矩阵空间用矩阵转换一次性计算出这种恶化变样。
        """
        B,V,L=log_x_t.shape
        t = t.unsqueeze(1).repeat(1,L)
        log_x_start = rearrange(log_x_t, 'b v l -> b l v')

        # 分头行动：因为这个任务里咱们同时在玩大类(category)和精准坐标(POI)
        # 1. 专门抽出来给大类(category)搞一步破坏
        log_x_start_category = log_x_start[batch.category_mask.bool()]
        t_tmp = t[batch.category_mask.bool()]
        selected_range = log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes]

        log_ct1 = extract(self.log_ct1, t_tmp, selected_range.shape)         
        log_1_min_ct1 = extract(self.log_1_min_ct1, t_tmp, selected_range.shape)      

        # 把这原先残存的真情报叠合上衰减破坏魔法和干扰乱投杂讯
        selected_range = selected_range + log_1_min_ct1
        log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes] = selected_range
        log_x_start_category = torch.cat([log_x_start_category[:,:-2],log_add_exp(log_x_start_category[:,-2:-1],log_ct1), log_x_start_category[:,-1:]],dim=-1)
        log_x_start[batch.category_mask.bool()] = log_x_start_category

        # 2. 也是这套道理再在三千多人的地点池(POI)里面用另一套高深的恶化表走一步
        log_x_start_poi = log_x_start[batch.poi_mask.bool()]
        selected_range = log_x_start_poi[:,self.num_spectial+self.type_classes:-2]
        t_tmp = t[batch.poi_mask.bool()]
        log_at = extract(self.log_at, t_tmp, selected_range.shape) 
        log_bt = extract(self.log_bt, t_tmp, selected_range.shape)             
        log_ct = extract(self.log_ct, t_tmp, selected_range.shape)             
        log_1_min_ct = extract(self.log_1_min_ct, t_tmp, selected_range.shape)        

        # 不单单是吃老本变乱码，里面甚至把“我突然跳到了别的正常坐标冒充”的情况(log_bt)加进去
        selected_range = log_add_exp(selected_range +log_at, log_bt)
        log_x_start_poi[:,self.num_spectial+self.type_classes:-2] = selected_range
        log_x_start_poi = torch.cat([log_x_start_poi[:,:-1],log_add_exp(log_x_start_poi[:,-1:]+log_1_min_ct, log_ct)],dim=-1)
        log_x_start[batch.poi_mask.bool()] = log_x_start_poi

        log_probs = rearrange(log_x_start, 'b l v -> b v l')

        return log_probs

    def q_pred(self, log_x_start, t, batch):           
        """
        前向破坏器（一步到位法）： q(xt|x0)
        因为离散马尔可夫链极度伟大的一点就是：
        我不用老老实实从1步跑到50步来制造考卷。我可以直接把累乘破败图表(cumprod 参数)拿出来，
        拿无暇的事件瞬间就能在矩阵里造出 50步惨遭破坏的模样发给考生！这极其加速！
        """
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        B,V,L=log_x_start.shape
        t = t.unsqueeze(1).repeat(1,L)

        log_x_start = rearrange(log_x_start, 'b v l -> b l v')

        # 对大类进行跨越变质
        log_x_start_category = log_x_start[batch.category_mask.bool()]
        selected_range = log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes]
        t_tmp = t[batch.category_mask.bool()]

        log_cumprod_ct1 = extract(self.log_cumprod_ct1, t_tmp, selected_range.shape)         
        log_1_min_cumprod_ct1 = extract(self.log_1_min_cumprod_ct1, t_tmp, selected_range.shape)       

        selected_range = selected_range + log_1_min_cumprod_ct1
        log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes] = selected_range
        log_x_start_category = torch.cat([log_x_start_category[:,:-2],log_add_exp(log_x_start_category[:,-2:-1],log_cumprod_ct1), log_x_start_category[:,-1:]],dim=-1)
        log_x_start[batch.category_mask.bool()] = log_x_start_category

        # 对地理位置坐标点跨域变质
        log_x_start_poi = log_x_start[batch.poi_mask.bool()]
        selected_range = log_x_start_poi[:,self.num_spectial+self.type_classes:-2]
        t_tmp = t[batch.poi_mask.bool()]

        log_cumprod_at = extract(self.log_cumprod_at, t_tmp, selected_range.shape)
        log_cumprod_bt = extract(self.log_cumprod_bt, t_tmp, selected_range.shape)         
        log_cumprod_ct = extract(self.log_cumprod_ct, t_tmp, selected_range.shape)         
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t_tmp, selected_range.shape)       

        selected_range = log_add_exp(selected_range +log_cumprod_at, log_cumprod_bt)
        log_x_start_poi[:,self.num_spectial+self.type_classes:-2] = selected_range
        log_x_start_poi = torch.cat([log_x_start_poi[:,:-1],log_add_exp(log_x_start_poi[:,-1:]+log_1_min_cumprod_ct, log_cumprod_ct)],dim=-1)
        log_x_start[batch.poi_mask.bool()] = log_x_start_poi

        log_probs = rearrange(log_x_start, 'b l v -> b v l')
            
        return log_probs

    def predict_start(self, log_x_t, cond_emb, t, batch):          
        """
        这就是模型本体上场大展拳脚的环节。 p(x0|xt) 
        你扔给模型一顿完全被打到烂的碎片(log_x_t)，带上它此刻受难的破坏步数(t)和周边情况指引(cond)，
        让装在我们本体里那个无敌强大的 Transformer 给脑补推理：“这TM原本完好无缺的纯情记录应该是啥样的大概率事件？” 
        """
        x_t = log_onehot_to_index(log_x_t)
        # 用高能半精度算（或者全精度算）交给老大哥Transformer破案
        if self.amp == True:
            with autocast():
                out = self.transformer(x_t, cond_emb, t, batch)
        else:
            out = self.transformer(x_t, cond_emb, t, batch)
            
        # 确保破出来的词典词数不多不少恰好合适
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-2
        assert out.size()[2:] == x_t.size()[1:]
        
        # 因为输出来的是分数，走一遍 log_softmax 流程按死在 0-1 百分比。
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]

        # 把刚才强行抽走的两个遮掩杂向位置补两把无所谓的占位冷板凳填回去凑全。
        zero_vector = torch.zeros(batch_size, 2, log_x_t.size(2)).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred
    
    def predict_start_with_truncate(self, log_x_t, cond_emb, t, batch, truncation_k=15):  
        """
        这个就是刚才那招的【高冷净化尊享版】。
        虽然你预测了很多词的可能还原发生概率。但为了避免它胡说八道瞎联想，我只允许取“最可能、得分最高的 15 个绝杀选项”(Top-K)！其他没上榜单的全部一脚踹飞，以 -70（极其微小）镇压！这也大大加速了推理和稳定。
        """
        x_t = log_onehot_to_index(log_x_t)
        if self.amp == True:
            with autocast():
                out = self.transformer(x_t, cond_emb, t, batch)
        else:
            out = self.transformer(x_t, cond_emb, t, batch)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-2
        assert out.size()[2:] == x_t.size()[1:]
        
        log_pred = F.log_softmax(out.double(), dim=1).float()

        # 镇压神技：取出排位前 15 大哥位置
        val, ind = log_pred.topk(k=truncation_k, dim=1)
        # 满地都是 -70
        probs = torch.full_like(log_pred, -70)
        # 但是在那前15个兄弟的位置铺上光明的大数（Scatter替换）
        log_pred = probs.scatter_(1, ind, val)

        batch_size = log_x_t.size()[0]
        zero_vector = torch.zeros(batch_size, 2, log_x_t.size(2)).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

 
    
    def q_posterior(self, log_x_start, log_x_t, t, batch):            
        """有了还原好最终版本和此刻情况，严谨算出它怎么退回到上一步 xt-1"""
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(log_x_start, t - 1, batch)
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_emb, t, batch):             
        """主降噪单步推理流程"""
        if self.parametrization == 'x0':
            # 先去猜终极大谜底可能是啥（保留前十五）
            log_x_recon = self.predict_start_with_truncate(log_x, cond_emb, t, batch)
            # 再由果索因，推退回前一步老老实实当垫脚
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t, batch=batch)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_emb, t, batch)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(self, log_x, cond_emb,  t, batch):               
        """推理阶段神技：凭空取一沙，步步变金丹的核心倒转环。"""
        # 利用刚上面那个解出它的真实原力场（上一步可能的百分比组队）
        model_log_prob, log_x_recon = self.p_pred(log_x, cond_emb, t, batch)
        # 用下面那个古老的甘贝尔摇骰子魔法，在这步强行钦定抽出来一个实在结果。
        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits):           
        """
        利用古老妖法：Gumbel-Max 采样抽个盲盒。
        因为模型脑补的对数概率是模棱两可连续的（比如 50%吃饭 45%看电影）。
        我不能永远这么纠结发下去影响下一步啊！我必须马上干脆定下到底是饭还是电影，
        把局定了！这套结合了噪音加噪随机的算法能优美的按照给出的分布
        把它摇成板上钉钉的确切离散事标，还能拥有随机防错的波动艺术。
        """
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        # 返回确认好的下一步落子（打码形式）
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t, batch):                 
        """顺藤摸瓜污染造题目的。先知道它在公式里有多坏了，再强行掷骰定下这波被毁到的确切实事标。"""
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, batch)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        """如果考场上老师发卷子，他怎么抽这次考你毁容毁到了第几步难度的大题？"""
        if method == 'importance':
            # 聪明发卷子法：哪里之前经常丢分出大错算不好，这次出它这个难度的几率就很高，加强死磕它。
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            # 懒汉发卷法：直接投个随机数抽难度呗
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    @property
    def device(self):
        return self.transformer.to_logits[-1].weight.device ##todo

    def training_losses(
        self,
        batch,
        is_train=True
        ):
        """【教练指挥官】整个模型进厂被殴打调教、被狂揍吃亏认错修正的总入口"""
        b, device = batch.batch_size, batch.device
        assert self.loss_type == 'vb_stochastic'
        x_start = batch.checkin_sequences
        
        # 老师出题：抽一个步骤 t，这事就破坏到这地步测你
        t, pt = self.sample_time(b, device, 'importance')
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        
        # 直接拿图纸造出被严重污染后乱飞狗跳的废案样卷 (x_t)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t, batch=batch) 

        # 翻译过去曾经的事怎么样的所有前因后果条件
        cond_emb = self.condition_encoder(batch)

        # 把这坨废件(xt)带着条件发给自己家牛逼学生猜：“你给我猜原本好好的样卷应该是啥样(recon)？”
        log_x0_recon = self.predict_start(log_xt, cond_emb, t=t, batch=batch) 
        log_x0_recon = log_x0_recon.transpose(1, 2)

        losses = {}
        # 发火打分环节：拿学生猜的成绩(recon)和起初的完美记录(x_start)强行比对！算交叉错位惩罚度（分差）算出来的就是你该长记性要往回扒的分。
        loss = F.cross_entropy(log_x0_recon.reshape(-1, log_x0_recon.shape[-1]), x_start.flatten(),ignore_index=3, reduce=False)
        loss = loss.reshape(x_start.size(0), -1)
        losses['loss'] = torch.mean(loss, -1)
        return losses['loss']

    def sample_fast(
            self,
            batch,
            content_token = None,
            **kwargs):
        """【一键奇迹发生室。成品开箱无中生有】学成以后放出来的终极大招推演模块"""
        B, L = batch.batch_size, batch.content_len

        device = self.log_at.device

        batch.device = device

        cond_emb = self.condition_encoder(batch)

        mask_poi=self.num_classes-1 # POI掩码
        mask_cat=self.num_classes-2 # 类别掩码
        
        bottom=torch.tensor([2],device=device)
        input=torch.ones(B,L,dtype=torch.int64,device=device) *3 ##padding

        # 神仙下界之前：用 0和3 这种极端杂数和掩码拼出全部都是空白没有实事的骨架皮囊垫上。
        for i in range(B):
            seq_len = batch.unpadded_length[i]
            # 类别部分用 mask_cat 占位
            head=torch.tensor([0] + [mask_cat] * seq_len ,device=device)
            # POI 部分用 mask_poi 占位
            body=torch.tensor([1] + [mask_poi] * seq_len ,device=device)

            tmp=torch.cat([head,body,bottom],dim=-1)
            input[i][:len(tmp)]=tmp

        log_z = index_to_log_onehot(input,self.num_classes)
        # 开始漫长且伟大的倒步旅行：从第 200 号最废物的噪音状态开始，让 Transformer 全速退档降噪。
        start_step = self.num_timesteps
        with torch.no_grad():
            for diffusion_index in range(start_step - 1, -1, -1):
                t = torch.full((B,), diffusion_index, device=device, dtype=torch.long)
                # 每跑一步就是给皮囊渡一次劫和去假抽真术
                log_z = self.p_sample(log_z, cond_emb, t, batch)  

        # 最末尾返回人间的不再是概率数字，而是明明白白的一举一行和坐标号串。
        content_token = log_onehot_to_index(log_z)

        return Batch(
            time=batch.time,
            condition1=batch.condition1, 
            condition2=batch.condition2,
            condition3=batch.condition3,
            condition4=batch.condition4,
            condition5=batch.condition5,
            condition6=batch.condition6,
            condition1_indicator=batch.condition1_indicator,
            condition2_indicator=batch.condition2_indicator,
            condition3_indicator=batch.condition3_indicator,
            condition4_indicator=batch.condition4_indicator,
            condition5_indicator=batch.condition5_indicator,
            condition6_indicator=batch.condition6_indicator,
            mask=batch.mask, 
            tmax=batch.tmax,
            checkin_sequences=content_token,
            category_mask=batch.category_mask,
            poi_mask=batch.poi_mask,
            tau=batch.tau,
            unpadded_length=batch.unpadded_length
        )

"""
========================================================================
【给外行新手的通俗讲解】 -- discrete_diffusion/diffusion_transformer.py
========================================================================

1. 这个大篇幅离散扩散到底有什么用？
   - 之前那个 `add_thin` 大佬告诉你的是一个人：明天 8:36 、下午 14:02 、晚上 21:00 这三个精确**时间发生做事**。
   - 但是这三个时间点他到底 **干嘛去了(分类)? 哪去干的(POI地点)?** 这就需要交给这个文件里的二神仙来做定夺补充完整。他们俩加在一起实现了从“无声的几点钟”升级到“生龙活虎的：他在 8:36去了朝阳大悦城吃了小笼包大类”。这便是真正的双轨点过程预测。 

2. 什么叫离散空间？它为啥特殊？怎么推断？
   - 因为人类的位置去向和干嘛是“断层”的，比如不是 A就是B，没有半个网吧这种数字说法。
   - 你既然无法像数学图像平滑加杂色。你只能：我从一个盒子里拿出写着这人去了“网吧”的纸条，我强行换成了上面乱码的纸条放回他人生轨道里（前向干扰q_sample）。
   - 然后要求我们的机器上帝（Transformer）拿着这破破烂烂到处是乱码插曲的生平时间线。通过它前后观察他曾做过啥去过哪。最终用 `p_sample` 利用 `Top-k 断舍离法` 强行把真纸条从垃圾桶抽出来完美恢复他的真实去向轨迹。非常生猛又伟大！
========================================================================
"""
