# Session: Marionette Project

## Learner Profile
- **Level**: 中级加速（已知基础语法、高阶函数、继承）
- **Language**: 中文
- **Started**: 2026-03-18
- **Mode**: 加速模式（20+ 小时/周）

## Strengths (Diagnosed)
- ✅ Python 基础语法
- ✅ 高阶函数（函数作为参数）
- ✅ 继承和 `super()`
- ✅ 基础类型注解 (`-> int`)

## Gaps (Identified)
- ⚠️ 装饰器原理和应用
- ⚠️ 高级类型注解（泛型、Union、Optional）
- ⚠️ 魔术方法（`__getattr__`, `__iter__` 等）
- ⚠️ PyTorch 和深度学习
- ⚠️ 项目特定概念（扩散模型、TPP等）

---

## Concept Map

| # | Concept | Prerequisites | Status | Score | Last Reviewed | Review Interval |
|---|---------|---------------|--------|-------|---------------|-----------------|
| 1 | **装饰器原理** | - | mastered | 100% | 2026-03-19 | 2d |
| 2 | **高级类型注解** | - | mastered | 100% | 2026-03-19 | 2d |
| 3 | **魔术方法** | 2 | mastered | 100% | 2026-03-19 | 2d |
| 4 | **PyTorch Tensor** | - | mastered | 100% | 2026-03-19 | 2d |
| 5 | **nn.Module 架构** | 4 | mastered | 85% | 2026-03-19 | 1d |
| 6 | **数据加载流程** | 3, 5 | mastered | 95% | 2026-03-19 | 1d |
| 7 | **项目配置系统** | - | mastered | 95% | 2026-03-19 | 1d |
| 8 | **扩散模型基础** | 4, 5 | mastered | 90% | 2026-03-19 | 1d |
| 9 | **Transformer 架构** | 8 | mastered | 85% | 2026-03-22 | 1d |
| 10 | **Marionette 整体架构** | 6, 7, 8, 9 | mastered | 90% | 2026-03-22 | 1d |

---

## Session Log
- [2026-03-18] 诊断完成：学习者具备 Python 基础、高阶函数、继承知识
- [2026-03-18] 概念 1-4：装饰器原理、高级类型注解、魔术方法、PyTorch Tensor — 全部掌握
- [2026-03-19] 概念 5：nn.Module 架构 — 掌握 (85%)
  - 深入分析了 PointClassifier 实现
  - 理解了 embedding 概念和维度计算
  - 掌握了 __init__ vs forward 的职责
  - 理解了 model(x) vs model.forward(x) 的区别
  - 学习了 nn.Module 的 hooks 机制
- [2026-03-19] 概念 6：数据加载流程 — 掌握 (95%)
  - 理解了 PyTorch 数据加载的 4 个核心组件
  - 掌握了 Dataset 和 DataLoader 的职责分工
  - 理解了 collate_fn 的作用和实现
  - 深入分析了 Batch 类的设计和魔术方法应用
  - 理解了变长序列的 padding 和 mask 机制
- [2026-03-19] 概念 8：扩散模型基础 — 掌握 (90%)
  - 理解了扩散模型的基本原理（前向/反向过程）
  - 深入学习了 noise() 方法（添加噪声：保留+添加）
  - 深入学习了 sample_posterior() 方法（去除噪声：分类+重组）
  - 理解了训练流程（两个损失函数：分类+强度）
  - 理解了采样流程（从纯噪声生成新数据）
  - 通过具体例子和问答形式完全理解了 Add-Thin 的实现机制
- [2026-03-22] 概念 9：Transformer 架构 — 掌握 (85%)
  - 理解了 RNN 的局限性和 Transformer 的动机
  - 掌握了 Self-Attention 机制（Q·K·V，加权求和）
  - 理解了 Multi-Head Attention 的作用（多种关系模式）
  - 学习了 Positional Encoding（sin/cos 相对位置编码）
  - 理解了 Encoder-Decoder 架构和 Cross-Attention
  - 理解了 Masked Self-Attention 的作用（防止训练时偷看）
  - 理解了 Feed Forward 层的作用（信息处理）
- [2026-03-22] 概念 10：Marionette 整体架构 — 掌握 (90%)
  - 理解了时空分离架构的设计思想
  - 掌握了 DiffusionTransformer 的组成（扩散模型 + Transformer）
  - 理解了双损失函数训练机制（时间+空间）
  - 理解了采样流程的时间-空间协作顺序
  - 理解了类别特定扩散策略的作用
  - 理解了条件嵌入的作用
  - **完成了全部 10 个概念的学习！** 🎉
