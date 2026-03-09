# Marionette 7 天深度教学 - 完整材料包

## 📦 材料清单

### 1. 交互式 Jupyter Notebooks (Day 1-7)

**位置**: `marionette-tutorial/notebooks/`

| 文件 | 内容 | 预计时间 |
|------|------|----------|
| `Day_01_环境搭建与项目概览.ipynb` | 环境配置、项目结构、配置系统 | 4 小时 |
| `Day_02_Python数据科学生态.ipynb` | NumPy, Pandas, PyTorch 基础 | 4 小时 |
| `Day_03_数据处理与批次管理.ipynb` | Sequence, Batch 类，数据预处理 | 4.5 小时 |
| `Day_04_扩散模型核心原理.ipynb` | Add-THIN, 离散扩散，条件生成 | 4.5 小时 |
| `Day_05_训练与采样.ipynb` | train.py, sample.py, W&B | 4.5 小时 |
| `Day_06_评估与可视化.ipynb` | 评估指标，Deck.gl 可视化 | 5 小时 |
| `Day_07_FastAPI后端与整合.ipynb` | FastAPI, 前后端整合 | 5 小时 |

**总计**: 约 30-35 小时（每天 4-5 小时）

### 2. 技术文档

**位置**: `marionette-tutorial/docs/`

| 文件 | 内容 |
|------|------|
| `Marionette项目技术文档.md` | 完整的技术文档，包含架构、API、部署指南 |
| `快速参考指南.md` | 核心类和方法速查表 |

### 3. 后端代码

**位置**: `marionette-tutorial/backend/`

| 文件 | 内容 |
|------|------|
| `api.py` | FastAPI 后端服务，包含完整的 API 实现 |

### 4. 前端代码

**位置**: `marionette-tutorial/frontend/`

| 文件 | 内容 |
|------|------|
| `components/App.tsx` | React + Deck.gl 双屏可视化组件 |

---

## 🚀 快速开始

### 方式一：Google Colab（推荐）

1. 将每个 `.ipynb` 文件上传到 Google Drive
2. 在 Google Colab 中打开
3. 启用 GPU 运行时
4. 从上到下依次运行代码块

### 方式二：本地 Jupyter

1. 安装依赖：
```bash
pip install jupyter numpy pandas torch matplotlib scipy
```

2. 启动 Jupyter：
```bash
jupyter notebook
```

3. 打开 `notebooks/` 目录下的 Notebook

---

## 📚 学习路径

### 第 1 天：环境搭建与项目概览
- ✅ 设置 Google Colab GPU 环境
- ✅ 安装项目依赖
- ✅ 理解项目目录结构
- ✅ 学习 Hydra 配置系统

### 第 2 天：Python 数据科学生态
- ✅ NumPy 数组操作
- ✅ Pandas 数据分析
- ✅ PyTorch Tensor 和自动微分
- ✅ 神经网络基础

### 第 3 天：数据处理与批次管理
- ✅ 理解 Istanbul 数据集格式
- ✅ 掌握 Sequence 类的实现
- ✅ 理解 Batch 类的 padding 逻辑
- ✅ 学习多数据集适配方法

### 第 4 天：扩散模型核心原理
- ✅ 扩散模型的直观理解
- ✅ Add-THIN 时间扩散模型
- ✅ 离散扩散空间模型
- ✅ 条件生成机制

### 第 5 天：训练与采样
- ✅ train.py 训练流程
- ✅ W&B 实验跟踪
- ✅ sample.py 采样流程
- ✅ 在 Colab 中运行训练

### 第 6 天：评估与可视化
- ✅ 理解 5 个评估任务
- ✅ 掌握评估指标计算
- ✅ 数据导出为 JSON
- ✅ Deck.gl 双屏可视化

### 第 7 天：FastAPI 后端与整合
- ✅ FastAPI 基础和 API 设计
- ✅ 模型推理接口
- ✅ 前后端整合
- ✅ 完整 Demo 部署

---

## 📊 最终产出

完成 7 天学习后，你将拥有：

1. ✅ **可运行的前后端 Demo**
   - FastAPI 后端 (`backend/api.py`)
   - React + Deck.gl 前端 (`frontend/components/App.tsx`)

2. ✅ **中文技术文档**
   - `docs/Marionette项目技术文档.md`
   - `docs/快速参考指南.md`

3. ✅ **交互式学习材料**
   - 7 个完整的 Jupyter Notebooks
   - 包含理论讲解、代码示例、练习题

---

## 🎯 学习目标检查清单

完成课程后，你应该能够：

- [ ] 从零搭建 Marionette 项目
- [ ] 解释每个核心类和方法的作用
- [ ] 理解扩散模型的工作原理
- [ ] 训练和采样轨迹数据
- [ ] 运行完整的评估流程
- [ ] 实现轨迹数据可视化
- [ ] 搭建 FastAPI 后端服务
- [ ] 完成前后端整合

---

## 🔧 后续工作

### 代码注释版

需要为以下关键文件添加详细中文注释：

1. `datamodule.py` - Sequence 和 Batch 类
2. `add_thin/diffusion/model.py` - AddThin 类
3. `discrete_diffusion/diffusion_transformer.py` - DiffusionTransformer 类
4. `tasks.py` - DensityEstimation 类

### 实际运行 Demo

1. **后端启动**:
```bash
cd marionette-tutorial/backend
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
```

2. **前端启动**:
```bash
cd marionette-tutorial/frontend
npm install
npm run dev
```

3. **访问**: http://localhost:5173

---

## 📝 研究计划书建议

基于你的前端背景，推荐的研究方向：

### 可视化驱动的生成优化

1. **交互式参数调优**
   - 通过可视化界面实时调整生成参数
   - 观察参数对生成结果的影响
   - 自动搜索最优参数组合

2. **生成结果对比分析**
   - 双屏同步展示真实 vs 生成数据
   - 实时计算统计差异
   - 可视化评估指标

3. **用户反馈循环**
   - 收集用户对生成质量的反馈
   - 将反馈用于模型优化
   - 主动学习式的生成改进

---

## 🙏 致谢

本项目基于以下工作：

- Marionette (KDD'25) 论文
- UM Data Intelligence Lab

---

## 📧 联系方式

如有问题，请参考：
- 项目技术文档: `docs/Marionette项目技术文档.md`
- 快速参考指南: `docs/快速参考指南.md`

---

## 🎉 祝你学习顺利！

完成这 7 天的学习后，你将：
- ✅ 深入理解扩散模型在轨迹生成中的应用
- ✅ 掌握 PyTorch 和 PyTorch Lightning
- ✅ 具备搭建深度学习项目的能力
- ✅ 能够进行相关研究工作

**下一步**: 基于这个基础，撰写研究计划书，申请进入 UM Data Intelligence Lab！

---

*最后更新: 2024-03-06*
*版本: 1.0*
