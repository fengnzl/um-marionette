#!/bin/bash
# Marionette 项目环境一键安装脚本
# 适用于 Google Colab 和 Ubuntu/Debian 系统

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Marionette 环境配置脚本"
echo "=========================================="
echo ""

# 检查是否在 Colab 中
if [ -f "/tools/node_modules/.bin/acpi" ]; then
    echo "检测到 Google Colab 环境"
    IN_COLAB=true
else
    echo "检测到本地环境"
    IN_COLAB=false
fi

echo "步骤 1/7: 安装 PyTorch 和相关依赖..."
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117 -q
pip install pytorch-lightning==1.9.5 -q
echo "✓ PyTorch 安装完成"
echo ""

echo "步骤 2/7: 安装数据处理库..."
pip install numpy==1.26.4 pandas==2.2.2 -q
echo "✓ 数据处理库安装完成"
echo ""

echo "步骤 3/7: 安装配置和日志工具..."
pip install hydra-core wandb einops rich -q
echo "✓ 配置工具安装完成"
echo ""

echo "步骤 4/7: 安装其他依赖..."
pip install torchtyping typeguard pyyaml scipy -q
echo "✓ 其他依赖安装完成"
echo ""

echo "步骤 5/7: 安装可视化库..."
pip install matplotlib seaborn -q
echo "✓ 可视化库安装完成"
echo ""

echo "步骤 6/7: 验证 GPU 可用性..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"✓ GPU 可用: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠ GPU 不可用，将使用 CPU（训练会很慢）")
EOF
echo ""

echo "步骤 7/7: 验证安装..."
python3 << 'EOF'
import sys
packages = [
    'torch', 'pytorch_lightning', 'numpy', 'pandas',
    'hydra', 'wandb', 'einops', 'rich'
]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg} 安装失败")
        sys.exit(1)
print("所有依赖验证通过！")
EOF
echo ""

echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "下一步："
if [ "$IN_COLAB" = true ]; then
    echo "1. 将 Marionette 项目文件上传到 Google Drive"
    echo "2. 在 Colab 中挂载 Drive: from google.colab import drive"
    echo "3. 开始学习 Day 1 教程"
else
    echo "1. 确保项目文件在当前目录"
    echo "2. 运行: jupyter notebook"
    echo "3. 打开 Day 1 教程"
fi
echo ""
