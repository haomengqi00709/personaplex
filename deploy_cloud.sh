#!/bin/bash

# PersonaPlex 云端 GPU 部署脚本
# 在云端 GPU 服务器上运行此脚本

set -e

echo "=========================================="
echo "PersonaPlex 云端 GPU 部署"
echo "=========================================="
echo ""

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到 NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  未检测到 NVIDIA GPU，将使用 CPU（较慢）"
    echo ""
fi

# 检查 Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ 错误: 未找到 Python"
    exit 1
fi

echo "使用 Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# 检查 HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  警告: HF_TOKEN 未设置"
    echo "   请设置: export HF_TOKEN=<YOUR_TOKEN>"
    echo ""
fi

# 安装 PyTorch (CUDA 版本)
echo "检查 PyTorch..."
if $PYTHON_CMD -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "✓ PyTorch 已安装"
else
    echo "安装 PyTorch (CUDA 11.8)..."
    $PYTHON_CMD -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi
echo ""

# 安装依赖
echo "安装 Python 依赖..."
$PYTHON_CMD -m pip install -r requirements_cloud.txt
echo ""

# 安装 PersonaPlex moshi 包
if [ -d "personaplex/moshi" ]; then
    echo "安装 PersonaPlex moshi 包..."
    $PYTHON_CMD -m pip install -e personaplex/moshi/
    echo ""
else
    echo "⚠️  警告: 未找到 personaplex/moshi 目录"
    echo "   请确保已克隆 PersonaPlex 仓库"
    echo ""
fi

# 检查必要文件
if [ ! -f "server_cloud.py" ]; then
    echo "❌ 错误: 未找到 server_cloud.py"
    exit 1
fi

if [ ! -f "index.html" ]; then
    echo "❌ 错误: 未找到 index.html"
    exit 1
fi

echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "启动服务器:"
echo "  python3 server_cloud.py 5001"
echo ""
echo "或者后台运行:"
echo "  nohup python3 server_cloud.py 5001 > server.log 2>&1 &"
echo ""

