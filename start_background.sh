#!/bin/bash

# PersonaPlex 后台启动脚本

cd "$(dirname "$0")"

# 检查 HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  警告: HF_TOKEN 未设置"
    echo "   请设置: export HF_TOKEN=<YOUR_TOKEN>"
    exit 1
fi

# 设置端口（默认 5001）
PORT=${1:-5001}

echo "=========================================="
echo "启动 PersonaPlex 服务器（后台运行）"
echo "=========================================="
echo "端口: $PORT"
echo "日志文件: server.log"
echo ""

# 后台运行
nohup python3 server_cloud_stateful.py $PORT > server.log 2>&1 &

# 获取进程 ID
PID=$!

echo "✓ 服务器已启动"
echo "  进程 ID: $PID"
echo "  查看日志: tail -f server.log"
echo "  停止服务: kill $PID"
echo ""
echo "服务器正在启动中，请稍候..."
echo ""

# 等待几秒，检查是否启动成功
sleep 3

if ps -p $PID > /dev/null; then
    echo "✓ 服务器运行中 (PID: $PID)"
else
    echo "✗ 服务器启动失败，请查看 server.log"
    exit 1
fi

