#!/bin/bash
# EventVAD V5 可视化系统 — 一键启动脚本
#
# 用法:
#   cd EventVAD/visualization
#   bash start.sh
#
# 服务:
#   - 后端 API + 前端: http://localhost:8501
#   - 前端独立访问:    http://localhost:8502
#   - API 文档:       http://localhost:8501/docs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║     EventVAD V5 可视化系统               ║"
echo "  ║     Video Anomaly Detection Visualizer   ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# ── 关闭已有实例 ──
echo "[0/3] 清理已有实例..."
pkill -f "uvicorn app:app.*8501" 2>/dev/null || true
pkill -f "http.server 8502" 2>/dev/null || true
sleep 1

# ── 检查 Python 依赖 ──
echo "[1/3] 检查 Python 依赖..."
pip install fastapi uvicorn python-multipart opencv-python-headless httpx -q 2>/dev/null || {
    echo "  ⚠ 部分依赖安装失败，请手动运行: pip install -r backend/requirements.txt"
}

# ── 启动后端 ──
echo "[2/3] 启动后端 API (port 8501)..."
cd backend
nohup python -m uvicorn app:app --host 0.0.0.0 --port 8501 --log-level info > /tmp/eventvad_backend.log 2>&1 &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

sleep 2

# 检查后端是否启动成功
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo "  ✓ 后端已启动 (PID: $BACKEND_PID)"
else
    echo "  ✗ 后端启动失败，查看日志: cat /tmp/eventvad_backend.log"
    exit 1
fi

# ── 启动前端 ──
echo "[3/3] 启动前端 (port 8502)..."
cd frontend
nohup python -m http.server 8502 --bind 0.0.0.0 > /tmp/eventvad_frontend.log 2>&1 &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

sleep 1

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║  系统已启动！                             ║"
echo "  ║                                          ║"
echo "  ║  前端 UI:   http://localhost:8502         ║"
echo "  ║  API 文档:  http://localhost:8501/docs    ║"
echo "  ║                                          ║"
echo "  ║  后端 PID: $BACKEND_PID                        ║"
echo "  ║  前端 PID: $FRONTEND_PID                        ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  停止服务: bash stop.sh"
echo "  查看日志: tail -f /tmp/eventvad_backend.log"
echo ""
