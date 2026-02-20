#!/bin/bash
# EventVAD V5 可视化系统 — 停止脚本

echo "正在停止 EventVAD 可视化服务..."
pkill -f "uvicorn app:app.*8501" 2>/dev/null && echo "  ✓ 后端已停止" || echo "  - 后端未运行"
pkill -f "http.server 8502" 2>/dev/null && echo "  ✓ 前端已停止" || echo "  - 前端未运行"
echo "完成。"
