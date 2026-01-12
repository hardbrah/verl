#!/bin/bash
# Rollout 生成启动脚本
# 配置请修改 configs/config.py

set -e
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(dirname "$SCRIPT_DIR")/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/rollout_$(date +%Y%m%d_%H%M%S).log"


cd "$SCRIPT_DIR"
python -u data_gen_vllm.py "$@" 2>&1 | tee -a "$LOG_FILE"
