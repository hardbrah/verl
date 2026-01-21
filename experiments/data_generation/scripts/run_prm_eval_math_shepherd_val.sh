#!/bin/bash
# ============================================================
# PRM评估脚本 - Math Shepherd验证集
# ============================================================
#
# 使用训练时的验证集评估PRM效果
# 数据集字段:
#   - response: 问题 + Step 1 的思维链
#   - step1_label: +/- 标签
#
# 使用方法:
#   bash run_prm_eval_math_shepherd_val.sh all
#   bash run_prm_eval_math_shepherd_val.sh extract
#   bash run_prm_eval_math_shepherd_val.sh score
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/eval_prm_math_shepherd_val.py"

# 配置参数
NPROC=${NPROC:-8}
BATCH_SIZE=${BATCH_SIZE:-64}

# 数据和模型路径
DATA_PATH="/data/chenhaotian/latentqa/data/math_shepherd_step1_fullkeys/val_balanced.jsonl"
MODEL_PATH="/data/models/Qwen/Qwen3-4B-Instruct-2507"
PRM_CHECKPOINT="/data/chenhaotian/latentqa/output/continue_train_dapo_rollout/best_model.pt"
OUTPUT_DIR="/data/chenhaotian/verl/experiments/data_generation/outputs/stage2_prm_eval_math_shepherd_val"

# PRM配置
THRESHOLD=${THRESHOLD:-0.5}

echo "============================================================"
echo "PRM评估脚本 - Math Shepherd验证集"
echo "============================================================"
echo "GPU数量: ${NPROC}"
echo "Batch size: ${BATCH_SIZE}"
echo "PRM阈值: ${THRESHOLD}"
echo "数据集: ${DATA_PATH}"
echo "模型: ${MODEL_PATH}"
echo "PRM checkpoint: ${PRM_CHECKPOINT}"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

MODE=${1:-"all"}

run_extract() {
    echo ">>> 阶段1: 提取Latent States (8卡)"
    echo ""
    
    torchrun --nproc_per_node=${NPROC} ${SCRIPT_PATH} \
        --stage extract \
        --batch_size ${BATCH_SIZE} \
        --data_path "${DATA_PATH}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}"
    
    echo ""
    echo ">>> 阶段1完成"
    echo ""
}

run_score() {
    echo ">>> 阶段2: 分布式打分和评估 (8卡)"
    echo ""
    
    torchrun --nproc_per_node=${NPROC} ${SCRIPT_PATH} \
        --stage score \
        --batch_size ${BATCH_SIZE} \
        --model_path "${MODEL_PATH}" \
        --prm_checkpoint "${PRM_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        --threshold ${THRESHOLD}
    
    echo ""
    echo ">>> 阶段2完成"
    echo ""
}

case ${MODE} in
    "extract")
        run_extract
        ;;
    "score")
        run_score
        ;;
    "all")
        run_extract
        run_score
        ;;
    *)
        echo "用法: bash run_prm_eval_math_shepherd_val.sh [extract|score|all]"
        exit 1
        ;;
esac

echo "============================================================"
echo "评估完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo "============================================================"
