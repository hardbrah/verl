#!/bin/bash
# ============================================================
# PRM评估脚本 - GSM8K数据集版本
# ============================================================
#
# 功能说明:
#   1. 阶段1 (extract): 使用8卡并行提取第15层latent states
#      - 使用gsm8k.py验证工具分类正负样本
#      - 选择数量相同的正负样本进行平衡
#   2. 阶段2 (score): 使用8卡并行对latent states进行PRM打分
#
# 使用方法:
#   # 只运行阶段2（如果latent states已缓存）:
#   bash run_prm_eval_gsm8k.sh score
#
#   # 运行两个阶段:
#   bash run_prm_eval_gsm8k.sh all
#
#   # 只运行阶段1:
#   bash run_prm_eval_gsm8k.sh extract
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/eval_prm_gsm8k.py"

# 配置参数
NPROC=${NPROC:-8}
BATCH_SIZE=${BATCH_SIZE:-64}

# 数据和模型路径
DATA_PATH="/data/chenhaotian/verl/experiments/data_generation/outputs/gsm8k/rollouts/rollout_20260117_110227/all_responses_rollout_20260117_110227.jsonl"
MODEL_PATH="/data/models/Qwen/Qwen3-4B-Instruct-2507"
PRM_CHECKPOINT="/data/chenhaotian/latentqa/output/continue_train_dapo_rollout/best_model.pt"
OUTPUT_DIR="/data/chenhaotian/verl/experiments/data_generation/outputs/stage2_prm_eval_gsm8k"

# PRM配置
THRESHOLD=${THRESHOLD:-0.5}
RESPONSE_MAX_TOKENS=40
SEED=${SEED:-42}

echo "============================================================"
echo "PRM评估脚本 - GSM8K数据集"
echo "============================================================"
echo "GPU数量: ${NPROC}"
echo "Batch size: ${BATCH_SIZE}"
echo "PRM阈值: ${THRESHOLD}"
echo "随机种子: ${SEED}"
echo "数据集: ${DATA_PATH}"
echo "模型: ${MODEL_PATH}"
echo "PRM checkpoint: ${PRM_CHECKPOINT}"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# 获取运行模式
MODE=${1:-"score"}

run_extract() {
    echo ">>> 阶段1: 提取Latent States (8卡)"
    echo ">>> 使用gsm8k验证工具分类正负样本并平衡"
    echo ">>> Response截断到前 ${RESPONSE_MAX_TOKENS} 个token"
    echo ""
    
    torchrun --nproc_per_node=${NPROC} ${SCRIPT_PATH} \
        --stage extract \
        --batch_size ${BATCH_SIZE} \
        --data_path "${DATA_PATH}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --response_max_tokens ${RESPONSE_MAX_TOKENS} \
        --seed ${SEED}
    
    echo ""
    echo ">>> 阶段1完成"
    echo ""
}

run_score() {
    echo ">>> 阶段2: 分布式打分和评估 (8卡)"
    echo ">>> 使用PRM checkpoint: ${PRM_CHECKPOINT}"
    echo ">>> 阈值: ${THRESHOLD}"
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
        echo "用法: bash run_prm_eval_gsm8k.sh [extract|score|all]"
        echo "  extract: 只运行阶段1（提取latent states）"
        echo "  score:   只运行阶段2（打分和评估）- 默认"
        echo "  all:     运行两个阶段"
        exit 1
        ;;
esac

echo "============================================================"
echo "评估完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo "  - prm_eval_results.jsonl: 详细结果"
echo "  - prm_eval_summary.json: 评估摘要"
echo "============================================================"
