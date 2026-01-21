#!/bin/bash
# ============================================================
# PRM评估脚本 - 使用8卡进行分布式打分
# ============================================================
#
# 功能说明:
#   1. 阶段1 (extract): 使用8卡并行从Qwen3-4B模型提取第15层latent states
#   2. 阶段2 (score_dist): 使用8卡并行对latent states进行PRM打分并评估
#
# 使用方法:
#   # 只运行阶段2（如果latent states已缓存）:
#   bash run_prm_eval.sh score
#
#   # 运行两个阶段:
#   bash run_prm_eval.sh all
#
#   # 只运行阶段1:
#   bash run_prm_eval.sh extract
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/eval_prm_rollout.py"

# 配置参数
NPROC=${NPROC:-8}
BATCH_SIZE=${BATCH_SIZE:-64}

# 数据和模型路径
DATA_PATH="/data/chenhaotian/verl/experiments/data_generation/outputs/qwen3_4b_1000query_16sample/rollout_20260107_134649.jsonl"
MODEL_PATH="/data/models/Qwen/Qwen3-4B-Instruct-2507"
PRM_CHECKPOINT="/data/chenhaotian/latentqa/output/continue_train_dapo_rollout/best_model.pt"
CACHE_DIR="/data/chenhaotian/verl/experiments/data_generation/outputs/stage2_prm_latent_cache"
OUTPUT_DIR="/data/chenhaotian/verl/experiments/data_generation/outputs/stage2_prm_eval"

# PRM配置
THRESHOLD=${THRESHOLD:-0.5}
RESPONSE_MAX_TOKENS=64

echo "============================================================"
echo "PRM评估脚本"
echo "============================================================"
echo "GPU数量: ${NPROC}"
echo "Batch size: ${BATCH_SIZE}"
echo "PRM阈值: ${THRESHOLD}"
echo "数据集: ${DATA_PATH}"
echo "模型: ${MODEL_PATH}"
echo "PRM checkpoint: ${PRM_CHECKPOINT}"
echo "缓存目录: ${CACHE_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# 获取运行模式
MODE=${1:-"score"}

run_extract() {
    echo ">>> 阶段1: 提取Latent States (8卡)"
    echo ">>> 使用模型 ${MODEL_PATH} 提取第15层的latent states"
    echo ">>> Response截断到前 ${RESPONSE_MAX_TOKENS} 个token"
    echo ""
    
    torchrun --nproc_per_node=${NPROC} ${SCRIPT_PATH} \
        --stage extract \
        --batch_size ${BATCH_SIZE} \
        --data_path "${DATA_PATH}" \
        --model_path "${MODEL_PATH}" \
        --cache_dir "${CACHE_DIR}" \
        --response_max_tokens ${RESPONSE_MAX_TOKENS}
    
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
        --stage score_dist \
        --batch_size ${BATCH_SIZE} \
        --model_path "${MODEL_PATH}" \
        --prm_checkpoint "${PRM_CHECKPOINT}" \
        --cache_dir "${CACHE_DIR}" \
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
        echo "用法: bash run_prm_eval.sh [extract|score|all]"
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
