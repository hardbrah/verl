#!/bin/bash
# GSM8K Rollout 生成脚本
# 
# 功能：
# 1. 使用 gsm8k_dataloader.py 采样并格式化 GSM8K 数据
# 2. 使用 data_gen_vllm.py 进行 rollout 生成
#
# 用法：
#   bash run_gsm8k_rollout.sh [sample_size] [n_samples]
#   
#   sample_size: 采样的问题数量（默认 1000）
#   n_samples: 每个问题生成的回复数量（默认 16）
#
# 示例：
#   bash run_gsm8k_rollout.sh 500 8    # 采样 500 个问题，每个生成 8 个回复

set -e

# ==================== 配置参数 ====================
# 从命令行参数获取，或使用默认值
SAMPLE_SIZE=${1:-8000}
N_SAMPLES=${2:-16}

# 模型路径
MODEL_PATH="/data/models/Qwen/Qwen3-4B-Instruct-2507"

# GSM8K 数据集路径
DATASET_PATH="/data/datasets/openai/gsm8k/main/train-00000-of-00001.parquet"

# 输出目录
OUTPUT_DIR="/data/chenhaotian/verl/experiments/data_generation/outputs/gsm8k"
ROLLOUT_OUTPUT_DIR="${OUTPUT_DIR}/rollouts"

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 随机种子
SEED=42

# vLLM 配置
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=10240
MAX_NUM_SEQS=512

# 采样参数
MAX_NEW_TOKENS=8192
TEMPERATURE=0.7
TOP_P=0.95
REPETITION_PENALTY=1.00

# 保存配置
SAVE_BATCH_SIZE=500

# ==================== 打印配置 ====================
echo "=========================================="
echo "GSM8K Rollout 生成配置"
echo "=========================================="
echo "数据配置:"
echo "  - 采样数量: ${SAMPLE_SIZE}"
echo "  - 数据集路径: ${DATASET_PATH}"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo ""
echo "模型配置:"
echo "  - 模型路径: ${MODEL_PATH}"
echo "  - 张量并行: ${TENSOR_PARALLEL_SIZE}"
echo "  - GPU 显存利用率: ${GPU_MEMORY_UTILIZATION}"
echo "  - 最大模型长度: ${MAX_MODEL_LEN}"
echo ""
echo "生成配置:"
echo "  - 每问题采样数: ${N_SAMPLES}"
echo "  - 最大生成 token: ${MAX_NEW_TOKENS}"
echo "  - 温度: ${TEMPERATURE}"
echo "  - Top-p: ${TOP_P}"
echo "=========================================="
echo ""

# ==================== Step 1: 数据准备 ====================
echo "Step 1: 准备 GSM8K 数据..."
echo "----------------------------------------"

python "${SCRIPT_DIR}/gsm8k_dataloader.py" \
    --sample-size ${SAMPLE_SIZE} \
    --model-path "${MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed ${SEED}

# 检查输出文件
PROMPTS_PATH="${OUTPUT_DIR}/formatted_prompts.json"
QUESTIONS_PATH="${OUTPUT_DIR}/sampled_questions.jsonl"

if [ ! -f "${PROMPTS_PATH}" ]; then
    echo "错误: 未找到格式化 prompts 文件: ${PROMPTS_PATH}"
    exit 1
fi

if [ ! -f "${QUESTIONS_PATH}" ]; then
    echo "错误: 未找到问题元数据文件: ${QUESTIONS_PATH}"
    exit 1
fi

echo ""
echo "数据准备完成！"
echo "  - Prompts: ${PROMPTS_PATH}"
echo "  - Questions: ${QUESTIONS_PATH}"
echo ""

# ==================== Step 2: Rollout 生成 ====================
echo "Step 2: 开始 Rollout 生成..."
echo "----------------------------------------"

python "${SCRIPT_DIR}/data_gen_vllm.py" \
    --model-path "${MODEL_PATH}" \
    --prompts-path "${PROMPTS_PATH}" \
    --questions-path "${QUESTIONS_PATH}" \
    --output-dir "${ROLLOUT_OUTPUT_DIR}" \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --n-samples ${N_SAMPLES} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --repetition-penalty ${REPETITION_PENALTY} \
    --save-batch-size ${SAVE_BATCH_SIZE} \
    --enable-resume

echo ""
echo "=========================================="
echo "GSM8K Rollout 生成完成！"
echo "=========================================="
echo "输出目录: ${ROLLOUT_OUTPUT_DIR}"
echo ""
