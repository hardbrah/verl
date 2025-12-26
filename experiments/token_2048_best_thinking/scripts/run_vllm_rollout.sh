#!/bin/bash
# vLLM Rollout 启动脚本
# 使用 vLLM 引擎进行轻量化的 rollout

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 进入项目目录
cd "$PROJECT_DIR"

echo "========================================="
echo "vLLM Rollout 脚本"
echo "========================================="
echo "项目目录: $PROJECT_DIR"
echo ""

# 检查是否安装了 vLLM
if ! python -c "import vllm" 2>/dev/null; then
    echo "错误: vLLM 未安装"
    echo "请运行: pip install vllm"
    exit 1
fi

# 默认参数（可以通过环境变量覆盖）
MODEL_PATH="${MODEL_PATH:-/datacenter/models/Qwen/Qwen3-4B-Instruct-2507}"
INPUT_PATH="${INPUT_PATH:-outputs/stage1_sampled_questions.parquet}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/vllm_rollout_output.parquet}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-"-1"}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"

# 显示配置
echo "配置参数："
echo "  模型路径: $MODEL_PATH"
echo "  输入文件: $INPUT_PATH"
echo "  输出文件: $OUTPUT_PATH"
echo "  采样数: $N_SAMPLES"
echo "  最大 tokens: $MAX_TOKENS"
echo "  温度: $TEMPERATURE"
echo "  Top-p: $TOP_P"
echo "  Top-k: $TOP_K"
echo "  张量并行: $TENSOR_PARALLEL_SIZE"
echo "  显存利用率: $GPU_MEMORY_UTILIZATION"
echo "  数据类型: $DTYPE"
echo "  随机种子: $SEED"
echo ""

# 检查输入文件是否存在
if [ ! -f "$INPUT_PATH" ]; then
    echo "错误: 输入文件不存在: $INPUT_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_PATH")"

# 执行 rollout
echo "开始执行 rollout..."
echo "========================================="

python scripts/vllm_rollout.py \
    --model_path "$MODEL_PATH" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --n_samples "$N_SAMPLES" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE" \
    --seed "$SEED"

echo ""
echo "========================================="
echo "Rollout 完成！"
echo "输出文件: $OUTPUT_PATH"
echo "========================================="









