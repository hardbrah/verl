#!/bin/bash
# vLLM Rollout 快捷命令脚本
# 提供常用操作的快捷方式

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 默认参数
MODEL_PATH="${MODEL_PATH:-/datacenter/models/Qwen/Qwen3-4B-Instruct-2507}"

function show_help() {
    cat << EOF
vLLM Rollout 快捷命令

用法: bash scripts/vllm_shortcuts.sh <命令> [参数]

命令:
  test          快速测试（使用少量数据）
  test-full     完整测试
  rollout       执行 rollout（需要指定输入输出文件）
  rollout-demo  使用演示数据执行 rollout
  analyze       分析结果
  flatten       展平结果
  export        导出为 JSONL
  help          显示此帮助信息

环境变量:
  MODEL_PATH    模型路径（默认: /datacenter/models/Qwen/Qwen3-4B-Instruct-2507）
  INPUT_PATH    输入文件路径
  OUTPUT_PATH   输出文件路径
  N_SAMPLES     采样数（默认: 8）
  MAX_TOKENS    最大 tokens（默认: 2048）

示例:
  # 快速测试
  bash scripts/vllm_shortcuts.sh test

  # 使用自定义模型测试
  MODEL_PATH=/path/to/model bash scripts/vllm_shortcuts.sh test

  # 执行 rollout
  INPUT_PATH=data/input.parquet OUTPUT_PATH=data/output.parquet \\
    bash scripts/vllm_shortcuts.sh rollout

  # 分析结果
  bash scripts/vllm_shortcuts.sh analyze data/output.parquet

EOF
}

function cmd_test() {
    echo "========================================="
    echo "快速测试 vLLM Rollout"
    echo "========================================="
    python scripts/test_vllm_rollout.py --model_path "$MODEL_PATH"
}

function cmd_test_full() {
    echo "========================================="
    echo "完整测试 vLLM Rollout"
    echo "========================================="
    python scripts/test_vllm_rollout.py --model_path "$MODEL_PATH" --full
}

function cmd_rollout() {
    if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
        echo "错误: 必须设置 INPUT_PATH 和 OUTPUT_PATH 环境变量"
        echo "示例: INPUT_PATH=input.parquet OUTPUT_PATH=output.parquet bash $0 rollout"
        exit 1
    fi
    
    echo "========================================="
    echo "执行 vLLM Rollout"
    echo "========================================="
    echo "输入: $INPUT_PATH"
    echo "输出: $OUTPUT_PATH"
    
    python scripts/vllm_rollout.py \
        --model_path "$MODEL_PATH" \
        --input "$INPUT_PATH" \
        --output "$OUTPUT_PATH" \
        --n_samples "${N_SAMPLES:-8}" \
        --max_tokens "${MAX_TOKENS:-2048}" \
        --temperature "${TEMPERATURE:-1.0}" \
        --top_p "${TOP_P:-0.95}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
}

function cmd_rollout_demo() {
    echo "========================================="
    echo "使用演示数据执行 Rollout"
    echo "========================================="
    
    # 创建演示数据
    python -c "
import pandas as pd
import os

demo_data = pd.DataFrame({
    'q_id': [0, 1, 2, 3, 4],
    'question': [
        'What is 2 + 2?',
        'Explain photosynthesis.',
        'Write a short poem about spring.',
        'What is the capital of France?',
        'How does a computer work?'
    ],
    'prompt': [
        [{'role': 'user', 'content': 'What is 2 + 2?'}],
        [{'role': 'user', 'content': 'Explain photosynthesis.'}],
        [{'role': 'user', 'content': 'Write a short poem about spring.'}],
        [{'role': 'user', 'content': 'What is the capital of France?'}],
        [{'role': 'user', 'content': 'How does a computer work?'}]
    ]
})

os.makedirs('outputs', exist_ok=True)
demo_data.to_parquet('outputs/demo_input.parquet', index=False)
print('✓ 演示数据已创建: outputs/demo_input.parquet')
"
    
    # 执行 rollout
    INPUT_PATH="outputs/demo_input.parquet" \
    OUTPUT_PATH="outputs/demo_output.parquet" \
    N_SAMPLES="${N_SAMPLES:-4}" \
    MAX_TOKENS="${MAX_TOKENS:-512}" \
    cmd_rollout
}

function cmd_analyze() {
    local input_file="${1:-}"
    if [ -z "$input_file" ]; then
        echo "错误: 必须指定输入文件"
        echo "用法: bash $0 analyze <input_file>"
        exit 1
    fi
    
    echo "========================================="
    echo "分析 Rollout 结果"
    echo "========================================="
    python scripts/process_rollout_results.py analyze --input "$input_file"
}

function cmd_flatten() {
    local input_file="${1:-}"
    local output_file="${2:-}"
    
    if [ -z "$input_file" ] || [ -z "$output_file" ]; then
        echo "错误: 必须指定输入和输出文件"
        echo "用法: bash $0 flatten <input_file> <output_file>"
        exit 1
    fi
    
    echo "========================================="
    echo "展平 Rollout 结果"
    echo "========================================="
    python scripts/process_rollout_results.py flatten --input "$input_file" --output "$output_file"
}

function cmd_export() {
    local input_file="${1:-}"
    local output_file="${2:-}"
    
    if [ -z "$input_file" ] || [ -z "$output_file" ]; then
        echo "错误: 必须指定输入和输出文件"
        echo "用法: bash $0 export <input_file> <output_file>"
        exit 1
    fi
    
    echo "========================================="
    echo "导出为 JSONL"
    echo "========================================="
    python scripts/process_rollout_results.py export --input "$input_file" --output "$output_file"
}

# 主逻辑
case "${1:-}" in
    test)
        cmd_test
        ;;
    test-full)
        cmd_test_full
        ;;
    rollout)
        cmd_rollout
        ;;
    rollout-demo)
        cmd_rollout_demo
        ;;
    analyze)
        cmd_analyze "${2:-}"
        ;;
    flatten)
        cmd_flatten "${2:-}" "${3:-}"
        ;;
    export)
        cmd_export "${2:-}" "${3:-}"
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "错误: 未知命令 '$1'"
        echo ""
        show_help
        exit 1
        ;;
esac



