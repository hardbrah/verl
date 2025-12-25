#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Token 2048 最佳思考实验 - 完整运行脚本

set -e  # 遇到错误立即退出
export CUDA_VISIBLE_DEVICES="6,7"
# export VLLM_USE_V1=0  # 强制使用 vLLM v0，避免 v1 的 CUDA 多进程问题

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
python -c "import multiprocessing as mp; print(mp.get_start_method(allow_none=True))"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""
}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

print_section "Token 2048 最佳思考实验"
print_info "工作目录: $SCRIPT_DIR"
print_info "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 记录开始时间
START_TIME=$(date +%s)

# ============================================================================
# 阶段1：采样1k问题并生成8个token_2048
# ============================================================================
print_section "阶段1：采样1k问题并生成8个token_2048"

if [ -f "outputs/stage1_output.parquet" ]; then
    print_warning "检测到阶段1输出已存在，跳过"
    print_info "如需重新运行，请删除 outputs/stage1_output.parquet"
else
    print_info "运行阶段1..."
    python scripts/stage1_sample.py || {
        print_error "阶段1执行失败"
        exit 1
    }
    print_success "阶段1完成"
fi

# 计算阶段1耗时
STAGE1_TIME=$(date +%s)
STAGE1_DURATION=$((STAGE1_TIME - START_TIME))
print_info "阶段1耗时: $(($STAGE1_DURATION / 60)) 分钟"

# ============================================================================
# 阶段2：准备继续生成的数据
# ============================================================================
print_section "阶段2：准备继续生成的数据"

if [ -f "outputs/stage2_input.parquet" ]; then
    print_warning "检测到阶段2输出已存在，跳过"
    print_info "如需重新运行，请删除 outputs/stage2_input.parquet"
else
    print_info "运行阶段2..."
    python scripts/stage2_prepare_data.py || {
        print_error "阶段2执行失败"
        exit 1
    }
    print_success "阶段2完成"
fi

STAGE2_TIME=$(date +%s)
STAGE2_DURATION=$((STAGE2_TIME - STAGE1_TIME))
print_info "阶段2耗时: $STAGE2_DURATION 秒"

# ============================================================================
# 阶段3：对每个token_2048继续生成100次
# ============================================================================
print_section "阶段3：对每个token_2048继续生成100次"
print_warning "这个阶段将花费较长时间（预计10小时左右）"

if [ -f "outputs/stage3_output.parquet" ]; then
    print_warning "检测到阶段3输出已存在，跳过"
    print_info "如需重新运行，请删除 outputs/stage3_output.parquet"
else
    print_info "运行阶段3..."
    python scripts/stage3_continuation.py || {
        print_error "阶段3执行失败"
        exit 1
    }
    print_success "阶段3完成"
fi

STAGE3_TIME=$(date +%s)
STAGE3_DURATION=$((STAGE3_TIME - STAGE2_TIME))
print_info "阶段3耗时: $(($STAGE3_DURATION / 60)) 分钟"

# ============================================================================
# 阶段4：计算准确率并选择最佳token_2048
# ============================================================================
print_section "阶段4：计算准确率并选择最佳token_2048"

print_info "运行阶段4..."
python scripts/stage4_select_best.py || {
    print_error "阶段4执行失败"
    exit 1
}
print_success "阶段4完成"

STAGE4_TIME=$(date +%s)
STAGE4_DURATION=$((STAGE4_TIME - STAGE3_TIME))
print_info "阶段4耗时: $STAGE4_DURATION 秒"

# ============================================================================
# 实验完成
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

print_section "实验全部完成！"
print_success "总耗时: $(($TOTAL_DURATION / 3600)) 小时 $(($TOTAL_DURATION % 3600 / 60)) 分钟"
print_info "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
print_info "输出文件："
echo "  ✓ outputs/stage1_output.parquet - 阶段1输出 (8000条)"
echo "  ✓ outputs/stage2_input.parquet - 阶段2输入 (8000条)"
echo "  ✓ outputs/stage3_output.parquet - 阶段3输出 (800000条)"
echo "  ✓ outputs/final_best.parquet - 最终结果 (1000条)"
echo "  ✓ outputs/analysis_report.txt - 分析报告"
echo "  ✓ outputs/accuracy_distribution.png - 可视化图表（如果matplotlib可用）"

echo ""
print_info "查看分析报告："
echo "  cat outputs/analysis_report.txt"

echo ""
print_info "快速统计："
python -c "
import pandas as pd
df = pd.read_parquet('outputs/final_best.parquet')
print(f'  平均最佳准确率: {df[\"best_acc\"].mean():.2%}')
print(f'  准确率标准差: {df[\"best_acc\"].std():.4f}')
print(f'  最高准确率: {df[\"best_acc\"].max():.2%}')
print(f'  最低准确率: {df[\"best_acc\"].min():.2%}')
" 2>/dev/null || print_warning "无法显示快速统计"

echo ""
print_success "实验成功完成！"

