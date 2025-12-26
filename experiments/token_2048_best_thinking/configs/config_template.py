#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模板文件
复制此文件为 config.py 并填写实际值
"""

# ==================== 通用配置 ====================
# 随机种子
RANDOM_SEED = None

# ==================== 路径配置 ====================
class PathConfig:
    """路径配置"""
    # 模型路径
    MODEL_PATH = None
    
    # 数据集路径
    DATASET_NAME = None
    DATASET_LOCAL_PATH = None
    
    # 输出目录
    OUTPUT_DIR = None
    
    # Stage 1: 采样和初始生成
    STAGE1_SAMPLED_QUESTIONS = None  # 采样的问题文件
    STAGE1_SAMPLING_INDICES = None   # 采样索引文件
    STAGE1_RAW_OUTPUT = None         # 原始生成结果
    STAGE1_OUTPUT = None             # 最终输出（不含扩展名）
    
    # Stage 2: 继续生成
    STAGE2_CONTINUATION_PROMPTS = None
    
    # Stage 3: 完整rollouts
    STAGE3_FORMATTED_PROMPTS = None
    STAGE3_ROLLOUTS = None
    FINAL_COMPLETE_ROLLOUTS = None


# ==================== 数据采样配置 ====================
class SamplingConfig:
    """数据采样配置（sample_token2048.py）"""
    # 采样数量
    SAMPLE_SIZE = None
    
    # 每个问题生成的样本数
    N_SAMPLES_STAGE1 = None
    
    # Stage 1 生成长度
    MAX_NEW_TOKENS_STAGE1 = None
    
    # 采样温度
    TEMPERATURE_STAGE1 = None
    
    # Top-p采样参数
    TOP_P_STAGE1 = None
    
    # 系统提示词
    SYSTEM_PROMPT = None


# ==================== vLLM 生成配置 ====================
class VLLMConfig:
    """vLLM生成配置（my_vllm.py）"""
    # 模型配置
    DTYPE = None
    TRUST_REMOTE_CODE = None
    
    # 并行配置
    TENSOR_PARALLEL_SIZE = None
    
    # 显存配置
    GPU_MEMORY_UTILIZATION = None
    
    # 模型长度限制
    MAX_MODEL_LEN = None
    MAX_NUM_SEQS = None
    
    # 采样参数
    N_SAMPLES = None
    MAX_NEW_TOKENS = None
    TEMPERATURE = None
    TOP_P = None
    REPETITION_PENALTY = None
    IGNORE_EOS = None
    
    # 数据处理
    PROMPT_LIMIT = None  # 限制处理的prompt数量，None表示全部处理


# ==================== Stage特定配置 ====================
class Stage1Config:
    """阶段1配置：采样问题并生成初始token_2048"""
    N_SAMPLES = None
    MAX_NEW_TOKENS = None
    TEMPERATURE = None
    TOP_P = None
    MAX_MODEL_LEN = None
    TENSOR_PARALLEL_SIZE = None


class Stage3Config:
    """阶段3配置：完整rollouts生成"""
    N_SAMPLES = None
    MAX_NEW_TOKENS = None
    TEMPERATURE = None
    TOP_P = None
    MAX_MODEL_LEN = None
    TENSOR_PARALLEL_SIZE = None
    GPU_MEMORY_UTILIZATION = None
    PROMPT_LIMIT = None  # 处理的prompt数量限制

