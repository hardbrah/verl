#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 vLLM 的 Rollout 生成脚本

功能：
1. 分批生成和保存 responses（避免内存溢出和丢失进度）
2. 记录详细的元信息（模型、参数、统计等）
3. 支持断点续传（中断后自动从上次进度继续）
4. 可选的 WandB 监控
5. 完成时自动合并所有分批文件

使用方式：
1. 先运行 dataloader.py 准备格式化的 prompts
2. 再运行本脚本进行生成（推荐使用 run_rollout.sh）

输出目录结构：
    rollouts/
        rollout_<timestamp>/
            meta.json                           # 整体元信息
            checkpoint.json                     # 断点续传信息
            responses_00001_01000.jsonl         # 分批保存
            responses_01001_02000.jsonl
            ...
            all_responses_rollout_xxx.jsonl     # 完成后合并的完整文件
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 添加父目录到模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from configs.config import PathConfig, VLLMConfig, RolloutConfig
from rollout_manager import (
    RolloutManager,
    ModelConfig,
    SamplingParams as RolloutSamplingParams,
)


# ==================== 生成器类 ====================

class RolloutGenerator:
    """
    Rollout 生成器
    
    负责使用 vLLM 生成 responses，并与 RolloutManager 协作保存结果。
    """
    
    def __init__(
        self,
        model_path: str,
        # vLLM 引擎配置
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 20480,
        max_num_seqs: int = 256,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        # 采样参数
        n_samples: int = 32,
        max_new_tokens: int = 16384,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        ignore_eos: bool = False,
        # Rollout 管理配置
        save_batch_size: int = 1000,
        output_base_dir: str = "outputs/rollouts",
        enable_resume: bool = True,
        enable_wandb: bool = False,
        wandb_project: str = "rollout",
        wandb_entity: Optional[str] = None,
    ):
        """初始化生成器"""
        self.model_path = model_path
        
        # vLLM 配置
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        
        # 采样参数
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.ignore_eos = ignore_eos
        
        # Rollout 管理
        self.save_batch_size = save_batch_size
        self.output_base_dir = output_base_dir
        self.enable_resume = enable_resume
        self.enable_wandb = enable_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        # 运行时组件（延迟初始化）
        self.tokenizer: Optional[AutoTokenizer] = None
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.rollout_manager: Optional[RolloutManager] = None
    
    def _init_tokenizer(self):
        """初始化 tokenizer"""
        print(f"加载 tokenizer: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
    
    def _init_llm(self):
        """初始化 vLLM 引擎"""
        print(f"初始化 vLLM 引擎: {self.model_path}")
        print(f"  - tensor_parallel_size: {self.tensor_parallel_size}")
        print(f"  - gpu_memory_utilization: {self.gpu_memory_utilization}")
        print(f"  - max_model_len: {self.max_model_len}")
        print(f"  - dtype: {self.dtype}")
        
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
        )
    
    def _init_sampling_params(self):
        """初始化采样参数"""
        stop_token_ids = None
        if self.tokenizer and self.tokenizer.eos_token_id:
            stop_token_ids = [self.tokenizer.eos_token_id]
            if self.tokenizer.pad_token_id:
                stop_token_ids.append(self.tokenizer.pad_token_id)
        
        self.sampling_params = SamplingParams(
            n=self.n_samples,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=self.repetition_penalty,
            ignore_eos=self.ignore_eos,
        )
        
        print(f"采样参数:")
        print(f"  - n_samples: {self.n_samples}")
        print(f"  - max_new_tokens: {self.max_new_tokens}")
        print(f"  - temperature: {self.temperature}")
        print(f"  - top_p: {self.top_p}")
        print(f"  - repetition_penalty: {self.repetition_penalty}")
    
    def _init_rollout_manager(
        self,
        total_questions: int,
        formatted_prompts_path: str,
        sampled_questions_path: str,
    ) -> int:
        """初始化 Rollout 管理器，返回起始索引"""
        model_config = ModelConfig(
            model_path=self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )
        
        sampling_params = RolloutSamplingParams(
            n_samples=self.n_samples,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            ignore_eos=self.ignore_eos,
        )
        
        self.rollout_manager = RolloutManager(
            output_base_dir=self.output_base_dir,
            save_batch_size=self.save_batch_size,
            enable_resume=self.enable_resume,
            enable_wandb=self.enable_wandb,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
        )
        
        start_index = self.rollout_manager.initialize(
            model_config=model_config,
            sampling_params=sampling_params,
            total_questions=total_questions,
            formatted_prompts_path=formatted_prompts_path,
            sampled_questions_path=sampled_questions_path,
        )
        
        return start_index
    
    def _load_data(
        self,
        formatted_prompts_path: str,
        sampled_questions_path: str,
    ) -> tuple:
        """加载输入数据"""
        print(f"加载格式化 prompts: {formatted_prompts_path}")
        with open(formatted_prompts_path, 'r') as f:
            formatted_prompts = json.load(f)
        
        print(f"加载问题元数据: {sampled_questions_path}")
        questions_data = []
        with open(sampled_questions_path, 'r') as f:
            for line in f:
                questions_data.append(json.loads(line))
        
        print(f"共 {len(formatted_prompts)} 个 prompts, {len(questions_data)} 个问题")
        
        return formatted_prompts, questions_data
    
    def _process_output(self, output, prompt: str) -> tuple:
        """
        处理单个 vLLM 输出
        
        Returns:
            (responses, prompt_tokens, completion_tokens_list, finish_reasons)
        """
        responses = []
        completion_tokens_list = []
        finish_reasons = []
        
        # 计算 prompt tokens
        prompt_tokens = len(self.tokenizer.encode(prompt))
        
        for o in output.outputs:
            responses.append(o.text)
            completion_tokens_list.append(len(o.token_ids))
            finish_reasons.append(o.finish_reason or "unknown")
        
        return responses, prompt_tokens, completion_tokens_list, finish_reasons
    
    def generate(
        self,
        formatted_prompts_path: str,
        sampled_questions_path: str,
    ) -> str:
        """
        执行 rollout 生成
        
        Args:
            formatted_prompts_path: 格式化 prompts 文件路径
            sampled_questions_path: 问题元数据文件路径
            
        Returns:
            输出目录路径
        """
        print("=" * 80)
        print("开始 Rollout 生成")
        print("=" * 80)
        
        # 1. 加载数据
        formatted_prompts, questions_data = self._load_data(
            formatted_prompts_path,
            sampled_questions_path,
        )
        total_questions = len(formatted_prompts)
        
        # 2. 初始化组件
        self._init_tokenizer()
        self._init_llm()
        self._init_sampling_params()
        
        start_index = self._init_rollout_manager(
            total_questions=total_questions,
            formatted_prompts_path=formatted_prompts_path,
            sampled_questions_path=sampled_questions_path,
        )
        
        # 3. 生成（vLLM 自动管理批处理和显存）
        print(f"\n开始生成（从问题 {start_index} 开始）...")
        print(f"总问题数: {total_questions}")
        print(f"保存批次大小: {self.save_batch_size}")
        print("=" * 80)
        
        # 获取待生成的 prompts
        prompts_to_generate = formatted_prompts[start_index:]
        questions_to_process = questions_data[start_index:]
        
        try:
            # 一次性传入所有 prompts，vLLM 自动管理批处理
            print(f"\n正在生成 {len(prompts_to_generate)} 个问题的 responses...")
            gen_start_time = time.time()
            outputs = self.llm.generate(prompts_to_generate, self.sampling_params)
            gen_time = time.time() - gen_start_time
            
            print(f"生成完成: {gen_time:.2f}s, {len(prompts_to_generate) / gen_time:.2f} 问题/秒")
            
            # 处理输出并保存
            print("\n正在保存结果...")
            for i, (output, question_data, prompt) in enumerate(
                zip(outputs, questions_to_process, prompts_to_generate)
            ):
                q_id = start_index + i
                
                responses, prompt_tokens, completion_tokens_list, finish_reasons = \
                    self._process_output(output, prompt)
                
                self.rollout_manager.add_responses(
                    q_id=q_id,
                    question=question_data['question'],
                    gt_answer=question_data['gt_answer'],
                    prompt=prompt,
                    responses=responses,
                    prompt_tokens=prompt_tokens,
                    completion_tokens_list=completion_tokens_list,
                    finish_reasons=finish_reasons,
                )
            
            # 4. 完成
            self.rollout_manager.finalize(status="completed")
            
        except KeyboardInterrupt:
            print("\n检测到中断信号，正在保存进度...")
            self.rollout_manager.finalize(status="interrupted")
            print("进度已保存，下次运行将自动续传")
            raise
        
        except Exception as e:
            print(f"\n发生错误: {e}")
            self.rollout_manager.finalize(status="error")
            raise
        
        return self.rollout_manager.get_output_dir()


# ==================== 便捷函数 ====================

def generate_rollouts(
    model_path: str = PathConfig.MODEL_PATH,
    formatted_prompts_path: str = PathConfig.FORMATTED_PROMPTS_JSON,
    sampled_questions_path: str = PathConfig.SAMPLED_QUESTIONS_JSONL,
    # vLLM 配置
    tensor_parallel_size: int = VLLMConfig.TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization: float = VLLMConfig.GPU_MEMORY_UTILIZATION,
    max_model_len: int = VLLMConfig.MAX_MODEL_LEN,
    max_num_seqs: int = VLLMConfig.MAX_NUM_SEQS,
    dtype: str = VLLMConfig.DTYPE,
    trust_remote_code: bool = VLLMConfig.TRUST_REMOTE_CODE,
    # 采样参数
    n_samples: int = VLLMConfig.N_SAMPLES,
    max_new_tokens: int = VLLMConfig.MAX_NEW_TOKENS,
    temperature: float = VLLMConfig.TEMPERATURE,
    top_p: float = VLLMConfig.TOP_P,
    repetition_penalty: float = VLLMConfig.REPETITION_PENALTY,
    # Rollout 配置
    save_batch_size: int = RolloutConfig.SAVE_BATCH_SIZE,
    output_base_dir: str = RolloutConfig.ROLLOUT_OUTPUT_DIR,
    enable_resume: bool = RolloutConfig.ENABLE_RESUME,
    enable_wandb: bool = RolloutConfig.ENABLE_WANDB,
    wandb_project: str = RolloutConfig.WANDB_PROJECT,
    wandb_entity: Optional[str] = RolloutConfig.WANDB_ENTITY,
) -> str:
    """
    便捷函数：执行 rollout 生成
    
    Returns:
        输出目录路径
    """
    generator = RolloutGenerator(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        save_batch_size=save_batch_size,
        output_base_dir=output_base_dir,
        enable_resume=enable_resume,
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    
    return generator.generate(
        formatted_prompts_path=formatted_prompts_path,
        sampled_questions_path=sampled_questions_path,
    )


# ==================== 命令行参数解析 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="vLLM Rollout 生成器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 路径参数
    parser.add_argument("--model-path", type=str, default=PathConfig.MODEL_PATH,
                        help="模型路径")
    parser.add_argument("--prompts-path", type=str, default=PathConfig.FORMATTED_PROMPTS_JSON,
                        help="格式化 prompts 文件路径")
    parser.add_argument("--questions-path", type=str, default=PathConfig.SAMPLED_QUESTIONS_JSONL,
                        help="问题元数据文件路径")
    parser.add_argument("--output-dir", type=str, default=RolloutConfig.ROLLOUT_OUTPUT_DIR,
                        help="输出目录")
    
    # vLLM 配置
    parser.add_argument("--tensor-parallel-size", type=int, default=VLLMConfig.TENSOR_PARALLEL_SIZE,
                        help="张量并行大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=VLLMConfig.GPU_MEMORY_UTILIZATION,
                        help="GPU 显存利用率")
    parser.add_argument("--max-model-len", type=int, default=VLLMConfig.MAX_MODEL_LEN,
                        help="模型最大长度")
    parser.add_argument("--max-num-seqs", type=int, default=VLLMConfig.MAX_NUM_SEQS,
                        help="最大并行序列数")
    parser.add_argument("--dtype", type=str, default=VLLMConfig.DTYPE,
                        help="模型数据类型")
    
    # 采样参数
    parser.add_argument("--n-samples", type=int, default=VLLMConfig.N_SAMPLES,
                        help="每个问题采样数量")
    parser.add_argument("--max-new-tokens", type=int, default=VLLMConfig.MAX_NEW_TOKENS,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=VLLMConfig.TEMPERATURE,
                        help="采样温度")
    parser.add_argument("--top-p", type=float, default=VLLMConfig.TOP_P,
                        help="Top-p 采样参数")
    parser.add_argument("--repetition-penalty", type=float, default=VLLMConfig.REPETITION_PENALTY,
                        help="重复惩罚")
    
    # Rollout 配置
    parser.add_argument("--save-batch-size", type=int, default=RolloutConfig.SAVE_BATCH_SIZE,
                        help="保存批次大小")
    parser.add_argument("--enable-resume", action="store_true", default=RolloutConfig.ENABLE_RESUME,
                        help="启用断点续传")
    parser.add_argument("--no-resume", action="store_true",
                        help="禁用断点续传")
    parser.add_argument("--enable-wandb", action="store_true", default=RolloutConfig.ENABLE_WANDB,
                        help="启用 WandB 监控")
    parser.add_argument("--wandb-project", type=str, default=RolloutConfig.WANDB_PROJECT,
                        help="WandB 项目名称")
    parser.add_argument("--wandb-entity", type=str, default=RolloutConfig.WANDB_ENTITY,
                        help="WandB 实体名称")
    
    return parser.parse_args()


def print_config(args):
    """打印配置信息"""
    print("=" * 80)
    print("Rollout 生成器 - 配置信息")
    print("=" * 80)
    print("\n[路径配置]")
    print(f"  模型路径:           {args.model_path}")
    print(f"  输入 prompts:       {args.prompts_path}")
    print(f"  问题元数据:         {args.questions_path}")
    print(f"  输出目录:           {args.output_dir}")
    print("\n[vLLM 配置]")
    print(f"  tensor_parallel:    {args.tensor_parallel_size}")
    print(f"  gpu_memory_util:    {args.gpu_memory_utilization}")
    print(f"  max_model_len:      {args.max_model_len}")
    print(f"  max_num_seqs:       {args.max_num_seqs}")
    print(f"  dtype:              {args.dtype}")
    print("\n[采样参数]")
    print(f"  n_samples:          {args.n_samples}")
    print(f"  max_new_tokens:     {args.max_new_tokens}")
    print(f"  temperature:        {args.temperature}")
    print(f"  top_p:              {args.top_p}")
    print(f"  repetition_penalty: {args.repetition_penalty}")
    print("\n[Rollout 配置]")
    print(f"  save_batch_size:    {args.save_batch_size}")
    print(f"  断点续传:            {'启用' if args.enable_resume and not args.no_resume else '禁用'}")
    print(f"  WandB 监控:          {'启用' if args.enable_wandb else '禁用'}")
    if args.enable_wandb:
        print(f"  WandB 项目:          {args.wandb_project}")
        print(f"  WandB 实体:          {args.wandb_entity}")
    print("=" * 80)


# ==================== 主入口 ====================

if __name__ == "__main__":
    args = parse_args()
    
    # 打印配置
    print_config(args)
    
    # 处理 resume 参数
    enable_resume = args.enable_resume and not args.no_resume
    
    # 执行生成
    output_dir = generate_rollouts(
        model_path=args.model_path,
        formatted_prompts_path=args.prompts_path,
        sampled_questions_path=args.questions_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        save_batch_size=args.save_batch_size,
        output_base_dir=args.output_dir,
        enable_resume=enable_resume,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    
    print("\n" + "=" * 80)
    print(f"生成完成！输出目录: {output_dir}")
    print("=" * 80)
