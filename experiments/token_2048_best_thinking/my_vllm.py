#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 vllm 的生成脚本，用于高效生成数据（Rollout）
"""

import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from configs.config import PathConfig, Stage3Config


def generate_responses(
    model_path: str,
    formatted_prompts_path: str,
    stage2_continuation_prompts_path: str,
    stage3_rollouts_path: str,
    final_complete_rollouts_path: str,
    n_samples: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    gpu_memory_utilization: float = 0.95,
    tensor_parallel_size: int = 2,
    max_model_len: int = 32768,
    max_num_seqs: int = 256,
    repetition_penalty: float = 1.05,
    ignore_eos: bool = False,
    prompt_limit: int = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
):
    """
    使用 vllm 高效生成响应
    
    Args:
        model_path: 模型路径
        formatted_prompts_path: 输入的格式化prompts文件路径
        stage2_continuation_prompts_path: stage2继续生成的prompts路径
        stage3_rollouts_path: stage3 rollouts输出路径
        final_complete_rollouts_path: 最终完整rollouts输出路径
        n_samples: 每个问题生成多少个响应 (Best-of-N / Rollout)
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: nucleus sampling参数
        gpu_memory_utilization: KV cache 预留显存比例
        tensor_parallel_size: 使用多少张卡进行张量并行
        max_model_len: 模型最大长度
        max_num_seqs: 最大并行序列数
        repetition_penalty: 重复惩罚
        ignore_eos: 是否忽略EOS token
        prompt_limit: 限制处理的prompt数量，None表示全部处理
        dtype: 模型数据类型
        trust_remote_code: 是否信任远程代码
    """
    # 1.读取 formatted_prompts.json
    with open(formatted_prompts_path, "r") as f:
        formatted_prompts = json.load(f)
    
    # 限制处理的prompt数量
    if prompt_limit is not None:
        formatted_prompts = formatted_prompts[:prompt_limit]
    
    print("Total number of prompts: ", len(formatted_prompts))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. 初始化 vllm 引擎
    print(f"Initializing vLLM engine with model {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    # 3. 设置采样参数
    # vllm 的 n 参数可以直接为一个 prompt 生成多个 output，比 for 循环快得多
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id] if tokenizer.eos_token_id else None,
        repetition_penalty=repetition_penalty,
        ignore_eos=ignore_eos,
    )

    # 4. 执行生成
    print(f"Generating {n_samples} responses for {len(formatted_prompts)} prompts...")
    # vllm 会自动处理 batching，不需要手动分 batch
    outputs = llm.generate(formatted_prompts, sampling_params)

    # 5. 解析结果
    all_responses = []
    
    # 6. 读取 stage2_continuation_prompts.jsonl
    # key:token_id q_id question gt_answer token_2048
    # outputs是使用上述每个条目再继续sample的结果
    # 将结果保存到 final_complete_rollouts_path
    for output in outputs:
        # output.outputs 是一个 list，长度为 n_samples
        batch_responses = [o.text for o in output.outputs]
        all_responses.append(batch_responses)
    print("Stage3 rollouts generated")
    print("="*100)
    with open(stage3_rollouts_path, "w") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=4)
    print("Stage3 rollouts saved")
    print("="*100)

    with open(final_complete_rollouts_path, "a") as f:
        with open(stage2_continuation_prompts_path, "r") as f_1:
            for line, responses in zip(f_1, all_responses):
                data = json.loads(line)
                token_id = data["token_id"]
                q_id = data["q_id"]
                question = data["question"]
                gt_answer = data["gt_answer"]
                token_2048 = data["token_2048"]
                for response in responses:
                    f.write(json.dumps({
                        "token_id": token_id,
                        "q_id": q_id,
                        "question": question,
                        "gt_answer": gt_answer,
                        "token_2048": token_2048,
                        "complete_rollout": token_2048+response,
                    }) + "\n")





if __name__ == "__main__":
    print("="*100)
    print("Starting generation...")
    print("="*100)
    generate_responses(
        model_path=PathConfig.MODEL_PATH,
        formatted_prompts_path=PathConfig.STAGE3_FORMATTED_PROMPTS,
        stage2_continuation_prompts_path=PathConfig.STAGE2_CONTINUATION_PROMPTS,
        stage3_rollouts_path=PathConfig.STAGE3_ROLLOUTS,
        final_complete_rollouts_path=PathConfig.FINAL_COMPLETE_ROLLOUTS,
        n_samples=Stage3Config.N_SAMPLES,
        max_new_tokens=Stage3Config.MAX_NEW_TOKENS,
        temperature=Stage3Config.TEMPERATURE,
        top_p=Stage3Config.TOP_P,
        max_model_len=Stage3Config.MAX_MODEL_LEN,
        tensor_parallel_size=Stage3Config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=Stage3Config.GPU_MEMORY_UTILIZATION,
        prompt_limit=Stage3Config.PROMPT_LIMIT,
    )
    print("="*100)
    print("Generation completed")
    print("="*100)