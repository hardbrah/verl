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
    input_json_path: str,
    output_format_json_path: str,
    output_rollouts_path: str,
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
        input_json_path: 输入的json文件路径
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
    with open(input_json_path, "r") as f:
        data = json.load(f)
    
    # 限制处理的prompt数量
    if prompt_limit is not None:
        data = data[:prompt_limit]
    
        
    print("Total number of prompts: ", len(data))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    questions = []
    for item in data:
        question = item["question"]
        questions.append(question)
    formatted_prompts = tokenizer.apply_chat_template(questions, add_generation_prompt=True, tokenize=False)
    with open(output_format_json_path, "w") as f:
        for question, formatted_prompt in zip(questions, formatted_prompts):
            f.write(json.dumps({
                "question": question,
                "formatted_prompt": formatted_prompt,
            }) + "\n")

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
    print(f"Generating {n_samples} responses for {len(data)} prompts...")
    # vllm 会自动处理 batching，不需要手动分 batch
    outputs = llm.generate(data, sampling_params)

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
    print("rollouts generated")
    print("="*100)
    with open(output_rollouts_path, "w") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=4)
    print("rollouts saved")
    print("="*100)
