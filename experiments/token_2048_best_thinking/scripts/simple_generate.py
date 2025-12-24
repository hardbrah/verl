#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的生成脚本，直接使用 transformers，不依赖 verl 的复杂框架
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 添加verl到Python路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))

def generate_responses(
    model_path: str,
    input_parquet: str,
    output_parquet: str,
    n_samples: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_size: int = 4,
    device: str = "cuda"
):
    """
    使用 transformers 直接生成响应
    
    Args:
        model_path: 模型路径
        input_parquet: 输入文件路径 (包含 prompt 列)
        output_parquet: 输出文件路径
        n_samples: 每个问题生成多少个响应
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: nucleus sampling参数
        batch_size: 批量大小
        device: 设备 (cuda 或 cpu)
    """
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print(f"Loading data from {input_parquet}")
    df = pd.read_parquet(input_parquet)
    
    # 确保有 prompt 列
    if 'prompt' not in df.columns:
        raise ValueError("Input data must have a 'prompt' column")
    
    all_responses = []
    
    # 处理每一行数据
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        prompt = row['prompt']
        
        # 为每个prompt生成n_samples个响应
        responses_for_this_prompt = []
        
        for sample_idx in range(n_samples):
            # 应用chat template
            if isinstance(prompt, list):
                # 如果prompt已经是chat格式的list
                text = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 如果是字符串
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            
            # 解码 (只取新生成的token)
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            responses_for_this_prompt.append(generated_text)
        
        all_responses.append(responses_for_this_prompt)
    
    # 将响应添加到dataframe
    df['responses'] = all_responses
    
    # 保存结果
    os.makedirs(os.path.dirname(output_parquet) or '.', exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Results saved to {output_parquet}")
    print(f"Generated {len(df)} prompts × {n_samples} responses = {len(df) * n_samples} total responses")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple generation script using transformers")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of samples per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    
    generate_responses(
        model_path=args.model_path,
        input_parquet=args.input,
        output_parquet=args.output,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size
    )

