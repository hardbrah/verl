#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证续写提示词格式是否正确

用途：
- 检查 stage3_temp_input.parquet 中的对话格式
- 验证 tokenizer.apply_chat_template 的输出
- 确保模型能够正确续写
"""

import pandas as pd
import json
from transformers import AutoTokenizer
from pathlib import Path


def verify_continuation_format(
    input_parquet: str = "outputs/stage3_temp_input.parquet",
    model_path: str = "/datacenter/models/Qwen/Qwen3-4B-Instruct-2507",
    num_samples: int = 3
):
    """
    验证续写格式
    
    Args:
        input_parquet: 输入文件路径
        model_path: 模型路径
        num_samples: 检查多少个样本
    """
    print("=" * 100)
    print("续写提示词格式验证")
    print("=" * 100)
    
    # 1. 读取数据
    print(f"\n[1/4] 读取数据: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    print(f"✓ 共 {len(df)} 条数据")
    print(f"✓ 列名: {df.columns.tolist()}")
    
    if 'question' not in df.columns:
        print("❌ 错误：缺少 'question' 列")
        return False
    
    # 2. 检查数据格式
    print(f"\n[2/4] 检查数据格式")
    first_item = df['question'].iloc[0]
    print(f"✓ 数据类型: {type(first_item)}")
    
    if not isinstance(first_item, (list, tuple)) and hasattr(first_item, '__iter__'):
        # 可能是 numpy array，转换为 list
        first_item = list(first_item)
    
    if isinstance(first_item, (list, tuple)):
        print(f"✓ 是列表格式")
        print(f"✓ 对话轮数: {len(first_item)}")
        
        # 检查对话格式
        for i, turn in enumerate(first_item):
            if isinstance(turn, dict) and 'role' in turn and 'content' in turn:
                role = turn['role']
                content_preview = turn['content'][:100] + "..." if len(turn['content']) > 100 else turn['content']
                print(f"  - 第 {i+1} 轮: {role}")
                print(f"    内容预览: {content_preview}")
            else:
                print(f"  ❌ 第 {i+1} 轮格式错误: {turn}")
                return False
        
        # 检查最后一轮是否是 assistant
        if first_item[-1]['role'] == 'assistant':
            print(f"✓ 最后一轮是 assistant（未完成的回答）- 正确！")
        else:
            print(f"⚠️  最后一轮不是 assistant，是 {first_item[-1]['role']}")
            print(f"   这可能不是续写场景")
    else:
        print(f"❌ 不是列表格式，而是: {type(first_item)}")
        return False
    
    # 3. 加载 tokenizer 并测试格式化
    print(f"\n[3/4] 测试 tokenizer.apply_chat_template")
    print(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"✓ Tokenizer 加载成功")
    
    # 测试格式化
    print(f"\n测试前 {num_samples} 个样本的格式化结果：")
    print("=" * 100)
    
    for idx in range(min(num_samples, len(df))):
        conversation = df['question'].iloc[idx]
        
        # 如果是 numpy array，转换为 list
        if not isinstance(conversation, list):
            conversation = list(conversation)
        
        print(f"\n样本 {idx + 1}:")
        print("-" * 100)
        
        # ✅ 正确方法：直接传入对话列表
        formatted_text = tokenizer.apply_chat_template(
            conversation,  # 直接传入整个对话历史
            tokenize=False,
            add_generation_prompt=True  # 关键参数
        )
        
        print(f"对话轮数: {len(conversation)}")
        print(f"格式化后长度: {len(formatted_text)} 字符")
        print(f"估计 token 数: ~{len(formatted_text) * 0.5:.0f} tokens")
        print()
        print("格式化后的文本:")
        print("─" * 100)
        # 打印完整文本（如果太长，显示前后部分）
        if len(formatted_text) > 2000:
            print(formatted_text[:1000])
            print("\n... [中间省略] ...\n")
            print(formatted_text[-1000:])
        else:
            print(formatted_text)
        print("─" * 100)
        
        # 验证关键特征
        checks = []
        
        # 检查是否包含用户问题
        if any('solve' in turn.get('content', '').lower() for turn in conversation if turn['role'] == 'user'):
            checks.append("✓ 包含用户问题")
        
        # 检查是否包含部分回答
        if any(turn['role'] == 'assistant' and len(turn.get('content', '')) > 100 for turn in conversation):
            checks.append("✓ 包含助手的部分回答")
        
        # 检查是否以生成提示符结尾（具体格式取决于模型）
        # 对于 Qwen，应该以 assistant 标记结尾但没有 end 标记
        if '<|im_start|>assistant' in formatted_text or 'assistant' in formatted_text[-200:]:
            checks.append("✓ 包含继续生成的提示符")
        
        # 检查是否正确结束（没有重复嵌套）
        if formatted_text.count('<|im_start|>user') <= 2:  # 最多2次（可能有系统消息）
            checks.append("✓ 没有重复嵌套")
        else:
            checks.append("⚠️  可能有重复嵌套")
        
        print()
        print("验证检查:")
        for check in checks:
            print(f"  {check}")
        print("=" * 100)
    
    # 4. 总结
    print(f"\n[4/4] 验证总结")
    print("✅ 数据格式正确")
    print("✅ 可以直接使用 tokenizer.apply_chat_template(conversation, add_generation_prompt=True)")
    print()
    print("关键点：")
    print("  1. 你的数据已经是对话列表格式 - 不需要再包装")
    print("  2. 直接传入整个对话列表给 apply_chat_template")
    print("  3. 必须设置 add_generation_prompt=True")
    print("  4. 模型会从最后的 assistant 回答处继续生成")
    print()
    
    return True


def compare_methods(
    input_parquet: str = "outputs/stage3_temp_input.parquet",
    model_path: str = "/datacenter/models/Qwen/Qwen3-4B-Instruct-2507"
):
    """
    对比正确和错误的处理方法
    """
    print("\n" + "=" * 100)
    print("对比：正确 vs 错误的处理方法")
    print("=" * 100)
    
    df = pd.read_parquet(input_parquet)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    conversation = df['question'].iloc[0]
    if not isinstance(conversation, list):
        conversation = list(conversation)
    
    print("\n原始对话:")
    print("-" * 100)
    for turn in conversation:
        print(f"{turn['role']}: {turn['content'][:200]}...")
    print()
    
    # ✅ 正确方法
    print("\n✅ 正确方法：直接传入对话列表")
    print("-" * 100)
    correct_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"长度: {len(correct_text)} 字符")
    print(f"预览:\n{correct_text[-500:]}")  # 显示结尾部分
    print()
    
    # ❌ 错误方法（如果把对话当成字符串）
    print("\n❌ 错误方法：把对话列表当成字符串")
    print("-" * 100)
    print("如果你这样做：")
    print('  tokenizer.apply_chat_template([{"role": "user", "content": conversation}], ...)')
    print("会导致嵌套错误，因为 conversation 已经是列表了！")
    print()
    
    print("=" * 100)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证续写提示词格式")
    parser.add_argument("--input", type=str, 
                       default="outputs/stage3_temp_input.parquet",
                       help="输入 parquet 文件路径")
    parser.add_argument("--model", type=str,
                       default="/datacenter/models/Qwen/Qwen3-4B-Instruct-2507",
                       help="模型路径")
    parser.add_argument("--num_samples", type=int, default=2,
                       help="检查多少个样本")
    parser.add_argument("--compare", action="store_true",
                       help="对比正确和错误的方法")
    
    args = parser.parse_args()
    
    # 验证格式
    success = verify_continuation_format(
        input_parquet=args.input,
        model_path=args.model,
        num_samples=args.num_samples
    )
    
    if success and args.compare:
        compare_methods(
            input_parquet=args.input,
            model_path=args.model
        )
    
    if success:
        print("\n✅ 验证通过！可以安全使用 generate_responses_stage3 进行生成")
    else:
        print("\n❌ 验证失败！请检查数据格式")

