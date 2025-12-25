#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版验证脚本 - 检查对话格式（不需要 transformers）
"""

import sys
from pathlib import Path

# 添加verl到Python路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))

import pandas as pd
import json


def verify_data_format(input_parquet: str = "outputs/stage3_temp_input.parquet"):
    """
    验证数据格式（不加载 tokenizer）
    """
    print("=" * 100)
    print("Stage3 续写数据格式验证")
    print("=" * 100)
    
    # 1. 读取数据
    print(f"\n[1/3] 读取数据: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    print(f"✓ 共 {len(df)} 条数据")
    print(f"✓ 列名: {df.columns.tolist()}")
    
    if 'question' not in df.columns:
        print("❌ 错误：缺少 'question' 列")
        return False
    
    # 2. 检查数据格式
    print(f"\n[2/3] 检查数据格式（前3个样本）")
    
    for idx in range(min(3, len(df))):
        print(f"\n样本 {idx + 1}:")
        print("-" * 100)
        
        conversation = df['question'].iloc[idx]
        
        # 检查是否是列表
        if not isinstance(conversation, (list, tuple)):
            # 可能是 numpy array
            conversation = list(conversation)
        
        print(f"✓ 数据类型: {type(conversation)}")
        print(f"✓ 对话轮数: {len(conversation)}")
        
        # 检查每一轮
        for i, turn in enumerate(conversation):
            if not isinstance(turn, dict):
                print(f"  ❌ 第 {i+1} 轮不是字典: {type(turn)}")
                return False
            
            if 'role' not in turn or 'content' not in turn:
                print(f"  ❌ 第 {i+1} 轮缺少 role 或 content")
                return False
            
            role = turn['role']
            content = turn['content']
            content_len = len(content)
            content_preview = content[:150] + "..." if len(content) > 150 else content
            
            print(f"  第 {i+1} 轮:")
            print(f"    - role: {role}")
            print(f"    - content 长度: {content_len} 字符")
            print(f"    - 内容预览: {content_preview}")
        
        # 验证最后一轮
        last_turn = conversation[-1]
        if last_turn['role'] == 'assistant':
            print(f"\n✅ 最后一轮是 assistant（未完成的回答）")
            print(f"   这是正确的续写格式！")
        else:
            print(f"\n⚠️  最后一轮是 {last_turn['role']}，不是 assistant")
        
        print("-" * 100)
    
    # 3. 说明正确用法
    print(f"\n[3/3] 正确的处理方法")
    print("=" * 100)
    print()
    print("你的数据格式是：")
    print()
    print("df['question'] = [")
    print("  {")
    print('    "role": "user",')
    print('    "content": "问题内容..."')
    print("  },")
    print("  {")
    print('    "role": "assistant",')
    print('    "content": "未完成的回答..."')
    print("  }")
    print("]")
    print()
    print("✅ 正确用法：")
    print()
    print("```python")
    print("# 直接传入整个对话列表")
    print("formatted_text = tokenizer.apply_chat_template(")
    print("    conversation,  # 直接使用，不要再包装")
    print("    tokenize=False,")
    print("    add_generation_prompt=True  # 关键参数")
    print(")")
    print("```")
    print()
    print("❌ 错误用法：")
    print()
    print("```python")
    print("# 不要这样做！会导致嵌套")
    print("formatted_text = tokenizer.apply_chat_template(")
    print('    [{"role": "user", "content": conversation}],  # 错误！')
    print("    tokenize=False,")
    print("    add_generation_prompt=True")
    print(")")
    print("```")
    print()
    print("=" * 100)
    print()
    print("🔑 关键点：")
    print("  1. 你的数据已经是对话列表格式")
    print("  2. 直接传入 tokenizer.apply_chat_template()")
    print("  3. 必须设置 add_generation_prompt=True")
    print("  4. 模型会自动从最后的 assistant 回答继续生成")
    print()
    print("✅ 验证通过！数据格式正确")
    print()
    
    return True


if __name__ == "__main__":
    verify_data_format()

