#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试文件：给定一个response，调用compute_score函数并打印结果
"""

from verl.utils.reward_score.math_dapo import compute_score
import json


def test_single_response(response: str, ground_truth: str):
    """
    测试单个response的得分
    
    Args:
        response: 模型生成的回答
        ground_truth: 正确答案
    """
    print("=" * 80)
    print("测试 compute_score 函数")
    print("=" * 80)
    
    print("\n输入的 Response:")
    print("-" * 80)
    print(response)
    
    print("\n正确答案 (Ground Truth):")
    print("-" * 80)
    print(ground_truth)
    
    # 调用compute_score函数
    print("\n调用 compute_score 函数...")
    print("-" * 80)
    result = compute_score(response, ground_truth)
    
    # 打印结果
    print("\n结果:")
    print("-" * 80)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 打印得分
    print("\n最终得分:")
    print("-" * 80)
    score = result.get("score", 0)
    if score == 1:
        print(f"✓ 正确！得分: {score}")
    else:
        print(f"✗ 错误！得分: {score}")
    
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    print("\n【示例1：正确答案】\n")
    response1 = """
        We are given:\n\n- $ m > n $, both positive integers.\n- $ a = \\text{lcm}(m, n) $\n- $ b = \\gcd(m, n) $\n- $ a = 30b $\n- $ m - n \\mid a $\n- We are to find $ \\frac{m + n}{b} $\n\n---\n\n### Step 1: Use the identity relating LCM and GCD\n\nWe know that:\n$$\n\\text{lcm}(m, n) \\cdot \\gcd(m, n) = m \\cdot n\n$$\nSo,\n$$\na \\cdot b = m n\n$$\nBut we're told $ a = 30b $, so:\n$$\n(30b) \\cdot b = m n \\Rightarrow 30b^2 = m n\n$$\n\n---\n\n### Step 2: Let $ d = \\gcd(m, n) $\n\nLet:\n- $ m = d \\cdot x $\n- $ n = d \\cdot y $\nwhere $ x > y $, $ \\gcd(x, y) = 1 $ (since we factored out the gcd)\n\nThen:\n- $ b = d $\n- $ a = \\text{lcm}(m, n) = d \\cdot \\text{lcm}(x, y) = d \\cdot x y $ since $ \\gcd(x, y) = 1 $\n\nNow use $ a = 30b $:\n$$\nd x y = 30 d \\Rightarrow x y = 30\n$$\n\nSince $ \\gcd(x, y) = 1 $, and $ x y = 30 $, we need **coprime positive integers** $ x, y $ such that $ x > y $, $ x y = 30 $.\n\nList all factor pairs of 30 with $ x > y $, $ \\gcd(x, y) = 1 $:\n\n- $ (30, 1) $: $ \\gcd(30,1)=1 $ \u2705  \n- $ (15, 2) $: $ \\gcd(15,2)=1 $ \u2705  \n- $ (10, 3) $: $ \\gcd(10,3)=1 $ \u2705  \n- $ (6, 5) $: $ \\gcd(6,5)=1 $ \u2705  \n\nCheck others:\n- $ (5,6) $: $ x < y $, skip\n- $ (3,10) $: $ x < y $, skip\n- $ (2,15) $: skip\n- $ (1,30) $: skip\n\nSo possible coprime pairs with $ x > y $:\n- $ (30,1) $\n- $ (15,2) $\n- $ (10,3) $\n- $ (6,5) $\n\nNow recall: $ m = d x $, $ n = d y $\n\nThen:\n- $ m - n = d(x - y) $\n- $ a = d x y = 30d $\n\nWe are told: $ m - n \\mid a $\n\nSo:\n$$\nd(x - y) \\mid 30d\n$$\nDivide both sides by $ d $ (since $ d > 0 $):\n$$\nx - y \\mid 30\n$$\n\nSo now check which of the above pairs satisfy $ x - y \\mid 30 $\n\nAll differences:\n- $ (30,1) $: $ 30 - 1 = 29 $ \u2192 29 divides 30? No \u274c\n- $ (15,2) $: $ 13 $ \u2192 13 \u2224 30 \u274c\n- $ (10,3) $: $ 7 $ \u2192 7 \u2224 30 \u274c\n- $ (6,5) $: $ 1 $ \u2192 1 \u2223 30 \u2705\n\nOnly pair satisfying $ x - y \\mid 30 $ is $ (6,5) $\n\n\u2705 So only valid pair: $ x = 6, y = 5 $\n\n---\n\n### Step 3: Compute $ \\frac{m + n}{b} $\n\nRecall:\n- $ m = d x = 6d $\n- $ n = d y = 5d $\n- $ b = d $\n\nSo:\n$$\nm + n = 6d + 5d = 11d\n$$\n$$\n\\frac{m + n}{b} = \\frac{11d}{d} = 11\n$$\n\n---\n\n### Final Answer:\nAnswer: 11
    """
    ground_truth1 = "11"

    response2 = """
    We are given:\n\n- $ m > n $, both positive integers.\n- $ a = \\text{lcm}(m, n) $\n- $ b = \\gcd(m, n) $\n- $ a = 30b $\n- $ m - n \\mid a $\n- We are to find $ \\frac{m + n}{b} $\n\n---\n\n### Step 1: Use the identity between LCM and GCD\n\nWe know that for any two positive integers $ m $ and $ n $:\n\n$$\n\\text{lcm}(m, n) \\cdot \\gcd(m, n) = m \\cdot n\n$$\n\nGiven $ a = \\text{lcm}(m,n) = 30b $, so:\n\n$$\na \\cdot b = m n \\Rightarrow 30b \\cdot b = m n \\Rightarrow 30b^2 = m n\n$$\n\nSo:\n\n$$\nmn = 30b^2 \\quad \\text{(Equation 1)}\n$$\n\n---\n\n### Step 2: Express $ m $ and $ n $ in terms of $ b $\n\nLet:\n\n$$\nm = b \\cdot x, \\quad n = b \\cdot y\n$$\n\nSince $ b = \\gcd(m, n) $, we can write $ m = bx $, $ n = by $, where $ \\gcd(x, y) = 1 $, and $ x > y $ since $ m > n $.\n\nNow plug into Equation 1:\n\n$$\nmn = (bx)(by) = b^2xy = 30b^2\n\\Rightarrow xy = 30\n$$\n\nSo $ x $ and $ y $ are **coprime positive integers** with $ x > y $ and $ xy = 30 $.\n\n---\n\n### Step 3: List coprime pairs $ (x, y) $ such that $ xy = 30 $, $ x > y $\n\nFactor pairs of 30:\n\n- $ (30, 1) $ \u2192 gcd(30,1)=1 \u2705\n- $ (15, 2) $ \u2192 gcd(15,2)=1 \u2705\n- $ (10, 3) $ \u2192 gcd(10,3)=1 \u2705\n- $ (6, 5) $ \u2192 gcd(6,5)=1 \u2705\n\nCheck others:\n- $ (5,6) $ \u2192 $ x < y $ \u2192 skip\n- $ (2,15) $ \u2192 $ x < y $ \u2192 skip\n\nSo valid coprime pairs with $ x > y $:  \n\u2192 $ (30,1), (15,2), (10,3), (6,5) $\n\nNow for each pair, compute:\n\n- $ m = bx $\n- $ n = by $\n- $ m - n = b(x - y) $\n- $ a = 30b $ \u2014 given\n- Condition: $ m - n \\mid a $ \u2192 $ b(x - y) \\mid 30b $\n\nWe can divide both sides by $ b $ (since $ b > 0 $):\n\n$$\nx - y \\mid 30\n$$\n\nSo for each pair, check whether $ x - y $ divides 30.\n\nLet\u2019s evaluate each:\n\n---\n\n1. $ (x,y) = (30,1) $\n\n- $ x - y = 29 $\n- Does 29 divide 30? \u2192 No \u274c\n\n---\n\n2. $ (15,2) $\n\n- $ x - y = 13 $\n- 13 \u2223 30? \u2192 30 \u00f7 13 \u2248 2.3 \u2192 No \u274c\n\n---\n\n3. $ (10,3) $\n\n- $ x - y = 7 $\n- 7 \u2223 30? \u2192 30 \u00f7 7 \u2248 4.28 \u2192 No \u274c\n\n---\n\n4. $ (6,5) $\n\n- $ x - y = 1 $\n- 1 \u2223 30? \u2192 Yes \u2705\n\nOnly this pair satisfies all conditions.\n\n---\n\n### Step 4: Plug in $ x=6, y=5 $\n\nSo:\n\n- $ m = 6b $\n- $ n = 5b $\n- $ m - n = b $\n- $ a = 30b $\n- Check: $ m - n = b \\mid 30b $? Yes \u2192 $ b \\mid 30b $ always true\n\nCondition is satisfied.\n\nNow compute $ \\frac{m + n}{b} $:\n\n$$\n\\frac{m + n}{b} = \\frac{6b + 5b}{b} = \\frac{11b}{b} = 11\n$$\n\n---\n\n\u2705 All conditions are satisfied, only one valid pair.\n\nAnswer: 11
    """
    
    result1 = test_single_response(response1, ground_truth1)
