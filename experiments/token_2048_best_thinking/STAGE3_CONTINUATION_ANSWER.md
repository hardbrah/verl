# Stage3 续写提示词构造指南

## 📋 问题回答

**问题**：这些都是模型被中断的思考过程，我想要模型接着这个思维过程思考，该怎样构造提示词？直接 `tokenizer.apply_chat_template` 就行了吗？

**回答**：可以用 `tokenizer.apply_chat_template`，但要注意正确的用法！

---

## ✅ 正确答案

### 你的数据格式

你的 `stage3_temp_input.parquet` 中，`question` 列已经是**完整的对话列表格式**：

```python
[
  {
    "role": "user",
    "content": "Solve the following math problem..."
  },
  {
    "role": "assistant",
    "content": "We are given that... [未完成的推理过程]"
  }
]
```

### 正确的处理方法

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# ✅ 正确：直接传入对话列表
for conversation in df['question']:
    formatted_text = tokenizer.apply_chat_template(
        conversation,              # 直接传入，不要再包装！
        tokenize=False,
        add_generation_prompt=True  # 关键参数！
    )
```

### 关键点说明

1. **你的数据已经是对话格式**
   - 不需要再构造 `[{"role": "user", "content": ...}]`
   - 直接传入整个 `conversation` 列表

2. **`add_generation_prompt=True` 是必须的**
   - 这个参数告诉 tokenizer 要添加继续生成的提示符
   - 对于 Qwen 模型，会添加 `<|im_start|>assistant` 但不添加 `<|im_end|>`
   - 模型会自然地从最后的 assistant 回答继续生成

3. **效果**
   - 模型会看到完整的对话历史
   - 模型会从最后的 assistant 回答处继续思考
   - 就像人类接着之前的思路继续推理

---

## ❌ 常见错误

### 错误 1：重复包装

```python
# ❌ 错误！会导致嵌套
formatted_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": conversation}],  # conversation 已经是列表了！
    tokenize=False,
    add_generation_prompt=True
)
```

**问题**：`conversation` 本身就是一个对话列表，再包装一层会导致格式错误。

### 错误 2：忘记 `add_generation_prompt=True`

```python
# ❌ 错误！模型不知道要继续生成
formatted_text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=False  # 或者不设置（默认 False）
)
```

**问题**：没有生成提示符，模型会认为对话已经结束，不会继续生成。

### 错误 3：只传入 assistant 的内容

```python
# ❌ 错误！丢失了用户问题
last_response = conversation[-1]['content']
formatted_text = tokenizer.apply_chat_template(
    [{"role": "assistant", "content": last_response}],
    tokenize=False,
    add_generation_prompt=True
)
```

**问题**：模型需要看到完整的对话上下文，包括用户的问题。

---

## 🔧 验证你的代码

运行验证脚本检查格式：

```bash
python scripts/check_continuation_format.py
```

你应该看到：
- ✅ 数据是列表格式
- ✅ 包含 2 轮对话（user + assistant）
- ✅ 最后一轮是 assistant（未完成）

---

## 📝 完整示例代码

```python
#!/usr/bin/env python3
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 1. 读取数据
df = pd.read_parquet('outputs/stage3_temp_input.parquet')

# 2. 加载 tokenizer
model_path = "/datacenter/models/Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 3. 格式化提示词
formatted_prompts = []
for conversation in df['question']:
    # conversation 是完整的对话列表：
    # [
    #   {"role": "user", "content": "..."},
    #   {"role": "assistant", "content": "未完成的回答..."}
    # ]
    
    # 直接传入，不要再包装
    text = tokenizer.apply_chat_template(
        conversation,              # 直接使用整个对话
        tokenize=False,
        add_generation_prompt=True  # 添加继续生成的提示
    )
    formatted_prompts.append(text)

# 4. 使用 vLLM 生成
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=32768
)

sampling_params = SamplingParams(
    n=8,                # 每个问题生成 8 次
    temperature=1.0,
    top_p=0.95,
    max_tokens=2048
)

# 5. 执行生成
outputs = llm.generate(formatted_prompts, sampling_params)

# 6. 提取结果
for idx, output in enumerate(outputs):
    continuations = [o.text for o in output.outputs]
    print(f"样本 {idx}: 生成了 {len(continuations)} 个续写")
```

---

## 🎯 核心要点总结

| 问题 | 答案 |
|------|------|
| 需要重新构造对话格式吗？ | ❌ 不需要，数据已经是对话格式 |
| 可以直接用 `apply_chat_template` 吗？ | ✅ 可以，直接传入对话列表 |
| 需要 `add_generation_prompt=True` 吗？ | ✅ 必须！否则模型不会继续生成 |
| 是否需要特殊处理？ | ❌ 不需要，标准流程即可 |

---

## 📚 相关文件

- **验证脚本**: `scripts/check_continuation_format.py`
- **生成脚本**: `scripts/simple_generate.py` (已更新 `generate_responses_stage3`)
- **详细指南**: `CONTINUATION_GUIDE.md`

---

## 🚀 快速开始

```bash
# 1. 验证数据格式
python scripts/check_continuation_format.py

# 2. 运行生成（使用更新后的 stage3 函数）
python scripts/simple_generate.py \
    --model_path /datacenter/models/Qwen/Qwen3-4B-Instruct-2507 \
    --input outputs/stage3_temp_input.parquet \
    --output outputs/stage3_output.parquet \
    --n_samples 8 \
    --max_new_tokens 2048
```

搞定！🎉

