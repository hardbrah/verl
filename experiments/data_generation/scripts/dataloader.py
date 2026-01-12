#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAPO-MATH-17K 数据集加载器

功能：
1. 加载 DAPO-MATH-17K 数据集（本地 parquet 文件）
2. 使用固定种子随机采样 N 个问题
3. 支持增量采样：跳过之前已采样过的问题，继续采样新问题
4. 使用 tokenizer 对 chat 格式的 prompt 进行格式化
5. 输出格式化后的 prompts，供 data_gen_vllm.py 使用

输出：
- sampled_questions.jsonl: 采样的问题元数据（q_id, question, gt_answer, prompt）
- formatted_prompts.json: 格式化后的 prompt 字符串列表（直接供 vLLM 使用）
- sampled_history.json: 已采样索引历史（用于增量采样）
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Set

# 添加父目录到模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from configs.config import (
    RANDOM_SEED,
    PathConfig,
    DataLoaderConfig,
)


class SamplingHistory:
    """采样历史管理器，用于跟踪已采样的数据集索引"""
    
    def __init__(self, history_path: str):
        self.history_path = history_path
        self.history: Dict[str, Any] = {
            "seed": None,
            "total_sampled": 0,
            "sampled_indices": [],  # 所有已采样的数据集索引
            "batches": [],  # 每次采样的记录
        }
        self._load()
    
    def _load(self):
        """加载历史记录"""
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
            print(f"已加载采样历史: {self.history_path}")
            print(f"  - 种子: {self.history['seed']}")
            print(f"  - 已采样总数: {self.history['total_sampled']}")
            print(f"  - 采样批次数: {len(self.history['batches'])}")
    
    def save(self):
        """保存历史记录"""
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"采样历史已保存: {self.history_path}")
    
    def get_sampled_indices(self) -> Set[int]:
        """获取所有已采样的索引集合"""
        return set(self.history["sampled_indices"])
    
    def get_seed(self) -> Optional[int]:
        """获取历史记录中的种子"""
        return self.history["seed"]
    
    def add_batch(self, seed: int, indices: List[int], batch_info: Dict[str, Any] = None):
        """
        添加一批采样记录
        
        Args:
            seed: 使用的随机种子
            indices: 本次采样的数据集索引列表
            batch_info: 额外的批次信息
        """
        # 检查种子一致性
        if self.history["seed"] is None:
            self.history["seed"] = seed
        elif self.history["seed"] != seed:
            raise ValueError(
                f"种子不一致！历史记录种子: {self.history['seed']}, 当前种子: {seed}\n"
                f"如需使用新种子，请删除历史文件: {self.history_path}"
            )
        
        # 添加新索引
        existing = set(self.history["sampled_indices"])
        new_indices = [idx for idx in indices if idx not in existing]
        
        self.history["sampled_indices"].extend(new_indices)
        self.history["total_sampled"] = len(self.history["sampled_indices"])
        
        # 记录批次信息
        batch_record = {
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(indices),
            "new_indices_count": len(new_indices),
            "indices": indices,
        }
        if batch_info:
            batch_record.update(batch_info)
        self.history["batches"].append(batch_record)
        
        self.save()
        
        return new_indices
    
    def clear(self):
        """清空历史记录"""
        self.history = {
            "seed": None,
            "total_sampled": 0,
            "sampled_indices": [],
            "batches": [],
        }
        if os.path.exists(self.history_path):
            os.remove(self.history_path)
            print(f"已清空采样历史: {self.history_path}")


class DAPOMathDataLoader:
    """DAPO-MATH-17K 数据集加载器"""
    
    def __init__(
        self,
        model_path: str = PathConfig.MODEL_PATH,
        dataset_path: str = PathConfig.DATASET_LOCAL_PATH,
        system_prompt: str = DataLoaderConfig.SYSTEM_PROMPT,
        seed: int = RANDOM_SEED,
        history_path: str = DataLoaderConfig.SAMPLED_HISTORY_PATH,
        enable_incremental: bool = DataLoaderConfig.ENABLE_INCREMENTAL_SAMPLING,
    ):
        """
        初始化数据加载器
        
        Args:
            model_path: 模型路径（用于加载 tokenizer）
            dataset_path: 数据集本地路径（parquet 格式）
            system_prompt: 系统提示词
            seed: 随机种子
            history_path: 采样历史文件路径
            enable_incremental: 是否启用增量采样
        """
        self.model_path = model_path
        self.dataset_path = os.path.expanduser(dataset_path)
        self.system_prompt = system_prompt
        self.seed = seed
        self.enable_incremental = enable_incremental
        
        self.tokenizer = None
        self.dataset = None
        self.sampled_data = None
        
        # 采样历史管理
        self.history = SamplingHistory(history_path) if enable_incremental else None
        
    def load_tokenizer(self) -> AutoTokenizer:
        """加载 tokenizer"""
        if self.tokenizer is None:
            print(f"正在加载 tokenizer: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            print(f"Tokenizer 加载完成")
        return self.tokenizer
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载数据集"""
        if self.dataset is not None:
            return self.dataset
            
        print(f"正在加载数据集: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"数据集文件不存在: {self.dataset_path}\n"
                f"请确保已下载 DAPO-MATH-17K 数据集"
            )
        
        df = pd.read_parquet(self.dataset_path)
        self.dataset = df.to_dict('records')
        print(f"数据集加载成功，共 {len(self.dataset)} 条数据")
        
        return self.dataset
    
    def sample_questions(
        self,
        sample_size: Optional[int] = None,
        save_indices: bool = True,
        indices_save_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        从数据集中采样问题（支持增量采样）
        
        Args:
            sample_size: 采样数量，None 表示全部
            save_indices: 是否保存采样索引
            indices_save_path: 索引保存路径
            
        Returns:
            采样后的数据列表
        """
        if self.dataset is None:
            self.load_dataset()
        
        total_size = len(self.dataset)
        
        if sample_size is None or sample_size >= total_size:
            print(f"使用全部 {total_size} 条数据")
            indices = list(range(total_size))
            sample_size = total_size
        else:
            # 获取已采样的索引
            already_sampled = set()
            if self.enable_incremental and self.history:
                already_sampled = self.history.get_sampled_indices()
                
                # 检查种子一致性
                history_seed = self.history.get_seed()
                if history_seed is not None and history_seed != self.seed:
                    raise ValueError(
                        f"种子不一致！历史记录种子: {history_seed}, 当前种子: {self.seed}\n"
                        f"如需使用新种子，请删除历史文件或设置 ENABLE_INCREMENTAL_SAMPLING=False"
                    )
                
                if already_sampled:
                    print(f"增量采样模式：已采样 {len(already_sampled)} 条，将跳过这些问题")
            
            # 计算可用的索引
            available_indices = [i for i in range(total_size) if i not in already_sampled]
            
            if len(available_indices) < sample_size:
                print(f"警告：可用数据不足！需要 {sample_size} 条，但只剩 {len(available_indices)} 条")
                print(f"将使用全部剩余 {len(available_indices)} 条数据")
                sample_size = len(available_indices)
            
            if len(available_indices) == 0:
                raise ValueError(
                    f"没有可用的数据！全部 {total_size} 条数据都已被采样过。\n"
                    f"如需重新采样，请删除历史文件: {self.history.history_path}"
                )
            
            # 使用固定种子采样
            # 关键：使用种子生成一个确定性的随机排列，然后按顺序取
            np.random.seed(self.seed)
            
            # 生成全部数据的随机排列
            full_permutation = np.random.permutation(total_size).tolist()
            
            # 从排列中按顺序选取未被采样过的索引
            indices = []
            for idx in full_permutation:
                if idx not in already_sampled:
                    indices.append(idx)
                    if len(indices) >= sample_size:
                        break
            
            print(f"随机采样 {len(indices)} 条数据（种子: {self.seed}）")
            print(f"  - 数据集总量: {total_size}")
            print(f"  - 已采样: {len(already_sampled)}")
            print(f"  - 本次采样: {len(indices)}")
            print(f"  - 剩余可用: {len(available_indices) - len(indices)}")
            
            # 保存采样索引到历史
            if self.enable_incremental and self.history:
                self.history.add_batch(
                    seed=self.seed,
                    indices=indices,
                    batch_info={
                        "sample_size_requested": sample_size,
                        "available_before": len(available_indices),
                    }
                )
            
            # 保存本次采样索引
            if save_indices and indices_save_path:
                os.makedirs(os.path.dirname(indices_save_path), exist_ok=True)
                with open(indices_save_path, 'w') as f:
                    json.dump({
                        'seed': self.seed,
                        'sample_size': len(indices),
                        'total_size': total_size,
                        'already_sampled': len(already_sampled),
                        'indices': indices
                    }, f, indent=2)
                print(f"本次采样索引已保存: {indices_save_path}")
        
        # 处理采样数据
        self.sampled_data = []
        for q_id, idx in enumerate(indices):
            item = self.dataset[idx]
            
            # 提取 prompt（chat 格式的列表）
            prompt = item.get('prompt', [])
            if hasattr(prompt, 'tolist'):
                prompt = prompt.tolist()
            
            # 从 prompt 中提取问题文本
            question_text = ""
            if isinstance(prompt, list) and len(prompt) > 0:
                for msg in prompt:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        question_text = msg.get('content', '')
                        break
            
            # 获取 ground_truth
            reward_model = item.get('reward_model', {})
            gt_answer = reward_model.get('ground_truth', '')
            
            # 确保 gt_answer 是字符串
            if isinstance(gt_answer, np.ndarray):
                gt_answer = gt_answer.item() if gt_answer.size == 1 else str(gt_answer)
            elif not isinstance(gt_answer, str):
                gt_answer = str(gt_answer)
            
            # 构建标准 chat 格式的 prompt
            chat_prompt = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question_text}
            ]
            
            self.sampled_data.append({
                'q_id': q_id,
                'dataset_idx': idx,  # 记录原始数据集索引
                'question': question_text,
                'gt_answer': gt_answer,
                'prompt': chat_prompt,
            })
        
        return self.sampled_data
    
    def format_prompts(
        self,
        add_generation_prompt: bool = True,
    ) -> List[str]:
        """
        使用 tokenizer 格式化 prompts
        
        Args:
            add_generation_prompt: 是否添加生成提示
            
        Returns:
            格式化后的 prompt 字符串列表
        """
        if self.sampled_data is None:
            raise ValueError("请先调用 sample_questions() 采样数据")
        
        if self.tokenizer is None:
            self.load_tokenizer()
        
        print(f"正在格式化 {len(self.sampled_data)} 条 prompts...")
        
        formatted_prompts = []
        for item in self.sampled_data:
            chat_prompt = item['prompt']
            
            # 使用 tokenizer 的 apply_chat_template 格式化
            formatted = self.tokenizer.apply_chat_template(
                chat_prompt,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            formatted_prompts.append(formatted)
        
        print(f"格式化完成，共 {len(formatted_prompts)} 条")
        return formatted_prompts
    
    def save_sampled_questions(
        self,
        output_path: str = PathConfig.SAMPLED_QUESTIONS_JSONL,
    ) -> str:
        """
        保存采样的问题元数据为 JSONL 格式
        
        Args:
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        if self.sampled_data is None:
            raise ValueError("请先调用 sample_questions() 采样数据")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.sampled_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"问题元数据已保存: {output_path}")
        return output_path
    
    def save_formatted_prompts(
        self,
        formatted_prompts: List[str],
        output_path: str = PathConfig.FORMATTED_PROMPTS_JSON,
    ) -> str:
        """
        保存格式化后的 prompts 为 JSON 格式（供 vLLM 使用）
        
        Args:
            formatted_prompts: 格式化后的 prompt 列表
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_prompts, f, ensure_ascii=False, indent=2)
        
        print(f"格式化 prompts 已保存: {output_path}")
        return output_path
    
    def prepare_for_generation(
        self,
        sample_size: Optional[int] = DataLoaderConfig.SAMPLE_SIZE,
        add_generation_prompt: bool = DataLoaderConfig.ADD_GENERATION_PROMPT,
        questions_output_path: str = PathConfig.SAMPLED_QUESTIONS_JSONL,
        prompts_output_path: str = PathConfig.FORMATTED_PROMPTS_JSON,
    ) -> tuple:
        """
        完整的数据准备流程：加载 -> 采样 -> 格式化 -> 保存
        
        Args:
            sample_size: 采样数量
            add_generation_prompt: 是否添加生成提示
            questions_output_path: 问题元数据输出路径
            prompts_output_path: 格式化 prompts 输出路径
            
        Returns:
            (questions_path, prompts_path) 保存的文件路径元组
        """
        print("=" * 80)
        print("DAPO-MATH-17K 数据准备流程")
        if self.enable_incremental:
            print("模式: 增量采样（跳过已采样问题）")
        else:
            print("模式: 普通采样")
        print("=" * 80)
        
        # 1. 加载数据集
        self.load_dataset()
        
        # 2. 采样问题
        indices_path = questions_output_path.replace('.jsonl', '_indices.json')
        self.sample_questions(
            sample_size=sample_size,
            save_indices=True,
            indices_save_path=indices_path if sample_size else None,
        )
        
        # 3. 格式化 prompts
        formatted_prompts = self.format_prompts(
            add_generation_prompt=add_generation_prompt,
        )
        
        # 4. 保存结果
        questions_path = self.save_sampled_questions(questions_output_path)
        prompts_path = self.save_formatted_prompts(formatted_prompts, prompts_output_path)
        
        print("=" * 80)
        print("数据准备完成！")
        print(f"  - 问题元数据: {questions_path}")
        print(f"  - 格式化 prompts: {prompts_path}")
        print(f"  - 数据量: {len(formatted_prompts)} 条")
        if self.enable_incremental and self.history:
            print(f"  - 累计已采样: {self.history.history['total_sampled']} 条")
        print("=" * 80)
        
        # 显示示例
        if self.sampled_data:
            print("\n示例数据（第1条）：")
            print("-" * 40)
            print(f"Dataset Index: {self.sampled_data[0]['dataset_idx']}")
            print(f"Question: {self.sampled_data[0]['question'][:200]}...")
            print(f"GT Answer: {self.sampled_data[0]['gt_answer']}")
            print(f"\nFormatted Prompt:\n{formatted_prompts[0][:500]}...")
        
        return questions_path, prompts_path
    
    def clear_history(self):
        """清空采样历史（重新开始采样）"""
        if self.history:
            self.history.clear()
            print("采样历史已清空，下次将从头开始采样")
        else:
            print("增量采样未启用，无需清空历史")


def main():
    """主函数：准备数据供 vLLM 生成使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DAPO-MATH-17K 数据加载器")
    parser.add_argument("--clear-history", action="store_true",
                        help="清空采样历史，重新开始采样")
    parser.add_argument("--no-incremental", action="store_true",
                        help="禁用增量采样（不跳过已采样问题）")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="采样数量（覆盖配置文件）")
    args = parser.parse_args()
    
    # 创建数据加载器
    loader = DAPOMathDataLoader(
        model_path=PathConfig.MODEL_PATH,
        dataset_path=PathConfig.DATASET_LOCAL_PATH,
        system_prompt=DataLoaderConfig.SYSTEM_PROMPT,
        seed=RANDOM_SEED,
        history_path=DataLoaderConfig.SAMPLED_HISTORY_PATH,
        enable_incremental=not args.no_incremental and DataLoaderConfig.ENABLE_INCREMENTAL_SAMPLING,
    )
    
    # 处理清空历史
    if args.clear_history:
        loader.clear_history()
        print("已清空历史，继续执行采样...")
    
    # 确定采样数量
    sample_size = args.sample_size if args.sample_size else DataLoaderConfig.SAMPLE_SIZE
    
    # 执行完整的数据准备流程
    questions_path, prompts_path = loader.prepare_for_generation(
        sample_size=sample_size,
        add_generation_prompt=DataLoaderConfig.ADD_GENERATION_PROMPT,
        questions_output_path=PathConfig.SAMPLED_QUESTIONS_JSONL,
        prompts_output_path=PathConfig.FORMATTED_PROMPTS_JSON,
    )
    
    print(f"\n下一步：运行 data_gen_vllm.py 进行生成")
    print(f"  python scripts/data_gen_vllm.py")


if __name__ == "__main__":
    main()
