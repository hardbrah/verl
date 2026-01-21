#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRM评估脚本 - GSM8K数据集版本

阶段1 (extract): 使用8卡并行提取latent states
阶段2 (score): 对latent states打分并计算准确率

特点：
- 使用gsm8k.py中的工具提取答案并验证正确性
- 选择数量相同的正负样本进行平衡评估
- 截取response前64个token，拼接question
- 提取第15层latent states送入PRM打分

使用方法:
# 阶段1: 提取latent states (8卡)
torchrun --nproc_per_node=8 eval_prm_gsm8k.py --stage extract --batch_size 128

# 阶段2: 打分和评估 (8卡)
torchrun --nproc_per_node=8 eval_prm_gsm8k.py --stage score --batch_size 64
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from glob import glob

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 添加项目路径
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

# 导入gsm8k验证工具
import importlib.util
gsm8k_path = os.path.join(project_root, "verl", "utils", "reward_score", "gsm8k.py")
spec = importlib.util.spec_from_file_location("gsm8k", gsm8k_path)
gsm8k_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gsm8k_module)
gsm8k_compute_score = gsm8k_module.compute_score
gsm8k_extract_solution = gsm8k_module.extract_solution


# ==================== 配置 ====================

@dataclass
class Config:
    # 数据集路径
    data_path: str = "/data/chenhaotian/verl/experiments/data_generation/outputs/gsm8k/rollouts/rollout_20260117_110227/all_responses_rollout_20260117_110227.jsonl"
    
    # 模型路径
    model_path: str = "/data/models/Qwen/Qwen3-4B-Instruct-2507"
    
    # PRM checkpoint路径
    prm_checkpoint_path: str = "/data/chenhaotian/verl/checkpoints/latentqa/math_shepherd_20260116/best_model.pt"
    
    # 输出目录
    output_dir: str = "/data/chenhaotian/verl/experiments/data_generation/outputs/prm_eval_gsm8k"
    
    # Latent states提取配置
    extract_layer: int = 15  # 提取第15层的latent states
    response_max_tokens: int = 64  # response最多取64个token
    
    # PRM阈值
    prm_threshold: float = 0.5
    
    # 数据类型
    dtype: torch.dtype = torch.bfloat16
    
    # 随机种子
    seed: int = 42


# ==================== 数据处理 ====================

def load_and_balance_data(data_path: str, seed: int = 42) -> Tuple[List[Dict], int, int]:
    """加载数据并进行正负样本平衡"""
    print(f"Loading data from {data_path}")
    
    with open(data_path, "r") as f:
        all_data = [json.loads(line) for line in f]
    
    print(f"Total samples: {len(all_data)}")
    
    # 使用gsm8k验证工具分类正负样本
    positive_samples = []
    negative_samples = []
    
    for item in tqdm(all_data, desc="Classifying samples"):
        score = gsm8k_compute_score(
            solution_str=item["response"],
            ground_truth=item["gt_answer"],
            method="strict"
        )
        item["is_correct"] = (score > 0)
        
        if item["is_correct"]:
            positive_samples.append(item)
        else:
            negative_samples.append(item)
    
    print(f"Positive (correct) samples: {len(positive_samples)}")
    print(f"Negative (incorrect) samples: {len(negative_samples)}")
    
    # 选择数量相同的正负样本
    min_count = min(len(positive_samples), len(negative_samples))
    print(f"Balancing to {min_count} samples each")
    
    random.seed(seed)
    if len(positive_samples) > min_count:
        positive_samples = random.sample(positive_samples, min_count)
    if len(negative_samples) > min_count:
        negative_samples = random.sample(negative_samples, min_count)
    
    # 合并并打乱
    balanced_data = positive_samples + negative_samples
    random.shuffle(balanced_data)
    
    print(f"Final balanced dataset: {len(balanced_data)} samples ({min_count} positive, {min_count} negative)")
    
    return balanced_data, min_count, min_count


# ==================== 数据集 ====================

class GSM8KDataset(Dataset):
    """GSM8K数据集"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        response_max_tokens: int = 64,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.response_max_tokens = response_max_tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        question = item["question"]
        response = item["response"]
        gt_answer = item["gt_answer"]
        
        # 只使用 question + 被截断的response（不包含system message或chat格式token）
        # Tokenize question部分
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        
        # Tokenize response部分，取前response_max_tokens个token
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        response_truncated_tokens = response_tokens[:self.response_max_tokens]
        
        # 拼接：question + response（与训练时一致）
        input_ids = question_tokens + response_truncated_tokens
        
        return {
            "idx": idx,
            "input_ids": input_ids,
            "question": question,
            "response": response,
            "gt_answer": gt_answer,
            "q_id": item.get("q_id", idx),
            "sample_idx": item.get("sample_idx", 0),
            "is_correct": item.get("is_correct", False),
        }


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict:
    """Collate function for DataLoader"""
    max_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids_padded = []
    attention_mask = []
    
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        # 右padding（与训练时一致）
        padded_ids = ids + [tokenizer.pad_token_id] * pad_len
        mask = [1] * len(ids) + [0] * pad_len
        
        input_ids_padded.append(padded_ids)
        attention_mask.append(mask)
    
    return {
        "idx": [item["idx"] for item in batch],
        "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "question": [item["question"] for item in batch],
        "response": [item["response"] for item in batch],
        "gt_answer": [item["gt_answer"] for item in batch],
        "q_id": [item["q_id"] for item in batch],
        "sample_idx": [item["sample_idx"] for item in batch],
        "is_correct": [item["is_correct"] for item in batch],
    }


# ==================== 模型 ====================

class RegressionHead(nn.Module):
    """PRM Regression Head"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return torch.sigmoid(x)


class PRMBackbone(nn.Module):
    """PRM Backbone - 接收latent states作为inputs_embeds"""
    
    def __init__(
        self,
        model_path: str,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        
        if hasattr(self.backbone, 'lm_head'):
            del self.backbone.lm_head
        
        self.regression_head = RegressionHead(hidden_size).to(dtype)
        self.hidden_size = hidden_size
        self.dtype = dtype
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # 使用最后一个有效token的hidden state（与训练时一致）
        batch_size = hidden_states.shape[0]
        seq_lengths = attention_mask.sum(dim=-1) - 1  # [batch]
        seq_lengths = seq_lengths.clamp(min=0)
        
        # 提取最后一个有效token的 hidden states
        pooled = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            seq_lengths
        ]
        
        scores = self.regression_head(pooled)
        return scores.squeeze(-1)


# ==================== 阶段1: 提取Latent States ====================

def extract_latent_states(config: Config, batch_size: int = 128):
    """阶段1: 提取latent states (分布式)"""
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"=== 阶段1: 提取Latent States (GSM8K) ===")
        print(f"World size: {world_size}")
        print(f"Model: {config.model_path}")
        print(f"Extract layer: {config.extract_layer}")
        print(f"Response max tokens: {config.response_max_tokens}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Rank 0 加载并平衡数据，然后广播
    if rank == 0:
        balanced_data, pos_count, neg_count = load_and_balance_data(config.data_path, config.seed)
        # 保存平衡后的数据供其他rank使用
        os.makedirs(config.output_dir, exist_ok=True)
        balanced_path = os.path.join(config.output_dir, "balanced_data.json")
        with open(balanced_path, "w") as f:
            json.dump(balanced_data, f)
        print(f"Saved balanced data to {balanced_path}")
    
    dist.barrier(device_ids=[local_rank])
    
    # 所有rank读取平衡后的数据
    balanced_path = os.path.join(config.output_dir, "balanced_data.json")
    with open(balanced_path, "r") as f:
        balanced_data = json.load(f)
    
    if rank == 0:
        print(f"All ranks loaded {len(balanced_data)} samples")
    
    # 创建数据集
    dataset = GSM8KDataset(
        data=balanced_data,
        tokenizer=tokenizer,
        response_max_tokens=config.response_max_tokens,
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda x: collate_fn(x, tokenizer),
        num_workers=4,
        pin_memory=True,
    )
    
    if rank == 0:
        print(f"Loading model from {config.model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=config.dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)
    model.eval()
    
    results = []
    extract_layer_idx = config.extract_layer + 1
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"[Rank {rank}] Extracting", disable=(rank != 0)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            
            latent_states = outputs.hidden_states[extract_layer_idx]
            
            for i in range(len(batch["idx"])):
                idx = batch["idx"][i]
                seq_len = int(attention_mask[i].sum().item())
                
                # 右padding: 有效token在前面，所以用 :seq_len
                latent = latent_states[i, :seq_len, :].cpu()
                valid_input_ids = input_ids[i, :seq_len].cpu().tolist()
                decoded_text = tokenizer.decode(valid_input_ids, skip_special_tokens=False)
                
                results.append({
                    "idx": idx,
                    "q_id": batch["q_id"][i],
                    "sample_idx": batch["sample_idx"][i],
                    "input_ids": valid_input_ids,
                    "decoded_text": decoded_text,
                    "latent_states": latent,
                    "question": batch["question"][i],
                    "response": batch["response"][i],
                    "gt_answer": batch["gt_answer"][i],
                    "is_correct": batch["is_correct"][i],
                })
    
    # 释放GPU显存
    del model
    torch.cuda.empty_cache()
    
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, f"latent_states_rank{rank}.pt")
    torch.save(results, output_path)
    
    if rank == 0:
        print(f"Saved {len(results)} samples to {output_path}")
    
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    
    # Rank 0 合并结果
    if rank == 0:
        print("Merging results from all ranks...")
        all_results = []
        for r in range(world_size):
            path = os.path.join(config.output_dir, f"latent_states_rank{r}.pt")
            data = torch.load(path, weights_only=False)
            all_results.extend(data)
        
        all_results.sort(key=lambda x: x["idx"])
        
        merged_path = os.path.join(config.output_dir, "latent_states_merged.pt")
        torch.save(all_results, merged_path)
        print(f"Merged {len(all_results)} samples to {merged_path}")
        
        for r in range(world_size):
            path = os.path.join(config.output_dir, f"latent_states_rank{r}.pt")
            os.remove(path)
        print("Cleaned up temporary files")
    
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


# ==================== 阶段2: 打分和评估 ====================

def score_and_evaluate_distributed(config: Config, batch_size: int = 64):
    """阶段2: 分布式打分和评估"""
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"=== 阶段2: 分布式打分和评估 (GSM8K) ===")
        print(f"World size: {world_size}")
    
    # 加载PRM checkpoint
    if rank == 0:
        print(f"Loading PRM checkpoint from {config.prm_checkpoint_path}")
    checkpoint = torch.load(config.prm_checkpoint_path, map_location="cpu", weights_only=False)
    
    model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
    hidden_size = model_config.hidden_size
    if rank == 0:
        print(f"Hidden size: {hidden_size}")
    
    if rank == 0:
        print(f"Creating PRM backbone...")
    prm_backbone = PRMBackbone(
        model_path=config.model_path,
        hidden_size=hidden_size,
        dtype=config.dtype,
    )
    
    state_dict = checkpoint["model_state_dict"]
    
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            new_key = key[len("backbone."):]
            backbone_state[new_key] = value
    
    if backbone_state:
        missing, unexpected = prm_backbone.backbone.load_state_dict(backbone_state, strict=False)
        if rank == 0:
            print(f"Loaded backbone weights: {len(backbone_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    
    regression_head_state = {}
    for key, value in state_dict.items():
        if key.startswith("regression_head."):
            new_key = key[len("regression_head."):]
            regression_head_state[new_key] = value
    
    if regression_head_state:
        prm_backbone.regression_head.load_state_dict(regression_head_state)
        if rank == 0:
            print(f"Loaded regression head weights: {list(regression_head_state.keys())}")
    
    prm_backbone = prm_backbone.to(device)
    prm_backbone.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载latent states
    merged_path = os.path.join(config.output_dir, "latent_states_merged.pt")
    if rank == 0:
        print(f"Loading merged latent states from {merged_path}")
    all_data = torch.load(merged_path, map_location="cpu", weights_only=False)
    if rank == 0:
        print(f"Loaded {len(all_data)} samples")
    
    # 分配给各rank
    my_data = [all_data[i] for i in range(len(all_data)) if i % world_size == rank]
    if rank == 0:
        print(f"Each rank processes ~{len(my_data)} samples")
    
    local_results = []
    
    for i in tqdm(range(0, len(my_data), batch_size), desc=f"[Rank {rank}] Scoring", disable=(rank != 0)):
        batch_items = my_data[i:i+batch_size]
        
        max_len = max(item["latent_states"].shape[0] for item in batch_items)
        
        latent_list = []
        mask_list = []
        
        for item in batch_items:
            latent = item["latent_states"]
            seq_len = latent.shape[0]
            pad_len = max_len - seq_len
            
            if pad_len > 0:
                pad_tensor = torch.zeros(pad_len, hidden_size, dtype=latent.dtype)
                # 右padding（与训练时一致）
                latent_padded = torch.cat([latent, pad_tensor], dim=0)
            else:
                latent_padded = latent
            
            # 右padding: 有效token在前，padding在后
            mask = [1] * seq_len + [0] * pad_len
            latent_list.append(latent_padded)
            mask_list.append(mask)
        
        batch_latent = torch.stack(latent_list).to(device)
        batch_mask = torch.tensor(mask_list, dtype=torch.long).to(device)
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=config.dtype):
            scores = prm_backbone(batch_latent, batch_mask)
        
        scores_list = scores.cpu().tolist()
        
        for j, item in enumerate(batch_items):
            local_results.append({
                "idx": item["idx"],
                "q_id": item["q_id"],
                "sample_idx": item["sample_idx"],
                "decoded_text": item["decoded_text"],
                "prm_score": scores_list[j],
                "question": item["question"],
                "response": item["response"],
                "gt_answer": item["gt_answer"],
                "is_correct": item["is_correct"],
            })
    
    # 保存本地结果
    local_output = os.path.join(config.output_dir, f"results_rank{rank}.pt")
    torch.save(local_results, local_output)
    print(f"[Rank {rank}] Saved {len(local_results)} results to {local_output}")
    
    del prm_backbone
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    dist.barrier(device_ids=[local_rank])
    
    # Rank 0 合并结果并计算指标
    if rank == 0:
        print("\nMerging results from all ranks...")
        all_results = []
        for r in range(world_size):
            path = os.path.join(config.output_dir, f"results_rank{r}.pt")
            data = torch.load(path, weights_only=False)
            all_results.extend(data)
            os.remove(path)
        
        # 按idx排序
        all_results.sort(key=lambda x: x["idx"])
        
        print(f"Total samples: {len(all_results)}")
        evaluate_and_save_results(all_results, config)
    
    dist.barrier(device_ids=[local_rank])
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate_and_save_results(results: List[Dict], config: Config):
    """评估结果并保存"""
    print("\n=== 评估结果 (GSM8K) ===")
    
    total = len(results)
    if total == 0:
        print("错误: 没有结果可以评估!")
        return
    
    threshold = config.prm_threshold
    
    # 使用is_correct作为ground truth（已经通过gsm8k验证）
    prm_predictions = [1 if item["prm_score"] >= threshold else 0 for item in results]
    ground_truths = [1 if item["is_correct"] else 0 for item in results]
    
    correct_predictions = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == g)
    accuracy = correct_predictions / total
    
    tp = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == 0 and g == 0)
    fn = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == 0 and g == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    prm_scores = [item["prm_score"] for item in results]
    prm_scores_correct = [item["prm_score"] for item in results if item["is_correct"]]
    prm_scores_wrong = [item["prm_score"] for item in results if not item["is_correct"]]
    
    print(f"数据集: GSM8K")
    print(f"阈值: {threshold}")
    print(f"总样本数: {total}")
    print(f"正确答案数: {sum(ground_truths)} ({sum(ground_truths)/total*100:.2f}%)")
    print(f"错误答案数: {total - sum(ground_truths)} ({(total-sum(ground_truths))/total*100:.2f}%)")
    print()
    print(f"PRM预测准确率: {accuracy:.4f} ({correct_predictions}/{total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print()
    print(f"混淆矩阵:")
    print(f"  TP (预测正确, 实际正确): {tp}")
    print(f"  FP (预测正确, 实际错误): {fp}")
    print(f"  TN (预测错误, 实际错误): {tn}")
    print(f"  FN (预测错误, 实际正确): {fn}")
    print()
    print(f"PRM分数统计:")
    print(f"  总体: mean={sum(prm_scores)/len(prm_scores):.4f}, min={min(prm_scores):.4f}, max={max(prm_scores):.4f}")
    if prm_scores_correct:
        print(f"  正确答案: mean={sum(prm_scores_correct)/len(prm_scores_correct):.4f}, min={min(prm_scores_correct):.4f}, max={max(prm_scores_correct):.4f}")
    if prm_scores_wrong:
        print(f"  错误答案: mean={sum(prm_scores_wrong)/len(prm_scores_wrong):.4f}, min={min(prm_scores_wrong):.4f}, max={max(prm_scores_wrong):.4f}")
    
    # 计算分数差异
    if prm_scores_correct and prm_scores_wrong:
        score_gap = sum(prm_scores_correct)/len(prm_scores_correct) - sum(prm_scores_wrong)/len(prm_scores_wrong)
        print(f"  正负样本分数差: {score_gap:.4f}")
    
    # 保存详细结果
    output_jsonl = os.path.join(config.output_dir, "prm_eval_results.jsonl")
    with open(output_jsonl, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n详细结果已保存到: {output_jsonl}")
    
    # 保存摘要
    summary = {
        "dataset": "gsm8k",
        "threshold": threshold,
        "total_samples": total,
        "correct_answers": sum(ground_truths),
        "wrong_answers": total - sum(ground_truths),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "prm_score_mean": sum(prm_scores) / len(prm_scores),
        "prm_score_min": min(prm_scores),
        "prm_score_max": max(prm_scores),
        "prm_score_correct_mean": sum(prm_scores_correct) / len(prm_scores_correct) if prm_scores_correct else None,
        "prm_score_wrong_mean": sum(prm_scores_wrong) / len(prm_scores_wrong) if prm_scores_wrong else None,
    }
    
    summary_path = os.path.join(config.output_dir, "prm_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"摘要已保存到: {summary_path}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="PRM评估脚本 - GSM8K版本")
    parser.add_argument("--stage", type=str, choices=["extract", "score", "all"], required=True,
                        help="运行阶段: extract=提取latent states, score=打分评估, all=两个阶段")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--data_path", type=str, default=None, help="数据集路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--prm_checkpoint", type=str, default=None, help="PRM checkpoint路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--response_max_tokens", type=int, default=64, help="Response最大token数")
    parser.add_argument("--threshold", type=float, default=0.5, help="PRM阈值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.data_path:
        config.data_path = args.data_path
    if args.model_path:
        config.model_path = args.model_path
    if args.prm_checkpoint:
        config.prm_checkpoint_path = args.prm_checkpoint
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.response_max_tokens:
        config.response_max_tokens = args.response_max_tokens
    if args.threshold:
        config.prm_threshold = args.threshold
    if args.seed:
        config.seed = args.seed
    
    if args.stage == "extract":
        extract_latent_states(config, batch_size=args.batch_size)
    elif args.stage == "score":
        score_and_evaluate_distributed(config, batch_size=args.batch_size)
    elif args.stage == "all":
        print("请分两步运行:")
        print(f"1. torchrun --nproc_per_node=8 {__file__} --stage extract --batch_size {args.batch_size}")
        print(f"2. torchrun --nproc_per_node=8 {__file__} --stage score --batch_size {args.batch_size}")


if __name__ == "__main__":
    main()
