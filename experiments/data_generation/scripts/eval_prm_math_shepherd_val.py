#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRM评估脚本 - Math Shepherd验证集版本

使用训练时的验证集评估PRM效果。

数据集字段:
- response: 问题 + Step 1 的思维链（作为输入）
- step1_label: +/- 标签（+ 表示正确，- 表示错误）

使用方法:
# 阶段1: 提取latent states (8卡)
torchrun --nproc_per_node=8 eval_prm_math_shepherd_val.py --stage extract --batch_size 128

# 阶段2: 打分和评估 (8卡)
torchrun --nproc_per_node=8 eval_prm_math_shepherd_val.py --stage score --batch_size 64
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ==================== 配置 ====================

@dataclass
class Config:
    # 数据集路径
    data_path: str = "/data/chenhaotian/latentqa/data/math_shepherd_step1_fullkeys/val_balanced.jsonl"
    
    # 模型路径
    model_path: str = "/data/models/Qwen/Qwen3-4B-Instruct-2507"
    
    # PRM checkpoint路径
    prm_checkpoint_path: str = "/data/chenhaotian/verl/checkpoints/latentqa/math_shepherd_20260116/best_model.pt"
    
    # 输出目录
    output_dir: str = "/data/chenhaotian/verl/experiments/data_generation/outputs/prm_eval_math_shepherd_val"
    
    # Latent states提取配置
    extract_layer: int = 15
    
    # PRM阈值
    prm_threshold: float = 0.5
    
    # 数据类型
    dtype: torch.dtype = torch.bfloat16


# ==================== 数据集 ====================

class MathShepherdValDataset(Dataset):
    """Math Shepherd验证集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
    ):
        self.tokenizer = tokenizer
        
        print(f"Loading data from {data_path}")
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.data)} samples")
        
        # 统计正负样本
        pos_count = sum(1 for item in self.data if item["step1_label"] == "+")
        neg_count = len(self.data) - pos_count
        print(f"Positive samples: {pos_count}, Negative samples: {neg_count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # response字段已经包含了问题和Step 1的思维链
        response = item["response"]
        step1_label = item["step1_label"]
        is_correct = (step1_label == "+")
        
        # Tokenize
        input_ids = self.tokenizer.encode(response, add_special_tokens=False)
        
        return {
            "idx": idx,
            "input_ids": input_ids,
            "response": response,
            "step1_label": step1_label,
            "is_correct": is_correct,
            "task": item.get("task", ""),
            "id": item.get("id", idx),
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
        "response": [item["response"] for item in batch],
        "step1_label": [item["step1_label"] for item in batch],
        "is_correct": [item["is_correct"] for item in batch],
        "task": [item["task"] for item in batch],
        "id": [item["id"] for item in batch],
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
    """PRM Backbone"""
    
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
    """阶段1: 提取latent states"""
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"=== 阶段1: 提取Latent States (Math Shepherd Val) ===")
        print(f"World size: {world_size}")
        print(f"Model: {config.model_path}")
        print(f"Extract layer: {config.extract_layer}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = MathShepherdValDataset(
        data_path=config.data_path,
        tokenizer=tokenizer,
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
                    "input_ids": valid_input_ids,
                    "decoded_text": decoded_text,
                    "latent_states": latent,
                    "response": batch["response"][i],
                    "step1_label": batch["step1_label"][i],
                    "is_correct": batch["is_correct"][i],
                    "task": batch["task"][i],
                    "id": batch["id"][i],
                })
    
    del model
    torch.cuda.empty_cache()
    
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, f"latent_states_rank{rank}.pt")
    torch.save(results, output_path)
    
    if rank == 0:
        print(f"Saved {len(results)} samples to {output_path}")
    
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    
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
        print(f"=== 阶段2: 分布式打分和评估 (Math Shepherd Val) ===")
        print(f"World size: {world_size}")
    
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
    
    merged_path = os.path.join(config.output_dir, "latent_states_merged.pt")
    if rank == 0:
        print(f"Loading merged latent states from {merged_path}")
    all_data = torch.load(merged_path, map_location="cpu", weights_only=False)
    if rank == 0:
        print(f"Loaded {len(all_data)} samples")
    
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
                "decoded_text": item["decoded_text"],
                "prm_score": scores_list[j],
                "response": item["response"],
                "step1_label": item["step1_label"],
                "is_correct": item["is_correct"],
                "task": item["task"],
                "id": item["id"],
            })
    
    local_output = os.path.join(config.output_dir, f"results_rank{rank}.pt")
    torch.save(local_results, local_output)
    print(f"[Rank {rank}] Saved {len(local_results)} results to {local_output}")
    
    del prm_backbone
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    dist.barrier(device_ids=[local_rank])
    
    if rank == 0:
        print("\nMerging results from all ranks...")
        all_results = []
        for r in range(world_size):
            path = os.path.join(config.output_dir, f"results_rank{r}.pt")
            data = torch.load(path, weights_only=False)
            all_results.extend(data)
            os.remove(path)
        
        all_results.sort(key=lambda x: x["idx"])
        
        print(f"Total samples: {len(all_results)}")
        evaluate_and_save_results(all_results, config)
    
    dist.barrier(device_ids=[local_rank])
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate_and_save_results(results: List[Dict], config: Config):
    """评估结果并保存"""
    print("\n=== 评估结果 (Math Shepherd Validation) ===")
    
    total = len(results)
    if total == 0:
        print("错误: 没有结果可以评估!")
        return
    
    threshold = config.prm_threshold
    
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
    
    # 按任务类型统计
    math_results = [r for r in results if r["task"] == "MATH"]
    gsm8k_results = [r for r in results if r["task"] == "GSM8K"]
    
    print(f"数据集: Math Shepherd Validation")
    print(f"阈值: {threshold}")
    print(f"总样本数: {total}")
    print(f"  - MATH: {len(math_results)}")
    print(f"  - GSM8K: {len(gsm8k_results)}")
    print(f"正确标签数 (+): {sum(ground_truths)} ({sum(ground_truths)/total*100:.2f}%)")
    print(f"错误标签数 (-): {total - sum(ground_truths)} ({(total-sum(ground_truths))/total*100:.2f}%)")
    print()
    print(f"=== 整体指标 ===")
    print(f"PRM预测准确率: {accuracy:.4f} ({correct_predictions}/{total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print()
    print(f"混淆矩阵:")
    print(f"  TP (预测+, 实际+): {tp}")
    print(f"  FP (预测+, 实际-): {fp}")
    print(f"  TN (预测-, 实际-): {tn}")
    print(f"  FN (预测-, 实际+): {fn}")
    print()
    print(f"PRM分数统计:")
    print(f"  总体: mean={sum(prm_scores)/len(prm_scores):.4f}, min={min(prm_scores):.4f}, max={max(prm_scores):.4f}")
    if prm_scores_correct:
        print(f"  正确(+): mean={sum(prm_scores_correct)/len(prm_scores_correct):.4f}, min={min(prm_scores_correct):.4f}, max={max(prm_scores_correct):.4f}")
    if prm_scores_wrong:
        print(f"  错误(-): mean={sum(prm_scores_wrong)/len(prm_scores_wrong):.4f}, min={min(prm_scores_wrong):.4f}, max={max(prm_scores_wrong):.4f}")
    
    if prm_scores_correct and prm_scores_wrong:
        score_gap = sum(prm_scores_correct)/len(prm_scores_correct) - sum(prm_scores_wrong)/len(prm_scores_wrong)
        print(f"  正负样本分数差: {score_gap:.4f}")
    
    # 按任务类型计算准确率
    if math_results:
        math_correct = sum(1 for r in math_results if (r["prm_score"] >= threshold) == r["is_correct"])
        print(f"\n=== MATH任务 ===")
        print(f"  样本数: {len(math_results)}")
        print(f"  准确率: {math_correct/len(math_results):.4f}")
    
    if gsm8k_results:
        gsm8k_correct = sum(1 for r in gsm8k_results if (r["prm_score"] >= threshold) == r["is_correct"])
        print(f"\n=== GSM8K任务 ===")
        print(f"  样本数: {len(gsm8k_results)}")
        print(f"  准确率: {gsm8k_correct/len(gsm8k_results):.4f}")
    
    # 保存详细结果
    output_jsonl = os.path.join(config.output_dir, "prm_eval_results.jsonl")
    with open(output_jsonl, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n详细结果已保存到: {output_jsonl}")
    
    # 保存摘要
    summary = {
        "dataset": "math_shepherd_validation",
        "threshold": threshold,
        "total_samples": total,
        "math_samples": len(math_results),
        "gsm8k_samples": len(gsm8k_results),
        "correct_labels": sum(ground_truths),
        "wrong_labels": total - sum(ground_truths),
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
        "math_accuracy": math_correct/len(math_results) if math_results else None,
        "gsm8k_accuracy": gsm8k_correct/len(gsm8k_results) if gsm8k_results else None,
    }
    
    summary_path = os.path.join(config.output_dir, "prm_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"摘要已保存到: {summary_path}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="PRM评估脚本 - Math Shepherd验证集")
    parser.add_argument("--stage", type=str, choices=["extract", "score", "all"], required=True,
                        help="运行阶段")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--data_path", type=str, default=None, help="数据集路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--prm_checkpoint", type=str, default=None, help="PRM checkpoint路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="PRM阈值")
    
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
    if args.threshold:
        config.prm_threshold = args.threshold
    
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
