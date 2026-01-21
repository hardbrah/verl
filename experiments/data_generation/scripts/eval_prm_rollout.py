#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRM效果测试脚本 - 两阶段实现

阶段1 (extract): 使用8卡并行提取latent states（已缓存）
阶段2 (score): 对latent states打分并计算准确率

使用方法:
# 阶段1: 提取latent states (8卡) - 如果需要重新提取
torchrun --nproc_per_node=8 eval_prm_rollout.py --stage extract --batch_size 128

# 阶段2: 打分和评估 (单卡)
python eval_prm_rollout.py --stage score

# 或一键运行两个阶段
python eval_prm_rollout.py --stage all
"""

import argparse
import json
import os
import sys
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

# 直接导入math_dapo模块
import importlib.util
math_dapo_path = os.path.join(project_root, "verl", "utils", "reward_score", "math_dapo.py")
spec = importlib.util.spec_from_file_location("math_dapo", math_dapo_path)
math_dapo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_dapo)
dapo_compute_score = math_dapo.compute_score


# ==================== 配置 ====================

@dataclass
class Config:
    # 数据集路径
    data_path: str = "/data/chenhaotian/verl/experiments/data_generation/outputs/qwen3_4b_1000query_16sample/rollout_20260107_134649.jsonl"
    
    # 模型路径
    model_path: str = "/data/models/Qwen/Qwen3-4B-Instruct-2507"
    
    # PRM checkpoint路径
    prm_checkpoint_path: str = "/data/chenhaotian/verl/checkpoints/latentqa/math_shepherd_20260116/best_model.pt"
    
    # 输出目录
    output_dir: str = "/data/chenhaotian/verl/experiments/data_generation/outputs/prm_eval"
    
    # 缓存目录（阶段1已生成的latent states）
    cache_dir: str = "/data/chenhaotian/verl/experiments/data_generation/outputs/prm_latent_cache"
    
    # Latent states提取配置
    extract_layer: int = 15  # 提取第15层的latent states
    response_max_tokens: int = 64  # response最多取64个token
    
    # PRM阈值
    prm_threshold: float = 0.5
    
    # 数据类型
    dtype: torch.dtype = torch.bfloat16


# ==================== 数据集 ====================

class RolloutDataset(Dataset):
    """Rollout数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        response_max_tokens: int = 64,
    ):
        self.tokenizer = tokenizer
        self.response_max_tokens = response_max_tokens
        
        # 加载数据
        print(f"Loading data from {data_path}")
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 获取question和response
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
            "response": response,  # 完整response用于后续验证
            "gt_answer": gt_answer,
            "q_id": item.get("q_id", idx),
            "sample_idx": item.get("sample_idx", 0),
        }


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict:
    """Collate function for DataLoader"""
    # Pad input_ids
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
        
        # 加载backbone模型
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        
        # 删除lm_head节省显存
        if hasattr(self.backbone, 'lm_head'):
            del self.backbone.lm_head
        
        # Regression head
        self.regression_head = RegressionHead(hidden_size).to(dtype)
        
        self.hidden_size = hidden_size
        self.dtype = dtype
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        Returns:
            scores: [batch_size]
        """
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
    
    # 初始化分布式
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"=== 阶段1: 提取Latent States ===")
        print(f"World size: {world_size}")
        print(f"Model: {config.model_path}")
        print(f"Extract layer: {config.extract_layer}")
        print(f"Response max tokens: {config.response_max_tokens}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    dataset = RolloutDataset(
        data_path=config.data_path,
        tokenizer=tokenizer,
        response_max_tokens=config.response_max_tokens,
    )
    
    # 分布式采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda x: collate_fn(x, tokenizer),
        num_workers=4,
        pin_memory=True,
    )
    
    # 加载模型
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
    
    # 提取hidden states
    results = []
    extract_layer_idx = config.extract_layer + 1  # +1因为第0个是embedding
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"[Rank {rank}] Extracting", disable=(rank != 0)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward获取hidden states
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            
            # 提取指定层的hidden states
            latent_states = outputs.hidden_states[extract_layer_idx]  # [B, L, H]
            
            # 保存每个样本的结果
            for i in range(len(batch["idx"])):
                idx = batch["idx"][i]
                seq_len = int(attention_mask[i].sum().item())
                
                # 右padding: 有效token在前面，所以用 :seq_len
                latent = latent_states[i, :seq_len, :].cpu()  # 只保留非padding部分
                
                # Decode出文本（保留special tokens）
                valid_input_ids = input_ids[i, :seq_len].cpu().tolist()
                decoded_text = tokenizer.decode(valid_input_ids, skip_special_tokens=False)
                
                results.append({
                    "idx": idx,
                    "q_id": batch["q_id"][i],
                    "sample_idx": batch["sample_idx"][i],
                    "input_ids": valid_input_ids,
                    "decoded_text": decoded_text,
                    "latent_states": latent,  # Tensor
                    "question": batch["question"][i],
                    "response": batch["response"][i],
                    "gt_answer": batch["gt_answer"][i],
                })
    
    # 释放GPU显存
    del model
    torch.cuda.empty_cache()
    
    # 保存结果
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, f"latent_states_rank{rank}.pt")
    torch.save(results, output_path)
    
    if rank == 0:
        print(f"Saved {len(results)} samples to {output_path}")
    
    # 同步所有进程
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    
    # Rank 0 合并所有结果
    if rank == 0:
        print("Merging results from all ranks...")
        all_results = []
        for r in range(world_size):
            path = os.path.join(config.output_dir, f"latent_states_rank{r}.pt")
            data = torch.load(path, weights_only=False)
            all_results.extend(data)
        
        # 按idx排序
        all_results.sort(key=lambda x: x["idx"])
        
        # 保存合并后的结果
        merged_path = os.path.join(config.output_dir, "latent_states_merged.pt")
        torch.save(all_results, merged_path)
        print(f"Merged {len(all_results)} samples to {merged_path}")
        
        # 清理临时文件
        for r in range(world_size):
            path = os.path.join(config.output_dir, f"latent_states_rank{r}.pt")
            os.remove(path)
        print("Cleaned up temporary files")
    
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


# ==================== 阶段2: 打分和评估 ====================

def load_cached_latent_states(cache_dir: str) -> List[str]:
    """从缓存目录加载所有latent states part文件路径"""
    print(f"Loading cached latent states from {cache_dir}")
    
    # 读取所有manifest文件
    manifest_files = sorted(glob(os.path.join(cache_dir, "manifest.rank*.json")))
    print(f"Found {len(manifest_files)} manifest files")
    
    all_parts = []
    for manifest_file in manifest_files:
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
        for part_info in manifest["parts"]:
            all_parts.append(part_info["part"])
    
    print(f"Total {len(all_parts)} part files to load")
    
    # 按文件名排序确保顺序一致
    all_parts = sorted(set(all_parts))
    
    return all_parts


def score_and_evaluate_distributed(config: Config, batch_size: int = 64):
    """阶段2: 使用分布式多卡对latent states打分并计算准确率"""
    
    # 初始化分布式
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"=== 阶段2: 分布式打分和评估 ===")
        print(f"World size: {world_size}")
    
    # 加载PRM checkpoint
    if rank == 0:
        print(f"Loading PRM checkpoint from {config.prm_checkpoint_path}")
    checkpoint = torch.load(config.prm_checkpoint_path, map_location="cpu", weights_only=False)
    
    # 获取hidden size
    model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
    hidden_size = model_config.hidden_size
    if rank == 0:
        print(f"Hidden size: {hidden_size}")
    
    # 创建PRM Backbone
    if rank == 0:
        print(f"Creating PRM backbone...")
    prm_backbone = PRMBackbone(
        model_path=config.model_path,
        hidden_size=hidden_size,
        dtype=config.dtype,
    )
    
    # 加载PRM权重
    state_dict = checkpoint["model_state_dict"]
    
    # 加载backbone权重
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            new_key = key[len("backbone."):]
            backbone_state[new_key] = value
    
    if backbone_state:
        missing, unexpected = prm_backbone.backbone.load_state_dict(backbone_state, strict=False)
        if rank == 0:
            print(f"Loaded backbone weights: {len(backbone_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    
    # 加载regression head权重
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
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载latent states（支持两种格式）
    # 格式1: latent_states_merged.pt (阶段1生成的)
    # 格式2: manifest.rank*.json + part_r*.pt (旧缓存格式)
    merged_path = os.path.join(config.output_dir, "latent_states_merged.pt")
    
    if os.path.exists(merged_path):
        # 使用阶段1生成的merged格式
        if rank == 0:
            print(f"Loading merged latent states from {merged_path}")
        all_data = torch.load(merged_path, map_location="cpu", weights_only=False)
        if rank == 0:
            print(f"Loaded {len(all_data)} samples")
        
        # 分配给各rank
        my_data = [all_data[i] for i in range(len(all_data)) if i % world_size == rank]
        if rank == 0:
            print(f"Each rank processes ~{len(my_data)} samples")
        
        # 处理
        local_results = []
        
        for i in tqdm(range(0, len(my_data), batch_size), desc=f"[Rank {rank}] Scoring", disable=(rank != 0)):
            batch_items = my_data[i:i+batch_size]
            
            # Pad latent states
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
                    "q_id": item["q_id"],
                    "sample_idx": item["sample_idx"],
                    "decoded_text": item["decoded_text"],
                    "prm_score": scores_list[j],
                    "question": item["question"],
                    "response": item["response"],
                    "gt_answer": item["gt_answer"],
                })
    else:
        # 使用旧的manifest格式
        all_part_files = load_cached_latent_states(config.cache_dir)
        my_parts = [p for i, p in enumerate(all_part_files) if i % world_size == rank]
        if rank == 0:
            print(f"Total {len(all_part_files)} parts, each rank processes ~{len(my_parts)} parts")
        
        local_results = []
        
        for part_file in tqdm(my_parts, desc=f"[Rank {rank}] Scoring", disable=(rank != 0)):
            part_data = torch.load(part_file, map_location="cpu", weights_only=False)
            
            input_ids = part_data["input_ids"]
            attention_mask = part_data["attention_mask"]
            latent_states = part_data["latent_states"]
            meta = part_data["meta"]
            
            batch_count = input_ids.shape[0]
            
            for i in range(0, batch_count, batch_size):
                end_idx = min(i + batch_size, batch_count)
                
                batch_latent = latent_states[i:end_idx].to(device)
                batch_mask = attention_mask[i:end_idx].to(device)
                batch_input_ids = input_ids[i:end_idx]
                
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=config.dtype):
                    scores = prm_backbone(batch_latent, batch_mask)
                
                scores_list = scores.cpu().tolist()
                
                for j in range(end_idx - i):
                    item_meta = meta[i + j]
                    item_input_ids = batch_input_ids[j]
                    item_mask = batch_mask[j].cpu()
                    
                    valid_len = int(item_mask.sum().item())
                    valid_input_ids = item_input_ids[-valid_len:].tolist()
                    decoded_text = tokenizer.decode(valid_input_ids, skip_special_tokens=False)
                    
                    local_results.append({
                        "q_id": item_meta["q_id"],
                        "sample_idx": item_meta["sample_idx"],
                        "decoded_text": decoded_text,
                        "prm_score": scores_list[j],
                        "question": item_meta["question"],
                        "response": item_meta["response"],
                        "gt_answer": item_meta["gt_answer"],
                    })
    
    # 保存本地结果
    os.makedirs(config.output_dir, exist_ok=True)
    local_output = os.path.join(config.output_dir, f"results_rank{rank}.pt")
    torch.save(local_results, local_output)
    print(f"[Rank {rank}] Saved {len(local_results)} results to {local_output}")
    
    # 释放GPU显存
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
        
        print(f"Total samples: {len(all_results)}")
        
        # 计算DAPO验证分数
        print("Computing DAPO verification scores...")
        for item in tqdm(all_results, desc="Verifying"):
            dapo_result = dapo_compute_score(
                solution_str=item["response"],
                ground_truth=item["gt_answer"],
            )
            item["dapo_score"] = dapo_result["score"]
            item["dapo_acc"] = dapo_result["acc"]
            item["dapo_pred"] = dapo_result["pred"]
        
        # 计算指标
        evaluate_and_save_results(all_results, config)
    
    dist.barrier(device_ids=[local_rank])
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate_and_save_results(results: List[Dict], config: Config):
    """评估结果并保存"""
    print("\n=== 评估结果 ===")
    
    total = len(results)
    if total == 0:
        print("错误: 没有结果可以评估!")
        return
    
    threshold = config.prm_threshold
    prm_predictions = [1 if item["prm_score"] >= threshold else 0 for item in results]
    ground_truths = [1 if item["dapo_acc"] else 0 for item in results]
    
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
    prm_scores_correct = [item["prm_score"] for item in results if item["dapo_acc"]]
    prm_scores_wrong = [item["prm_score"] for item in results if not item["dapo_acc"]]
    
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
        print(f"  正确答案: mean={sum(prm_scores_correct)/len(prm_scores_correct):.4f}")
    if prm_scores_wrong:
        print(f"  错误答案: mean={sum(prm_scores_wrong)/len(prm_scores_wrong):.4f}")
    
    # 保存详细结果
    output_jsonl = os.path.join(config.output_dir, "prm_eval_results.jsonl")
    with open(output_jsonl, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n详细结果已保存到: {output_jsonl}")
    
    # 保存摘要
    summary = {
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


def score_and_evaluate(config: Config, batch_size: int = 64):
    """阶段2: 对latent states打分并计算准确率"""
    
    print(f"=== 阶段2: 打分和评估 ===")
    
    device = torch.device("cuda:0")
    
    # 加载PRM checkpoint
    print(f"Loading PRM checkpoint from {config.prm_checkpoint_path}")
    checkpoint = torch.load(config.prm_checkpoint_path, map_location="cpu", weights_only=False)
    
    # 获取hidden size
    model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
    hidden_size = model_config.hidden_size
    print(f"Hidden size: {hidden_size}")
    
    # 创建PRM Backbone
    print(f"Creating PRM backbone...")
    prm_backbone = PRMBackbone(
        model_path=config.model_path,
        hidden_size=hidden_size,
        dtype=config.dtype,
    )
    
    # 加载PRM权重
    state_dict = checkpoint["model_state_dict"]
    
    # 加载backbone权重
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            new_key = key[len("backbone."):]
            backbone_state[new_key] = value
    
    if backbone_state:
        missing, unexpected = prm_backbone.backbone.load_state_dict(backbone_state, strict=False)
        print(f"Loaded backbone weights: {len(backbone_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing keys (may include lm_head): {missing[:5]}...")
    
    # 加载regression head权重
    regression_head_state = {}
    for key, value in state_dict.items():
        if key.startswith("regression_head."):
            new_key = key[len("regression_head."):]
            regression_head_state[new_key] = value
    
    if regression_head_state:
        prm_backbone.regression_head.load_state_dict(regression_head_state)
        print(f"Loaded regression head weights: {list(regression_head_state.keys())}")
    
    prm_backbone = prm_backbone.to(device)
    prm_backbone.eval()
    
    # 加载tokenizer用于decode
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 获取所有part文件
    part_files = load_cached_latent_states(config.cache_dir)
    
    # 处理所有part文件
    print("\nScoring latent states...")
    results = []
    global_idx = 0
    
    for part_file in tqdm(part_files, desc="Processing parts"):
        # 加载part数据
        part_data = torch.load(part_file, map_location="cpu", weights_only=False)
        
        input_ids = part_data["input_ids"]  # [B, L]
        attention_mask = part_data["attention_mask"]  # [B, L]
        latent_states = part_data["latent_states"]  # [B, L, H]
        meta = part_data["meta"]  # List of dicts
        
        batch_count = input_ids.shape[0]
        
        # 分批处理
        for i in range(0, batch_count, batch_size):
            end_idx = min(i + batch_size, batch_count)
            
            batch_latent = latent_states[i:end_idx].to(device)
            batch_mask = attention_mask[i:end_idx].to(device)
            batch_input_ids = input_ids[i:end_idx]
            
            # Forward
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=config.dtype):
                scores = prm_backbone(batch_latent, batch_mask)
            
            scores_list = scores.cpu().tolist()
            
            # 保存结果
            for j in range(end_idx - i):
                item_meta = meta[i + j]
                item_input_ids = batch_input_ids[j]
                item_mask = batch_mask[j].cpu()
                
                # 获取有效token的input_ids
                valid_len = item_mask.sum().item()
                valid_input_ids = item_input_ids[-valid_len:].tolist()
                
                # Decode出文本（保留special tokens）
                decoded_text = tokenizer.decode(valid_input_ids, skip_special_tokens=False)
                
                results.append({
                    "idx": global_idx,
                    "q_id": item_meta["q_id"],
                    "sample_idx": item_meta["sample_idx"],
                    "decoded_text": decoded_text,
                    "prm_score": scores_list[j],
                    "question": item_meta["question"],
                    "response": item_meta["response"],
                    "gt_answer": item_meta["gt_answer"],
                })
                global_idx += 1
    
    print(f"Total samples processed: {len(results)}")
    
    # 计算DAPO验证分数
    print("Computing DAPO verification scores...")
    for item in tqdm(results, desc="Verifying"):
        dapo_result = dapo_compute_score(
            solution_str=item["response"],
            ground_truth=item["gt_answer"],
        )
        item["dapo_score"] = dapo_result["score"]
        item["dapo_acc"] = dapo_result["acc"]
        item["dapo_pred"] = dapo_result["pred"]
    
    # 计算准确率
    print("\n=== 评估结果 ===")
    
    # 使用0.5阈值
    threshold = config.prm_threshold
    prm_predictions = [1 if item["prm_score"] >= threshold else 0 for item in results]
    ground_truths = [1 if item["dapo_acc"] else 0 for item in results]
    
    # 统计
    total = len(results)
    correct_predictions = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == g)
    accuracy = correct_predictions / total
    
    # True/False Positive/Negative
    tp = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == 0 and g == 0)
    fn = sum(1 for p, g in zip(prm_predictions, ground_truths) if p == 0 and g == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # PRM分数统计
    prm_scores = [item["prm_score"] for item in results]
    prm_scores_correct = [item["prm_score"] for item in results if item["dapo_acc"]]
    prm_scores_wrong = [item["prm_score"] for item in results if not item["dapo_acc"]]
    
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
        print(f"  正确答案: mean={sum(prm_scores_correct)/len(prm_scores_correct):.4f}")
    if prm_scores_wrong:
        print(f"  错误答案: mean={sum(prm_scores_wrong)/len(prm_scores_wrong):.4f}")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 保存详细结果到JSONL
    output_jsonl = os.path.join(config.output_dir, "prm_eval_results.jsonl")
    with open(output_jsonl, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n详细结果已保存到: {output_jsonl}")
    
    # 保存摘要
    summary = {
        "threshold": threshold,
        "total_samples": total,
        "correct_answers": sum(ground_truths),
        "wrong_answers": total - sum(ground_truths),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
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
    parser = argparse.ArgumentParser(description="PRM评估脚本")
    parser.add_argument("--stage", type=str, choices=["extract", "score", "score_dist", "all"], required=True,
                        help="运行阶段: extract=提取latent states, score=单卡打分, score_dist=分布式打分, all=两个阶段")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--data_path", type=str, default=None, help="数据集路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--prm_checkpoint", type=str, default=None, help="PRM checkpoint路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--cache_dir", type=str, default=None, help="缓存目录（阶段1已生成的latent states）")
    parser.add_argument("--response_max_tokens", type=int, default=64, help="Response最大token数")
    parser.add_argument("--threshold", type=float, default=0.5, help="PRM阈值")
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    if args.data_path:
        config.data_path = args.data_path
    if args.model_path:
        config.model_path = args.model_path
    if args.prm_checkpoint:
        config.prm_checkpoint_path = args.prm_checkpoint
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.response_max_tokens:
        config.response_max_tokens = args.response_max_tokens
    if args.threshold:
        config.prm_threshold = args.threshold
    
    # 运行
    if args.stage == "extract":
        extract_latent_states(config, batch_size=args.batch_size)
    elif args.stage == "score":
        score_and_evaluate(config, batch_size=args.batch_size)
    elif args.stage == "score_dist":
        score_and_evaluate_distributed(config, batch_size=args.batch_size)
    elif args.stage == "all":
        print("请分两步运行:")
        print(f"1. torchrun --nproc_per_node=8 {__file__} --stage extract --batch_size {args.batch_size}")
        print(f"2. torchrun --nproc_per_node=8 {__file__} --stage score_dist --batch_size {args.batch_size}")


if __name__ == "__main__":
    main()
