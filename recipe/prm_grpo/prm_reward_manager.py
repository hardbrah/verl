# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PRM Reward Manager for PRM-GRPO.

This reward manager uses a Process Reward Model (PRM) to score truncated
thinking chains using latent states extracted from layer 15.

Key features:
1. Extracts latent states from Actor's layer 15
2. Passes latent states to PRM backbone for scoring
3. Logs all important metrics to JSON files for each step
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import torch

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager

import sys
import os
# Add the directory containing this file to sys.path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from prm_model import PRMScorer


# NOTE: We don't use @register decorator here because this module is loaded via importlib
# The reward manager is specified in the config via reward_manager.source=importlib
class PRMRewardManager(AbstractRewardManager):
    """
    Reward manager that uses a Process Reward Model (PRM) for scoring.
    
    Instead of using outcome-based rewards (checking if the final answer is correct),
    this reward manager:
    1. Extracts latent states from input sequences (layer 15)
    2. Passes latent states to PRM backbone
    3. Uses PRM score as the reward
    
    All important data (prompts, responses, rewards, etc.) are logged to JSON files.
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,  # Not used, but required by interface
        reward_fn_key: str = "data_source",
        prm_checkpoint_path: str = None,
        prm_backbone_path: str = None,
        prm_extract_layer: int = 15,
        prm_max_length: int = 256,
        prm_device: str = "cuda",
        log_dir: str = None,
        **kwargs,
    ) -> None:
        """
        Initialize the PRM Reward Manager.
        
        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: Number of batches of decoded responses to print for debugging.
            compute_score: Not used (for interface compatibility).
            reward_fn_key: Key to access data source in non_tensor_batch.
            prm_checkpoint_path: Path to the PRM checkpoint.
            prm_backbone_path: Path to the PRM backbone model.
            prm_extract_layer: Layer to extract latent states from (default: 15).
            prm_max_length: Maximum sequence length for PRM input.
            prm_device: Device to run PRM on.
            log_dir: Directory to save step-level JSON logs.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        
        # Initialize PRM scorer
        assert prm_checkpoint_path is not None, "prm_checkpoint_path is required"
        assert prm_backbone_path is not None, "prm_backbone_path is required"
        
        print(f"[PRMRewardManager] Initializing...")
        print(f"  - PRM checkpoint: {prm_checkpoint_path}")
        print(f"  - PRM backbone: {prm_backbone_path}")
        print(f"  - Extract layer: {prm_extract_layer}")
        print(f"  - Max length: {prm_max_length}")
        
        self.prm_scorer = PRMScorer(
            prm_checkpoint_path=prm_checkpoint_path,
            backbone_path=prm_backbone_path,
            extract_layer=prm_extract_layer,
            device=prm_device,
            max_length=prm_max_length,
        )
        
        self.prm_max_length = prm_max_length
        self.prm_extract_layer = prm_extract_layer
        
        # Setup logging directory
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "logs",
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"  - Log directory: {self.log_dir}")
        
        # Step counter for logging
        self.step_counter = 0
        
        print(f"[PRMRewardManager] Initialized successfully")
        
    def _log_step_data(
        self,
        step: int,
        prompts: list[str],
        responses: list[str],
        prm_scores: list[float],
        ground_truths: list[str],
        data_sources: list[str],
        extra_info: dict,
    ):
        """Log step data to JSON file."""
        log_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(prompts),
            "samples": [],
            "statistics": {},
        }
        
        # Add individual samples
        for i in range(len(prompts)):
            sample = {
                "index": i,
                "prompt": prompts[i][:500] + "..." if len(prompts[i]) > 500 else prompts[i],
                "response": responses[i],
                "prm_score": prm_scores[i],
                "ground_truth": ground_truths[i] if i < len(ground_truths) else None,
                "data_source": data_sources[i] if i < len(data_sources) else None,
            }
            log_data["samples"].append(sample)
        
        # Add statistics
        scores_tensor = torch.tensor(prm_scores)
        log_data["statistics"] = {
            "prm_score_mean": float(scores_tensor.mean()),
            "prm_score_std": float(scores_tensor.std()),
            "prm_score_min": float(scores_tensor.min()),
            "prm_score_max": float(scores_tensor.max()),
            "prm_score_median": float(scores_tensor.median()),
        }
        
        # Add extra info
        log_data["extra_info"] = extra_info
        
        # Save to file
        log_file = os.path.join(self.log_dir, f"step_{step:06d}.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """
        Compute PRM rewards for a batch of data.
        
        Args:
            data: DataProto containing prompts and responses.
            return_dict: Whether to return a dict with extra info.
            
        Returns:
            Reward tensor or dict with reward tensor and extra info.
        """
        # NOTE: We always compute PRM scores, even if rm_scores exists
        # This ensures we use PRM-based rewards instead of any pre-computed scores
        print(f"[PRMRewardManager] Computing rewards for batch...")
        
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        already_print_data_sources = {}
        
        # Collect all prompts and responses for batch processing
        prompts_list = []
        responses_list = []
        valid_response_lengths = []
        ground_truths = []
        data_sources = []
        
        for i in range(batch_size):
            data_item = data[i]
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode prompt and response
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            prompts_list.append(prompt_str)
            responses_list.append(response_str)
            valid_response_lengths.append(valid_response_length.item())
            
            # Get ground truth and data source for logging
            gt = data_item.non_tensor_batch.get("reward_model", {})
            if isinstance(gt, dict):
                ground_truths.append(str(gt.get("ground_truth", "")))
            else:
                ground_truths.append("")
            
            ds = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            data_sources.append(ds)
        
        # Batch score with PRM
        with torch.no_grad():
            scores, raw_scores = self.prm_scorer.score(prompts_list, responses_list)
            scores = scores.cpu()
            raw_scores = raw_scores.cpu()
        
        # Convert to list for logging
        prm_scores_list = scores.tolist()
        
        # Assign rewards to the last valid token of each response
        for i in range(batch_size):
            valid_response_length = valid_response_lengths[i]
            score = scores[i].item()
            
            # Place reward at the last valid response token
            if valid_response_length > 0:
                reward_tensor[i, int(valid_response_length) - 1] = score
            
            reward_extra_info["prm_score"].append(score)
            reward_extra_info["prm_raw_score"].append(raw_scores[i].item())
            
            # Debug printing
            data_source = data_sources[i]
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n[PRM-GRPO Sample {i}]")
                print(f"[prompt] {prompts_list[i][:300]}...")
                print(f"[response] {responses_list[i][:300]}...")
                print(f"[prm_score] {score:.4f}")
                print(f"[ground_truth] {ground_truths[i]}")
                print("-" * 50)
        
        # Log step data to JSON
        self.step_counter += 1
        extra_log_info = {
            "batch_size": batch_size,
            "prm_max_length": self.prm_max_length,
            "prm_extract_layer": self.prm_extract_layer,
        }
        
        self._log_step_data(
            step=self.step_counter,
            prompts=prompts_list,
            responses=responses_list,
            prm_scores=prm_scores_list,
            ground_truths=ground_truths,
            data_sources=data_sources,
            extra_info=extra_log_info,
        )
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor


class PRMGRPOStepLogger:
    """
    Logger for PRM-GRPO training steps.
    
    Records all important data for each training step:
    - Prompts and responses (rollouts)
    - PRM scores (rewards)
    - Advantages
    - KL divergence
    - Standard deviation
    - Other metrics
    """
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.step_counter = 0
    
    def log_step(
        self,
        global_step: int,
        batch_data: dict,
        metrics: dict,
        is_train: bool = True,
    ):
        """
        Log data for a training/validation step.
        
        Args:
            global_step: Global training step
            batch_data: Dictionary containing batch information
            metrics: Dictionary containing computed metrics
            is_train: Whether this is a training step
        """
        log_entry = {
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
            "is_train": is_train,
            "metrics": metrics,
            "batch_info": {},
        }
        
        # Add batch information
        if "prompts" in batch_data:
            log_entry["batch_info"]["num_samples"] = len(batch_data["prompts"])
        
        if "prm_scores" in batch_data:
            scores = batch_data["prm_scores"]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().tolist()
            log_entry["batch_info"]["prm_scores"] = {
                "values": scores[:10] if len(scores) > 10 else scores,  # First 10
                "mean": float(torch.tensor(scores).mean()),
                "std": float(torch.tensor(scores).std()),
                "min": float(min(scores)),
                "max": float(max(scores)),
            }
        
        if "advantages" in batch_data:
            advs = batch_data["advantages"]
            if isinstance(advs, torch.Tensor):
                advs = advs.cpu()
                # Get non-zero advantages
                non_zero_advs = advs[advs != 0]
                if len(non_zero_advs) > 0:
                    log_entry["batch_info"]["advantages"] = {
                        "mean": float(non_zero_advs.mean()),
                        "std": float(non_zero_advs.std()),
                        "min": float(non_zero_advs.min()),
                        "max": float(non_zero_advs.max()),
                    }
        
        if "kl" in batch_data:
            kl = batch_data["kl"]
            if isinstance(kl, torch.Tensor):
                kl = kl.cpu()
                log_entry["batch_info"]["kl"] = {
                    "mean": float(kl.mean()),
                    "std": float(kl.std()),
                }
        
        # Save samples for detailed inspection (first few)
        if "prompts" in batch_data and "responses" in batch_data:
            samples = []
            num_samples = min(5, len(batch_data["prompts"]))
            for i in range(num_samples):
                sample = {
                    "prompt": batch_data["prompts"][i][:500] if len(batch_data["prompts"][i]) > 500 else batch_data["prompts"][i],
                    "response": batch_data["responses"][i][:500] if len(batch_data["responses"][i]) > 500 else batch_data["responses"][i],
                }
                if "prm_scores" in batch_data:
                    sample["prm_score"] = batch_data["prm_scores"][i] if i < len(batch_data["prm_scores"]) else None
                if "advantages" in batch_data and isinstance(batch_data["advantages"], torch.Tensor):
                    # Get the advantage for this sample (sum over response tokens)
                    sample["advantage"] = float(batch_data["advantages"][i].sum())
                samples.append(sample)
            log_entry["samples"] = samples
        
        # Save to file
        prefix = "train" if is_train else "val"
        log_file = os.path.join(self.log_dir, f"{prefix}_step_{global_step:06d}.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        return log_entry
    
    def log_validation(
        self,
        global_step: int,
        val_metrics: dict,
        val_samples: list = None,
    ):
        """
        Log validation results.
        
        Args:
            global_step: Global training step
            val_metrics: Validation metrics
            val_samples: Optional list of validation samples
        """
        log_entry = {
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
            "validation_metrics": val_metrics,
        }
        
        if val_samples:
            log_entry["samples"] = val_samples[:10]  # First 10 samples
        
        log_file = os.path.join(self.log_dir, f"validation_step_{global_step:06d}.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        return log_entry
