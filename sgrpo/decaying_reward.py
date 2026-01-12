# Copyright 2024 S-GRPO Implementation
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
Decaying reward computation for S-GRPO.

The decaying reward encourages the model to produce correct answers with
shorter chain-of-thought sequences. Rewards decrease exponentially as
more correct answers are found.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


class DecayingRewardManager(AbstractRewardManager):
    """
    Reward manager implementing the decaying reward function for S-GRPO.
    
    The decaying reward works as follows:
    - For a given query, answers are ordered from shortest to longest CoT
    - For correct answer_i: reward_i = 1 / (2^N_right)
      where N_right is the count of correct answers from the start to answer_i
    - For incorrect answer_i: reward_i = 0
    
    This encourages the model to:
    1. Find correct answers
    2. Prefer shorter chains of thought that still produce correct answers
    """
    
    def __init__(
        self,
        tokenizer,
        compute_score: Callable,
        num_examine: int = 0,
        reward_fn_key: str = "reward_model",
        num_truncations: int = 4,  # m
        **kwargs,
    ):
        """
        Initialize the decaying reward manager.
        
        Args:
            tokenizer: HuggingFace tokenizer
            compute_score: Function to compute correctness of an answer
            num_examine: Number of samples to examine for debugging
            reward_fn_key: Key for reward function in data
            num_truncations: Number of truncations per sample (m)
        """
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.num_truncations = num_truncations
    
    def __call__(
        self, 
        data: DataProto, 
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, Any]:
        """
        Compute decaying rewards for a batch of S-GRPO data.
        
        The data should contain responses ordered as:
        [sample_0_cot_1, sample_0_cot_2, ..., sample_0_cot_m, sample_0_cot_0,
         sample_1_cot_1, sample_1_cot_2, ..., sample_1_cot_m, sample_1_cot_0,
         ...]
        
        Args:
            data: DataProto containing prompts, responses, and metadata
            return_dict: If True, return dict with reward_tensor and extra_info
            
        Returns:
            Reward tensor or dict with rewards and extra info
        """
        batch_size = len(data.batch["responses"])
        response_length = data.batch["responses"].shape[1]
        device = data.batch["responses"].device
        
        # Get the number of sequences per original sample
        num_seqs_per_sample = self.num_truncations + 1  # m + 1 (m truncated + 1 original)
        
        # Compute correctness for each response
        correctness_list = []
        answers_list = []
        
        for i in range(batch_size):
            # Decode the response
            response = data.batch["responses"][i]
            response_mask = data.batch["response_mask"][i]
            valid_response = response[response_mask == 1]
            response_text = self.tokenizer.decode(valid_response, skip_special_tokens=True)
            
            # Extract answer and check correctness
            # Get ground truth from the data
            sample_idx = i // num_seqs_per_sample
            if "reward_model" in data.non_tensor_batch:
                reward_info = data.non_tensor_batch["reward_model"]
                if isinstance(reward_info, np.ndarray) and len(reward_info) > sample_idx:
                    sample_reward_info = reward_info[i] if len(reward_info) == batch_size else reward_info[sample_idx]
                    ground_truth = sample_reward_info.get("ground_truth", None) if isinstance(sample_reward_info, dict) else None
                else:
                    ground_truth = None
            else:
                ground_truth = None
            
            # Use compute_score to get correctness
            try:
                if ground_truth is not None:
                    score = self.compute_score(response_text, ground_truth)
                else:
                    # Try to get data_source for more context
                    data_source = None
                    if "data_source" in data.non_tensor_batch:
                        ds = data.non_tensor_batch["data_source"]
                        data_source = ds[i] if len(ds) == batch_size else ds[sample_idx]
                    
                    score = self.compute_score(
                        response_text,
                        data_source=data_source,
                    )
                is_correct = score > 0.5 if isinstance(score, (int, float)) else bool(score)
            except Exception:
                is_correct = False
            
            correctness_list.append(is_correct)
            answers_list.append(response_text)
        
        # Compute decaying rewards
        rewards = self._compute_decaying_rewards(
            correctness_list=correctness_list,
            num_seqs_per_sample=num_seqs_per_sample,
        )
        
        # Create token-level reward tensor (reward only at the last token)
        reward_tensor = torch.zeros(batch_size, response_length, device=device)
        
        for i in range(batch_size):
            # Find the last valid token position
            response_mask = data.batch["response_mask"][i]
            last_valid_idx = (response_mask.sum() - 1).clamp(min=0)
            reward_tensor[i, last_valid_idx] = rewards[i]
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "correctness": correctness_list,
                    "raw_rewards": rewards.tolist(),
                },
            }
        
        return reward_tensor
    
    def _compute_decaying_rewards(
        self,
        correctness_list: List[bool],
        num_seqs_per_sample: int,
    ) -> torch.Tensor:
        """
        Compute decaying rewards based on correctness.
        
        For each sample group, rewards are computed in order:
        - If answer_i is correct: reward_i = 1 / (2^N_right)
          where N_right is the number of correct answers BEFORE answer_i (not including itself)
        - If answer_i is incorrect: reward_i = 0
        
        This means:
        - First correct answer:  N_right=0, reward=1/(2^0)=1
        - Second correct answer: N_right=1, reward=1/(2^1)=0.5
        - Third correct answer:  N_right=2, reward=1/(2^2)=0.25
        
        Args:
            correctness_list: List of correctness booleans for all responses
            num_seqs_per_sample: Number of sequences per original sample (m+1)
            
        Returns:
            Tensor of rewards
        """
        total_samples = len(correctness_list)
        num_complete_groups = total_samples // num_seqs_per_sample
        remainder = total_samples % num_seqs_per_sample
        
        if remainder != 0:
            print(f"Warning: batch_size ({total_samples}) is not divisible by "
                  f"num_seqs_per_sample ({num_seqs_per_sample}). "
                  f"Remainder {remainder} samples will receive 0 reward.")
        
        rewards = []
        
        for sample_idx in range(num_complete_groups):
            start_idx = sample_idx * num_seqs_per_sample
            end_idx = start_idx + num_seqs_per_sample
            sample_correctness = correctness_list[start_idx:end_idx]
            
            n_right = 0  # Number of correct answers BEFORE current position
            sample_rewards = []
            
            for is_correct in sample_correctness:
                if is_correct:
                    # Compute reward using n_right (correct answers BEFORE this one)
                    reward = 1.0 / (2 ** n_right)
                    # Then increment the counter for next iterations
                    n_right += 1
                else:
                    reward = 0.0
                sample_rewards.append(reward)
            
            rewards.extend(sample_rewards)
        
        # Handle remainder samples (assign 0 reward)
        for _ in range(remainder):
            rewards.append(0.0)
        
        return torch.tensor(rewards, dtype=torch.float32)


class SGRPORewardManager(AbstractRewardManager):
    """
    Complete reward manager for S-GRPO that handles both:
    1. Answer extraction from truncated CoTs
    2. Decaying reward computation
    
    This manager is designed to work with the S-GRPO trainer.
    """
    
    def __init__(
        self,
        tokenizer,
        compute_score: Callable,
        num_examine: int = 0,
        reward_fn_key: str = "reward_model",
        num_truncations: int = 4,
        answer_extractor: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize the S-GRPO reward manager.
        
        Args:
            tokenizer: HuggingFace tokenizer
            compute_score: Function to verify answer correctness
            num_examine: Number of samples to examine
            reward_fn_key: Key for reward function in data
            num_truncations: Number of truncations per sample
            answer_extractor: Optional custom answer extraction function
        """
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.num_truncations = num_truncations
        self.answer_extractor = answer_extractor or self._default_extract_answer
    
    def _default_extract_answer(self, response_text: str, data_source: Optional[str] = None) -> str:
        """Default answer extraction logic.
        
        Args:
            response_text: The response text to extract answer from
            data_source: The data source identifier to determine extraction method
            
        Returns:
            Extracted answer string
        """
        import re
        
        # For GSM8K dataset, use the same extraction logic as GRPO
        if data_source == "openai/gsm8k":
            from verl.utils.reward_score.gsm8k import extract_solution
            answer = extract_solution(response_text, method="strict")
            if answer is not None:
                return answer
            # If strict mode fails, try flexible mode
            answer = extract_solution(response_text, method="flexible")
            if answer is not None:
                return answer
            # Fallback to default patterns if GSM8K extraction fails
        
        # Try common patterns for other datasets
        patterns = [
            r"\\boxed{([^}]+)}",
            r"The answer is[:\s]+(.+?)(?:\.|$)",
            r"Answer[:\s]+(.+?)(?:\.|$)",
            r"Therefore[,:\s]+(.+?)(?:\.|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Return last line as fallback
        lines = response_text.strip().split('\n')
        return lines[-1].strip() if lines else response_text
    
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, Any]:
        """
        Compute S-GRPO rewards for a batch.
        
        Args:
            data: DataProto with responses (both truncated and original)
            return_dict: Whether to return extra info
            
        Returns:
            Reward tensor or dict with rewards and extra info
        """
        batch_size = len(data.batch["responses"])
        response_length = data.batch["responses"].shape[1]
        device = data.batch["responses"].device
        
        num_seqs_per_sample = self.num_truncations + 1
        
        # Process each response
        correctness_list = []
        answers_list = []
        scores_list = []
        
        for i in range(batch_size):
            response = data.batch["responses"][i]
            response_mask = data.batch["response_mask"][i]
            valid_response = response[response_mask == 1]
            response_text = self.tokenizer.decode(valid_response, skip_special_tokens=True)
            
            # Get ground truth and data source first (needed for answer extraction)
            sample_idx = i // num_seqs_per_sample
            ground_truth = self._get_ground_truth(data, i, sample_idx, batch_size)
            data_source = self._get_data_source(data, i, sample_idx, batch_size)
            
            # Compute score/correctness
            # IMPORTANT: Pass the full response_text to compute_score, not the extracted answer!
            # This matches GRPO's behavior where compute_score internally extracts the answer.
            # For GSM8K, gsm8k.compute_score expects the full response and calls extract_solution internally.
            try:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_text,  # Pass full response, not extracted answer
                    ground_truth=ground_truth,
                )
                
                if isinstance(score, bool):
                    is_correct = score
                    score = 1.0 if score else 0.0
                else:
                    is_correct = score > 0.5
            except Exception as e:
                print(f"Warning: Error computing score: {e}")
                is_correct = False
                score = 0.0
            
            # Extract answer for logging purposes only (after computing score)
            if self.answer_extractor == self._default_extract_answer:
                answer = self._default_extract_answer(response_text, data_source=data_source)
            else:
                answer = self.answer_extractor(response_text)
            answers_list.append(answer)
            
            correctness_list.append(is_correct)
            scores_list.append(score)
        
        # Compute decaying rewards
        rewards = self._compute_decaying_rewards(correctness_list, num_seqs_per_sample)
        
        # Create token-level reward tensor
        reward_tensor = torch.zeros(batch_size, response_length, device=device)
        
        for i in range(batch_size):
            response_mask = data.batch["response_mask"][i]
            last_valid_idx = (response_mask.sum() - 1).clamp(min=0)
            reward_tensor[i, last_valid_idx] = rewards[i]
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "correctness": correctness_list,
                    "raw_scores": scores_list,
                    "decaying_rewards": rewards.tolist(),
                    "answers": answers_list,
                },
            }
        
        return reward_tensor
    
    def _get_ground_truth(
        self,
        data: DataProto,
        i: int,
        sample_idx: int,
        batch_size: int,
    ) -> Optional[Any]:
        """Extract ground truth from data."""
        if "reward_model" not in data.non_tensor_batch:
            return None
        
        reward_info = data.non_tensor_batch["reward_model"]
        if isinstance(reward_info, np.ndarray):
            if len(reward_info) == batch_size:
                item = reward_info[i]
            else:
                item = reward_info[sample_idx]
            
            if isinstance(item, dict):
                return item.get("ground_truth", None)
        
        return None
    
    def _get_data_source(
        self,
        data: DataProto,
        i: int,
        sample_idx: int,
        batch_size: int,
    ) -> Optional[str]:
        """Extract data source from data."""
        if "data_source" not in data.non_tensor_batch:
            return None
        
        ds = data.non_tensor_batch["data_source"]
        if len(ds) == batch_size:
            return ds[i]
        return ds[sample_idx]
    
    def _compute_decaying_rewards(
        self,
        correctness_list: List[bool],
        num_seqs_per_sample: int,
    ) -> torch.Tensor:
        """Compute decaying rewards for correctness list.
        
        N_right is the count of correct answers BEFORE current answer.
        - First correct:  reward = 1/(2^0) = 1
        - Second correct: reward = 1/(2^1) = 0.5
        - Third correct:  reward = 1/(2^2) = 0.25
        """
        total_samples = len(correctness_list)
        num_complete_groups = total_samples // num_seqs_per_sample
        remainder = total_samples % num_seqs_per_sample
        
        if remainder != 0:
            print(f"Warning: batch_size ({total_samples}) is not divisible by "
                  f"num_seqs_per_sample ({num_seqs_per_sample}). "
                  f"Remainder {remainder} samples will receive 0 reward.")
        
        rewards = []
        
        for sample_idx in range(num_complete_groups):
            start_idx = sample_idx * num_seqs_per_sample
            end_idx = start_idx + num_seqs_per_sample
            sample_correctness = correctness_list[start_idx:end_idx]
            
            n_right = 0  # Correct answers BEFORE current position
            for is_correct in sample_correctness:
                if is_correct:
                    # Compute reward first, then increment
                    reward = 1.0 / (2 ** n_right)
                    n_right += 1
                else:
                    reward = 0.0
                rewards.append(reward)
        
        # Handle remainder samples (assign 0 reward)
        for _ in range(remainder):
            rewards.append(0.0)
        
        return torch.tensor(rewards, dtype=torch.float32)


def create_sgrpo_reward_manager(
    config,
    tokenizer,
    num_truncations: int = 4,
    **kwargs,
) -> SGRPORewardManager:
    """
    Factory function to create an S-GRPO reward manager.
    
    Args:
        config: Configuration object
        tokenizer: Tokenizer
        num_truncations: Number of truncations per sample
        **kwargs: Additional arguments
        
    Returns:
        SGRPORewardManager instance
    """
    from verl.trainer.ppo.reward import get_custom_reward_fn
    from verl.utils.reward_score import default_compute_score
    
    # Get custom reward function or use default
    compute_score = get_custom_reward_fn(config)
    if compute_score is None:
        compute_score = default_compute_score
    
    return SGRPORewardManager(
        tokenizer=tokenizer,
        compute_score=compute_score,
        num_truncations=num_truncations,
        **kwargs,
    )
