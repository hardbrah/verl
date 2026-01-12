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
Core algorithms for S-GRPO.
Implements the decaying reward function and S-GRPO advantage estimation.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch


def compute_decaying_reward(
    answers: List[str],
    verify_fn: Callable[[str], bool],
    ground_truth: Any = None,
) -> List[float]:
    """
    Compute decaying reward for a sequence of answers.
    
    The answers should be ordered from CoT1, CoT2, ..., CoTm, CoT0,
    where CoT1 to CoTm are truncated chains of thought (length increasing),
    and CoT0 is the complete chain of thought.
    
    Reward rule:
    - If answer_i is correct: reward_i = 1 / (2^N_right)
      where N_right is the count of correct answers BEFORE answer_i (not including itself)
    - If answer_i is incorrect: reward_i = 0
    
    This means:
    - First correct answer:  N_right=0, reward=1/(2^0)=1
    - Second correct answer: N_right=1, reward=1/(2^1)=0.5
    - Third correct answer:  N_right=2, reward=1/(2^2)=0.25
    
    Args:
        answers: List of extracted answers in order [answer1, answer2, ..., answerm, answer0]
        verify_fn: Function to verify if an answer is correct
                   Takes (answer, ground_truth) as arguments and returns bool
        ground_truth: The ground truth for verification
        
    Returns:
        List of rewards corresponding to each answer
    """
    rewards = []
    n_right = 0  # Number of correct answers BEFORE current position
    
    for answer in answers:
        # Verify if the answer is correct
        if ground_truth is not None:
            is_correct = verify_fn(answer, ground_truth)
        else:
            is_correct = verify_fn(answer)
        
        if is_correct:
            # Compute reward using n_right (correct answers BEFORE this one)
            reward = 1.0 / (2 ** n_right)
            # Then increment the counter for next iterations
            n_right += 1
        else:
            reward = 0.0
            
        rewards.append(reward)
    
    return rewards


def compute_decaying_reward_batch(
    answers_batch: List[List[str]],
    correctness_batch: List[List[bool]],
) -> List[List[float]]:
    """
    Compute decaying reward for a batch of answer sequences.
    
    Args:
        answers_batch: Batch of answer lists, each list contains 
                       [answer1, ..., answerm, answer0] for one query
        correctness_batch: Batch of correctness lists, each list contains
                           [correct1, ..., correctm, correct0] for one query
        
    Returns:
        Batch of reward lists
    """
    rewards_batch = []
    
    for correctness_list in correctness_batch:
        rewards = []
        n_right = 0  # Number of correct answers BEFORE current position
        
        for is_correct in correctness_list:
            if is_correct:
                # Compute reward using n_right (correct answers BEFORE this one)
                reward = 1.0 / (2 ** n_right)
                # Then increment the counter for next iterations
                n_right += 1
            else:
                reward = 0.0
            rewards.append(reward)
        
        rewards_batch.append(rewards)
    
    return rewards_batch


def compute_sgrpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    skip_advantage: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for S-GRPO, using GRPO-style advantage estimation
    but WITHOUT dividing by standard deviation.
    
    This follows the Dr.GRPO style where advantages are not scaled by std.
    
    Args:
        token_level_rewards: (torch.Tensor)
            shape is (bs, response_length)
        response_mask: (torch.Tensor)
            shape is (bs, response_length)
        index: (np.ndarray)
            Index array for grouping (same query has same index)
        epsilon: (float)
            Small value to avoid numerical issues
        skip_advantage: (np.ndarray, optional)
            Boolean array indicating which samples should have advantage=0
            (e.g., samples with too-short responses that couldn't be truncated properly)
            
    Returns:
        advantages: (torch.Tensor)
            shape is (bs, response_length)
        returns: (torch.Tensor)
            shape is (bs, response_length)
    """
    # Sum token-level rewards to get sequence-level scores
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    
    id2score: Dict[Any, List[torch.Tensor]] = defaultdict(list)
    id2mean: Dict[Any, torch.Tensor] = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # Group scores by index
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # Compute mean for each group
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
            else:
                raise ValueError(f"No score in prompt index: {idx}")
        
        # Compute advantage: score - mean (no division by std)
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]
        
        # Broadcast to token level
        advantages = scores.unsqueeze(-1) * response_mask
        
        # Zero out advantages for samples that should be skipped
        if skip_advantage is not None:
            skip_mask = torch.tensor(skip_advantage, dtype=torch.bool, device=advantages.device)
            advantages[skip_mask] = 0.0
    
    return advantages, advantages


def compute_sgrpo_advantage_vectorized(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized version of S-GRPO advantage computation.
    
    Args:
        token_level_rewards: (torch.Tensor)
            shape is (bs, response_length)
        response_mask: (torch.Tensor)
            shape is (bs, response_length)
        index: (np.ndarray)
            Index array for grouping
        epsilon: (float)
            Small value for numerical stability
            
    Returns:
        advantages: (torch.Tensor)
            shape is (bs, response_length)
        returns: (torch.Tensor)
            shape is (bs, response_length)
    """
    from verl.utils import as_torch_index, group_mean_std
    
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)  # (bs,)
        g = as_torch_index(index, device=scores.device)
        
        # Get group mean (we don't need std for S-GRPO)
        mean_g, _, _ = group_mean_std(scores, g, eps=epsilon)
        
        # Advantage = score - group_mean (no normalization by std)
        scalars = scores - mean_g[g]
        
        # Broadcast to token level
        advantages = scalars.unsqueeze(-1) * response_mask
        
    return advantages, advantages


def register_sgrpo_advantage_estimator():
    """
    Register S-GRPO advantage estimator with verl's registry.
    """
    from verl.trainer.ppo.core_algos import register_adv_est
    
    @register_adv_est("sgrpo")
    def _compute_sgrpo_advantage(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index: np.ndarray,
        epsilon: float = 1e-6,
        norm_adv_by_std_in_grpo: bool = False,  # Always False for S-GRPO
        config: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """S-GRPO advantage estimator."""
        return compute_sgrpo_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
        )
    
    return _compute_sgrpo_advantage
