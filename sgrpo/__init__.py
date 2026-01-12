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
S-GRPO: Sampling-based GRPO with Decaying Reward

This module implements the S-GRPO algorithm which extends GRPO with:
1. Chain-of-thought truncation sampling
2. Forced answer generation from truncated thoughts
3. Decaying reward to encourage shorter correct solutions
4. GRPO-style advantage estimation without std normalization
"""

__all__ = [
    "compute_sgrpo_advantage",
    "compute_decaying_reward",
    "compute_decaying_reward_batch",
    "register_sgrpo_advantage_estimator",
    "SGRPODataProcessor",
    "create_sgrpo_uid_mapping",
    "DecayingRewardManager",
    "SGRPORewardManager",
    "create_sgrpo_reward_manager",
    "RaySGRPOTrainer",
    "run_sgrpo",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in ("compute_sgrpo_advantage", "compute_decaying_reward", 
                "compute_decaying_reward_batch", "register_sgrpo_advantage_estimator"):
        from sgrpo.core_algos import (
            compute_sgrpo_advantage,
            compute_decaying_reward,
            compute_decaying_reward_batch,
            register_sgrpo_advantage_estimator,
        )
        return locals()[name]
    
    if name in ("SGRPODataProcessor", "create_sgrpo_uid_mapping"):
        from sgrpo.data_processor import SGRPODataProcessor, create_sgrpo_uid_mapping
        return locals()[name]
    
    if name in ("DecayingRewardManager", "SGRPORewardManager", "create_sgrpo_reward_manager"):
        from sgrpo.decaying_reward import (
            DecayingRewardManager,
            SGRPORewardManager,
            create_sgrpo_reward_manager,
        )
        return locals()[name]
    
    if name == "RaySGRPOTrainer":
        from sgrpo.ray_trainer import RaySGRPOTrainer
        return RaySGRPOTrainer
    
    if name == "run_sgrpo":
        from sgrpo.main_sgrpo import run_sgrpo
        return run_sgrpo
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
