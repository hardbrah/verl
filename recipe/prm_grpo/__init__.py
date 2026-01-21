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
PRM-GRPO: GRPO with Process Reward Model using Latent States.

This recipe replaces the outcome-based reward in GRPO with a Process Reward Model
that scores truncated thinking chains using latent states extracted from the Actor.

Architecture:
1. Actor model forward pass extracts latent states from layer 15
2. Latent states are injected into PRM backbone's layer 0 (as inputs_embeds)
3. PRM backbone + regression head outputs score in [0, 1]

Key components:
- LatentExtractor: Extracts hidden states from specified layer
- PRMBackbone: Processes latent states and outputs score
- PRMScorer: High-level scorer for easy integration
- PRMRewardManager: Reward manager for verl training loop
- PRMGRPOStepLogger: Logs detailed step data to JSON files

Reference:
- /data/chenhaotian/latentqa/math_sheperd/preprocess_latent_states.py
- /data/chenhaotian/latentqa/math_sheperd/train_math_shepherd_step1_scorer.py
"""

from .prm_model import (
    LatentExtractor,
    PRMBackbone,
    ProcessRewardModel,
    PRMScorer,
    RegressionHead,
    load_prm_from_checkpoint,
)
from .prm_reward_manager import PRMRewardManager, PRMGRPOStepLogger

__all__ = [
    # Model components
    "LatentExtractor",
    "PRMBackbone",
    "ProcessRewardModel",
    "PRMScorer",
    "RegressionHead",
    "load_prm_from_checkpoint",
    # Reward manager
    "PRMRewardManager",
    "PRMGRPOStepLogger",
]
