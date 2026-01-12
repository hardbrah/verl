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
S-GRPO Trainer with Ray-based single controller.

S-GRPO (Sampling-based GRPO with Decaying Reward) extends GRPO by:
1. Generating ONE complete response (CoT0) per query
2. Truncating CoT0 at m uniformly sampled positions to get CoT1, ..., CoTm
3. Forcing the model to answer from each truncated CoT with a special prompt
4. Using a decaying reward that encourages correct answers from shorter CoTs
5. Using GRPO advantage estimation without std normalization

Key difference from GRPO:
- GRPO: samples N complete responses per query, uses them as a group
- S-GRPO: samples 1 complete response, creates m truncations, uses m+1 sequences as a group
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.model import compute_position_id_with_mask
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.utils import Role
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from sgrpo.core_algos import compute_sgrpo_advantage, register_sgrpo_advantage_estimator
from sgrpo.data_processor import SGRPODataProcessor, create_sgrpo_uid_mapping
from sgrpo.decaying_reward import SGRPORewardManager, create_sgrpo_reward_manager


# Register the S-GRPO advantage estimator
register_sgrpo_advantage_estimator()


# Default prompt to force the model to stop thinking and output answer
DEFAULT_FORCE_ANSWER_PROMPT = "Time is limited, stop thinking and start answering.\n</think>\n\n"


class RaySGRPOTrainer(RayPPOTrainer):
    """
    S-GRPO Trainer that extends the PPO trainer with sampling-based
    chain-of-thought truncation and decaying rewards.
    
    Algorithm:
    For each query:
    1. Generate ONE complete response CoT0 with n tokens
    2. Uniformly sample m positions from [1, n] to get truncation points
    3. Create truncated sequences: CoT1, CoT2, ..., CoTm (lengths increasing)
    4. For each truncated CoT_i, append force_answer_prompt and generate answer
    5. Compute decaying rewards for all m+1 responses (CoT1...CoTm, CoT0)
    6. Use GRPO advantage estimation (without std normalization) on this group
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: Dict[Role, Any],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """Initialize the S-GRPO trainer."""
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        
        # S-GRPO specific configuration
        sgrpo_config = config.get("sgrpo", {})
        self.num_truncations = sgrpo_config.get("num_truncations", 4)
        self.force_answer_prompt = sgrpo_config.get(
            "force_answer_prompt",
            DEFAULT_FORCE_ANSWER_PROMPT
        )
        self.answer_max_tokens = sgrpo_config.get("answer_max_tokens", 256)
        self.min_truncation_ratio = sgrpo_config.get("min_truncation_ratio", 0.1)
        self.max_truncation_ratio = sgrpo_config.get("max_truncation_ratio", 0.9)
        
        # Tokenize the force answer prompt once
        self.force_answer_prompt_ids = tokenizer.encode(
            self.force_answer_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )[0]
        
        # Initialize S-GRPO reward manager
        self.sgrpo_reward_fn = create_sgrpo_reward_manager(
            config=config,
            tokenizer=tokenizer,
            num_truncations=self.num_truncations,
        )
    
    def _sample_truncation_positions(
        self, 
        response_length: int,
    ) -> Optional[List[int]]:
        """
        Randomly sample m truncation positions from the response (uniform, without replacement).
        
        Args:
            response_length: Total tokens in the complete response (n)
            
        Returns:
            Sorted list of m truncation positions (ascending order), 
            or None if response is too short to sample enough positions.
        """
        m = self.num_truncations
        
        # Calculate valid range for truncation
        min_pos = max(1, int(response_length * self.min_truncation_ratio))
        max_pos = min(response_length - 1, int(response_length * self.max_truncation_ratio))
        
        # Calculate number of available positions
        available_positions = max_pos - min_pos + 1 if max_pos >= min_pos else 0
        
        # If not enough positions available, return None to indicate skip
        if available_positions < m:
            return None
        
        # Random uniform sampling without replacement
        positions = sorted(np.random.choice(
            range(min_pos, max_pos + 1), 
            size=m, 
            replace=False
        ).tolist())
        
        return positions
    
    def _create_truncated_inputs_for_generation(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[int]], List[int], set]:
        """
        Create input sequences for generating answers from truncated CoTs.
        
        For each sample, creates m truncated sequences:
        input = prompt + truncated_response + force_answer_prompt
        
        Args:
            prompts: (batch_size, prompt_length)
            responses: (batch_size, response_length)
            attention_mask: (batch_size, total_length)
            response_mask: (batch_size, response_length)
            
        Returns:
            Tuple of:
            - Dictionary with padded input_ids, attention_mask, position_ids
            - List of truncation positions for each sample
            - List mapping each truncated sequence to its original sample index
            - Set of sample indices that should be skipped (too short to sample enough truncations)
        """
        batch_size = prompts.shape[0]
        prompt_length = prompts.shape[1]
        device = prompts.device
        
        force_prompt_ids = self.force_answer_prompt_ids.to(device)
        force_prompt_len = len(force_prompt_ids)
        
        all_input_ids = []
        all_truncation_positions = []
        sample_indices = []
        
        # Track which samples should be skipped due to too-short responses
        skip_samples = set()
        
        for i in range(batch_size):
            # Get valid prompt tokens
            prompt_mask = attention_mask[i, :prompt_length]
            valid_prompt = prompts[i][prompt_mask == 1]
            
            # Get valid response tokens
            valid_response_mask = response_mask[i]
            valid_response = responses[i][valid_response_mask == 1]
            response_length = len(valid_response)
            
            # Sample truncation positions (returns None if response too short)
            truncation_positions = self._sample_truncation_positions(response_length)
            
            if truncation_positions is None:
                # Response too short to sample enough truncation positions
                # Mark this sample to skip (advantage will be 0)
                skip_samples.add(i)
                # Use placeholder positions (will be filtered later)
                truncation_positions = [1] * self.num_truncations
            
            all_truncation_positions.append(truncation_positions)
            
            # Create m truncated sequences for this sample
            for pos in truncation_positions:
                # Truncate response at position pos
                truncated_response = valid_response[:pos]
                
                # Concatenate: prompt + truncated_response + force_prompt
                input_ids = torch.cat([
                    valid_prompt,
                    truncated_response,
                    force_prompt_ids,
                ])
                
                all_input_ids.append(input_ids)
                sample_indices.append(i)
        
        # Pad all sequences to same length
        max_length = max(len(ids) for ids in all_input_ids)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for input_ids in all_input_ids:
            padding_length = max_length - len(input_ids)
            
            # Left padding for generation
            padded_ids = torch.cat([
                torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype, device=device),
                input_ids,
            ])
            mask = torch.cat([
                torch.zeros(padding_length, dtype=torch.long, device=device),
                torch.ones(len(input_ids), dtype=torch.long, device=device),
            ])
            
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(mask)
        
        input_ids_batch = torch.stack(padded_input_ids)
        attention_mask_batch = torch.stack(padded_attention_mask)
        position_ids_batch = attention_mask_batch.cumsum(-1) - 1
        position_ids_batch.masked_fill_(attention_mask_batch == 0, 0)
        
        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "position_ids": position_ids_batch,
        }, all_truncation_positions, sample_indices, skip_samples
    
    def _generate_truncated_answers(
        self,
        batch: DataProto,
        timing_raw: Dict,
    ) -> Tuple[DataProto, Dict[str, Any]]:
        """
        Generate answers from truncated chain-of-thoughts.
        
        Process:
        1. For each complete response, sample m truncation positions
        2. Create m inputs: prompt + truncated_cot + force_prompt
        3. Generate answers for each truncated input
        
        Args:
            batch: DataProto containing prompts and complete responses
            timing_raw: Timing dictionary for profiling
            
        Returns:
            Tuple of (DataProto with generated answers, metadata)
        """
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        response_mask = batch.batch["response_mask"]
        
        with marked_timer("create_truncated_inputs", timing_raw, color="purple"):
            # Create truncated inputs for generation
            truncated_inputs, truncation_positions, sample_indices, skip_samples = \
                self._create_truncated_inputs_for_generation(
                    prompts=prompts,
                    responses=responses,
                    attention_mask=attention_mask,
                    response_mask=response_mask,
                )
        
        with marked_timer("generate_truncated_answers", timing_raw, color="orange"):
            # Create DataProto for generation
            gen_batch = DataProto.from_single_dict(truncated_inputs)
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": True,
                "do_sample": True,
                "max_new_tokens": self.answer_max_tokens,
                "global_steps": self.global_steps,
            }
            
            # Pad and generate
            # For async mode, we need to provide prompt_ids for the agent loop
            # This bypasses apply_chat_template to avoid double <|im_start|>assistant tokens
            if self.async_rollout_mode:
                # Extract valid prompt_ids directly (already contains proper chat template)
                prompt_ids_list = []
                input_ids_batch = truncated_inputs["input_ids"]
                attention_mask_batch = truncated_inputs["attention_mask"]
                for i in range(len(input_ids_batch)):
                    # Get valid tokens (non-padding)
                    input_ids = input_ids_batch[i]
                    attention_mask = attention_mask_batch[i]
                    valid_ids = input_ids[attention_mask == 1].tolist()
                    prompt_ids_list.append(valid_ids)
                
                # Pass prompt_ids directly to skip apply_chat_template in agent_loop
                gen_batch.non_tensor_batch["prompt_ids"] = np.array(prompt_ids_list, dtype=object)
                # raw_prompt is still needed for compatibility, but won't be used for tokenization
                gen_batch.non_tensor_batch["raw_prompt"] = np.array(
                    [[{"role": "user", "content": ""}]] * len(input_ids_batch), dtype=object
                )
                
                # Copy data_source and reward_model from original batch for reward_loop
                # Each truncated sample maps back to its original sample via sample_indices
                if "data_source" in batch.non_tensor_batch:
                    original_data_sources = batch.non_tensor_batch["data_source"]
                    truncated_data_sources = [original_data_sources[idx] for idx in sample_indices]
                    gen_batch.non_tensor_batch["data_source"] = np.array(truncated_data_sources, dtype=object)
                
                if "reward_model" in batch.non_tensor_batch:
                    original_reward_models = batch.non_tensor_batch["reward_model"]
                    truncated_reward_models = [original_reward_models[idx] for idx in sample_indices]
                    gen_batch.non_tensor_batch["reward_model"] = np.array(truncated_reward_models, dtype=object)
                
                size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            else:
                size_divisor = self.actor_rollout_wg.world_size
            
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
            
            if not self.async_rollout_mode:
                truncated_outputs = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
            else:
                truncated_outputs = self.async_rollout_manager.generate_sequences(gen_batch_padded)
            truncated_outputs = unpad_dataproto(truncated_outputs, pad_size=pad_size)
        
        metadata = {
            "truncation_positions": truncation_positions,
            "sample_indices": sample_indices,
            "num_truncations": self.num_truncations,
            "truncated_inputs": truncated_inputs,  # Save for logging
            "skip_samples": skip_samples,  # Samples to skip due to too-short responses
        }
        
        return truncated_outputs, metadata
    
    def _build_sgrpo_training_batch(
        self,
        original_batch: DataProto,
        truncated_outputs: DataProto,
        metadata: Dict[str, Any],
    ) -> DataProto:
        """
        Build the final training batch for S-GRPO.
        
        Organizes data as:
        [sample_0_cot_1, sample_0_cot_2, ..., sample_0_cot_m, sample_0_cot_0,
         sample_1_cot_1, sample_1_cot_2, ..., sample_1_cot_m, sample_1_cot_0,
         ...]
        
        Where:
        - cot_1 to cot_m: truncated CoTs with forced answers
        - cot_0: original complete response
        
        All sequences from the same sample share the same UID for group advantage.
        
        IMPORTANT: For truncated sequences, the response includes:
        - truncated_response (from original CoT)
        - force_answer_prompt
        - forced_answer (generated by model)
        This ensures response_length correctly reflects the full response length.
        
        Args:
            original_batch: Original batch with complete responses
            truncated_outputs: Generated answers from truncated CoTs
            metadata: Processing metadata
            
        Returns:
            Combined DataProto for training
        """
        batch_size = len(original_batch.batch["input_ids"])
        num_truncations = metadata["num_truncations"]
        sample_indices = metadata["sample_indices"]
        truncation_positions = metadata["truncation_positions"]  # List of lists: [[pos1, pos2, ...], ...]
        skip_samples = metadata.get("skip_samples", set())  # Samples with too-short responses
        
        device = original_batch.batch["input_ids"].device
        force_prompt_ids = self.force_answer_prompt_ids.to(device)
        
        # Get original UIDs
        original_uids = original_batch.non_tensor_batch.get(
            "uid",
            np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        )
        
        # We need to interleave: for each sample, m truncated + 1 original
        # Order: [trunc_1, trunc_2, ..., trunc_m, original] for each sample
        
        # Build lists for all data
        all_prompts = []
        all_responses = []
        all_response_masks = []
        all_uids = []
        all_reward_model_info = []
        all_data_sources = []
        all_skip_flags = []  # Track which sequences should have advantage=0
        
        truncated_idx = 0
        
        for sample_idx in range(batch_size):
            uid = original_uids[sample_idx]
            should_skip = sample_idx in skip_samples  # Mark if this sample should be skipped
            
            # Get original response for this sample (to extract truncated parts)
            original_response = original_batch.batch["responses"][sample_idx]
            original_response_mask = original_batch.batch["response_mask"][sample_idx]
            valid_original_response = original_response[original_response_mask == 1]
            
            # Get truncation positions for this sample
            sample_truncation_positions = truncation_positions[sample_idx]
            
            # Add m truncated sequences for this sample
            for trunc_idx in range(num_truncations):
                if truncated_idx < len(truncated_outputs.batch["responses"]):
                    # Get the forced answer (only the newly generated part)
                    forced_answer = truncated_outputs.batch["responses"][truncated_idx]
                    forced_answer_mask = truncated_outputs.batch["response_mask"][truncated_idx]
                    valid_forced_answer = forced_answer[forced_answer_mask == 1]
                    
                    # Get the truncated response from original
                    trunc_pos = sample_truncation_positions[trunc_idx]
                    truncated_response = valid_original_response[:trunc_pos]
                    
                    # Build complete response: truncated_response + force_prompt + forced_answer
                    complete_response = torch.cat([
                        truncated_response,
                        force_prompt_ids,
                        valid_forced_answer,
                    ])
                    
                    # Use the original prompt (not the truncated input prompt)
                    all_prompts.append(original_batch.batch["prompts"][sample_idx])
                    all_responses.append(complete_response)
                    
                    # Create response mask (all 1s for the complete response)
                    complete_response_mask = torch.ones(len(complete_response), dtype=torch.long, device=device)
                    all_response_masks.append(complete_response_mask)
                    
                    all_uids.append(uid)  # Same UID for grouping
                    all_skip_flags.append(should_skip)  # Mark for advantage=0
                    
                    # Copy reward model info from original sample
                    if "reward_model" in original_batch.non_tensor_batch:
                        rm_info = original_batch.non_tensor_batch["reward_model"]
                        all_reward_model_info.append(rm_info[sample_idx] if len(rm_info) > sample_idx else {})
                    
                    if "data_source" in original_batch.non_tensor_batch:
                        ds = original_batch.non_tensor_batch["data_source"]
                        all_data_sources.append(ds[sample_idx] if len(ds) > sample_idx else "unknown")
                    
                    truncated_idx += 1
            
            # Add original complete response for this sample
            all_prompts.append(original_batch.batch["prompts"][sample_idx])
            all_responses.append(original_batch.batch["responses"][sample_idx])
            
            if "response_mask" in original_batch.batch:
                all_response_masks.append(original_batch.batch["response_mask"][sample_idx])
            
            all_uids.append(uid)  # Same UID for grouping
            all_skip_flags.append(should_skip)  # Mark for advantage=0
            
            if "reward_model" in original_batch.non_tensor_batch:
                rm_info = original_batch.non_tensor_batch["reward_model"]
                all_reward_model_info.append(rm_info[sample_idx] if len(rm_info) > sample_idx else {})
            
            if "data_source" in original_batch.non_tensor_batch:
                ds = original_batch.non_tensor_batch["data_source"]
                all_data_sources.append(ds[sample_idx] if len(ds) > sample_idx else "unknown")
        
        # Pad all sequences to the same length
        max_prompt_len = max(p.shape[0] for p in all_prompts)
        max_response_len = max(r.shape[0] for r in all_responses)
        
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        def pad_tensor(tensor, max_len, pad_value=0):
            if len(tensor) >= max_len:
                return tensor[:max_len]
            padding = torch.full(
                (max_len - len(tensor),), 
                pad_value, 
                dtype=tensor.dtype, 
                device=tensor.device
            )
            return torch.cat([tensor, padding])
        
        padded_prompts = torch.stack([pad_tensor(p, max_prompt_len, pad_token_id) for p in all_prompts])
        padded_responses = torch.stack([pad_tensor(r, max_response_len, pad_token_id) for r in all_responses])
        
        if all_response_masks:
            padded_response_masks = torch.stack([pad_tensor(m, max_response_len, 0) for m in all_response_masks])
        else:
            # Compute response masks
            padded_response_masks = (padded_responses != pad_token_id).long()
        
        # Build input_ids by concatenating prompts and responses
        # input_ids = [prompts, responses] concatenated along sequence dimension
        input_ids = torch.cat([padded_prompts, padded_responses], dim=1)
        
        # Build attention_mask: 1 for non-padding tokens, 0 for padding tokens
        # For prompts: check if token is not padding
        prompt_attention_mask = (padded_prompts != pad_token_id).long()
        # For responses: use response_mask which already indicates valid tokens
        attention_mask = torch.cat([prompt_attention_mask, padded_response_masks], dim=1)
        
        # Build position_ids based on attention_mask using cumsum
        # This correctly handles padding by assigning position 0 to all padding tokens
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build the combined batch
        combined_batch_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "prompts": padded_prompts,
            "responses": padded_responses,
            "attention_mask": attention_mask,
            "response_mask": padded_response_masks,
        }
        
        combined_batch = DataProto.from_single_dict(combined_batch_dict)
        
        # Set non-tensor batch
        combined_batch.non_tensor_batch["uid"] = np.array(all_uids, dtype=object)
        
        if all_reward_model_info:
            combined_batch.non_tensor_batch["reward_model"] = np.array(all_reward_model_info, dtype=object)
        
        if all_data_sources:
            combined_batch.non_tensor_batch["data_source"] = np.array(all_data_sources, dtype=object)
        
        # Store skip flags for advantage computation
        # True means this sample should have advantage=0 (response too short)
        combined_batch.non_tensor_batch["skip_advantage"] = np.array(all_skip_flags, dtype=bool)
        
        return combined_batch
    
    def _compute_sgrpo_rewards(
        self,
        batch: DataProto,
        timing_raw: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, List]]:
        """
        Compute decaying rewards for S-GRPO.
        
        Args:
            batch: DataProto with all responses (truncated + original)
            timing_raw: Timing dictionary
            
        Returns:
            Tuple of (reward tensor, extra info dict)
        """
        with marked_timer("compute_decaying_reward", timing_raw, color="yellow"):
            result = self.sgrpo_reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            reward_extra_info = result.get("reward_extra_info", {})
        
        return reward_tensor, reward_extra_info
    
    def _log_sgrpo_samples(
        self,
        original_batch: DataProto,
        truncated_inputs: Dict[str, torch.Tensor],
        sgrpo_batch: DataProto,
        reward_extra_info: Dict[str, List],
        metadata: Dict[str, Any],
        dump_path: str,
    ):
        """
        Log S-GRPO samples to disk for debugging.
        
        Saves detailed information about:
        1. Original prompts
        2. Complete responses (CoT0)
        3. Truncated inputs (with force_answer_prompt)
        4. Generated answers from truncated CoTs
        5. Extracted answers and correctness
        6. Decaying rewards
        
        Args:
            original_batch: Original batch with complete responses
            truncated_inputs: Inputs used for truncated answer generation
            sgrpo_batch: Final combined batch for training
            reward_extra_info: Reward computation extra info
            metadata: S-GRPO processing metadata
            dump_path: Directory to save samples
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"sgrpo_samples_{self.global_steps}.jsonl")
        
        original_batch_size = len(original_batch.batch["input_ids"])
        num_truncations = metadata.get("num_truncations", self.num_truncations)
        num_seqs_per_sample = num_truncations + 1
        
        # Decode original prompts and complete responses
        original_prompts = self.tokenizer.batch_decode(
            original_batch.batch["prompts"], skip_special_tokens=True
        )
        complete_responses = self.tokenizer.batch_decode(
            original_batch.batch["responses"], skip_special_tokens=True
        )
        
        # Decode truncated inputs (prompt + truncated_cot + force_prompt)
        if truncated_inputs is not None and "input_ids" in truncated_inputs:
            truncated_input_texts = self.tokenizer.batch_decode(
                truncated_inputs["input_ids"], skip_special_tokens=False
            )
        else:
            truncated_input_texts = []
        
        # Decode all responses in sgrpo_batch
        all_responses = self.tokenizer.batch_decode(
            sgrpo_batch.batch["responses"], skip_special_tokens=True
        )
        
        # Get rewards and correctness
        correctness = reward_extra_info.get("correctness", [])
        decaying_rewards = reward_extra_info.get("decaying_rewards", [])
        extracted_answers = reward_extra_info.get("answers", [])
        
        # Get ground truths
        ground_truths = []
        if "reward_model" in original_batch.non_tensor_batch:
            rm_info = original_batch.non_tensor_batch["reward_model"]
            for i in range(original_batch_size):
                if i < len(rm_info) and isinstance(rm_info[i], dict):
                    ground_truths.append(rm_info[i].get("ground_truth", None))
                else:
                    ground_truths.append(None)
        
        lines = []
        truncated_idx = 0
        
        for sample_idx in range(original_batch_size):
            sample_entry = {
                "step": self.global_steps,
                "sample_idx": sample_idx,
                "original_prompt": original_prompts[sample_idx] if sample_idx < len(original_prompts) else "",
                "complete_response_cot0": complete_responses[sample_idx] if sample_idx < len(complete_responses) else "",
                "ground_truth": ground_truths[sample_idx] if sample_idx < len(ground_truths) else None,
                "truncations": [],
            }
            
            # Add m truncated sequences
            for trunc_idx in range(num_truncations):
                global_idx = sample_idx * num_seqs_per_sample + trunc_idx
                
                trunc_entry = {
                    "truncation_idx": trunc_idx + 1,  # 1-indexed for clarity
                    "type": f"cot_{trunc_idx + 1}",
                }
                
                # Truncated input (with force_answer_prompt)
                if truncated_idx < len(truncated_input_texts):
                    trunc_entry["truncated_input"] = truncated_input_texts[truncated_idx]
                    truncated_idx += 1
                
                # Generated response
                if global_idx < len(all_responses):
                    trunc_entry["response"] = all_responses[global_idx]
                
                # Extracted answer
                if global_idx < len(extracted_answers):
                    trunc_entry["extracted_answer"] = extracted_answers[global_idx]
                
                # Correctness and reward
                if global_idx < len(correctness):
                    trunc_entry["is_correct"] = correctness[global_idx]
                if global_idx < len(decaying_rewards):
                    trunc_entry["decaying_reward"] = decaying_rewards[global_idx]
                
                sample_entry["truncations"].append(trunc_entry)
            
            # Add original complete response (CoT0)
            global_idx = sample_idx * num_seqs_per_sample + num_truncations
            cot0_entry = {
                "truncation_idx": 0,  # 0 for original
                "type": "cot_0 (complete)",
                "response": complete_responses[sample_idx] if sample_idx < len(complete_responses) else "",
            }
            
            if global_idx < len(extracted_answers):
                cot0_entry["extracted_answer"] = extracted_answers[global_idx]
            if global_idx < len(correctness):
                cot0_entry["is_correct"] = correctness[global_idx]
            if global_idx < len(decaying_rewards):
                cot0_entry["decaying_reward"] = decaying_rewards[global_idx]
            
            sample_entry["truncations"].append(cot0_entry)
            
            lines.append(json.dumps(sample_entry, ensure_ascii=False))
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        
        print(f"Dumped S-GRPO samples to {filename}")
    
    def _log_sgrpo_metrics_to_wandb(
        self,
        sgrpo_batch: DataProto,
        reward_extra_info: Dict[str, List],
        num_samples_to_log: int = 5,
    ) -> Dict[str, Any]:
        """
        Prepare S-GRPO specific metrics and sample tables for wandb logging.
        
        Args:
            sgrpo_batch: The S-GRPO training batch
            reward_extra_info: Extra info from reward computation
            num_samples_to_log: Number of samples to include in tables
            
        Returns:
            Dictionary of metrics to log
        """
        metrics = {}
        
        # Decode responses for sample logging
        all_responses = self.tokenizer.batch_decode(
            sgrpo_batch.batch["responses"][:num_samples_to_log * (self.num_truncations + 1)],
            skip_special_tokens=True
        )
        
        correctness = reward_extra_info.get("correctness", [])
        decaying_rewards = reward_extra_info.get("decaying_rewards", [])
        extracted_answers = reward_extra_info.get("answers", [])
        
        # Compute per-truncation statistics
        num_seqs_per_sample = self.num_truncations + 1
        total_samples = len(sgrpo_batch.batch["responses"]) // num_seqs_per_sample
        
        # Accuracy per truncation position
        for trunc_idx in range(num_seqs_per_sample):
            trunc_correctness = []
            trunc_rewards = []
            
            for sample_idx in range(total_samples):
                global_idx = sample_idx * num_seqs_per_sample + trunc_idx
                if global_idx < len(correctness):
                    trunc_correctness.append(correctness[global_idx])
                if global_idx < len(decaying_rewards):
                    trunc_rewards.append(decaying_rewards[global_idx])
            
            if trunc_correctness:
                if trunc_idx < self.num_truncations:
                    metrics[f"sgrpo/accuracy_cot_{trunc_idx + 1}"] = sum(trunc_correctness) / len(trunc_correctness)
                    if trunc_rewards:
                        metrics[f"sgrpo/mean_reward_cot_{trunc_idx + 1}"] = np.mean(trunc_rewards)
                else:
                    metrics["sgrpo/accuracy_cot_0_complete"] = sum(trunc_correctness) / len(trunc_correctness)
                    if trunc_rewards:
                        metrics["sgrpo/mean_reward_cot_0_complete"] = np.mean(trunc_rewards)
        
        return metrics
    
    def fit(self):
        """
        The training loop of S-GRPO.
        
        For each query:
        1. Generate ONE complete response (CoT0)
        2. Sample m truncation positions uniformly
        3. Create truncated CoTs and generate forced answers
        4. Compute decaying rewards for all m+1 responses
        5. Compute S-GRPO advantages (GRPO without std normalization)
        6. Update the model
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        self._load_checkpoint()
        
        current_epoch = self.global_steps // len(self.train_dataloader)
        
        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="S-GRPO Training"
        )
        
        self.global_steps += 1
        last_val_metrics = None
        
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                
                # Add UIDs - one per original query
                original_batch_size = len(batch.batch["input_ids"])
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(original_batch_size)],
                    dtype=object,
                )
                
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                
                is_last_step = self.global_steps >= self.total_training_steps
                
                with marked_timer("step", timing_raw):
                    # Step 1: Generate ONE complete response per query (CoT0)
                    with marked_timer("gen_complete", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        if "timing" in gen_batch_output.meta_info:
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)
                    
                    # Combine with original batch (now has complete responses)
                    batch = batch.union(gen_batch_output)
                    
                    # Ensure response_mask exists
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # Step 2 & 3: Create truncated CoTs and generate forced answers
                    truncated_outputs, sgrpo_metadata = self._generate_truncated_answers(
                        batch=batch,
                        timing_raw=timing_raw,
                    )
                    
                    # Step 4: Build combined training batch
                    # Order: [cot_1, cot_2, ..., cot_m, cot_0] for each sample
                    with marked_timer("build_sgrpo_batch", timing_raw, color="cyan"):
                        sgrpo_batch = self._build_sgrpo_training_batch(
                            original_batch=batch,
                            truncated_outputs=truncated_outputs,
                            metadata=sgrpo_metadata,
                        )
                    
                    # Compute global token info
                    sgrpo_batch.meta_info["global_token_num"] = torch.sum(
                        sgrpo_batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    
                    # Step 5: Compute decaying rewards
                    reward_tensor, reward_extra_info = self._compute_sgrpo_rewards(
                        batch=sgrpo_batch,
                        timing_raw=timing_raw,
                    )
                    sgrpo_batch.batch["token_level_scores"] = reward_tensor
                    sgrpo_batch.batch["token_level_rewards"] = reward_tensor
                    
                    if reward_extra_info:
                        sgrpo_batch.non_tensor_batch.update({
                            k: np.array(v) for k, v in reward_extra_info.items()
                        })
                    
                    # Step 6: Compute log probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(sgrpo_batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = sgrpo_batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        metrics["actor/entropy"] = entropy_agg.detach().item()
                        old_log_prob.batch.pop("entropys")
                        sgrpo_batch = sgrpo_batch.union(old_log_prob)
                    
                    # Reference policy log probs if needed
                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(sgrpo_batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(sgrpo_batch)
                            sgrpo_batch = sgrpo_batch.union(ref_log_prob)
                    
                    # Step 7: Compute S-GRPO advantages
                    # All m+1 sequences with same UID form a group
                    # Samples with too-short responses have advantage=0
                    with marked_timer("sgrpo_adv", timing_raw, color="brown"):
                        skip_advantage = sgrpo_batch.non_tensor_batch.get("skip_advantage", None)
                        advantages, returns = compute_sgrpo_advantage(
                            token_level_rewards=sgrpo_batch.batch["token_level_rewards"],
                            response_mask=sgrpo_batch.batch["response_mask"],
                            index=sgrpo_batch.non_tensor_batch["uid"],
                            skip_advantage=skip_advantage,
                        )
                        sgrpo_batch.batch["advantages"] = advantages
                        sgrpo_batch.batch["returns"] = returns
                    
                    # Step 8: Update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            sgrpo_batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            sgrpo_batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(sgrpo_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                
                # Validation
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                
                # Checkpoint saving
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                
                # Log S-GRPO samples if rollout_data_dir is configured
                sgrpo_data_dir = self.config.trainer.get("rollout_data_dir", None)
                rollout_data_dump_freq = self.config.trainer.get("rollout_data_dump_freq", -1)
                should_dump_samples = (
                    sgrpo_data_dir is not None 
                    and rollout_data_dump_freq > 0 
                    and self.global_steps % rollout_data_dump_freq == 0
                )
                if should_dump_samples:
                    with marked_timer("dump_sgrpo_samples", timing_raw, color="green"):
                        # Get truncated inputs from metadata if available
                        truncated_inputs = sgrpo_metadata.get("truncated_inputs", None)
                        # Add experiment_name as subdirectory to avoid overwriting
                        experiment_name = self.config.trainer.get("experiment_name", "default")
                        sample_dump_path = os.path.join(sgrpo_data_dir, experiment_name)
                        self._log_sgrpo_samples(
                            original_batch=batch,
                            truncated_inputs=truncated_inputs,
                            sgrpo_batch=sgrpo_batch,
                            reward_extra_info=reward_extra_info,
                            metadata=sgrpo_metadata,
                            dump_path=sample_dump_path,
                        )
                
                # Collect metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                    "sgrpo/num_truncations": self.num_truncations,
                    "sgrpo/original_batch_size": original_batch_size,
                    "sgrpo/expanded_batch_size": len(sgrpo_batch.batch["responses"]),
                })
                
                # S-GRPO specific metrics
                if "correctness" in reward_extra_info:
                    correctness = reward_extra_info["correctness"]
                    metrics["sgrpo/accuracy"] = sum(correctness) / len(correctness) if correctness else 0
                
                if "decaying_rewards" in reward_extra_info:
                    rewards = reward_extra_info["decaying_rewards"]
                    metrics["sgrpo/mean_decaying_reward"] = np.mean(rewards) if rewards else 0
                
                # Add per-truncation metrics for detailed tracking
                wandb_metrics = self._log_sgrpo_metrics_to_wandb(
                    sgrpo_batch=sgrpo_batch,
                    reward_extra_info=reward_extra_info,
                )
                metrics.update(wandb_metrics)
                
                metrics.update(compute_data_metrics(batch=sgrpo_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=sgrpo_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=sgrpo_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                logger.log(data=metrics, step=self.global_steps)
                
                progress_bar.update(1)
                self.global_steps += 1
                
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
