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
Data processor for S-GRPO.
Handles truncation of chain-of-thought responses and forced answer generation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from dataclasses import dataclass


# Default prompt to force the model to stop thinking and output answer
DEFAULT_FORCE_ANSWER_PROMPT = "Time is limited, stop thinking and start answering.\n</think>\n\n"


@dataclass
class TruncatedCoT:
    """Represents a truncated chain of thought."""
    # Original query (prompt) tokens
    prompt_ids: torch.Tensor
    # Truncated response tokens
    truncated_response_ids: torch.Tensor
    # Position where truncation occurred
    truncation_position: int
    # Original full response length
    original_length: int
    # Index of this truncated CoT (0 to m-1 for truncated, m for original)
    truncation_index: int
    # Original sample index in the batch
    sample_index: int


class SGRPODataProcessor:
    """
    Data processor for S-GRPO algorithm.
    
    Handles:
    1. Sampling truncation points from a complete response
    2. Creating truncated chain-of-thought sequences
    3. Appending force-answer prompts
    4. Extracting answers from generated responses
    """
    
    def __init__(
        self,
        tokenizer,
        num_truncations: int = 4,  # m: number of truncation points
        force_answer_prompt: str = DEFAULT_FORCE_ANSWER_PROMPT,
        min_truncation_ratio: float = 0.1,  # Minimum ratio of response to keep
        max_truncation_ratio: float = 0.9,  # Maximum ratio of response to keep
        answer_max_tokens: int = 256,  # Max tokens for forced answer generation
    ):
        """
        Initialize the S-GRPO data processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            num_truncations: Number of truncation points (m)
            force_answer_prompt: Prompt to force model to output answer
            min_truncation_ratio: Minimum ratio of response length to keep
            max_truncation_ratio: Maximum ratio of response length to keep
            answer_max_tokens: Maximum tokens for answer generation
        """
        self.tokenizer = tokenizer
        self.num_truncations = num_truncations
        self.force_answer_prompt = force_answer_prompt
        self.min_truncation_ratio = min_truncation_ratio
        self.max_truncation_ratio = max_truncation_ratio
        self.answer_max_tokens = answer_max_tokens
        
        # Tokenize the force answer prompt
        self.force_answer_prompt_ids = tokenizer.encode(
            force_answer_prompt, 
            add_special_tokens=False,
            return_tensors="pt"
        )[0]
    
    def sample_truncation_points(
        self, 
        response_length: int,
        num_points: Optional[int] = None,
    ) -> List[int]:
        """
        Sample truncation points uniformly from the response.
        
        Args:
            response_length: Total number of tokens in the response (n)
            num_points: Number of points to sample (default: self.num_truncations)
            
        Returns:
            Sorted list of truncation positions (from small to large)
        """
        if num_points is None:
            num_points = self.num_truncations
        
        # Calculate valid range for truncation
        min_pos = max(1, int(response_length * self.min_truncation_ratio))
        max_pos = min(response_length - 1, int(response_length * self.max_truncation_ratio))
        
        if max_pos <= min_pos:
            # If response is too short, use uniform spacing
            positions = list(range(1, min(response_length, num_points + 1)))
        else:
            # Uniform sampling from [min_pos, max_pos]
            positions = np.linspace(min_pos, max_pos, num_points, dtype=int).tolist()
            # Remove duplicates and sort
            positions = sorted(set(positions))
        
        return positions
    
    def create_truncated_sequences(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        sample_index: int,
        truncation_positions: Optional[List[int]] = None,
    ) -> List[TruncatedCoT]:
        """
        Create truncated chain-of-thought sequences from a full response.
        
        Args:
            prompt_ids: Token IDs for the prompt
            response_ids: Token IDs for the full response (CoT0)
            sample_index: Index of this sample in the original batch
            truncation_positions: Optional pre-computed truncation positions
            
        Returns:
            List of TruncatedCoT objects, ordered from shortest to longest,
            with the original (full) response at the end
        """
        response_length = len(response_ids)
        
        if truncation_positions is None:
            truncation_positions = self.sample_truncation_points(response_length)
        
        truncated_sequences = []
        
        # Create truncated sequences (CoT1, CoT2, ..., CoTm)
        for i, pos in enumerate(truncation_positions):
            truncated = TruncatedCoT(
                prompt_ids=prompt_ids,
                truncated_response_ids=response_ids[:pos],
                truncation_position=pos,
                original_length=response_length,
                truncation_index=i,
                sample_index=sample_index,
            )
            truncated_sequences.append(truncated)
        
        # Add the original full response (CoT0) at the end
        original = TruncatedCoT(
            prompt_ids=prompt_ids,
            truncated_response_ids=response_ids,
            truncation_position=response_length,
            original_length=response_length,
            truncation_index=len(truncation_positions),  # m
            sample_index=sample_index,
        )
        truncated_sequences.append(original)
        
        return truncated_sequences
    
    def prepare_force_answer_inputs(
        self,
        truncated_cots: List[TruncatedCoT],
        pad_token_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for forced answer generation.
        
        Concatenates: prompt + truncated_response + force_answer_prompt
        
        Args:
            truncated_cots: List of truncated CoT sequences
            pad_token_id: Token ID for padding
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', 'position_ids'
        """
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        all_input_ids = []
        max_length = 0
        
        # Concatenate sequences
        for cot in truncated_cots:
            input_ids = torch.cat([
                cot.prompt_ids,
                cot.truncated_response_ids,
                self.force_answer_prompt_ids.to(cot.prompt_ids.device),
            ])
            all_input_ids.append(input_ids)
            max_length = max(max_length, len(input_ids))
        
        # Pad sequences
        padded_input_ids = []
        attention_masks = []
        
        for input_ids in all_input_ids:
            padding_length = max_length - len(input_ids)
            
            # Left padding (for generation)
            padded = torch.cat([
                torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype, device=input_ids.device),
                input_ids,
            ])
            mask = torch.cat([
                torch.zeros(padding_length, dtype=torch.long, device=input_ids.device),
                torch.ones(len(input_ids), dtype=torch.long, device=input_ids.device),
            ])
            
            padded_input_ids.append(padded)
            attention_masks.append(mask)
        
        # Stack into batch
        input_ids_batch = torch.stack(padded_input_ids)
        attention_mask_batch = torch.stack(attention_masks)
        position_ids_batch = attention_mask_batch.cumsum(-1) - 1
        position_ids_batch.masked_fill_(attention_mask_batch == 0, 0)
        
        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "position_ids": position_ids_batch,
        }
    
    def extract_answer(
        self,
        full_response: str,
        answer_pattern: Optional[str] = None,
    ) -> str:
        """
        Extract the final answer from a response.
        
        This is a simple extraction that looks for common answer patterns.
        Can be overridden for task-specific answer extraction.
        
        Args:
            full_response: The full generated response
            answer_pattern: Optional regex pattern for answer extraction
            
        Returns:
            Extracted answer string
        """
        import re
        
        # Try common answer patterns
        patterns = [
            r"\\boxed{([^}]+)}",  # LaTeX boxed answer
            r"The answer is[:\s]+(.+?)(?:\.|$)",  # "The answer is X"
            r"Answer[:\s]+(.+?)(?:\.|$)",  # "Answer: X"
            r"= ([^=\n]+)$",  # Final equation result
        ]
        
        if answer_pattern:
            patterns.insert(0, answer_pattern)
        
        for pattern in patterns:
            match = re.search(pattern, full_response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return the last line or last few words
        lines = full_response.strip().split('\n')
        if lines:
            return lines[-1].strip()
        
        return full_response.strip()
    
    def process_batch(
        self,
        batch: "DataProto",
    ) -> Tuple["DataProto", Dict[str, Any]]:
        """
        Process a batch for S-GRPO training.
        
        This method:
        1. Takes a batch with complete responses
        2. Creates truncated versions of each response
        3. Prepares inputs for forced answer generation
        
        Args:
            batch: DataProto containing prompts and complete responses
            
        Returns:
            Tuple of (processed_batch, metadata)
        """
        from verl import DataProto
        
        prompts = batch.batch["prompts"]  # (batch_size, prompt_length)
        responses = batch.batch["responses"]  # (batch_size, response_length)
        
        batch_size = prompts.shape[0]
        
        all_truncated = []
        sample_to_truncations = {}  # Map sample index to its truncated sequences
        
        for i in range(batch_size):
            prompt_ids = prompts[i][batch.batch["attention_mask"][i, :prompts.shape[1]] == 1]
            response_ids = responses[i][batch.batch["response_mask"][i] == 1]
            
            # Create truncated sequences
            truncated_seqs = self.create_truncated_sequences(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                sample_index=i,
            )
            
            sample_to_truncations[i] = truncated_seqs
            all_truncated.extend(truncated_seqs)
        
        # Prepare inputs for forced answer generation
        force_answer_inputs = self.prepare_force_answer_inputs(all_truncated)
        
        metadata = {
            "original_batch_size": batch_size,
            "num_truncations": self.num_truncations,
            "truncated_sequences": all_truncated,
            "sample_to_truncations": sample_to_truncations,
            "force_answer_prompt": self.force_answer_prompt,
        }
        
        # Create new batch for forced answer generation
        processed_batch = DataProto.from_single_dict({
            "input_ids": force_answer_inputs["input_ids"],
            "attention_mask": force_answer_inputs["attention_mask"],
            "position_ids": force_answer_inputs["position_ids"],
        })
        
        return processed_batch, metadata


def create_sgrpo_uid_mapping(
    original_uids: np.ndarray,
    num_truncations: int,
) -> np.ndarray:
    """
    Create UID mapping for S-GRPO where all truncated sequences from the same
    sample share the same UID (for advantage computation).
    
    Args:
        original_uids: Original UIDs for each sample
        num_truncations: Number of truncations per sample (m)
        
    Returns:
        Extended UIDs where each original UID is repeated (m+1) times
    """
    # Each sample generates (m+1) sequences: m truncated + 1 original
    extended_uids = np.repeat(original_uids, num_truncations + 1)
    return extended_uids


def reorder_rewards_for_sgrpo(
    rewards: torch.Tensor,
    num_samples: int,
    num_truncations: int,
) -> torch.Tensor:
    """
    Reorder rewards to match S-GRPO's expected order.
    
    S-GRPO expects rewards in order: [CoT1, CoT2, ..., CoTm, CoT0] for each sample.
    
    Args:
        rewards: Tensor of rewards in generation order
        num_samples: Number of original samples
        num_truncations: Number of truncations per sample (m)
        
    Returns:
        Reordered rewards tensor
    """
    # Rewards should already be in the correct order if generation was done correctly
    # This function is provided for cases where reordering is needed
    return rewards
