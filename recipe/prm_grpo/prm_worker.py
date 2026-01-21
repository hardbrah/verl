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
PRM Reward Model Worker for PRM-GRPO.

This worker loads a Process Reward Model (PRM) and computes rewards based on
latent states extracted from layer 15 of the backbone model.

Architecture:
1. Extract latent states from layer 15 of the backbone model
2. Feed latent states as inputs_embeds to the PRM backbone (starting from layer 0)
3. Apply regression head to get the final score (0-1)
"""

import datetime
import os
import warnings
from typing import Optional

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.fs import copy_to_local
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.utils.config import omega_conf_to_dataclass

device_name = get_device_name()


class RegressionHead(nn.Module):
    """Regression head for PRM scoring."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return torch.sigmoid(hidden_states)


class LatentExtractor(nn.Module):
    """Extracts latent states from a specified layer of a transformer model."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        extract_layer: int = 15,
    ):
        super().__init__()
        self.model = model
        self.extract_layer = extract_layer
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract latent states from the specified layer.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            latent_states: [batch_size, seq_len, hidden_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        
        # Get hidden states from the specified layer
        # hidden_states is a tuple of (embedding_output, layer_1_output, ..., layer_n_output)
        # So layer 15 output is at index 16 (0-indexed: embedding at 0, layer 0 at 1, ...)
        latent_states = outputs.hidden_states[self.extract_layer + 1]
        
        return latent_states


class PRMBackbone(nn.Module):
    """
    PRM backbone that takes latent states as inputs_embeds and outputs scores.
    
    This model:
    1. Takes latent states (from layer 15 of actor) as inputs_embeds
    2. Processes them through the transformer layers (starting from layer 0)
    3. Applies a regression head to get the final score
    """
    
    def __init__(
        self,
        model_path: str,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        
        # Load the backbone model
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        )
        
        # Remove the LM head if present (we use regression head instead)
        if hasattr(self.backbone, 'lm_head'):
            del self.backbone.lm_head
            
        # Add regression head
        self.regression_head = RegressionHead(hidden_size).to(dtype)
        
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with latent states as inputs_embeds.
        
        Args:
            inputs_embeds: [batch_size, seq_len, hidden_size] - latent states from layer 15
            attention_mask: [batch_size, seq_len]
            
        Returns:
            scores: [batch_size] - PRM scores (0-1)
        """
        # Use the model's inner model (without lm_head) with inputs_embeds
        # This injects the latent states at layer 0
        outputs = self.backbone.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Pool: get the last valid token's hidden state
        batch_size = hidden_states.shape[0]
        sequence_lengths = attention_mask.sum(dim=-1) - 1  # Get index of last valid token
        sequence_lengths = sequence_lengths.clamp(min=0)
        
        # Gather the last valid token's hidden state for each sample
        pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        # Apply regression head
        scores = self.regression_head(pooled)  # [batch_size, 1]
        
        return scores.squeeze(-1)  # [batch_size]


class PRMRewardModelWorker(Worker, DistProfilerExtension):
    """
    Process Reward Model Worker for PRM-GRPO.
    
    This worker:
    1. Loads a LatentExtractor (for extracting latent states from layer 15)
    2. Loads a PRMBackbone (for scoring latent states)
    3. Computes PRM scores for input sequences
    """
    
    def __init__(self, config: DictConfig):
        Worker.__init__(self)
        
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )
        
        self.config = config
        
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        
        # Note: world_size is a property from Worker base class, don't reassign
        self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)
        
        # PRM-specific config
        self.prm_checkpoint_path = config.prm.checkpoint_path
        self.prm_backbone_path = config.prm.backbone_path
        self.prm_extract_layer = config.prm.get("extract_layer", 15)
        self.prm_max_length = config.prm.get("max_length", 256)
        
        # Normalize batch size config
        if config.get("micro_batch_size") is not None:
            config.micro_batch_size //= torch.distributed.get_world_size()
            config.micro_batch_size_per_gpu = config.micro_batch_size
            
    def _build_model(self, config: DictConfig):
        """Build the PRM model components."""
        print(f"[PRMRewardModelWorker] Building PRM model on rank {self.rank}...")
        print(f"  - PRM checkpoint: {self.prm_checkpoint_path}")
        print(f"  - PRM backbone: {self.prm_backbone_path}")
        print(f"  - Extract layer: {self.prm_extract_layer}")
        
        trust_remote_code = config.model.get("trust_remote_code", False)
        dtype = torch.bfloat16
        
        # Download model to local if needed
        local_backbone_path = copy_to_local(self.prm_backbone_path, use_shm=config.model.get("use_shm", False))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_backbone_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model config
        model_config = AutoConfig.from_pretrained(
            local_backbone_path,
            trust_remote_code=trust_remote_code,
        )
        hidden_size = model_config.hidden_size
        
        # Build LatentExtractor (for extracting latent states from layer 15)
        print(f"[PRMRewardModelWorker] Loading LatentExtractor model...")
        extractor_model = AutoModelForCausalLM.from_pretrained(
            local_backbone_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        )
        self.latent_extractor = LatentExtractor(
            model=extractor_model,
            extract_layer=self.prm_extract_layer,
        ).to(get_device_id())
        
        # Build PRMBackbone
        print(f"[PRMRewardModelWorker] Loading PRMBackbone model...")
        self.prm_backbone = PRMBackbone(
            model_path=local_backbone_path,
            hidden_size=hidden_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(get_device_id())
        
        # Load PRM checkpoint (regression head weights)
        print(f"[PRMRewardModelWorker] Loading PRM checkpoint: {self.prm_checkpoint_path}")
        checkpoint = torch.load(self.prm_checkpoint_path, map_location="cpu", weights_only=False)
        
        # Debug: Print checkpoint structure
        print(f"[PRMRewardModelWorker] Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"[PRMRewardModelWorker] Checkpoint keys: {list(checkpoint.keys())}")
        
        # Load state dict (only regression head and backbone weights that are in checkpoint)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print(f"[PRMRewardModelWorker] Using 'model_state_dict' key")
        else:
            state_dict = checkpoint
            print(f"[PRMRewardModelWorker] Using checkpoint directly as state_dict")
        
        # Debug: Print all state dict keys
        print(f"[PRMRewardModelWorker] State dict keys (first 20): {list(state_dict.keys())[:20]}")
        print(f"[PRMRewardModelWorker] Total keys in state dict: {len(state_dict)}")
            
        # Try to load weights for regression head
        regression_head_state = {}
        backbone_state = {}
        for key, value in state_dict.items():
            if "regression_head" in key:
                # Remove prefix if present
                new_key = key.replace("regression_head.", "")
                regression_head_state[new_key] = value
                print(f"[PRMRewardModelWorker] Found regression_head key: {key} -> {new_key}")
            elif "backbone" in key:
                # Store backbone weights for potential loading
                backbone_state[key] = value
                
        if regression_head_state:
            self.prm_backbone.regression_head.load_state_dict(regression_head_state, strict=False)
            print(f"[PRMRewardModelWorker] Loaded regression head weights: {list(regression_head_state.keys())}")
        else:
            print(f"[PRMRewardModelWorker] WARNING: No regression head weights found in checkpoint!")
            print(f"[PRMRewardModelWorker] Looking for keys containing: 'head', 'scorer', 'output', 'value'...")
            for key in state_dict.keys():
                if any(k in key.lower() for k in ['head', 'scorer', 'output', 'value', 'proj']):
                    print(f"[PRMRewardModelWorker]   Potential key: {key}")
            
        # Set to eval mode
        self.latent_extractor.eval()
        self.prm_backbone.eval()
        
        # Debug: Print regression head weights statistics
        print(f"[PRMRewardModelWorker] Regression head weights:")
        for name, param in self.prm_backbone.regression_head.named_parameters():
            print(f"  - {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
        
        print(f"[PRMRewardModelWorker] Model built successfully on rank {self.rank}")
        print(f"[PRMRewardModelWorker] Model config: hidden_size={hidden_size}, extract_layer={self.prm_extract_layer}, max_length={self.prm_max_length}")
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the PRM model."""
        self._build_model(config=self.config)
        
    def _forward_micro_batch(self, micro_batch: dict, debug: bool = False) -> torch.Tensor:
        """
        Forward pass for a micro batch.
        
        Args:
            micro_batch: Dictionary containing input_ids, attention_mask, position_ids
            debug: Whether to print debug information
            
        Returns:
            PRM scores for the micro batch
        """
        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            
            if debug:
                print(f"[PRMRewardModelWorker] Input shape: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
            
            # Truncate to max length if needed
            if input_ids.shape[1] > self.prm_max_length:
                input_ids = input_ids[:, :self.prm_max_length]
                attention_mask = attention_mask[:, :self.prm_max_length]
                if debug:
                    print(f"[PRMRewardModelWorker] Truncated to max_length={self.prm_max_length}")
                
            # Step 1: Extract latent states from layer 15
            latent_states = self.latent_extractor(input_ids, attention_mask)
            
            if debug:
                print(f"[PRMRewardModelWorker] Latent states shape: {latent_states.shape}")
                print(f"[PRMRewardModelWorker] Latent states stats - mean: {latent_states.mean():.4f}, "
                      f"std: {latent_states.std():.4f}, min: {latent_states.min():.4f}, max: {latent_states.max():.4f}")
                # Check if latent states are all the same (potential issue)
                if latent_states.std() < 1e-6:
                    print(f"[PRMRewardModelWorker] WARNING: Latent states have very low variance!")
            
            # Step 2: Feed latent states to PRM backbone and get scores
            scores = self.prm_backbone(latent_states, attention_mask)
            
            if debug:
                print(f"[PRMRewardModelWorker] Scores shape: {scores.shape}")
                print(f"[PRMRewardModelWorker] Scores: {scores[:5].tolist()}")  # Print first 5 scores
            
            return scores
            
    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor) -> torch.Tensor:
        """
        Expand scalar scores to token-level rewards.
        
        The reward is placed at the last valid token position.
        """
        batch_size = data.batch.batch_size[0]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
            
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores
        
        # Select only the response part
        token_level_scores = token_level_scores[:, -response_length:]
        
        return token_level_scores
        
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """
        Compute PRM scores for a batch of data.
        
        Args:
            data: DataProto containing input_ids, attention_mask, etc.
            
        Returns:
            DataProto containing rm_scores
        """
        # Track call count for debugging
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        
        # Enable debug for first few calls
        debug = (self._call_count <= 2 and self.rank == 0)
        
        if debug:
            print(f"\n[PRMRewardModelWorker] ========== compute_rm_score call #{self._call_count} ==========")
            print(f"[PRMRewardModelWorker] Data batch keys: {list(data.batch.keys())}")
        
        # Move data to device
        data = data.to(get_device_id())
        
        # Get input data
        rm_input_ids = data.batch["input_ids"]
        rm_attention_mask = data.batch["attention_mask"]
        
        if debug:
            print(f"[PRMRewardModelWorker] Total batch - input_ids: {rm_input_ids.shape}, attention_mask: {rm_attention_mask.shape}")
            # Decode first sample to see what we're scoring
            if hasattr(self, 'tokenizer'):
                first_sample = self.tokenizer.decode(rm_input_ids[0], skip_special_tokens=True)
                print(f"[PRMRewardModelWorker] First sample (truncated): {first_sample[:200]}...")
        
        # Split into micro batches and process
        micro_batch_size = self.config.get("micro_batch_size_per_gpu", 8)
        batch_size = rm_input_ids.shape[0]
        
        all_scores = []
        for i in range(0, batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, batch_size)
            micro_batch = {
                "input_ids": rm_input_ids[i:end_idx],
                "attention_mask": rm_attention_mask[i:end_idx],
            }
            # Only debug first micro batch
            scores = self._forward_micro_batch(micro_batch, debug=(debug and i == 0))
            all_scores.append(scores)
            
        scores = torch.cat(all_scores, dim=0)  # [batch_size]
        
        # Expand to token level
        token_level_scores = self._expand_to_token_level(data, scores)
        
        # Create output
        output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
        output = output.to("cpu")
        
        # Print some samples for debugging
        if self.rank == 0:
            print(f"[PRMRewardModelWorker] PRM scores - mean: {scores.mean():.4f}, "
                  f"std: {scores.std():.4f}, min: {scores.min():.4f}, max: {scores.max():.4f}")
            # Alert if std is 0
            if scores.std() < 1e-6:
                print(f"[PRMRewardModelWorker] ALERT: All scores are identical! This indicates a problem.")
                  
        return output
