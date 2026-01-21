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
Process Reward Model (PRM) implementation for PRM-GRPO.

This module implements a PRM that scores truncated thinking chains using latent states.

Architecture:
1. Extract latent states from layer 15 of the Actor model (all tokens)
2. Inject latent states into layer 0 of the PRM backbone (as inputs_embeds)
3. Pass through transformer layers and use regression head to output score

Reference:
- /data/chenhaotian/latentqa/math_sheperd/preprocess_latent_states.py
- /data/chenhaotian/latentqa/math_sheperd/train_math_shepherd_step1_scorer.py
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, Tuple


class RegressionHead(nn.Module):
    """Regression head for PRM scoring."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, hidden_size] - last token hidden states
        Returns:
            scores: [batch_size, 1] - scores between 0 and 1
        """
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return torch.sigmoid(x)


class LatentExtractor(nn.Module):
    """
    Extracts latent states from a specified layer of the model.
    This is used to extract hidden states from the Actor model.
    """
    
    def __init__(
        self,
        model_name: str,
        extract_layer: int = 15,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        # Load model for extracting latent states
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.extract_layer = extract_layer
        self.hidden_size = self.model.config.hidden_size
        self.dtype = dtype
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract hidden states from the specified layer.
        
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
        )
        
        # hidden_states[0] is embedding, hidden_states[i+1] is layer i output
        latent_states = outputs.hidden_states[self.extract_layer + 1]
        
        return latent_states


class PRMBackbone(nn.Module):
    """
    PRM Backbone that takes latent states as input and outputs scores.
    
    The latent states are injected into layer 0 (as inputs_embeds) and
    passed through the transformer layers. The last token's hidden state
    is used to compute the final score via a regression head.
    """
    
    def __init__(
        self,
        model_name: str,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        # Load backbone model
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        
        self.hidden_size = hidden_size
        self.dtype = dtype
        
        # Regression head
        self.regression_head = RegressionHead(hidden_size)
        self.regression_head = self.regression_head.to(dtype)
        
        # Remove original lm_head to save memory
        if hasattr(self.backbone, 'lm_head'):
            del self.backbone.lm_head
    
    def forward(
        self,
        latent_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent_states: [batch_size, seq_len, hidden_size] - extracted latent states
            attention_mask: [batch_size, seq_len]
        
        Returns:
            scores: [batch_size] - scores between 0 and 1
        """
        # Inject latent states into layer 0 (as inputs_embeds)
        outputs = self.backbone.model(
            inputs_embeds=latent_states,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get last layer hidden states
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Get the last valid token's hidden state for each sample
        batch_size = latent_states.shape[0]
        seq_lengths = attention_mask.sum(dim=1) - 1  # [batch]
        
        # Extract pooled representation
        pooled = torch.zeros(batch_size, self.hidden_size, 
                           device=hidden_states.device, dtype=self.dtype)
        for i in range(batch_size):
            last_idx = int(seq_lengths[i].item())
            pooled[i] = hidden_states[i, last_idx, :]
        
        # Compute scores via regression head
        scores = self.regression_head(pooled)  # [batch, 1]
        
        return scores.squeeze(-1)  # [batch]


class ProcessRewardModel(nn.Module):
    """
    Complete Process Reward Model for PRM-GRPO.
    
    This model:
    1. Uses LatentExtractor to extract layer-15 hidden states from input
    2. Passes latent states to PRMBackbone for scoring
    
    Note: In the actual training loop, the Actor model should be used for
    latent extraction to share computation. This class provides a standalone
    implementation for inference.
    """
    
    def __init__(
        self,
        model_name: str,
        extract_layer: int = 15,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        # Latent extractor (can be shared with Actor)
        self.extractor = LatentExtractor(
            model_name=model_name,
            extract_layer=extract_layer,
            dtype=dtype,
        )
        
        # PRM backbone
        self.backbone = PRMBackbone(
            model_name=model_name,
            hidden_size=self.extractor.hidden_size,
            dtype=dtype,
        )
        
        self.extract_layer = extract_layer
        self.hidden_size = self.extractor.hidden_size
        self.dtype = dtype
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass: extract latent states and compute scores.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            scores: [batch_size] - scores between 0 and 1
        """
        # Extract latent states from layer 15
        latent_states = self.extractor(input_ids, attention_mask)
        
        # Compute scores
        scores = self.backbone(latent_states, attention_mask)
        
        return scores
    
    def score_from_latent(
        self,
        latent_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores from pre-extracted latent states.
        This is useful when latent states are extracted during Actor forward.
        
        Args:
            latent_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            scores: [batch_size] - scores between 0 and 1
        """
        return self.backbone(latent_states, attention_mask)


def load_prm_from_checkpoint(
    checkpoint_path: str,
    backbone_path: str,
    extract_layer: int = 15,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PRMBackbone, int]:
    """
    Load PRM backbone from a checkpoint.
    
    The checkpoint format should match train_math_shepherd_step1_scorer.py output:
    - model_state_dict: contains backbone.xxx and regression_head.xxx
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        backbone_path: Path to the backbone model (for config)
        extract_layer: Layer from which latent states were extracted
        device: Device to load the model on
        dtype: Data type for the model
        
    Returns:
        Tuple of (PRMBackbone model, hidden_size)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    
    # Get configuration
    extract_layer = config.get("extract_layer", extract_layer)
    
    # Get hidden size from the checkpoint or model
    state_dict = checkpoint["model_state_dict"]
    
    # Determine hidden size from regression head weights
    if "regression_head.dense.weight" in state_dict:
        hidden_size = state_dict["regression_head.dense.weight"].shape[0]
    else:
        # Fallback: load from model config
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)
        hidden_size = model_config.hidden_size
    
    print(f"Loading PRM backbone from {checkpoint_path}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Extract layer: {extract_layer}")
    
    # Create PRM backbone
    prm_backbone = PRMBackbone(
        model_name=backbone_path,
        hidden_size=hidden_size,
        dtype=dtype,
    )
    
    # Load state dict
    # Handle both full model checkpoints and backbone-only checkpoints
    backbone_state = {}
    regression_head_state = {}
    
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            # Remove 'backbone.' prefix
            new_key = key[len("backbone."):]
            backbone_state[new_key] = value
        elif key.startswith("regression_head."):
            new_key = key[len("regression_head."):]
            regression_head_state[new_key] = value
    
    # Load weights
    if backbone_state:
        # Load backbone model weights
        missing, unexpected = prm_backbone.backbone.model.load_state_dict(backbone_state, strict=False)
        if missing:
            print(f"  - Missing backbone keys: {len(missing)}")
        if unexpected:
            print(f"  - Unexpected backbone keys: {len(unexpected)}")
    
    if regression_head_state:
        prm_backbone.regression_head.load_state_dict(regression_head_state)
        print(f"  - Loaded regression head weights")
    
    # Move to device
    prm_backbone = prm_backbone.to(device)
    prm_backbone.eval()
    
    print(f"  - Device: {device}")
    print(f"  - Dtype: {dtype}")
    
    return prm_backbone, hidden_size


class PRMScorer:
    """
    A scorer class that wraps the PRM for easy scoring in the reward manager.
    
    This scorer handles:
    1. Loading the PRM checkpoint
    2. Extracting latent states from input sequences
    3. Computing PRM scores
    
    Note: Uses lazy initialization to support loading in non-GPU processes.
    The models are only loaded when score() is first called.
    """
    
    def __init__(
        self,
        prm_checkpoint_path: str,
        backbone_path: str,
        tokenizer_path: Optional[str] = None,
        extract_layer: int = 15,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 256,
    ):
        """
        Args:
            prm_checkpoint_path: Path to the PRM checkpoint
            backbone_path: Path to the backbone model
            tokenizer_path: Path to the tokenizer (defaults to backbone_path)
            extract_layer: Layer to extract latent states from
            device: Device to load the model on
            dtype: Data type for the model
            max_length: Maximum sequence length
        """
        # Store initialization parameters for lazy loading
        self.prm_checkpoint_path = prm_checkpoint_path
        self.backbone_path = backbone_path
        self.tokenizer_path = tokenizer_path or backbone_path
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.extract_layer = extract_layer
        
        # Models will be loaded lazily
        self._initialized = False
        self.tokenizer = None
        self.extractor = None
        self.prm_backbone = None
        self.hidden_size = None
        
        print(f"[PRMScorer] Initialized with lazy loading")
        print(f"  - PRM checkpoint: {prm_checkpoint_path}")
        print(f"  - Backbone: {backbone_path}")
        print(f"  - Extract layer: {extract_layer}")
        print(f"  - Max length: {max_length}")
        print(f"  - Device: {device}")
    
    def _ensure_initialized(self):
        """Lazy initialization of models. Called before first use."""
        if self._initialized:
            return
        
        import torch.cuda
        
        # Determine actual device
        if self.device == "cuda" and not torch.cuda.is_available():
            print(f"[PRMScorer] Warning: CUDA not available, using CPU")
            self.device = "cpu"
        
        print(f"[PRMScorer] Performing lazy initialization on device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load latent extractor
        print(f"[PRMScorer] Loading latent extractor from {self.backbone_path}")
        self.extractor = LatentExtractor(
            model_name=self.backbone_path,
            extract_layer=self.extract_layer,
            dtype=self.dtype,
        )
        self.extractor = self.extractor.to(self.device)
        self.extractor.eval()
        
        # Load PRM backbone
        self.prm_backbone, self.hidden_size = load_prm_from_checkpoint(
            checkpoint_path=self.prm_checkpoint_path,
            backbone_path=self.backbone_path,
            extract_layer=self.extract_layer,
            device=self.device,
            dtype=self.dtype,
        )
        
        self._initialized = True
        print(f"[PRMScorer] Lazy initialization complete")
        
    @torch.no_grad()
    def score(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score a batch of prompt-response pairs.
        
        Args:
            prompts: List of prompts (questions)
            responses: List of responses (truncated thinking chains)
            
        Returns:
            Tuple of (scores tensor [batch_size], raw_logits tensor [batch_size])
        """
        # Ensure models are loaded
        self._ensure_initialized()
        
        # Combine prompts and responses
        texts = [p + r for p, r in zip(prompts, responses)]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Extract latent states
        latent_states = self.extractor(input_ids, attention_mask)
        
        # Score with PRM backbone
        scores = self.prm_backbone(latent_states, attention_mask)
        
        return scores, scores  # Return scores twice (scores, raw_scores)
    
    @torch.no_grad()
    def score_from_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score from pre-tokenized inputs.
        
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            
        Returns:
            Tuple of (scores tensor [batch_size], raw_logits tensor [batch_size])
        """
        # Ensure models are loaded
        self._ensure_initialized()
        
        # Move to device if needed
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Truncate to max length
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
        
        # Extract latent states
        latent_states = self.extractor(input_ids, attention_mask)
        
        # Score with PRM backbone
        scores = self.prm_backbone(latent_states, attention_mask)
        
        return scores, scores
    
    @torch.no_grad()
    def score_from_latent(
        self,
        latent_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score from pre-extracted latent states.
        
        This is the most efficient method when latent states are already
        computed during Actor forward pass.
        
        Args:
            latent_states: [batch_size, seq_length, hidden_size]
            attention_mask: [batch_size, seq_length]
            
        Returns:
            Tuple of (scores tensor [batch_size], raw_logits tensor [batch_size])
        """
        # Ensure models are loaded
        self._ensure_initialized()
        
        latent_states = latent_states.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Score with PRM backbone
        scores = self.prm_backbone(latent_states, attention_mask)
        
        return scores, scores
