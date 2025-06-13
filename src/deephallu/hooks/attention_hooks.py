"""
Attention Hook Functions for Vision-Language Model Modifications

This module provides hook functions to modify attention mechanisms in VLMs:
1. Vision embeddings sparse attention with symmetric (bidirectional) attention
2. Text embeddings causal attention 
3. Cross-modal attention between vision and text
4. Embedding fusion strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
import math
import numpy as np
from functools import partial


class AttentionModifier:
    """Main class for modifying attention mechanisms in VLMs"""
    
    def __init__(self, 
                 vision_sparsity_ratio: float = 0.1,
                 vision_attention_type: str = "symmetric",
                 text_attention_type: str = "causal",
                 cross_attention_type: str = "bidirectional"):
        """
        Args:
            vision_sparsity_ratio: Ratio of attention weights to keep for vision tokens
            vision_attention_type: Type of attention for vision tokens ("symmetric", "bidirectional")
            text_attention_type: Type of attention for text tokens ("causal", "autoregressive")
            cross_attention_type: Type of cross-modal attention ("bidirectional", "vision_to_text", "text_to_vision")
        """
        self.vision_sparsity_ratio = vision_sparsity_ratio
        self.vision_attention_type = vision_attention_type
        self.text_attention_type = text_attention_type
        self.cross_attention_type = cross_attention_type
        
        self.hooks = []
        self.attention_maps = {}
        self.modified_attentions = {}
        
    def create_sparse_attention_mask(self, 
                                   attention_weights: torch.Tensor,
                                   sparsity_ratio: float,
                                   method: str = "topk") -> torch.Tensor:
        """
        Create sparse attention mask
        
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
            sparsity_ratio: Ratio of weights to keep
            method: Sparsification method ("topk", "threshold", "random")
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        if method == "topk":
            # Keep top-k attention weights
            k = int(seq_len * sparsity_ratio)
            k = max(1, k)  # Ensure at least one connection
            
            # Get top-k indices for each query
            topk_values, topk_indices = torch.topk(attention_weights, k, dim=-1)
            
            # Create sparse mask
            mask = torch.zeros_like(attention_weights)
            mask.scatter_(-1, topk_indices, 1.0)
            
        elif method == "threshold":
            # Keep weights above threshold
            threshold = torch.quantile(attention_weights.flatten(), 1 - sparsity_ratio)
            mask = (attention_weights >= threshold).float()
            
        elif method == "random":
            # Random sparsification
            mask = torch.rand_like(attention_weights) < sparsity_ratio
            mask = mask.float()
            
        return mask
    
    def apply_symmetric_attention(self, 
                                attention_weights: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply symmetric (bidirectional) attention for vision tokens
        
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
            mask: Optional attention mask
        """
        # Make attention symmetric: A_ij = A_ji
        symmetric_attention = (attention_weights + attention_weights.transpose(-2, -1)) / 2.0
        
        if mask is not None:
            symmetric_attention = symmetric_attention * mask
            
        # Renormalize
        symmetric_attention = F.softmax(symmetric_attention, dim=-1)
        
        return symmetric_attention
    
    def apply_causal_attention(self, 
                             attention_weights: torch.Tensor,
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply causal attention for text tokens
        
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
            mask: Optional attention mask
        """
        seq_len = attention_weights.size(-1)
        
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_weights.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply causal mask
        attention_weights = attention_weights.masked_fill(causal_mask == 0, float('-inf'))
        
        if mask is not None:
            attention_weights = attention_weights * mask
            
        # Apply softmax
        causal_attention = F.softmax(attention_weights, dim=-1)
        
        return causal_attention
    
    def create_mixed_attention_mask(self, 
                                  vision_seq_len: int,
                                  text_seq_len: int,
                                  batch_size: int,
                                  num_heads: int,
                                  device: torch.device) -> torch.Tensor:
        """
        Create attention mask for mixed vision-text sequences
        
        Vision tokens: symmetric attention among themselves
        Text tokens: causal attention among themselves and to vision
        Cross-modal: bidirectional between vision and text
        """
        total_seq_len = vision_seq_len + text_seq_len
        mask = torch.ones(total_seq_len, total_seq_len, device=device)
        
        # Vision-to-vision: symmetric (bidirectional)
        vision_mask = torch.ones(vision_seq_len, vision_seq_len, device=device)
        mask[:vision_seq_len, :vision_seq_len] = vision_mask
        
        # Text-to-text: causal
        text_causal_mask = torch.tril(torch.ones(text_seq_len, text_seq_len, device=device))
        mask[vision_seq_len:, vision_seq_len:] = text_causal_mask
        
        # Vision-to-text: bidirectional (depending on cross_attention_type)
        if self.cross_attention_type == "bidirectional":
            # Vision can attend to all text, text can attend to all vision
            pass  # Already set to 1
        elif self.cross_attention_type == "vision_to_text":
            # Only vision can attend to text
            mask[vision_seq_len:, :vision_seq_len] = 0
        elif self.cross_attention_type == "text_to_vision":
            # Only text can attend to vision  
            mask[:vision_seq_len, vision_seq_len:] = 0
            
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    
    def attention_hook(self, 
                      module: nn.Module, 
                      input_tuple: Tuple,
                      output: torch.Tensor,
                      vision_seq_len: int,
                      text_seq_len: int,
                      layer_name: str) -> torch.Tensor:
        """
        Hook function to modify attention weights
        
        Args:
            module: The attention module
            input_tuple: Input to the attention module
            output: Output from the attention module
            vision_seq_len: Length of vision sequence
            text_seq_len: Length of text sequence  
            layer_name: Name of the layer for logging
        """
        # Extract attention weights from output
        if isinstance(output, tuple):
            hidden_states, attention_weights = output
        else:
            hidden_states = output
            attention_weights = None
            
        # If we can't get attention weights, try to extract from module
        if attention_weights is None and hasattr(module, 'attention_weights'):
            attention_weights = module.attention_weights
            
        if attention_weights is not None:
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Store original attention for analysis
            self.attention_maps[layer_name] = attention_weights.detach().clone()
            
            # Create mixed attention mask
            mixed_mask = self.create_mixed_attention_mask(
                vision_seq_len, text_seq_len, batch_size, num_heads, attention_weights.device
            )
            
            # Split attention weights
            vision_attention = attention_weights[:, :, :vision_seq_len, :vision_seq_len]
            text_attention = attention_weights[:, :, vision_seq_len:, vision_seq_len:]
            cross_attention_v2t = attention_weights[:, :, :vision_seq_len, vision_seq_len:]
            cross_attention_t2v = attention_weights[:, :, vision_seq_len:, :vision_seq_len]
            
            # Apply sparsity to vision attention
            if self.vision_sparsity_ratio < 1.0:
                vision_sparse_mask = self.create_sparse_attention_mask(
                    vision_attention, self.vision_sparsity_ratio
                )
                vision_attention = vision_attention * vision_sparse_mask
            
            # Apply symmetric attention to vision tokens
            vision_attention = self.apply_symmetric_attention(vision_attention)
            
            # Apply causal attention to text tokens
            text_attention = self.apply_causal_attention(text_attention)
            
            # Reconstruct full attention matrix
            modified_attention = torch.zeros_like(attention_weights)
            modified_attention[:, :, :vision_seq_len, :vision_seq_len] = vision_attention
            modified_attention[:, :, vision_seq_len:, vision_seq_len:] = text_attention
            modified_attention[:, :, :vision_seq_len, vision_seq_len:] = cross_attention_v2t
            modified_attention[:, :, vision_seq_len:, :vision_seq_len] = cross_attention_t2v
            
            # Apply mixed mask
            modified_attention = modified_attention * mixed_mask
            
            # Store modified attention
            self.modified_attentions[layer_name] = modified_attention.detach().clone()
            
            # Return modified output
            if isinstance(output, tuple):
                return (hidden_states, modified_attention)
            else:
                return hidden_states
                
        return output
    
    def register_hooks(self, 
                      model: nn.Module,
                      vision_seq_len: int,
                      text_seq_len: int,
                      target_layers: Optional[List[str]] = None) -> None:
        """
        Register hooks on specified attention layers
        
        Args:
            model: The VLM model
            vision_seq_len: Length of vision sequence
            text_seq_len: Length of text sequence
            target_layers: List of layer names to hook (if None, hook all attention layers)
        """
        for name, module in model.named_modules():
            # Check if this is an attention layer
            if any(attention_name in name.lower() for attention_name in ['attention', 'attn']):
                if target_layers is None or any(target in name for target in target_layers):
                    hook_fn = partial(
                        self.attention_hook,
                        vision_seq_len=vision_seq_len,
                        text_seq_len=text_seq_len,
                        layer_name=name
                    )
                    hook = module.register_forward_hook(hook_fn)
                    self.hooks.append(hook)
                    print(f"Registered hook on layer: {name}")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("All hooks removed")
    
    def get_attention_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze attention patterns and modifications
        
        Returns:
            Dictionary with analysis results for each layer
        """
        analysis = {}
        
        for layer_name in self.attention_maps:
            original_attn = self.attention_maps[layer_name]
            modified_attn = self.modified_attentions.get(layer_name)
            
            layer_analysis = {
                'attention_entropy': float(self._compute_attention_entropy(original_attn)),
                'attention_sparsity': float(self._compute_attention_sparsity(original_attn)),
            }
            
            if modified_attn is not None:
                layer_analysis.update({
                    'modified_entropy': float(self._compute_attention_entropy(modified_attn)),
                    'modified_sparsity': float(self._compute_attention_sparsity(modified_attn)),
                    'attention_change': float(torch.norm(original_attn - modified_attn).item())
                })
            
            analysis[layer_name] = layer_analysis
            
        return analysis
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention weights"""
        # Add small epsilon for numerical stability
        eps = 1e-8
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1)
        return entropy.mean()
    
    def _compute_attention_sparsity(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute sparsity of attention weights"""
        threshold = 0.01  # Consider weights below this as "sparse"
        sparse_ratio = (attention_weights < threshold).float().mean()
        return sparse_ratio


class EmbeddingFuser:
    """Class for fusing vision and text embeddings"""
    
    def __init__(self, 
                 fusion_method: str = "concat",
                 hidden_dim: int = 768,
                 fusion_dim: Optional[int] = None):
        """
        Args:
            fusion_method: Method for fusing embeddings ("concat", "add", "gated", "attention")
            hidden_dim: Hidden dimension of embeddings
            fusion_dim: Dimension after fusion (if None, same as hidden_dim)
        """
        self.fusion_method = fusion_method
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim or hidden_dim
        
        # Initialize fusion layers based on method
        if fusion_method == "gated":
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.transform = nn.Linear(hidden_dim * 2, self.fusion_dim)
        elif fusion_method == "attention":
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
            self.transform = nn.Linear(hidden_dim, self.fusion_dim)
        elif fusion_method == "concat":
            self.transform = nn.Linear(hidden_dim * 2, self.fusion_dim)
        elif fusion_method == "add":
            self.transform = nn.Linear(hidden_dim, self.fusion_dim)
            
    def fuse_embeddings(self, 
                       vision_embeddings: torch.Tensor,
                       text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse vision and text embeddings
        
        Args:
            vision_embeddings: [batch_size, vision_seq_len, hidden_dim]
            text_embeddings: [batch_size, text_seq_len, hidden_dim]
            
        Returns:
            fused_embeddings: [batch_size, total_seq_len, fusion_dim]
        """
        if self.fusion_method == "concat":
            # Simple concatenation
            fused = torch.cat([vision_embeddings, text_embeddings], dim=1)
            return self.transform(fused)
            
        elif self.fusion_method == "add":
            # Element-wise addition (requires same sequence length)
            min_len = min(vision_embeddings.size(1), text_embeddings.size(1))
            vision_truncated = vision_embeddings[:, :min_len, :]
            text_truncated = text_embeddings[:, :min_len, :]
            fused = vision_truncated + text_truncated
            return self.transform(fused)
            
        elif self.fusion_method == "gated":
            # Gated fusion
            batch_size = vision_embeddings.size(0)
            vision_seq_len = vision_embeddings.size(1)
            text_seq_len = text_embeddings.size(1)
            
            # Repeat to match sequence lengths for gating
            vision_expanded = vision_embeddings.unsqueeze(2).expand(-1, -1, text_seq_len, -1)
            text_expanded = text_embeddings.unsqueeze(1).expand(-1, vision_seq_len, -1, -1)
            
            # Concatenate for gating
            combined = torch.cat([vision_expanded, text_expanded], dim=-1)
            
            # Apply gate
            gate_weights = torch.sigmoid(self.gate(combined))
            gated = gate_weights * vision_expanded + (1 - gate_weights) * text_expanded
            
            # Reshape and transform
            gated = gated.view(batch_size, -1, self.hidden_dim)
            return self.transform(gated)
            
        elif self.fusion_method == "attention":
            # Cross-attention fusion
            # Use vision as query, text as key/value
            fused, _ = self.attention(
                query=vision_embeddings.transpose(0, 1),  # [vision_seq_len, batch_size, hidden_dim]
                key=text_embeddings.transpose(0, 1),      # [text_seq_len, batch_size, hidden_dim]
                value=text_embeddings.transpose(0, 1)     # [text_seq_len, batch_size, hidden_dim]
            )
            fused = fused.transpose(0, 1)  # [batch_size, vision_seq_len, hidden_dim]
            
            # Concatenate with original text embeddings
            final_fused = torch.cat([fused, text_embeddings], dim=1)
            return self.transform(final_fused)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


def visualize_attention_patterns(attention_weights: torch.Tensor,
                                vision_seq_len: int,
                                text_seq_len: int,
                                save_path: Optional[str] = None) -> None:
    """
    Visualize attention patterns for vision and text tokens
    
    Args:
        attention_weights: [batch_size, num_heads, seq_len, seq_len]
        vision_seq_len: Length of vision sequence
        text_seq_len: Length of text sequence
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Average across batch and heads
    avg_attention = attention_weights.mean(dim=(0, 1)).cpu().numpy()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Vision-to-vision attention
    v2v_attn = avg_attention[:vision_seq_len, :vision_seq_len]
    sns.heatmap(v2v_attn, ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Vision-to-Vision Attention')
    
    # Text-to-text attention
    t2t_attn = avg_attention[vision_seq_len:, vision_seq_len:]
    sns.heatmap(t2t_attn, ax=axes[0, 1], cmap='Reds')
    axes[0, 1].set_title('Text-to-Text Attention')
    
    # Vision-to-text attention
    v2t_attn = avg_attention[:vision_seq_len, vision_seq_len:]
    sns.heatmap(v2t_attn, ax=axes[1, 0], cmap='Greens')
    axes[1, 0].set_title('Vision-to-Text Attention')
    
    # Text-to-vision attention
    t2v_attn = avg_attention[vision_seq_len:, :vision_seq_len]
    sns.heatmap(t2v_attn, ax=axes[1, 1], cmap='Purples')
    axes[1, 1].set_title('Text-to-Vision Attention')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show() 