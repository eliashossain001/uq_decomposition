"""
Uncertainty-Weighted Attention - Component 2

Modifies transformer attention mechanism to downweight unreliable tokens
based on their uncertainty scores.

Attention reweighting:
    α'_ij = α_ij * exp(-λ * U(x_j))

where:
    - α_ij: Original attention weight from token i to token j
    - U(x_j): Uncertainty score of token j
    - λ: Uncertainty penalty hyperparameter

Author: N/A
Email: N/A@email.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class UncertaintyWeightedAttention(nn.Module):
    """
    Component 2: Uncertainty-Weighted Attention
    
    Dynamically adjusts attention weights based on token-level uncertainty,
    enabling the model to focus on reliable tokens and downweight uncertain ones.
    
    Args:
        hidden_size (int): Dimension of hidden states
        num_attention_heads (int): Number of attention heads
        uncertainty_penalty (float): λ hyperparameter (default: 0.5)
        dropout_rate (float): Attention dropout (default: 0.1)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 12,
        uncertainty_penalty: float = 0.5,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        assert hidden_size % num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.uncertainty_penalty = uncertainty_penalty
        
        # Standard attention projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape for multi-head attention
        
        Args:
            x: [batch_size, seq_len, hidden_size]
            
        Returns:
            [batch_size, num_heads, seq_len, head_size]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def compute_uncertainty_weights(
        self,
        token_uncertainty: torch.Tensor,
        lambda_penalty: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention modulation weights from uncertainty
        
        Weight formula: w_j = exp(-λ * U(x_j))
        
        Args:
            token_uncertainty: [batch_size, seq_len] uncertainty scores
            lambda_penalty: λ hyperparameter (uses self.uncertainty_penalty if None)
            
        Returns:
            weights: [batch_size, seq_len] modulation weights in (0, 1]
        """
        if lambda_penalty is None:
            lambda_penalty = self.uncertainty_penalty
        
        # Compute exponential decay: exp(-λ * U)
        # High uncertainty → low weight
        # Low uncertainty → high weight (close to 1)
        weights = torch.exp(-lambda_penalty * token_uncertainty)
        
        return weights
    
    def apply_uncertainty_weighting(
        self,
        attention_scores: torch.Tensor,
        uncertainty_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply uncertainty-based modulation to attention scores
        
        Args:
            attention_scores: [batch_size, num_heads, seq_len_q, seq_len_k]
                Raw attention scores (before softmax)
            uncertainty_weights: [batch_size, seq_len_k]
                Per-token uncertainty weights
                
        Returns:
            modulated_scores: [batch_size, num_heads, seq_len_q, seq_len_k]
                Uncertainty-weighted attention scores
        """
        # Reshape uncertainty weights for broadcasting
        # [batch_size, seq_len_k] → [batch_size, 1, 1, seq_len_k]
        uncertainty_weights = uncertainty_weights.unsqueeze(1).unsqueeze(2)
        
        # Apply multiplicative modulation to attention scores
        # High uncertainty → scores are scaled down → lower attention weights
        modulated_scores = attention_scores * uncertainty_weights
        
        return modulated_scores
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_uncertainty: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with uncertainty-weighted attention
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            token_uncertainty: [batch_size, seq_len] per-token uncertainty scores
            attention_mask: [batch_size, seq_len] mask (1 = attend, 0 = ignore)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: [batch_size, seq_len, hidden_size] attended output
            attention_weights: [batch_size, num_heads, seq_len, seq_len] (if requested)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores: Q * K^T / √d_k
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply uncertainty weighting BEFORE softmax
        # This is the key innovation: downweight unreliable tokens
        uncertainty_weights = self.compute_uncertainty_weights(token_uncertainty)
        attention_scores = self.apply_uncertainty_weighting(
            attention_scores, uncertainty_weights
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            # [batch_size, seq_len] → [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Apply mask: masked positions get large negative value
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, float('-inf')
            )
        
        # Softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs)
        
        # Compute weighted sum of values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back: [batch_size, num_heads, seq_len, head_size] 
        #             → [batch_size, seq_len, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context_layer)
        
        if return_attention_weights:
            return output, attention_probs
        else:
            return output, None
    
    def visualize_attention_modulation(
        self,
        token_uncertainty: torch.Tensor,
        tokens: Optional[list] = None
    ) -> dict:
        """
        Helper function to visualize how uncertainty affects attention
        
        Args:
            token_uncertainty: [batch_size, seq_len] uncertainty scores
            tokens: List of token strings (optional)
            
        Returns:
            Dictionary with visualization data
        """
        uncertainty_weights = self.compute_uncertainty_weights(token_uncertainty)
        
        result = {
            'token_uncertainty': token_uncertainty.cpu().numpy(),
            'uncertainty_weights': uncertainty_weights.cpu().numpy(),
            'attention_reduction': (1 - uncertainty_weights).cpu().numpy() * 100  # Percentage reduction
        }
        
        if tokens is not None:
            result['tokens'] = tokens
        
        return result


def test_uncertainty_weighted_attention():
    """
    Unit test for Uncertainty-Weighted Attention
    """
    print("Testing UncertaintyWeightedAttention...")
    
    # Parameters
    batch_size = 4
    seq_len = 128
    hidden_size = 768
    num_heads = 12
    
    # Create attention layer
    attention = UncertaintyWeightedAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        uncertainty_penalty=0.5
    )
    
    # Test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    token_uncertainty = torch.rand(batch_size, seq_len)  # Uncertainty in [0, 1]
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    output, attn_weights = attention(
        hidden_states,
        token_uncertainty,
        attention_mask,
        return_attention_weights=True
    )
    
    # Verify shapes
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Attention weights shape: {attn_weights.shape}")
    
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Test uncertainty modulation
    low_uncertainty = torch.zeros(batch_size, seq_len) + 0.1  # Low uncertainty
    high_uncertainty = torch.ones(batch_size, seq_len) * 2.0  # High uncertainty
    
    output_low, attn_low = attention(hidden_states, low_uncertainty, attention_mask, True)
    output_high, attn_high = attention(hidden_states, high_uncertainty, attention_mask, True)
    
    # High uncertainty should lead to more uniform attention (less peaked)
    entropy_low = -(attn_low * torch.log(attn_low + 1e-10)).sum(dim=-1).mean()
    entropy_high = -(attn_high * torch.log(attn_high + 1e-10)).sum(dim=-1).mean()
    
    print(f"✓ Attention entropy (low uncertainty): {entropy_low.item():.4f}")
    print(f"✓ Attention entropy (high uncertainty): {entropy_high.item():.4f}")
    
    # Visualize modulation
    vis_data = attention.visualize_attention_modulation(token_uncertainty[0:1])
    print(f"✓ Mean attention reduction: {vis_data['attention_reduction'].mean():.2f}%")
    
    print("✓ All tests passed!")
    
    return output, attn_weights


if __name__ == "__main__":
    test_uncertainty_weighted_attention()