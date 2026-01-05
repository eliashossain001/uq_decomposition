import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class BayesianEmbedding(nn.Module):
    """
    Component 1: Bayesian Embedding Calibration
    
    Applies Monte Carlo Dropout to transformer embeddings to approximate
    the posterior distribution over embeddings given input.
    
    Key features:
    1. MC Dropout sampling at embedding layer
    2. Computes mean and variance over samples
    3. Enables epistemic uncertainty quantification
    
    Args:
        embedding_layer (nn.Module): Base embedding layer (word + position embeddings)
        dropout_rate (float): Dropout probability p (default: 0.3)
        mc_samples (int): Number of Monte Carlo samples M (default: 10)
    """
    
    def __init__(
        self,
        embedding_layer: nn.Module,
        dropout_rate: float = 0.3,
        mc_samples: int = 10
    ):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def enable_dropout(self):
        """Enable dropout even during evaluation (for MC sampling)"""
        self.dropout.train()
        
    def disable_dropout(self):
        """Disable dropout (standard inference)"""
        self.dropout.eval()
    
    def sample_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single forward pass with dropout enabled
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        # Get base embeddings
        embeddings = self.embedding_layer(input_ids)
        
        # Apply dropout (approximates sampling from posterior)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def forward_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty quantification via MC Dropout
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            num_samples: Number of MC samples (uses self.mc_samples if None)
            
        Returns:
            mean_embeddings: E[z | x] [batch_size, seq_len, hidden_size]
            std_embeddings: √Var[z | x] [batch_size, seq_len, hidden_size]
            all_samples: All M samples [M, batch_size, seq_len, hidden_size]
        """
        if num_samples is None:
            num_samples = self.mc_samples
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Enable dropout for sampling
        self.enable_dropout()
        
        # Collect MC samples
        samples = []
        for m in range(num_samples):
            # Sample embedding with dropout
            sample = self.sample_embedding(input_ids, attention_mask)
            samples.append(sample)
        
        # Stack samples: [M, batch_size, seq_len, hidden_size]
        all_samples = torch.stack(samples, dim=0)
        
        # Compute statistics
        mean_embeddings = all_samples.mean(dim=0)  # [batch_size, seq_len, hidden_size]
        std_embeddings = all_samples.std(dim=0, unbiased=True)  # [batch_size, seq_len, hidden_size]
        
        return mean_embeddings, std_embeddings, all_samples
    
    def compute_token_uncertainty(
        self,
        std_embeddings: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute per-token uncertainty from embedding variance
        
        Args:
            std_embeddings: [batch_size, seq_len, hidden_size]
            reduction: How to reduce hidden dimension ('mean' or 'sum')
            
        Returns:
            token_uncertainty: [batch_size, seq_len]
        """
        if reduction == 'mean':
            # Average variance across hidden dimensions
            uncertainty = std_embeddings.mean(dim=-1)
        elif reduction == 'sum':
            # Sum variance across hidden dimensions
            uncertainty = std_embeddings.sum(dim=-1)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return uncertainty
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True
    ) -> dict:
        """
        Main forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_uncertainty: Whether to compute uncertainty
            
        Returns:
            Dictionary containing:
                - embeddings: Mean embeddings [batch_size, seq_len, hidden_size]
                - std: Standard deviation [batch_size, seq_len, hidden_size]
                - token_uncertainty: Per-token uncertainty [batch_size, seq_len]
                - all_samples: All MC samples [M, batch_size, seq_len, hidden_size]
        """
        if return_uncertainty:
            mean_emb, std_emb, all_samples = self.forward_with_uncertainty(
                input_ids, attention_mask
            )
            token_unc = self.compute_token_uncertainty(std_emb, reduction='mean')
            
            return {
                'embeddings': mean_emb,
                'std': std_emb,
                'token_uncertainty': token_unc,
                'all_samples': all_samples
            }
        else:
            # Standard forward without uncertainty
            self.disable_dropout()
            embeddings = self.embedding_layer(input_ids)
            return {
                'embeddings': embeddings,
                'std': None,
                'token_uncertainty': None,
                'all_samples': None
            }


class BayesianLinear(nn.Module):
    """
    Bayesian Linear layer with MC Dropout
    
    Used for classification head with uncertainty
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.3,
        mc_samples: int = 10
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def enable_dropout(self):
        """Enable dropout for MC sampling"""
        self.dropout.train()
        
    def forward_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with uncertainty via MC Dropout
        
        Args:
            x: Input [batch_size, in_features]
            num_samples: Number of MC samples
            
        Returns:
            mean_output: E[y | x] [batch_size, out_features]
            std_output: √Var[y | x] [batch_size, out_features]
            all_samples: [M, batch_size, out_features]
        """
        if num_samples is None:
            num_samples = self.mc_samples
        
        self.enable_dropout()
        
        samples = []
        for m in range(num_samples):
            # Apply dropout and linear
            dropped = self.dropout(x)
            output = self.linear(dropped)
            samples.append(output)
        
        # Stack: [M, batch_size, out_features]
        all_samples = torch.stack(samples, dim=0)
        
        # Compute statistics
        mean_output = all_samples.mean(dim=0)
        std_output = all_samples.std(dim=0, unbiased=True)
        
        return mean_output, std_output, all_samples
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> torch.Tensor:
        """Standard forward"""
        if return_uncertainty:
            mean_out, _, _ = self.forward_with_uncertainty(x)
            return mean_out
        else:
            return self.linear(x)


def test_bayesian_embedding():
    """
    Unit test for Bayesian Embedding
    """
    print("Testing BayesianEmbedding...")
    
    # Parameters
    vocab_size = 30522  # BERT vocab
    hidden_size = 768
    max_position_embeddings = 512
    batch_size = 4
    seq_len = 128
    mc_samples = 10
    
    # Create base embedding layer
    base_embedding = nn.Embedding(vocab_size, hidden_size)
    
    # Create Bayesian wrapper
    bayesian_emb = BayesianEmbedding(
        embedding_layer=base_embedding,
        dropout_rate=0.3,
        mc_samples=mc_samples
    )
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward with uncertainty
    outputs = bayesian_emb(input_ids, attention_mask, return_uncertainty=True)
    
    # Verify shapes
    print(f"✓ Mean embeddings shape: {outputs['embeddings'].shape}")
    print(f"✓ Std embeddings shape: {outputs['std'].shape}")
    print(f"✓ Token uncertainty shape: {outputs['token_uncertainty'].shape}")
    print(f"✓ All samples shape: {outputs['all_samples'].shape}")
    
    # Verify values
    assert outputs['embeddings'].shape == (batch_size, seq_len, hidden_size)
    assert outputs['std'].shape == (batch_size, seq_len, hidden_size)
    assert outputs['token_uncertainty'].shape == (batch_size, seq_len)
    assert outputs['all_samples'].shape == (mc_samples, batch_size, seq_len, hidden_size)
    
    # Check that std is positive
    assert (outputs['std'] >= 0).all(), "Standard deviation should be non-negative"
    
    # Check that uncertainty varies across tokens
    print(f"✓ Mean token uncertainty: {outputs['token_uncertainty'].mean().item():.6f}")
    print(f"✓ Std of token uncertainty: {outputs['token_uncertainty'].std().item():.6f}")
    
    print("✓ All tests passed!")
    
    return outputs


if __name__ == "__main__":
    test_bayesian_embedding()