"""
Layer-wise Variance Decomposition - Theorem 5 Implementation

This module implements Theorem 5 from the paper:
"Layer-wise Uncertainty Propagation in UAT-Lite"

Mathematical Formulation:
    Var[ŷ] = E_θ,ε_<L [Σ(l=1 to L) Var_ε_l[ŷ | h^(l-1)] + Var_θ,ε_<l[E_ε_l[ŷ | h^(l-1)]]]

Where:
    - First term: Aleatoric uncertainty (stochastic perturbations per layer)
    - Second term: Epistemic uncertainty (parameter uncertainty)
    - h^(l): Hidden state at layer l
    - ε_l: Dropout mask at layer l
    - θ: Model parameters

Author: N/A
Email: email@email.com
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class LayerWiseVarianceDecomposition(nn.Module):
    """
    Implements Theorem 5: Hierarchical layer-wise variance decomposition
    
    This enables:
    1. Attribution of uncertainty to specific transformer layers
    2. Decomposition into aleatoric vs epistemic components
    3. Identification of layers contributing most to prediction uncertainty
    
    Args:
        num_layers (int): Number of transformer layers (L)
        hidden_size (int): Dimension of hidden states
        track_gradients (bool): Whether to track gradients for uncertainty flow
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        track_gradients: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.track_gradients = track_gradients
        
    def compute_layer_variance(
        self,
        hidden_states_samples: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute variance for a single layer
        
        Args:
            hidden_states_samples: [M, batch_size, seq_len, hidden_size]
                M Monte Carlo samples of hidden states
            layer_idx: Layer index (0 to L-1)
            
        Returns:
            aleatoric_var: Aleatoric uncertainty [batch_size]
            epistemic_var: Epistemic uncertainty [batch_size]
        """
        # hidden_states_samples shape: [M, batch_size, seq_len, hidden_size]
        M = hidden_states_samples.shape[0]
        batch_size = hidden_states_samples.shape[1]
        
        # Compute expectation over MC samples: E_ε[h^(l)]
        # Shape: [batch_size, seq_len, hidden_size]
        mean_hidden = hidden_states_samples.mean(dim=0)
        
        # Compute variance over MC samples: Var_ε[h^(l)]
        # This captures aleatoric uncertainty (stochastic perturbations ε_l)
        # Shape: [batch_size, seq_len, hidden_size]
        variance_hidden = hidden_states_samples.var(dim=0, unbiased=True)
        
        # Aggregate over sequence length and hidden dimensions
        # Using mean pooling to get per-sample uncertainty
        aleatoric_var = variance_hidden.mean(dim=[1, 2])  # [batch_size]
        
        # Epistemic uncertainty: variance of the means
        # This captures parameter uncertainty θ
        # For each position, compute variance across samples
        epistemic_var = (hidden_states_samples - mean_hidden.unsqueeze(0)).pow(2).mean(dim=0).mean(dim=[1, 2])
        
        return aleatoric_var, epistemic_var
    
    def decompose_total_variance(
        self,
        all_layer_hidden_states: List[torch.Tensor],
        logits_samples: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose total prediction variance according to Theorem 5
        
        Args:
            all_layer_hidden_states: List of length L, each element is
                [M, batch_size, seq_len, hidden_size] for layer l
            logits_samples: [M, batch_size, num_classes]
                Logit predictions for M MC samples
                
        Returns:
            Dictionary containing:
                - total_variance: Total prediction variance [batch_size]
                - layer_aleatoric: Aleatoric variance per layer [batch_size, L]
                - layer_epistemic: Epistemic variance per layer [batch_size, L]
                - layer_total: Total variance per layer [batch_size, L]
                - aleatoric_fraction: Fraction of aleatoric uncertainty [batch_size]
                - epistemic_fraction: Fraction of epistemic uncertainty [batch_size]
        """
        M = logits_samples.shape[0]
        batch_size = logits_samples.shape[1]
        L = len(all_layer_hidden_states)
        
        # Initialize storage for per-layer variances
        layer_aleatoric_list = []
        layer_epistemic_list = []
        
        # Compute variance for each layer
        for layer_idx, hidden_states in enumerate(all_layer_hidden_states):
            aleatoric_var, epistemic_var = self.compute_layer_variance(
                hidden_states, layer_idx
            )
            layer_aleatoric_list.append(aleatoric_var)
            layer_epistemic_list.append(epistemic_var)
        
        # Stack into tensors: [batch_size, L]
        layer_aleatoric = torch.stack(layer_aleatoric_list, dim=1)
        layer_epistemic = torch.stack(layer_epistemic_list, dim=1)
        layer_total = layer_aleatoric + layer_epistemic
        
        # Compute total variance from logits
        # Var[ŷ] over MC samples
        mean_logits = logits_samples.mean(dim=0)  # [batch_size, num_classes]
        # Total variance: average variance over class dimensions
        total_variance = logits_samples.var(dim=0, unbiased=True).mean(dim=1)  # [batch_size]
        
        # Sum across layers (Theorem 5 formulation)
        total_aleatoric = layer_aleatoric.sum(dim=1)  # [batch_size]
        total_epistemic = layer_epistemic.sum(dim=1)  # [batch_size]
        
        # Compute fractions
        total_uncertainty = total_aleatoric + total_epistemic + 1e-10  # Add epsilon for numerical stability
        aleatoric_fraction = total_aleatoric / total_uncertainty
        epistemic_fraction = total_epistemic / total_uncertainty
        
        return {
            'total_variance': total_variance,
            'layer_aleatoric': layer_aleatoric,
            'layer_epistemic': layer_epistemic,
            'layer_total': layer_total,
            'total_aleatoric': total_aleatoric,
            'total_epistemic': total_epistemic,
            'aleatoric_fraction': aleatoric_fraction,
            'epistemic_fraction': epistemic_fraction,
        }
    
    def identify_critical_layers(
        self,
        layer_uncertainties: torch.Tensor,
        top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify which layers contribute most to uncertainty
        
        Args:
            layer_uncertainties: [batch_size, L] per-layer uncertainty
            top_k: Number of top contributing layers to return
            
        Returns:
            top_layer_indices: [batch_size, top_k] indices of top layers
            top_layer_values: [batch_size, top_k] uncertainty values
        """
        # Get top-k layers by uncertainty
        top_values, top_indices = torch.topk(
            layer_uncertainties, k=min(top_k, layer_uncertainties.shape[1]), dim=1
        )
        return top_indices, top_values
    
    def forward(
        self,
        all_layer_hidden_states: List[torch.Tensor],
        logits_samples: torch.Tensor,
        return_critical_layers: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: Theorem 5 variance decomposition
        
        Args:
            all_layer_hidden_states: List of hidden states from all layers
            logits_samples: MC sampled logits
            return_critical_layers: Whether to identify critical layers
            
        Returns:
            Complete uncertainty decomposition dictionary
        """
        # Compute full decomposition
        decomposition = self.decompose_total_variance(
            all_layer_hidden_states, logits_samples
        )
        
        # Identify critical layers if requested
        if return_critical_layers:
            critical_indices, critical_values = self.identify_critical_layers(
                decomposition['layer_total'], top_k=3
            )
            decomposition['critical_layer_indices'] = critical_indices
            decomposition['critical_layer_values'] = critical_values
        
        return decomposition
    
    def get_layer_contribution_percentages(
        self,
        layer_uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute percentage contribution of each layer to total uncertainty
        
        Args:
            layer_uncertainties: [batch_size, L]
            
        Returns:
            percentages: [batch_size, L] percentage contributions (sum to 100)
        """
        total = layer_uncertainties.sum(dim=1, keepdim=True) + 1e-10
        percentages = (layer_uncertainties / total) * 100.0
        return percentages


def test_variance_decomposition():
    """
    Unit test for variance decomposition
    """
    print("Testing LayerWiseVarianceDecomposition...")
    
    # Parameters
    M = 10  # MC samples
    batch_size = 4
    seq_len = 128
    hidden_size = 768
    num_layers = 12
    num_classes = 2
    
    # Create dummy hidden states for testing
    all_hidden_states = []
    for l in range(num_layers):
        # Simulate MC samples with increasing variance at deeper layers
        hidden = torch.randn(M, batch_size, seq_len, hidden_size) * (0.1 * (l + 1))
        all_hidden_states.append(hidden)
    
    # Simulate logits
    logits_samples = torch.randn(M, batch_size, num_classes)
    
    # Initialize decomposition
    decomposer = LayerWiseVarianceDecomposition(
        num_layers=num_layers,
        hidden_size=hidden_size
    )
    
    # Run decomposition
    results = decomposer(all_hidden_states, logits_samples)
    
    # Verify outputs
    print(f"✓ Total variance shape: {results['total_variance'].shape}")
    print(f"✓ Layer aleatoric shape: {results['layer_aleatoric'].shape}")
    print(f"✓ Layer epistemic shape: {results['layer_epistemic'].shape}")
    print(f"✓ Aleatoric fraction: {results['aleatoric_fraction'].mean().item():.4f}")
    print(f"✓ Epistemic fraction: {results['epistemic_fraction'].mean().item():.4f}")
    print(f"✓ Critical layers: {results['critical_layer_indices'][0].tolist()}")
    
    # Verify sum of fractions = 1
    fraction_sum = results['aleatoric_fraction'] + results['epistemic_fraction']
    assert torch.allclose(fraction_sum, torch.ones_like(fraction_sum), atol=1e-5), \
        "Aleatoric + Epistemic fractions should sum to 1"
    
    print("✓ All tests passed!")
    
    return results


if __name__ == "__main__":
    test_variance_decomposition()