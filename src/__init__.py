"""
UAT-Lite: Bayesian Uncertainty Quantification for Transformers

This package provides a complete implementation of UAT-Lite,
including all three components and Theorem 5.
"""

__version__ = "1.0.0"
__author__ = "Author Name"
__email__ = "Author Email"

from .models import (
    UATLite,
    BayesianEmbedding,
    UncertaintyWeightedAttention,
    ConfidenceDecision,
    LayerWiseVarianceDecomposition,
)

__all__ = [
    'UATLite',
    'BayesianEmbedding',
    'UncertaintyWeightedAttention',
    'ConfidenceDecision',
    'LayerWiseVarianceDecomposition',
]