"""
Causal Head Gating (CHG): A framework for interpreting attention head roles in transformers.

This package provides tools for identifying which attention heads in transformer models
are causally necessary or sufficient for specific tasks, following the methodology from
"Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers"
(Nam et al., NeurIPS 2025).

Quick Start:
    >>> from causal_head_gating import CHGAnalyzer
    >>> analyzer = CHGAnalyzer.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> results = analyzer.fit(texts=prompts, targets=targets)
    >>> print(results.necessary_heads())

For more control:
    >>> from causal_head_gating import CHG, CHGTrainer, CHGDataset
    >>> # Manual setup and training...
"""

from .api import CHGAnalyzer
from .core.chg import CHG
from .core.trainer import CHGTrainer
from .data.datasets import CHGDataset, MaskedSequenceDataset
from .analysis.masks import CHGResults

__version__ = "0.1.0"
__author__ = "Andrew Nam, Jonathan Hanke"
__all__ = [
    "CHGAnalyzer",
    "CHG",
    "CHGTrainer",
    "CHGDataset",
    "MaskedSequenceDataset",
    "CHGResults",
]
