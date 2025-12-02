"""Basic import tests for the causal_head_gating package."""

import pytest


def test_import_main_package():
    """Test that the main package can be imported."""
    import causal_head_gating
    assert hasattr(causal_head_gating, "__version__")


def test_import_core_classes():
    """Test that core classes can be imported."""
    from causal_head_gating import CHG, CHGTrainer, CHGDataset, CHGResults


def test_import_high_level_api():
    """Test that the high-level API can be imported."""
    from causal_head_gating import CHGAnalyzer


def test_import_data_utilities():
    """Test that data utilities can be imported."""
    from causal_head_gating.data import MaskedSequenceDataset, PromptTokenizer
    from causal_head_gating.data import load_math_dataset, create_fewshot_dataset, get_aba_abb_path


def test_import_utils():
    """Test that utility functions can be imported."""
    from causal_head_gating.utils import to_long_df, TensorDict
