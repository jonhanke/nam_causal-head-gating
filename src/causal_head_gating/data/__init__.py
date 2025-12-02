"""Data utilities for CHG training."""

from pathlib import Path

from .datasets import CHGDataset, MaskedSequenceDataset, TensorDictDataset
from .tokenization import PromptTokenizer
from .formatters import (
    create_fewshot_dataset,
    assign_split,
    create_prompts,
    generate_prompt_permutations,
    load_math_dataset,
)

# HuggingFace dataset repo for package data
HF_DATASET_REPO = "jonhanke-nam/nam-causal-head-gating"


def get_aba_abb_path() -> Path:
    """Download and return the path to the ABA/ABB dataset.

    Downloads from HuggingFace Hub on first use, then cached locally.

    Returns:
        Path to the aba_abb data.tsv file.

    Example:
        >>> from causal_head_gating.data import get_aba_abb_path
        >>> data_path = get_aba_abb_path()
        >>> dataset = CHGDataset.from_tsv(str(data_path), tokenizer=tokenizer)
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="aba_abb/data.tsv",
        repo_type="dataset",
    )
    return Path(path)


__all__ = [
    "CHGDataset",
    "MaskedSequenceDataset",
    "TensorDictDataset",
    "PromptTokenizer",
    "create_fewshot_dataset",
    "assign_split",
    "create_prompts",
    "generate_prompt_permutations",
    "load_math_dataset",
    "get_aba_abb_path",
    "HF_DATASET_REPO",
]
