"""
Dataset classes for CHG training.

Provides TensorDict-based datasets with support for variable-length sequences
and automatic padding/truncation.
"""

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils.tensor_dict import TensorDict


class TensorDictDataset(TensorDict):
    """
    PyTorch Dataset wrapper for TensorDicts.

    Extends TensorDict with Dataset interface and DataLoader helpers.
    All tensors must have the same first dimension (batch size).

    Example:
        >>> dataset = TensorDictDataset(
        ...     input_ids=torch.randint(0, 1000, (100, 64)),
        ...     labels=torch.randint(0, 10, (100,))
        ... )
        >>> for batch in dataset.to_dataloader(batch_size=8):
        ...     print(batch["input_ids"].shape)
    """

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(next(iter(self._dict.values())))

    def __getitem__(self, index):
        """Get a single sample by index, or a tensor by string key."""
        # Delegate string keys to parent TensorDict behavior
        if isinstance(index, str):
            return self._dict[index]
        if isinstance(index, (list, tuple)) and index and isinstance(index[0], str):
            return TensorDict(**{k: self._dict[k] for k in index})
        # Integer index: return a single sample as TensorDict
        return TensorDict(**{k: v[index] for k, v in self._dict.items()})

    def collate_fn(self, batch: List[TensorDict]) -> TensorDict:
        """Stack a list of TensorDicts into a batch."""
        return TensorDict.stack(batch)

    def to_dataloader(
        self,
        infinite: bool = False,
        **kwargs: Any,
    ) -> Union[DataLoader, Iterator[TensorDict]]:
        """
        Create a DataLoader from this dataset.

        Args:
            infinite: If True, return an infinite iterator that loops forever.
            **kwargs: Arguments passed to DataLoader (batch_size, shuffle, etc.).

        Returns:
            DataLoader or infinite iterator.
        """
        def infinite_loader() -> Iterator[TensorDict]:
            while True:
                yield from DataLoader(self, collate_fn=self.collate_fn, **kwargs)

        if infinite:
            return infinite_loader()
        return DataLoader(self, collate_fn=self.collate_fn, **kwargs)

    def to_iterator(self, **kwargs: Any) -> Iterator[TensorDict]:
        """Create an infinite iterator over the dataset."""
        return iter(self.to_dataloader(infinite=True, **kwargs))


class MaskedSequenceDataset(TensorDictDataset):
    """
    Dataset for variable-length sequences with automatic batch truncation.

    This dataset automatically truncates batches to remove unnecessary padding,
    improving training efficiency for variable-length sequences.

    Required field:
        - input_ids: Token ID tensor of shape [num_samples, max_seq_len]

    Common additional fields:
        - loss_masks: Boolean tensor indicating which tokens to supervise

    Args:
        pad_token_id: Token ID used for padding.
        pad_right: If True, truncate padding from the right (default).
            If False, truncate from the left.
        **kwargs: Named tensors for the dataset.

    Example:
        >>> dataset = MaskedSequenceDataset(
        ...     pad_token_id=tokenizer.pad_token_id,
        ...     input_ids=input_ids,
        ...     loss_masks=loss_masks,
        ... )
        >>> for batch in dataset.to_dataloader(batch_size=32):
        ...     # Batch is automatically truncated to shortest sequence
        ...     print(batch["input_ids"].shape)
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_right: bool = True,
        **kwargs: Tensor,
    ):
        if "input_ids" not in kwargs:
            raise ValueError("MaskedSequenceDataset requires 'input_ids' field")
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.pad_right = pad_right

    def map(self, f: Callable[[Tensor], Tensor]) -> "MaskedSequenceDataset":
        """Apply a function to all tensors, returning a new dataset."""
        tensors = {}
        for k, v in self._dict.items():
            if isinstance(v, TensorDict):
                tensors[k] = v.map(f)
            elif isinstance(v, Tensor):
                tensors[k] = f(v)
            else:
                raise TypeError(f"Cannot apply map to field '{k}' of type {type(v)}")
        return MaskedSequenceDataset(self.pad_token_id, self.pad_right, **tensors)

    def collate_fn(self, batch: List[TensorDict]) -> TensorDict:
        """Stack batch and truncate padding."""
        batch_td = TensorDict.stack(batch)
        tokens = batch_td["input_ids"]

        if self.pad_right:
            # Find first column that's all padding
            match = (tokens == self.pad_token_id).all(0)
            if match.any():
                idx = match.byte().argmax(-1).item()
                for k in batch_td:
                    if batch_td[k].ndim == 2 and batch_td[k].shape[1] == tokens.shape[1]:
                        batch_td._dict[k] = batch_td[k][:, :idx]
        else:
            # Find last column that's all padding
            match = (tokens == self.pad_token_id).all(0)
            if match.any():
                idx = match.byte().argmin(-1).item()
                for k in batch_td:
                    if batch_td[k].ndim == 2 and batch_td[k].shape[1] == tokens.shape[1]:
                        batch_td._dict[k] = batch_td[k][:, idx:]

        return batch_td


class CHGDataset:
    """
    Factory class for creating CHG-compatible datasets.

    Provides convenient methods for creating MaskedSequenceDataset from various
    input formats including raw text, HuggingFace datasets, and pattern tasks.

    Example:
        >>> # From raw text
        >>> dataset = CHGDataset.from_texts(
        ...     texts=["The capital of France is", "2 + 2 ="],
        ...     targets=["Paris", "4"],
        ...     tokenizer=tokenizer,
        ... )
        >>>
        >>> # From HuggingFace dataset
        >>> dataset = CHGDataset.from_huggingface(
        ...     dataset_name="openai/gsm8k",
        ...     tokenizer=tokenizer,
        ...     input_column="question",
        ...     target_column="answer",
        ... )
    """

    @staticmethod
    def from_texts(
        texts: List[str],
        targets: List[str],
        tokenizer: Any,
        max_length: Optional[int] = None,
        device: str = "cpu",
    ) -> MaskedSequenceDataset:
        """
        Create a dataset from raw text pairs.

        Args:
            texts: List of input/prompt texts.
            targets: List of target texts to predict.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length (truncates if exceeded).
            device: Device to place tensors on.

        Returns:
            MaskedSequenceDataset ready for CHG training.

        Note:
            The loss_masks will mark only the target tokens for supervision,
            not the prompt tokens.
        """
        from .tokenization import PromptTokenizer

        prompt_tokenizer = PromptTokenizer(tokenizer, device=device)
        input_ids, loss_masks = prompt_tokenizer.tokenize_batch(texts, targets)

        if max_length is not None and input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]
            loss_masks = loss_masks[:, :max_length]

        return MaskedSequenceDataset(
            pad_token_id=tokenizer.pad_token_id,
            input_ids=input_ids,
            loss_masks=loss_masks,
        )

    @staticmethod
    def from_tokens(
        input_ids: Tensor,
        loss_masks: Tensor,
        pad_token_id: int,
    ) -> MaskedSequenceDataset:
        """
        Create a dataset from pre-tokenized data.

        Args:
            input_ids: Token IDs of shape [num_samples, seq_len].
            loss_masks: Boolean masks of shape [num_samples, seq_len].
            pad_token_id: Token ID used for padding.

        Returns:
            MaskedSequenceDataset ready for CHG training.
        """
        return MaskedSequenceDataset(
            pad_token_id=pad_token_id,
            input_ids=input_ids,
            loss_masks=loss_masks,
        )

    @staticmethod
    def from_huggingface(
        dataset_name: str,
        tokenizer: Any,
        input_column: str,
        target_column: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_length: Optional[int] = None,
        device: str = "cpu",
    ) -> MaskedSequenceDataset:
        """
        Create a dataset from a HuggingFace dataset.

        Args:
            dataset_name: Name of the HuggingFace dataset (e.g., "openai/gsm8k").
            tokenizer: HuggingFace tokenizer.
            input_column: Column name for input text.
            target_column: Column name for target text.
            split: Dataset split to use.
            max_samples: Maximum number of samples to use (None = all).
            max_length: Maximum sequence length.
            device: Device to place tensors on.

        Returns:
            MaskedSequenceDataset ready for CHG training.
        """
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        texts = dataset[input_column]
        targets = dataset[target_column]

        return CHGDataset.from_texts(
            texts=texts,
            targets=targets,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
        )

    @staticmethod
    def from_pattern_task(
        pattern: str,
        tokenizer: Any,
        num_samples: int = 10000,
        num_lines: int = 5,
        use_vocab_tokens: bool = True,
        vocab_size: int = 26,
        seed: Optional[int] = None,
        device: str = "cpu",
    ) -> MaskedSequenceDataset:
        """
        Create a synthetic pattern completion dataset.

        These datasets test specific capabilities like induction heads.
        The pattern task presents sequences like:
            "word1^ word2^ word1\\n word3^ word4^ word3\\n ... word_n^ word_m^"
        and the model must predict the next token based on the pattern.

        Args:
            pattern: Pattern type:
                - "aba": A B A pattern (tests copying/induction)
                - "abb": A B B pattern (tests repetition detection)
            tokenizer: HuggingFace tokenizer.
            num_samples: Number of samples to generate.
            num_lines: Number of pattern lines per sample (default 5, last is incomplete).
            use_vocab_tokens: If True, use real vocabulary tokens (like original data).
                If False, use simple letters A-Z.
            vocab_size: Number of unique tokens to sample from (only used if use_vocab_tokens=False).
            seed: Random seed for reproducibility.
            device: Device to place tensors on.

        Returns:
            MaskedSequenceDataset with pattern completion task.

        Example:
            >>> # Using real vocab tokens (matches original aba_abb.tsv format)
            >>> dataset = CHGDataset.from_pattern_task("aba", tokenizer, num_samples=1000)
            >>>
            >>> # Using simple letters (simpler, faster)
            >>> dataset = CHGDataset.from_pattern_task("aba", tokenizer, use_vocab_tokens=False)
        """
        import random

        if seed is not None:
            random.seed(seed)

        if pattern not in ("aba", "abb"):
            raise ValueError(f"Unknown pattern: {pattern}. Must be 'aba' or 'abb'.")

        if use_vocab_tokens:
            # Use real vocabulary tokens like the original aba_abb.tsv
            # Filter to get reasonable tokens (not special, not too short)
            vocab = tokenizer.get_vocab()
            valid_tokens = [
                tok for tok, idx in vocab.items()
                if len(tok) >= 2
                and not tok.startswith("<")
                and not tok.startswith("[")
                and idx < tokenizer.vocab_size
                and tok.isalpha() or (tok.startswith("▁") and tok[1:].isalpha())
            ]
            # Remove leading space marker if present (common in sentencepiece)
            token_pool = []
            for tok in valid_tokens:
                clean = tok.lstrip("▁").lstrip("Ġ")  # Handle different tokenizer formats
                if len(clean) >= 2 and clean.isalpha():
                    token_pool.append(clean)
            token_pool = list(set(token_pool))[:5000]  # Limit pool size
            if len(token_pool) < 100:
                # Fallback if tokenizer doesn't have enough valid tokens
                use_vocab_tokens = False

        if not use_vocab_tokens:
            # Simple letter-based patterns
            token_pool = [chr(ord("A") + i) for i in range(vocab_size)]

        texts = []
        targets = []

        for _ in range(num_samples):
            lines = []
            # Generate (num_lines - 1) complete patterns
            for _ in range(num_lines - 1):
                a, b = random.sample(token_pool, 2)
                if pattern == "aba":
                    lines.append(f" {a}^ {b}^ {a}")
                else:  # abb
                    lines.append(f" {a}^ {b}^ {b}")

            # Generate incomplete pattern (last line)
            a, b = random.sample(token_pool, 2)
            lines.append(f" {a}^ {b}^")

            text = "\n".join(lines)
            # Target is the token that completes the pattern
            target = f" {a}" if pattern == "aba" else f" {b}"

            texts.append(text)
            targets.append(target)

        return CHGDataset.from_texts(
            texts=texts,
            targets=targets,
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def from_tsv(
        tsv_path: str,
        tokenizer: Any,
        prompt_column: str = "prompt",
        target_column: str = "target",
        max_samples: Optional[int] = None,
        device: str = "cpu",
        last_token_only: bool = False,
    ) -> MaskedSequenceDataset:
        """
        Create a dataset from a TSV file (like aba_abb.tsv).

        This method loads pre-generated pattern data from disk, which is useful
        for exact reproducibility with the original CHG experiments.

        Args:
            tsv_path: Path to the TSV file.
            tokenizer: HuggingFace tokenizer.
            prompt_column: Column name for prompts.
            target_column: Column name for targets.
            max_samples: Maximum samples to load (None = all).
            device: Device to place tensors on.
            last_token_only: If True, only supervise the last token (original ABA/ABB behavior).
                If False, use PromptTokenizer to supervise all target tokens.

        Returns:
            MaskedSequenceDataset ready for CHG training.

        Example:
            >>> # Exact reproduction of original ABA/ABB notebook
            >>> dataset = CHGDataset.from_tsv("data/aba_abb.tsv", tokenizer, last_token_only=True)
        """
        import pandas as pd

        df = pd.read_csv(tsv_path, sep="\t")

        if max_samples is not None:
            df = df.head(max_samples)

        texts = df[prompt_column].tolist()
        targets = df[target_column].tolist()

        if last_token_only:
            # Original ABA/ABB behavior: concatenate and only supervise last token
            return CHGDataset.from_concatenated_texts(
                texts=[p + t for p, t in zip(texts, targets)],
                tokenizer=tokenizer,
                device=device,
            )
        else:
            return CHGDataset.from_texts(
                texts=texts,
                targets=targets,
                tokenizer=tokenizer,
                device=device,
            )

    @staticmethod
    def from_concatenated_texts(
        texts: List[str],
        tokenizer: Any,
        device: str = "cpu",
    ) -> MaskedSequenceDataset:
        """
        Create a dataset from pre-concatenated texts with loss only on the last token.

        This exactly replicates the original ABA/ABB notebook behavior where:
        1. prompt + target are concatenated
        2. Simple tokenization is applied
        3. Loss mask is set to True only for the last token

        Args:
            texts: List of full text strings (prompt + target already concatenated).
            tokenizer: HuggingFace tokenizer.
            device: Device to place tensors on.

        Returns:
            MaskedSequenceDataset with loss only on last token.

        Example:
            >>> texts = [row['prompt'] + row['target'] for row in data]
            >>> dataset = CHGDataset.from_concatenated_texts(texts, tokenizer)
        """
        # Simple tokenization like original notebook
        tokens = tokenizer(texts)['input_ids']
        input_ids = torch.tensor(tokens, device=device)

        # Loss only on last token (original behavior)
        loss_masks = torch.zeros_like(input_ids, dtype=torch.bool)
        loss_masks[:, -1] = True

        return MaskedSequenceDataset(
            pad_token_id=tokenizer.pad_token_id,
            input_ids=input_ids,
            loss_masks=loss_masks,
        )
