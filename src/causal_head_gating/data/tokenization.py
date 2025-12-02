"""
Tokenization utilities for CHG.

Provides PromptTokenizer for creating loss_masks that separate prompts from targets.
"""

from typing import Any, List, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


class PromptTokenizer:
    """
    Tokenizer wrapper that separates prompts from targets for loss masking.

    CHG training typically only supervises the target tokens, not the prompt.
    This class adds a special marker token between prompt and target, then
    creates loss_masks that mark only the target tokens for supervision.

    Attributes:
        tokenizer: The wrapped HuggingFace tokenizer.
        device: Device for output tensors.
        marker: The marker string used to separate prompt and target.
        marker_id: Token ID of the marker.

    Example:
        >>> prompt_tokenizer = PromptTokenizer(tokenizer)
        >>> input_ids, loss_masks = prompt_tokenizer.tokenize_batch(
        ...     prompts=["What is 2+2?"],
        ...     targets=["4"],
        ... )
        >>> # loss_masks will be True only for the "4" token
    """

    def __init__(
        self,
        tokenizer: Any,
        device: str = "cpu",
        marker: str = " <|TARGET|> ",
    ):
        """
        Initialize the PromptTokenizer.

        Args:
            tokenizer: A HuggingFace tokenizer.
            device: Device to place output tensors on.
            marker: Special marker string to insert between prompt and target.
                Should be unique and not appear in normal text.
        """
        # Ensure marker has spaces
        if not marker.startswith(" "):
            marker = " " + marker
        if not marker.endswith(" "):
            marker = marker + " "

        # Add marker as special token if not already present
        if marker not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [marker]})

        self.tokenizer = tokenizer
        self.device = device
        self.marker = marker
        self.marker_id = tokenizer(marker, add_special_tokens=False)["input_ids"][0]

    def tokenize(self, prompt: str, target: str) -> Tuple[Tensor, Tensor]:
        """
        Tokenize a single prompt-target pair.

        Args:
            prompt: The input/prompt text.
            target: The target text to predict.

        Returns:
            Tuple of (input_ids, loss_mask) tensors.
            - input_ids: Token IDs for the full sequence.
            - loss_mask: Boolean mask, True for target tokens.
        """
        # Normalize whitespace
        prompt = prompt.rstrip()
        if not target.startswith(" "):
            target = " " + target

        # Tokenize with marker
        full_text = prompt + self.marker + target
        input_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Find and remove marker
        idx = input_ids.index(self.marker_id)
        new_ids = input_ids[:idx] + input_ids[idx + 1 :]

        # Create mask: False for prompt, True for target
        mask = [False] * idx + [True] * (len(new_ids) - idx)

        return (
            torch.tensor(new_ids, dtype=torch.long, device=self.device),
            torch.tensor(mask, dtype=torch.bool, device=self.device),
        )

    def tokenize_batch(
        self,
        prompts: List[str],
        targets: List[str],
    ) -> Tuple[Tensor, Tensor]:
        """
        Tokenize a batch of prompt-target pairs with padding.

        Args:
            prompts: List of input/prompt texts.
            targets: List of target texts.

        Returns:
            Tuple of (input_ids, loss_masks) tensors with padding.
            - input_ids: Shape [batch_size, max_seq_len]
            - loss_masks: Shape [batch_size, max_seq_len]
        """
        # Normalize whitespace
        prompts = [p.rstrip() for p in prompts]
        targets = [(" " + t) if not t.startswith(" ") else t for t in targets]

        # Tokenize with markers
        full_texts = [p + self.marker + t for p, t in zip(prompts, targets)]
        tokenized = self.tokenizer(
            full_texts,
            padding=False,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids_list = tokenized["input_ids"]

        # Process each sequence
        input_ids_trimmed = []
        mask_trimmed = []

        for ids in input_ids_list:
            idx = ids.index(self.marker_id)
            new_ids = ids[:idx] + ids[idx + 1 :]
            mask = [False] * idx + [True] * (len(new_ids) - idx)

            input_ids_trimmed.append(torch.tensor(new_ids, dtype=torch.long))
            mask_trimmed.append(torch.tensor(mask, dtype=torch.bool))

        # Pad sequences
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_trimmed,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        mask_padded = torch.nn.utils.rnn.pad_sequence(
            mask_trimmed,
            batch_first=True,
            padding_value=False,
        )

        # Add one extra padding token at the end (for target prediction)
        input_ids_padded = F.pad(input_ids_padded, (0, 1), value=self.tokenizer.pad_token_id)
        mask_padded = F.pad(mask_padded, (0, 1), value=False)

        return input_ids_padded.to(self.device), mask_padded.to(self.device)
