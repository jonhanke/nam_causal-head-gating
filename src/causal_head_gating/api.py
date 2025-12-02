"""
High-level API for Causal Head Gating analysis.

This module provides CHGAnalyzer, a user-friendly interface for performing
CHG analysis on transformer models with minimal boilerplate.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .analysis.masks import CHGResults
from .core.chg import CHG
from .core.trainer import CHGTrainer
from .data.datasets import CHGDataset, MaskedSequenceDataset
from .models.adapters import ModelAdapter, get_adapter


class CHGAnalyzer:
    """
    High-level interface for Causal Head Gating analysis.

    CHGAnalyzer provides a simple API for analyzing which attention heads
    in a transformer model are causally necessary or sufficient for specific
    tasks. It handles model loading, dataset preparation, training, and
    result analysis.

    Attributes:
        model: The transformer model being analyzed.
        tokenizer: The tokenizer for the model.
        adapter: Model adapter for architecture-specific access.

    Example:
        >>> from causal_head_gating import CHGAnalyzer
        >>>
        >>> # Load model
        >>> analyzer = CHGAnalyzer.from_pretrained("meta-llama/Llama-3.2-1B")
        >>>
        >>> # Analyze on custom data
        >>> results = analyzer.fit(
        ...     texts=["The capital of France is", "2 + 2 equals"],
        ...     targets=["Paris", "4"],
        ...     num_updates=500,
        ... )
        >>>
        >>> # Examine results
        >>> print(results.summary())
        >>> necessary_heads = results.necessary_heads()
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        adapter: Optional[ModelAdapter] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize CHGAnalyzer with a model and tokenizer.

        Args:
            model: A HuggingFace transformer model (should be in eval mode
                with frozen parameters).
            tokenizer: The corresponding tokenizer.
            adapter: Optional model adapter. Auto-detected if not provided.
            device: Device to use. Auto-detected if not provided.

        Note:
            Use CHGAnalyzer.from_pretrained() for easier initialization.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter if adapter is not None else get_adapter(model)

        # Ensure model is ready for analysis
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self._device = self._resolve_device(device)

    def _resolve_device(self, device: Optional[str]) -> str:
        """Determine the device to use."""
        if device is not None:
            return device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> "CHGAnalyzer":
        """
        Create a CHGAnalyzer from a pretrained model.

        Args:
            model_name_or_path: HuggingFace model identifier or local path.
            device: Device to load model on ('cuda', 'mps', 'cpu', or None for auto).
            torch_dtype: Data type for model weights (e.g., torch.float16).
            trust_remote_code: Whether to trust remote code for custom models.
            **model_kwargs: Additional arguments passed to from_pretrained().

        Returns:
            Initialized CHGAnalyzer.

        Example:
            >>> analyzer = CHGAnalyzer.from_pretrained(
            ...     "meta-llama/Llama-3.2-1B",
            ...     torch_dtype=torch.float16,
            ... )
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Set default dtype if not specified
        if torch_dtype is None and device in ("cuda", "mps"):
            torch_dtype = torch.float16

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        return cls(model, tokenizer, device=device)

    def fit(
        self,
        texts: List[str],
        targets: List[str],
        num_masks: int = 10,
        num_updates: int = 500,
        num_reg_updates: int = 500,
        batch_size: int = 32,
        l1_weight: float = 0.1,
        lr: float = 1.0,
        gradient_accum_steps: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        callback: Optional[Callable[[Tensor, Dict[str, Any]], None]] = None,
    ) -> CHGResults:
        """
        Run CHG analysis on text data.

        This method trains learnable masks over attention heads to identify
        which heads are necessary and sufficient for the task of predicting
        the targets given the texts.

        Args:
            texts: List of input/prompt texts.
            targets: List of target strings to predict.
            num_masks: Number of independent mask configurations to learn.
            num_updates: Training updates for unregularized stage.
            num_reg_updates: Training updates for each regularized stage.
            batch_size: Training batch size.
            l1_weight: Weight for L1 regularization.
            lr: Learning rate.
            gradient_accum_steps: Gradient accumulation steps.
            save_path: Optional path to save masks during training.
            verbose: Whether to show progress.
            callback: Optional function called after each update.

        Returns:
            CHGResults containing trained masks and analysis methods.

        Example:
            >>> results = analyzer.fit(
            ...     texts=["What is the capital of France?"],
            ...     targets=["Paris"],
            ...     num_updates=100,
            ... )
            >>> print(results.necessary_heads())
        """
        # Create dataset
        dataset = CHGDataset.from_texts(
            texts=texts,
            targets=targets,
            tokenizer=self.tokenizer,
            device=self._device,
        )

        return self._fit_dataset(
            dataset=dataset,
            num_masks=num_masks,
            num_updates=num_updates,
            num_reg_updates=num_reg_updates,
            batch_size=batch_size,
            l1_weight=l1_weight,
            lr=lr,
            gradient_accum_steps=gradient_accum_steps,
            save_path=save_path,
            verbose=verbose,
            callback=callback,
        )

    def fit_dataset(
        self,
        dataset: MaskedSequenceDataset,
        num_masks: int = 10,
        num_updates: int = 500,
        num_reg_updates: int = 500,
        batch_size: int = 32,
        l1_weight: float = 0.1,
        lr: float = 1.0,
        gradient_accum_steps: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        callback: Optional[Callable[[Tensor, Dict[str, Any]], None]] = None,
    ) -> CHGResults:
        """
        Run CHG analysis on a pre-prepared dataset.

        Use this method when you have already prepared a MaskedSequenceDataset,
        for example when using custom preprocessing or few-shot prompting.

        Args:
            dataset: MaskedSequenceDataset with input_ids and loss_masks.
            num_masks: Number of independent mask configurations to learn.
            num_updates: Training updates for unregularized stage.
            num_reg_updates: Training updates for each regularized stage.
            batch_size: Training batch size.
            l1_weight: Weight for L1 regularization.
            lr: Learning rate.
            gradient_accum_steps: Gradient accumulation steps.
            save_path: Optional path to save masks during training.
            verbose: Whether to show progress.
            callback: Optional function called after each update.

        Returns:
            CHGResults containing trained masks and analysis methods.
        """
        return self._fit_dataset(
            dataset=dataset,
            num_masks=num_masks,
            num_updates=num_updates,
            num_reg_updates=num_reg_updates,
            batch_size=batch_size,
            l1_weight=l1_weight,
            lr=lr,
            gradient_accum_steps=gradient_accum_steps,
            save_path=save_path,
            verbose=verbose,
            callback=callback,
        )

    def fit_pattern_task(
        self,
        pattern: str = "aba",
        num_samples: int = 10000,
        num_masks: int = 10,
        num_updates: int = 500,
        num_reg_updates: int = 500,
        batch_size: int = 32,
        l1_weight: float = 0.1,
        lr: float = 1.0,
        gradient_accum_steps: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> CHGResults:
        """
        Run CHG analysis on a synthetic pattern task.

        Pattern tasks are useful for identifying specific circuits like
        induction heads (aba pattern) or repetition detection (abb pattern).

        Args:
            pattern: Pattern type ('aba' or 'abb').
            num_samples: Number of pattern samples to generate.
            num_masks: Number of independent mask configurations.
            num_updates: Training updates for unregularized stage.
            num_reg_updates: Training updates for each regularized stage.
            batch_size: Training batch size.
            l1_weight: Weight for L1 regularization.
            lr: Learning rate.
            gradient_accum_steps: Gradient accumulation steps.
            save_path: Optional path to save masks.
            verbose: Whether to show progress.

        Returns:
            CHGResults containing trained masks.

        Example:
            >>> # Find induction heads
            >>> results = analyzer.fit_pattern_task("aba")
            >>> print(results.necessary_heads())
        """
        dataset = CHGDataset.from_pattern_task(
            pattern=pattern,
            tokenizer=self.tokenizer,
            num_samples=num_samples,
            device=self._device,
        )

        return self._fit_dataset(
            dataset=dataset,
            num_masks=num_masks,
            num_updates=num_updates,
            num_reg_updates=num_reg_updates,
            batch_size=batch_size,
            l1_weight=l1_weight,
            lr=lr,
            gradient_accum_steps=gradient_accum_steps,
            save_path=save_path,
            verbose=verbose,
        )

    def fit_huggingface(
        self,
        dataset_name: str,
        input_column: str,
        target_column: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        num_masks: int = 10,
        num_updates: int = 500,
        num_reg_updates: int = 500,
        batch_size: int = 32,
        l1_weight: float = 0.1,
        lr: float = 1.0,
        gradient_accum_steps: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> CHGResults:
        """
        Run CHG analysis on a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "openai/gsm8k").
            input_column: Column name for input text.
            target_column: Column name for target text.
            split: Dataset split to use.
            max_samples: Maximum samples to use (None = all).
            num_masks: Number of independent mask configurations.
            num_updates: Training updates for unregularized stage.
            num_reg_updates: Training updates for each regularized stage.
            batch_size: Training batch size.
            l1_weight: Weight for L1 regularization.
            lr: Learning rate.
            gradient_accum_steps: Gradient accumulation steps.
            save_path: Optional path to save masks.
            verbose: Whether to show progress.

        Returns:
            CHGResults containing trained masks.

        Example:
            >>> results = analyzer.fit_huggingface(
            ...     dataset_name="openai/gsm8k",
            ...     input_column="question",
            ...     target_column="answer",
            ...     max_samples=1000,
            ... )
        """
        dataset = CHGDataset.from_huggingface(
            dataset_name=dataset_name,
            tokenizer=self.tokenizer,
            input_column=input_column,
            target_column=target_column,
            split=split,
            max_samples=max_samples,
            device=self._device,
        )

        return self._fit_dataset(
            dataset=dataset,
            num_masks=num_masks,
            num_updates=num_updates,
            num_reg_updates=num_reg_updates,
            batch_size=batch_size,
            l1_weight=l1_weight,
            lr=lr,
            gradient_accum_steps=gradient_accum_steps,
            save_path=save_path,
            verbose=verbose,
        )

    def _fit_dataset(
        self,
        dataset: MaskedSequenceDataset,
        num_masks: int,
        num_updates: int,
        num_reg_updates: int,
        batch_size: int,
        l1_weight: float,
        lr: float,
        gradient_accum_steps: int,
        save_path: Optional[Union[str, Path]],
        verbose: bool,
        callback: Optional[Callable[[Tensor, Dict[str, Any]], None]] = None,
    ) -> CHGResults:
        """Internal method to run training."""
        trainer = CHGTrainer(
            model=self.model,
            dataset=dataset,
            num_masks=num_masks,
            batch_size=batch_size,
            l1_weight=l1_weight,
            lr=lr,
            gradient_accum_steps=gradient_accum_steps,
            adapter=self.adapter,
        )

        # Collect all masks and metrics
        all_masks: Dict[str, List[Tensor]] = {"none": [], "positive": [], "negative": []}
        all_metrics: List[Dict[str, Any]] = []

        for mask, metrics in trainer.fit(
            num_updates=num_updates,
            num_reg_updates=num_reg_updates,
            save_path=save_path,
            verbose=verbose,
            callback=callback,
        ):
            stage = metrics["regularization"]
            all_masks[stage].append(mask)
            all_metrics.append(metrics)

        # Stack masks into tensors
        stacked_masks = {
            k: torch.stack(v, dim=0) if v else None
            for k, v in all_masks.items()
        }

        # Create model config info
        model_config = {
            "num_layers": self.adapter.num_layers,
            "num_heads": self.adapter.num_heads,
            "model_class": self.model.__class__.__name__,
        }

        return CHGResults(
            masks=stacked_masks,
            metrics=all_metrics,
            model_config=model_config,
        )

    @property
    def num_layers(self) -> int:
        """Number of transformer layers in the model."""
        return self.adapter.num_layers

    @property
    def num_heads(self) -> int:
        """Number of attention heads per layer."""
        return self.adapter.num_heads
