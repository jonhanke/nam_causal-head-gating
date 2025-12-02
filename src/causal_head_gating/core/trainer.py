"""
CHGTrainer: Three-stage training pipeline for learning attention head masks.

The training proceeds in three stages:
1. Unregularized: Learn task-relevant masks without sparsity bias
2. Positive L1: Find minimal necessary heads (push masks toward 0)
3. Negative L1: Find maximal sufficient heads (push masks toward 1)
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, Literal, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import trange

from .chg import CHG
from ..data.datasets import MaskedSequenceDataset
from ..models.adapters import ModelAdapter


class CHGTrainer:
    """
    Trainer for learning attention head masks via Causal Head Gating.

    This trainer implements the three-stage CHG training pipeline:
    1. **Unregularized stage**: Trains masks to minimize task loss without any
       sparsity penalty, allowing all task-relevant heads to have high mask values.
    2. **Positive L1 stage**: Starting from unregularized masks, applies positive
       L1 regularization to push masks toward 0, identifying the minimal set of
       heads necessary for the task.
    3. **Negative L1 stage**: Starting from unregularized masks, applies negative
       L1 regularization to push masks toward 1, identifying the maximal set of
       heads sufficient for the task.

    Attributes:
        model: The transformer model being analyzed.
        dataset: The training dataset.
        masked_model: CHG wrapper around the model.
        mask_logits: Mask parameters for unregularized training.
        mask_logits_pos: Mask parameters for positive L1 training.
        mask_logits_neg: Mask parameters for negative L1 training.

    Example:
        >>> trainer = CHGTrainer(model, dataset, num_masks=10)
        >>> for mask, metrics in trainer.fit(num_updates=500, num_reg_updates=500):
        ...     print(f"Stage: {metrics['regularization']}, NLL: {metrics['nll']:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: MaskedSequenceDataset,
        num_masks: Optional[int] = None,
        mask_init_range: Tuple[float, float] = (0.0, 1.0),
        batch_size: int = 32,
        l1_weight: float = 0.1,
        l1_clamp: float = 4.0,
        lr: float = 1.0,
        lr_end: Optional[float] = None,
        lr_reg: float = 1.0,
        lr_reg_end: Optional[float] = None,
        gradient_accum_steps: int = 1,
        grad_norm: Optional[float] = 1.0,
        adapter: Optional[ModelAdapter] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the CHG trainer.

        Args:
            model: The transformer model to analyze. Should have frozen parameters.
            dataset: MaskedSequenceDataset with input_ids and loss_masks.
            num_masks: Number of mask configurations to learn simultaneously.
                If None, learns a single mask.
            mask_init_range: Range (low, high) for uniform mask initialization.
            batch_size: Training batch size.
            l1_weight: Weight for L1 regularization in stages 2 and 3.
            l1_clamp: Clamp mask logits to [-l1_clamp, l1_clamp] before L1 computation.
                This prevents extreme values and encourages binary solutions.
            lr: Learning rate for unregularized stage.
            lr_end: End learning rate for linear schedule (None = constant LR).
            lr_reg: Learning rate for regularized stages.
            lr_reg_end: End learning rate for regularized stages.
            gradient_accum_steps: Number of micro-batches per update.
            grad_norm: Max gradient norm for clipping (None = no clipping).
            adapter: Optional model adapter. If None, auto-detected.
            device: Device to train on. If None, uses model's device.
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.l1_weight = l1_weight
        self.l1_clamp = l1_clamp
        self.lr = lr
        self.lr_end = lr_end
        self.lr_reg = lr_reg
        self.lr_reg_end = lr_reg_end
        self.gradient_accum_steps = gradient_accum_steps
        self.grad_norm = grad_norm

        # Initialize CHG wrapper
        self.masked_model = CHG(model, adapter=adapter)

        # Create mask parameters
        self.mask_logits = self.masked_model.create_masks(
            num_masks=num_masks, mask_init_range=mask_init_range
        )
        self.mask_logits_pos = nn.Parameter(torch.zeros_like(self.mask_logits))
        self.mask_logits_neg = nn.Parameter(torch.zeros_like(self.mask_logits))

        # Set up mixed precision training
        self._device_type = self._get_device_type(device)
        self.scaler = torch.amp.GradScaler(enabled=(self._device_type == "cuda"))

        # Create data iterator
        self.iterator = dataset.to_iterator(batch_size=batch_size, shuffle=True)

        # Training state
        self._steps_since_update = 0
        self._update_nll = 0.0
        self._update_L1 = 0.0
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[LinearLR] = None

    def _get_device_type(self, device: Optional[str]) -> str:
        """Determine device type for AMP."""
        if device is not None:
            return device.split(":")[0]
        model_device = self.masked_model.device
        return str(model_device).split(":")[0]

    def _reset_optimizer(
        self,
        mask_logits: nn.Parameter,
        lr: float,
        end_lr: Optional[float],
        total_steps: int,
    ) -> None:
        """Reset optimizer and scheduler for a new training stage."""
        self._optimizer = torch.optim.Adam([mask_logits], lr=lr)
        if end_lr is not None:
            self._scheduler = LinearLR(
                self._optimizer,
                start_factor=1.0,
                end_factor=end_lr / lr,
                total_iters=total_steps,
            )
        else:
            self._scheduler = None

    def _zero_grad(self) -> None:
        """Zero gradients and reset accumulation counters."""
        if self._optimizer is not None:
            self._optimizer.zero_grad()
        self._steps_since_update = 0
        self._update_nll = 0.0
        self._update_L1 = 0.0

    def _update(self, mask_logits: nn.Parameter) -> None:
        """Perform optimizer step with gradient clipping."""
        self.scaler.unscale_(self._optimizer)
        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(mask_logits, max_norm=self.grad_norm)
        self.scaler.step(self._optimizer)
        self.scaler.update()
        if self._scheduler is not None:
            self._scheduler.step()
        self._zero_grad()

    def _get_loss(
        self,
        mask_logits: nn.Parameter,
        input_ids: Tensor,
        loss_masks: Tensor,
        l1_weight: float,
    ) -> Tuple[Tensor, float, float]:
        """
        Compute training loss (NLL + L1 regularization).

        Args:
            mask_logits: Current mask parameters in logit space.
            input_ids: Batch of token IDs.
            loss_masks: Batch of loss masks.
            l1_weight: L1 regularization weight (can be negative for stage 3).

        Returns:
            Tuple of (loss tensor, nll value, L1 value).
        """
        with torch.amp.autocast(self._device_type):
            # Compute log-probability with masks
            logp = self.masked_model.get_logp(
                mask_logits.sigmoid(),
                input_ids,
                loss_masks,
                agg="mean",
            )
            nll = -logp.mean() / self.gradient_accum_steps

            # L1 regularization on clamped logits
            if l1_weight != 0:
                clamped = mask_logits.clamp(-self.l1_clamp, self.l1_clamp)
                L1 = l1_weight * clamped.mean() / self.gradient_accum_steps
            else:
                L1 = torch.tensor(0.0)

            # Total loss: minimize NLL, maximize L1 term (hence subtraction)
            # For positive L1: pushes logits down (masks toward 0)
            # For negative L1: pushes logits up (masks toward 1)
            loss = nll - L1

        L1_val = L1.item() if isinstance(L1, Tensor) else L1
        return loss, nll.item(), L1_val

    def _train_step(
        self,
        mask_logits: nn.Parameter,
        l1_weight: float,
    ) -> Tuple[float, float]:
        """
        Execute a single training step.

        Returns accumulated NLL and L1 values for the current update.
        """
        batch = next(self.iterator)
        input_ids = batch["input_ids"]
        loss_masks = batch["loss_masks"]

        loss, nll, L1 = self._get_loss(mask_logits, input_ids, loss_masks, l1_weight)
        self.scaler.scale(loss).backward()

        self._update_nll += nll
        self._update_L1 += L1
        self._steps_since_update += 1

        if self._steps_since_update >= self.gradient_accum_steps:
            self._update(mask_logits)

        return self._update_nll, self._update_L1

    def fit_stage(
        self,
        stage: Literal["none", "positive", "negative"],
        num_updates: int,
        show_progress: bool = True,
        callback: Optional[Callable[[Tensor, Dict[str, Any]], None]] = None,
    ) -> Generator[Tuple[Tensor, Dict[str, Any]], None, None]:
        """
        Train masks for a single stage.

        Args:
            stage: Training stage:
                - 'none': Unregularized (no L1)
                - 'positive': Positive L1 (find necessary heads)
                - 'negative': Negative L1 (find sufficient heads)
            num_updates: Number of optimizer updates.
            show_progress: Whether to show progress bar.
            callback: Optional function called after each update with (mask, metrics).

        Yields:
            Tuples of (mask, metrics) after each optimizer update.
            - mask: Current mask values in [0, 1] range, shape [num_layers, num_heads]
                or [num_masks, num_layers, num_heads].
            - metrics: Dict with 'regularization', 'num_update', 'nll', 'L1'.
        """
        if stage == "none":
            mask_logits = self.mask_logits
            l1_weight = 0.0
            lr, end_lr = self.lr, self.lr_end
        elif stage == "positive":
            mask_logits = self.mask_logits_pos
            l1_weight = self.l1_weight
            lr, end_lr = self.lr_reg, self.lr_reg_end
        elif stage == "negative":
            mask_logits = self.mask_logits_neg
            l1_weight = -self.l1_weight
            lr, end_lr = self.lr_reg, self.lr_reg_end
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self._reset_optimizer(mask_logits, lr, end_lr, num_updates)
        self._zero_grad()

        updates = 0
        total_steps = num_updates * self.gradient_accum_steps

        for step in trange(total_steps, disable=not show_progress, desc=f"Stage: {stage}"):
            nll, L1 = self._train_step(mask_logits, l1_weight)

            # Yield after each optimizer update
            if step == 0 or self._steps_since_update == 0:
                updates += 1
                mask = mask_logits.sigmoid().detach().cpu()
                metrics = {
                    "regularization": stage,
                    "num_update": updates,
                    "nll": nll,
                    "L1": L1,
                }
                if callback is not None:
                    callback(mask, metrics)
                yield mask, metrics

    def fit(
        self,
        num_updates: int = 500,
        num_reg_updates: int = 500,
        save_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        callback: Optional[Callable[[Tensor, Dict[str, Any]], None]] = None,
    ) -> Generator[Tuple[Tensor, Dict[str, Any]], None, None]:
        """
        Run the full three-stage CHG training pipeline.

        This method trains through all three stages sequentially:
        1. Unregularized stage (num_updates steps)
        2. Positive L1 stage (num_reg_updates steps)
        3. Negative L1 stage (num_reg_updates steps)

        After the unregularized stage, the positive and negative mask parameters
        are initialized from the unregularized result.

        Args:
            num_updates: Number of updates for unregularized stage.
            num_reg_updates: Number of updates for each regularized stage.
            save_path: Optional path to save masks after each stage. Should end in '.pt'.
            verbose: Whether to print progress messages and show progress bars.
            callback: Optional function called after each update.

        Yields:
            Tuples of (mask, metrics) after each optimizer update across all stages.

        Example:
            >>> trainer = CHGTrainer(model, dataset, num_masks=10)
            >>> all_metrics = []
            >>> for mask, metrics in trainer.fit():
            ...     all_metrics.append(metrics)
            >>> # Analyze final masks
            >>> final_necessary = trainer.mask_logits_pos.sigmoid()
            >>> final_sufficient = trainer.mask_logits_neg.sigmoid()
        """
        # Validate save path
        if save_path is not None:
            save_path = Path(save_path)
            if save_path.suffix != ".pt":
                raise ValueError("save_path must end with .pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)

        all_masks: Dict[str, list] = {"none": [], "positive": [], "negative": []}

        # Training loop
        stages = [
            ("none", num_updates),
            ("positive", num_reg_updates),
            ("negative", num_reg_updates),
        ]

        for stage, steps in stages:
            if verbose:
                print(f"\nFitting masks with regularization: {stage}")

            for mask, metrics in self.fit_stage(stage, steps, show_progress=verbose, callback=callback):
                all_masks[stage].append(mask)
                yield mask, metrics

            # Clean up optimizer
            del self._optimizer
            self._optimizer = None
            if self._scheduler is not None:
                del self._scheduler
                self._scheduler = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # Initialize regularized masks from unregularized result
            if stage == "none":
                with torch.no_grad():
                    self.mask_logits_pos.copy_(self.mask_logits)
                    self.mask_logits_neg.copy_(self.mask_logits)

            # Save checkpoint
            if save_path is not None:
                masks_to_save = {
                    k: torch.stack(v, dim=0) if v else None
                    for k, v in all_masks.items()
                }
                torch.save(masks_to_save, save_path)
                if verbose:
                    print(f"  Saved masks to {save_path}")

    def get_final_masks(self) -> Dict[str, Tensor]:
        """
        Get the final trained masks from all stages.

        Returns:
            Dict with keys 'unregularized', 'necessary', 'sufficient', each
            containing mask values in [0, 1] range.
        """
        return {
            "unregularized": self.mask_logits.sigmoid().detach().cpu(),
            "necessary": self.mask_logits_pos.sigmoid().detach().cpu(),
            "sufficient": self.mask_logits_neg.sigmoid().detach().cpu(),
        }
