"""
Core CHG (Causal Head Gating) class for applying learnable masks to attention heads.

This module provides the main mechanism for gating attention head outputs
during forward passes, enabling causal analysis of head importance.
"""

import contextlib
from functools import partial
from typing import Literal, Optional, Tuple, Union

import einops
import torch
from torch import nn, Tensor

from ..models.adapters import ModelAdapter, get_adapter
from ..utils.helpers import cross_entropy


class CHG:
    """
    Causal Head Gating wrapper for transformer models.

    This class wraps a HuggingFace transformer model to add attention head gating
    capability. It uses forward hooks on attention output projections to intercept
    and scale attention head outputs by learnable masks.

    The gating is applied AFTER attention is computed but BEFORE the final output
    projection mixes the heads together. This allows for causal analysis of
    individual head contributions.

    Attributes:
        model: The wrapped transformer model.
        adapter: The model adapter for accessing architecture-specific internals.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> chg = CHG(model)
        >>> masks = chg.create_masks()
        >>> output = chg(input_ids, masks)
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: Optional[ModelAdapter] = None,
    ):
        """
        Initialize CHG wrapper around a transformer model.

        Args:
            model: A HuggingFace transformer model (e.g., from AutoModelForCausalLM).
            adapter: Optional model adapter. If None, auto-detected from model type.

        Raises:
            ValueError: If model architecture is not supported and no adapter provided.
        """
        self.model = model
        self.adapter = adapter if adapter is not None else get_adapter(model)
        self._mask_hooks: list = []

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self.adapter.device

    @property
    def num_layers(self) -> int:
        """Get the number of transformer layers."""
        return self.adapter.num_layers

    @property
    def num_heads(self) -> int:
        """Get the number of attention heads per layer."""
        return self.adapter.num_heads

    @property
    def head_dim(self) -> int:
        """Get the dimension of each attention head."""
        return self.adapter.head_dim

    @property
    def mask_shape(self) -> Tuple[int, int]:
        """Get the shape of a single mask: (num_layers, num_heads)."""
        return (self.num_layers, self.num_heads)

    def create_masks(
        self,
        num_masks: Optional[int] = None,
        mask_init_range: Tuple[float, float] = (0.0, 1.0),
        as_parameter: bool = True,
        as_logits: bool = True,
    ) -> Union[nn.Parameter, Tensor]:
        """
        Create learnable mask tensors for attention heads.

        Args:
            num_masks: Number of mask configurations to create. If None, creates
                a single mask of shape [num_layers, num_heads].
            mask_init_range: Range (low, high) for uniform initialization.
            as_parameter: If True, wrap masks as nn.Parameter for optimization.
            as_logits: If True, store masks in logit space (sigmoid applied during forward).

        Returns:
            Mask tensor of shape [num_layers, num_heads] or [num_masks, num_layers, num_heads].
            If as_parameter=True, returns nn.Parameter.

        Note:
            Masks are initialized uniformly in the specified range, then converted
            to logit space if as_logits=True. During forward pass, sigmoid is
            applied to convert back to [0, 1] range.
        """
        if num_masks is None:
            masks = torch.rand(self.num_layers, self.num_heads, device=self.device)
        else:
            masks = torch.rand(num_masks, self.num_layers, self.num_heads, device=self.device)

        # Scale to init range
        low, high = mask_init_range
        masks = low + (high - low) * masks

        # Convert to logit space if requested
        if as_logits:
            # Clamp to avoid inf in logit
            masks = masks.clamp(1e-6, 1 - 1e-6)
            masks = torch.log(masks / (1 - masks))

        if as_parameter:
            masks = nn.Parameter(masks)

        return masks

    def remove_hooks(self) -> None:
        """Remove all registered mask hooks."""
        for hook in self._mask_hooks:
            hook.remove()
        self._mask_hooks.clear()

    @contextlib.contextmanager
    def register_mask_hooks(self, masks: Tensor):
        """
        Context manager that registers forward hooks for mask application.

        The hooks intercept the input to each attention output projection,
        reshape it to expose individual heads, apply the mask, and reshape back.

        Args:
            masks: Mask tensor of shape [batch*num_masks, num_layers, num_heads].
                   Values should be in [0, 1] range (after sigmoid if using logits).

        Yields:
            None. Hooks are active within the context.

        Note:
            Hooks are automatically removed when exiting the context, even if
            an exception occurs.
        """

        def pre_hook(module: nn.Module, input: Tuple[Tensor, ...], layer_idx: int) -> Tuple[Tensor, ...]:
            x = input[0]  # Shape: [batch, seq_len, hidden_dim]
            bsz, seq_len, hidden_dim = x.shape

            # Reshape to expose individual heads
            x = x.view(bsz, seq_len, self.num_heads, self.head_dim)

            # Get mask for this layer: [batch, 1, num_heads, 1]
            mask = masks[:, layer_idx].view(-1, 1, self.num_heads, 1)
            mask = mask.to(x.device, dtype=x.dtype)

            # Apply mask
            x = x * mask

            # Reshape back
            x = x.view(bsz, seq_len, hidden_dim)
            return (x,)

        # Register hooks on all layers
        layers = self.adapter.get_layers()
        for i, layer in enumerate(layers):
            o_proj = self.adapter.get_attention_output_proj(layer)
            hook = o_proj.register_forward_pre_hook(partial(pre_hook, layer_idx=i))
            self._mask_hooks.append(hook)

        try:
            yield
        finally:
            self.remove_hooks()

    def forward(
        self,
        input_ids: Tensor,
        masks: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with optional attention head masking.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_length].
            masks: Optional mask tensor. Can be:
                - None: No masking, standard forward pass
                - Shape [num_layers, num_heads]: Single mask applied to all samples
                - Shape [num_masks, num_layers, num_heads]: Multiple masks evaluated in parallel

        Returns:
            Logits tensor. Shape depends on masks:
                - No masks: [batch_size, seq_length, vocab_size]
                - Single mask: [batch_size, seq_length, vocab_size]
                - Multiple masks: [batch_size, num_masks, seq_length, vocab_size]

        Note:
            When using multiple masks, the input is replicated internally to evaluate
            all mask configurations efficiently in a single forward pass.
        """
        if masks is None:
            return self.model(input_ids).logits

        single_mask = False
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
            single_mask = True

        batch_size = input_ids.size(0)
        num_masks = masks.size(0)

        # Replicate inputs for each mask configuration
        input_ids = einops.repeat(input_ids, "b s -> (b k) s", k=num_masks)
        masks = einops.repeat(masks, "k l h -> (b k) l h", b=batch_size)

        with self.register_mask_hooks(masks):
            output = self.model(input_ids).logits
            # Reshape to separate mask dimension
            output = output.view(batch_size, num_masks, *output.shape[1:])
            return output[:, 0] if single_mask else output

    def __call__(self, *args, **kwargs) -> Tensor:
        """Shorthand for forward()."""
        return self.forward(*args, **kwargs)

    def get_logp(
        self,
        masks: Tensor,
        input_ids: Tensor,
        loss_masks: Tensor,
        agg: Literal["none", "sum", "mean"] = "sum",
        return_accuracy: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute log-probability of target tokens under masked model.

        This is the core metric used for CHG training - it measures how well
        the model can predict targets when certain heads are masked.

        Args:
            masks: Attention masks in [0, 1] range. Shape [num_layers, num_heads]
                or [num_masks, num_layers, num_heads].
            input_ids: Token IDs of shape [batch_size, seq_length].
            loss_masks: Boolean mask indicating which tokens to include in loss.
                Shape [batch_size, seq_length]. True = include token.
            agg: Aggregation method for log-probs:
                - 'none': Return per-token log-probs
                - 'sum': Sum log-probs over tokens
                - 'mean': Average log-probs over tokens
            return_accuracy: If True, also return prediction accuracy.

        Returns:
            Log-probability tensor. Shape depends on masks and aggregation.
            If return_accuracy=True, returns tuple of (logp, accuracy).

        Example:
            >>> chg = CHG(model)
            >>> masks = torch.ones(chg.num_layers, chg.num_heads)  # All heads on
            >>> logp = chg.get_logp(masks, input_ids, loss_masks)
        """
        if agg not in ["none", "sum", "mean"]:
            raise ValueError(f"Unsupported aggregation method: {agg}")

        single_mask = False
        if masks.ndim == 2:
            single_mask = True
            masks = masks.unsqueeze(0)

        num_masks = masks.size(0)

        # Forward pass (excluding last token which has no target)
        logits = self(input_ids[:, :-1], masks)

        # Prepare targets and masks for all mask configurations
        targets = einops.repeat(input_ids[:, 1:], "b l -> b m l", m=num_masks)
        loss_masks_expanded = einops.repeat(loss_masks[:, 1:], "b l -> b m l", m=num_masks)

        # Compute accuracy if requested
        if return_accuracy:
            pred = logits.argmax(-1)
            correct = (pred == targets) * loss_masks_expanded
            accuracy = correct.sum(-1) / loss_masks_expanded.sum(-1).clamp(min=1)

        # Compute negative log-likelihood
        nll = cross_entropy(logits, targets, reduction="none")
        logp = -(nll * loss_masks_expanded)

        # Aggregate
        if agg != "none":
            logp = logp.sum(dim=-1)
        if agg == "mean":
            logp = logp / loss_masks_expanded.sum(dim=-1).clamp(min=1)

        # Remove mask dimension if single mask
        if single_mask:
            logp = logp.squeeze(1)
            if return_accuracy:
                accuracy = accuracy.squeeze(1)

        if return_accuracy:
            return logp, accuracy
        return logp
