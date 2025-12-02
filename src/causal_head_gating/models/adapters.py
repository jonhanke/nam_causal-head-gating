"""
Model adapters for accessing transformer internals across different architectures.

This module provides a unified interface for accessing attention layers and their
output projections across different HuggingFace transformer model architectures.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Type
import torch.nn as nn


class ModelAdapter(ABC):
    """
    Abstract base class for accessing transformer model internals.

    Different model architectures organize their layers differently. This adapter
    provides a unified interface for CHG to work with any supported architecture.

    Attributes:
        model: The HuggingFace transformer model being adapted.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the adapter with a model.

        Args:
            model: A HuggingFace transformer model (e.g., from AutoModelForCausalLM).
        """
        self.model = model

    @abstractmethod
    def get_layers(self) -> nn.ModuleList:
        """
        Get the list of transformer layers.

        Returns:
            ModuleList of transformer layers.
        """
        pass

    @abstractmethod
    def get_attention_output_proj(self, layer: nn.Module) -> nn.Module:
        """
        Get the attention output projection (W_O) from a layer.

        This is the projection applied after attention heads are computed,
        which is where CHG applies its masks.

        Args:
            layer: A single transformer layer.

        Returns:
            The output projection module (typically a Linear layer).
        """
        pass

    @property
    def num_layers(self) -> int:
        """Get the number of transformer layers."""
        return self.model.config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        """Get the number of attention heads per layer."""
        return self.model.config.num_attention_heads

    @property
    def head_dim(self) -> int:
        """Get the dimension of each attention head."""
        hidden_size = self.model.config.hidden_size
        return hidden_size // self.num_heads

    @property
    def device(self):
        """Get the device the model is on."""
        return next(self.model.parameters()).device


class LlamaAdapter(ModelAdapter):
    """
    Adapter for Llama-style models.

    Supports:
        - meta-llama/Llama-2-*
        - meta-llama/Llama-3-*
        - meta-llama/Llama-3.1-*
        - meta-llama/Llama-3.2-*
    """

    def get_layers(self) -> nn.ModuleList:
        return self.model.model.layers

    def get_attention_output_proj(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn.o_proj


class MistralAdapter(ModelAdapter):
    """
    Adapter for Mistral-style models.

    Supports:
        - mistralai/Mistral-*
        - mistralai/Mixtral-*
    """

    def get_layers(self) -> nn.ModuleList:
        return self.model.model.layers

    def get_attention_output_proj(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn.o_proj


class GPT2Adapter(ModelAdapter):
    """
    Adapter for GPT-2 style models.

    Supports:
        - gpt2, gpt2-medium, gpt2-large, gpt2-xl
        - openai-community/gpt2*
    """

    def get_layers(self) -> nn.ModuleList:
        return self.model.transformer.h

    def get_attention_output_proj(self, layer: nn.Module) -> nn.Module:
        return layer.attn.c_proj

    @property
    def num_heads(self) -> int:
        return self.model.config.n_head


class GPTNeoXAdapter(ModelAdapter):
    """
    Adapter for GPT-NeoX style models.

    Supports:
        - EleutherAI/pythia-*
        - EleutherAI/gpt-neox-*
    """

    def get_layers(self) -> nn.ModuleList:
        return self.model.gpt_neox.layers

    def get_attention_output_proj(self, layer: nn.Module) -> nn.Module:
        return layer.attention.dense


class FalconAdapter(ModelAdapter):
    """
    Adapter for Falcon-style models.

    Supports:
        - tiiuae/falcon-*
    """

    def get_layers(self) -> nn.ModuleList:
        return self.model.transformer.h

    def get_attention_output_proj(self, layer: nn.Module) -> nn.Module:
        return layer.self_attention.dense


class QwenAdapter(ModelAdapter):
    """
    Adapter for Qwen-style models.

    Supports:
        - Qwen/Qwen2-*
        - Qwen/Qwen2.5-*
    """

    def get_layers(self) -> nn.ModuleList:
        return self.model.model.layers

    def get_attention_output_proj(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn.o_proj


class GemmaAdapter(ModelAdapter):
    """
    Adapter for Gemma-style models.

    Supports:
        - google/gemma-*
        - google/gemma-2-*
    """

    def get_layers(self) -> nn.ModuleList:
        return self.model.model.layers

    def get_attention_output_proj(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn.o_proj


class PhiAdapter(ModelAdapter):
    """
    Adapter for Phi-style models.

    Supports:
        - microsoft/phi-*
    """

    def get_layers(self) -> nn.ModuleList:
        return self.model.model.layers

    def get_attention_output_proj(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn.dense


# Mapping from model architecture names to adapter classes
ADAPTER_REGISTRY: dict[str, Type[ModelAdapter]] = {
    "LlamaForCausalLM": LlamaAdapter,
    "MistralForCausalLM": MistralAdapter,
    "MixtralForCausalLM": MistralAdapter,
    "GPT2LMHeadModel": GPT2Adapter,
    "GPTNeoXForCausalLM": GPTNeoXAdapter,
    "FalconForCausalLM": FalconAdapter,
    "Qwen2ForCausalLM": QwenAdapter,
    "GemmaForCausalLM": GemmaAdapter,
    "Gemma2ForCausalLM": GemmaAdapter,
    "PhiForCausalLM": PhiAdapter,
    "Phi3ForCausalLM": PhiAdapter,
}

# List of supported architectures for documentation
SUPPORTED_ARCHITECTURES = list(ADAPTER_REGISTRY.keys())


def get_adapter(model: nn.Module) -> ModelAdapter:
    """
    Automatically detect the model architecture and return the appropriate adapter.

    Args:
        model: A HuggingFace transformer model.

    Returns:
        An appropriate ModelAdapter instance for the model.

    Raises:
        ValueError: If the model architecture is not supported.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> adapter = get_adapter(model)
        >>> print(adapter.num_heads)  # 32
    """
    model_class_name = model.__class__.__name__

    if model_class_name in ADAPTER_REGISTRY:
        adapter_class = ADAPTER_REGISTRY[model_class_name]
        return adapter_class(model)

    # Try to find a matching adapter by checking the model's config
    if hasattr(model, 'config'):
        config = model.config
        model_type = getattr(config, 'model_type', None)

        # Map model_type to adapter
        model_type_mapping = {
            'llama': LlamaAdapter,
            'mistral': MistralAdapter,
            'mixtral': MistralAdapter,
            'gpt2': GPT2Adapter,
            'gpt_neox': GPTNeoXAdapter,
            'falcon': FalconAdapter,
            'qwen2': QwenAdapter,
            'gemma': GemmaAdapter,
            'gemma2': GemmaAdapter,
            'phi': PhiAdapter,
            'phi3': PhiAdapter,
        }

        if model_type in model_type_mapping:
            adapter_class = model_type_mapping[model_type]
            return adapter_class(model)

    supported = ", ".join(SUPPORTED_ARCHITECTURES)
    raise ValueError(
        f"Unsupported model architecture: {model_class_name}. "
        f"Supported architectures: {supported}. "
        f"You can create a custom adapter by subclassing ModelAdapter."
    )
