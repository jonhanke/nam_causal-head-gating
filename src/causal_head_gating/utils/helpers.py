"""
General utility functions for the CHG package.
"""

import random
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F


def format_memory_size(
    num_bytes: int,
    precision: int = 2,
    unit: Optional[Literal["B", "KB", "MB", "GB", "TB"]] = None,
) -> str:
    """
    Format a byte count as a human-readable string.

    Args:
        num_bytes: Number of bytes.
        precision: Decimal places for the formatted number.
        unit: Specific unit to use (auto-detected if None).

    Returns:
        Formatted string like "1.23 GB".

    Example:
        >>> format_memory_size(1024 * 1024 * 1024)
        '1.00 GB'
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    if unit is None:
        scale = min(len(units) - 1, int(num_bytes).bit_length() // 10)
        unit = units[scale]
    else:
        unit = unit.upper()
        if unit not in units:
            raise ValueError(f"Invalid unit '{unit}', must be one of {units}")
        scale = units.index(unit)

    scaled = num_bytes / (1024**scale)
    return f"{scaled:.{precision}f} {unit}"


def seed_all(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """
    Cross-entropy loss with intuitive dimension ordering.

    The input tensor should have the class dimension as the last dimension,
    which is more intuitive for sequence models where shape is [batch, seq, vocab].

    Args:
        input: Logits tensor of shape [..., num_classes].
        target: Target indices of shape [...].
        reduction: How to reduce the loss ('none', 'mean', or 'sum').

    Returns:
        Loss tensor, reduced according to the reduction parameter.

    Example:
        >>> logits = torch.randn(32, 128, 50000)  # [batch, seq, vocab]
        >>> targets = torch.randint(0, 50000, (32, 128))
        >>> loss = cross_entropy(logits, targets)
    """
    # Move class dimension from last to second position for F.cross_entropy
    dims = [0, -1] + list(range(1, len(input.shape) - 1))
    input = input.permute(dims)
    return F.cross_entropy(input, target, reduction=reduction)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available device or validate a specified device.

    Args:
        device: Requested device ('cuda', 'mps', 'cpu', or None for auto).

    Returns:
        A torch.device object.

    Example:
        >>> device = get_device()  # Auto-detect best device
        >>> device = get_device("cuda")  # Explicitly request CUDA
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def check_device() -> None:
    """
    Print information about available compute devices.

    Useful for debugging and verifying GPU availability.
    """
    if torch.cuda.is_available():
        print("CUDA is available!")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"\nGPU {i}: {gpu_name}")

            total_memory = torch.cuda.get_device_properties(i).total_memory
            reserved_memory = torch.cuda.memory_reserved(i)
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = reserved_memory - allocated_memory

            print(f"  Total Memory: {total_memory / (1024 ** 3):.2f} GB")
            print(f"  Reserved Memory: {reserved_memory / (1024 ** 3):.2f} GB")
            print(f"  Allocated Memory: {allocated_memory / (1024 ** 3):.2f} GB")
            print(f"  Free Memory: {free_memory / (1024 ** 3):.2f} GB")

    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon) is available!")
    else:
        print("No GPU available. Using CPU.")


def logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute log of mean of exponentials in a numerically stable way.

    Args:
        x: Input tensor.
        dim: Dimension along which to compute.

    Returns:
        log(mean(exp(x))) computed stably.
    """
    return x.logsumexp(dim) - torch.log(
        torch.tensor(x.shape[dim], dtype=x.dtype, device=x.device)
    )


def to_long_df(
    array: torch.Tensor,
    dim_names: list,
    value_name: str = "value",
    **kwargs,
):
    """
    Convert a multi-dimensional tensor to a long-format pandas DataFrame.

    This function transforms a multi-dimensional array into a DataFrame in long format,
    where each row represents a unique combination of indices from each dimension.

    Args:
        array: Multi-dimensional tensor to convert.
        dim_names: Names for each dimension (become column names).
        value_name: Name for the value column, or list of names if last dim should be split.
        **kwargs: Additional arrays to include as columns.

    Returns:
        DataFrame in long format with columns for each dimension and values.

    Example:
        >>> masks = torch.randn(3, 500, 16, 16)  # [stages, steps, layers, heads]
        >>> df = to_long_df(masks, ['stage', 'step', 'layer', 'head'])
        >>> df.head()
           stage  step  layer  head     value
        0      0     0      0     0  0.123456
        1      0     0      0     1 -0.234567
        ...
    """
    import pandas as pd

    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()

    shape = array.shape

    if isinstance(value_name, str):
        array = array.flatten()
        index = pd.MultiIndex.from_product([range(i) for i in shape], names=dim_names)
        df = pd.DataFrame(array, columns=[value_name], index=index).reset_index()
    else:
        # value_name is a list - split last dimension into columns
        array = array.reshape(-1, len(value_name))
        index = pd.MultiIndex.from_product([range(i) for i in shape[:-1]], names=dim_names)
        df = pd.DataFrame(array, columns=value_name, index=index).reset_index()

    # Add any additional arrays
    for k, v in kwargs.items():
        i = len(v.shape)
        v = v.flatten()
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        while len(v) < len(df):
            v = np.repeat(v, shape[i])
            i += 1
        df[k] = v

    return df
