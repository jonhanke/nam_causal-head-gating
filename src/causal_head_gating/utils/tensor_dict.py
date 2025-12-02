"""
TensorDict: A dictionary-like container for torch.Tensors with enhanced functionality.
"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Callable, Iterator, List, Optional, TypeVar

import torch
import torch.nn.functional as F

from .helpers import format_memory_size


T = TypeVar("T", bound="TensorDict")


class TensorDict(Mapping):
    """
    A dictionary-like container for torch.Tensors with enhanced structure,
    device consistency, and batch-level operations.

    Compared to a plain dict of tensors, TensorDict:
      - Enforces all tensors share the same device.
      - Supports attribute-style access (e.g., td.x instead of td['x']).
      - Propagates tensor operations (e.g., td.cuda(), td.mean()) to all fields.
      - Supports nested TensorDicts and recursive transformations (e.g., td.map(fn)).
      - Includes utilities for memory usage, partitioning, stacking, and printing.

    Ideal for organizing minibatches, model outputs, and structured data.

    Example:
        >>> td = TensorDict(obs=torch.randn(32, 4), act=torch.randint(0, 3, (32,)))
        >>> td = td.cuda().map(lambda x: x * 2)
        >>> print(td.obs.shape)
        torch.Size([32, 4])
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize a TensorDict from keyword arguments.

        Args:
            **kwargs: Named tensors or nested TensorDicts/Mappings.

        Raises:
            ValueError: If tensors are on different devices.
        """
        self._dict: OrderedDict[str, Any] = OrderedDict()
        devices = {v.device for v in kwargs.values() if isinstance(v, torch.Tensor)}
        if len(devices) > 1:
            raise ValueError(
                f"All tensors must be on the same device. Devices found: {devices}"
            )
        for k, v in kwargs.items():
            self[k] = v

    @property
    def device(self) -> Optional[torch.device]:
        """Get the device of the tensors in this TensorDict."""
        for v in self._dict.values():
            if isinstance(v, torch.Tensor):
                return v.device
            elif isinstance(v, TensorDict):
                return v.device
        return None

    @staticmethod
    def cat(tds: List["TensorDict"], dim: int = 0, pad_value: Optional[float] = None) -> "TensorDict":
        """
        Concatenate a list of TensorDicts along the specified dimension.

        Args:
            tds: List of TensorDicts to concatenate.
            dim: Dimension along which to concatenate.
            pad_value: If set, pad tensors to match shapes before concatenating.

        Returns:
            A new TensorDict with concatenated tensors.
        """
        keys = tds[0].keys()
        new_td = {}
        for k in keys:
            values = [td[k] for td in tds]
            if isinstance(values[0], torch.Tensor):
                tensors = values
                if pad_value is not None:
                    tensors = _pad_to_same(tensors, pad_value=pad_value, exclude_dims=dim)
                new_td[k] = torch.cat(tensors, dim=dim)
            else:
                new_td[k] = TensorDict.cat(values, dim=dim, pad_value=pad_value)
        return TensorDict(**new_td)

    @staticmethod
    def stack(tds: List["TensorDict"], dim: int = 0, pad_value: Optional[float] = None) -> "TensorDict":
        """
        Stack a list of TensorDicts along a new dimension.

        Args:
            tds: List of TensorDicts to stack.
            dim: Position of the new dimension.
            pad_value: If set, pad tensors to match shapes before stacking.

        Returns:
            A new TensorDict with stacked tensors.
        """
        keys = tds[0].keys()
        new_td = {}
        for k in keys:
            values = [td[k] for td in tds]
            if isinstance(values[0], torch.Tensor):
                tensors = values
                if pad_value is not None:
                    tensors = _pad_to_same(tensors, pad_value=pad_value)
                new_td[k] = torch.stack(tensors, dim=dim)
            else:
                new_td[k] = TensorDict.stack(values, dim=dim, pad_value=pad_value)
        return TensorDict(**new_td)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._dict[key]
        if isinstance(key, (list, tuple)) and key and isinstance(key[0], str):
            return TensorDict(**{k: self._dict[k] for k in key})
        return TensorDict(**{k: v[key] for k, v in self._dict.items()})

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, (torch.Tensor, TensorDict)):
            if self._dict:
                self._check_device(value)
            self._dict[key] = value
        elif isinstance(value, Mapping):
            td_value = TensorDict(**value)
            if self._dict:
                self._check_device(td_value)
            self._dict[key] = td_value
        else:
            device = self.device or "cpu"
            self._dict[key] = torch.tensor(value, device=device)

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
        if "_dict" in self.__dict__ and item in self._dict:
            return self._dict[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if "_dict" in self.__dict__ and key in self._dict:
            if not isinstance(value, (torch.Tensor, TensorDict)):
                raise TypeError(f"Cannot assign non-tensor to registered tensor field '{key}'")
            self._check_device(value)
            self._dict[key] = value
        else:
            super().__setattr__(key, value)

    def __delitem__(self, key: str) -> None:
        del self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def __repr__(self) -> str:
        def fmt(k: str, v: Any) -> str:
            if isinstance(v, torch.Tensor):
                shape = f"({v.shape[0]})" if v.ndim == 1 else f"{tuple(v.shape)}"
                dtype = str(v.dtype).replace("torch.", "")
                return f"{k}[{dtype}]: {shape}"
            elif isinstance(v, TensorDict):
                return f"{k}: {v.__class__.__name__}"
            else:
                return f"{k}: {type(v).__name__}"

        keys = ", ".join(fmt(k, v) for k, v in self._dict.items())
        mem = self.memory_bytes_str()
        return f"{self.__class__.__name__}[{mem}] ({keys})"

    def _check_device(self, item: Any) -> None:
        if hasattr(item, "device") and self.device is not None and item.device != self.device:
            raise ValueError(
                f"Tried to set Tensor on device {item.device} to TensorDict on device {self.device}."
            )

    def update(self, other: Mapping[str, Any]) -> None:
        """Update the TensorDict with key-value pairs from another mapping."""
        if not isinstance(other, (Mapping, TensorDict)):
            raise TypeError("update() expects a Mapping or TensorDict")
        for k, v in other.items():
            self[k] = v

    def map(self: T, f: Callable[[torch.Tensor], torch.Tensor]) -> T:
        """
        Apply a function to all tensors in the TensorDict.

        Args:
            f: Function to apply to each tensor.

        Returns:
            A new TensorDict with transformed tensors.
        """
        tensors = {}
        for k, v in self._dict.items():
            if isinstance(v, TensorDict):
                tensors[k] = v.map(f)
            elif isinstance(v, torch.Tensor):
                tensors[k] = f(v)
            else:
                raise TypeError(f"Cannot apply map to non-tensor field '{k}' of type {type(v)}")
        return self.__class__(**tensors)

    def save(self, path: str) -> None:
        """Save the TensorDict to disk as a torch file."""
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls: type[T], path: str, map_location: Optional[str] = None) -> T:
        """Load a TensorDict from disk."""
        state = torch.load(path, map_location=map_location, weights_only=False)
        return cls(**state)

    def to_dict(self) -> dict:
        """Convert to a standard Python dict (for saving)."""
        out = {}
        for k, v in self._dict.items():
            if isinstance(v, TensorDict):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    def partition(
        self,
        parts: List[int] | List[float],
        seed: int = 0,
        drop_remainder: bool = True,
    ) -> List["TensorDict"]:
        """
        Partition the dataset into multiple splits.

        Args:
            parts: List of partition sizes (counts or proportions).
            seed: Random seed for shuffling.
            drop_remainder: Whether to drop samples beyond sum of parts.

        Returns:
            List of TensorDict partitions.
        """
        lengths = [v.shape[0] for v in self._dict.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All entries must have the same length along dim=0")

        N = lengths[0]
        is_proportions = all(isinstance(p, float) for p in parts)
        is_counts = all(isinstance(p, int) for p in parts)
        if not (is_proportions or is_counts):
            raise ValueError("Parts must be all floats or all ints")

        counts = [int(N * p) for p in parts] if is_proportions else parts
        if sum(counts) > N:
            raise ValueError("Requested partition sizes exceed dataset length")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        idx = torch.randperm(N, generator=generator, device=self.device)

        splits, cursor = [], 0
        for count in counts:
            sel = idx[cursor : cursor + count]
            subdict = {k: v[sel] for k, v in self._dict.items()}
            splits.append(TensorDict(**subdict))
            cursor += count

        if not drop_remainder and cursor < N:
            sel = idx[cursor:]
            subdict = {k: v[sel] for k, v in self._dict.items()}
            splits.append(TensorDict(**subdict))

        return splits

    def memory_bytes(self) -> int:
        """Get total memory usage in bytes."""
        return sum(
            v.memory_bytes() if isinstance(v, TensorDict) else v.numel() * v.element_size()
            for v in self._dict.values()
            if isinstance(v, (TensorDict, torch.Tensor))
        )

    def memory_bytes_str(self, precision: int = 2, unit: Optional[str] = None) -> str:
        """Get formatted memory usage string."""
        return format_memory_size(self.memory_bytes(), precision=precision, unit=unit)


def _pad_to_same(
    tensors: List[torch.Tensor],
    pad_value: float = 0,
    exclude_dims: Optional[int | List[int]] = None,
) -> List[torch.Tensor]:
    """Pad tensors to have the same shape."""
    if exclude_dims is None:
        exclude_dims = []
    elif isinstance(exclude_dims, int):
        exclude_dims = [exclude_dims]

    ndims = tensors[0].ndim
    max_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        for d in range(ndims):
            if d not in exclude_dims:
                max_shape[d] = max(max_shape[d], t.shape[d])

    out = []
    for t in tensors:
        shape_diff = [
            max_shape[d] - t.shape[d] if d not in exclude_dims else 0 for d in range(ndims)
        ]
        pad = []
        for d in reversed(range(ndims)):
            pad.extend([0, int(shape_diff[d])])
        if any(shape_diff):
            t = F.pad(t, pad, value=pad_value)
        out.append(t)
    return out


# Add common tensor methods to TensorDict
_TENSOR_METHODS = [
    "detach",
    "cpu",
    "cuda",
    "float",
    "double",
    "half",
    "int",
    "long",
    "bfloat16",
    "short",
    "byte",
    "bool",
    "clone",
    "contiguous",
    "squeeze",
    "unsqueeze",
    "mean",
    "sum",
    "softmax",
    "log_softmax",
    "transpose",
    "permute",
    "abs",
    "sqrt",
    "tanh",
    "relu",
    "to",
]


def _make_method(name: str) -> Callable:
    def method(self: TensorDict, *args: Any, **kwargs: Any) -> TensorDict:
        return self.map(lambda t: getattr(t, name)(*args, **kwargs))
    return method


for _name in _TENSOR_METHODS:
    setattr(TensorDict, _name, _make_method(_name))
