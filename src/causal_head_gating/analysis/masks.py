"""
Analysis utilities for CHG mask results.

Provides CHGResults container and head taxonomy classification.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import Tensor


class HeadTaxonomy(Enum):
    """
    Classification of attention head roles based on CHG analysis.

    Based on the comparison of necessary (positive L1) and sufficient (negative L1) masks:
    - FACILITATING: Head helps task performance (high in both necessary and sufficient)
    - INTERFERING: Head hurts task performance (low in necessary, high in sufficient)
    - IRRELEVANT: Head has no effect (low in both)
    """

    FACILITATING = "facilitating"
    INTERFERING = "interfering"
    IRRELEVANT = "irrelevant"


@dataclass
class CHGResults:
    """
    Container for CHG analysis results with analysis methods.

    This class stores the trained masks from all three CHG stages and provides
    methods for analyzing and visualizing the results.

    Attributes:
        masks: Dict with keys 'none', 'positive', 'negative', each containing
            mask tensors of shape [num_updates, num_masks, num_layers, num_heads]
            or [num_updates, num_layers, num_heads] for single mask.
        metrics: List of metric dicts from training (optional).
        model_config: Dict with model configuration info (optional).

    Example:
        >>> results = CHGResults.load("masks.pt")
        >>> necessary = results.necessary_heads(threshold=0.5)
        >>> print(f"Found {len(necessary)} necessary heads")
        >>> df = results.to_dataframe()
    """

    masks: Dict[str, Tensor]
    metrics: Optional[List[Dict[str, Any]]] = None
    model_config: Optional[Dict[str, Any]] = None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CHGResults":
        """
        Load CHGResults from a saved file.

        Args:
            path: Path to the .pt file saved by CHGTrainer.

        Returns:
            CHGResults instance.
        """
        data = torch.load(path, weights_only=False)

        if isinstance(data, dict) and any(k in data for k in ["none", "positive", "negative"]):
            # Standard format from CHGTrainer
            return cls(masks=data)
        else:
            raise ValueError(f"Unrecognized file format at {path}")

    def save(self, path: Union[str, Path]) -> None:
        """
        Save CHGResults to a file.

        Args:
            path: Destination path (should end in .pt).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "masks": self.masks,
            "metrics": self.metrics,
            "model_config": self.model_config,
        }
        torch.save(data, path)

    def _get_final_masks(self, stage: str) -> Tensor:
        """Get the final mask from a training stage."""
        masks = self.masks.get(stage)
        if masks is None:
            raise ValueError(f"No masks found for stage '{stage}'")
        # Return last update
        return masks[-1]

    @property
    def num_layers(self) -> int:
        """Get the number of layers."""
        mask = next(iter(self.masks.values()))
        return mask.shape[-2]

    @property
    def num_heads(self) -> int:
        """Get the number of heads per layer."""
        mask = next(iter(self.masks.values()))
        return mask.shape[-1]

    @property
    def num_masks(self) -> Optional[int]:
        """Get the number of mask configurations (None if single mask)."""
        mask = next(iter(self.masks.values()))
        if mask.ndim == 4:
            return mask.shape[1]
        return None

    def necessary_heads(
        self,
        threshold: float = 0.5,
        mask_idx: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get heads identified as necessary for the task.

        Necessary heads are those with high mask values after positive L1
        regularization (which pushes masks toward 0). Heads that remain
        "on" despite this pressure are necessary for task performance.

        Args:
            threshold: Mask value threshold for considering a head "on".
            mask_idx: If multiple masks, which one to analyze (default: average all).

        Returns:
            List of (layer_idx, head_idx) tuples for necessary heads.
        """
        masks = self._get_final_masks("positive")
        return self._heads_above_threshold(masks, threshold, mask_idx)

    def sufficient_heads(
        self,
        threshold: float = 0.5,
        mask_idx: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get heads identified as sufficient for the task.

        Sufficient heads are those with high mask values after negative L1
        regularization (which pushes masks toward 1). The set of heads that
        are "on" represents a sufficient set for task performance.

        Args:
            threshold: Mask value threshold for considering a head "on".
            mask_idx: If multiple masks, which one to analyze (default: average all).

        Returns:
            List of (layer_idx, head_idx) tuples for sufficient heads.
        """
        masks = self._get_final_masks("negative")
        return self._heads_above_threshold(masks, threshold, mask_idx)

    def _heads_above_threshold(
        self,
        masks: Tensor,
        threshold: float,
        mask_idx: Optional[int],
    ) -> List[Tuple[int, int]]:
        """Helper to find heads above a threshold."""
        if masks.ndim == 3:
            # Multiple masks: [num_masks, num_layers, num_heads]
            if mask_idx is not None:
                masks = masks[mask_idx]
            else:
                masks = masks.mean(dim=0)
        # Now masks is [num_layers, num_heads]

        heads = []
        for layer_idx in range(masks.shape[0]):
            for head_idx in range(masks.shape[1]):
                if masks[layer_idx, head_idx] >= threshold:
                    heads.append((layer_idx, head_idx))
        return heads

    def head_taxonomy(
        self,
        threshold: float = 0.5,
        mask_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Classify each head according to CHG taxonomy.

        The taxonomy is based on comparing necessary and sufficient masks:
        - FACILITATING: necessary=high, sufficient=high (helps the task)
        - INTERFERING: necessary=low, sufficient=high (hurts the task)
        - IRRELEVANT: necessary=low, sufficient=low (no effect)

        Args:
            threshold: Threshold for binary classification.
            mask_idx: If multiple masks, which one to analyze.

        Returns:
            DataFrame with columns: layer, head, necessary_mask, sufficient_mask, taxonomy
        """
        necessary = self._get_final_masks("positive")
        sufficient = self._get_final_masks("negative")

        # Handle multiple masks
        if necessary.ndim == 3:
            if mask_idx is not None:
                necessary = necessary[mask_idx]
                sufficient = sufficient[mask_idx]
            else:
                necessary = necessary.mean(dim=0)
                sufficient = sufficient.mean(dim=0)

        records = []
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                nec_val = necessary[layer_idx, head_idx].item()
                suf_val = sufficient[layer_idx, head_idx].item()

                is_necessary = nec_val >= threshold
                is_sufficient = suf_val >= threshold

                if is_necessary and is_sufficient:
                    taxonomy = HeadTaxonomy.FACILITATING
                elif not is_necessary and is_sufficient:
                    taxonomy = HeadTaxonomy.INTERFERING
                else:
                    taxonomy = HeadTaxonomy.IRRELEVANT

                records.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "necessary_mask": nec_val,
                    "sufficient_mask": suf_val,
                    "taxonomy": taxonomy.value,
                })

        return pd.DataFrame.from_records(records)

    def to_dataframe(
        self,
        stage: str = "positive",
        include_all_updates: bool = False,
    ) -> pd.DataFrame:
        """
        Convert masks to a long-format DataFrame for analysis.

        Args:
            stage: Which stage's masks to convert ('none', 'positive', 'negative').
            include_all_updates: If True, include all training updates.
                If False, only include the final masks.

        Returns:
            DataFrame with columns: update, mask_idx (if multiple), layer, head, value
        """
        masks = self.masks.get(stage)
        if masks is None:
            raise ValueError(f"No masks found for stage '{stage}'")

        if not include_all_updates:
            masks = masks[-1:]  # Keep as 3D/4D for consistent handling

        records = []
        for update_idx, update_masks in enumerate(masks):
            if update_masks.ndim == 2:
                # Single mask: [num_layers, num_heads]
                for layer_idx in range(update_masks.shape[0]):
                    for head_idx in range(update_masks.shape[1]):
                        records.append({
                            "update": update_idx,
                            "layer": layer_idx,
                            "head": head_idx,
                            "value": update_masks[layer_idx, head_idx].item(),
                        })
            else:
                # Multiple masks: [num_masks, num_layers, num_heads]
                for mask_idx in range(update_masks.shape[0]):
                    for layer_idx in range(update_masks.shape[1]):
                        for head_idx in range(update_masks.shape[2]):
                            records.append({
                                "update": update_idx,
                                "mask_idx": mask_idx,
                                "layer": layer_idx,
                                "head": head_idx,
                                "value": update_masks[mask_idx, layer_idx, head_idx].item(),
                            })

        return pd.DataFrame.from_records(records)

    def summary(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Get a summary of the CHG analysis results.

        Args:
            threshold: Threshold for head classification.

        Returns:
            Dict with summary statistics.
        """
        necessary = self.necessary_heads(threshold)
        sufficient = self.sufficient_heads(threshold)
        taxonomy = self.head_taxonomy(threshold)

        total_heads = self.num_layers * self.num_heads
        taxonomy_counts = taxonomy["taxonomy"].value_counts().to_dict()

        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "total_heads": total_heads,
            "num_necessary": len(necessary),
            "num_sufficient": len(sufficient),
            "pct_necessary": len(necessary) / total_heads * 100,
            "pct_sufficient": len(sufficient) / total_heads * 100,
            "taxonomy_counts": taxonomy_counts,
        }

    def __repr__(self) -> str:
        stages = list(self.masks.keys())
        return (
            f"CHGResults(stages={stages}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads})"
        )
