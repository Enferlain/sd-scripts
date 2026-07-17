"""Optional strategy features supported by some model families."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from library.metadata.dataclasses.model import ModelFamilyMetadataContribution


class ModelFamilyMetadataStrategy(ABC):
    """Optional capability for families with additional realization facts."""

    @abstractmethod
    def resolve_model_family_metadata(
        self,
        cfg: Any,
        *,
        run_identifier: str,
        realization_identifier: str,
    ) -> ModelFamilyMetadataContribution:
        """Resolve explicit versioned family-local facts for one realization."""
        raise NotImplementedError


class WeightedPromptStrategy(ABC):
    """Optional capability for model families that support weighted prompts."""

    @abstractmethod
    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Tokenize text and return prompt weights alongside token tensors.

        Args:
            text: Text or list of text to tokenize.

        Returns:
            Tuple of token tensors and weight tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_tokens_with_weights(
        self,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Encode token tensors with prompt-weight application.
        """
        raise NotImplementedError


__all__ = ["ModelFamilyMetadataStrategy", "WeightedPromptStrategy"]
