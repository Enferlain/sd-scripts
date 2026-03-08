"""
Compatibility shim for the merged text-encoding contract.

The canonical runtime text-encoding contract now lives in
``library.strategies.base.training.TextEncodingStrategy``.
"""

from typing import Any

import torch

from library.strategies.base.tokenization import TokenizeStrategy
from library.strategies.base.training import TextEncodingStrategy as _TextEncodingStrategy


class TextEncodingStrategy(_TextEncodingStrategy):
    @classmethod
    def set_strategy(cls, strategy):
        _TextEncodingStrategy.set_strategy(strategy)

    @classmethod
    def get_strategy(cls):
        return _TextEncodingStrategy.get_strategy()

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        raise NotImplementedError

    def encode_tokens_with_weights(
        self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor], weights: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        raise NotImplementedError


__all__ = ["TextEncodingStrategy"]
