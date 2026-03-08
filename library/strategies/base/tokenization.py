"""
Compatibility shim for the merged tokenization contract.

The canonical runtime tokenization contract now lives in
``library.strategies.base.training.TokenizationStrategy``.
"""

import torch
from transformers import CLIPTokenizer

from library.models.sd.tokenizer import get_clip_input_ids, get_clip_weighted_input_ids
from library.strategies.base.training import TokenizationStrategy


class TokenizeStrategy(TokenizationStrategy):
    @classmethod
    def set_strategy(cls, strategy):
        TokenizationStrategy.set_strategy(strategy)

    @classmethod
    def get_strategy(cls):
        return TokenizationStrategy.get_strategy()

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        raise NotImplementedError

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        raise NotImplementedError

    def _get_weighted_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_clip_weighted_input_ids(tokenizer, text, max_length)

    def _get_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: int | None = None, weighted: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return get_clip_input_ids(tokenizer, text, max_length, weighted=weighted)


__all__ = ["TokenizeStrategy"]
