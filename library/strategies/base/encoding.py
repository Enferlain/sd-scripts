from typing import Optional, Any

import torch

from library.strategies.base.tokenization import TokenizeStrategy


class TextEncodingStrategy:
    """
    Base class for text encoding strategy.
    """

    _strategy = None  # strategy instance: actual strategy class

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncodingStrategy"]:
        return cls._strategy

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens into embeddings and outputs.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of TextModel
            tokens: List of token tensors for each TextModel

        Returns:
            List of output embeddings for each architecture
        """
        raise NotImplementedError

    def encode_tokens_with_weights(
        self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor], weights: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Encode tokens into embeddings and outputs with weights.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of TextModel
            tokens: List of token tensors for each TextModel
            weights: List of weight tensors for each TextModel

        Returns:
            List of output embeddings for each architecture
        """
        raise NotImplementedError
