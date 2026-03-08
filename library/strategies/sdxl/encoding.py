from typing import Any

import torch

from library.models.sdxl.text_encoder import encode_input_ids_sdxl, apply_hidden_state_weights_sdxl
from library.strategies.sdxl.tokenization import SdxlTokenizeStrategy
from library.strategies.base.training import TextEncodingStrategy, TokenizationStrategy


class SdxlTextEncodingStrategy(TextEncodingStrategy):
    """
    Text encoding strategy for SDXL.
    """

    def __init__(self) -> None:
        pass

    def encode_tokens(self, tokenize_strategy: TokenizationStrategy, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of models, [text_encoder1, text_encoder2, unwrapped text_encoder2 (optional)].
                If text_encoder2 is wrapped by accelerate, unwrapped_text_encoder2 is required
            tokens: List of tokens, for text_encoder1 and text_encoder2

        Returns:
            List of encoded tensors
        """
        if len(models) == 2:
            text_encoder1, text_encoder2 = models
            unwrapped_text_encoder2 = None
        else:
            text_encoder1, text_encoder2, unwrapped_text_encoder2 = models
        tokens1, tokens2 = tokens
        assert isinstance(tokenize_strategy, SdxlTokenizeStrategy)
        sdxl_tokenize_strategy: SdxlTokenizeStrategy = tokenize_strategy
        tokenizer1, tokenizer2 = sdxl_tokenize_strategy.tokenizer1, sdxl_tokenize_strategy.tokenizer2

        hidden_states1, hidden_states2, pool2 = encode_input_ids_sdxl(
            tokens1,
            tokens2,
            tokenizer1,
            tokenizer2,
            text_encoder1,
            text_encoder2,
            unwrapped_text_encoder2=unwrapped_text_encoder2,
        )
        return [hidden_states1, hidden_states2, pool2]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizationStrategy,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Encode tokens with weights.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of models
            tokens: List of token tensors
            weights: List of weight tensors

        Returns:
            List of encoded tensors
        """
        hidden_states1, hidden_states2, pool2 = self.encode_tokens(tokenize_strategy, models, tokens)

        hidden_states1, hidden_states2 = apply_hidden_state_weights_sdxl(
            hidden_states1,
            hidden_states2,
            weights[0],
            weights[1],
        )
        return [hidden_states1, hidden_states2, pool2]
