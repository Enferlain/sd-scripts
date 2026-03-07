from typing import Any

import torch

from library.models.sd.text_encoder import get_hidden_states_sd, apply_hidden_state_weights_sd
from library.strategies.base.encoding import TextEncodingStrategy
from library.strategies.base.tokenization import TokenizeStrategy
from library.strategies.sd.tokenization import SdTokenizeStrategy


class SdTextEncodingStrategy(TextEncodingStrategy):
    """
    Text encoding strategy for SD1.5 and SD2.0.
    """

    def __init__(self, clip_skip: int | None = None) -> None:
        self.clip_skip = clip_skip

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens.

        Args:
            tokenize_strategy: TokenizeStrategy instance
            models: List of models
            tokens: List of token tensors

        Returns:
            List of encoded tensors
        """
        text_encoder = models[0]
        tokens_tensor = tokens[0]
        assert isinstance(tokenize_strategy, SdTokenizeStrategy)
        sd_tokenize_strategy: SdTokenizeStrategy = tokenize_strategy

        encoder_hidden_states = get_hidden_states_sd(
            tokens_tensor,
            sd_tokenize_strategy.tokenizer,
            text_encoder,
            clip_skip=self.clip_skip,
        )
        return [encoder_hidden_states]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        encoder_hidden_states = self.encode_tokens(tokenize_strategy, models, tokens)[0]
        return [apply_hidden_state_weights_sd(encoder_hidden_states, weights[0])]
