from typing import Any

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPTextModel

from library.models.sdxl.text_encoder import pool_workaround, get_hidden_states_sdxl
from library.strategies.sdxl.tokenization import SdxlTokenizeStrategy
from library.strategies.base.encoding import TextEncodingStrategy
from library.strategies.base.tokenization import TokenizeStrategy


class SdxlTextEncodingStrategy(TextEncodingStrategy):
    """
    Text encoding strategy for SDXL.
    """

    def __init__(self) -> None:
        pass

    def _pool_workaround(
        self, text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
    ):
        """
        Delegate to shared utility function.
        """
        return pool_workaround(text_encoder, last_hidden_state, input_ids, eos_token_id)

    def _get_hidden_states_sdxl(
        self,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        text_encoder1: CLIPTextModel | torch.nn.Module,
        text_encoder2: CLIPTextModelWithProjection | torch.nn.Module,
        unwrapped_text_encoder2: CLIPTextModelWithProjection | None = None,
    ):
        """
        Wrapper around shared utility that derives max_token_length from input shape.

        The input_ids have shape [b, n, 77] where n is the number of 77-token chunks.
        """
        # Derive max_token_length from input shape
        if input_ids1.size()[1] == 1:
            max_token_length = None
        else:
            max_token_length = input_ids1.size()[1] * input_ids1.size()[2]

        # Move to device (don't reshape here - get_hidden_states_sdxl does it and needs original shape for b_size)
        input_ids1 = input_ids1.to(next(text_encoder1.parameters()).device)
        input_ids2 = input_ids2.to(next(text_encoder2.parameters()).device)

        # Use unwrapped encoder for pool workaround if provided
        unwrapped_te2 = unwrapped_text_encoder2 or text_encoder2

        # Call shared utility - pass None for accelerator since we handle unwrapping here
        return get_hidden_states_sdxl(
            max_token_length,
            input_ids1,
            input_ids2,
            tokenizer1,
            tokenizer2,
            text_encoder1,
            unwrapped_te2,
            weight_dtype=None,
            accelerator=None,
        )

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
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

        hidden_states1, hidden_states2, pool2 = self._get_hidden_states_sdxl(
            tokens1, tokens2, tokenizer1, tokenizer2, text_encoder1, text_encoder2, unwrapped_text_encoder2
        )
        return [hidden_states1, hidden_states2, pool2]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
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

        weights_tensors = [w.to(hidden_states1.device) for w in weights]

        # apply weights
        if weights_tensors[0].shape[1] == 1:  # no max_token_length
            # weights: ((b, 1, 77), (b, 1, 77)), hidden_states: (b, 77, 768), (b, 77, 768)
            hidden_states1 = hidden_states1 * weights_tensors[0].squeeze(1).unsqueeze(2)
            hidden_states2 = hidden_states2 * weights_tensors[1].squeeze(1).unsqueeze(2)
        else:
            # weights: ((b, n, 77), (b, n, 77)), hidden_states: (b, n*75+2, 768), (b, n*75+2, 768)
            for weight, hidden_states in zip(weights_tensors, [hidden_states1, hidden_states2]):
                for i in range(weight.shape[1]):
                    hidden_states[:, i * 75 + 1 : i * 75 + 76] = hidden_states[:, i * 75 + 1 : i * 75 + 76] * weight[:, i, 1:-1].unsqueeze(
                        -1
                    )

        return [hidden_states1, hidden_states2, pool2]
