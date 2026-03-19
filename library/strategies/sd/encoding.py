from typing import Any

import torch

from library.models.sd.text_encoder import get_hidden_states_sd, apply_hidden_state_weights_sd
from library.strategies.sd.tokenization import tokenize_sd_captions
from transformers import CLIPTokenizer

from library.strategies.base.training import TextEncodingStrategy


def encode_sd_tokens(tokenizer: CLIPTokenizer, clip_skip: int | None, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
    """Encode SD token tensors using a loaded tokenizer and optional clip-skip."""
    text_encoder = models[0]
    tokens_tensor = tokens[0]

    encoder_hidden_states = get_hidden_states_sd(
        tokens_tensor,
        tokenizer,
        text_encoder,
        clip_skip=clip_skip,
    )
    return [encoder_hidden_states]


def encode_sd_tokens_with_weights(
    tokenizer: CLIPTokenizer,
    clip_skip: int | None,
    models: list[Any],
    tokens: list[torch.Tensor],
    weights: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Encode SD token tensors and apply prompt weights."""
    encoder_hidden_states = encode_sd_tokens(tokenizer, clip_skip, models, tokens)[0]
    return [apply_hidden_state_weights_sd(encoder_hidden_states, weights[0])]


class SdTextEncodingStrategy(TextEncodingStrategy):
    """
    Text encoding strategy for SD1.5 and SD2.0.
    """

    def __init__(self, tokenizer: CLIPTokenizer, clip_skip: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.clip_skip = clip_skip

    def encode_tokens(self, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens.

        Args:
            models: List of models
            tokens: List of token tensors

        Returns:
            List of encoded tensors
        """
        return encode_sd_tokens(self.tokenizer, self.clip_skip, models, tokens)

    def encode_tokens_with_weights(
        self,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return encode_sd_tokens_with_weights(self.tokenizer, self.clip_skip, models, tokens, weights)

    def encode_te_outputs_in_memory(
        self,
        text_encoders: list[Any],
        tokenizers: list[Any],
        caption: str,
        max_token_length: int,
        device: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute SD text encoder outputs for a single caption."""
        input_ids = tokenize_sd_captions(tokenizers[0], [caption], max_token_length).to(device)

        with torch.no_grad():
            hidden_state = get_hidden_states_sd(input_ids, tokenizers[0], text_encoders[0], clip_skip=self.clip_skip)
        return {"hidden_state": hidden_state.squeeze(0).cpu()}

    def get_models_for_text_encoding(self, cfg: Any, accelerator: Any, text_encoders: list[Any]) -> list[Any]:
        """Return SD text encoders as-is for live encoding."""
        return text_encoders
