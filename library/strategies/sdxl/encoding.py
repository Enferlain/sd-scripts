from typing import Any

import torch

from library.models.sdxl.text_encoder import encode_input_ids_sdxl, apply_hidden_state_weights_sdxl
from library.models.sd.tokenizer import tokenize_clip_captions
from transformers import CLIPTokenizer

from library.strategies.base.training import TextEncodingStrategy


def encode_sdxl_tokens(
    tokenizers: list[CLIPTokenizer],
    models: list[Any],
    tokens: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Encode SDXL token tensors using loaded tokenizer state."""
    if len(models) == 2:
        text_encoder1, text_encoder2 = models
        unwrapped_text_encoder2 = None
    else:
        text_encoder1, text_encoder2, unwrapped_text_encoder2 = models
    tokens1, tokens2 = tokens
    tokenizer1, tokenizer2 = tokenizers

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


def encode_sdxl_tokens_with_weights(
    tokenizers: list[CLIPTokenizer],
    models: list[Any],
    tokens: list[torch.Tensor],
    weights: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Encode SDXL token tensors and apply prompt weights."""
    hidden_states1, hidden_states2, pool2 = encode_sdxl_tokens(tokenizers, models, tokens)
    hidden_states1, hidden_states2 = apply_hidden_state_weights_sdxl(
        hidden_states1,
        hidden_states2,
        weights[0],
        weights[1],
    )
    return [hidden_states1, hidden_states2, pool2]


class SdxlTextEncodingStrategy(TextEncodingStrategy):
    """
    Text encoding strategy for SDXL.
    """

    def __init__(self, tokenizers: list[CLIPTokenizer]) -> None:
        self.tokenizers = tokenizers

    def encode_tokens(self, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens.

        Args:
            models: List of models, [text_encoder1, text_encoder2, unwrapped text_encoder2 (optional)].
                If text_encoder2 is wrapped by accelerate, unwrapped_text_encoder2 is required
            tokens: List of tokens, for text_encoder1 and text_encoder2

        Returns:
            List of encoded tensors
        """
        return encode_sdxl_tokens(self.tokenizers, models, tokens)

    def encode_tokens_with_weights(
        self,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Encode tokens with weights.

        Args:
            models: List of models
            tokens: List of token tensors
            weights: List of weight tensors

        Returns:
            List of encoded tensors
        """
        return encode_sdxl_tokens_with_weights(self.tokenizers, models, tokens, weights)

    def encode_te_outputs_in_memory(
        self,
        text_encoders: list[Any],
        tokenizers: list[Any],
        caption: str,
        max_token_length: int,
        device: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute SDXL text encoder outputs for a single caption."""
        input_ids1 = tokenize_clip_captions(tokenizers[0], [caption], max_token_length).to(device)
        input_ids2 = tokenize_clip_captions(tokenizers[1], [caption], max_token_length).to(device)

        with torch.no_grad():
            hidden_state1, hidden_state2, pool2 = encode_input_ids_sdxl(
                input_ids1,
                input_ids2,
                tokenizers[0],
                tokenizers[1],
                text_encoders[0],
                text_encoders[1],
            )
        return {
            "hidden_state1": hidden_state1.squeeze(0).cpu(),
            "hidden_state2": hidden_state2.squeeze(0).cpu(),
            "pool2": pool2.squeeze(0).cpu(),
        }

    def get_models_for_text_encoding(self, cfg: Any, accelerator: Any, text_encoders: list[Any]) -> list[Any]:
        """Return SDXL text encoders plus unwrapped text encoder 2."""
        return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]
