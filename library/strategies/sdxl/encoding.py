from typing import Any

import torch

from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from library.strategies.shared.clip.tokenization import tokenize_clip_captions
from transformers import CLIPTokenizer

from library.strategies.base.contracts import TextEncodingStrategy


def pool_workaround(
    text_encoder: CLIPTextModelWithProjection | torch.nn.Module, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
) -> torch.Tensor:
    """Work around CLIP-G pooling returning the max-token state instead of EOS."""
    eos_token_mask = (input_ids == eos_token_id).int()
    eos_token_index = torch.argmax(eos_token_mask, dim=1)
    eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]
    pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
    pooled_output = pooled_output.to(last_hidden_state.dtype)
    return pooled_output


def get_hidden_states_sdxl(
    max_token_length: int | None,
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel | torch.nn.Module,
    text_encoder2: CLIPTextModelWithProjection | torch.nn.Module,
    weight_dtype: torch.dtype | None = None,
    accelerator: Accelerator | None = None,
    unwrapped_text_encoder2: CLIPTextModelWithProjection | torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get hidden states for SDXL from its dual text encoders."""
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))

    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]

    pool_encoder2 = unwrapped_text_encoder2
    if pool_encoder2 is None:
        pool_encoder2 = text_encoder2 if accelerator is None else accelerator.unwrap_model(text_encoder2)
    pool2 = pool_workaround(pool_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        states_list = [hidden_states1[:, 0].unsqueeze(1)]
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i : i + tokenizer1.model_max_length - 2])
        states_list.append(hidden_states1[:, -1].unsqueeze(1))
        hidden_states1 = torch.cat(states_list, dim=1)

        states_list = [hidden_states2[:, 0].unsqueeze(1)]
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2]
            states_list.append(chunk)
        states_list.append(hidden_states2[:, -1].unsqueeze(1))
        hidden_states2 = torch.cat(states_list, dim=1)

        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def encode_input_ids_sdxl(
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel | torch.nn.Module,
    text_encoder2: CLIPTextModelWithProjection | torch.nn.Module,
    weight_dtype: torch.dtype | None = None,
    unwrapped_text_encoder2: CLIPTextModelWithProjection | torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode SDXL token tensors into hidden states and pooled output."""
    max_token_length = None if input_ids1.size(1) == 1 else input_ids1.size(1) * input_ids1.size(2)

    input_ids1 = input_ids1.to(next(text_encoder1.parameters()).device)
    input_ids2 = input_ids2.to(next(text_encoder2.parameters()).device)

    return get_hidden_states_sdxl(
        max_token_length,
        input_ids1,
        input_ids2,
        tokenizer1,
        tokenizer2,
        text_encoder1,
        text_encoder2,
        weight_dtype=weight_dtype,
        accelerator=None,
        unwrapped_text_encoder2=unwrapped_text_encoder2,
    )


def apply_hidden_state_weights_sdxl(
    hidden_states1: torch.Tensor,
    hidden_states2: torch.Tensor,
    weights1: torch.Tensor,
    weights2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply prompt weights to SDXL hidden states."""
    weights_tensors = [weights1.to(hidden_states1.device), weights2.to(hidden_states1.device)]

    if weights_tensors[0].shape[1] == 1:
        hidden_states1 = hidden_states1 * weights_tensors[0].squeeze(1).unsqueeze(2)
        hidden_states2 = hidden_states2 * weights_tensors[1].squeeze(1).unsqueeze(2)
        return hidden_states1, hidden_states2

    for weight, hidden_states in zip(weights_tensors, [hidden_states1, hidden_states2]):
        for i in range(weight.shape[1]):
            hidden_states[:, i * 75 + 1 : i * 75 + 76] = hidden_states[:, i * 75 + 1 : i * 75 + 76] * weight[:, i, 1:-1].unsqueeze(-1)

    return hidden_states1, hidden_states2


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

    def __init__(self, tokenizers: list[CLIPTokenizer] | None = None) -> None:
        self._tokenizers = tokenizers

    def _get_tokenizers(self) -> list[CLIPTokenizer]:
        """Resolve tokenizer state from explicit construction or the tokenization facet."""
        tokenizers = getattr(self, "_tokenizers", None)
        if tokenizers is not None:
            return tokenizers

        tokenizers = getattr(self, "tokenizers", None)
        if tokenizers is None:
            raise RuntimeError(
                "SDXL text encoding requires tokenizers. Pass them to "
                "SdxlTextEncodingStrategy(...) or mix in SdxlTokenizeStrategy."
            )
        return tokenizers

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
        return encode_sdxl_tokens(self._get_tokenizers(), models, tokens)

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
        return encode_sdxl_tokens_with_weights(self._get_tokenizers(), models, tokens, weights)

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
