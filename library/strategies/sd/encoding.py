from typing import Any

import torch

from library.strategies.sd.tokenization import tokenize_sd_captions
from transformers import CLIPTokenizer

from library.strategies.base.contracts import TextEncodingStrategy


def get_hidden_states_sd(
    input_ids: torch.Tensor,
    tokenizer: CLIPTokenizer,
    text_encoder: torch.nn.Module,
    clip_skip: int | None = None,
    weight_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Encode SD/SD2 CLIP input IDs into hidden states."""
    batch_size = input_ids.size(0)
    max_token_length = input_ids.size(1) * input_ids.size(2)
    model_max_length = tokenizer.model_max_length

    flat_input_ids = input_ids.reshape((-1, model_max_length)).to(text_encoder.device)

    if clip_skip is None:
        encoder_hidden_states = text_encoder(flat_input_ids)[0]
    else:
        enc_out = text_encoder(flat_input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    encoder_hidden_states = encoder_hidden_states.reshape((batch_size, -1, encoder_hidden_states.shape[-1]))

    if max_token_length != model_max_length:
        if tokenizer.pad_token_id != tokenizer.eos_token_id:
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]
            for i in range(1, max_token_length, model_max_length):
                chunk = encoder_hidden_states[:, i : i + model_max_length - 2]
                if i > 0:
                    for j in range(len(chunk)):
                        if flat_input_ids[j, 1] == tokenizer.eos_token:
                            chunk[j, 0] = chunk[j, 1]
                states_list.append(chunk)
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
            encoder_hidden_states = torch.cat(states_list, dim=1)
        else:
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]
            for i in range(1, max_token_length, model_max_length):
                states_list.append(encoder_hidden_states[:, i : i + model_max_length - 2])
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
            encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def apply_hidden_state_weights_sd(encoder_hidden_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Apply prompt weights to SD hidden states."""
    weights = weights.to(encoder_hidden_states.device)

    if weights.shape[1] == 1:
        return encoder_hidden_states * weights.squeeze(1).unsqueeze(2)

    for i in range(weights.shape[1]):
        encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] = encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] * weights[
            :, i, 1:-1
        ].unsqueeze(-1)

    return encoder_hidden_states


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

    def __init__(self, tokenizer: CLIPTokenizer | None = None, clip_skip: int | None = None) -> None:
        self._tokenizer = tokenizer
        self._clip_skip = clip_skip
        self.clip_skip = clip_skip

    def _get_tokenizer(self) -> CLIPTokenizer:
        """Resolve tokenizer state from explicit construction or the tokenization facet."""
        if self._tokenizer is not None:
            return self._tokenizer

        tokenizers = getattr(self, "tokenizers", None)
        if not tokenizers:
            raise RuntimeError(
                "SD text encoding requires a tokenizer. Pass one to "
                "SdTextEncodingStrategy(...) or mix in SdTokenizeStrategy."
            )
        return tokenizers[0]

    def _get_clip_skip(self) -> int | None:
        """Resolve clip-skip from explicit construction or owning strategy state."""
        if self._clip_skip is not None:
            return self._clip_skip
        return getattr(self, "clip_skip", None)

    def encode_tokens(self, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens.

        Args:
            models: List of models
            tokens: List of token tensors

        Returns:
            List of encoded tensors
        """
        return encode_sd_tokens(self._get_tokenizer(), self._get_clip_skip(), models, tokens)

    def encode_tokens_with_weights(
        self,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return encode_sd_tokens_with_weights(self._get_tokenizer(), self._get_clip_skip(), models, tokens, weights)

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
