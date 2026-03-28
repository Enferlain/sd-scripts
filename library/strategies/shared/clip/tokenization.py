"""Shared CLIP-family tokenization behavior helpers."""

import logging
from typing import cast

import torch
from transformers import CLIPTokenizer

from library.data.prompt_utils import parse_prompt_attention


logger = logging.getLogger(__name__)


def get_clip_weighted_input_ids(
    tokenizer: CLIPTokenizer,
    text: str,
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build weighted CLIP-family token ids for prompt syntax handling."""

    def get_prompts_with_weights(prompt: str, content_max_length: int) -> tuple[list[int], list[float]]:
        truncated = False
        texts_and_weights = parse_prompt_attention(prompt)
        tokens: list[int] = []
        weights: list[float] = []

        for word, weight in texts_and_weights:
            token = tokenizer(word).input_ids[1:-1]
            tokens += token
            weights += [float(weight)] * len(token)
            if len(tokens) > content_max_length:
                truncated = True
                break

        if len(tokens) > content_max_length:
            truncated = True
            tokens = tokens[:content_max_length]
            weights = weights[:content_max_length]

        if truncated:
            logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")

        return tokens, weights

    def pad_tokens_and_weights(
        tokens: list[int],
        weights: list[float],
        padded_max_length: int,
        bos: int,
        eos: int,
        pad: int,
    ) -> tuple[list[int], list[float]]:
        tokens = [bos] + tokens + [eos] + [pad] * (padded_max_length - 2 - len(tokens))
        weights = [1.0] + weights + [1.0] * (padded_max_length - 1 - len(weights))
        return tokens, weights

    effective_max_length = tokenizer.model_max_length if max_length is None else max_length
    tokens, weights = get_prompts_with_weights(text, effective_max_length - 2)
    tokens, weights = pad_tokens_and_weights(
        tokens,
        weights,
        effective_max_length,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    )
    return torch.tensor(tokens).unsqueeze(0), torch.tensor(weights).unsqueeze(0)


def get_clip_input_ids(
    tokenizer: CLIPTokenizer,
    text: str,
    max_length: int | None = None,
    weighted: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Build CLIP-family token ids with the current long-prompt chunking rules."""
    effective_max_length = tokenizer.model_max_length - 2 if max_length is None else max_length

    weights: torch.Tensor | None = None
    if weighted:
        input_ids, weights = get_clip_weighted_input_ids(tokenizer, text, effective_max_length)
    else:
        input_ids = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=effective_max_length,
            return_tensors="pt",
        ).input_ids
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

    if effective_max_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            for i in range(1, effective_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                ids_chunk = (
                    input_ids[0].unsqueeze(0),
                    input_ids[i : i + tokenizer.model_max_length - 2],
                    input_ids[-1].unsqueeze(0),
                )
                iids_list.append(torch.cat(ids_chunk))
        else:
            for i in range(1, effective_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                ids_chunk = torch.cat(
                    (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                )

                if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                    ids_chunk[-1] = tokenizer.eos_token_id
                if ids_chunk[1] == tokenizer.pad_token_id:
                    ids_chunk[1] = tokenizer.eos_token_id

                iids_list.append(ids_chunk)

        input_ids = torch.stack(iids_list)

        if weighted and weights is not None:
            weights = weights.squeeze(0)
            new_weights = torch.ones(input_ids.shape)
            for i in range(1, effective_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                batch_index = i // (tokenizer.model_max_length - 2)
                new_weights[batch_index, 1 : 1 + tokenizer.model_max_length - 2] = weights[i : i + tokenizer.model_max_length - 2]
            weights = new_weights

    if weighted and weights is not None:
        return input_ids, weights
    return input_ids


def tokenize_clip_captions(tokenizer: CLIPTokenizer, captions: list[str], max_token_length: int | None) -> torch.Tensor:
    """Tokenize captions using the current CLIP-family chunking rules."""
    effective_max_length = tokenizer.model_max_length if max_token_length is None else max_token_length + 2
    input_ids = [cast(torch.Tensor, get_clip_input_ids(tokenizer, caption, effective_max_length)) for caption in captions]
    return torch.stack(input_ids, dim=0)


__all__ = [
    "get_clip_input_ids",
    "get_clip_weighted_input_ids",
    "tokenize_clip_captions",
]
