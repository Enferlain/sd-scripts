import logging
import os
from typing import Any

import torch

from transformers import CLIPTokenizer

from library.constants import re_attention


logger = logging.getLogger(__name__)


def load_tokenizer(
    model_class: Any,
    model_id: str,
    subfolder: str | None = None,
    tokenizer_cache_dir: str | None = None,
) -> Any:
    """
    Load a tokenizer from cache or the upstream model repo.
    """
    tokenizer = None
    local_tokenizer_path: str | None = None

    if tokenizer_cache_dir:
        local_tokenizer_path = os.path.join(tokenizer_cache_dir, model_id.replace("/", "_"))
        if os.path.exists(local_tokenizer_path):
            logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
            tokenizer = model_class.from_pretrained(local_tokenizer_path)

    if tokenizer is None:
        tokenizer = model_class.from_pretrained(model_id, subfolder=subfolder)

    if local_tokenizer_path is not None and not os.path.exists(local_tokenizer_path):
        logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
        tokenizer.save_pretrained(local_tokenizer_path)

    return tokenizer


def parse_prompt_attention(text: str) -> list[list[str | float]]:
    r"""
    Parse prompt attention syntax into ``[text, weight]`` spans.

    Accepted tokens are:
    - ``(abc)`` -> increases attention to ``abc`` by 1.1
    - ``(abc:3.12)`` -> increases attention to ``abc`` by 3.12
    - ``[abc]`` -> decreases attention to ``abc`` by 1 / 1.1
    """

    res: list[list[str | float]] = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position: int, multiplier: float) -> None:
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for match in re_attention.finditer(text):
        token = match.group(0)
        weight = match.group(1)

        if token.startswith("\\"):
            res.append([token[1:], 1.0])
        elif token == "(":
            round_brackets.append(len(res))
        elif token == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif token == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif token == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([token, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        return [["", 1.0]]

    # Merge adjacent runs with identical weights.
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] = str(res[i][0]) + str(res[i + 1][0])
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_clip_weighted_input_ids(
    tokenizer: CLIPTokenizer,
    text: str,
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build weighted CLIP-family token ids for SD/SDXL prompt syntax.

    ``max_length`` includes BOS/EOS when provided.
    """

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
        tokens, weights, effective_max_length, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    )
    return torch.tensor(tokens).unsqueeze(0), torch.tensor(weights).unsqueeze(0)


def get_clip_input_ids(
    tokenizer: CLIPTokenizer,
    text: str,
    max_length: int | None = None,
    weighted: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Build CLIP-family token ids for the current SD / SDXL tokenizer behavior.

    ``max_length`` follows the existing strategy-layer convention: if
    provided, it already includes BOS/EOS.
    """

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
            # v1-style CLIP tokenizers use EOS as pad.
            for i in range(1, effective_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                ids_chunk = (
                    input_ids[0].unsqueeze(0),
                    input_ids[i : i + tokenizer.model_max_length - 2],
                    input_ids[-1].unsqueeze(0),
                )
                iids_list.append(torch.cat(ids_chunk))
        else:
            # v2/SDXL-style tokenizers preserve PAD separately from EOS.
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
                new_weights[batch_index, 1 : 1 + tokenizer.model_max_length - 2] = weights[
                    i : i + tokenizer.model_max_length - 2
                ]
            weights = new_weights

    if weighted and weights is not None:
        return input_ids, weights
    return input_ids


def tokenize_clip_captions(tokenizer: CLIPTokenizer, captions: list[str], max_token_length: int | None) -> torch.Tensor:
    """
    Tokenize captions using the current CLIP-family chunking rules.

    ``max_token_length`` follows training config semantics and excludes
    BOS/EOS when provided.
    """

    effective_max_length = tokenizer.model_max_length if max_token_length is None else max_token_length + 2
    input_ids = [get_clip_input_ids(tokenizer, caption, effective_max_length) for caption in captions]
    return torch.stack(input_ids, dim=0)
