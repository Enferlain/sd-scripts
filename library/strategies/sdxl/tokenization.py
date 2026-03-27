import logging
import os
from typing import Any, cast

import torch
from transformers import CLIPTokenizer

from library.constants import TOKENIZER1_PATH, TOKENIZER2_PATH
from library.data.prompt_utils import parse_prompt_attention
from library.strategies.base.contracts import TokenizationStrategy


logger = logging.getLogger(__name__)


def load_tokenizer(
    model_class: Any,
    model_id: str,
    subfolder: str | None = None,
    tokenizer_cache_dir: str | None = None,
) -> Any:
    """Load a tokenizer from cache or the upstream model repo."""
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


def get_clip_weighted_input_ids(
    tokenizer: CLIPTokenizer,
    text: str,
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build weighted CLIP-family token ids for SDXL prompt syntax."""

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
    """Build CLIP-family token ids for SDXL tokenization behavior."""
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
    input_ids = [get_clip_input_ids(tokenizer, caption, effective_max_length) for caption in captions]
    return torch.stack(input_ids, dim=0)


def build_sdxl_tokenizers(max_length: int | None, tokenizer_cache_dir: str | None = None) -> tuple[list[CLIPTokenizer], int]:
    """Load the SDXL tokenizer runtime state used by training strategies."""
    tokenizer1 = load_tokenizer(CLIPTokenizer, TOKENIZER1_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
    tokenizer2 = load_tokenizer(CLIPTokenizer, TOKENIZER2_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
    tokenizer2.pad_token_id = 0

    resolved_max_length = tokenizer1.model_max_length if max_length is None else max_length + 2
    return [tokenizer1, tokenizer2], resolved_max_length


def tokenize_sdxl_text(tokenizer1: CLIPTokenizer, tokenizer2: CLIPTokenizer, max_length: int, text: str | list[str]) -> list[torch.Tensor]:
    """Tokenize SDXL text for training/runtime use without a separate strategy object."""
    text = [text] if isinstance(text, str) else text
    return [
        torch.stack([cast(torch.Tensor, get_clip_input_ids(tokenizer1, t, max_length)) for t in text], dim=0),
        torch.stack([cast(torch.Tensor, get_clip_input_ids(tokenizer2, t, max_length)) for t in text], dim=0),
    ]


def tokenize_sdxl_text_with_weights(
    tokenizer1: CLIPTokenizer, tokenizer2: CLIPTokenizer, max_length: int, text: str | list[str]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Tokenize SDXL text and return prompt weights for training/runtime use."""
    text = [text] if isinstance(text, str) else text
    tokens1_list, tokens2_list = [], []
    weights1_list, weights2_list = [], []
    for t in text:
        tokens1, weights1 = get_clip_input_ids(tokenizer1, t, max_length, weighted=True)
        tokens2, weights2 = get_clip_input_ids(tokenizer2, t, max_length, weighted=True)
        tokens1_list.append(tokens1)
        tokens2_list.append(tokens2)
        weights1_list.append(weights1)
        weights2_list.append(weights2)
    return [torch.stack(tokens1_list, dim=0), torch.stack(tokens2_list, dim=0)], [
        torch.stack(weights1_list, dim=0),
        torch.stack(weights2_list, dim=0),
    ]


class SdxlTokenizeStrategy(TokenizationStrategy):
    """
    Tokenize strategy for SDXL.
    """

    def __init__(self, max_length: int | None, tokenizer_cache_dir: str | None = None) -> None:
        """
        Args:
            max_length: Max length of tokens
            tokenizer_cache_dir: Directory to cache the tokenizer
        """
        tokenizers, self.max_length = build_sdxl_tokenizers(max_length, tokenizer_cache_dir)
        self.tokenizer1, self.tokenizer2 = tokenizers

    @property
    def tokenizers(self) -> list[CLIPTokenizer]:
        """Return the tokenizer instances owned by this helper strategy."""
        return [self.tokenizer1, self.tokenizer2]

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        """
        Tokenize text.

        Args:
            text: Text or list of text to tokenize

        Returns:
            List of token tensors
        """
        return tokenize_sdxl_text(self.tokenizer1, self.tokenizer2, self.max_length, text)

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Tokenize text with weights.

        Args:
            text: Text or list of text to tokenize

        Returns:
            Tuple of lists of token tensors and weight tensors
        """
        return tokenize_sdxl_text_with_weights(self.tokenizer1, self.tokenizer2, self.max_length, text)

    def tokenize_captions(self, tokenizers: list[CLIPTokenizer], captions: list[str], max_token_length: int) -> list[torch.Tensor]:
        """Tokenize captions using the SDXL helper strategy's dual tokenizers."""
        return [
            tokenize_clip_captions(tokenizers[0], captions, max_token_length),
            tokenize_clip_captions(tokenizers[1], captions, max_token_length),
        ]
