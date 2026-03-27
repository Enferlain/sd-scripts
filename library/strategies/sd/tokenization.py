import logging
import os
from typing import Any, cast

import torch
from transformers import CLIPTokenizer

from library.constants import V2_STABLE_DIFFUSION_ID, TOKENIZER_ID
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
    """Build weighted CLIP-family token ids for SD prompt syntax."""

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
    """Build CLIP-family token ids for SD tokenization behavior."""
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


def build_sd_tokenizers(v2: bool, max_length: int | None, tokenizer_cache_dir: str | None = None) -> tuple[list[CLIPTokenizer], int]:
    """Load the SD tokenizer runtime state used by training strategies."""
    logger.info(f"Using {'v2' if v2 else 'v1'} tokenizer")
    if v2:
        tokenizer = load_tokenizer(CLIPTokenizer, V2_STABLE_DIFFUSION_ID, subfolder="tokenizer", tokenizer_cache_dir=tokenizer_cache_dir)
    else:
        tokenizer = load_tokenizer(CLIPTokenizer, TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)

    resolved_max_length = tokenizer.model_max_length if max_length is None else max_length + 2
    return [tokenizer], resolved_max_length


def tokenize_sd_captions(tokenizer: CLIPTokenizer, captions: list[str], max_token_length: int | None) -> torch.Tensor:
    """Tokenize SD captions from an already-loaded tokenizer instance.

    This mirrors ``SdTokenizeStrategy.tokenize()`` without requiring callers to
    construct a full strategy object.
    """
    return tokenize_clip_captions(tokenizer, captions, max_token_length)


def tokenize_sd_text(tokenizer: CLIPTokenizer, max_length: int, text: str | list[str]) -> list[torch.Tensor]:
    """Tokenize SD text for training/runtime use without a separate strategy object."""
    text = [text] if isinstance(text, str) else text
    input_ids = [cast(torch.Tensor, get_clip_input_ids(tokenizer, t, max_length)) for t in text]
    return [torch.stack(input_ids, dim=0)]


def tokenize_sd_text_with_weights(
    tokenizer: CLIPTokenizer, max_length: int, text: str | list[str]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Tokenize SD text and return prompt weights for training/runtime use."""
    text = [text] if isinstance(text, str) else text
    tokens_list = []
    weights_list = []
    for t in text:
        tokens, weights = get_clip_input_ids(tokenizer, t, max_length, weighted=True)
        tokens_list.append(tokens)
        weights_list.append(weights)
    return [torch.stack(tokens_list, dim=0)], [torch.stack(weights_list, dim=0)]


class SdTokenizeStrategy(TokenizationStrategy):
    """
    Tokenize strategy for SD1.5 and SD2.0.
    """

    def __init__(self, v2: bool, max_length: int | None, tokenizer_cache_dir: str | None = None) -> None:
        """
        Args:
            v2: Whether to use v2 tokenizer
            max_length: Max length of tokens. max_length does not include <BOS> and <EOS> (None, 75, 150, 225)
            tokenizer_cache_dir: Directory to cache the tokenizer
        """
        tokenizers, self.max_length = build_sd_tokenizers(v2, max_length, tokenizer_cache_dir)
        self.tokenizer = tokenizers[0]

    @property
    def tokenizers(self) -> list[CLIPTokenizer]:
        """Return the tokenizer instances owned by this helper strategy."""
        return [self.tokenizer]

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        """
        Tokenize text.

        Args:
            text: Text or list of text to tokenize

        Returns:
            List of token tensors
        """
        return tokenize_sd_text(self.tokenizer, self.max_length, text)

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Tokenize text with weights.

        Args:
            text: Text or list of text to tokenize

        Returns:
            Tuple of lists of token tensors and weight tensors
        """
        return tokenize_sd_text_with_weights(self.tokenizer, self.max_length, text)

    def tokenize_captions(self, tokenizers: list[CLIPTokenizer], captions: list[str], max_token_length: int) -> list[torch.Tensor]:
        """Tokenize captions using the SD helper strategy's CLIP tokenizer."""
        return [tokenize_sd_captions(tokenizers[0], captions, max_token_length)]
