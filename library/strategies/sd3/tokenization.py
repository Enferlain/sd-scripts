import logging
from typing import Any, cast

import torch
from transformers import CLIPTokenizer, T5TokenizerFast

from library.models.sd.tokenizer import load_tokenizer
from library.strategies.base.contracts import TokenizationStrategy


logger = logging.getLogger(__name__)


CLIP_L_TOKENIZER_ID = "openai/clip-vit-large-patch14"
CLIP_G_TOKENIZER_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
T5_XXL_TOKENIZER_ID = "google/t5-v1_1-xxl"
DEFAULT_SD3_T5_MAX_LENGTH = 256

def resolve_sd3_t5_max_length(max_length: int | None) -> int:
    """Resolve the effective SD3 T5 sequence length."""
    return DEFAULT_SD3_T5_MAX_LENGTH if max_length is None else max_length


def build_sd3_tokenizers(
    t5xxl_max_length: int | None,
    tokenizer_cache_dir: str | None = None,
) -> tuple[list[Any], int]:
    """Load the SD3 tokenizer runtime state used by training strategies."""
    tokenizer_l = load_tokenizer(CLIPTokenizer, CLIP_L_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
    tokenizer_g = load_tokenizer(CLIPTokenizer, CLIP_G_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
    tokenizer_t5 = load_tokenizer(T5TokenizerFast, T5_XXL_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
    tokenizer_g.pad_token_id = 0
    resolved_t5_max_length = resolve_sd3_t5_max_length(t5xxl_max_length)
    return [tokenizer_l, tokenizer_g, tokenizer_t5], resolved_t5_max_length


def tokenize_sd3_text(
    tokenizer_l: CLIPTokenizer,
    tokenizer_g: CLIPTokenizer,
    tokenizer_t5: T5TokenizerFast,
    t5xxl_max_length: int,
    text: str | list[str],
) -> list[torch.Tensor]:
    """Tokenize SD3 text into CLIP-L, CLIP-G, and T5 tensors plus attention masks."""
    text = [text] if isinstance(text, str) else text

    l_tokens = tokenizer_l(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    g_tokens = tokenizer_g(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    t5_tokens = tokenizer_t5(text, max_length=t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt")

    return [
        cast(torch.Tensor, l_tokens["input_ids"]),
        cast(torch.Tensor, g_tokens["input_ids"]),
        cast(torch.Tensor, t5_tokens["input_ids"]),
        cast(torch.Tensor, l_tokens["attention_mask"]),
        cast(torch.Tensor, g_tokens["attention_mask"]),
        cast(torch.Tensor, t5_tokens["attention_mask"]),
    ]


def tokenize_sd3_captions(
    tokenizers: list[Any],
    captions: list[str],
    max_token_length: int | None,
) -> list[torch.Tensor]:
    """Tokenize captions for per-epoch token caching."""
    tokenizer_l, tokenizer_g, tokenizer_t5 = tokenizers
    t5_max_length = resolve_sd3_t5_max_length(max_token_length)
    tokens = tokenize_sd3_text(tokenizer_l, tokenizer_g, tokenizer_t5, t5_max_length, captions)
    return tokens[:3]


class Sd3TokenizeStrategy(TokenizationStrategy):
    """Tokenization strategy for SD3."""

    def __init__(self, t5xxl_max_length: int | None = None, tokenizer_cache_dir: str | None = None) -> None:
        tokenizers, self.t5xxl_max_length = build_sd3_tokenizers(t5xxl_max_length, tokenizer_cache_dir)
        self.tokenizer_l, self.tokenizer_g, self.tokenizer_t5 = tokenizers

    @property
    def tokenizers(self) -> list[Any]:
        """Return the tokenizer instances owned by this helper strategy."""
        return [self.tokenizer_l, self.tokenizer_g, self.tokenizer_t5]

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        """Tokenize text into SD3 token ids and attention masks."""
        return tokenize_sd3_text(self.tokenizer_l, self.tokenizer_g, self.tokenizer_t5, self.t5xxl_max_length, text)

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """SD3 upstream does not define weighted prompt tokenization in the current port."""
        raise NotImplementedError("SD3 prompt weighting is not implemented in the current strategy port")

    def tokenize_captions(self, tokenizers: list[Any], captions: list[str], max_token_length: int) -> list[torch.Tensor]:
        """Tokenize captions for SD3 per-epoch token caching."""
        return tokenize_sd3_captions(tokenizers, captions, max_token_length)
