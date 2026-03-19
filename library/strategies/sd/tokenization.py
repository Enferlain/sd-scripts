from typing import cast

import torch
from transformers import CLIPTokenizer

from library.constants import V2_STABLE_DIFFUSION_ID, TOKENIZER_ID
from library.models.sd.tokenizer import get_clip_input_ids, load_tokenizer, tokenize_clip_captions
from library.strategies.base.training import TokenizationStrategy
import logging


logger = logging.getLogger(__name__)


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


def tokenize_sd_text_with_weights(tokenizer: CLIPTokenizer, max_length: int, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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
