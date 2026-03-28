import logging
from typing import cast

import torch
from transformers import CLIPTokenizer

from library.constants import TOKENIZER1_PATH, TOKENIZER2_PATH
from library.models.sd.tokenizer import load_tokenizer
from library.strategies.base.contracts import TokenizationStrategy
from library.strategies.shared.clip.tokenization import get_clip_input_ids, tokenize_clip_captions


logger = logging.getLogger(__name__)

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
