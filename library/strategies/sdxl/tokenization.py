from typing import cast

import torch
from transformers import CLIPTokenizer

from library.constants import TOKENIZER1_PATH, TOKENIZER2_PATH
from library.strategies.base.tokenization import TokenizeStrategy


class SdxlTokenizeStrategy(TokenizeStrategy):
    """
    Tokenize strategy for SDXL.
    """

    def __init__(self, max_length: int | None, tokenizer_cache_dir: str | None = None) -> None:
        """
        Args:
            max_length: Max length of tokens
            tokenizer_cache_dir: Directory to cache the tokenizer
        """
        self.tokenizer1 = self._load_tokenizer(CLIPTokenizer, TOKENIZER1_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
        self.tokenizer2 = self._load_tokenizer(CLIPTokenizer, TOKENIZER2_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
        self.tokenizer2.pad_token_id = 0  # use 0 as pad token for tokenizer2

        if max_length is None:
            self.max_length = self.tokenizer1.model_max_length
        else:
            self.max_length = max_length + 2

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        """
        Tokenize text.

        Args:
            text: Text or list of text to tokenize

        Returns:
            List of token tensors
        """
        text = [text] if isinstance(text, str) else text
        # _get_input_ids returns Tensor when weighted=False (default)
        return [
            torch.stack([cast(torch.Tensor, self._get_input_ids(self.tokenizer1, t, self.max_length)) for t in text], dim=0),
            torch.stack([cast(torch.Tensor, self._get_input_ids(self.tokenizer2, t, self.max_length)) for t in text], dim=0),
        ]

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Tokenize text with weights.

        Args:
            text: Text or list of text to tokenize

        Returns:
            Tuple of lists of token tensors and weight tensors
        """
        text = [text] if isinstance(text, str) else text
        tokens1_list, tokens2_list = [], []
        weights1_list, weights2_list = [], []
        for t in text:
            tokens1, weights1 = self._get_input_ids(self.tokenizer1, t, self.max_length, weighted=True)
            tokens2, weights2 = self._get_input_ids(self.tokenizer2, t, self.max_length, weighted=True)
            tokens1_list.append(tokens1)
            tokens2_list.append(tokens2)
            weights1_list.append(weights1)
            weights2_list.append(weights2)
        return [torch.stack(tokens1_list, dim=0), torch.stack(tokens2_list, dim=0)], [
            torch.stack(weights1_list, dim=0),
            torch.stack(weights2_list, dim=0),
        ]
