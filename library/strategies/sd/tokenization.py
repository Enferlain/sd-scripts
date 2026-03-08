from typing import cast

import torch
from transformers import CLIPTokenizer

from library.constants import V2_STABLE_DIFFUSION_ID, TOKENIZER_ID
from library.strategies.base.tokenization import TokenizeStrategy
import logging


logger = logging.getLogger(__name__)


def tokenize_sd_captions(tokenizer: CLIPTokenizer, captions: list[str], max_token_length: int | None) -> torch.Tensor:
    """Tokenize SD captions from an already-loaded tokenizer instance.

    This mirrors ``SdTokenizeStrategy.tokenize()`` without requiring callers to
    construct a full strategy object just to reuse ``TokenizeStrategy`` chunking
    logic for long prompts.
    """
    base_strategy = TokenizeStrategy()
    if max_token_length is None:
        effective_max_length = tokenizer.model_max_length
    else:
        effective_max_length = max_token_length + 2

    input_ids = [cast(torch.Tensor, base_strategy._get_input_ids(tokenizer, caption, effective_max_length)) for caption in captions]
    return torch.stack(input_ids, dim=0)


class SdTokenizeStrategy(TokenizeStrategy):
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
        logger.info(f"Using {'v2' if v2 else 'v1'} tokenizer")
        if v2:
            self.tokenizer = self._load_tokenizer(
                CLIPTokenizer, V2_STABLE_DIFFUSION_ID, subfolder="tokenizer", tokenizer_cache_dir=tokenizer_cache_dir
            )
        else:
            self.tokenizer = self._load_tokenizer(CLIPTokenizer, TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)

        if max_length is None:
            self.max_length = self.tokenizer.model_max_length
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
        input_ids = [cast(torch.Tensor, self._get_input_ids(self.tokenizer, t, self.max_length)) for t in text]
        return [torch.stack(input_ids, dim=0)]

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Tokenize text with weights.

        Args:
            text: Text or list of text to tokenize

        Returns:
            Tuple of lists of token tensors and weight tensors
        """
        text = [text] if isinstance(text, str) else text
        tokens_list = []
        weights_list = []
        for t in text:
            tokens, weights = self._get_input_ids(self.tokenizer, t, self.max_length, weighted=True)
            tokens_list.append(tokens)
            weights_list.append(weights)
        return [torch.stack(tokens_list, dim=0)], [torch.stack(weights_list, dim=0)]
