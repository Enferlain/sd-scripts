import os
import logging
import numpy as np
import torch

from typing import Any, cast
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from library.constants import TOKENIZER1_PATH, TOKENIZER2_PATH
from library.models.text_encoder_util import pool_workaround, get_hidden_states_sdxl
from library.strategies.strategy_base import TokenizeStrategy, TextEncodingStrategy, TextEncoderOutputsCachingStrategy
from library.data.data_structures import ImageInfo
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


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


class SdxlTextEncodingStrategy(TextEncodingStrategy):
    """
    Text encoding strategy for SDXL.
    """

    def __init__(self) -> None:
        pass

    def _pool_workaround(
        self, text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
    ):
        """
        Delegate to shared utility function.
        """
        return pool_workaround(text_encoder, last_hidden_state, input_ids, eos_token_id)

    def _get_hidden_states_sdxl(
        self,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        text_encoder1: CLIPTextModel | torch.nn.Module,
        text_encoder2: CLIPTextModelWithProjection | torch.nn.Module,
        unwrapped_text_encoder2: CLIPTextModelWithProjection | None = None,
    ):
        """
        Wrapper around shared utility that derives max_token_length from input shape.

        The input_ids have shape [b, n, 77] where n is the number of 77-token chunks.
        """
        # Derive max_token_length from input shape
        if input_ids1.size()[1] == 1:
            max_token_length = None
        else:
            max_token_length = input_ids1.size()[1] * input_ids1.size()[2]

        # Flatten and move to device
        input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))
        input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))
        input_ids1 = input_ids1.to(next(text_encoder1.parameters()).device)
        input_ids2 = input_ids2.to(next(text_encoder2.parameters()).device)

        # Use unwrapped encoder for pool workaround if provided
        unwrapped_te2 = unwrapped_text_encoder2 or text_encoder2

        # Call shared utility - pass None for accelerator since we handle unwrapping here
        return get_hidden_states_sdxl(
            max_token_length,
            input_ids1,
            input_ids2,
            tokenizer1,
            tokenizer2,
            text_encoder1,
            unwrapped_te2,
            weight_dtype=None,
            accelerator=None,
        )

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of models, [text_encoder1, text_encoder2, unwrapped text_encoder2 (optional)].
                If text_encoder2 is wrapped by accelerate, unwrapped_text_encoder2 is required
            tokens: List of tokens, for text_encoder1 and text_encoder2

        Returns:
            List of encoded tensors
        """
        if len(models) == 2:
            text_encoder1, text_encoder2 = models
            unwrapped_text_encoder2 = None
        else:
            text_encoder1, text_encoder2, unwrapped_text_encoder2 = models
        tokens1, tokens2 = tokens
        assert isinstance(tokenize_strategy, SdxlTokenizeStrategy)
        sdxl_tokenize_strategy: SdxlTokenizeStrategy = tokenize_strategy
        tokenizer1, tokenizer2 = sdxl_tokenize_strategy.tokenizer1, sdxl_tokenize_strategy.tokenizer2

        hidden_states1, hidden_states2, pool2 = self._get_hidden_states_sdxl(
            tokens1, tokens2, tokenizer1, tokenizer2, text_encoder1, text_encoder2, unwrapped_text_encoder2
        )
        return [hidden_states1, hidden_states2, pool2]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Encode tokens with weights.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of models
            tokens: List of token tensors
            weights: List of weight tensors

        Returns:
            List of encoded tensors
        """
        hidden_states1, hidden_states2, pool2 = self.encode_tokens(tokenize_strategy, models, tokens)

        weights_tensors = [w.to(hidden_states1.device) for w in weights]

        # apply weights
        if weights_tensors[0].shape[1] == 1:  # no max_token_length
            # weights: ((b, 1, 77), (b, 1, 77)), hidden_states: (b, 77, 768), (b, 77, 768)
            hidden_states1 = hidden_states1 * weights_tensors[0].squeeze(1).unsqueeze(2)
            hidden_states2 = hidden_states2 * weights_tensors[1].squeeze(1).unsqueeze(2)
        else:
            # weights: ((b, n, 77), (b, n, 77)), hidden_states: (b, n*75+2, 768), (b, n*75+2, 768)
            for weight, hidden_states in zip(weights_tensors, [hidden_states1, hidden_states2]):
                for i in range(weight.shape[1]):
                    hidden_states[:, i * 75 + 1 : i * 75 + 76] = hidden_states[:, i * 75 + 1 : i * 75 + 76] * weight[:, i, 1:-1].unsqueeze(
                        -1
                    )

        return [hidden_states1, hidden_states2, pool2]


class SdxlTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """
    Text encoder outputs caching strategy for SDXL.
    """

    SDXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_te_outputs.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int | None,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        is_weighted: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial, is_weighted)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        """
        Get path to the cached text encoder outputs npz file.

        Args:
            image_abs_path: Absolute path to the image file

        Returns:
            Path to the npz file
        """
        return os.path.splitext(image_abs_path)[0] + SdxlTextEncoderOutputsCachingStrategy.SDXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        """
        Check if the text encoder outputs are cached in disk.

        Args:
            npz_path: Path to the npz file

        Returns:
            True if cached, False otherwise
        """
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "hidden_state1" not in npz or "hidden_state2" not in npz or "pool2" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> list[np.ndarray]:
        """
        Load text encoder outputs from npz file.

        Args:
            npz_path: Path to the npz file

        Returns:
            List of text encoder outputs
        """
        data = np.load(npz_path)
        hidden_state1 = data["hidden_state1"]
        hidden_state2 = data["hidden_state2"]
        pool2 = data["pool2"]
        return [hidden_state1, hidden_state2, pool2]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: list[Any], text_encoding_strategy: TextEncodingStrategy, batch: list[ImageInfo]
    ) -> None:
        """
        Cache batch outputs.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of TextModel
            text_encoding_strategy: TextEncodingStrategy
            batch: List of ImageInfo
        """
        infos = batch
        assert isinstance(text_encoding_strategy, SdxlTextEncodingStrategy)
        sdxl_text_encoding_strategy: SdxlTextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        if self.is_weighted:
            tokens_list, weights_list = tokenize_strategy.tokenize_with_weights(captions)
            with torch.no_grad():
                hidden_state1, hidden_state2, pool2 = sdxl_text_encoding_strategy.encode_tokens_with_weights(
                    tokenize_strategy, models, tokens_list, weights_list
                )
        else:
            tokens1, tokens2 = tokenize_strategy.tokenize(captions)
            with torch.no_grad():
                hidden_state1, hidden_state2, pool2 = sdxl_text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, [tokens1, tokens2]
                )

        if hidden_state1.dtype == torch.bfloat16:
            hidden_state1 = hidden_state1.float()
        if hidden_state2.dtype == torch.bfloat16:
            hidden_state2 = hidden_state2.float()
        if pool2.dtype == torch.bfloat16:
            pool2 = pool2.float()

        hidden_state1 = hidden_state1.cpu().numpy()
        hidden_state2 = hidden_state2.cpu().numpy()
        pool2 = pool2.cpu().numpy()

        for i, info in enumerate(infos):
            hidden_state1_i = hidden_state1[i]
            hidden_state2_i = hidden_state2[i]
            pool2_i = pool2[i]

            if self.cache_to_disk:
                assert info.text_encoder_outputs_npz is not None, "text_encoder_outputs_npz must be set when cache_to_disk is True"
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_state1=hidden_state1_i,
                    hidden_state2=hidden_state2_i,
                    pool2=pool2_i,
                )
            else:
                info.text_encoder_outputs = [hidden_state1_i, hidden_state2_i, pool2_i]
