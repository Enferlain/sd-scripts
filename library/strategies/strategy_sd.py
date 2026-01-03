import os
import torch
import logging

from typing import Any, cast
from transformers import CLIPTokenizer

from library.constants import HIGH_VRAM, V2_STABLE_DIFFUSION_ID, TOKENIZER_ID
from library.strategies.strategy_base import LatentsCachingStrategy, TokenizeStrategy, TextEncodingStrategy
from library.data.data_structures import ImageInfo
from library.utils.common_utils import setup_logging
from library.utils.device_utils import clean_memory_on_device

setup_logging()
logger = logging.getLogger(__name__)


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


class SdTextEncodingStrategy(TextEncodingStrategy):
    """
    Text encoding strategy for SD1.5 and SD2.0.
    """

    def __init__(self, clip_skip: int | None = None) -> None:
        self.clip_skip = clip_skip

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens.

        Args:
            tokenize_strategy: TokenizeStrategy instance
            models: List of models
            tokens: List of token tensors

        Returns:
            List of encoded tensors
        """
        text_encoder = models[0]
        tokens_tensor = tokens[0]
        assert isinstance(tokenize_strategy, SdTokenizeStrategy)
        sd_tokenize_strategy: SdTokenizeStrategy = tokenize_strategy

        # tokens_tensor: b,n,77
        b_size = tokens_tensor.size()[0]
        max_token_length = tokens_tensor.size()[1] * tokens_tensor.size()[2]
        model_max_length = sd_tokenize_strategy.tokenizer.model_max_length
        tokens_tensor = tokens_tensor.reshape((-1, model_max_length))  # batch_size*3, 77

        tokens_tensor = tokens_tensor.to(text_encoder.device)

        if self.clip_skip is None:
            encoder_hidden_states = text_encoder(tokens_tensor)[0]
        else:
            enc_out = text_encoder(tokens_tensor, output_hidden_states=True, return_dict=True)
            encoder_hidden_states = enc_out["hidden_states"][-self.clip_skip]
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

        # bs*3, 77, 768 or 1024
        encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

        if max_token_length != model_max_length:
            v1 = sd_tokenize_strategy.tokenizer.pad_token_id == sd_tokenize_strategy.tokenizer.eos_token_id
            if not v1:
                # v2: Restore the triplet of <BOS>...<EOS> <PAD> ... to <BOS>...<EOS> <PAD> ...
                states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
                for i in range(1, max_token_length, model_max_length):
                    chunk = encoder_hidden_states[:, i : i + model_max_length - 2]  # From after <BOS> to before the last
                    if i > 0:
                        for j in range(len(chunk)):
                            if tokens_tensor[j, 1] == sd_tokenize_strategy.tokenizer.eos_token:
                                # Empty, i.e., <BOS> <EOS> <PAD> ... pattern
                                chunk[j, 0] = chunk[j, 1]  # Copy the value of the next <PAD>
                    states_list.append(chunk)  # From after <BOS> to before <EOS>
                states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # Either <EOS> or <PAD>
                encoder_hidden_states = torch.cat(states_list, dim=1)
            else:
                # v1: Restore the triplet of <BOS>...<EOS> to <BOS>...<EOS>
                states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
                for i in range(1, max_token_length, model_max_length):
                    states_list.append(encoder_hidden_states[:, i : i + model_max_length - 2])  # From after <BOS> to before <EOS>
                states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
                encoder_hidden_states = torch.cat(states_list, dim=1)

        return [encoder_hidden_states]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        encoder_hidden_states = self.encode_tokens(tokenize_strategy, models, tokens)[0]

        weights_tensor = weights[0].to(encoder_hidden_states.device)

        # apply weights
        if weights_tensor.shape[1] == 1:  # no max_token_length
            # weights: ((b, 1, 77), (b, 1, 77)), hidden_states: (b, 77, 768), (b, 77, 768)
            encoder_hidden_states = encoder_hidden_states * weights_tensor.squeeze(1).unsqueeze(2)
        else:
            # weights: ((b, n, 77), (b, n, 77)), hidden_states: (b, n*75+2, 768), (b, n*75+2, 768)
            for i in range(weights_tensor.shape[1]):
                encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] = encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] * weights_tensor[
                    :, i, 1:-1
                ].unsqueeze(-1)

        return [encoder_hidden_states]


class SdSdxlLatentsCachingStrategy(LatentsCachingStrategy):
    """
    Latents caching strategy for SD1.5, SD2.0 and SDXL.
    """

    # SD and SDXL use the same caching format, separated only by cache file suffix (_sd vs _sdxl)
    # and we keep the old npz for the backward compatibility.

    SD_OLD_LATENTS_NPZ_SUFFIX = ".npz"
    SD_LATENTS_NPZ_SUFFIX = "_sd.npz"
    SDXL_LATENTS_NPZ_SUFFIX = "_sdxl.npz"

    def __init__(self, sd: bool, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.sd = sd
        self.suffix = SdSdxlLatentsCachingStrategy.SD_LATENTS_NPZ_SUFFIX if sd else SdSdxlLatentsCachingStrategy.SDXL_LATENTS_NPZ_SUFFIX

    @property
    def cache_suffix(self) -> str:
        return self.suffix

    def get_latents_npz_path(self, absolute_path: str, image_size: tuple[int, int]) -> str:
        """
        Get path to the cached latents npz file.

        Args:
            absolute_path: Absolute path to the image file
            image_size: Image size (width, height)

        Returns:
            Path to the npz file
        """
        # support old .npz
        old_npz_file = os.path.splitext(absolute_path)[0] + SdSdxlLatentsCachingStrategy.SD_OLD_LATENTS_NPZ_SUFFIX
        if os.path.exists(old_npz_file):
            return old_npz_file
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.suffix

    def is_disk_cached_latents_expected(self, bucket_reso: tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        """
        Check if the latents are cached in disk.

        Args:
            bucket_reso: Resolution of the bucket
            npz_path: Path to the npz file
            flip_aug: Whether to flip images
            alpha_mask: Whether to apply alpha mask

        Returns:
            True if cached, False otherwise
        """
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask)

    def cache_batch_latents(
        self,
        model: Any,
        batch: list[ImageInfo],
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
        random_crop_padding_percent: float = 0.05,
    ) -> None:
        """
        Cache batch latents.

        Args:
            model: VAE model
            batch: List of ImageInfo
            flip_aug: Whether to flip images
            alpha_mask: Whether to apply alpha mask
            random_crop: Whether to random crop images
            random_crop_padding_percent: Padding percent for random crop
        """
        vae = model
        image_infos = batch

        def encode_by_vae(img_tensor: torch.Tensor) -> torch.Tensor:
            return vae.encode(img_tensor).latent_dist.sample()

        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae,
            vae_device,
            vae_dtype,
            image_infos,
            flip_aug,
            alpha_mask,
            random_crop,
            random_crop_padding_percent=random_crop_padding_percent,
        )

        if not HIGH_VRAM:
            clean_memory_on_device(vae.device)
