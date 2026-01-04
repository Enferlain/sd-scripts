# base class for platform strategies. this file defines the interface for strategies

import os
import numpy as np
import torch
import logging

from typing import Any, Optional
from collections.abc import Callable
from transformers import CLIPTokenizer

from library.constants import re_attention
from library.utils.common_utils import setup_logging
from library.data._deprecated.caching import load_images_and_masks_for_caching
from library.data._deprecated.data_structures import ImageInfo

setup_logging()
logger = logging.getLogger(__name__)


class TokenizeStrategy:
    _strategy = None  # strategy instance: actual strategy class

    _re_attention = re_attention

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TokenizeStrategy"]:
        return cls._strategy

    def _load_tokenizer(self, model_class: Any, model_id: str, subfolder: str | None = None, tokenizer_cache_dir: str | None = None) -> Any:
        """
        Load tokenizer from cache or download it.

        Args:
            model_class: Tokenizer class (e.g. CLIPTokenizer)
            model_id: Model ID (e.g. "openai/clip-vit-large-patch14")
            subfolder: Subfolder in the model repo (e.g. "tokenizer")
            tokenizer_cache_dir: Directory to cache the tokenizer

        Returns:
            Tokenizer instance
        """
        tokenizer = None
        local_tokenizer_path: str | None = None
        if tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(tokenizer_cache_dir, model_id.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = model_class.from_pretrained(local_tokenizer_path)  # same for v1 and v2

        if tokenizer is None:
            tokenizer = model_class.from_pretrained(model_id, subfolder=subfolder)

        if local_tokenizer_path is not None and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        return tokenizer

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        """
        Tokenize text.

        Args:
            text: Text or list of text to tokenize

        Returns:
            List of token tensors
        """
        raise NotImplementedError

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Tokenize text with weights.

        Args:
            text: Text or list of text to tokenize

        Returns:
            Tuple of lists of token tensors and weight tensors
            ([tokens1, tokens2, ...], [weights1, weights2, ...])
        """
        raise NotImplementedError

    def _get_weighted_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get weighted input ids.

        Args:
            tokenizer: Tokenizer instance
            text: Text to tokenize
            max_length: Max length of tokens (including starting and ending tokens)

        Returns:
            Tuple of token tensor and weight tensor
        """

        def parse_prompt_attention(text):
            r"""
            Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
            Accepted tokens are:
            (abc) - increases attention to abc by a multiplier of 1.1
            (abc:3.12) - increases attention to abc by a multiplier of 3.12
            [abc] - decreases attention to abc by a multiplier of 1.1
            \( - literal character '('
            \[ - literal character '['
            \) - literal character ')'
            \] - literal character ']'
            \\ - literal character '\'
            anything else - just text
            >>> parse_prompt_attention('normal text')
            [['normal text', 1.0]]
            >>> parse_prompt_attention('an (important) word')
            [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
            >>> parse_prompt_attention('(unbalanced')
            [['unbalanced', 1.1]]
            >>> parse_prompt_attention('\(literal\]')
            [['(literal]', 1.0]]
            >>> parse_prompt_attention('(unnecessary)(parens)')
            [['unnecessaryparens', 1.1]]
            >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
            [['a ', 1.0],
            ['house', 1.5730000000000004],
            [' ', 1.1],
            ['on', 1.0],
            [' a ', 1.1],
            ['hill', 0.55],
            [', sun, ', 1.1],
            ['sky', 1.4641000000000006],
            ['.', 1.1]]
            """

            res: list[list[str | float]] = []
            round_brackets = []
            square_brackets = []

            round_bracket_multiplier = 1.1
            square_bracket_multiplier = 1 / 1.1

            def multiply_range(start_position, multiplier):
                for p in range(start_position, len(res)):
                    res[p][1] *= multiplier

            for m in TokenizeStrategy._re_attention.finditer(text):
                text = m.group(0)
                weight = m.group(1)

                if text.startswith("\\"):
                    res.append([text[1:], 1.0])
                elif text == "(":
                    round_brackets.append(len(res))
                elif text == "[":
                    square_brackets.append(len(res))
                elif weight is not None and len(round_brackets) > 0:
                    multiply_range(round_brackets.pop(), float(weight))
                elif text == ")" and len(round_brackets) > 0:
                    multiply_range(round_brackets.pop(), round_bracket_multiplier)
                elif text == "]" and len(square_brackets) > 0:
                    multiply_range(square_brackets.pop(), square_bracket_multiplier)
                else:
                    res.append([text, 1.0])

            for pos in round_brackets:
                multiply_range(pos, round_bracket_multiplier)

            for pos in square_brackets:
                multiply_range(pos, square_bracket_multiplier)

            if len(res) == 0:
                res = [["", 1.0]]

            # merge runs of identical weights
            i = 0
            while i + 1 < len(res):
                if res[i][1] == res[i + 1][1]:
                    res[i][0] = str(res[i][0]) + str(res[i + 1][0])
                    res.pop(i + 1)
                else:
                    i += 1

            return res

        def get_prompts_with_weights(text: str, max_length: int):
            r"""
            Tokenize a list of prompts and return its tokens with weights of each token. max_length does not include starting and ending token.

            No padding, starting or ending token is included.
            """
            truncated = False

            texts_and_weights = parse_prompt_attention(text)
            tokens = []
            weights = []
            for word, weight in texts_and_weights:
                # tokenize and discard the starting and the ending token
                token = tokenizer(word).input_ids[1:-1]
                tokens += token
                # copy the weight by length of token
                weights += [weight] * len(token)
                # stop if the text is too long (longer than truncation limit)
                if len(tokens) > max_length:
                    truncated = True
                    break
            # truncate
            if len(tokens) > max_length:
                truncated = True
                tokens = tokens[:max_length]
                weights = weights[:max_length]
            if truncated:
                logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
            return tokens, weights

        def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad):
            r"""
            Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
            """
            tokens = [bos] + tokens + [eos] + [pad] * (max_length - 2 - len(tokens))
            weights = [1.0] + weights + [1.0] * (max_length - 1 - len(weights))
            return tokens, weights

        if max_length is None:
            max_length = tokenizer.model_max_length

        tokens, weights = get_prompts_with_weights(text, max_length - 2)
        tokens, weights = pad_tokens_and_weights(
            tokens, weights, max_length, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        return torch.tensor(tokens).unsqueeze(0), torch.tensor(weights).unsqueeze(0)

    def _get_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: int | None = None, weighted: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Get input ids for SD1.5/2.0/SDXL.

        Args:
            tokenizer: Tokenizer instance
            text: Text to tokenize
            max_length: Max length of tokens
            weighted: Whether to return weights

        Returns:
            Input ids tensor (or tuple of input ids and weights if weighted=True)
        """
        if max_length is None:
            max_length = tokenizer.model_max_length - 2

        weights: torch.Tensor | None = None
        if weighted:
            input_ids, weights = self._get_weighted_input_ids(tokenizer, text, max_length)
        else:
            input_ids = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids

        if max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # When 77 or more, it becomes "<BOS> .... <EOS> <EOS> <EOS>" totaling 227 etc., so convert it to "<BOS>...<EOS>" triplet
                # 1111 seems to split by comma, but for now simply
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # When 77 or more, it becomes "<BOS> .... <EOS> <PAD> <PAD>..." totaling 227 etc., so convert it to "<BOS>...<EOS> <PAD> <PAD> ..." triplet
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # If the end is <EOS> <PAD> or <PAD> <PAD>, nothing needs to be done
                    # If the end is x <PAD/EOS>, change the end to <EOS> (if x <EOS>, no change result)
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # If the beginning is <BOS> <PAD> ..., change it to <BOS> <EOS> <PAD> ...
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77

            if weighted and weights is not None:
                weights = weights.squeeze(0)
                new_weights = torch.ones(input_ids.shape)
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    b = i // (tokenizer.model_max_length - 2)
                    new_weights[b, 1 : 1 + tokenizer.model_max_length - 2] = weights[i : i + tokenizer.model_max_length - 2]
                weights = new_weights

        if weighted and weights is not None:
            return input_ids, weights
        return input_ids


class TextEncodingStrategy:
    """
    Base class for text encoding strategy.
    """

    _strategy = None  # strategy instance: actual strategy class

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncodingStrategy"]:
        return cls._strategy

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens into embeddings and outputs.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of TextModel
            tokens: List of token tensors for each TextModel

        Returns:
            List of output embeddings for each architecture
        """
        raise NotImplementedError

    def encode_tokens_with_weights(
        self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor], weights: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Encode tokens into embeddings and outputs with weights.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of TextModel
            tokens: List of token tensors for each TextModel
            weights: List of weight tensors for each TextModel

        Returns:
            List of output embeddings for each architecture
        """
        raise NotImplementedError


class TextEncoderOutputsCachingStrategy:
    """
    Base class for text encoder outputs caching strategy.
    """

    _strategy = None  # strategy instance: actual strategy class

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int | None,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        is_weighted: bool = False,
    ) -> None:
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check
        self._is_partial = is_partial
        self._is_weighted = is_weighted

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncoderOutputsCachingStrategy"]:
        return cls._strategy

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def is_partial(self):
        return self._is_partial

    @property
    def is_weighted(self):
        return self._is_weighted

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        """
        Get path to the cached text encoder outputs npz file.

        Args:
            image_abs_path: Absolute path to the image file

        Returns:
            Path to the npz file
        """
        raise NotImplementedError

    def load_outputs_npz(self, npz_path: str) -> list[np.ndarray]:
        """
        Load text encoder outputs from npz file.

        Args:
            npz_path: Path to the npz file

        Returns:
            List of text encoder outputs
        """
        raise NotImplementedError

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        """
        Check if the text encoder outputs are cached in disk.

        Args:
            npz_path: Path to the npz file

        Returns:
            True if cached, False otherwise
        """
        raise NotImplementedError

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: list[Any], text_encoding_strategy: TextEncodingStrategy, batch: list
    ):
        """
        Cache batch outputs.

        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of TextModel
            text_encoding_strategy: TextEncodingStrategy
            batch: Batch of data
        """
        raise NotImplementedError


class LatentsCachingStrategy:
    """
    Base class for latents caching strategy.
    """

    # TODO commonize utillity functions to this class, such as npz handling etc.

    _strategy = None  # strategy instance: actual strategy class

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["LatentsCachingStrategy"]:
        return cls._strategy

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def cache_suffix(self):
        """
        Get the suffix for the cache file.
        """
        raise NotImplementedError

    def get_image_size_from_disk_cache_path(self, absolute_path: str, npz_path: str) -> tuple[int | None, int | None]:
        """
        Get image size from disk cache path.

        Args:
            absolute_path: Absolute path to the image file
            npz_path: Path to the npz file

        Returns:
            Width and height
        """
        w, h = os.path.splitext(npz_path)[0].split("_")[-2].split("x")
        return int(w), int(h)

    def get_latents_npz_path(self, absolute_path: str, image_size: tuple[int, int]) -> str:
        """
        Get path to the cached latents npz file.

        Args:
            absolute_path: Absolute path to the image file
            image_size: Image size (width, height)

        Returns:
            Path to the npz file
        """
        raise NotImplementedError

    def is_disk_cached_latents_expected(self, bucket_reso: tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool) -> bool:
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
        raise NotImplementedError

    def cache_batch_latents(
        self, model: Any, batch: list, flip_aug: bool, alpha_mask: bool, random_crop: bool, random_crop_padding_percent: float = 0.05
    ):
        """
        Cache batch latents.

        Args:
            model: Model instance (VAE)
            batch: Batch of data
            flip_aug: Whether to flip images
            alpha_mask: Whether to apply alpha mask
            random_crop: Whether to random crop images
            random_crop_padding_percent: Padding percent for random crop
        """
        raise NotImplementedError

    def _default_is_disk_cached_latents_expected(
        self,
        latents_stride: int,
        bucket_reso: tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        apply_alpha_mask: bool,
        multi_resolution: bool = False,
    ) -> bool:
        """
        Args:
            latents_stride: stride of latents
            bucket_reso: resolution of the bucket
            npz_path: path to the npz file
            flip_aug: whether to flip images
            apply_alpha_mask: whether to apply alpha mask
            multi_resolution: whether to use multi-resolution latents

        Returns:
            bool
        """
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        expected_latents_size = (bucket_reso[1] // latents_stride, bucket_reso[0] // latents_stride)  # bucket_reso is (W, H)

        # e.g. "_32x64", HxW
        key_reso_suffix = f"_{expected_latents_size[0]}x{expected_latents_size[1]}" if multi_resolution else ""

        try:
            npz = np.load(npz_path)
            if "latents" + key_reso_suffix not in npz:
                return False
            if flip_aug and "latents_flipped" + key_reso_suffix not in npz:
                return False
            if apply_alpha_mask and "alpha_mask" + key_reso_suffix not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def _default_cache_batch_latents(
        self,
        encode_by_vae: Callable,
        vae_device: torch.device,
        vae_dtype: torch.dtype,
        image_infos: list[ImageInfo],
        flip_aug: bool,
        apply_alpha_mask: bool,
        random_crop: bool,
        multi_resolution: bool = False,
        random_crop_padding_percent: float = 0.05,
    ):
        """
        Default implementation for cache_batch_latents. Image loading, VAE, flipping, alpha mask handling are common.

        Args:
            encode_by_vae: function to encode images by VAE
            vae_device: device to use for VAE
            vae_dtype: dtype to use for VAE
            image_infos: list of ImageInfo
            flip_aug: whether to flip images
            apply_alpha_mask: whether to apply alpha mask
            random_crop: whether to random crop images
            multi_resolution: whether to use multi-resolution latents

        Returns:
            None
        """
        img_tensor, alpha_masks, original_sizes, crop_ltrbs = load_images_and_masks_for_caching(
            image_infos, apply_alpha_mask, random_crop, random_crop_padding_percent=random_crop_padding_percent
        )
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latents_tensors = encode_by_vae(img_tensor).to("cpu")
        if flip_aug:
            img_tensor = torch.flip(img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = encode_by_vae(img_tensor).to("cpu")
        else:
            flipped_latents = [None] * len(latents_tensors)

        # for info, latents, flipped_latent, alpha_mask in zip(image_infos, latents_tensors, flipped_latents, alpha_masks):
        for i in range(len(image_infos)):
            info = image_infos[i]
            latents = latents_tensors[i]
            flipped_latent = flipped_latents[i]
            alpha_mask = alpha_masks[i]
            original_size = original_sizes[i]
            crop_ltrb = crop_ltrbs[i]

            latents_size = latents.shape[1:3]  # H, W
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}" if multi_resolution else ""  # e.g. "_32x64", HxW

            if self.cache_to_disk:
                self.save_latents_to_disk(
                    info.latents_npz,
                    latents,
                    original_size,
                    crop_ltrb,
                    flipped_latent,  # Can be None when flip_aug=False
                    alpha_mask,  # Can be ndarray when loaded from image
                    key_reso_suffix,
                )
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrb
                info.latents = latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                info.alpha_mask = alpha_mask

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: tuple[int, int]
    ) -> tuple[np.ndarray | None, list[int] | None, list[int] | None, np.ndarray | None, np.ndarray | None]:
        """
        for SD/SDXL

        Args:
            npz_path (str): Path to the npz file.
            bucket_reso (Tuple[int, int]): The resolution of the bucket.

        Returns:
            Tuple[
                Optional[np.ndarray],
                Optional[List[int]],
                Optional[List[int]],
                Optional[np.ndarray],
                Optional[np.ndarray]
            ]: Latent np tensors, original size, crop (left top, right bottom), flipped latents, alpha mask
        """
        return self._default_load_latents_from_disk(None, npz_path, bucket_reso)

    def _default_load_latents_from_disk(
        self, latents_stride: int | None, npz_path: str, bucket_reso: tuple[int, int]
    ) -> tuple[np.ndarray | None, list[int] | None, list[int] | None, np.ndarray | None, np.ndarray | None]:
        """
        Args:
            latents_stride (Optional[int]): Stride for latents. If None, load all latents.
            npz_path (str): Path to the npz file.
            bucket_reso (Tuple[int, int]): The resolution of the bucket.

        Returns:
            Tuple[
                Optional[np.ndarray],
                Optional[List[int]],
                Optional[List[int]],
                Optional[np.ndarray],
                Optional[np.ndarray]
            ]: Latent np tensors, original size, crop (left top, right bottom), flipped latents, alpha mask
        """
        if latents_stride is None:
            key_reso_suffix = ""
        else:
            latents_size = (bucket_reso[1] // latents_stride, bucket_reso[0] // latents_stride)  # bucket_reso is (W, H)
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}"  # e.g. "_32x64", HxW

        npz = np.load(npz_path)
        if "latents" + key_reso_suffix not in npz:
            raise ValueError(f"latents{key_reso_suffix} not found in {npz_path}")

        latents = npz["latents" + key_reso_suffix]
        original_size = npz["original_size" + key_reso_suffix].tolist()
        crop_ltrb = npz["crop_ltrb" + key_reso_suffix].tolist()
        flipped_latents = npz.get("latents_flipped" + key_reso_suffix)
        alpha_mask = npz.get("alpha_mask" + key_reso_suffix)
        return latents, original_size, crop_ltrb, flipped_latents, alpha_mask

    def save_latents_to_disk(
        self,
        npz_path,
        latents_tensor,
        original_size,
        crop_ltrb,
        flipped_latents_tensor=None,
        alpha_mask=None,
        key_reso_suffix="",
    ):
        """
        Args:
            npz_path (str): Path to the npz file.
            latents_tensor (torch.Tensor): Latent tensor
            original_size (List[int]): Original size of the image
            crop_ltrb (List[int]): Crop left top right bottom
            flipped_latents_tensor (Optional[torch.Tensor]): Flipped latent tensor
            alpha_mask (Optional[torch.Tensor]): Alpha mask
            key_reso_suffix (str): Key resolution suffix

        Returns:
            None
        """
        kwargs = {}

        if os.path.exists(npz_path):
            # load existing npz and update it
            npz = np.load(npz_path)
            for key in npz.files:
                kwargs[key] = npz[key]

        # TODO float() is needed if vae is in bfloat16. Remove it if vae is float16.
        kwargs["latents" + key_reso_suffix] = latents_tensor.float().cpu().numpy()
        kwargs["original_size" + key_reso_suffix] = np.array(original_size)
        kwargs["crop_ltrb" + key_reso_suffix] = np.array(crop_ltrb)
        if flipped_latents_tensor is not None:
            kwargs["latents_flipped" + key_reso_suffix] = (
                flipped_latents_tensor.float().cpu().numpy()
            )  # flipped_latents_tensor checked for None above
        if alpha_mask is not None:
            kwargs["alpha_mask" + key_reso_suffix] = (
                alpha_mask.float().cpu().numpy()
            )  # alpha_mask is Tensor at this point (checked for None above)
        np.savez(npz_path, **kwargs)
