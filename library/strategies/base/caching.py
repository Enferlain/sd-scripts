import logging
import os
from typing import Optional, Any, Callable

import numpy as np
import torch

from library.data._deprecated.caching import load_images_and_masks_for_caching
from library.data._deprecated.data_structures import ImageInfo

from library.strategies.base.encoding import TextEncodingStrategy
from library.strategies.base.tokenization import TokenizeStrategy
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


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
