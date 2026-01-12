"""
SD 1.5/2.0 caching strategies for the new data pipeline.

These strategies implement the CachingStrategy interface from library/data/pipeline/caching_engine.py
and are designed to work with CacheEntry dataclasses, not the legacy ImageInfo.
"""

import logging
import os
from pathlib import Path
from typing import Any
import random
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from library.constants import HIGH_VRAM
from library.data._deprecated.data_structures import ImageInfo

from library.data.caching_engine import CachingStrategy
from library.data.structures import CacheData, CacheEntry
from library.strategies.base.caching import LatentsCachingStrategy
from library.utils.common_utils import setup_logging
from library.utils.device_utils import clean_memory_on_device

setup_logging()
logger = logging.getLogger(__name__)

# SD VAE scale factor (used in latent space)
SD_VAE_LATENT_SCALE = 0.18215


class SdLatentsPipelineStrategy(CachingStrategy):
    """
    Latent caching strategy for SD 1.5 and SD 2.0.

    Encodes images to VAE latents and saves as .safetensors files.
    Each cache file contains:
    - latents: [C, H/8, W/8] tensor
    - latents_flipped: (optional) horizontally flipped latents for augmentation
    - Metadata: original_size, crop_ltrb, bucket_reso

    Args:
        cache_suffix: File suffix for cache files (default: "_sd_latents.safetensors")
        flip_aug: Whether to also cache horizontally flipped latents
        dtype: Data type for saved latents ("fp16", "bf16", "fp32")
    """

    def __init__(
        self,
        cache_suffix: str = "_sd_latents.safetensors",
        flip_aug: bool = False,
        dtype: str = "fp16",
    ) -> None:
        self.cache_suffix = cache_suffix
        self.flip_aug = flip_aug
        self.dtype = dtype
        self._torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

    # Note: get_entry_cache_path() is inherited from CachingStrategy base class
    # and returns entry.latent_cache_path by default - no override needed

    def encode_batch(
        self,
        images: torch.Tensor,
        model: Any,
        entries: list[CacheEntry],
    ) -> list[dict[str, Any]]:
        """
        Encode a batch of images to VAE latents.

        Args:
            images: Batch of image tensors [B, C, H, W] in [-1, 1] range.
            model: VAE model with encode() method.
            entries: Corresponding CacheEntry objects for metadata.

        Returns:
            List of dicts with 'latents' tensor and metadata for each image.
        """
        vae = model
        device = vae.device
        vae_dtype = vae.dtype

        # Move images to VAE device and dtype
        images = images.to(device=device, dtype=vae_dtype)

        # Encode to latents
        with torch.no_grad():
            latent_dist = vae.encode(images).latent_dist
            latents = latent_dist.sample() * SD_VAE_LATENT_SCALE

        # Encode flipped if requested
        flipped_latents = None
        if self.flip_aug:
            images_flipped = torch.flip(images, dims=[3])  # Flip width dimension
            with torch.no_grad():
                latent_dist_flipped = vae.encode(images_flipped).latent_dist
                flipped_latents = latent_dist_flipped.sample() * SD_VAE_LATENT_SCALE

        # Convert to target dtype and CPU
        latents = latents.to(dtype=self._torch_dtype).cpu()
        if flipped_latents is not None:
            flipped_latents = flipped_latents.to(dtype=self._torch_dtype).cpu()

        # Build output for each entry
        results = []
        for i, entry in enumerate(entries):
            data: dict[str, Any] = {
                "latents": latents[i],
                "metadata": {
                    "original_size": f"{entry.original_size[0]},{entry.original_size[1]}",
                    "bucket_reso": f"{entry.bucket_reso[0]},{entry.bucket_reso[1]}",
                    "resized_size": f"{entry.resized_size[0]},{entry.resized_size[1]}",
                    # crop_ltrb: left, top, right, bottom (for SDXL conditioning)
                    "crop_ltrb": "0,0,0,0",  # Default no crop, can be computed if needed
                },
            }
            if flipped_latents is not None:
                data["latents_flipped"] = flipped_latents[i]
            results.append(data)

        return results

    def save_cache(self, data: dict[str, Any], path: Path) -> None:
        """
        Save encoded latents to a .safetensors file.

        Args:
            data: Dict with 'latents' tensor and optional 'latents_flipped', plus 'metadata'.
            path: Output file path.
        """
        tensors = {"latents": data["latents"]}
        if "latents_flipped" in data:
            tensors["latents_flipped"] = data["latents_flipped"]

        metadata = data.get("metadata", {})
        save_file(tensors, str(path), metadata=metadata)

    def load_cache(self, path: Path) -> CacheData:
        """
        Load cached latents from a .safetensors file.

        Args:
            path: Cache file path.

        Returns:
            CacheData with latents (and optionally latents_flipped, alpha_mask).
        """
        from library.data.structures import CacheData
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            latents = f.get_tensor("latents")
            latents_flipped = f.get_tensor("latents_flipped") if "latents_flipped" in f.keys() else None  # noqa: SIM118
            alpha_mask = f.get_tensor("alpha_mask") if "alpha_mask" in f.keys() else None  # noqa: SIM118

        return CacheData(
            latents=latents,
            latents_flipped=latents_flipped,
            alpha_mask=alpha_mask,
        )

    def is_cache_valid(
        self,
        path: Path,
        entry: CacheEntry,
        flip_aug: bool = False,
        alpha_mask: bool = False,
    ) -> bool:
        """
        Check if cache file is valid for the given entry and config.

        Validates:
        - 'latents' key exists
        - Latent shape matches bucket resolution (H/8, W/8)
        - 'latents_flipped' present if flip_aug is True
        - 'alpha_mask' present if alpha_mask is True
        - Stored bucket_reso matches entry

        Args:
            path: Cache file path.
            entry: The CacheEntry to validate against.
            flip_aug: Whether flipped latents are required.
            alpha_mask: Whether alpha mask is required.

        Returns:
            True if cache is valid, False if it needs re-caching.
        """
        from safetensors import safe_open

        try:
            with safe_open(str(path), framework="pt") as f:
                keys = set(f.keys())

                # Check required key
                if "latents" not in keys:
                    logger.debug(f"Cache {path}: missing 'latents' key")
                    return False

                # Check latent shape
                latents = f.get_tensor("latents")
                expected_h = entry.bucket_reso[1] // 8
                expected_w = entry.bucket_reso[0] // 8
                if latents.shape != (4, expected_h, expected_w):
                    logger.debug(f"Cache {path}: shape mismatch. Expected (4, {expected_h}, {expected_w}), got {tuple(latents.shape)}")
                    return False

                # Check flip_aug if required
                if flip_aug and "latents_flipped" not in keys:
                    logger.debug(f"Cache {path}: flip_aug required but 'latents_flipped' missing")
                    return False

                # Check alpha_mask if required
                if alpha_mask and "alpha_mask" not in keys:
                    logger.debug(f"Cache {path}: alpha_mask required but missing")
                    return False

                # Check metadata matches entry (optional but useful)
                metadata = f.metadata()
                if metadata:
                    stored_bucket = metadata.get("bucket_reso", "")
                    expected_bucket = f"{entry.bucket_reso[0]},{entry.bucket_reso[1]}"
                    if stored_bucket and stored_bucket != expected_bucket:
                        logger.debug(f"Cache {path}: bucket_reso mismatch. Stored '{stored_bucket}', expected '{expected_bucket}'")
                        return False

            return True

        except Exception as e:
            logger.debug(f"Cache validation error for {path}: {e}")
            return False

    def preprocess_image(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
        resized_size: tuple[int, int] | None = None,
        random_crop: bool = False,
        random_crop_padding_percent: float = 0.05,
        resize_interpolation: str | None = None,
    ) -> torch.Tensor:
        """
        Preprocess an image for VAE encoding.

        Resizes maintaining aspect ratio to resized_size, then crops to target_size.
        This prevents distortion when image aspect ratio doesn't exactly match bucket.

        Args:
            image: PIL Image.
            target_size: Final bucket resolution (width, height) after cropping.
            resized_size: Intermediate size before crop (width, height). If None,
                uses target_size directly (legacy behavior, may cause distortion).
            random_crop: If True, use random crop offset. If False, center crop.
            random_crop_padding_percent: Extra padding when random crop enabled (0.05 = 5%).
            resize_interpolation: Interpolation method. Supported values:
                - None: Auto-select (HAMMING for downscale, LANCZOS for upscale)
                - 'lanczos', 'hamming', 'area', 'bilinear', 'bicubic', 'nearest'

        Returns:
            Tensor [C, H, W] ready for batching.
        """

        # Convert to RGB if needed (do this first to simplify later operations)
        if image.mode != "RGB":
            image = image.convert("RGB")

        target_w, target_h = target_size

        # Determine resize target
        if resized_size is None:
            resize_w, resize_h = target_w, target_h
        else:
            resize_w, resize_h = resized_size
            # Apply random crop padding if enabled
            if random_crop:
                resize_w = int(resize_w * (1.0 + random_crop_padding_percent))
                resize_h = int(resize_h * (1.0 + random_crop_padding_percent))

        # Resize if needed
        orig_w, orig_h = image.size
        if orig_w != resize_w or orig_h != resize_h:
            # Determine interpolation method
            if resize_interpolation is None:
                # Auto-select: HAMMING for downscale, LANCZOS for upscale
                if orig_w >= resize_w and orig_h >= resize_h:
                    pil_interp = Image.Resampling.HAMMING
                else:
                    pil_interp = Image.Resampling.LANCZOS
                image = image.resize((resize_w, resize_h), pil_interp)
            elif resize_interpolation == "area":
                # Use cv2 INTER_AREA for true area-based downscaling
                import cv2

                arr = np.array(image)
                arr = cv2.resize(arr, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
                image = Image.fromarray(arr)
            else:
                # Map string to PIL resampling
                pil_map = {
                    "lanczos": Image.Resampling.LANCZOS,
                    "hamming": Image.Resampling.HAMMING,
                    "bilinear": Image.Resampling.BILINEAR,
                    "bicubic": Image.Resampling.BICUBIC,
                    "nearest": Image.Resampling.NEAREST,
                }
                pil_interp = pil_map.get(resize_interpolation.lower(), Image.Resampling.LANCZOS)
                image = image.resize((resize_w, resize_h), pil_interp)

        # Crop to target size if needed
        current_w, current_h = image.size

        if current_w > target_w:
            trim = current_w - target_w
            left = trim // 2 if not random_crop else random.randint(0, trim)
            image = image.crop((left, 0, left + target_w, current_h))

        if current_h > target_h:
            trim = current_h - target_h
            top = trim // 2 if not random_crop else random.randint(0, trim)
            image = image.crop((0, top, image.size[0], top + target_h))

        # Convert to tensor and normalize to [-1, 1]
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        return tensor


# TODO LEGACY SCRIPT, PENDING UPDATE OF NON SDXL_PEFT TRAINING SCRIPTS AND STRATEGY
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
