"""
SDXL caching strategies for the new data pipeline.

These strategies implement the CacheHandler interface from library/data/pipeline/caching_engine.py
and are designed to work with CacheEntry dataclasses, not the legacy ImageInfo.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from library.data._deprecated.data_structures import ImageInfo
from library.data.caching_engine import CacheHandler
from library.data.structures import CacheData, CacheEntry, ModelConditioning
from library.strategies.base.training import TextEncodingStrategy, TokenizationStrategy
from library.strategies.sdxl.encoding import SdxlTextEncodingStrategy
from library.strategies.base.caching import TextEncoderOutputsCachingStrategy

from library.utils.hash_utils import stable_string_hash


logger = logging.getLogger(__name__)

# SDXL VAE scale factor (different from SD)
SDXL_VAE_LATENT_SCALE = 0.13025


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
        self, tokenize_strategy: TokenizationStrategy, models: list[Any], text_encoding_strategy: TextEncodingStrategy, batch: list[ImageInfo]
    ) -> None:
        """
        Cache batch outputs.

        Args:
            tokenize_strategy: TokenizationStrategy
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


@dataclass
class SdxlConditioning(ModelConditioning):
    """
    SDXL-specific micro-conditioning metadata.

    SDXL uses original image size, crop coordinates, and target size as conditioning
    inputs to improve generation quality. These are stored in cache metadata and
    extracted during loading. See SDXL paper section 2.2.
    """

    original_size_hw: tuple[int, int]
    """Original image size (height, width) before any processing."""

    crop_top_left: tuple[int, int]
    """Crop offset (top, left) in bucket pixel space."""

    target_size_hw: tuple[int, int]
    """Target/bucket resolution (height, width) the image was resized to."""


def get_crop_ltrb(
    bucket_reso: tuple[int, int],
    original_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Calculate crop coordinates for SDXL micro-conditioning.

    When an image is resized to fit a bucket, it may need to be cropped.
    This computes where the center crop happens in the original image's
    coordinate space.

    Ported from BucketManager.get_crop_ltrb() for compatibility with
    Stability AI's training approach.

    Args:
        bucket_reso: Target bucket resolution (width, height).
        original_size: Original image size (width, height).

    Returns:
        Tuple of (crop_left, crop_top, crop_right, crop_bottom) in original
        image pixel coordinates.
    """
    bucket_w, bucket_h = bucket_reso
    orig_w, orig_h = original_size

    bucket_ar = bucket_w / bucket_h
    image_ar = orig_w / orig_h

    if bucket_ar > image_ar:
        # Bucket is wider than image → match height, crop top/bottom
        resized_width = bucket_h * image_ar
        resized_height = bucket_h
    else:
        # Bucket is taller than image → match width, crop left/right
        resized_width = bucket_w
        resized_height = bucket_w / image_ar

    # Center crop offsets
    crop_left = int((bucket_w - resized_width) // 2)
    crop_top = int((bucket_h - resized_height) // 2)
    crop_right = int(crop_left + resized_width)
    crop_bottom = int(crop_top + resized_height)

    return crop_left, crop_top, crop_right, crop_bottom


class SdxlLatentsPipelineStrategy(CacheHandler):
    """
    Latent caching strategy for SDXL.

    Encodes images to VAE latents and saves as .safetensors files.
    Each cache file contains:
    - latents: [C, H/8, W/8] tensor
    - latents_flipped: (optional) horizontally flipped latents for augmentation
    - Metadata: original_size, crop_ltrb, bucket_reso (used in SDXL conditioning)

    Args:
        cache_suffix: File suffix for cache files (default: "_sdxl_latents.safetensors")
        flip_aug: Whether to also cache horizontally flipped latents
        dtype: Data type for saved latents ("fp16", "bf16", "fp32")
    """

    def __init__(
        self,
        cache_suffix: str = "_sdxl_latents.safetensors",
        flip_aug: bool = False,
        dtype: str = "fp16",
    ) -> None:
        self.cache_suffix = cache_suffix
        self.flip_aug = flip_aug
        self.dtype = dtype
        self._torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

    # Note: get_entry_cache_path() is inherited from CacheHandler base class
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
            latents = latent_dist.sample() * SDXL_VAE_LATENT_SCALE

        # Encode flipped if requested
        flipped_latents = None
        if self.flip_aug:
            images_flipped = torch.flip(images, dims=[3])  # Flip width dimension
            with torch.no_grad():
                latent_dist_flipped = vae.encode(images_flipped).latent_dist
                flipped_latents = latent_dist_flipped.sample() * SDXL_VAE_LATENT_SCALE

        # Convert to target dtype and CPU
        latents = latents.to(dtype=self._torch_dtype).cpu()
        if flipped_latents is not None:
            flipped_latents = flipped_latents.to(dtype=self._torch_dtype).cpu()

        # Build output for each entry
        results = []
        for i, entry in enumerate(entries):
            # Compute crop coordinates for SDXL micro-conditioning
            crop_l, crop_t, crop_r, crop_b = get_crop_ltrb(entry.bucket_reso, entry.original_size)

            data: dict[str, Any] = {
                "latents": latents[i],
                "metadata": {
                    "original_size": f"{entry.original_size[0]},{entry.original_size[1]}",
                    "bucket_reso": f"{entry.bucket_reso[0]},{entry.bucket_reso[1]}",
                    "resized_size": f"{entry.resized_size[0]},{entry.resized_size[1]}",
                    # Crop coordinates in bucket pixel space (for SDXL conditioning)
                    "crop_ltrb": f"{crop_l},{crop_t},{crop_r},{crop_b}",
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
            CacheData with latents and SdxlConditioning (parsed from metadata).
        """
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata() or {}
            latents = f.get_tensor("latents")
            latents_flipped = f.get_tensor("latents_flipped") if "latents_flipped" in f.keys() else None  # noqa: SIM118
            alpha_mask = f.get_tensor("alpha_mask") if "alpha_mask" in f.keys() else None  # noqa: SIM118

        # Parse SDXL conditioning from metadata
        original_size_hw = (0, 0)
        crop_top_left = (0, 0)
        target_size_hw = (0, 0)

        if "original_size" in metadata:
            w, h = map(int, metadata["original_size"].split(","))
            original_size_hw = (h, w)  # Convert to HW format

        if "crop_ltrb" in metadata:
            l, t, _r, _b = map(int, metadata["crop_ltrb"].split(","))
            crop_top_left = (t, l)  # (top, left) format

        if "bucket_reso" in metadata:
            w, h = map(int, metadata["bucket_reso"].split(","))
            target_size_hw = (h, w)  # Convert WH to HW format

        conditioning = SdxlConditioning(
            original_size_hw=original_size_hw,
            crop_top_left=crop_top_left,
            target_size_hw=target_size_hw,
        )

        return CacheData(
            latents=latents,
            latents_flipped=latents_flipped,
            alpha_mask=alpha_mask,
            conditioning=conditioning,
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
                - 'lanczos': High-quality upscale/downscale
                - 'hamming': PIL's sharpest downscale filter
                - 'area': cv2 INTER_AREA (moiré-free downscaling, uses OpenCV)
                - 'bilinear': Fast but lower quality
                - 'bicubic': Balanced quality/speed

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


class SdxlTextEncoderPipelineStrategy(CacheHandler):
    """
    Text encoder output caching strategy for SDXL.

    Encodes captions using both CLIP text encoders and saves outputs as .safetensors.
    Each cache file contains:
    - hidden_state1: Output from CLIP-L (text_encoder 1)
    - hidden_state2: Output from CLIP-G (text_encoder 2)
    - pool2: Pooled output from CLIP-G (used in time embeddings)
    - Metadata: caption hash, token lengths

    Args:
        cache_suffix: File suffix for cache files (default: "_sdxl_te.safetensors")
        max_token_length: Maximum token length (None = 75, or 150, 225 for longer prompts)
        dtype: Data type for saved embeddings ("fp16", "bf16", "fp32")
    """

    def __init__(
        self,
        cache_suffix: str = "_sdxl_te.safetensors",
        max_token_length: int | None = None,
        dtype: str = "fp16",
    ) -> None:
        self.cache_suffix = cache_suffix
        self.max_token_length = max_token_length
        self.dtype = dtype
        self._torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

    def get_entry_cache_path(self, entry: CacheEntry) -> str | None:
        """Return te_cache_path instead of latent_cache_path."""
        return entry.te_cache_path

    def encode_batch(
        self,
        images: torch.Tensor,
        model: Any,
        entries: list[CacheEntry],
    ) -> list[dict[str, Any]]:
        """
        Encode captions using SDXL's dual text encoders.

        Note: For text encoder caching, 'images' tensor is not used.
        We pull captions from the entries instead.

        Args:
            images: Unused (text encoding doesn't need images).
            model: Tuple of (text_encoder1, text_encoder2, tokenizer1, tokenizer2).
            entries: CacheEntry objects containing captions.

        Returns:
            List of dicts with 'hidden_state1', 'hidden_state2', 'pool2' for each entry.
        """
        text_encoder1, text_encoder2, tokenizer1, tokenizer2 = model
        device = text_encoder1.device

        captions = [entry.caption for entry in entries]

        # Tokenize with both tokenizers
        tokens1 = tokenizer1(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer1.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)

        tokens2 = tokenizer2(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer2.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)

        # Encode with both text encoders
        with torch.no_grad():
            # Text encoder 1 (CLIP-L)
            enc1_out = text_encoder1(tokens1, output_hidden_states=True)
            hidden_state1 = enc1_out.hidden_states[-2]  # Penultimate layer

            # Text encoder 2 (CLIP-G with projection)
            enc2_out = text_encoder2(tokens2, output_hidden_states=True)
            hidden_state2 = enc2_out.hidden_states[-2]  # Penultimate layer
            pool2 = enc2_out.text_embeds  # Pooled output

        # Convert to target dtype and CPU
        hidden_state1 = hidden_state1.to(dtype=self._torch_dtype).cpu()
        hidden_state2 = hidden_state2.to(dtype=self._torch_dtype).cpu()
        pool2 = pool2.to(dtype=self._torch_dtype).cpu()

        # Build output for each entry
        results = []
        for i, entry in enumerate(entries):
            data: dict[str, Any] = {
                "hidden_state1": hidden_state1[i],
                "hidden_state2": hidden_state2[i],
                "pool2": pool2[i],
                "metadata": {
                    "caption_hash": str(stable_string_hash(entry.caption)),  # For validation
                },
            }
            results.append(data)

        return results

    def save_cache(self, data: dict[str, Any], path: Path) -> None:
        """
        Save text encoder outputs to a .safetensors file.

        Args:
            data: Dict with 'hidden_state1', 'hidden_state2', 'pool2', plus 'metadata'.
            path: Output file path.
        """
        tensors = {
            "hidden_state1": data["hidden_state1"],
            "hidden_state2": data["hidden_state2"],
            "pool2": data["pool2"],
        }

        metadata = data.get("metadata", {})
        save_file(tensors, str(path), metadata=metadata)

    def load_cache(self, path: Path) -> CacheData:
        """
        Load cached text encoder outputs from a .safetensors file.

        Args:
            path: Cache file path.

        Returns:
            CacheData with TE outputs in the `aux` dict.
        """
        from safetensors import safe_open

        extra = {}
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open requires .keys()
                extra[key] = f.get_tensor(key)

        return CacheData(aux=extra)

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
        - Required keys exist (hidden_state1, hidden_state2, pool2)
        - Optionally checks caption_hash matches

        Args:
            path: Cache file path.
            entry: The CacheEntry to validate against.
            flip_aug: Unused for text encoder (no flip augmentation).
            alpha_mask: Unused for text encoder.

        Returns:
            True if cache is valid, False if it needs re-caching.
        """
        from safetensors import safe_open

        try:
            with safe_open(str(path), framework="pt") as f:
                keys = set(f.keys())

                # Check required keys
                required_keys = {"hidden_state1", "hidden_state2", "pool2"}
                missing = required_keys - keys
                if missing:
                    logger.debug(f"Cache {path}: missing keys {missing}")
                    return False

                # Optionally check caption hash (detect caption changes)
                metadata = f.metadata()
                if metadata and "caption_hash" in metadata:
                    stored_hash = metadata["caption_hash"]
                    expected_hash = str(stable_string_hash(entry.caption))
                    if stored_hash != expected_hash:
                        logger.debug(f"Cache {path}: caption changed. Stored hash '{stored_hash}', expected '{expected_hash}'")
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
        No-op for text encoder caching (we don't use images).

        Returns a dummy tensor since the interface requires it.
        """
        # Return empty tensor - text encoder doesn't use images
        return torch.empty(0)
