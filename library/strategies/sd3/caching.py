import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from library.data.caching_engine import CacheBackend
from library.data.structures import CacheData, CacheEntry
from library.strategies.base.contracts import CachingStrategy
from library.strategies.sd3.encoding import encode_sd3_tokens
from library.strategies.sd3.tokenization import DEFAULT_SD3_T5_MAX_LENGTH, tokenize_sd3_text
from library.utils.hash_utils import stable_string_hash


logger = logging.getLogger(__name__)


class Sd3CachingStrategy(CachingStrategy):
    """Training-facing SD3 caching facet implementation."""

    def create_latent_caching_strategy(self, cfg: Any) -> "Sd3LatentsPipelineStrategy":
        """Create the new-pipeline latent cache handler for SD3."""
        latent_dtype = "fp32" if cfg.performance.precision.no_half_vae else "fp16"
        return Sd3LatentsPipelineStrategy(
            flip_aug=cfg.data.preprocessing.flip_aug,
            dtype=latent_dtype,
        )

    def create_te_caching_strategy(self, cfg: Any) -> "Sd3TextEncoderPipelineStrategy":
        """Create the new-pipeline text-encoder cache handler for SD3."""
        t5_max_length = getattr(cfg.model, "t5xxl_max_token_length", None)
        if t5_max_length is None:
            t5_max_length = cfg.training.max_token_length or DEFAULT_SD3_T5_MAX_LENGTH
        return Sd3TextEncoderPipelineStrategy(
            t5_max_length=t5_max_length,
        )

    def get_token_cache_encoder_names(self) -> list[str]:
        """Return the SD3 token-cache encoder names."""
        return ["clip_l", "clip_g", "t5"]

    def build_te_cache_model_bundle(self, cfg: Any, accelerator: Any, text_encoders: list[Any], tokenizers: list[Any]) -> Any:
        """Return the model bundle used by SD3 TE caching."""
        del cfg, accelerator
        return (*text_encoders, *tokenizers)


class Sd3LatentsPipelineStrategy(CacheBackend):
    """
    Latent caching strategy for SD3.

    Encodes images to SD3 VAE latents and saves as .safetensors files.
    Each cache file contains:
    - latents: [C, H/8, W/8] tensor with 16 channels
    - latents_flipped: (optional) horizontally flipped latents for augmentation
    - Metadata: original_size, crop_ltrb, bucket_reso, resized_size
    """

    def __init__(
        self,
        cache_suffix: str = "_sd3_latents.safetensors",
        flip_aug: bool = False,
        dtype: str = "fp16",
    ) -> None:
        self.cache_suffix = cache_suffix
        self.flip_aug = flip_aug
        self.dtype = dtype
        self._torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

    def encode_batch(
        self,
        images: torch.Tensor,
        model: Any,
        entries: list[CacheEntry],
    ) -> list[dict[str, Any]]:
        """Encode a batch of images to SD3 VAE latents."""
        vae = model
        device = vae.device
        vae_dtype = vae.dtype

        images = images.to(device=device, dtype=vae_dtype)

        with torch.no_grad():
            latents = vae.encode(images)

        flipped_latents = None
        if self.flip_aug:
            images_flipped = torch.flip(images, dims=[3])
            with torch.no_grad():
                flipped_latents = vae.encode(images_flipped)

        latents = latents.to(dtype=self._torch_dtype).cpu()
        if flipped_latents is not None:
            flipped_latents = flipped_latents.to(dtype=self._torch_dtype).cpu()

        results = []
        for i, entry in enumerate(entries):
            data: dict[str, Any] = {
                "latents": latents[i],
                "metadata": {
                    "original_size": f"{entry.original_size[0]},{entry.original_size[1]}",
                    "bucket_reso": f"{entry.bucket_reso[0]},{entry.bucket_reso[1]}",
                    "resized_size": f"{entry.resized_size[0]},{entry.resized_size[1]}",
                    "crop_ltrb": "0,0,0,0",
                },
            }
            if flipped_latents is not None:
                data["latents_flipped"] = flipped_latents[i]
            results.append(data)

        return results

    def save_cache(self, data: dict[str, Any], path: Path) -> None:
        """Save encoded latents to a .safetensors file."""
        tensors = {"latents": data["latents"]}
        if "latents_flipped" in data:
            tensors["latents_flipped"] = data["latents_flipped"]

        save_file(tensors, str(path), metadata=data.get("metadata", {}))

    def load_cache(self, path: Path) -> CacheData:
        """Load cached SD3 latents from a .safetensors file."""
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
        """Check if SD3 latent cache matches the expected shape and metadata."""
        from safetensors import safe_open

        try:
            with safe_open(str(path), framework="pt") as f:
                keys = set(f.keys())
                if "latents" not in keys:
                    logger.debug(f"Cache {path}: missing 'latents' key")
                    return False

                latents = f.get_tensor("latents")
                expected_h = entry.bucket_reso[1] // 8
                expected_w = entry.bucket_reso[0] // 8
                if latents.shape != (16, expected_h, expected_w):
                    logger.debug(f"Cache {path}: shape mismatch. Expected (16, {expected_h}, {expected_w}), got {tuple(latents.shape)}")
                    return False

                if flip_aug and "latents_flipped" not in keys:
                    logger.debug(f"Cache {path}: flip_aug required but 'latents_flipped' missing")
                    return False

                if alpha_mask and "alpha_mask" not in keys:
                    logger.debug(f"Cache {path}: alpha_mask required but missing")
                    return False

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
        """Preprocess an image for SD3 VAE encoding."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        target_w, target_h = target_size
        if resized_size is None:
            resize_w, resize_h = target_w, target_h
        else:
            resize_w, resize_h = resized_size
            if random_crop:
                resize_w = int(resize_w * (1.0 + random_crop_padding_percent))
                resize_h = int(resize_h * (1.0 + random_crop_padding_percent))

        orig_w, orig_h = image.size
        if orig_w != resize_w or orig_h != resize_h:
            if resize_interpolation is None:
                if orig_w >= resize_w and orig_h >= resize_h:
                    pil_interp = Image.Resampling.HAMMING
                else:
                    pil_interp = Image.Resampling.LANCZOS
                image = image.resize((resize_w, resize_h), pil_interp)
            elif resize_interpolation == "area":
                import cv2

                arr = np.array(image)
                arr = cv2.resize(arr, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
                image = Image.fromarray(arr)
            else:
                pil_map = {
                    "lanczos": Image.Resampling.LANCZOS,
                    "hamming": Image.Resampling.HAMMING,
                    "bilinear": Image.Resampling.BILINEAR,
                    "bicubic": Image.Resampling.BICUBIC,
                    "nearest": Image.Resampling.NEAREST,
                }
                pil_interp = pil_map.get(resize_interpolation.lower(), Image.Resampling.LANCZOS)
                image = image.resize((resize_w, resize_h), pil_interp)

        current_w, current_h = image.size
        if current_w > target_w:
            trim = current_w - target_w
            left = trim // 2 if not random_crop else random.randint(0, trim)
            image = image.crop((left, 0, left + target_w, current_h))

        if current_h > target_h:
            trim = current_h - target_h
            top = trim // 2 if not random_crop else random.randint(0, trim)
            image = image.crop((0, top, image.size[0], top + target_h))

        arr = np.array(image, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)


class Sd3TextEncoderPipelineStrategy(CacheBackend):
    """
    Text encoder output caching strategy for SD3.

    Each cache file contains:
    - lg_out: concatenated CLIP-L/CLIP-G hidden states
    - t5_out: T5 hidden states
    - lg_pooled: concatenated pooled CLIP-L/CLIP-G output
    - clip_l_attn_mask, clip_g_attn_mask, t5_attn_mask
    """

    def __init__(
        self,
        cache_suffix: str = "_sd3_te.safetensors",
        t5_max_length: int = DEFAULT_SD3_T5_MAX_LENGTH,
        dtype: str = "fp16",
        apply_lg_attn_mask: bool = False,
        apply_t5_attn_mask: bool = False,
    ) -> None:
        self.cache_suffix = cache_suffix
        self.t5_max_length = t5_max_length
        self.dtype = dtype
        self.apply_lg_attn_mask = apply_lg_attn_mask
        self.apply_t5_attn_mask = apply_t5_attn_mask
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
        """Encode captions using SD3's three text encoders."""
        del images
        text_encoder_l, text_encoder_g, text_encoder_t5, tokenizer_l, tokenizer_g, tokenizer_t5 = model
        captions = [entry.caption for entry in entries]

        tokens = tokenize_sd3_text(tokenizer_l, tokenizer_g, tokenizer_t5, self.t5_max_length, captions)

        with torch.no_grad():
            conditioning = encode_sd3_tokens(
                [text_encoder_l, text_encoder_g, text_encoder_t5],
                tokens,
                apply_lg_attn_mask=self.apply_lg_attn_mask,
                apply_t5_attn_mask=self.apply_t5_attn_mask,
                enable_dropout=False,
            )

        conditioning = conditioning.move_to("cpu", weight_dtype=self._torch_dtype)

        results = []
        for i, entry in enumerate(entries):
            sample_conditioning = conditioning.select(i)
            data: dict[str, Any] = sample_conditioning.to_te_output_dict()
            data["metadata"] = {
                "caption_hash": str(stable_string_hash(entry.caption)),
                "t5_max_length": str(self.t5_max_length),
                "apply_lg_attn_mask": str(self.apply_lg_attn_mask),
                "apply_t5_attn_mask": str(self.apply_t5_attn_mask),
            }
            results.append(data)

        return results

    def save_cache(self, data: dict[str, Any], path: Path) -> None:
        """Save SD3 text encoder outputs to a .safetensors file."""
        tensors = {
            "lg_out": data["lg_out"],
            "lg_pooled": data["lg_pooled"],
            "clip_l_attn_mask": data["clip_l_attn_mask"],
            "clip_g_attn_mask": data["clip_g_attn_mask"],
            "t5_attn_mask": data["t5_attn_mask"],
        }
        if "t5_out" in data:
            tensors["t5_out"] = data["t5_out"]
        save_file(tensors, str(path), metadata=data.get("metadata", {}))

    def load_cache(self, path: Path) -> CacheData:
        """Load cached SD3 text encoder outputs from disk."""
        from safetensors import safe_open

        extra = {}
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                extra[key] = f.get_tensor(key)

        return CacheData(aux=extra)

    def is_cache_valid(
        self,
        path: Path,
        entry: CacheEntry,
        flip_aug: bool = False,
        alpha_mask: bool = False,
    ) -> bool:
        """Check that the SD3 TE cache has the expected tensors and metadata."""
        del flip_aug, alpha_mask
        from safetensors import safe_open

        try:
            with safe_open(str(path), framework="pt") as f:
                keys = set(f.keys())
                required_keys = {"lg_out", "lg_pooled", "clip_l_attn_mask", "clip_g_attn_mask", "t5_attn_mask", "t5_out"}
                missing = required_keys - keys
                if missing:
                    logger.debug(f"Cache {path}: missing keys {missing}")
                    return False

                metadata = f.metadata() or {}
                if "caption_hash" in metadata:
                    stored_hash = metadata["caption_hash"]
                    expected_hash = str(stable_string_hash(entry.caption))
                    if stored_hash != expected_hash:
                        logger.debug(f"Cache {path}: caption changed. Stored hash '{stored_hash}', expected '{expected_hash}'")
                        return False

                if metadata.get("t5_max_length") not in (None, str(self.t5_max_length)):
                    logger.debug(f"Cache {path}: t5_max_length mismatch")
                    return False
                if metadata.get("apply_lg_attn_mask") not in (None, str(self.apply_lg_attn_mask)):
                    logger.debug(f"Cache {path}: apply_lg_attn_mask mismatch")
                    return False
                if metadata.get("apply_t5_attn_mask") not in (None, str(self.apply_t5_attn_mask)):
                    logger.debug(f"Cache {path}: apply_t5_attn_mask mismatch")
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
        """Return a dummy tensor because SD3 text encoding does not use images."""
        del image, target_size, resized_size, random_crop, random_crop_padding_percent, resize_interpolation
        return torch.empty(0)
