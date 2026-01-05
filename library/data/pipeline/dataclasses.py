"""
Core dataclasses for the new data pipeline.

These replace the monolithic ImageInfo with focused, single-responsibility structures.
"""

from abc import ABC
from dataclasses import dataclass, field

import torch


# =============================================================================
# Cache Loading Types (returned by CachingStrategy.load_cache)
# =============================================================================


class ModelConditioning(ABC):  # noqa: B024 - Marker class, no abstract methods
    """
    Base class for model-specific conditioning data.

    Each model type (SD1.5, SDXL, Flux, SD3) has different conditioning requirements.
    This abstract base enables type-safe composition without coupling the dataloader
    to any specific model.

    The training loop, which IS model-specific, casts this to the concrete type.
    """

    pass


@dataclass
class CacheData:
    """
    Model-agnostic cache data returned by CachingStrategy.load_cache().

    This is the universal container for loaded cache data. The `conditioning`
    field holds model-specific data via composition, keeping the dataloader
    agnostic to model details.

    For VAE caching: use `latents`, `latents_flipped`, `alpha_mask`.
    For TE caching: use `aux` dict with encoder-specific outputs.
    """

    latents: torch.Tensor | None = None
    """VAE-encoded latents [C, H, W] (None for TE caching)."""

    latents_flipped: torch.Tensor | None = None
    """Horizontally flipped latents for augmentation (optional)."""

    alpha_mask: torch.Tensor | None = None
    """Alpha channel mask for inpainting (optional)."""

    conditioning: ModelConditioning | None = None
    """Model-specific conditioning data (SDXL crops, etc.)."""

    aux: dict[str, torch.Tensor] = field(default_factory=dict)
    """Additional tensors (e.g., TE outputs: hidden_state1, hidden_state2, pool2)."""


# =============================================================================
# Manifest Entry Types
# =============================================================================


@dataclass
class CacheEntry:
    """
    Represents a single cached image with all its metadata.

    This is the atomic unit in the manifest - one entry per image.
    Replaces the grab-bag ImageInfo with a cleaner, focused structure.
    """

    # Identification
    id: str
    """Unique identifier (used as cache filename base)."""

    image_path: str
    """Absolute path to the source image file."""

    # Image dimensions
    original_size: tuple[int, int]
    """Original image size (width, height) before any processing."""

    bucket_reso: tuple[int, int]
    """Target bucket resolution (width, height) for training."""

    resized_size: tuple[int, int]  # Is this not target_size?
    """Actual size after resize/crop to fit bucket."""

    # Caption data
    caption: str
    """Primary caption text."""

    tags: list[str] = field(default_factory=list)
    """Parsed comma-separated tags from caption."""

    # Training config
    num_repeats: int = 1
    """How many times to repeat this image per epoch."""

    is_reg: bool = False
    """Whether this is a regularization image."""

    split: str = "train"
    """Dataset split: 'train' or 'val'."""

    # Cache paths (populated during caching phase)
    latent_cache_path: str | None = None
    """Path to cached VAE latent .safetensors file."""

    te_cache_path: str | None = None
    """Path to cached text encoder output .safetensors file (when caching TE outputs)."""

    # Augmentation flags (determine if cache is valid)
    has_flipped: bool = False
    """Whether flipped latent is also cached."""

    has_alpha_mask: bool = False
    """Whether alpha mask is included in cache."""


@dataclass
class Bucket:
    """
    Represents a resolution bucket with its images.

    Used for memory-aware batch organization.
    """

    resolution: tuple[int, int]
    """Bucket resolution (width, height)."""

    image_ids: list[str] = field(default_factory=list)
    """IDs of images assigned to this bucket."""

    recommended_batch_size: int = 1
    """Pre-computed safe batch size for this resolution based on VRAM budget."""

    @property
    def count(self) -> int:
        """Number of images in this bucket."""
        return len(self.image_ids)

    def memory_per_image(
        self,
        latent_channels: int = 4,
        latent_scale_factor: int = 8,
        latent_dtype: str = "fp16",
    ) -> int:
        """
        Estimated memory per latent in this bucket (bytes).

        Args:
            latent_channels: Number of latent channels (4 for SD/SDXL, 16 for Flux 1, 32 for Flux 2).
            latent_scale_factor: Spatial downscale factor.
            latent_dtype: Data type for latents ("fp16", "bf16", or "fp32").

        Returns:
            Memory in bytes.
        """
        dtype_bytes = {"fp16": 2, "bf16": 2, "fp32": 4}
        bytes_per_element = dtype_bytes.get(latent_dtype, 2)

        w, h = self.resolution
        latent_w, latent_h = w // latent_scale_factor, h // latent_scale_factor
        return latent_w * latent_h * latent_channels * bytes_per_element


@dataclass
class BatchInfo:
    """
    Complete information for a single training batch.

    Contains all data needed to load and process the batch without
    additional lookups during the training loop.
    """

    image_ids: list[str]
    """Base image IDs (without repeat suffix). For loading latents."""

    bucket_reso: tuple[int, int]
    """Resolution of all images in this batch."""

    # Repeat tracking for per-repeat token/TE caching
    repeat_indices: list[int] = field(default_factory=list)
    """Repeat index for each sample (0 for single-repeat images)."""

    # Caption data (populated in Phase 3 after processing)
    processed_captions: list[str] = field(default_factory=list)
    """Captions after dropout, tag shuffle, wildcard resolution."""

    # Tokenized inputs (optional - populated if not using TE cache)
    # Dict keyed by encoder name, e.g. {"clip_l": [...], "clip_g": [...], "t5": [...]}
    # SD: {"clip"}
    # SDXL: {"clip_l", "clip_g"}
    # SD3/Flux: {"clip_l", "clip_g", "t5"}
    input_ids: dict[str, list[list[int]]] = field(default_factory=dict)
    """Tokenized captions per text encoder. Key = encoder name, value = batch of token sequences."""

    @property
    def batch_size(self) -> int:
        """Number of images in this batch."""
        return len(self.image_ids)

    def get_sample_key(self, idx: int) -> str:
        """Get unique sample key for the given batch index.

        For token/TE caching, this distinguishes between repeats of the same image.
        Format: "img_id" for repeat 0, or "img_id#repeat_idx" for repeat > 0.
        """
        img_id = self.image_ids[idx]
        if not self.repeat_indices:
            return img_id
        repeat_idx = self.repeat_indices[idx]
        return img_id if repeat_idx == 0 else f"{img_id}#{repeat_idx}"


@dataclass
class EpochManifest:
    """
    Pre-computed batch order for one epoch.

    Generated in Phase 3 (epoch preparation), consumed in Phase 4 (training).
    Contains the exact sequence of batches to iterate through.
    """

    epoch: int
    """Epoch number this manifest is for."""

    seed: int
    """Random seed used to generate this ordering."""

    batches: list[BatchInfo] = field(default_factory=list)
    """Ordered list of batches with full metadata."""

    @property
    def num_batches(self) -> int:
        """Total number of batches in this epoch."""
        return len(self.batches)

    @property
    def num_images(self) -> int:
        """Total number of images across all batches."""
        return sum(batch.batch_size for batch in self.batches)


@dataclass
class DatasetManifest:
    """
    Complete dataset manifest - the output of Phase 1 (dataset preparation).

    This is the single source of truth for a dataset, persisted as JSON.
    """

    version: str = "2.0"
    """Manifest format version."""

    created_at: str = ""
    """ISO timestamp when manifest was created."""

    # Configuration used to generate this manifest
    base_resolution: tuple[int, int] = (1024, 1024)
    bucket_reso_steps: int = 64
    min_bucket_reso: int = 256
    max_bucket_reso: int = 2048

    # VAE configuration (affects latent dimensions and storage)
    latent_channels: int = 4
    """Number of latent channels (4 for SD/SDXL, 16 for Flux 1, 32 for Flux 2)."""

    latent_scale_factor: int = 8
    """Spatial downscale factor (8 for most VAEs)."""

    latent_dtype: str = "fp16"
    """Data type for cached latents: 'fp16', 'bf16', or 'fp32'.
    Use 'fp32' for 'no half VAE' mode which can improve training quality at the cost of storage."""

    # Note: Flux 2 also uses patch_size [2, 2] which further affects latent dims.
    # This may need to be extended in the future for full Flux 2 support.

    # Data
    entries: dict[str, CacheEntry] = field(default_factory=dict)
    """Map of image_id -> CacheEntry."""

    buckets: dict[str, Bucket] = field(default_factory=dict)
    """Map of 'WxH' -> Bucket."""

    def get_entry(self, image_id: str) -> CacheEntry | None:
        """Get entry by ID."""
        return self.entries.get(image_id)

    def get_bucket(self, resolution: tuple[int, int]) -> Bucket | None:
        """Get bucket by resolution."""
        key = f"{resolution[0]}x{resolution[1]}"
        return self.buckets.get(key)
