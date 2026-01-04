"""
Core dataclasses for the new data pipeline.

These replace the monolithic ImageInfo with focused, single-responsibility structures.
"""

from dataclasses import dataclass, field


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

    resized_size: tuple[int, int]
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
    """Path to cached text encoder output .safetensors file (SDXL only)."""

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

    @property
    def count(self) -> int:
        """Number of images in this bucket."""
        return len(self.image_ids)

    @property
    def memory_per_image(self) -> int:
        """Estimated memory per latent in this bucket (bytes)."""
        w, h = self.resolution
        # Latent is 4 channels, 1/8 resolution, fp16
        latent_w, latent_h = w // 8, h // 8
        return latent_w * latent_h * 4 * 2  # 4 channels * 2 bytes (fp16)


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

    batches: list[list[str]] = field(default_factory=list)
    """Ordered list of batches, each batch is a list of image IDs."""

    @property
    def num_batches(self) -> int:
        """Total number of batches in this epoch."""
        return len(self.batches)

    @property
    def num_images(self) -> int:
        """Total number of images across all batches."""
        return sum(len(batch) for batch in self.batches)


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
