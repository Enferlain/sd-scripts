"""SDXL-specific conditioning payloads used by strategy facets."""

from dataclasses import dataclass

from library.strategies.base.contracts import ModelConditioning


@dataclass
class SdxlConditioning(ModelConditioning):
    """
    SDXL micro-conditioning metadata carried alongside cached latents.

    SDXL uses original image size, crop coordinates, and target size as
    auxiliary conditioning inputs during denoiser calls.
    """

    original_size_hw: tuple[int, int]
    """Original image size (height, width) before any processing."""

    crop_top_left: tuple[int, int]
    """Crop offset (top, left) in bucket pixel space."""

    target_size_hw: tuple[int, int]
    """Target/bucket resolution (height, width) the image was resized to."""
