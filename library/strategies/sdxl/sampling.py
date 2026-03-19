from typing import Any

import torch

from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from library.strategies.base.training import SampleGenerationStrategy
from library.training.sample_generation import sample_images_common


class SdxlSampleGenerationStrategy(SampleGenerationStrategy):
    """Sample-generation facet for SDXL training strategies."""

    def sample_images(
        self,
        accelerator: Any,
        cfg: Any,
        epoch: int,
        global_step: int,
        device: torch.device,
        vae: Any,
        tokenizers: list[Any],
        text_encoders: list[Any],
        unet: Any,
    ) -> None:
        """Generate sample images for SDXL."""
        sample_images_common(
            SdxlStableDiffusionLongPromptWeightingPipeline,
            accelerator,
            cfg.output.sampling,
            cfg.training,
            cfg.output.saving,
            cfg.loss,
            epoch,
            global_step,
            device,
            vae,
            tokenizers,
            text_encoders,
            unet,
            strategy=self,
        )
