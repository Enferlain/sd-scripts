from typing import Any

import torch

from library.pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from library.strategies.base.training import SampleGenerationStrategy
from library.training.sample_generation import sample_images_common


class SdSampleGenerationStrategy(SampleGenerationStrategy):
    """Sample-generation facet for SD 1.5/2.0 training strategies."""

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
        denoiser: Any,
    ) -> None:
        """Generate sample images for SD."""
        sample_images_common(
            StableDiffusionLongPromptWeightingPipeline,
            accelerator,
            cfg.output.sampling,
            cfg.training,
            cfg.output.saving,
            cfg.loss,
            epoch,
            global_step,
            device,
            vae,
            tokenizers[0],
            text_encoders[0],
            denoiser,
            strategy=self,
        )
