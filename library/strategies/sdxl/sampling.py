from typing import Any

import gc
import torch

from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from library.strategies.base.contracts import SampleGenerationStrategy
from library.training.sample_generation import (
    get_my_scheduler,
    sample_images_check,
    sample_images_common,
)
from library.utils.device_utils import clean_memory_on_device


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
        denoiser: Any,
    ) -> None:
        """Generate sample images for SDXL."""
        if not sample_images_check(cfg.output.sampling, epoch, global_step):
            return

        org_vae_device = vae.device
        org_vae_dtype = vae.dtype
        denoiser = accelerator.unwrap_model(denoiser)
        org_denoiser_device = denoiser.device
        text_encoders = [accelerator.unwrap_model(te) for te in text_encoders]
        org_te_devices = [te.device for te in text_encoders]

        if cfg.output.sampling.sample_vae_dtype is not None:
            sample_dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            sample_vae_dtype = sample_dtype_map.get(cfg.output.sampling.sample_vae_dtype)
            if sample_vae_dtype is not None and sample_vae_dtype != org_vae_dtype:
                vae.to(dtype=sample_vae_dtype)

        vae.to(accelerator.device)
        try:
            pipeline = SdxlStableDiffusionLongPromptWeightingPipeline(
                text_encoder=text_encoders,
                vae=vae,
                tokenizer=tokenizers,
                unet=denoiser,
                scheduler=get_my_scheduler(
                    sample_sampler=cfg.output.sampling.sample_sampler,
                    prediction_type=cfg.objective.prediction,
                ),
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                clip_skip=cfg.training.clip_skip,
                strategy=self,
            )
            sample_images_common(
                accelerator,
                cfg.output.sampling,
                cfg.training,
                cfg.output.saving,
                cfg.objective,
                cfg.loss,
                epoch,
                global_step,
                pipeline,
            )
        finally:
            vae.to(device=org_vae_device, dtype=org_vae_dtype)
            denoiser.to(org_denoiser_device)
            for te, org_device in zip(text_encoders, org_te_devices):
                te.to(org_device)
            gc.collect()
            clean_memory_on_device(accelerator.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
