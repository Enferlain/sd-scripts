from __future__ import annotations

import gc
from typing import Any

import torch
from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from library.strategies.base.contracts import SampleGenerationStrategy
from library.training.sample_generation import (
    SamplingRequest,
    get_my_scheduler,
    sample_images_check,
    sample_images_common,
)
from library.utils.device_utils import clean_memory_on_device


class SdxlFlowSamplingBackend:
    """Thin adapter that routes SDXL RF sampling through the SDXL pipeline layer."""

    _is_sampling_backend = True
    def __init__(self, pipeline: SdxlStableDiffusionLongPromptWeightingPipeline):
        self.pipeline = pipeline

    def to(self, device: torch.device) -> None:
        self.pipeline.to(device)

    def generate_image(self, accelerator: Any, request: SamplingRequest):
        generator = torch.Generator(device=accelerator.device)
        if request.seed is None:
            generator.seed()
        else:
            generator.manual_seed(request.seed)

        with accelerator.autocast(), torch.no_grad():
            latents = self.pipeline.flow_text2img(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.sample_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
                shift=request.flow_shift if request.flow_shift is not None else 2.5,
                output_type="latent",
            )
        gc.collect()
        clean_memory_on_device(accelerator.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        image = self.pipeline.latents_to_image(latents)[0]

        del latents
        gc.collect()
        clean_memory_on_device(accelerator.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return image


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
        del device
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
            if cfg.objective.path != "rectified_flow":
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
                return

            denoiser.to(accelerator.device)
            backend = SdxlFlowSamplingBackend(pipeline)
            sample_images_common(
                accelerator,
                cfg.output.sampling,
                cfg.training,
                cfg.output.saving,
                cfg.objective,
                cfg.loss,
                epoch,
                global_step,
                backend,
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


__all__ = [
    "SdxlFlowSamplingBackend",
    "SdxlSampleGenerationStrategy",
]
