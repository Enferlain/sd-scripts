from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from library.constants import SDXL_VAE_LATENT_SCALE
from library.pipelines.flow import (
    DiscreteFlowModelSampling,
    get_discrete_flow_sigmas,
    starts_at_max_denoise,
)
from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from library.strategies.base.contracts import SampleGenerationStrategy
from library.strategies.sdxl.conditioning import SdxlConditioning
from library.training.sample_generation import (
    SamplingRequest,
    get_my_scheduler,
    sample_images_check,
    sample_images_common,
)
from library.utils.device_utils import clean_memory_on_device


@dataclass
class SdxlFlowSamplingBackend:
    """SDXL-specific RF sampling backend used by the shared sampling orchestrator."""

    _is_sampling_backend = True
    strategy: Any
    cfg: Any
    denoiser: Any
    vae: Any
    text_encoders: list[Any]
    device: torch.device

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, weights = self.strategy.tokenize_with_weights(prompt)
        models = [*self.text_encoders, self.text_encoders[-1]]
        hidden1, hidden2, pool2 = self.strategy.encode_tokens_with_weights(models, tokens, weights)
        return hidden1, hidden2, pool2

    def _build_sampling_batch(self, request: SamplingRequest, latents: torch.Tensor) -> dict[str, Any]:
        conditioning = SdxlConditioning(
            original_size_hw=(request.height, request.width),
            crop_top_left=(0, 0),
            target_size_hw=(request.height, request.width),
        )
        return {"conditionings": [conditioning, conditioning], "latents": latents}

    def _do_sample(self, *, accelerator: Any, request: SamplingRequest) -> torch.Tensor:
        latent = torch.zeros(1, 4, request.height // 8, request.width // 8, device=self.device, dtype=self.vae.dtype)

        generator = torch.manual_seed(request.seed) if request.seed is not None else None
        noise = (
            torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator, device="cpu")
            .to(latent.dtype)
            .to(self.device)
        )

        model_sampling = DiscreteFlowModelSampling(shift=request.flow_shift if request.flow_shift is not None else 2.5)
        sigmas = get_discrete_flow_sigmas(model_sampling, request.sample_steps).to(self.device)
        noise_scaled = model_sampling.noise_scaling(sigmas[0], noise, latent, starts_at_max_denoise(model_sampling, sigmas))

        cond = self._encode_prompt(request.prompt)
        neg_cond = self._encode_prompt(request.negative_prompt or "")
        text_conds = tuple(torch.cat([positive, negative], dim=0).to(device=self.device, dtype=self.vae.dtype) for positive, negative in zip(cond, neg_cond))
        x = noise_scaled.to(device=self.device, dtype=self.vae.dtype)

        for i in range(len(sigmas) - 1):
            sigma_hat = sigmas[i]
            timestep = model_sampling.timestep(sigma_hat).float()
            timesteps = torch.tensor([timestep, timestep], device=self.device, dtype=torch.float32)

            x_c_nc = torch.cat([x, x], dim=0)
            sampling_batch = self._build_sampling_batch(request, x_c_nc)
            model_output = self.strategy.call_denoiser(
                self.cfg,
                accelerator,
                self.denoiser,
                x_c_nc,
                timesteps,
                text_conds,
                sampling_batch,
                self.vae.dtype,
            ).float()

            pos_out, neg_out = model_output.chunk(2)
            denoised = neg_out + (pos_out - neg_out) * request.guidance_scale

            sigma_hat_dims = sigma_hat[(...,) + (None,) * (x.ndim - sigma_hat.ndim)]
            derivative = (x - denoised) / sigma_hat_dims
            dt = sigmas[i + 1] - sigma_hat
            x = (x + derivative * dt).to(self.vae.dtype)

        return x

    def generate_image(self, accelerator: Any, request: SamplingRequest) -> Image.Image:
        with accelerator.autocast(), torch.no_grad():
            latents = self._do_sample(accelerator=accelerator, request=request)

        clean_memory_on_device(self.device)
        with torch.no_grad():
            image = self.vae.decode(latents.to(device=self.vae.device, dtype=self.vae.dtype) / SDXL_VAE_LATENT_SCALE).sample.float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
        pil_image = Image.fromarray(decoded_np)

        del latents, image
        gc.collect()
        clean_memory_on_device(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return pil_image


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
            if cfg.objective.path != "rectified_flow":
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
                return

            denoiser.to(accelerator.device)
            for text_encoder in text_encoders:
                text_encoder.to(accelerator.device)

            backend = SdxlFlowSamplingBackend(
                strategy=self,
                cfg=cfg,
                denoiser=denoiser,
                vae=vae,
                text_encoders=text_encoders,
                device=accelerator.device,
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
