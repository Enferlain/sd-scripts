from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from library.pipelines.flow import (
    DiscreteFlowModelSampling,
    get_discrete_flow_sigmas,
    starts_at_max_denoise,
)
from library.strategies.base.contracts import SampleGenerationStrategy
from library.strategies.sd3.encoding import concat_sd3_encodings, encode_sd3_tokens
from library.training.sample_generation import SamplingRequest, sample_images_check, sample_images_common
from library.utils.device_utils import clean_memory_on_device


@dataclass
class Sd3FlowSamplingBackend:
    """SD3-specific RF sampling backend used by the shared sampling orchestrator."""

    _is_sampling_backend = True
    strategy: Any
    denoiser: Any
    vae: Any
    text_encoders: list[Any]
    device: torch.device

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.strategy.tokenize_to_payload(prompt).to(self.device)
        conditioning = encode_sd3_tokens(
            self.text_encoders,
            tokens,
            apply_lg_attn_mask=bool(getattr(self.strategy, "apply_lg_attn_mask", False)),
            apply_t5_attn_mask=bool(getattr(self.strategy, "apply_t5_attn_mask", False)),
            enable_dropout=False,
        )
        return concat_sd3_encodings(conditioning)

    def _do_sample(
        self,
        *,
        request: SamplingRequest,
        accelerator: Any,
    ) -> torch.Tensor:
        latent = torch.zeros(1, 16, request.height // 8, request.width // 8, device=self.device, dtype=self.vae.dtype)

        generator = torch.manual_seed(request.seed) if request.seed is not None else None
        noise = (
            torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator, device="cpu")
            .to(latent.dtype)
            .to(self.device)
        )

        model_sampling = DiscreteFlowModelSampling(shift=request.flow_shift if request.flow_shift is not None else 3.0)
        sigmas = get_discrete_flow_sigmas(model_sampling, request.sample_steps).to(self.device)
        noise_scaled = model_sampling.noise_scaling(sigmas[0], noise, latent, starts_at_max_denoise(model_sampling, sigmas))

        cond = self._encode_prompt(request.prompt)
        neg_cond = self._encode_prompt(request.negative_prompt or "")
        c_crossattn = torch.cat([cond[0], neg_cond[0]]).to(device=self.device, dtype=self.vae.dtype)
        y = torch.cat([cond[1], neg_cond[1]]).to(device=self.device, dtype=self.vae.dtype)
        x = noise_scaled.to(device=self.device, dtype=self.vae.dtype)

        for i in range(len(sigmas) - 1):
            sigma_hat = sigmas[i]
            timestep = model_sampling.timestep(sigma_hat).float()
            timestep = torch.tensor([timestep, timestep], device=self.device, dtype=torch.float32)

            x_c_nc = torch.cat([x, x], dim=0)
            if hasattr(self.denoiser, "prepare_block_swap_before_forward"):
                self.denoiser.prepare_block_swap_before_forward()
            model_output = self.denoiser(x_c_nc, timestep, context=c_crossattn, y=y).float()
            batched = model_sampling.calculate_denoised(sigma_hat, model_output, x)

            pos_out, neg_out = batched.chunk(2)
            denoised = neg_out + (pos_out - neg_out) * request.guidance_scale

            sigma_hat_dims = sigma_hat[(...,) + (None,) * (x.ndim - sigma_hat.ndim)]
            derivative = (x - denoised) / sigma_hat_dims
            dt = sigmas[i + 1] - sigma_hat
            x = (x + derivative * dt).to(self.vae.dtype)

        if hasattr(self.denoiser, "prepare_block_swap_before_forward"):
            self.denoiser.prepare_block_swap_before_forward()
        return x

    def generate_image(self, accelerator: Any, request: SamplingRequest) -> Image.Image:
        with accelerator.autocast(), torch.no_grad():
            latents = self._do_sample(request=request, accelerator=accelerator)

        clean_memory_on_device(self.device)
        latents = self.vae.process_out(latents.to(device=self.vae.device, dtype=self.vae.dtype))
        image = self.vae.decode(latents).float()
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


class Sd3SampleGenerationStrategy(SampleGenerationStrategy):
    """Sample-generation facet for SD3."""

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
        """Generate sample images for SD3."""
        del tokenizers, device
        if not sample_images_check(cfg.output.sampling, epoch, global_step):
            return

        denoiser = accelerator.unwrap_model(denoiser)
        text_encoders = [accelerator.unwrap_model(text_encoder) for text_encoder in text_encoders]

        org_vae_device = vae.device
        org_vae_dtype = vae.dtype
        org_denoiser_device = denoiser.device
        org_te_devices = [text_encoder.device for text_encoder in text_encoders]

        if cfg.output.sampling.sample_vae_dtype is not None:
            sample_dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            sample_vae_dtype = sample_dtype_map.get(cfg.output.sampling.sample_vae_dtype)
            if sample_vae_dtype is not None and sample_vae_dtype != org_vae_dtype:
                vae.to(dtype=sample_vae_dtype)

        try:
            vae.to(accelerator.device)
            denoiser.to(accelerator.device)
            for text_encoder in text_encoders:
                text_encoder.to(accelerator.device)

            backend = Sd3FlowSamplingBackend(
                strategy=self,
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
            for text_encoder, org_device in zip(text_encoders, org_te_devices):
                text_encoder.to(org_device)
            gc.collect()
            clean_memory_on_device(accelerator.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


__all__ = [
    "Sd3FlowSamplingBackend",
    "Sd3SampleGenerationStrategy",
]
