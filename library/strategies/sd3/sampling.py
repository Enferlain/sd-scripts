from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from accelerate.state import PartialState

from library.strategies.base.contracts import SampleGenerationStrategy
from library.strategies.sd3.encoding import concat_sd3_encodings, encode_sd3_tokens
from library.training.sample_generation import get_sampling_prompt_dicts, sample_images_check
from library.utils.device_utils import clean_memory_on_device


logger = logging.getLogger(__name__)


class ModelSamplingDiscreteFlow:
    """Sampler helper for SD3 discrete-flow sigma/timestep calculations."""

    def __init__(self, shift: float = 1.0) -> None:
        self.shift = shift
        timesteps = 1000
        self.sigmas = self.sigma(torch.arange(1, timesteps + 1, 1))

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor | float) -> torch.Tensor | float:
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma: torch.Tensor, model_output: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(
        self,
        sigma: torch.Tensor,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
        max_denoise: bool = False,
    ) -> torch.Tensor:
        del max_denoise
        return sigma * noise + (1.0 - sigma) * latent_image


def get_all_sigmas(sampling: ModelSamplingDiscreteFlow, steps: int) -> torch.Tensor:
    """Build the SD3 sigma schedule for the requested number of inference steps."""
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    timesteps = torch.linspace(start, end, steps)
    sigmas = [sampling.sigma(timestep) for timestep in timesteps]
    sigmas.append(torch.tensor(0.0))
    return torch.stack(sigmas).float()


def max_denoise(model_sampling: ModelSamplingDiscreteFlow, sigmas: torch.Tensor) -> bool:
    """Return whether the denoising schedule starts at or above the model's max sigma."""
    max_sigma = float(model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return torch.isclose(torch.tensor(max_sigma), torch.tensor(sigma), rtol=1e-5).item() or sigma > max_sigma


class Sd3SampleGenerationStrategy(SampleGenerationStrategy):
    """Sample-generation facet for SD3."""

    def _encode_prompt(
        self,
        prompt: str,
        text_encoders: list[Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a prompt into SD3 context and pooled conditioning tensors."""
        tokens = self.tokenize(prompt)
        tokens = [token.to(device) for token in tokens]
        outputs = encode_sd3_tokens(
            text_encoders,
            tokens,
            apply_lg_attn_mask=bool(getattr(self, "apply_lg_attn_mask", False)),
            apply_t5_attn_mask=bool(getattr(self, "apply_t5_attn_mask", False)),
            enable_dropout=False,
        )
        return concat_sd3_encodings(outputs[0], outputs[1], outputs[2])

    def _do_sample(
        self,
        *,
        height: int,
        width: int,
        seed: int | None,
        cond: tuple[torch.Tensor, torch.Tensor],
        neg_cond: tuple[torch.Tensor, torch.Tensor],
        denoiser: Any,
        steps: int,
        guidance_scale: float,
        dtype: torch.dtype,
        device: torch.device,
        flow_shift: float,
    ) -> torch.Tensor:
        """Run the SD3 Euler flow-matching sampling loop."""
        latent = torch.zeros(1, 16, height // 8, width // 8, device=device, dtype=dtype)

        generator = torch.manual_seed(seed) if seed is not None else None
        noise = (
            torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator, device="cpu")
            .to(latent.dtype)
            .to(device)
        )

        model_sampling = ModelSamplingDiscreteFlow(shift=flow_shift)
        sigmas = get_all_sigmas(model_sampling, steps).to(device)
        noise_scaled = model_sampling.noise_scaling(sigmas[0], noise, latent, max_denoise(model_sampling, sigmas))

        c_crossattn = torch.cat([cond[0], neg_cond[0]]).to(device=device, dtype=dtype)
        y = torch.cat([cond[1], neg_cond[1]]).to(device=device, dtype=dtype)
        x = noise_scaled.to(device=device, dtype=dtype)

        for i in range(len(sigmas) - 1):
            sigma_hat = sigmas[i]
            timestep = model_sampling.timestep(sigma_hat).float()
            timestep = torch.tensor([timestep, timestep], device=device, dtype=torch.float32)

            x_c_nc = torch.cat([x, x], dim=0)
            if hasattr(denoiser, "prepare_block_swap_before_forward"):
                denoiser.prepare_block_swap_before_forward()
            model_output = denoiser(x_c_nc, timestep, context=c_crossattn, y=y).float()
            batched = model_sampling.calculate_denoised(sigma_hat, model_output, x)

            pos_out, neg_out = batched.chunk(2)
            denoised = neg_out + (pos_out - neg_out) * guidance_scale

            sigma_hat_dims = sigma_hat[(...,) + (None,) * (x.ndim - sigma_hat.ndim)]
            derivative = (x - denoised) / sigma_hat_dims
            dt = sigmas[i + 1] - sigma_hat
            x = (x + derivative * dt).to(dtype)

        if hasattr(denoiser, "prepare_block_swap_before_forward"):
            denoiser.prepare_block_swap_before_forward()
        return x

    def _sample_single_prompt(
        self,
        accelerator: Any,
        cfg: Any,
        denoiser: Any,
        vae: Any,
        save_dir: str,
        prompt_dict: dict[str, Any],
        epoch: int,
        global_step: int,
    ) -> None:
        """Generate and save one SD3 sample image."""
        sampling_cfg = cfg.output.sampling
        saving_cfg = cfg.output.saving

        prompt = prompt_dict.get("prompt", sampling_cfg.sample_prompt or "")
        negative_prompt = prompt_dict.get("negative_prompt", sampling_cfg.sample_negative_prompt) or ""
        sample_steps = prompt_dict.get("sample_steps", sampling_cfg.sample_steps if sampling_cfg.sample_steps is not None else 30)
        width = prompt_dict.get("width", sampling_cfg.sample_width if sampling_cfg.sample_width is not None else 1024)
        height = prompt_dict.get("height", sampling_cfg.sample_height if sampling_cfg.sample_height is not None else 1024)
        guidance_scale = prompt_dict.get(
            "scale",
            prompt_dict.get("guidance_scale", sampling_cfg.sample_cfg_scale if sampling_cfg.sample_cfg_scale is not None else 7.5),
        )
        seed = prompt_dict.get("seed", sampling_cfg.sample_seed)
        flow_shift = float(prompt_dict.get("flow_shift", getattr(cfg.model, "sample_flow_shift", 3.0)))

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        else:
            torch.seed()
            if torch.cuda.is_available():
                torch.cuda.seed()

        width = max(64, width - width % 8)
        height = max(64, height - height % 8)
        logger.info("prompt: %s", prompt)
        logger.info("negative_prompt: %s", negative_prompt)
        logger.info("height: %s", height)
        logger.info("width: %s", width)
        logger.info("sample_steps: %s", sample_steps)
        logger.info("scale: %s", guidance_scale)
        logger.info("flow_shift: %s", flow_shift)
        if seed is not None:
            logger.info("seed: %s", seed)

        with accelerator.autocast(), torch.no_grad():
            cond = self._encode_prompt(prompt, self._sampling_text_encoders, accelerator.device)
            neg_cond = self._encode_prompt(negative_prompt, self._sampling_text_encoders, accelerator.device)
            latents = self._do_sample(
                height=height,
                width=width,
                seed=seed,
                cond=cond,
                neg_cond=neg_cond,
                denoiser=denoiser,
                steps=sample_steps,
                guidance_scale=guidance_scale,
                dtype=vae.dtype,
                device=accelerator.device,
                flow_shift=flow_shift,
            )

        clean_memory_on_device(accelerator.device)
        latents = vae.process_out(latents.to(device=vae.device, dtype=vae.dtype))
        image = vae.decode(latents).float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
        pil_image = Image.fromarray(decoded_np)

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_index = prompt_dict["enum"]
        img_filename = (
            f"{'' if saving_cfg.output_name is None else saving_cfg.output_name + '_'}"
            f"{num_suffix}_{prompt_index:02d}_{ts_str}{seed_suffix}.png"
        )
        pil_image.save(os.path.join(save_dir, img_filename))

        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            wandb_tracker = accelerator.get_tracker("wandb")
            import wandb

            wandb_tracker.log({f"sample_{prompt_index}": wandb.Image(pil_image, caption=prompt)}, commit=False)

        del cond, neg_cond, latents, image, pil_image
        gc.collect()
        clean_memory_on_device(accelerator.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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

        prompts = get_sampling_prompt_dicts(cfg.output.sampling)
        if prompts is None:
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

        distributed_state = PartialState()
        save_dir = os.path.join(cfg.output.saving.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)

        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        except Exception:
            cuda_rng_state = None

        try:
            vae.to(accelerator.device)
            denoiser.to(accelerator.device)
            for text_encoder in text_encoders:
                text_encoder.to(accelerator.device)
            self._sampling_text_encoders = text_encoders

            if distributed_state.num_processes <= 1:
                with torch.no_grad():
                    for prompt_dict in prompts:
                        self._sample_single_prompt(accelerator, cfg, denoiser, vae, save_dir, prompt_dict, epoch, global_step)
            else:
                per_process_prompts = [prompts[i :: distributed_state.num_processes] for i in range(distributed_state.num_processes)]
                with torch.no_grad(), distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                    for prompt_dict in prompt_dict_lists[0]:
                        self._sample_single_prompt(accelerator, cfg, denoiser, vae, save_dir, prompt_dict, epoch, global_step)
        finally:
            if hasattr(self, "_sampling_text_encoders"):
                del self._sampling_text_encoders
            vae.to(device=org_vae_device, dtype=org_vae_dtype)
            denoiser.to(org_denoiser_device)
            for text_encoder, org_device in zip(text_encoders, org_te_devices):
                text_encoder.to(org_device)
            torch.set_rng_state(rng_state)
            if torch.cuda.is_available() and cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
            gc.collect()
            clean_memory_on_device(accelerator.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


__all__ = [
    "ModelSamplingDiscreteFlow",
    "Sd3SampleGenerationStrategy",
    "get_all_sigmas",
    "max_denoise",
]
