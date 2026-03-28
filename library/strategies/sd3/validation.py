import ast
import random
from typing import Any

import torch
from tqdm import tqdm

from library.losses.loss import conditional_loss
from library.strategies.base.contracts import ValidationStrategy
from library.strategies.sd3.diffusion import encode_sd3_images_to_latents, shift_scale_sd3_latents
from library.training.diffusion import prepare_latents
from library.training.trainer_utils import restore_rng_state, switch_rng_state


class Sd3ValidationStrategy(ValidationStrategy):
    """Validation facet for SD3 flow-matching training."""

    def process_val_batch(
        self,
        batch: Any,
        text_encoders: list[Any],
        denoiser: Any,
        trainable_model: Any,
        vae: Any,
        noise_scheduler: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        train_text_encoder: bool = True,
        train_denoiser: bool = True,
        timesteps_list: list[int] | None = None,
    ) -> torch.Tensor:
        """Process a batch for SD3 validation loss."""
        if timesteps_list is None:
            timesteps_list = [50, 350, 500, 650, 950]

        with torch.autograd.grad_mode.inference_mode(mode=True):
            latents = prepare_latents(
                batch,
                cfg.data.caching,
                accelerator.device,
                vae,
                vae_dtype,
                1.0,
                log_fn=accelerator.print,
                encode_images_to_latents_fn=encode_sd3_images_to_latents,
                shift_scale_latents_fn=shift_scale_sd3_latents,
            )
            total_loss = torch.zeros(1, device=latents.device)

            text_encoder_conds = self._get_text_conds(
                batch=batch,
                text_encoders=text_encoders,
                accelerator=accelerator,
                cfg=cfg,
                train_text_encoder=train_text_encoder,
                is_train=False,
                weight_dtype=weight_dtype,
            )

            batch_size = latents.shape[0]
            for fixed_timestep_value in timesteps_list:
                fixed_timesteps = torch.full((batch_size,), fixed_timestep_value, dtype=torch.long, device=latents.device)
                noise_pred, target, _, _ = self.get_noise_pred_and_target(
                    cfg,
                    accelerator,
                    noise_scheduler,
                    latents,
                    batch,
                    text_encoder_conds,
                    denoiser,
                    trainable_model,
                    weight_dtype,
                    train_denoiser,
                    fixed_timesteps=fixed_timesteps,
                    is_train=False,
                )

                loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)
                loss = loss.mean([1, 2, 3]).mean()
                total_loss += loss

        return total_loss / len(timesteps_list)

    def calculate_val_loss(
        self,
        global_step: int,
        epoch_step: int,
        train_dataloader: Any,
        val_loss_recorder: Any,
        val_dataloader: Any,
        cyclic_val_dataloader: Any,
        trainable_model: Any,
        text_encoders: list[Any],
        denoiser: Any,
        vae: Any,
        noise_scheduler: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        epoch: int,
        batch: Any | None = None,
        train_text_encoder: bool = True,
    ) -> tuple[float | None, float | None]:
        """Calculate validation loss for SD3."""
        del epoch_step, train_dataloader, epoch, batch

        rng_states = switch_rng_state(int(cfg.validation.validation_seed) if cfg.validation.validation_seed else 23, accelerator)
        timesteps_list = ast.literal_eval(cfg.validation.validation_timesteps)

        accelerator.print("")
        accelerator.print("Validating...")
        total_loss = 0.0
        with torch.no_grad():
            validation_steps = (
                min(int(cfg.validation.max_validation_steps), len(val_dataloader))
                if cfg.validation.max_validation_steps is not None
                else len(val_dataloader)
            )
            val_dataloader_seed = random.randint(global_step, 0x7FFFFFFF)
            val_dataloader_state = random.Random(val_dataloader_seed).getstate()
            for _val_step in tqdm(range(validation_steps), desc="Validation Steps"):
                val_original_state = random.getstate()
                random.setstate(val_dataloader_state)
                batch = next(cyclic_val_dataloader)
                val_dataloader_state = random.getstate()
                random.setstate(val_original_state)
                loss = self.process_val_batch(
                    batch,
                    text_encoders,
                    denoiser,
                    trainable_model,
                    vae,
                    noise_scheduler,
                    vae_dtype,
                    weight_dtype,
                    accelerator,
                    cfg,
                    train_text_encoder=train_text_encoder,
                    timesteps_list=timesteps_list,
                )
                total_loss += loss.detach().item()
            current_val_loss = total_loss / validation_steps
            val_loss_recorder.add(current_val_loss)

        average_val_loss: float = val_loss_recorder.average
        restore_rng_state(rng_states, accelerator)
        return current_val_loss, average_val_loss


__all__ = ["Sd3ValidationStrategy"]
