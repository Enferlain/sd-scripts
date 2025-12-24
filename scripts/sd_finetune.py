import hydra
import math
import os
import toml
import torch
import logging

from omegaconf import OmegaConf
from tqdm import tqdm
from multiprocessing import Value
from diffusers import DDPMScheduler

from library.performance import deepspeed_utils
from library.strategies import strategy_sd, strategy_base
from library.utils.device_utils import init_ipex, clean_memory_on_device
from library.utils.common_utils import setup_logging
from library.utils.torch_utils import set_torch_cuda_reduced_precision, set_seed_from_config, prepare_dtype
from library.config.config_util import BlueprintGenerator, generate_dataset_group_by_blueprint
from library.data.dataset import load_arbitrary_dataset, collator_class, debug_dataset
from library.training.model_prep import replace_unet_modules, patch_accelerator_for_fp16_training
from library.training.sd_model_prep import load_target_model
from library.training.diffusion import get_noise_noisy_latents_and_timesteps
from library.training.optimizer import get_optimizer, get_scheduler_fix
from library.training.sd_sample_generation import sample_images
from library.training.trainer_utils import prepare_accelerator, append_lr_to_logs
from library.losses.loss import LossRecorder, get_huber_threshold_if_needed, conditional_loss
from library.config.dataclasses.sd_finetune import SDFineTuneConfig
from library.config.validation import prepare_config, validate_config

from library.training.checkpointing import (
    resume_from_local_or_hf_if_specified,
    save_state_on_train_end,
)

from library.training.sd_checkpointing import (
    save_sd_model_on_epoch_end_or_stepwise,
    save_sd_model_on_train_end
)

from library.training.noise_utils import (
    fix_noise_scheduler_betas_for_zero_terminal_snr,
    prepare_scheduler_for_custom_training
)

from library.losses.loss_weighting import (
    apply_snr_weight,
    apply_debiased_estimation,
    scale_v_prediction_loss_like_noise_prediction,
)


init_ipex()

setup_logging()
logger = logging.getLogger(__name__)


def train(cfg: SDFineTuneConfig):

    setup_logging(cfg.logging, reset=True)
    set_torch_cuda_reduced_precision(cfg.training)
    deepspeed_utils.prepare_deepspeed_config(cfg.training)

    cache_latents = cfg.dataset.cache_latents

    set_seed_from_config(cfg.training)

    tokenize_strategy = strategy_sd.SdTokenizeStrategy(cfg.model.v2, cfg.training.max_token_length, cfg.model.tokenizer_cache_dir)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    if cache_latents:
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, cfg.dataset.cache_latents_to_disk, cfg.dataset.vae_batch_size, cfg.dataset.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    if cfg.dataset.dataset_class is None:
        blueprint_generator = BlueprintGenerator()
        blueprint = blueprint_generator.generate(cfg)
        train_dataset_group, val_dataset_group = generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        # load_arbitrary_dataset now accepts DatasetConfig directly
        train_dataset_group = load_arbitrary_dataset(dataset_config)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if cfg.training.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(64)

    if cfg.dataset.debug_dataset:
        debug_dataset(train_dataset_group)
        return
    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify the metadata file and train_data_dir option.")
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used"

    logger.info("prepare accelerator")
    accelerator = prepare_accelerator(cfg.training)

    weight_dtype, save_dtype = prepare_dtype(cfg.performance, cfg.saving)
    vae_dtype = torch.float32 if cfg.performance.no_half_vae else weight_dtype

    text_encoder, vae, unet, load_stable_diffusion_format = load_target_model(cfg.model, cfg.performance, weight_dtype, accelerator)

    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = cfg.model.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = cfg.model.pretrained_model_name_or_path

    if cfg.saving.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = cfg.saving.use_safetensors
    else:
        save_stable_diffusion_format = cfg.saving.save_model_as.lower() == "ckpt" or cfg.saving.save_model_as.lower() == "safetensors"
        use_safetensors = cfg.saving.use_safetensors or ("safetensors" in cfg.saving.save_model_as.lower())

    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    if cfg.performance.diffusers_xformers:
        accelerator.print("Use xformers by Diffusers")
        set_diffusers_xformers_flag(unet, True)
    else:
        accelerator.print("Disable Diffusers' xformers")
        set_diffusers_xformers_flag(unet, False)
        replace_unet_modules(unet, cfg.performance.mem_eff_attn, cfg.performance.xformers, cfg.performance.sdpa)

    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    training_models = []
    if cfg.performance.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    training_models.append(unet)

    if cfg.fine_tune.train_text_encoder:
        accelerator.print("enable text encoder training")
        if cfg.performance.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
        training_models.append(text_encoder)
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        if cfg.performance.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            text_encoder.train()
        else:
            text_encoder.eval()

    text_encoding_strategy = strategy_sd.SdTextEncodingStrategy(cfg.training.clip_skip)
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    for m in training_models:
        m.requires_grad_(True)

    trainable_params = []
    if cfg.fine_tune.learning_rate_te is None or not cfg.fine_tune.train_text_encoder:
        for m in training_models:
            trainable_params.extend(m.parameters())
    else:
        trainable_params = [
            {"params": list(unet.parameters()), "lr": cfg.optimizer.learning_rate},
            {"params": list(text_encoder.parameters()), "lr": cfg.fine_tune.learning_rate_te},
        ]

    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = get_optimizer(cfg.optimizer, trainable_params=trainable_params)

    train_dataset_group.set_current_strategies()

    n_workers = min(cfg.training.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=cfg.training.persistent_data_loader_workers,
    )

    if cfg.training.max_train_epochs is not None:
        cfg.training.max_train_steps = cfg.training.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / cfg.training.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {cfg.training.max_train_epochs} epochs is: {cfg.training.max_train_steps}"
        )

    train_dataset_group.set_max_train_steps(cfg.training.max_train_steps)

    lr_scheduler = get_scheduler_fix(cfg.optimizer, optimizer, accelerator.num_processes)

    if cfg.performance.full_fp16:
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder.to(weight_dtype)

    if cfg.training.deepspeed:
        if cfg.fine_tune.train_text_encoder:
            ds_model = deepspeed_utils.prepare_deepspeed_model(cfg.training, unet=unet, text_encoder=text_encoder)
        else:
            ds_model = deepspeed_utils.prepare_deepspeed_model(cfg.training, unet=unet)
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]
    else:
        if cfg.fine_tune.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    if cfg.performance.full_fp16:
        patch_accelerator_for_fp16_training(accelerator)

    resume_from_local_or_hf_if_specified(accelerator, cfg.saving)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)
    if (cfg.saving.save_n_epoch_ratio is not None) and (cfg.saving.save_n_epoch_ratio > 0):
        cfg.saving.save_every_n_epochs = math.floor(num_train_epochs / cfg.saving.save_n_epoch_ratio) or 1

    total_batch_size = cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps
    accelerator.print("running training")
    accelerator.print(f"  num examples: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
    accelerator.print(f"  num epochs: {num_train_epochs}")
    accelerator.print(f"  batch size per device: {cfg.training.train_batch_size}")
    accelerator.print(
        f"  total train batch size (with parallel & distributed & accumulation): {total_batch_size}"
    )
    accelerator.print(f"  gradient accumulation steps = {cfg.training.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps: {cfg.training.max_train_steps}")

    progress_bar = tqdm(range(cfg.training.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )

    if cfg.regularization.zero_terminal_snr:
        fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)

    if accelerator.is_main_process:
        init_kwargs = {}
        if cfg.logging.wandb_run_name:
            init_kwargs["wandb"] = {"name": cfg.logging.wandb_run_name}
        if cfg.logging.log_tracker_config is not None:
            init_kwargs = toml.load(cfg.logging.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if cfg.logging.log_tracker_name is None else cfg.logging.log_tracker_name,
            config=OmegaConf.to_container(cfg),
            init_kwargs=init_kwargs,
        )

    sample_images(
        accelerator, cfg.sampling, cfg.training, cfg.saving, 0, global_step, accelerator.device, vae, tokenize_strategy.tokenizer, text_encoder, unet
    )
    if len(accelerator.trackers) > 0:
        accelerator.log({}, step=0)

    loss_recorder = LossRecorder()
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(*training_models):
                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample().to(weight_dtype)
                    latents = latents * 0.18215
                b_size = latents.shape[0]

                with torch.set_grad_enabled(cfg.fine_tune.train_text_encoder):
                    if cfg.dataset.weighted_captions:
                        input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                        encoder_hidden_states = text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy, [text_encoder], input_ids_list, weights_list
                        )[0]
                    else:
                        input_ids = batch["input_ids_list"][0].to(accelerator.device)
                        encoder_hidden_states = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [text_encoder], [input_ids]
                        )[0]
                    if cfg.performance.full_fp16:
                        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

                noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(cfg.regularization, cfg.timestep, cfg.training, noise_scheduler, latents)

                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if cfg.loss.v_parameterization:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                huber_c = get_huber_threshold_if_needed(cfg.loss, timesteps, noise_scheduler)
                if cfg.loss.min_snr_gamma or cfg.loss.scale_v_pred_loss_like_noise_pred or cfg.loss.debiased_estimation_loss:
                    loss = conditional_loss(noise_pred.float(), target.float(), cfg.loss.loss_type, "none", huber_c, scale=float(cfg.loss.loss_scale))
                    loss = loss.mean([1, 2, 3])

                    if cfg.loss.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, cfg.loss.min_snr_gamma, cfg.loss.v_parameterization)
                    if cfg.loss.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if cfg.loss.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, cfg.loss.v_parameterization)

                    loss = loss.mean()
                else:
                    loss = conditional_loss(noise_pred.float(), target.float(), cfg.loss.loss_type, "mean", huber_c, scale=float(cfg.loss.loss_scale))

                accelerator.backward(loss)
                if accelerator.sync_gradients and cfg.optimizer.max_grad_norm != 0.0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, cfg.optimizer.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                sample_images(
                    accelerator, cfg.sampling, cfg.training, cfg.saving, None, global_step, accelerator.device, vae, tokenize_strategy.tokenizer, text_encoder, unet
                )

                if cfg.saving.save_every_n_steps is not None and global_step % cfg.saving.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        save_sd_model_on_epoch_end_or_stepwise(
                            cfg.saving,
                            cfg.metadata,
                            cfg.loss,
                            cfg.model.v2,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder),
                            accelerator.unwrap_model(unet),
                            vae,
                        )

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                append_lr_to_logs(logs, lr_scheduler, cfg.optimizer.optimizer_type, including_unet=True)
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.training.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if cfg.saving.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                save_sd_model_on_epoch_end_or_stepwise(
                    cfg.saving,
                    cfg.metadata,
                    cfg.loss,
                    cfg.model.v2,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder),
                    accelerator.unwrap_model(unet),
                    vae,
                )

        sample_images(
            accelerator, cfg.sampling, cfg.training, cfg.saving, epoch + 1, global_step, accelerator.device, vae, tokenize_strategy.tokenizer, text_encoder, unet
        )

    is_main_process = accelerator.is_main_process
    if is_main_process:
        unet = accelerator.unwrap_model(unet)
        text_encoder = accelerator.unwrap_model(text_encoder)

    accelerator.end_training()

    if is_main_process and (cfg.saving.save_state or cfg.saving.save_state_on_train_end):
        save_state_on_train_end(cfg.saving, accelerator)

    del accelerator

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        save_sd_model_on_train_end(
            cfg.saving,
            cfg.metadata,
            cfg.loss,
            cfg.model.v2,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder,
            unet,
            vae,
        )
        logger.info("model saved.")


@hydra.main(config_path="../configs", config_name="sd_finetune", version_base=None)
def main(cfg: SDFineTuneConfig):
    prepare_config(cfg)
    validate_config(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
