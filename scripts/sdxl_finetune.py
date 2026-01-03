import hydra
import math
import os
import torch
import logging

from multiprocessing import Value
from tqdm import tqdm
from diffusers import DDPMScheduler

import library.config.config_util as config_util

from library.constants import SDXL_VAE_LATENT_SCALE
from library.models.sdxl_model_util import get_size_embeddings
from library.utils.device_utils import init_ipex, clean_memory_on_device
from library.utils.common_utils import setup_logging
from library.utils.torch_utils import set_torch_cuda_reduced_precision, set_seed_from_config, prepare_dtype
from library.performance import deepspeed_utils
from library.models.sdxl_original_unet import SdxlUNet2DConditionModel
from library.strategies import strategy_sdxl, strategy_sd, strategy_base
from library.data.dataset import load_arbitrary_dataset, collator_class, debug_dataset
from library.training.checkpointing import resume_from_local_or_hf_if_specified, save_state_on_train_end
from library.training.sdxl_checkpointing import save_sd_model_on_epoch_end_or_stepwise, save_sd_model_on_train_end
from library.models.sdxl_model_prep import load_target_model
from library.training.sdxl_sample_generation import sample_images
from library.training.diffusion import get_noise_noisy_latents_and_timesteps
from library.models.model_prep import replace_unet_modules, patch_accelerator_for_fp16_training
from library.optimizers.scheduler import get_scheduler_fix
from library.optimizers.optimizer_factory import get_optimizer
from library.training.trainer_utils import prepare_accelerator, append_lr_to_logs
from library.logging.step_logging import append_lr_to_logs_with_names
from library.losses.loss import LossRecorder, get_huber_threshold_if_needed, conditional_loss
from library.config.dataclasses.sdxl_finetune import SDXLFineTuneConfig
from library.config.config_validation import prepare_config, validate_config

from library.config.config_util import (
    BlueprintGenerator,
)

from library.losses.loss_weighting import (
    apply_masked_loss,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_snr_weight,
)

from library.training.noise_utils import fix_noise_scheduler_betas_for_zero_terminal_snr, prepare_scheduler_for_custom_training

init_ipex()

setup_logging()
logger = logging.getLogger(__name__)

UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23


def get_block_params_to_optimize(unet: SdxlUNet2DConditionModel, block_lrs: list[float]) -> list[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


# Register Hydra schema for this script
from library.config.schemas import register_sdxl_finetune

register_sdxl_finetune()


@hydra.main(version_base=None, config_path="../configs", config_name="sdxl_finetune")
def train(cfg: SDXLFineTuneConfig):
    prepare_config(cfg)
    validate_config(cfg)

    if cfg.training.dry_run:
        print("Dry run completed successfully.")
        return

    set_torch_cuda_reduced_precision(cfg.performance.precision)
    deepspeed_utils.prepare_deepspeed_config(cfg.performance.deepspeed)
    setup_logging(cfg.output.logging, reset=True)

    if cfg.optimizer.learning_rates.blocks:
        block_lrs = [float(lr) for lr in cfg.optimizer.learning_rates.blocks.split(",")]
    else:
        block_lrs = None

    cache_latents = cfg.data.caching.cache_latents
    use_dreambooth_method = cfg.data.source.in_json is None

    set_seed_from_config(cfg.training)

    tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(cfg.training.max_token_length, cfg.model.tokenizer_cache_dir)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    tokenizers = [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    if cfg.data.caching.cache_latents:
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, cfg.data.caching.cache_latents_to_disk, cfg.data.caching.vae_batch_size, cfg.data.caching.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    if cfg.data.source.dataset_class is None:
        blueprint_generator = BlueprintGenerator()
        blueprint = blueprint_generator.generate(cfg)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = load_arbitrary_dataset(cfg.data, cfg.training.max_token_length)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if cfg.data.loader.max_workers == 0 else None
    collator = collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if cfg.data.preprocessing.debug_dataset:
        debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify the metadata file and train_data_dir option.")
        return

    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), "when caching latents, either color_aug or random_crop cannot be used"

    if cfg.performance.caching.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(), (
            "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"
        )

    logger.info("prepare accelerator")
    accelerator = prepare_accelerator(
        cfg.performance.precision,
        cfg.performance.compilation,
        cfg.performance.distributed,
        cfg.performance.deepspeed,
    )

    weight_dtype, save_dtype = prepare_dtype(cfg.performance.precision, cfg.output.saving)
    vae_dtype = torch.float32 if cfg.performance.precision.no_half_vae else weight_dtype

    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = load_target_model(
        cfg.model,
        cfg.performance.memory,
        cfg.performance.caching,
        cfg.performance.precision,
        accelerator,
        "sdxl",
        weight_dtype,
    )

    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = cfg.model.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = cfg.model.pretrained_model_name_or_path

    if cfg.output.saving.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = cfg.output.saving.use_safetensors
    else:
        save_stable_diffusion_format = (
            cfg.output.saving.save_model_as.lower() == "ckpt" or cfg.output.saving.save_model_as.lower() == "safetensors"
        )
        use_safetensors = cfg.output.saving.use_safetensors or ("safetensors" in cfg.output.saving.save_model_as.lower())

    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    if cfg.performance.attention.diffusers_xformers:
        accelerator.print("Use xformers by Diffusers")
        set_diffusers_xformers_flag(vae, True)
    else:
        accelerator.print("Disable Diffusers' xformers")
        replace_unet_modules(
            unet, cfg.performance.attention.mem_eff_attn, cfg.performance.attention.xformers, cfg.performance.attention.sdpa
        )
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(cfg.performance.attention.xformers)

    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    if cfg.performance.memory.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    train_unet = cfg.optimizer.learning_rates.base != 0
    train_text_encoder1 = False
    train_text_encoder2 = False

    text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # Train text encoder if TE LR > 0 (based on LR-based training control)
    from library.optimizers.optimizer_utils import should_train_text_encoder

    train_te_based_on_lr = should_train_text_encoder(cfg.optimizer.learning_rates)
    if train_te_based_on_lr:
        accelerator.print("enable text encoder training")
        if cfg.performance.memory.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()

        # Schema 1: Resolve TE LRs from optimizer.learning_rates.text_encoders
        # It can be a list [lr_te1, lr_te2] or a single scalar (applied to both)
        lr_te_schema = cfg.optimizer.learning_rates.text_encoders

        if lr_te_schema is not None:
            if isinstance(lr_te_schema, list):
                lr_te1 = lr_te_schema[0]
                lr_te2 = lr_te_schema[1] if len(lr_te_schema) > 1 else lr_te_schema[0]
            else:
                lr_te1 = lr_te_schema
                lr_te2 = lr_te_schema
        else:
            # No TE-specific LR, use base LR
            lr_te1 = cfg.optimizer.learning_rates.base
            lr_te2 = cfg.optimizer.learning_rates.base

        train_text_encoder1 = lr_te1 != 0
        train_text_encoder2 = lr_te2 != 0

        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder2.requires_grad_(train_text_encoder2)
        text_encoder1.train(train_text_encoder1)
        text_encoder2.train(train_text_encoder2)
    else:
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

        if cfg.performance.caching.cache_text_encoder_outputs:
            text_encoder_output_caching_strategy = strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                cfg.performance.caching.cache_text_encoder_outputs_to_disk, None, False, is_weighted=cfg.data.caption.weighted_captions
            )  # TODO: Expected type 'int', got 'None' instead
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_output_caching_strategy)

            text_encoder1.to(accelerator.device)
            text_encoder2.to(accelerator.device)
            with accelerator.autocast():
                train_dataset_group.new_cache_text_encoder_outputs([text_encoder1, text_encoder2], accelerator)

        accelerator.wait_for_everyone()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)

    training_models = []
    params_to_optimize = []
    if train_unet:
        training_models.append(unet)
        if block_lrs is None:
            params_to_optimize.append(
                {"params": list(unet.parameters()), "lr": cfg.optimizer.learning_rates.unet or cfg.optimizer.learning_rates.base}
            )
        else:
            params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs))

    if train_text_encoder1:
        training_models.append(text_encoder1)
        params_to_optimize.append(
            {"params": list(text_encoder1.parameters()), "lr": lr_te1}
        )  # TODO: Local variable 'lr_te1' might be referenced before assignment
    if train_text_encoder2:
        training_models.append(text_encoder2)
        params_to_optimize.append(
            {"params": list(text_encoder2.parameters()), "lr": lr_te2}
        )  # TODO: Local variable 'lr_te2' might be referenced before assignment

    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    accelerator.print("prepare optimizer, data loader etc.")

    if cfg.optimizer.fused_optimizer_groups:
        n_total_params = sum(len(params["params"]) for params in params_to_optimize)
        params_per_group = math.ceil(n_total_params / cfg.optimizer.fused_optimizer_groups)

        grouped_params = []
        param_group = []
        param_group_lr = -1
        for group in params_to_optimize:
            lr = group["lr"]
            for p in group["params"]:
                if lr != param_group_lr:
                    if param_group:
                        grouped_params.append({"params": param_group, "lr": param_group_lr})
                        param_group = []
                    param_group_lr = lr

                param_group.append(p)

                if len(param_group) == params_per_group:
                    grouped_params.append({"params": param_group, "lr": param_group_lr})
                    param_group = []
                    param_group_lr = -1

        if param_group:
            grouped_params.append({"params": param_group, "lr": param_group_lr})

        optimizers = []
        for group in grouped_params:
            _, _, optimizer = get_optimizer(cfg.optimizer, cfg.optimizer.learning_rates, cfg.optimizer.scheduler, trainable_params=[group])
            optimizers.append(optimizer)
        optimizer = optimizers[0]

        logger.info(f"using {len(optimizers)} optimizers for fused optimizer groups")

    else:
        _, _, optimizer = get_optimizer(
            cfg.optimizer, cfg.optimizer.learning_rates, cfg.optimizer.scheduler, trainable_params=params_to_optimize
        )

    train_dataset_group.set_current_strategies()

    n_workers = min(cfg.data.loader.max_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=cfg.data.loader.persistent_workers,
    )

    if cfg.training.max_train_epochs is not None:
        cfg.training.max_train_steps = cfg.training.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / cfg.training.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {cfg.training.max_train_epochs} epochs is / 指定エポックまでのステップ数: {cfg.training.max_train_steps}"
        )

    train_dataset_group.set_max_train_steps(cfg.training.max_train_steps)

    if cfg.optimizer.fused_optimizer_groups:
        lr_schedulers = [
            get_scheduler_fix(cfg.optimizer.scheduler, cfg.optimizer, cfg.training, optimizer, accelerator.num_processes)
            for optimizer in optimizers
        ]
        lr_scheduler = lr_schedulers[0]
    else:
        lr_scheduler = get_scheduler_fix(cfg.optimizer.scheduler, cfg.optimizer, cfg.training, optimizer, accelerator.num_processes)

    if cfg.performance.precision.full_fp16:
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif cfg.performance.precision.full_bf16:
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)

    if train_text_encoder1:
        text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
        text_encoder1.text_model.final_layer_norm.requires_grad_(False)

    if cfg.performance.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            cfg.performance.precision,
            unet=unet if train_unet else None,
            text_encoder1=text_encoder1 if train_text_encoder1 else None,
            text_encoder2=text_encoder2 if train_text_encoder2 else None,
        )
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(ds_model, optimizer, train_dataloader, lr_scheduler)
        training_models = [ds_model]

    else:
        if train_unet:
            unet = accelerator.prepare(unet)
        if train_text_encoder1:
            text_encoder1 = accelerator.prepare(text_encoder1)
        if train_text_encoder2:
            text_encoder2 = accelerator.prepare(text_encoder2)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    if cfg.performance.caching.cache_text_encoder_outputs:
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)

    if cfg.performance.precision.full_fp16:
        patch_accelerator_for_fp16_training(accelerator)

    resume_from_local_or_hf_if_specified(accelerator, cfg.output.saving, cfg.output.huggingface)

    if cfg.optimizer.fused_backward_pass:
        import library.optimizers.adafactor_fused

        library.optimizers.adafactor_fused.patch_adafactor_fused(optimizer)
        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def __grad_hook(tensor: torch.Tensor, param_group=param_group):
                        if accelerator.sync_gradients and cfg.optimizer.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(tensor, cfg.optimizer.max_grad_norm)
                        optimizer.step_param(tensor, param_group)
                        tensor.grad = None

                    parameter.register_post_accumulate_grad_hook(__grad_hook)

    elif cfg.optimizer.fused_optimizer_groups:
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        global optimizer_hooked_count
        global num_parameters_per_group
        global parameter_optimizer_map

        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, optimizer in enumerate(optimizers):
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def optimizer_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and cfg.optimizer.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, cfg.optimizer.max_grad_norm)

                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)

                        parameter.register_post_accumulate_grad_hook(optimizer_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)
    if (cfg.output.saving.save_n_epoch_ratio is not None) and (cfg.output.saving.save_n_epoch_ratio > 0):
        cfg.output.saving.save_every_n_epochs = math.floor(num_train_epochs / cfg.output.saving.save_n_epoch_ratio) or 1

    accelerator.print("running training")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {cfg.training.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {cfg.training.max_train_steps}")

    progress_bar = tqdm(range(cfg.training.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )

    if cfg.loss.regularization.zero_terminal_snr:
        fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)

    if accelerator.is_main_process:
        init_kwargs = {}
        if cfg.output.logging.wandb_run_name:
            init_kwargs["wandb"] = {"name": cfg.output.logging.wandb_run_name}
        if cfg.output.logging.log_tracker_config is not None:
            init_kwargs = cfg.output.logging.log_tracker_config
        library.logging.step_logging.init_trackers(  # TODO: Local variable 'library' might be referenced before assignment
            "finetuning" if cfg.output.logging.log_tracker_name is None else cfg.output.logging.log_tracker_name,
            init_kwargs=init_kwargs,  # TODO: Unexpected argument
        )  # TODO: Parameter 'logging_config' unfilled, Parameter 'default_tracker_name' unfilled

    sample_images(
        accelerator,
        cfg.output.sampling,
        cfg.training,
        cfg.output.saving,
        cfg.loss,
        0,
        global_step,
        accelerator.device,
        vae,
        tokenizers,
        [text_encoder1, text_encoder2],
        unet,
    )
    if len(accelerator.trackers) > 0:
        accelerator.log({}, step=0)

    loss_recorder = LossRecorder()
    epoch = 0  # Initialize before loop to handle edge case of 0 epochs
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if cfg.optimizer.fused_optimizer_groups:
                optimizer_hooked_count = dict.fromkeys(range(len(optimizers)), 0)

            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                latents = latents * SDXL_VAE_LATENT_SCALE

                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_outputs_list
                    encoder_hidden_states1 = encoder_hidden_states1.to(accelerator.device, dtype=weight_dtype)
                    encoder_hidden_states2 = encoder_hidden_states2.to(accelerator.device, dtype=weight_dtype)
                    pool2 = pool2.to(accelerator.device, dtype=weight_dtype)
                else:
                    input_ids1, input_ids2 = batch["input_ids_list"]
                    with torch.set_grad_enabled(train_te_based_on_lr):
                        if cfg.data.caption.weighted_captions:
                            input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(
                                batch["captions"]
                            )  #  TODO: Need more values to unpack
                            encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoding_strategy.encode_tokens_with_weights(
                                tokenize_strategy,
                                [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                input_ids_list,
                                weights_list,
                            )
                        else:
                            input_ids1 = input_ids1.to(accelerator.device)
                            input_ids2 = input_ids2.to(accelerator.device)
                            encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                [input_ids1, input_ids2],
                            )
                        if cfg.performance.precision.full_fp16:
                            encoder_hidden_states1 = encoder_hidden_states1.to(weight_dtype)
                            encoder_hidden_states2 = encoder_hidden_states2.to(weight_dtype)
                            pool2 = pool2.to(weight_dtype)

                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
                    cfg.loss.regularization, noise_scheduler, latents, output_dtype=weight_dtype
                )  # TODO: Parameter 'noise_scheduler' unfilled, Parameter 'latents' unfilled

                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                if cfg.loss.v_parameterization:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                huber_c = get_huber_threshold_if_needed(cfg.loss, timesteps, noise_scheduler)
                if (
                    cfg.loss.snr.min_snr_gamma
                    or cfg.loss.snr.scale_v_pred_loss_like_noise_pred
                    or cfg.loss.snr.v_pred_like_loss
                    or cfg.loss.snr.debiased_estimation_loss
                    or cfg.loss.masked
                ):
                    loss = conditional_loss(
                        noise_pred.float(), target.float(), cfg.loss.loss_type, "none", huber_c, scale=float(cfg.loss.loss_scale)
                    )
                    if cfg.loss.masked or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])

                    if cfg.loss.snr.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, cfg.loss.snr.min_snr_gamma, cfg.loss.v_parameterization)
                    if cfg.loss.snr.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if cfg.loss.snr.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(
                            loss, timesteps, noise_scheduler, cfg.loss.snr.v_pred_like_loss
                        )  # TODO: Expected type 'Tensor', got 'float' instead
                    if cfg.loss.snr.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, cfg.loss.v_parameterization)

                    loss = loss.mean()
                else:
                    loss = conditional_loss(
                        noise_pred.float(), target.float(), cfg.loss.loss_type, "mean", huber_c, scale=float(cfg.loss.loss_scale)
                    )

                accelerator.backward(loss)

                if not (cfg.optimizer.fused_backward_pass or cfg.optimizer.fused_optimizer_groups):
                    if accelerator.sync_gradients and cfg.optimizer.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, cfg.optimizer.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    lr_scheduler.step()
                    if cfg.optimizer.fused_optimizer_groups:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                sample_images(
                    accelerator,
                    cfg.output.sampling,
                    cfg.training,
                    cfg.output.saving,
                    cfg.loss,
                    epoch + 1,
                    global_step,
                    accelerator.device,
                    vae,
                    tokenizers,
                    [text_encoder1, text_encoder2],
                    unet,
                )

                if cfg.output.saving.save_every_n_steps is not None and global_step % cfg.output.saving.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        save_sd_model_on_epoch_end_or_stepwise(
                            cfg.output.saving,
                            cfg.output.metadata,
                            cfg.loss,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder1),
                            accelerator.unwrap_model(text_encoder2),
                            accelerator.unwrap_model(unet),
                            vae,
                            logit_scale,
                            ckpt_info,
                        )

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                if block_lrs is None:
                    append_lr_to_logs(logs, lr_scheduler, cfg.optimizer.optimizer_type, including_unet=train_unet)
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler, cfg.optimizer.optimizer_type)

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

        if cfg.output.saving.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                save_sd_model_on_epoch_end_or_stepwise(
                    cfg.output.saving,
                    cfg.output.metadata,
                    cfg.loss,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder1),
                    accelerator.unwrap_model(text_encoder2),
                    accelerator.unwrap_model(unet),
                    vae,
                    logit_scale,
                    ckpt_info,
                )

        sample_images(
            accelerator,
            cfg.output.sampling,
            cfg.training,
            cfg.output.saving,
            cfg.loss,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            tokenizers,
            [text_encoder1, text_encoder2],
            unet,
        )

    is_main_process = accelerator.is_main_process
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()

    if cfg.output.saving.save_state or cfg.output.saving.save_state_on_train_end:
        save_state_on_train_end(cfg.output.saving, accelerator)

    del accelerator

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        save_sd_model_on_train_end(
            cfg.output.saving,
            cfg.output.metadata,
            cfg.loss,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info,
        )
        logger.info("model saved.")


if __name__ == "__main__":
    train()
