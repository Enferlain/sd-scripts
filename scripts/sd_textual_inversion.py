import hydra
import math
import os
import toml
import logging
import torch

from tqdm import tqdm
from multiprocessing import Value
from typing import Any, List, Optional, Union
from diffusers import DDPMScheduler
from transformers import CLIPTokenizer
from omegaconf import OmegaConf

import library.utils.huggingface_util as huggingface_util

from library.models import model_util
from library.utils import sai_model_spec
from library.performance import deepspeed_utils
from library.strategies import strategy_sd, strategy_base
from library.utils.torch_utils import prepare_dtype, set_seed_from_config
from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex, clean_memory_on_device
from library.data.prompt_templates import imagenet_templates_small, imagenet_style_templates_small
from library.data.dataset import DatasetGroup, MinimalDataset, load_arbitrary_dataset, collator_class, debug_dataset
from library.config.dataclasses.sd_textual_inversion import TextualInversionConfig

from library.training.model_prep import replace_unet_modules, patch_accelerator_for_fp16_training
from library.training.sd_model_prep import load_target_model
from library.training.trainer_utils import prepare_accelerator
from library.training.diffusion import get_noise_noisy_latents_and_timesteps
from library.training.optimizer import get_optimizer, get_scheduler_fix
from library.training.sd_sample_generation import sample_images
from library.losses.loss import conditional_loss, get_huber_threshold_if_needed
from library.config.validation import prepare_config, validate_config, validate_sd_textual_inversion

from library.config.config_util import (
    BlueprintGenerator,
    generate_dataset_group_by_blueprint,
    generate_dreambooth_subsets_config_by_subdirs,
)

from library.training.checkpointing import (
    resume_from_local_or_hf_if_specified,
    get_sai_model_spec,
    save_and_remove_state_stepwise,
    get_step_ckpt_name,
    get_remove_step_no,
    get_epoch_ckpt_name,
    get_remove_epoch_no,
    save_and_remove_state_on_epoch_end,
    save_state_on_train_end,
    get_last_ckpt_name
)

from library.training.noise_utils import (
    fix_noise_scheduler_betas_for_zero_terminal_snr,
    prepare_scheduler_for_custom_training
)

from library.losses.loss_weighting import (
    apply_debiased_estimation,
    add_v_prediction_like_loss,
    scale_v_prediction_loss_like_noise_prediction,
    apply_snr_weight,
    apply_masked_loss
)

init_ipex()

setup_logging()
logger = logging.getLogger(__name__)


class TextualInversionTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    def validate_extra_config(self, config, train_dataset_group: Union[DatasetGroup, MinimalDataset], val_dataset_group: Optional[
        DatasetGroup]):
        validate_sd_textual_inversion(config, train_dataset_group, val_dataset_group)

    def load_target_model(self, sd_models_config, performance_config, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = load_target_model(sd_models_config, performance_config, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(sd_models_config.v2, sd_models_config.v_parameterization), [text_encoder], vae, unet

    def get_tokenize_strategy(self, sd_models_config, training_config):
        return strategy_sd.SdTokenizeStrategy(sd_models_config.v2, training_config.max_token_length, sd_models_config.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, dataset_config):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, dataset_config.cache_latents_to_disk, dataset_config.vae_batch_size, dataset_config.skip_cache_check
        )
        return latents_caching_strategy

    def assert_token_string(self, token_string, tokenizers: CLIPTokenizer):
        pass

    def get_text_encoding_strategy(self, training_config):
        return strategy_sd.SdTextEncodingStrategy(training_config.clip_skip)

    def get_models_for_text_encoding(self, config, accelerator, text_encoders) -> List[Any]:
        return text_encoders

    def call_unet(self, config, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def sample_images(
        self, accelerator, sampling_config, training_config, saving_config, epoch, global_step, device, vae, tokenizers, text_encoders, unet, prompt_replacement
    ):
        sample_images(
            accelerator, sampling_config, training_config, saving_config, epoch, global_step, device, vae, tokenizers[0], text_encoders[0], unet, prompt_replacement
        )

    def save_weights(self, file, updated_embs, save_dtype, metadata):
        state_dict = {"emb_params": updated_embs[0]}

        if save_dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(save_dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)  # can be loaded in Web UI

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            data = load_file(file)
        else:
            # compatible to Web UI's file format
            data = torch.load(file, map_location="cpu")
            if type(data) != dict:
                raise ValueError(f"weight file is not dict / 重みファイルがdict形式ではありません: {file}")

            if "string_to_param" in data:  # textual inversion embeddings
                data = data["string_to_param"]
                if hasattr(data, "_parameters"):  # support old PyTorch?
                    data = getattr(data, "_parameters")

        emb = next(iter(data.values()))
        if type(emb) != torch.Tensor:
            raise ValueError(f"weight file does not contains Tensor / 重みファイルのデータがTensorではありません: {file}")

        if len(emb.size()) == 1:
            emb = emb.unsqueeze(0)

        return [emb]

    def train(self, config: TextualInversionConfig):
        ti_config = config.textual_inversion
        training_config = config.training
        dataset_config = config.dataset
        model_config = config.model
        optimizer_config = config.optimizer
        saving_config = config.saving
        
        if saving_config.output_name is None:
            saving_config.output_name = ti_config.token_string
        use_template = ti_config.use_object_template or ti_config.use_style_template

        # verify_training_args(args) # skipped, Hydra validation assumed
        # prepare_dataset_args(args, True) # skipped
        setup_logging(config.logging, reset=True)

        cache_latents = dataset_config.cache_latents

        set_seed_from_config(training_config)

        tokenize_strategy = self.get_tokenize_strategy(model_config, training_config)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)

        latents_caching_strategy = self.get_latents_caching_strategy(dataset_config)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        logger.info("prepare accelerator")
        accelerator = prepare_accelerator(training_config)

        weight_dtype, save_dtype = prepare_dtype(config.performance, saving_config)
        vae_dtype = torch.float32 if config.performance.no_half_vae else weight_dtype

        model_version, text_encoders, vae, unet = self.load_target_model(model_config, config.performance, weight_dtype, accelerator)

        init_token_ids_list = []
        if ti_config.init_word is not None:
            for i, tokenizer in enumerate(tokenizers):
                init_token_ids = tokenizer.encode(ti_config.init_word, add_special_tokens=False)
                if len(init_token_ids) > 1 and len(init_token_ids) != ti_config.num_vectors_per_token:
                    accelerator.print(
                        f"token length for init words is not same to num_vectors_per_token, init words is repeated or truncated: tokenizer {i+1}, length {len(init_token_ids)}"
                    )
                init_token_ids_list.append(init_token_ids)
        else:
            init_token_ids_list = [None] * len(tokenizers)

        self.assert_token_string(ti_config.token_string, tokenizers)

        token_strings = [ti_config.token_string] + [f"{ti_config.token_string}{i+1}" for i in range(ti_config.num_vectors_per_token - 1)]
        token_ids_list = []
        token_embeds_list = []
        for i, (tokenizer, text_encoder, init_token_ids) in enumerate(zip(tokenizers, text_encoders, init_token_ids_list)):
            num_added_tokens = tokenizer.add_tokens(token_strings)
            assert (
                num_added_tokens == ti_config.num_vectors_per_token
            ), f"tokenizer has same word to token string. please use another one: tokenizer {i+1}, {ti_config.token_string}"

            token_ids = tokenizer.convert_tokens_to_ids(token_strings)
            accelerator.print(f"tokens are added for tokenizer {i+1}: {token_ids}")
            assert (
                min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(token_ids) - 1
            ), f"token ids is not ordered : tokenizer {i+1}, {token_ids}"
            assert (
                len(tokenizer) - 1 == token_ids[-1]
            ), f"token ids is not end of tokenize: tokenizer {i+1}, {token_ids}, {len(tokenizer)}"
            token_ids_list.append(token_ids)

            text_encoder.resize_token_embeddings(len(tokenizer))

            token_embeds = text_encoder.get_input_embeddings().weight.data
            if init_token_ids is not None:
                for i, token_id in enumerate(token_ids):
                    token_embeds[token_id] = token_embeds[init_token_ids[i % len(init_token_ids)]]
                    # accelerator.print(token_id, token_embeds[token_id].mean(), token_embeds[token_id].min())
            token_embeds_list.append(token_embeds)

        if ti_config.weights is not None:
            embeddings_list = self.load_weights(ti_config.weights)
            assert len(token_ids) == len(
                embeddings_list[0]
            ), f"num_vectors_per_token is mismatch for weights: {len(embeddings_list[0])}"
            for token_ids, embeddings, token_embeds in zip(token_ids_list, embeddings_list, token_embeds_list):
                for token_id, embedding in zip(token_ids, embeddings):
                    token_embeds[token_id] = embedding
            accelerator.print(f"weights loaded")

        accelerator.print(f"create embeddings for {ti_config.num_vectors_per_token} tokens, for {ti_config.token_string}")

        if dataset_config.dataset_class is None:
            blueprint_generator = BlueprintGenerator()
            # BlueprintGenerator.generate() expects a RootConfig-like object with a .dataset attribute.
            blueprint = blueprint_generator.generate(config)
            train_dataset_group, val_dataset_group = generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            # load_arbitrary_dataset now accepts DatasetConfig directly
            train_dataset_group = load_arbitrary_dataset(dataset_config)
            val_dataset_group = None

        self.validate_extra_config(None, train_dataset_group, val_dataset_group)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if training_config.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        if use_template:
            accelerator.print(f"use template for training captions. is object: {ti_config.use_object_template}")
            templates = imagenet_templates_small if ti_config.use_object_template else imagenet_style_templates_small
            replace_to = " ".join(token_strings)
            captions = []
            for tmpl in templates:
                captions.append(tmpl.format(replace_to))
            train_dataset_group.add_replacement("", captions)

            if ti_config.num_vectors_per_token > 1:
                prompt_replacement = (ti_config.token_string, replace_to)
            else:
                prompt_replacement = None
        else:
            if ti_config.num_vectors_per_token > 1:
                replace_to = " ".join(token_strings)
                train_dataset_group.add_replacement(ti_config.token_string, replace_to)
                prompt_replacement = (ti_config.token_string, replace_to)
            else:
                prompt_replacement = None

        if dataset_config.debug_dataset:
            debug_dataset(train_dataset_group, show_input_ids=True)
            return
        if len(train_dataset_group) == 0:
            accelerator.print("No data found. Please verify arguments")
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used"

        replace_unet_modules(unet, config.performance.mem_eff_attn, config.performance.xformers, config.performance.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(config.performance.xformers)

        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

            train_dataset_group.new_cache_latents(vae, accelerator)

            clean_memory_on_device(accelerator.device)
            accelerator.wait_for_everyone()

        if training_config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for text_encoder in text_encoders:
                text_encoder.gradient_checkpointing_enable()

        accelerator.print("prepare optimizer, data loader etc.")
        trainable_params = []
        for text_encoder in text_encoders:
            trainable_params += text_encoder.get_input_embeddings().parameters()
        _, _, optimizer = get_optimizer(optimizer_config, trainable_params)

        train_dataset_group.set_current_strategies()

        n_workers = min(training_config.max_data_loader_n_workers, os.cpu_count())
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=training_config.persistent_data_loader_workers,
        )

        if training_config.max_train_epochs is not None:
            training_config.max_train_steps = training_config.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / training_config.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {training_config.max_train_epochs} epochs is: {training_config.max_train_steps}"
            )

        train_dataset_group.set_max_train_steps(training_config.max_train_steps)

        lr_scheduler = get_scheduler_fix(config.optimizer, config.dataset, config.training, optimizer, accelerator.num_processes)

        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        text_encoders = [accelerator.prepare(text_encoder) for text_encoder in text_encoders]

        index_no_updates_list = []
        orig_embeds_params_list = []
        for tokenizer, token_ids, text_encoder in zip(tokenizers, token_ids_list, text_encoders):
            index_no_updates = torch.arange(len(tokenizer)) < token_ids[0]
            index_no_updates_list.append(index_no_updates)

            orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.detach().clone()
            orig_embeds_params_list.append(orig_embeds_params)

            text_encoder.requires_grad_(True)
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            unwrapped_text_encoder.text_model.encoder.requires_grad_(False)
            unwrapped_text_encoder.text_model.final_layer_norm.requires_grad_(False)
            unwrapped_text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        unet.requires_grad_(False)
        unet.to(accelerator.device, dtype=weight_dtype)
        if training_config.gradient_checkpointing:
            unet.train()
        else:
            unet.eval()

        text_encoding_strategy = self.get_text_encoding_strategy(training_config)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if training_config.full_fp16:
            patch_accelerator_for_fp16_training(accelerator)
            for text_encoder in text_encoders:
                text_encoder.to(weight_dtype)
        if training_config.full_bf16:
            for text_encoder in text_encoders:
                text_encoder.to(weight_dtype)

        resume_from_local_or_hf_if_specified(accelerator, saving_config)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_config.gradient_accumulation_steps)
        num_train_epochs = math.ceil(training_config.max_train_steps / num_update_steps_per_epoch)
        if (saving_config.save_n_epoch_ratio is not None) and (saving_config.save_n_epoch_ratio > 0):
            saving_config.save_every_n_epochs = math.floor(num_train_epochs / saving_config.save_n_epoch_ratio) or 1

        total_batch_size = training_config.train_batch_size * accelerator.num_processes * training_config.gradient_accumulation_steps
        accelerator.print("running training")
        accelerator.print(f"  num train images * repeats: {train_dataset_group.num_train_images}")
        accelerator.print(f"  num reg images: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
        accelerator.print(f"  num epochs: {num_train_epochs}")
        accelerator.print(f"  batch size per device: {training_config.train_batch_size}")
        accelerator.print(
            f"  total train batch size (with parallel & distributed & accumulation): {total_batch_size}"
        )
        accelerator.print(f"  gradient accumulation steps = {training_config.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps: {training_config.max_train_steps}")

        progress_bar = tqdm(range(training_config.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
        global_step = 0

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )

        if training_config.zero_terminal_snr:
            fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)

        if accelerator.is_main_process:
            init_kwargs = {}
            if config.logging.wandb_run_name:
                init_kwargs["wandb"] = {"name": config.logging.wandb_run_name}
            if config.logging.log_tracker_config is not None:
                init_kwargs = config.logging.log_tracker_config
            accelerator.init_trackers(
                "textual_inversion" if config.logging.log_tracker_name is None else config.logging.log_tracker_name,
                config=OmegaConf.to_container(config),
                init_kwargs=init_kwargs,
            )

        def save_model(ckpt_name, embs_list, steps, epoch_no, force_sync_upload=False):
            os.makedirs(saving_config.output_dir, exist_ok=True)
            ckpt_file = os.path.join(saving_config.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")

            sai_metadata = sai_model_spec.get_sai_model_spec_from_config(
                state_dict=None,
                metadata_config=config.metadata,
                is_sdxl=self.is_sdxl,
                is_v2=model_config.v2,
                v_parameterization=config.loss.v_parameterization,
                is_lora=False,
                is_textual_inversion=True,
            )

            self.save_weights(ckpt_file, embs_list, save_dtype, sai_metadata)
            if config.huggingface.huggingface_repo_id is not None:
                huggingface_util.upload(config.huggingface, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(saving_config.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        self.sample_images(
            accelerator,
            config.sampling,
            config.training,
            config.saving,
            0,
            global_step,
            accelerator.device,
            vae,
            tokenizers,
            text_encoders,
            unet,
            prompt_replacement,
        )
        if len(accelerator.trackers) > 0:
            accelerator.log({}, step=0)

        # training loop
        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            for text_encoder in text_encoders:
                text_encoder.train()

            loss_total = 0

            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step
                with accelerator.accumulate(text_encoders[0]):
                    with torch.no_grad():
                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                        else:
                            # latentに変換
                            latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample().to(dtype=weight_dtype)
                        latents = latents * self.vae_scale_factor

                    input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                    text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy, self.get_models_for_text_encoding(model_config, accelerator, text_encoders), input_ids
                    )
                    if training_config.full_fp16:
                        text_encoder_conds = [c.to(weight_dtype) for c in text_encoder_conds]

                    noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
                        training_config, noise_scheduler, latents
                    )

                    with accelerator.autocast():
                        noise_pred = self.call_unet(
                            model_config, accelerator, unet, noisy_latents, timesteps, text_encoder_conds, batch, weight_dtype
                        )

                    if config.loss.v_parameterization:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    huber_c = get_huber_threshold_if_needed(config.loss, timesteps, noise_scheduler)
                    loss = conditional_loss(noise_pred.float(), target.float(), config.loss.loss_type, "none", huber_c, scale=float(config.loss.loss_scale))
                    if config.masked_loss.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])

                    loss_weights = batch["loss_weights"]
                    loss = loss * loss_weights

                    if config.loss.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, config.loss.min_snr_gamma, config.loss.v_parameterization)
                    if config.loss.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if config.loss.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, config.loss.v_pred_like_loss)
                    if config.loss.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, config.loss.v_parameterization)

                    loss = loss.mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and optimizer_config.max_grad_norm != 0.0:
                        params_to_clip = accelerator.unwrap_model(text_encoder).get_input_embeddings().parameters()
                        accelerator.clip_grad_norm_(params_to_clip, optimizer_config.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        for text_encoder, orig_embeds_params, index_no_updates in zip(
                            text_encoders, orig_embeds_params_list, index_no_updates_list
                        ):
                            input_embeddings_weight = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
                            input_embeddings_weight[index_no_updates] = orig_embeds_params.to(input_embeddings_weight.dtype)[
                                index_no_updates
                            ]

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.sample_images(
                        accelerator,
                        config.sampling,
                        config.training,
                        config.saving,
                        None,
                        global_step,
                        accelerator.device,
                        vae,
                        tokenizers,
                        text_encoders,
                        unet,
                        prompt_replacement,
                    )

                    if saving_config.save_every_n_steps is not None and global_step % saving_config.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            updated_embs_list = []
                            for text_encoder, token_ids in zip(text_encoders, token_ids_list):
                                updated_embs = (
                                    accelerator.unwrap_model(text_encoder)
                                    .get_input_embeddings()
                                    .weight[token_ids]
                                    .data.detach()
                                    .clone()
                                )
                                updated_embs_list.append(updated_embs)

                            ckpt_name = get_step_ckpt_name(saving_config, "." + saving_config.save_model_as, global_step)
                            save_model(ckpt_name, updated_embs_list, global_step, epoch)

                            if saving_config.save_state:
                                save_and_remove_state_stepwise(saving_config, accelerator, global_step)

                            remove_step_no = get_remove_step_no(saving_config, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = get_step_ckpt_name(saving_config, "." + saving_config.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)

                current_loss = loss.detach().item()
                if len(accelerator.trackers) > 0:
                    logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
                    if (
                        optimizer_config.optimizer_type.lower().startswith("DAdapt".lower()) or optimizer_config.optimizer_type.lower() == "Prodigy".lower()
                    ):
                        logs["lr/d*lr"] = (
                            lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"]
                        )
                    accelerator.log(logs, step=global_step)

                loss_total += current_loss
                avr_loss = loss_total / (step + 1)
                logs = {"loss": avr_loss}
                progress_bar.set_postfix(**logs)

                if global_step >= training_config.max_train_steps:
                    break

            if len(accelerator.trackers) > 0:
                logs = {"loss/epoch": loss_total / len(train_dataloader)}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            updated_embs_list = []
            for text_encoder, token_ids in zip(text_encoders, token_ids_list):
                updated_embs = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[token_ids].data.detach().clone()
                updated_embs_list.append(updated_embs)

            if saving_config.save_every_n_epochs is not None:
                saving = (epoch + 1) % saving_config.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if accelerator.is_main_process and saving:
                    ckpt_name = get_epoch_ckpt_name(saving_config, "." + saving_config.save_model_as, epoch + 1)
                    save_model(ckpt_name, updated_embs_list, epoch + 1, global_step)

                    remove_epoch_no = get_remove_epoch_no(saving_config, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = get_epoch_ckpt_name(saving_config, "." + saving_config.save_model_as, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if saving_config.save_state:
                        save_and_remove_state_on_epoch_end(saving_config, accelerator, epoch + 1)

            self.sample_images(
                accelerator,
                config.sampling,
                config.training,
                config.saving,
                epoch + 1,
                global_step,
                accelerator.device,
                vae,
                tokenizers,
                text_encoders,
                unet,
                prompt_replacement,
            )
            accelerator.log({})

        is_main_process = accelerator.is_main_process
        if is_main_process:
            text_encoder = accelerator.unwrap_model(text_encoder)
            updated_embs = text_encoder.get_input_embeddings().weight[token_ids].data.detach().clone()

        accelerator.end_training()

        if is_main_process and (saving_config.save_state or saving_config.save_state_on_train_end):
            save_state_on_train_end(saving_config, accelerator)

        if is_main_process:
            ckpt_name = get_last_ckpt_name(saving_config, "." + saving_config.save_model_as)
            save_model(ckpt_name, updated_embs_list, global_step, num_train_epochs, force_sync_upload=True)

            logger.info("model saved.")


@hydra.main(config_path="../configs", config_name="sd_textual_inversion", version_base=None)
def main(config: TextualInversionConfig):
    prepare_config(config)
    validate_config(config)
    trainer = TextualInversionTrainer()
    trainer.train(config)


if __name__ == "__main__":
    main()
