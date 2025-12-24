
import torch

from library.utils import sai_model_spec
from library.models import sdxl_model_util

from library.training.checkpointing import (
    save_sd_model_on_train_end_common,
    save_sd_model_on_epoch_end_or_stepwise_common
)
from library.config.dataclasses.saving import SavingConfig
from library.config.dataclasses.metadata import MetadataConfig
from library.config.dataclasses.loss import LossConfig
from library.config.dataclasses.huggingface import HuggingFaceConfig
from typing import Optional
# TODO: TrainingConfig was only used for v_parameterization (which was a bug - it's in LossConfig).
# Consider adding clip_skip from TrainingConfig to metadata in the future.

def save_sd_model_on_train_end(
        saving_config: SavingConfig,
        metadata_config: MetadataConfig,
        loss_config: LossConfig,
        src_path: str,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        save_dtype: torch.dtype,
        epoch: int,
        global_step: int,
        text_encoder1,
        text_encoder2,
        unet,
        vae,
        logit_scale,
        ckpt_info,
        hf_config: Optional[HuggingFaceConfig] = None,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = sai_model_spec.get_sai_model_spec_from_config(
            state_dict=None,
            metadata_config=metadata_config,
            is_sdxl=True,
            is_v2=False, # SDXL is not v2
            v_parameterization=loss_config.v_parameterization,
            is_lora=False,
            is_textual_inversion=False,
            is_stable_diffusion_ckpt=True,
        )
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            sai_metadata,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )

    save_sd_model_on_train_end_common(
        saving_config, save_stable_diffusion_format, use_safetensors, epoch, global_step, sd_saver, diffusers_saver, hf_config
    )


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
        saving_config: SavingConfig,
        metadata_config: MetadataConfig,
        loss_config: LossConfig,
        on_epoch_end: bool,
        accelerator,
        src_path,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        save_dtype: torch.dtype,
        epoch: int,
        num_train_epochs: int,
        global_step: int,
        text_encoder1,
        text_encoder2,
        unet,
        vae,
        logit_scale,
        ckpt_info,
        hf_config: Optional[HuggingFaceConfig] = None,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = sai_model_spec.get_sai_model_spec_from_config(
            state_dict=None,
            metadata_config=metadata_config,
            is_sdxl=True,
            is_v2=False,
            v_parameterization=loss_config.v_parameterization,
            is_lora=False,
            is_textual_inversion=False,
            is_stable_diffusion_ckpt=True,
        )
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            sai_metadata,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )

    save_sd_model_on_epoch_end_or_stepwise_common(
        saving_config,
        on_epoch_end,
        accelerator,
        save_stable_diffusion_format,
        use_safetensors,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        diffusers_saver,
        hf_config,
    )
