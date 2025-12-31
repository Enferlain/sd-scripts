
import torch

from typing import Optional

import library.models.sd_model_util
from library.utils import model_metadata
from library.models import model_util
from library.config.dataclasses.loss import LossConfig

from library.config.dataclasses.output import (
    SavingConfig,
    MetadataConfig,
    HuggingFaceConfig
)

from library.training.checkpointing import (
    save_sd_model_on_train_end_common,
    save_sd_model_on_epoch_end_or_stepwise_common
)


def save_sd_model_on_train_end(
        saving_config: SavingConfig,
        metadata_config: MetadataConfig,
        loss_config: LossConfig,
        v2: bool,
        src_path: str,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        save_dtype: torch.dtype,
        epoch: int,
        global_step: int,
        text_encoder,
        unet,
        vae,
        hf_config: Optional[HuggingFaceConfig] = None,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        modelspec_metadata = model_metadata.get_model_metadata_from_config(
            state_dict=None,  # TODO: Expected type 'dict', got 'None' instead
            metadata_config=metadata_config,
            is_sdxl=False,
            is_v2=v2,
            v_parameterization=loss_config.v_parameterization,
            is_lora=False,
            is_textual_inversion=False,
            is_stable_diffusion_ckpt=True,
        )
        library.models.sd_model_util.save_stable_diffusion_checkpoint(
            v2, ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, modelspec_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        library.models.sd_model_util.save_diffusers_checkpoint(
            v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
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
        v2: bool,
        on_epoch_end: bool,
        accelerator,
        src_path: str,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        save_dtype: torch.dtype,
        epoch: int,
        num_train_epochs: int,
        global_step: int,
        text_encoder,
        unet,
        vae,
        hf_config: Optional[HuggingFaceConfig] = None,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        modelspec_metadata = model_metadata.get_model_metadata_from_config(
            state_dict=None,  # TODO: Expected type 'dict', got 'None' instead
            metadata_config=metadata_config,
            is_sdxl=False,
            is_v2=v2,
            v_parameterization=loss_config.v_parameterization,
            is_lora=False,
            is_textual_inversion=False,
            is_stable_diffusion_ckpt=True,
        )
        library.models.sd_model_util.save_stable_diffusion_checkpoint(
            v2, ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, modelspec_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        library.models.sd_model_util.save_diffusers_checkpoint(
            v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
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
