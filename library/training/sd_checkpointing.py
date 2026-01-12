import torch
from typing import TYPE_CHECKING
import library.models.sd.conversion
from library.utils import model_metadata
from library.config.dataclasses.loss import LossConfig
from library.config.dataclasses.output import SavingConfig, MetadataConfig, HuggingFaceConfig
from library.training.checkpointing import save_sd_model_on_train_end_common, save_sd_model_on_epoch_end_or_stepwise_common

if TYPE_CHECKING:
    from accelerate import Accelerator


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
    hf_config: HuggingFaceConfig | None = None,
) -> None:
    """
    Saves the Stable Diffusion (v1.5/v2) model at the end of training.

    Args:
        saving_config: Configuration for saving outputs.
        metadata_config: Configuration for metadata generation.
        loss_config: Configuration for loss parameters (used for v_parameterization).
        v2: Boolean indicating if the model is V2.
        src_path: Path to the source model.
        save_stable_diffusion_format: Whether to save in SD format.
        use_safetensors: Whether to use SafeTensors format.
        save_dtype: Data type for saving weights.
        epoch: Final epoch number.
        global_step: Final global step count.
        text_encoder: Text encoder model.
        unet: UNet model.
        vae: VAE model.
        hf_config: Configuration for Hugging Face integration.
    """

    def sd_saver(ckpt_file, epoch_no, global_step):
        modelspec_metadata = model_metadata.get_model_metadata_from_config(
            state_dict=None,
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


def save_sd_model_on_epoch_end_or_stepwise(
    saving_config: SavingConfig,
    metadata_config: MetadataConfig,
    loss_config: LossConfig,
    v2: bool,
    on_epoch_end: bool,
    accelerator: "Accelerator",
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
    hf_config: HuggingFaceConfig | None = None,
) -> None:
    """
    Saves the Stable Diffusion (v1.5/v2) model at epoch end or stepwise.

    This function integrates epoch and step saving logic as metadata includes both,
    and arguments are largely shared.

    Args:
        saving_config: Configuration for saving outputs.
        metadata_config: Configuration for metadata generation.
        loss_config: Configuration for loss parameters.
        v2: Boolean indicating if the model is V2.
        on_epoch_end: True if saving at epoch end, False if stepwise.
        accelerator: Accelerator instance.
        src_path: Path to the source model.
        save_stable_diffusion_format: Whether to save in SD format.
        use_safetensors: Whether to use SafeTensors format.
        save_dtype: Data type for saving weights.
        epoch: Current epoch number.
        num_train_epochs: Total number of training epochs.
        global_step: Current global step count.
        text_encoder: Text encoder model.
        unet: UNet model.
        vae: VAE model.
        hf_config: Configuration for Hugging Face integration.
    """

    def sd_saver(ckpt_file, epoch_no, global_step):
        modelspec_metadata = model_metadata.get_model_metadata_from_config(
            state_dict=None,
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
