"""SD3 model components."""

from library.models import LoadedModelComponentSpec
from library.models.sd3.conversion import save_models
from library.models.sd3.loader import (
    analyze_state_dict_state,
    load_clip_g,
    load_clip_l,
    load_target_model,
    load_mmdit,
    load_t5xxl,
    load_vae,
)
from library.models.sd3.mmdit import MMDiT, SD3Params, create_sd3_mmdit
from library.models.sd3.vae import SDVAE, VAE_SCALE_FACTOR, VAE_SHIFT_FACTOR

LOADED_MODEL_COMPONENT_SPECS = (
    LoadedModelComponentSpec(key="text_encoder1", public_name="clip_l", roles=("text_encoder",)),
    LoadedModelComponentSpec(key="text_encoder2", public_name="clip_g", roles=("text_encoder",)),
    LoadedModelComponentSpec(key="text_encoder3", public_name="t5xxl", roles=("text_encoder",)),
    LoadedModelComponentSpec(key="vae", public_name="vae", roles=("vae",)),
    LoadedModelComponentSpec(key="denoiser", public_name="mmdit", roles=("denoiser",)),
)


__all__ = [
    "MMDiT",
    "SD3Params",
    "SDVAE",
    "VAE_SCALE_FACTOR",
    "VAE_SHIFT_FACTOR",
    "analyze_state_dict_state",
    "create_sd3_mmdit",
    "save_models",
    "load_clip_g",
    "load_clip_l",
    "load_target_model",
    "load_mmdit",
    "load_t5xxl",
    "load_vae",
    "LOADED_MODEL_COMPONENT_SPECS",
]
