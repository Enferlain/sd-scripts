"""SD model components."""

from library.models import LoadedModelComponentSpec

LOADED_MODEL_COMPONENT_SPECS = (
    LoadedModelComponentSpec(key="text_encoder1", public_name="clip_l", roles=("text_encoder",)),
    LoadedModelComponentSpec(key="vae", public_name="vae", roles=("vae",)),
    LoadedModelComponentSpec(key="denoiser", public_name="unet", roles=("denoiser",)),
)

__all__ = ["LOADED_MODEL_COMPONENT_SPECS"]
