"""SD model components."""

from library.models.parameter_dump import NamedParameterComponentNames

NAMED_PARAMETER_COMPONENT_NAMES = NamedParameterComponentNames(
    text_encoder_names=("clip_l",),
    vae_name="vae",
    denoiser_name="unet",
)

__all__ = ["NAMED_PARAMETER_COMPONENT_NAMES"]
