# based on https://github.com/Stability-AI/ModelSpec
import os
import time
import datetime
import base64
import logging
import mimetypes
import subprocess
import safetensors

from dataclasses import dataclass, field, asdict, is_dataclass

from library.utils.common_utils import setup_logging
from library.config.dataclasses.output import MetadataConfig
# Type hints only to avoid circular imports if possible, though these are dataclasses so distinct modules usually fine

setup_logging()
logger = logging.getLogger(__name__)


r"""
# Metadata Example
metadata = {
    # === Must ===
    "modelspec.sai_model_spec": "1.0.0", # Required version ID for the spec
    "modelspec.architecture": "stable-diffusion-xl-v1-base", # Architecture, reference the ID of the original model of the arch to match the ID
    "modelspec.implementation": "sgm",
    "modelspec.title": "Example Model Version 1.0", # Clean, human-readable title. May use your own phrasing/language/etc
    # === Should ===
    "modelspec.author": "Example Corp", # Your name or company name
    "modelspec.description": "This is my example model to show you how to do it!", # Describe the model in your own words/language/etc. Focus on what users need to know
    "modelspec.date": "2023-07-20", # ISO-8601 compliant date of when the model was created
    # === Can ===
    "modelspec.license": "ExampleLicense-1.0", # eg CreativeML Open RAIL, etc.
    "modelspec.usage_hint": "Use keyword 'example'" # In your own language, very short hints about how the user should use the model
}
"""

BASE_METADATA = {
    # === MUST ===
    "modelspec.sai_model_spec": "1.0.1",
    "modelspec.architecture": None,
    "modelspec.implementation": None,
    "modelspec.title": None,
    "modelspec.resolution": None,
    # === SHOULD ===
    "modelspec.description": None,
    "modelspec.author": None,
    "modelspec.date": None,
    "modelspec.hash_sha256": None,
    # === CAN===
    "modelspec.implementation_version": None,
    "modelspec.license": None,
    "modelspec.usage_hint": None,
    "modelspec.thumbnail": None,
    "modelspec.tags": None,
    "modelspec.merged_from": None,
    "modelspec.trigger_phrase": None,
    "modelspec.prediction_type": None,
    "modelspec.timestep_range": None,
    "modelspec.encoder_layer": None,
    "modelspec.preprocessor": None,
    "modelspec.is_negative_embedding": None,
    "modelspec.unet_dtype": None,
    "modelspec.vae_dtype": None,
}

# 別に使うやつだけ定義
MODELSPEC_TITLE = "modelspec.title"

ARCH_SD_V1 = "stable-diffusion-v1"
ARCH_SD_V2_512 = "stable-diffusion-v2-512"
ARCH_SD_V2_768_V = "stable-diffusion-v2-768-v"
ARCH_SD_XL_V1_BASE = "stable-diffusion-xl-v1-base"
ARCH_SD3_M = "stable-diffusion-3"  # may be followed by "-m" or "-5-large" etc.
# ARCH_SD3_UNKNOWN = "stable-diffusion-3"
ARCH_FLUX_1_DEV = "flux-1-dev"
ARCH_FLUX_1_SCHNELL = "flux-1-schnell"
ARCH_FLUX_1_CHROMA = "chroma"  # for Flux Chroma
ARCH_FLUX_1_UNKNOWN = "flux-1"
ARCH_LUMINA_2 = "lumina-2"
ARCH_LUMINA_UNKNOWN = "lumina"
ARCH_HUNYUAN_IMAGE_2_1 = "hunyuan-image-2.1"
ARCH_HUNYUAN_IMAGE_UNKNOWN = "hunyuan-image"

ADAPTER_LORA = "lora"
ADAPTER_TEXTUAL_INVERSION = "textual-inversion"

IMPL_STABILITY_AI = "https://github.com/Stability-AI/generative-models"
IMPL_COMFY_UI = "https://github.com/comfyanonymous/ComfyUI"
IMPL_DIFFUSERS = "diffusers"
IMPL_FLUX = "https://github.com/black-forest-labs/flux"
IMPL_CHROMA = "https://huggingface.co/lodestones/Chroma"
IMPL_LUMINA = "https://github.com/Alpha-VLLM/Lumina-Image-2.0"
IMPL_HUNYUAN_IMAGE = "https://github.com/Tencent-Hunyuan/HunyuanImage-2.1"

PRED_TYPE_EPSILON = "epsilon"
PRED_TYPE_V = "v"


@dataclass
class ModelSpecMetadata:
    """
    ModelSpec 1.0.1 compliant metadata for safetensors models.
    All fields correspond to modelspec.* keys in the final metadata.
    """

    # === MUST ===
    architecture: str
    implementation: str
    title: str
    resolution: str
    sai_model_spec: str = "1.0.1"

    # === SHOULD ===
    description: str | None = None
    author: str | None = None
    date: str | None = None
    hash_sha256: str | None = None

    # === CAN ===
    implementation_version: str | None = None
    license: str | None = None
    usage_hint: str | None = None
    thumbnail: str | None = None
    tags: str | None = None
    merged_from: str | None = None
    trigger_phrase: str | None = None
    prediction_type: str | None = None
    timestep_range: str | None = None
    encoder_layer: str | None = None
    preprocessor: str | None = None
    is_negative_embedding: str | None = None
    unet_dtype: str | None = None
    vae_dtype: str | None = None

    # === Additional metadata ===
    additional_fields: dict[str, str] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, str]:
        """Convert dataclass to metadata dictionary with modelspec. prefixes."""
        metadata = {}

        # Add all non-None fields with modelspec prefix
        for field_name, value in self.__dict__.items():
            if field_name == "additional_fields":
                # Handle additional fields separately
                for key, val in value.items():
                    if key.startswith("modelspec."):
                        metadata[key] = val
                    else:
                        metadata[f"modelspec.{key}"] = val
            elif value is not None:
                metadata[f"modelspec.{field_name}"] = value

        return metadata

    @classmethod
    def from_config(cls, metadata_config: MetadataConfig, timestamp: float | None = None, **kwargs) -> "ModelSpecMetadata":
        """
        Create ModelSpecMetadata from MetadataConfig.

        Args:
            metadata_config (MetadataConfig): The metadata configuration.
            timestamp (float, optional): The timestamp. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelSpecMetadata: The created metadata instance.
        """
        if timestamp is None:
            timestamp = time.time()

        # Extract standard fields from the config
        # We look for fields in MetadataConfig that match "metadata_{name}"
        metadata_fields = {}
        # Handle both real dataclasses and OmegaConf DictConfig
        if is_dataclass(metadata_config) and not isinstance(metadata_config, type):
            config_fields = asdict(metadata_config).keys()
        else:
            # OmegaConf DictConfig or dict-like
            config_fields = metadata_config.keys() if hasattr(metadata_config, "keys") else dir(metadata_config)
        for config_field in config_fields:
            if config_field.startswith("metadata_"):
                value = getattr(metadata_config, config_field)
                if value is not None:
                    # Remove metadata_ prefix
                    field_name = config_field[9:]  # len("metadata_") = 9
                    metadata_fields[field_name] = value

        # Handle known standard fields
        standard_fields = {
            "title": metadata_fields.pop("title", None),
            "author": metadata_fields.pop("author", None),
            "description": metadata_fields.pop("description", None),
            "license": metadata_fields.pop("license", None),
            "tags": metadata_fields.pop("tags", None),
        }

        # Remove None values
        standard_fields = {k: v for k, v in standard_fields.items() if v is not None}

        # Merge with kwargs and remaining metadata fields
        all_fields = {**standard_fields, **kwargs}
        if metadata_fields:
            all_fields["additional_fields"] = metadata_fields

        if "date" not in all_fields:
            # remove microsecond from time
            int_ts = int(timestamp)
            # time to iso-8601 compliant date
            all_fields["date"] = datetime.datetime.fromtimestamp(int_ts).isoformat()

        # Ensure we have the required fields or let the constructor/post-init handle defaults?
        # The constructor expects architecture etc, which should be passed in kwargs.

        return cls(**all_fields)


def determine_architecture(
    v2: bool, v_parameterization: bool, sdxl: bool, lora: bool, textual_inversion: bool, model_config: dict[str, str] | None = None
) -> str:
    """
    Determine model architecture string from parameters.

    Args:
        v2 (bool): Whether the model is SD v2.
        v_parameterization (bool): Whether the model uses v-parameterization.
        sdxl (bool): Whether the model is SDXL.
        lora (bool): Whether the model is a LoRA.
        textual_inversion (bool): Whether the model is a Textual Inversion.
        model_config (dict[str, str], optional): Configuration for specific models (Flux, SD3, etc.).

    Returns:
        str: The model architecture string.
    """

    model_config = model_config or {}

    if sdxl:
        arch = ARCH_SD_XL_V1_BASE
    elif "sd3" in model_config:
        arch = ARCH_SD3_M + "-" + model_config["sd3"]
    elif "flux" in model_config:
        flux_type = model_config["flux"]
        if flux_type == "dev":
            arch = ARCH_FLUX_1_DEV
        elif flux_type == "schnell":
            arch = ARCH_FLUX_1_SCHNELL
        elif flux_type == "chroma":
            arch = ARCH_FLUX_1_CHROMA
        else:
            arch = ARCH_FLUX_1_UNKNOWN
    elif "lumina" in model_config:
        lumina_type = model_config["lumina"]
        if lumina_type == "lumina2":
            arch = ARCH_LUMINA_2
        else:
            arch = ARCH_LUMINA_UNKNOWN
    elif "hunyuan_image" in model_config:
        hunyuan_image_type = model_config["hunyuan_image"]
        if hunyuan_image_type == "2.1":
            arch = ARCH_HUNYUAN_IMAGE_2_1
        else:
            arch = ARCH_HUNYUAN_IMAGE_UNKNOWN
    elif v2:
        arch = ARCH_SD_V2_768_V if v_parameterization else ARCH_SD_V2_512
    else:
        arch = ARCH_SD_V1

    # Add adapter suffix
    if lora:
        arch += f"/{ADAPTER_LORA}"
    elif textual_inversion:
        arch += f"/{ADAPTER_TEXTUAL_INVERSION}"

    return arch


def determine_implementation(
    lora: bool,
    textual_inversion: bool,
    sdxl: bool,
    model_config: dict[str, str] | None = None,
    is_stable_diffusion_ckpt: bool | None = None,
) -> str:
    """
    Determine implementation string from parameters.

    Args:
        lora (bool): Whether the model is a LoRA.
        textual_inversion (bool): Whether the model is a Textual Inversion.
        sdxl (bool): Whether the model is SDXL.
        model_config (dict[str, str], optional): Configuration for specific models (Flux, SD3, etc.).
        is_stable_diffusion_ckpt (bool, optional): Whether the model is a Stable Diffusion checkpoint.

    Returns:
        str: The implementation string.
    """

    model_config = model_config or {}

    if "flux" in model_config:
        if model_config["flux"] == "chroma":
            return IMPL_CHROMA
        else:
            return IMPL_FLUX
    elif "lumina" in model_config:
        return IMPL_LUMINA
    elif (lora and sdxl) or textual_inversion or is_stable_diffusion_ckpt:
        return IMPL_STABILITY_AI
    else:
        return IMPL_DIFFUSERS


def get_implementation_version() -> str:
    """
    Get the current implementation version as sd-scripts/{commit_hash}.

    Returns:
        str: The implementation version string.
    """
    try:
        # Get the git commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),  # Go up to sd-scripts root
            timeout=5,
        )

        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            return f"sd-scripts/{commit_hash}"
        else:
            logger.warning("Failed to get git commit hash, using fallback")
            return "sd-scripts/unknown"

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"Could not determine git commit: {e}")
        return "sd-scripts/unknown"


def file_to_data_url(file_path: str) -> str:
    """
    Convert a file path to a data URL for embedding in metadata.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The data URL.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # Default to binary if we can't detect
        mime_type = "application/octet-stream"

    # Read file and encode as base64
    with open(file_path, "rb") as f:
        file_data = f.read()

    encoded_data = base64.b64encode(file_data).decode("ascii")

    return f"data:{mime_type};base64,{encoded_data}"


def determine_resolution(
    reso: int | tuple[int, int] | None = None,
    sdxl: bool = False,
    model_config: dict[str, str] | None = None,
    v2: bool = False,
    v_parameterization: bool = False,
) -> str:
    """
    Determine resolution string from parameters.

    Args:
        reso (Union[int, tuple[int, int]], optional): The resolution.
        sdxl (bool, optional): Whether the model is SDXL. Defaults to False.
        model_config (dict[str, str], optional): Configuration for specific models. Defaults to None.
        v2 (bool, optional): Whether the model is SD v2. Defaults to False.
        v_parameterization (bool, optional): Whether the model uses v-parameterization. Defaults to False.

    Returns:
        str: The resolution string (e.g., "1024x1024").
    """

    model_config = model_config or {}

    if reso is not None:
        # Handle comma separated string
        if isinstance(reso, str):
            parts = [int(x) for x in reso.split(",")]
            reso_tuple: tuple[int, int] = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], parts[0])
        # Handle single int
        elif isinstance(reso, int):
            reso_tuple = (reso, reso)
        # Handle tuple - ensure we have exactly 2 elements
        else:
            reso_tuple = reso if len(reso) >= 2 else (reso[0], reso[0])
    else:
        # Determine default resolution based on model type
        if sdxl or "sd3" in model_config or "flux" in model_config or "lumina" in model_config:
            reso_tuple = (1024, 1024)
        elif v2 and v_parameterization:
            reso_tuple = (768, 768)
        else:
            reso_tuple = (512, 512)

    return f"{reso_tuple[0]}x{reso_tuple[1]}"


# removed load_bytes_in_safetensorsw, precalculate_safetensors_hashes and update_hash_sha256 as these were unused


def build_metadata_dataclass(
    state_dict: dict | None,
    v2: bool,
    v_parameterization: bool,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    timestamp: float,
    title: str | None = None,
    reso: int | tuple[int, int] | None = None,
    is_stable_diffusion_ckpt: bool | None = None,
    author: str | None = None,
    description: str | None = None,
    license: str | None = None,
    tags: str | None = None,
    merged_from: str | None = None,
    timesteps: tuple[int, int] | None = None,
    clip_skip: int | None = None,
    model_config: dict | None = None,
    optional_metadata: dict | None = None,
) -> ModelSpecMetadata:
    """
    Build ModelSpec 1.0.1 compliant metadata dataclass.

    Args:
        state_dict (dict, optional): The model state dictionary.
        v2 (bool): Whether the model is SD v2.
        v_parameterization (bool): Whether the model uses v-parameterization.
        sdxl (bool): Whether the model is SDXL.
        lora (bool): Whether the model is a LoRA.
        textual_inversion (bool): Whether the model is a Textual Inversion.
        timestamp (float): The timestamp of model creation.
        title (str, optional): The model title.
        reso (Union[int, tuple[int, int]], optional): The resolution.
        is_stable_diffusion_ckpt (bool, optional): Whether the model is a Stable Diffusion checkpoint.
        author (str, optional): The model author.
        description (str, optional): The model description.
        license (str, optional): The model license.
        tags (str, optional): The model tags.
        merged_from (str, optional): The model merge source.
        timesteps (tuple[int, int], optional): The timesteps range.
        clip_skip (int, optional): The clip skip value.
        model_config (dict, optional): Dict containing model type info, e.g. {"flux": "dev"}, {"sd3": "large"}.
        optional_metadata (dict, optional): Dict of additional metadata fields to include.

    Returns:
        ModelSpecMetadata: The populated metadata dataclass.
    """

    # Use helper functions for complex logic
    architecture = determine_architecture(v2, v_parameterization, sdxl, lora, textual_inversion, model_config)

    if not lora and not textual_inversion and is_stable_diffusion_ckpt is None:
        is_stable_diffusion_ckpt = True  # default is stable diffusion ckpt if not lora and not textual_inversion

    implementation = determine_implementation(lora, textual_inversion, sdxl, model_config, is_stable_diffusion_ckpt)

    if title is None:
        if lora:
            title = "LoRA"
        elif textual_inversion:
            title = "TextualInversion"
        else:
            title = "Checkpoint"
        title += f"@{timestamp}"

    # remove microsecond from time
    int_ts = int(timestamp)
    # time to iso-8601 compliant date
    date = datetime.datetime.fromtimestamp(int_ts).isoformat()

    # Use helper function for resolution
    resolution = determine_resolution(reso, sdxl, model_config, v2, v_parameterization)

    # Handle prediction type - Flux models don't use prediction_type
    model_config = model_config or {}
    prediction_type = None
    if "flux" not in model_config:
        if v_parameterization:
            prediction_type = PRED_TYPE_V
        else:
            prediction_type = PRED_TYPE_EPSILON

    # Handle timesteps
    timestep_range = None
    if timesteps is not None:
        if isinstance(timesteps, (str, int)):
            timesteps = (timesteps, timesteps)
        if len(timesteps) == 1:
            timesteps = (timesteps[0], timesteps[0])
        timestep_range = f"{timesteps[0]},{timesteps[1]}"

    # Handle encoder layer (clip skip)
    encoder_layer = None
    if clip_skip is not None:
        encoder_layer = f"{clip_skip}"

    # Hash calculation is available in library.utils.hash_utils.precalculate_safetensors_hashes()
    # but not used here because hash is calculated during file saving, not metadata building

    # Process thumbnail - convert file path to data URL if needed
    processed_optional_metadata = optional_metadata.copy() if optional_metadata else {}
    if "thumbnail" in processed_optional_metadata:
        thumbnail_value = processed_optional_metadata["thumbnail"]
        # Check if it's already a data URL or if it's a file path
        if thumbnail_value and not thumbnail_value.startswith("data:"):
            try:
                processed_optional_metadata["thumbnail"] = file_to_data_url(thumbnail_value)
                logger.info(f"Converted thumbnail file {thumbnail_value} to data URL")
            except FileNotFoundError as e:
                logger.warning(f"Thumbnail file not found, skipping: {e}")
                del processed_optional_metadata["thumbnail"]
            except Exception as e:
                logger.warning(f"Failed to convert thumbnail to data URL: {e}")
                del processed_optional_metadata["thumbnail"]

    # Automatically set implementation version if not provided
    if "implementation_version" not in processed_optional_metadata:
        processed_optional_metadata["implementation_version"] = get_implementation_version()

    # Create the dataclass
    metadata = ModelSpecMetadata(
        architecture=architecture,
        implementation=implementation,
        title=title,
        description=description,
        author=author,
        date=date,
        license=license,
        tags=tags,
        merged_from=merged_from,
        resolution=resolution,
        prediction_type=prediction_type,
        timestep_range=timestep_range,
        encoder_layer=encoder_layer,
        additional_fields=processed_optional_metadata,
    )

    return metadata


def build_metadata(
    state_dict: dict | None,
    v2: bool,
    v_parameterization: bool,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    timestamp: float,
    title: str | None = None,
    reso: int | tuple[int, int] | None = None,
    is_stable_diffusion_ckpt: bool | None = None,
    author: str | None = None,
    description: str | None = None,
    license: str | None = None,
    tags: str | None = None,
    merged_from: str | None = None,
    timesteps: tuple[int, int] | None = None,
    clip_skip: int | None = None,
    model_config: dict | None = None,
    optional_metadata: dict | None = None,
) -> dict[str, str]:
    """
    Build ModelSpec 1.0.1 compliant metadata for safetensors models.
    Legacy function that returns dict - prefer build_metadata_dataclass for new code.

    Args:
        state_dict (dict, optional): The model state dictionary.
        v2 (bool): Whether the model is SD v2.
        v_parameterization (bool): Whether the model uses v-parameterization.
        sdxl (bool): Whether the model is SDXL.
        lora (bool): Whether the model is a LoRA.
        textual_inversion (bool): Whether the model is a Textual Inversion.
        timestamp (float): The timestamp of model creation.
        title (str, optional): The model title.
        reso (Union[int, tuple[int, int]], optional): The resolution.
        is_stable_diffusion_ckpt (bool, optional): Whether the model is a Stable Diffusion checkpoint.
        author (str, optional): The model author.
        description (str, optional): The model description.
        license (str, optional): The model license.
        tags (str, optional): The model tags.
        merged_from (str, optional): The model merge source.
        timesteps (tuple[int, int], optional): The timesteps range.
        clip_skip (int, optional): The clip skip value.
        model_config (dict, optional): Dict containing model type info, e.g. {"flux": "dev"}, {"sd3": "large"}.
        optional_metadata (dict, optional): Dict of additional metadata fields to include.

    Returns:
        dict[str, str]: The metadata dictionary.
    """
    # Use the dataclass function and convert to dict
    metadata_obj = build_metadata_dataclass(
        state_dict=state_dict,
        v2=v2,
        v_parameterization=v_parameterization,
        sdxl=sdxl,
        lora=lora,
        textual_inversion=textual_inversion,
        timestamp=timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=author,
        description=description,
        license=license,
        tags=tags,
        merged_from=merged_from,
        timesteps=timesteps,
        clip_skip=clip_skip,
        model_config=model_config,
        optional_metadata=optional_metadata,
    )

    return metadata_obj.to_metadata_dict()


def get_title(metadata: dict) -> str | None:
    """
    Get the title from the metadata dictionary.

    Args:
        metadata (dict): The metadata dictionary.

    Returns:
        str | None: The title, or None if not found.
    """
    return metadata.get(MODELSPEC_TITLE)


def load_metadata_from_safetensors(model: str) -> dict:
    """
    Load metadata from a safetensors file.

    Args:
        model (str): The path to the safetensors file.

    Returns:
        dict: The metadata dictionary.
    """
    if not model.endswith(".safetensors"):
        return {}

    with safetensors.safe_open(model, framework="pt") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


def build_merged_from(models: list[str]) -> str:
    """
    Build a comma-separated string of model titles from a list of model paths.

    Args:
        models (list[str]): A list of paths to model files.

    Returns:
        str: A comma-separated string of model titles.
    """

    def get_title(model: str):
        metadata = load_metadata_from_safetensors(model)
        title = metadata.get(MODELSPEC_TITLE, None)
        if title is None:
            title = os.path.splitext(os.path.basename(model))[0]  # use filename
        return title

    titles = [get_title(model) for model in models]
    return ", ".join(titles)


def get_model_metadata_from_config(
    state_dict: dict | None,
    metadata_config: MetadataConfig,
    is_sdxl: bool,
    is_v2: bool,
    v_parameterization: bool,
    is_lora: bool,
    is_textual_inversion: bool,
    resolution: int | tuple[int, int] = (512, 512),
    min_timestep: int | None = None,
    max_timestep: int | None = None,
    clip_skip: int | None = None,
    is_stable_diffusion_ckpt: bool | None = None,
    flux_type: str | None = None,
    lumina_type: str | None = None,
    hunyuan_image_type: str | None = None,
    optional_metadata: dict[str, str] | None = None,
) -> dict:
    """
    Get SAI Model Spec using configuration objects directly.

    Args:
        state_dict (dict): The model state dictionary.
        metadata_config (MetadataConfig): The metadata configuration.
        is_sdxl (bool): Whether the model is SDXL.
        is_v2 (bool): Whether the model is SD v2.
        v_parameterization (bool): Whether the model uses v-parameterization.
        is_lora (bool): Whether the model is a LoRA.
        is_textual_inversion (bool): Whether the model is a Textual Inversion.
        resolution (Union[int, tuple[int, int]], optional): The resolution. Defaults to (512, 512).
        min_timestep (int, optional): The minimum timesteps. Defaults to None.
        max_timestep (int, optional): The maximum timesteps. Defaults to None.
        clip_skip (int, optional): The clip skip value. Defaults to None.
        is_stable_diffusion_ckpt (bool, optional): Whether the model is a Stable Diffusion checkpoint. Defaults to None.
        flux_type (str, optional): The Flux model type. Defaults to None.
        lumina_type (str, optional): The Lumina model type. Defaults to None.
        hunyuan_image_type (str, optional): The Hunyuan Image model type. Defaults to None.
        optional_metadata (dict, optional): Additional metadata. Defaults to None.

    Returns:
        dict: The metadata dictionary.
    """
    timestamp = time.time()

    title = metadata_config.metadata_title

    # Timesteps logic
    timesteps = None
    if min_timestep is not None or max_timestep is not None:
        min_ts = min_timestep if min_timestep is not None else 0
        max_ts = max_timestep if max_timestep is not None else 1000
        timesteps = (min_ts, max_ts)

    # Model Config Dict
    model_config_dict = {}
    if flux_type is not None:
        model_config_dict["flux"] = flux_type
    if lumina_type is not None:
        model_config_dict["lumina"] = lumina_type
    if hunyuan_image_type is not None:
        model_config_dict["hunyuan_image"] = hunyuan_image_type

    # determine_architecture etc are done inside build_metadata_dataclass

    if not is_lora and not is_textual_inversion and is_stable_diffusion_ckpt is None:
        is_stable_diffusion_ckpt = True

    if title is None:
        if is_lora:
            title = "LoRA"
        elif is_textual_inversion:
            title = "TextualInversion"
        else:
            title = "Checkpoint"
        title += f"@{timestamp}"

    # Helper to merge optional metadata and extract from config
    extracted_metadata = {}
    # Handle both real dataclasses and OmegaConf DictConfig
    if is_dataclass(metadata_config) and not isinstance(metadata_config, type):
        config_fields = asdict(metadata_config).keys()
    else:
        # OmegaConf DictConfig or dict-like
        config_fields = metadata_config.keys() if hasattr(metadata_config, "keys") else dir(metadata_config)
    for config_field in config_fields:
        if config_field.startswith("metadata_"):
            value = getattr(metadata_config, config_field)
            if value is not None:
                field_name = config_field[9:]
                if field_name not in ["title", "author", "description", "license", "tags"]:
                    extracted_metadata[field_name] = value

    all_optional_metadata = {**extracted_metadata}
    if optional_metadata:
        all_optional_metadata.update(optional_metadata)

    # Using build_metadata_dataclass which we have locally
    metadata_obj = build_metadata_dataclass(
        state_dict,
        is_v2,
        v_parameterization,
        is_sdxl,
        is_lora,
        is_textual_inversion,
        timestamp,
        title=title,
        reso=resolution,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=metadata_config.metadata_author,
        description=metadata_config.metadata_description,
        license=metadata_config.metadata_license,
        tags=metadata_config.metadata_tags,
        timesteps=timesteps,
        clip_skip=clip_skip,
        model_config=model_config_dict,
        optional_metadata=all_optional_metadata,
    )

    return metadata_obj.to_metadata_dict()
