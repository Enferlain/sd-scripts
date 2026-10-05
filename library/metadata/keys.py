"""Shared metadata namespace constants."""

DEFAULT_METADATA_NAMESPACE = "kuro"

KURO_PREFIX = "kuro."
KURO_SCHEMA_VERSION_KEY = "kuro.schema_version"
KURO_SCHEMA_VERSION = "1"

KOHYA_SS_PREFIX = "ss_"
MODELSPEC_PREFIX = "modelspec."
MODELSPEC_VERSION_KEY = "modelspec.sai_model_spec"
MODELSPEC_VERSION = "1.0.1"

# Canonical artifact facts rendered by the ModelSpec projection, not extensions.
MODELSPEC_FACT_KEYS = (
    "architecture",
    "implementation",
    "title",
    "resolution",
    "description",
    "author",
    "date",
    "hash_sha256",
    "implementation_version",
    "license",
    "usage_hint",
    "thumbnail",
    "tags",
    "merged_from",
    "trigger_phrase",
    "prediction_type",
    "timestep_range",
    "encoder_layer",
    "preprocessor",
    "is_negative_embedding",
    "unet_dtype",
    "vae_dtype",
)

SS_METADATA_KEY_V2 = "ss_v2"
SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_ADAPTER_MODULE = "ss_adapter_module"
SS_METADATA_KEY_ADAPTER_RANK = "ss_adapter_rank"
SS_METADATA_KEY_ADAPTER_ALPHA = "ss_adapter_alpha"
SS_METADATA_KEY_ADAPTER_ARGS = "ss_adapter_args"
