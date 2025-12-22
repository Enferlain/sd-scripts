"""
Regression tests for library/constants.py

These tests verify that critical constants haven't been accidentally modified.
If a constant needs to change intentionally, update both the source and test.
"""

import pytest

from library.constants import (
    # Core
    EPSILON,
    HIGH_VRAM,
    TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
    
    # Image extensions - minimum set (dynamic extensions like AVIF/JXL may vary)
    IMAGE_EXTENSIONS,
    
    # Scheduler
    SCHEDULER_LINEAR_START,
    SCHEDULER_LINEAR_END,
    SCHEDULER_TIMESTEPS,
    SCHEDLER_SCHEDULE,
    
    # Checkpointing
    EPOCH_STATE_NAME,
    EPOCH_FILE_NAME,
    STEP_FILE_NAME,
    DEFAULT_EPOCH_NAME,
    DEFAULT_STEP_NAME,
    DEFAULT_LAST_OUTPUT_NAME,
    
    # Metadata keys
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_MINIMUM_KEYS,
    
    # Model parameters (SD 1.x)
    NUM_TRAIN_TIMESTEPS,
    BETA_START,
    BETA_END,
    UNET_PARAMS_MODEL_CHANNELS,
    UNET_PARAMS_CHANNEL_MULT,
    UNET_PARAMS_ATTENTION_RESOLUTIONS,
    UNET_PARAMS_IMAGE_SIZE,
    UNET_PARAMS_IN_CHANNELS,
    UNET_PARAMS_OUT_CHANNELS,
    UNET_PARAMS_NUM_RES_BLOCKS,
    UNET_PARAMS_CONTEXT_DIM,
    UNET_PARAMS_NUM_HEADS,
    
    # VAE parameters
    VAE_PARAMS_Z_CHANNELS,
    VAE_PARAMS_RESOLUTION,
    VAE_PARAMS_IN_CHANNELS,
    VAE_PARAMS_OUT_CH,
    VAE_PARAMS_CH,
    VAE_PARAMS_CH_MULT,
    VAE_PARAMS_NUM_RES_BLOCKS,
    VAE_PREFIX,
    
    # V2 parameters
    V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
    V2_UNET_PARAMS_CONTEXT_DIM,
    
    # Reference models
    DIFFUSERS_REF_MODEL_ID_V1,
    DIFFUSERS_REF_MODEL_ID_V2,
    
    # Original UNet
    BLOCK_OUT_CHANNELS,
    TIMESTEP_INPUT_DIM,
    TIME_EMBED_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    LAYERS_PER_BLOCK,
    NORM_GROUPS,
    NORM_EPS,
    DOWN_BLOCK_TYPES,
    UP_BLOCK_TYPES,
    
    # SDXL
    SDXL_KEY_PREFIX,
    VAE_SCALE_FACTOR,
    MODEL_VERSION_SDXL_BASE_V1_0,
    DIFFUSERS_REF_MODEL_ID_SDXL,
    DIFFUSERS_SDXL_UNET_CONFIG,
    SDXL_IN_CHANNELS,
    SDXL_OUT_CHANNELS,
    ADM_SDXL_IN_CHANNELS,
    SDXL_CONTEXT_DIM,
    SDXL_MODEL_CHANNELS,
    SDXL_TIME_EMBED_DIM,
    
    # Tokenizers
    TOKENIZER1_PATH,
    TOKENIZER2_PATH,
    TOKENIZER_ID,
    V2_STABLE_DIFFUSION_ID,
)


# =============================================================================
# Core Constants
# =============================================================================

@pytest.mark.unit
class TestCoreConstants:
    """Verify core constants haven't changed."""
    
    def test_epsilon(self):
        assert EPSILON == 1e-6
    
    def test_high_vram_default(self):
        assert HIGH_VRAM is False
    
    def test_text_encoder_cache_suffix(self):
        assert TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX == "_te_outputs.npz"


# =============================================================================
# Image Extensions
# =============================================================================

@pytest.mark.unit
class TestImageExtensions:
    """Verify minimum image extensions are present."""
    
    def test_core_extensions_present(self):
        # These must always be present
        core = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        for ext in core:
            assert ext in IMAGE_EXTENSIONS or ext.upper() in IMAGE_EXTENSIONS


# =============================================================================
# Scheduler Constants
# =============================================================================

@pytest.mark.unit
class TestSchedulerConstants:
    """Verify scheduler parameters match diffusers defaults."""
    
    def test_scheduler_linear_start(self):
        assert SCHEDULER_LINEAR_START == 0.00085
    
    def test_scheduler_linear_end(self):
        assert SCHEDULER_LINEAR_END == 0.0120
    
    def test_scheduler_timesteps(self):
        assert SCHEDULER_TIMESTEPS == 1000
    
    def test_scheduler_schedule(self):
        assert SCHEDLER_SCHEDULE == "scaled_linear"


# =============================================================================
# Checkpointing Format Strings
# =============================================================================

@pytest.mark.unit
class TestCheckpointingConstants:
    """Verify checkpoint naming patterns."""
    
    def test_epoch_format_strings(self):
        assert EPOCH_STATE_NAME == "{}-{:06d}-state"
        assert EPOCH_FILE_NAME == "{}-{:06d}"
        assert STEP_FILE_NAME == "{}-step{:08d}"
    
    def test_default_names(self):
        assert DEFAULT_EPOCH_NAME == "epoch"
        assert DEFAULT_STEP_NAME == "at"
        assert DEFAULT_LAST_OUTPUT_NAME == "last"


# =============================================================================
# Metadata Keys
# =============================================================================

@pytest.mark.unit
class TestMetadataKeys:
    """Verify metadata key constants."""
    
    def test_metadata_key_values(self):
        assert SS_METADATA_KEY_V2 == "ss_v2"
        assert SS_METADATA_KEY_BASE_MODEL_VERSION == "ss_base_model_version"
        assert SS_METADATA_KEY_NETWORK_MODULE == "ss_network_module"
        assert SS_METADATA_KEY_NETWORK_DIM == "ss_network_dim"
        assert SS_METADATA_KEY_NETWORK_ALPHA == "ss_network_alpha"
        assert SS_METADATA_KEY_NETWORK_ARGS == "ss_network_args"
    
    def test_minimum_keys_contains_all(self):
        assert len(SS_METADATA_MINIMUM_KEYS) == 6
        assert SS_METADATA_KEY_V2 in SS_METADATA_MINIMUM_KEYS
        assert SS_METADATA_KEY_NETWORK_MODULE in SS_METADATA_MINIMUM_KEYS


# =============================================================================
# SD 1.x Model Parameters
# =============================================================================

@pytest.mark.unit
class TestSD1ModelParams:
    """Verify SD 1.x architecture constants."""
    
    def test_training_params(self):
        assert NUM_TRAIN_TIMESTEPS == 1000
        assert BETA_START == 0.00085
        assert BETA_END == 0.0120
    
    def test_unet_params(self):
        assert UNET_PARAMS_MODEL_CHANNELS == 320
        assert UNET_PARAMS_CHANNEL_MULT == [1, 2, 4, 4]
        assert UNET_PARAMS_ATTENTION_RESOLUTIONS == [4, 2, 1]
        assert UNET_PARAMS_IMAGE_SIZE == 64
        assert UNET_PARAMS_IN_CHANNELS == 4
        assert UNET_PARAMS_OUT_CHANNELS == 4
        assert UNET_PARAMS_NUM_RES_BLOCKS == 2
        assert UNET_PARAMS_CONTEXT_DIM == 768
        assert UNET_PARAMS_NUM_HEADS == 8
    
    def test_vae_params(self):
        assert VAE_PARAMS_Z_CHANNELS == 4
        assert VAE_PARAMS_RESOLUTION == 256
        assert VAE_PARAMS_IN_CHANNELS == 3
        assert VAE_PARAMS_OUT_CH == 3
        assert VAE_PARAMS_CH == 128
        assert VAE_PARAMS_CH_MULT == [1, 2, 4, 4]
        assert VAE_PARAMS_NUM_RES_BLOCKS == 2
        assert VAE_PREFIX == "first_stage_model."


# =============================================================================
# SD 2.x Parameters
# =============================================================================

@pytest.mark.unit
class TestSD2ModelParams:
    """Verify SD 2.x architecture constants."""
    
    def test_v2_attention_head_dim(self):
        assert V2_UNET_PARAMS_ATTENTION_HEAD_DIM == [5, 10, 20, 20]
    
    def test_v2_context_dim(self):
        assert V2_UNET_PARAMS_CONTEXT_DIM == 1024


# =============================================================================
# Original UNet Constants
# =============================================================================

@pytest.mark.unit
class TestOriginalUNetConstants:
    """Verify original UNet architecture constants."""
    
    def test_block_out_channels(self):
        assert BLOCK_OUT_CHANNELS == (320, 640, 1280, 1280)
        assert TIMESTEP_INPUT_DIM == 320
        assert TIME_EMBED_DIM == 1280
    
    def test_channel_counts(self):
        assert IN_CHANNELS == 4
        assert OUT_CHANNELS == 4
        assert LAYERS_PER_BLOCK == 2
    
    def test_normalization(self):
        assert NORM_GROUPS == 32
        assert NORM_EPS == 1e-5
    
    def test_block_types(self):
        assert len(DOWN_BLOCK_TYPES) == 4
        assert len(UP_BLOCK_TYPES) == 4
        assert DOWN_BLOCK_TYPES[-1] == "DownBlock2D"
        assert UP_BLOCK_TYPES[0] == "UpBlock2D"


# =============================================================================
# SDXL Constants
# =============================================================================

@pytest.mark.unit
class TestSDXLConstants:
    """Verify SDXL architecture constants."""
    
    def test_sdxl_key_prefix(self):
        assert SDXL_KEY_PREFIX == "conditioner.embedders.1.model."
    
    def test_vae_scale_factor(self):
        # Critical: this value affects latent space scaling
        assert VAE_SCALE_FACTOR == 0.13025
    
    def test_model_version(self):
        assert MODEL_VERSION_SDXL_BASE_V1_0 == "sdxl_base_v1-0"
    
    def test_sdxl_channels(self):
        assert SDXL_IN_CHANNELS == 4
        assert SDXL_OUT_CHANNELS == 4
        assert ADM_SDXL_IN_CHANNELS == 2816
        assert SDXL_CONTEXT_DIM == 2048
        assert SDXL_MODEL_CHANNELS == 320
        assert SDXL_TIME_EMBED_DIM == 1280
    
    def test_sdxl_unet_config_structure(self):
        # Verify key config values
        assert DIFFUSERS_SDXL_UNET_CONFIG["in_channels"] == 4
        assert DIFFUSERS_SDXL_UNET_CONFIG["out_channels"] == 4
        assert DIFFUSERS_SDXL_UNET_CONFIG["cross_attention_dim"] == 2048
        assert DIFFUSERS_SDXL_UNET_CONFIG["sample_size"] == 128
        assert DIFFUSERS_SDXL_UNET_CONFIG["block_out_channels"] == [320, 640, 1280]
        assert len(DIFFUSERS_SDXL_UNET_CONFIG) > 30  # Has many keys


# =============================================================================
# Reference Model IDs
# =============================================================================

@pytest.mark.unit
class TestReferenceModelIDs:
    """Verify HuggingFace model IDs are correct."""
    
    def test_diffusers_ref_models(self):
        assert DIFFUSERS_REF_MODEL_ID_V1 == "runwayml/stable-diffusion-v1-5"
        assert DIFFUSERS_REF_MODEL_ID_V2 == "stabilityai/stable-diffusion-2-1"
        assert DIFFUSERS_REF_MODEL_ID_SDXL == "stabilityai/stable-diffusion-xl-base-1.0"
    
    def test_tokenizer_paths(self):
        assert TOKENIZER1_PATH == "openai/clip-vit-large-patch14"
        assert TOKENIZER2_PATH == "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        assert TOKENIZER_ID == "openai/clip-vit-large-patch14"
        assert V2_STABLE_DIFFUSION_ID == "stabilityai/stable-diffusion-2"
