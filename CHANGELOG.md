# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2025-12-23]

### Added

- **Test Suite Expansion (897 total tests)**
  - `test_optimizations_deepspeed.py` - Unit tests for DeepSpeed config/plugin preparation (17 tests)
  - `test_optimizations_offloading.py` - Unit tests for CPU offloading utilities (42 tests)
  - `test_training_sdxl_checkpointing.py` - Unit tests for SDXL checkpointing wrappers (16 tests)
  - `test_models_model_util.py` - Added tests for `renew_*_paths`, `get_model_version_str`, `conv_attn_to_linear`, `controlnet_conversion_map`, `reshape_weight_for_sd`, `linear_transformer_to_conv` (+21 tests, 49 total)
  - `test_models_sdxl_model_util.py` - Added tests for `convert_unet_state_dict`, bidirectional SDXL↔Diffusers conversion with roundtrip verification (+8 tests, 27 total)
  - `test_data_structures.py` - Added `BucketManager.make_buckets` test with mocked `model_util`
  - `test_dataset_bucketing.py` - Added unit tests for `dataset.py` bucketing logic (6 tests)
  - `test_training_sdxl_model_prep.py` - Added unit tests for `sdxl_model_prep.py` (10 tests)
  - `test_models_conversion.py` - Added unit tests for model conversion functions (15 tests: VAE attention/resnet paths, checkpoint assignment, VAE state dict conversion)
  - `test_fp8_optimization.py` - Added tests for `apply_fp8_monkey_patch` and `fp8_linear_forward_patch` (7 tests)
  - `test_optimizations_offloading.py` - Added tests for `Offloader` and `ModelOffloader` classes (11 tests)
  - Audited ROADMAP "Heavy Mocking" section with accurate ✅/🔶/❌ status for each module

## [2025-12-22]

### Added

- **Centralized Config Validation Module** (`library/config/validation.py`)
  - `prepare_config(cfg)` - Auto-fixups: cache flags, optimizer shortcuts, backward compat
  - `validate_config(cfg)` - Cross-config errors: `adaptive_noise_scale`, `v_pred` conflicts, `full_fp16/bf16`, `fp8_base`, SDXL `block_lr`
  - Script-specific validators: `validate_sd_peft`, `validate_sdxl_peft`, `validate_sd_textual_inversion`, `validate_sdxl_textual_inversion`
  - Comprehensive unit tests in `tests/unit/test_validation.py` (24 tests)
  - `tests/unit/test_losses_loss_weighting.py` - Tests for SNR weighting formulas, v-prediction modes, and masking.
  - `tests/unit/test_utils_device.py` - Tests for device memory cleanup, synchronization, and device selection with strict hardware mocking.
  - `tests/unit/test_losses_edm2_loss.py` - Tests for EDM2 adaptive loss weighting components and configuration logic.
  - `tests/unit/test_utils_jpeg_xl.py` - Tests for JXL bitstream parsing, container structure using synthetic binary data, refactoring `JXLBitstream` for correctness.
  - `tests/unit/test_networks_lora_utils.py` - Tests for LoRA state dict filtering and weight merging (Linear & Conv2d) with mocked loading hooks.
  - `tests/unit/test_sai_model_spec.py` - Tests for ModelSpec metadata generation, config extraction, and resolution logic.
  - `tests/unit/test_utils_torch.py` - Added tests for `set_seed_from_config` mocking underlying seed setters.
  - `tests/unit/test_training_sample_generation.py` - Tests for parsing logic, scheduler factories, and sampling trigger checks in `sample_generation.py`.
  - `tests/unit/test_models_text_encoder_util.py` - Tests for CLIP pooling workaround (finding EOS tokens) and SDXL hidden state chunking/reshaping.
  - `tests/unit/test_config_util.py` - Tests for `BlueprintGenerator`, dataset type detection (DreamBooth/FineTune/ControlNet), validation split logic, and subdirectory parsing.
  - `tests/unit/test_constants.py` - Regression tests for critical constants: scheduler params, UNet/VAE architecture values, SDXL configs, and HuggingFace model IDs.
  - `tests/unit/test_sdxl_model_util.py` - Tests for `timestep_embedding`, `get_size_embeddings`, and `make_unet_conversion_map` (pure math functions).
  - `tests/unit/test_model_util.py` - Tests for `shave_segments`, `is_safetensors`, `create_unet_diffusers_config`, and `create_vae_diffusers_config`.
  - `tests/unit/test_caching.py` - Tests for latent cache validation (`is_disk_cached_latents_is_expected`) and text encoder output file I/O.
  - `tests/unit/test_original_unet.py` - Tests for `get_timestep_embedding`, `resize_like`, and `get_parameter_dtype/device` utilities from original_unet.py.
  - `tests/unit/test_fp8_optimization.py` - Tests for FP8 quantization: `calculate_fp8_maxval`, `quantize_fp8`, and `quantize_weight` with block/channel/tensor modes.
  - `tests/unit/test_networks_lora.py` - Tests for LoRA block LR parsing, dims/alphas calculation, block index resolution, and weight removal utilities.
  - `tests/unit/test_optimizers_adafactor_fused.py` - Tests for stochastic rounding (`copy_stochastic_`) and optimizer patching.
  - `tests/unit/test_networks_lora_diffusers.py` - Tests for UNet conversion map between Stability AI and Diffusers naming formats.
  - `tests/unit/test_pipelines_lpw.py` - Tests for prompt attention weight parsing and token/weight padding utilities.
  - `tests/unit/test_utils_huggingface.py` - Tests for HuggingFace Hub API (`exists_repo`, `list_dir`) with mocked network calls.
  - `tests/unit/test_strategies_base.py` - Tests for strategy base classes: singleton patterns, tokenizer loading, NPZ save/load, weighted input parsing, and cache validation (39 tests).
  - `tests/unit/test_strategies_sd.py` - Tests for SD 1.5/2.0 strategies: tokenization (v1/v2), text encoding with clip_skip, latent caching with VAE mocking (17 tests).
  - `tests/unit/test_strategies_sdxl.py` - Tests for SDXL strategies: dual tokenizers, pool workaround, dual text encoders, text encoder output caching (21 tests).
  - Updated `tests/unit/test_caching.py` - Added heavy mocking tests for `load_images_and_masks_for_caching`, `cache_batch_latents` (VAE), `cache_batch_text_encoder_outputs` (11 new tests, 24 total).
  - Updated `tests/unit/test_training_trainer_utils.py` - Added heavy mocking tests for `prepare_accelerator`, `init_trackers`, `determine_grad_sync_context` (11 new tests, 27 total).

### Fixed

- **Checkpointing v_parameterization Bug**

  - Fixed `checkpointing.py` and `sdxl_checkpointing.py` accessing `training_config.v_parameterization` which doesn't exist on `TrainingConfig` (the field is on `LossConfig`)
  - Replaced `training_config` parameter with `loss_config` in `save_sd_model_on_train_end` and `save_sd_model_on_epoch_end_or_stepwise` functions
  - Updated call sites in `sd_finetune.py` and `sdxl_finetune.py` to pass `cfg.loss` instead of `cfg.training`

- **Hydra Config Field Reference Bugs**

  - Fixed `sd_finetune.py` and `sd_textual_inversion.py` referencing non-existent config fields
  - Fixed `training_config.min_snr_gamma` → `config.loss.min_snr_gamma` (and similar loss fields)
  - Fixed `training_config.zero_terminal_snr` → `config.regularization.zero_terminal_snr`
  - Fixed undefined `sd_models_config` variable → `model_config`
  - Fixed `training_config.max_grad_norm` → `optimizer_config.max_grad_norm`

- **SDXL Fine-tune Config Fixes**

  - Fixed `prepare_deepspeed_model(cfg.performance)` → `prepare_deepspeed_model(cfg.training)`
  - Fixed `cfg.sdxl.fused_backward_pass` → `cfg.optimizer.fused_backward_pass`
  - Fixed `cfg.optimizer.fused_backward_pass` → `cfg.training.fused_backward_pass` (correction: check actual usage if needed, but assuming general fixes here)
  - Added missing `OmegaConf` import for tracker config serialization
  - Fixed `JXLBitstream` in `jpeg_xl_util.py`: `IndexError` on partial reads due to incorrect bit/byte offset tracking.
  - Fixed `sai_model_spec.py`: Missing `import time` causing NameError on timestamp generation.
  - Fixed `torch_utils.py` tests: Mocked `set_seed` to avoid dependency on global state/missing libraries in unit tests.

- **SD Peft Pure Hydra Migration**

  - **Completed `sd_peft.py` migration** to Pure Hydra (removed ~100 undefined `args` references)
  - Refactored `sd_peft.py` to use `cfg.*` paths correctly (e.g. `cfg.training`, `cfg.logging`)
  - Removed unused `import argparse` from `sdxl_peft.py`
  - Fixed method signatures: `generate_step_logs(args)` → `generate_step_logs(cfg)`, `save_timestep_distribution_plot(args)` → `save_timestep_distribution_plot(cfg)`
  - Fixed `cfg.la_sampler` → `self.la_sampler` (runtime object, not config)
  - Fixed call sites: `sample_images_check(cfg)` → `sample_images_check(cfg.sampling)`, `calculate_val_loss_check(cfg)` → `calculate_val_loss_check(cfg.training)`
  - Fixed `prepare_deepspeed_model(cfg)` → `prepare_deepspeed_model(cfg.training)`
  - Fixed `cfg.training.vae` → `cfg.model.vae`
  - Fixed `plot_edm2_loss_weighting_check()` and `plot_edm2_loss_weighting()` - added missing `cfg.training` and `cfg.saving.output_name` parameters
  - Removed stale TODO comment about `no_metadata` field (it exists in `SavingConfig`)

- **Textual Inversion Cleanup**

  - Renamed unused `args` parameters to `config` in `sd_textual_inversion.py` and `sdxl_textual_inversion.py` for consistency

- **Function Naming Cleanup**

  - Renamed `args_set_seed()` → `set_seed_from_config()` in `torch_utils.py`
  - Renamed `prepare_deepspeed_args()` → `prepare_deepspeed_config()` in `deepspeed_utils.py`

- **Library Type Safety**

  - Refactored `prepare_accelerator` in `trainer_utils.py` to accept typed configs (`PerformanceConfig`, `LoggingConfig`, `TrainingConfig`) instead of `DictConfig`
  - Refactored `init_trackers` in `trainer_utils.py` to accept any config type with `.logging` sub-config
  - Refactored `prepare_deepspeed_plugin` and `prepare_deepspeed_config` in `deepspeed_utils.py` to accept `PerformanceConfig` and `TrainingConfig`
  - Added missing `no_metadata` field to `SavingConfig` dataclass

- **Dataclass Naming Consistency**
  - Renamed `sd_models: ModelConfig` → `model: ModelConfig` in 6 dataclasses to match YAML config naming

### Changed

- **HuggingFace Upload Refactor**

  - Refactored `huggingface_util.upload()` to accept `HuggingFaceConfig` instead of `argparse.Namespace`
  - Updated callers in `sd_textual_inversion.py` and `sd_peft.py` to use `config.huggingface`

- **Config Validation Architecture**
  - Removed `__post_init__` methods from 5 dataclasses (`DatasetConfig`, `RegularizationConfig`, `SamplingConfig`, `OptimizerConfig`, `SDXLConfig`)
  - Trainer `validate_extra_config` methods now delegate to centralized module
  - All 6 training scripts call `prepare_config()` and `validate_config()` at entry point
  - Removed duplicate precision validation asserts from `sd_peft.py`, `sd_finetune.py`, `sdxl_finetune.py`

### Removed

- **Legacy Argparse Code**
  - Removed dead imports of `add_logging_arguments`, `add_prompt_parsing_arguments`, `add_loss_weighting_arguments` from scripts
  - Removed deprecated `add_loss_weighting_arguments()` from `loss_weighting.py`
  - Removed deprecated `add_logging_arguments()` from `common_utils.py`
  - Removed deprecated `add_prompt_parsing_arguments()` from `prompt_utils.py`
  - Removed unused `import argparse` from `loss_weighting.py` and `prompt_utils.py`
  - Removed `sdxl_data_utils.py` (unused, superseded by strategy pattern)
  - Removed dead `get_hidden_states(args)` from `text_encoder_util.py` (superseded by strategy)
  - Removed `add_model_spec_arguments()` from `sai_model_spec.py` (superseded by MetadataConfig)
  - Removed `ModelSpecMetadata.from_args()` from `sai_model_spec.py` (use `from_config()` instead)
  - Renamed `generate_user_config_from_args()` → `generate_user_config_from_dataset()` in `config_util.py`

## [2025-12-20]

### Added

- **Hydra Configuration System**

  - `configs/sd_finetune.yaml` - Hydra config for SD 1.5/2.0 fine-tuning
  - `configs/sd_textual_inversion.yaml` - Hydra config for SD 1.5/2.0 textual inversion
  - `configs/sd_peft.yaml` - Hydra config for SD 1.5/2.0 PEFT/LoRA training
  - `configs/sdxl_finetune.yaml` - Hydra config for SDXL fine-tuning
  - `configs/sdxl_textual_inversion.yaml` - Hydra config for SDXL textual inversion
  - `configs/sdxl_peft.yaml` - Hydra config for SDXL PEFT/LoRA training
  - `library/config/config_util.py` - Added `RootConfig` Protocol for type-safe config handling

- **Testing Infrastructure**

  - `tests/conftest.py` - Pytest fixtures for Hydra, configs, temporary directories, and mock objects
  - `tests/unit/test_configs.py` - 28 tests for configuration dataclasses (instantiation, defaults, Hydra composition, overrides)
  - `tests/unit/test_training_optimizer.py` - 26 tests for optimizer module (creation, detection, schedulers, utilities)
  - `tests/unit/test_training_checkpointing.py` - 23 tests for checkpointing module (naming, removal, metadata)
  - `tests/unit/test_training_diffusion.py` - 17 tests for diffusion utilities (timesteps, noisy latents)
  - `tests/unit/test_training_noise_utils.py` - 21 tests for noise utilities (SNR, pyramid noise, noise offset)
  - `tests/unit/test_data_image_utils.py` - 14 tests for image utilities (globbing, loading, cropping)
  - `tests/unit/test_utils_torch.py` - 12 tests for torch utilities (dtype preparation, mixed precision)
  - `tests/unit/test_training_trainer_utils.py` - 16 tests for trainer utilities (validation check, LR logging)
  - `tests/unit/test_data_prompt_utils.py` - 16 tests for prompt utilities (attention parsing, token padding)
  - Enhanced `pytest.ini` with test markers (`unit`, `integration`, `config`, `training`, `data`, `slow`, `requires_gpu`)
  - Restructured tests to `tests/unit/` directory to avoid import naming collisions
  - Added pytest-cov for code coverage reporting
  - `tests/unit/test_utils_safetensors.py` - Tests for `safetensors_utils` (I/O, metadata, header parsing, `mmap` logic, key finding).
  - `tests/unit/test_utils_common.py` - Tests for `common_utils` (`str_to_dtype`, `resize_image` with alpha/interpolation checks, `GradualLatent`, `swap_weight_devices`).
  - `tests/unit/test_data_image_utils.py` - Tests for `image_utils` (`glob_images` escaping, `load_image` mode conversion, `trim_and_resize` logic).
  - `tests/unit/test_losses_loss.py` - Comprehensive tests for loss functions (`stable_mse`, `stable_smooth_l1`, etc.) and Recorders (`LossRecorder`, `EMARecorder`).

### Changed

- **Script Naming Standardization** - Renamed to `{model}_{method}.py` pattern:

  - `fine_tune.py` → `sd_finetune.py`
  - `train_textual_inversion.py` → `sd_textual_inversion.py`
  - `train_network.py` → `sd_peft.py`
  - `sdxl_train.py` → `sdxl_finetune.py`
  - `sdxl_train_textual_inversion.py` → `sdxl_textual_inversion.py`
  - `sdxl_train_network.py` → `sdxl_peft.py`

- **Dataclass File Naming** - Matched to script names in `library/config/dataclasses/`

- **Library Refactoring**

  - `library/training/trainer_utils.py` - `calculate_val_loss_check()` now accepts `TrainingConfig` directly
  - `library/config/config_util.py` - Added `RootConfig` Protocol, removed stale circular import workaround
  - `library/data/dataset.py` - `load_arbitrary_dataset()` accepts `DatasetConfig` directly

### Removed

- **Legacy Code**

  - `ArgsAdapter` class removed from `sd_peft.py` (formerly `train_network.py`)
  - `ConfigAdapter` shim removed from migrated scripts
  - `setup_parser()` and argparse removed from all migrated scripts

### Fixed

- Type hints in `config_util.py` - replaced string hint `"RootConfig"` with proper Protocol class

- **Config Audit Cleanup**

  - Removed duplicate `logging_dir` from `PerformanceConfig` (canonical: `LoggingConfig`)
  - Removed duplicate `vae` from `TrainingConfig` (canonical: `ModelConfig`)
  - Removed duplicate HuggingFace fields from `SavingConfig` (canonical: `HuggingFaceConfig`)
  - Fixed `sdxl_peft.py` to use `ModelConfig` instead of `ModelLoadingConfig`
  - Added missing `BucketsConfig` to `SDXLPeftConfig`
  - Removed orphan `v_parameterization` from `sd_models/default.yaml`
  - Added missing `optimizer_schedulefree_wrapper` fields to `optimizer/default.yaml`
  - Removed unused `NetworkConfig` import from `sd_textual_inversion.py`
  - Renamed `test_train_network_config.py` to `test_sd_peft_config.py`

- **SDXL Configuration Refactoring**

  - Renamed `sdxl_training` config group to `sdxl`
  - Renamed `configs/sdxl_training` directory to `configs/sdxl`
  - Renamed `SDXLTrainingConfig` to `SDXLConfig` in `library/config/dataclasses/sdxl.py`
  - Updates to `sdxl_finetune.py` and `sdxl_peft.py` to use explicit dataclass fields instead of inheritance
  - Fixed duplicate arguments and import errors in `sd_peft.py` and `sdxl_model_prep.py`
  - Removed legacy `library.config.arguments` usage from `sdxl_peft.py`, `sd_finetune.py`, and `sd_textual_inversion.py`
  - Fixed double `@dataclass` decorator in `library/config/dataclasses/model.py`

- **Model Config Refactor**

  - Renamed config group `sd_models` to `model`
  - Merged `SDModelsConfig` and `ModelLoadingConfig` into `ModelConfig` in `library/config/dataclasses/model.py`
  - Moved `v_parameterization` from `SDModelsConfig` to `LossConfig` (and updated scripts to use `config.loss.v_parameterization`)
  - Moved `vae` and `vae_conv2d_padding_mode` from `TrainingConfig`/`SDModelsConfig` to `ModelConfig`
  - Moved `use_ramtorch` and `direct_ramtorch` from `NetworkConfig` to `PerformanceConfig`
  - Updated all scripts and YAMLs to reflect these changes

- **Type hint bug in `library/utils/torch_utils.py`**

  - `prepare_dtype()` function was incorrectly typed to accept `TrainingConfig` but accessed fields from both `PerformanceConfig` (mixed_precision) and `SavingConfig` (save_precision). Refactored to properly accept both config types as separate parameters.
