# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2025-12-31]

### Changed

- **Dataclass Field Metadata Documentation**

  - Added comprehensive `metadata={"help": ...}` to all fields across config files:
    - `training.py`: All 11 fields in `TrainingConfig`
    - `loss.py`: All fields in `HuberConfig`, `SNRConfig`, `RegularizationConfig`, `LossConfig`
    - `performance.py`: All fields in `PrecisionConfig`, `MemoryConfig`, `AttentionConfig`, `CompilationConfig`, `DistributedConfig`
    - `output.py`: All fields in `SavingConfig`, `LoggingConfig`, `HuggingFaceConfig`, `SamplingConfig`, `MetadataConfig`
    - `sd_textual_inversion.py` / `sdxl_textual_inversion.py`: All fields in `TextualInversionSpecificConfig`
    - `peft.py`: All fields in `PeftConfig` with improved help text
    - `data.py`: Added missing `subsets` field help in `SourceConfig`
  - Added NOTE comments about TF32 enable/disable flags being mutually exclusive (could be consolidated)
  - Added TODO comment about `highvram` config field being unused (never sets `HIGH_VRAM` constant)

- **PeftConfig Field Naming Consistency**

  - Renamed config fields to use `adapter_` prefix consistently:
    - `weights` → `adapter_weights`
    - `module` → `adapter_module`
    - `args` → `adapter_args`
  - Updated all references in `sd_peft.py`, `sdxl_peft.py`, `training_metadata.py`
  - YAML config (`configs/peft/default.yaml`) already had correct naming

- **Library Reorganization - Model Prep Move**

  - Moved `model_prep.py`, `sd_model_prep.py`, `sdxl_model_prep.py` from `library/training/` → `library/models/` (model loading belongs with models, not training)
  - Updated test patches in `test_training_model_prep.py` and `test_training_sdxl_model_prep.py` to use new `library.models.*` paths

- **Library Reorganization - Optimizer Module Split**

  - Moved `optimizer.py` from `library/training/` → `library/optimizers/` and split into:
    - `optimizer_factory.py` - `get_optimizer()` factory function (431 lines)
    - `optimizer_utils.py` - `prepare_optimizer()` + helper functions (210 lines)
    - `scheduler.py` - `get_scheduler_fix()` + `get_dummy_scheduler()` (200 lines)
  - Moved `get_dummy_scheduler()` and `parse_string_to_type()` to `scheduler.py`
  - Fixed circular import by giving each file its own logger via `setup_logging()` pattern

- **Extracted `create_training_metadata()`**
  - Moved from `library/utils/model_metadata.py` → `library/training/training_metadata.py`
  - Training-specific `ss_*` metadata now in dedicated file

### Added

- **Configurable Hash Algorithm** (`output.saving.hash_algorithm`)

  - New config option to choose hash algorithm for model checksums
  - Options: `md5`, `sha1`, `sha256` (default), `sha512`, `blake3`
  - `blake3` is fastest (~3s for 6GB, saturates NVMe) but requires `pip install blake3`
  - `sha256` recommended for compatibility with A1111/ComfyUI/ModelSpec
  - Added `calculate_hash(filename, algorithm)` function in `hash_utils.py`

- **DataLoader Config Migration** (`data.loader`)

  - Moved `max_data_loader_n_workers` → `data.loader.max_workers`
  - Moved `persistent_data_loader_workers` → `data.loader.persistent_workers`
  - Removed unused `config_file` and `output_config` from `TrainingConfig`
  - Updated `prepare_deepspeed_config()` to accept `LoaderConfig` instead of `TrainingConfig`

- **ROADMAP Updates**
  - Added detailed explanations for `prepare_accelerator` split and `calculate_val_loss_check` TODOs
  - Marked lazy imports cleanup as complete

## [2025-12-30]

### Changed

- **Library Reorganization - Phase 1 Complete**

  - **Deleted `peft_common.py`** - All functions moved to specialized modules:

    - `prepare_datasets` → `library/data/dataset_setup.py`
    - `calculate_initial_step` → `library/training/trainer_utils.py`
    - `register_adapter_state_hooks` → `library/training/checkpointing.py`
    - `generate_step_logs`, `step_logging`, `epoch_logging` → `library/logging/step_logging.py`
    - `create_training_metadata` → `library/utils/model_metadata.py`
    - `resolve_adapter_kwargs` → `library/adapters/lora_utils.py`
    - `init_timestep_sampler`, `parse_dynamic_timestep_schedule` → `library/timestep/timestep_utils.py`
    - `setup_live_plotter` → `library/logging/training_plots.py`

  - **Refactored `common_utils.py`** (503 → 69 lines - 86% reduction):

    - Moved `swap_weight_devices`, `weighs_to_device`, `str_to_dtype` → `library/utils/torch_utils.py`
    - Moved `pil_resize`, `resize_image`, `get_cv2_interpolation`, `get_pil_interpolation`, `validate_interpolation_fn` → `library/data/image_utils.py`
    - Moved `GradualLatent`, `EulerAncestralDiscreteSchedulerGL` → `library/pipelines/gradual_latent.py` (NEW)
    - Remaining: `setup_logging`, `exists`, `default`, `fire_in_thread`

  - **Moved `init_trackers`** from `trainer_utils.py` → `library/logging/step_logging.py`

  - **Fixed circular imports**:

    - Removed unused `model_metadata` import from `checkpointing.py`
    - Added `setup_logging()` pattern to `dataset_setup.py`, `trainer_utils.py`, `checkpointing.py`
    - Applied duck-typing workaround in `model_metadata.py` for DreamBoothDataset detection

  - Updated smoke tests to import from new module locations
  - Updated documentation comments in `sd_peft.py`, `sdxl_peft.py`, `sd_peft_old.py`

- **Textual Inversion Config Pattern Refactoring**

  - Refactored `TextualInversionTrainer` methods to follow Strategies pattern (receive full `cfg`, use `cfg.*` access):
    - `load_target_model(cfg, weight_dtype, accelerator)`
    - `get_tokenize_strategy(cfg)`
    - `get_latents_caching_strategy(cfg)`
    - `get_text_encoding_strategy(cfg)`
  - Updated `SdxlTextualInversionTrainer` overrides to match new base class signatures
  - Fixed type hint: `assert_token_string` now correctly takes `tokenizers: List[Any]` instead of `CLIPTokenizer`
  - Fixed confusing `token_ids` reference to use `token_ids_list[-1]` for clarity

- **Config Validation Fix**

  - Fixed `should_train_text_encoder(cfg.optimizer)` → `should_train_text_encoder(cfg.optimizer.learning_rates)` in `config_validation.py`

### Fixed

- **Stale Test Fixtures**

  - Updated `test_config_validation.py` fixtures to use new config schema (`data.caching`, `loss.regularization`, `loss.snr`, `model.model_type`)
  - Updated `test_utils_torch.py` fixtures to use `PrecisionConfig` instead of `PerformanceConfig`
  - Updated `test_utils_sai_model_spec.py` to reference `model_metadata` instead of old `sai_model_spec` module
  - Updated `test_training_sdxl_checkpointing.py` patches and function names (`sai_model_spec` → `model_metadata`, `get_sai_model_spec_from_config` → `get_model_metadata_from_config`)
  - Removed stale `make_bucket_resolutions` tests from `test_models_model_util.py` (function was deleted)

- **Model Utility Refactoring**

  - Split `model_util.py` into shared utilities and SD-specific `sd_model_util.py`
  - Moved `make_bucket_resolutions` to `data_structures.py` (where `BucketManager` uses it)
  - Fixed circular import between `model_util.py` and `sd_model_util.py` by keeping helper functions in shared module
  - Added `__post_init__` to `OptimizerConfig` to handle legacy `use_8bit_adam` and `use_lion_optimizer` flags

- **Test Suite Fixes (13 tests fixed)**

  - `test_sd_peft_config.py`: Updated config key assertions (`buckets` → `data`, `dataset` → `model`)
  - `test_data_structures.py`: Updated `test_make_buckets_calls_model_util` to test real function (no longer mocks)
  - `test_utils_common.py`: Fixed mock paths (`common_utils` → `torch_utils`, `gradual_latent`)
  - `test_training_sample_generation.py`: Added missing `loss_config` parameter
  - `test_training_sdxl_checkpointing.py` (8 tests): Fixed to use `mock_sdxl_util` instead of importing real module
  - `test_models_bucket_resolutions.py`: Added new test file for `make_bucket_resolutions` function

### Removed

- Removed obsolete TODO/FIXME comments from textual inversion scripts (IDE type warnings, resolved issues)

## [2025-12-29]

### Changed

- **Configs sanitization continued:**

  - `trainer_utils.py`: Refactored `prepare_accelerator` to use `PrecisionConfig`, `CompilationConfig`, `DistributedConfig`, `DeepSpeedConfig` instead of parent `PerformanceConfig`
  - `trainer_utils.py`: Refactored `determine_grad_sync_context` to use `precision_config: PrecisionConfig` instead of `args`
  - `deepspeed_utils.py`: Fixed bugs where `prepare_deepspeed_plugin` accessed non-existent fields (e.g., `performance_config.mixed_precision` instead of `precision_config.mixed_precision`)
  - `deepspeed_utils.py`: Refactored `prepare_deepspeed_plugin` to use `DeepSpeedConfig`, `PrecisionConfig`, `TrainingConfig` instead of parent `PerformanceConfig`
  - `deepspeed_utils.py`: Refactored `prepare_deepspeed_config` to use `DeepSpeedConfig` instead of parent `PerformanceConfig`
  - `sd_model_prep.py`: Refactored `load_target_model` to use `MemoryConfig` instead of parent `PerformanceConfig`
  - `sdxl_model_prep.py`: Refactored `load_target_model` to use `ModelConfig`, `MemoryConfig`, `CachingConfig`, `PrecisionConfig` instead of full `SDXLFineTuneConfig`
  - `sdxl_model_prep.py`: Refactored `_load_target_model` to use `ModelConfig` instead of full config
  - `checkpointing.py`: Renamed `config` → `saving_config` for consistency; added `hf_config: HuggingFaceConfig` to `resume_from_local_or_hf_if_specified`
  - `sample_generation.py`: Added `loss_config: LossConfig` parameter; fixed `v_parameterization` access (was incorrectly using `training_config`, now correctly uses `loss_config`)
  - `optimizer.py`: Refactored `prepare_optimizer` to add explicit `learning_rates: LearningRatesConfig` parameter
  - `optimizer.py`: Refactored `get_optimizer` to add explicit `learning_rates: LearningRatesConfig` and `scheduler_config: SchedulerConfig` parameters
  - `optimizer.py`: Refactored `get_scheduler_fix` to use `scheduler_config: SchedulerConfig` and `optimizer_config: OptimizerConfig` instead of full params
  - `sd_textual_inversion.py`: Migrated from `*_config` aliases to direct `cfg.*` access pattern; fixed `prepare_accelerator` and `prepare_dtype` calls with correct sub-configs

### Fixed

- **Truncated `init_trackers` calls**

  - Fixed syntax errors in `sd_finetune.py`, `sdxl_finetune.py`, and `sd_textual_inversion.py` where `accelerator.init_trackers()` calls were malformed (double commas, missing `init_kwargs`, missing closing paren)

- **Null check for `hf_config`**

  - Fixed `resume_from_local_or_hf_if_specified` to check `hf_config is None` before accessing `resume_from_huggingface`

- **Test parameter naming**

  - Fixed `test_training_checkpointing.py` to use `adapter_args` instead of `network_args` and `ss_adapter_args` instead of `ss_network_args`

## [2025-12-28]

### Fixed

- **Epoch Variable Initialization**

  - Added `epoch = 0` initialization before training loops in `sd_finetune.py` and `sdxl_finetune.py` to prevent potential "referenced before assignment" errors when `num_train_epochs` is 0

- **noisy_latents dtype Handling**

  - Added `output_dtype` parameter to `get_noise_noisy_latents_and_timesteps()` in `diffusion.py`
  - Updated all callers (peft strategies, finetune scripts) to pass `output_dtype=weight_dtype`
  - Removed redundant `.to(weight_dtype)` casts from `call_unet` methods
  - Fixed incorrect function signature in `sd_textual_inversion.py` (pre-existing bug)

- **Optimizer Wrapper Guard**
  - Added guard in `optimizer.py` to raise clear error when `base_optimizer_type` is missing for ScheduleFreeWrapper/snoo_asgd optimizers

### Changed

- **Removed Legacy UI Workarounds**

  - Removed auto-adjustment of `first_cycle_max_steps` and `warmup_steps` based on `validation_split` in scheduler setup
  - Callers are now responsible for passing correct values

- **VAE Scale Factor Naming Cleanup**

  - Added `SD_VAE_LATENT_SCALE = 0.18215` constant to `constants.py`
  - Renamed `SDXL_VAE_SCALE_FACTOR` → `SDXL_VAE_LATENT_SCALE` for clarity
  - Renamed strategy field `vae_scale_factor` → `vae_latent_scale` in peft strategies and textual inversion trainers
  - lpw pipelines retain `vae_scale_factor` (spatial 8x, matches diffusers naming)

- **Model Metadata Module Rename**

  - Renamed `sai_model_spec.py` → `model_metadata.py` (now model-agnostic for Flux, Lumina, Hunyuan, etc.)
  - Renamed variable `sai_metadata` → `modelspec_metadata` across all files
  - Renamed strategy method `get_sai_model_spec()` → `get_model_metadata()`
  - Updated module imports throughout codebase

- **Naming Convention Cleanup**

  - Renamed `config` → `cfg` in `resume_from_local_or_hf_if_specified()` for consistency

- **Config Pattern Standardization in Library Utilities**
  - Established pattern: library utilities receive the **smallest container** that has what they need
  - `torch_utils.py`: Refactored `prepare_dtype`, `match_mixed_precision`, `set_torch_cuda_reduced_precision` to use `PrecisionConfig` instead of opaque `cfg`
  - `torch_utils.py`: Refactored `set_seed_from_config` to use `training_config: TrainingConfig`
  - `trainer_utils.py`: Fixed `init_trackers` to use `logging_config: LoggingConfig` directly
  - `sdxl_model_prep.py`: Fixed wrong `cfg.sd_models.*` → `cfg.model.*` and `cfg.performance.precision` type mismatch
  - `timestep_utils.py`: Refactored `parse_dynamic_timestep_schedule` and `init_timestep_sampler` to use `timestep_config: TimestepConfig`
  - `dataset_utils.py`: Refactored `load_arbitrary_dataset` to use `data_config: DataConfig` + `max_token_length: int`
  - Updated all corresponding tests to use new signatures

### Removed

- Stale TODO comments about TrainingConfig/v_parameterization in checkpointing modules
- Legacy `# TODO HYDRA` comment
- Hardcoded block_lr 23-value validation in `config_validation.py` (deferred for model-agnostic block/layer granular training)
- Unused `make_bucket_resolutions()` test code from `model_util.py`

## [2025-12-27]

### Changed

- **Optimizer Scheduler Config Nesting**

  - Created `SchedulerConfig` dataclass with 9 LR scheduler fields
  - Nested under `OptimizerConfig.scheduler`
  - Config access paths updated: `cfg.optimizer.lr_scheduler` → `cfg.optimizer.scheduler.lr_scheduler`, etc.
  - YAML updated: `configs/optimizer/default.yaml` now has `scheduler:` section

- **Learning Rate Consolidation**

  - Removed redundant `OptimizerConfig.learning_rate` field
  - `LearningRatesConfig.base` is now the canonical base LR (default: `2.0e-6`)
  - Config access paths updated: `cfg.optimizer.learning_rate` → `cfg.optimizer.learning_rates.base`

- **Learning Rate Parameter Passing Consolidation**

  - Refactored adapter `prepare_optimizer_params` methods (`lora.py`, `dylora.py`, `oft.py`) to accept `LearningRatesConfig` object instead of individual `text_encoder_lr`, `unet_lr`, and `learning_rate` float params
  - Removed legacy `text_encoder_lr` return value from `prepare_optimizer()` in `optimizer.py`
  - Updated `create_training_metadata()` to read `ss_text_encoder_lr` directly from `cfg.optimizer.learning_rates.text_encoders`
  - `cfg.optimizer.learning_rates` is now the single source of truth for all learning rates throughout the training pipeline

- **Model Config Restructuring**

  - Added `model_type` field to `ModelConfig` (`sd15 | sd2 | sdxl | flux`)
  - Removed `v2` boolean field - replaced with `model_type == "sd2"` checks
  - Configs are now self-documenting (model architecture visible at a glance)

- **Unified Validation Config**

  - Created top-level `ValidationConfig` with all 6 validation fields
  - Moved from `cfg.training.*` and `cfg.dataset.*` → `cfg.validation.*`
  - New file: `library/config/dataclasses/validation.py`, `configs/validation/default.yaml`
  - Added to all root config dataclasses and Hydra defaults

- **Loss Config Consolidation**

  - Created nested `LossConfig` with 5 sub-configs: `HuberConfig`, `SNRConfig`, `MaskedLossConfig`, `RegularizationConfig`, `EDM2Config`
  - Config access paths updated: `cfg.masked_loss.*` → `cfg.loss.masked.*`, `cfg.regularization.*` → `cfg.loss.regularization.*`
  - YAML configs consolidated: removed `configs/masked_loss/`, `configs/regularization/` → merged into `configs/loss/default.yaml`
  - Updated all 6 root config dataclasses and YAML files

- **Timestep Config Restructuring**

  - Replaced 22 flat `mix_adaptive_*` fields with 4 per-sampler nested dataclasses
  - New nested sections: `mix_adaptive`, `tempered_adaptive`, `gaussian_mid_snr`, `snr_windowed`
  - Config access paths updated: `cfg.timestep.mix_adaptive_bins` → `cfg.timestep.mix_adaptive.bins`, etc.
  - Each sampler type now has its own complete config section

- **Timestep and Logging Folder Reorganization**

  - Moved `library/timestep_samplers/` → `library/timestep/samplers/`
  - Created `library/timestep/timestep_utils.py` with `init_timestep_sampler`, `parse_dynamic_timestep_schedule`
  - Moved `tools/visualization/` → `library/logging/live_plotter/`
  - Created `library/logging/training_plots.py` with `save_timestep_distribution_plot`, `close_live_plotter`, `get_plotter_settings`, `setup_live_plotter`
  - Scripts (`sd_peft.py`, `sdxl_peft.py`) now import directly from new modules

- **Data Config Restructuring**

  - Merged `DatasetConfig` (~35 fields) + `BucketsConfig` (5 fields) into unified `DataConfig`
  - Created 5 nested sub-configs: `SourceConfig`, `PreprocessingConfig`, `CaptionConfig`, `BucketingConfig`, `CachingConfig`
  - New files: `library/config/dataclasses/data.py`, `configs/data/default.yaml`
  - Removed: `library/config/dataclasses/buckets.py`, `configs/dataset/`, `configs/buckets/`
  - Config access paths updated:
    - `cfg.dataset.train_data_dir` → `cfg.data.source.train_data_dir`
    - `cfg.dataset.cache_latents` → `cfg.data.caching.cache_latents`
    - `cfg.dataset.shuffle_caption` → `cfg.data.caption.shuffle_caption`
    - `cfg.dataset.flip_aug` → `cfg.data.preprocessing.flip_aug`
    - `cfg.buckets.enable_bucket` → `cfg.data.bucketing.enable_bucket`
  - Updated function signatures: `prepare_optimizer` (removed unused dataset_config param), `get_scheduler_fix` (now takes validation_split float), `load_arbitrary_dataset` (now takes root cfg)
  - `BlueprintGenerator` refactored to auto-search sub-configs for field values

- **DeepSpeedConfig Consolidation**

  - Moved `DeepSpeedConfig` from separate `deepspeed.py` into `performance.py` with other sub-configs
  - Deleted: `library/config/dataclasses/deepspeed.py`
  - Updated test imports to use new location
  - `peft_common.py` reduced from 974 → 650 lines

## [2025-12-26]

### Changed

- **Network → Adapter Terminology Rename**

  - **Folder:** `library/networks/` → `library/adapters/`
  - **Classes:** `LoRANetwork` → `LoRAAdapter`, `OFTNetwork` → `OFTAdapter`, `DyLoRANetwork` → `DyLoRAAdapter`
  - **Variable:** `network` → `adapter` across all scripts and library modules (~200+ occurrences)
  - **Functions:** `create_network` → `create_adapter`, `prepare_network` → `prepare_adapter`, etc.
  - **Config fields renamed:**
    - `dim` → `adapter_rank`
    - `alpha` → `adapter_alpha`
    - `dim_from_weights` → `adapter_rank_from_weights`
    - `neuron_dropout` kept (distinct from `rank_dropout`/`module_dropout`)
  - **Metadata keys:** `ss_network_*` → `ss_adapter_*` in safetensors metadata

- **Output Config Consolidation**

  - Created `OutputConfig` dataclass nesting 5 related configs: `SavingConfig`, `LoggingConfig`, `HuggingFaceConfig`, `SamplingConfig`, `MetadataConfig`
  - Config access paths updated: `cfg.saving.*` → `cfg.output.saving.*`, `cfg.logging.*` → `cfg.output.logging.*`, etc.
  - YAML configs consolidated: removed `configs/saving/`, `configs/logging/`, `configs/huggingface/`, `configs/sampling/`, `configs/metadata/` → single `configs/output/default.yaml`
  - Updated all 6 root config dataclasses and YAML files

- **Loss Config Consolidation**

  - Created nested `LossConfig` with 5 sub-configs: `HuberConfig`, `SNRConfig`, `MaskedLossConfig`, `RegularizationConfig`, `EDM2Config`
  - Config access paths updated: `cfg.masked_loss.*` → `cfg.loss.masked.*`, `cfg.regularization.*` → `cfg.loss.regularization.*`
  - YAML configs consolidated: removed `configs/masked_loss/`, `configs/regularization/` → merged into `configs/loss/default.yaml`
  - Updated all 6 root config dataclasses and YAML files

- **SD Fine-Tune Legacy Cleanup**

  - Removed `SDFineTuneSpecificConfig` dataclass and `fine_tune:` YAML section
  - `train_text_encoder` is now controlled via LR: set `optimizer.learning_rates.text_encoders` to enable TE training
  - `learning_rate_te` removed (use `optimizer.learning_rates.text_encoders` instead)

## [2025-12-25]

### Changed

- **Configuration Schema Refactor (Schema 1: Structured/Verbose)**

  - **Unified Learning Rate Configuration**: Consolidated all learning rate settings under `cfg.optimizer.learning_rates`:

    - `unet`: Dedicated field for UNet LR.
    - `text_encoders`: Dedicated field for Text Encoder LR(s). Supports separate LRs for SDXL via list `[lr_te1, lr_te2]`.
    - `blocks`: Per-block learning rates (moved from `cfg.sdxl.block_lr`).
    - Falls back to base `cfg.optimizer.learning_rate` when specific LRs are not set.
    - **Breaking**: Legacy fields (`unet_lr`, `text_encoder_lr` in PeftConfig, `learning_rate_te1/te2`, `block_lr` in SDXLConfig) removed.

  - **Config Key Rename: `cfg.network` → `cfg.peft`**:

    - Renamed config group from `network` to `peft` for semantic clarity.
    - All scripts and library modules updated to use `cfg.peft.*`.
    - YAML config moved from `configs/network/default.yaml` to `configs/peft/default.yaml`.

  - **PEFT Configuration Cleanup**:

    - Removed unused `@property` aliases from `PeftConfig` (they didn't work with Hydra/YAML).
    - Removed legacy fallback logic from `optimizer.py`.

  - **Script Migrations**:

    - **`sd_peft.py` / `sdxl_peft.py`**: Fully migrated to use `cfg.peft.*` and `cfg.optimizer.learning_rates.*`.
    - **`sdxl_finetune.py`**: Updated to use `cfg.optimizer.learning_rates.blocks`.
    - `validation.py` and tests updated accordingly.

  - **LR-Based Training Control**:

    - Removed `train_unet_only` and `train_text_encoder_only` boolean flags from `PeftConfig`.
    - Training control now inferred from learning rates: setting a component's LR to 0 disables its training.
    - Added `should_train_text_encoder()` and `should_train_unet()` helper functions to `optimizer.py`.
    - Updated `peft_strategy_base.py` methods `is_train_text_encoder()` and added `is_train_unet()` to delegate to LR helpers.
    - Updated validation, logging, and script logic to use LR-based detection.

  - **SDXLConfig Dissolution**:

    - Moved `cache_text_encoder_outputs`, `cache_text_encoder_outputs_to_disk`, `disable_mmap_load_safetensors` to `PerformanceConfig`.
    - Moved `fused_optimizer_groups` to `OptimizerConfig`.
    - Removed `train_text_encoder` (use LR-based control via `optimizer.learning_rates.text_encoders`).
    - `SDXLConfig` is now empty; kept for future SDXL-specific settings.

  - **PerformanceConfig Subcategories**:

    - Restructured `PerformanceConfig` with nested dataclasses for better organization:
      - `precision`: mixed_precision, full_fp16, full_bf16, fp8_base, fp8_base_unet, no_half_vae, cuda precision ops
      - `memory`: gradient_checkpointing, cpu_offload_checkpointing, lowram, highvram, ramtorch
      - `attention`: mem_eff_attn, xformers, sdpa, diffusers_xformers
      - `compilation`: torch_compile, dynamo_backend
      - `distributed`: ddp_timeout, ddp_gradient_as_bucket_view, ddp_static_graph
      - `caching`: cache_text_encoder_outputs, cache_text_encoder_outputs_to_disk, disable_mmap_load_safetensors
    - Updated all scripts, strategies, and validation to use nested paths (e.g., `cfg.performance.precision.mixed_precision`).

## [2025-12-24]

### Changed

- **Dataset Module Refactoring**

  - Split `library/data/dataset.py` (2351 → ~1060 lines) into modular files:
    - `dreambooth_dataset.py`: DreamBoothDataset class
    - `finetuning_dataset.py`: FineTuningDataset class
    - `controlnet_dataset.py`: ControlNetDataset class
    - `minimal_dataset.py`: MinimalDataset class
    - `dataset_group.py`: DatasetGroup class
    - `dataset_utils.py`: Utility functions (collator_class, load_arbitrary_dataset, split_train_val, debug_dataset, ImageLoadingDataset)
  - BaseDataset remains in `dataset.py` with backwards-compatible re-exports
  - No breaking changes: all existing imports continue to work

- **Config Consolidation: `diffusers_xformers`**

  - Moved `diffusers_xformers` from `SDXLConfig` and `SDFineTuneSpecificConfig` to `PerformanceConfig`
  - Updated `sd_finetune.py` and `sdxl_finetune.py` to use `cfg.performance.attention.diffusers_xformers`
  - Standardized `sd_finetune.py` to use `cfg` variable name (matching `sdxl_finetune.py`)

- **SAI Model Spec Consolidation**

  - Updated `peft_strategy_sd.py` and `peft_strategy_sdxl.py` to use `get_sai_model_spec_from_config()` instead of legacy argparse-based function
  - Removed ~135 lines of duplicate legacy code from `checkpointing.py`:
    - Removed `get_sai_model_spec()` (legacy, used argparse)
    - Removed `get_sai_model_spec_dataclass()` (legacy, unused)
  - Canonical function is now `library.utils.sai_model_spec.get_sai_model_spec_from_config()`

- **Text Encoder Utility Consolidation**

  - Consolidated duplicate `get_hidden_states_sdxl()` and `pool_workaround()` between `text_encoder_util.py` and `strategy_sdxl.py`
  - `SdxlTextEncodingStrategy` methods now delegate to shared utilities
  - Removed ~60 lines of duplicate code from `strategy_sdxl.py`

- **Dataclass Config Cleanup**

  - Consolidated `no_half_vae` to `PerformanceConfig` (canonical), removed from `SDXLConfig`
  - Updated `sdxl_finetune.py` to use `cfg.performance.precision.no_half_vae`
  - Improved `text_encoder_lr` documentation in `PeftConfig` (explains `Any` type, future unification plans)

- **Checkpointing Module Split**

  - Created `sd_checkpointing.py` with SD1.5/2-specific save functions
  - Removed SD-specific code from `checkpointing.py` (now generic-only)
  - Updated `sd_finetune.py` to import from `sd_checkpointing.py`
  - Mirrors existing pattern: `checkpointing.py` (generic) + `sdxl_checkpointing.py` (SDXL)

- **Model Prep Module Split**

  - Created `sd_model_prep.py` with SD1.5/2-specific model loading functions
  - `model_prep.py` now contains only generic utilities (`replace_unet_modules`, `patch_accelerator_for_fp16_training`, `set_padding_mode_for_vae_conv2d_modules`)
  - Updated SD scripts, strategies, and tools to import from `sd_model_prep.py`
  - Removed unused `load_target_model` import from `sdxl_peft.py`

- **Sample Generation Module Split**

  - Created `sd_sample_generation.py` with SD-specific `sample_images` wrapper
  - `sample_generation.py` now contains only generic utilities (`sample_images_common`, `sample_images_check`, etc.)
  - Mirrors existing `sdxl_sample_generation.py` pattern

- **Folder Rename: `optimizations/` → `performance/`**

  - Renamed `library/optimizations/` to `library/performance/` for consistency with `PerformanceConfig`
  - Updated all imports in scripts

- **Created `AGENTS.md`**
  - Agent instructions for working on the repository (venv location, test commands, project structure, conventions)

## [2025-12-23]

### Changed

- **PEFT Strategy-Based Refactoring (Phase 1-2)**

  - Reduced `sd_peft.py` from 927 lines to 50 lines (95% reduction)
  - Removed `SDPeftTrainer` class - all functionality extracted to modular components
  - Created `library/strategies/peft_strategy_base.py` (16 methods) - ABC interfaces for PEFT training
  - Created `library/strategies/peft_strategy_sd.py` (21 methods) - SD1.5/2 implementations
  - Created `library/training/peft_common.py` (6 functions) - shared logging/plotting utilities
  - Refactored `library/training/peft_trainer.py` to use strategy pattern and standalone functions
  - `train()` function now accepts `strategies: PeftTrainingStrategy` parameter

- **Phase 4: Training Loop Cleanup**

  - Extracted `prepare_datasets()` to `peft_common.py` (~47 lines saved per script)
  - Extracted `calculate_initial_step()` to `peft_common.py` (~42 lines saved per script)
  - Extracted `parse_dynamic_timestep_schedule()` to `peft_common.py` (~10 lines saved per script)
  - Extracted `register_adapter_state_hooks()` to `peft_common.py` (~40 lines saved per script)
  - Total: ~140 lines reduced from each PEFT script (1173 → 1035 lines)

### Fixed

- **HuggingFace Upload Bug in Checkpointing**
  - The Hydra migration accidentally replaced `if args.huggingface_repo_id is not None` with `if saving_config.resume is not None` (wrong!) and stubbed out upload calls with `pass`
  - Added `hf_config: Optional[HuggingFaceConfig] = None` parameter to 9 functions in `checkpointing.py` and `sdxl_checkpointing.py`
  - Restored proper upload logic for model checkpoints and training state
  - Functions affected: `save_sd_model_on_epoch_end_or_stepwise`, `save_sd_model_on_train_end`, `save_and_remove_state_*` variants

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
  - `test_dataset_bucketing.py` - Added 8 tests for `BaseDataset.__getitem__` (cached/disk latents, image loading, flip aug, batching)
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
  - Fixed `plot_edm2_loss_weighting_check()` and `plot_edm2_loss_weighting()` - added missing `cfg.training` and `cfg.output.saving.output_name` parameters
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
  - Removed unused `PeftConfig` import from `sd_textual_inversion.py`
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
  - Moved `use_ramtorch` and `direct_ramtorch` from `PeftConfig` to `PerformanceConfig`
  - Updated all scripts and YAMLs to reflect these changes

- **Type hint bug in `library/utils/torch_utils.py`**

  - `prepare_dtype()` function was incorrectly typed to accept `TrainingConfig` but accessed fields from both `PerformanceConfig` (mixed_precision) and `SavingConfig` (save_precision). Refactored to properly accept both config types as separate parameters.
