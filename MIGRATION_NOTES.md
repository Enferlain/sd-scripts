# [OUTDATED] Hydra Migration Notes

> **NOTE:** This document is outdated. Please refer to `HYDRA_MIGRATION_PROGRESS.md` for the latest status and architectural decisions.

This document details the migration of `scripts/train_network.py` and related library components from `argparse`/`toml` to Hydra/`dataclasses`.

## Overview

The migration aims to modernize the configuration management of the project, making it more modular, type-safe, and easier to manage. `train_network.py` has been fully refactored to use Hydra.

## Key Changes

### 1. New Configuration Groups

New dataclasses and default YAML files were created to accommodate `train_network.py` specific arguments:

- **`buckets`**: Extracted from `dataset` to handle bucket resolution settings.
  - `enable_bucket`, `min_bucket_reso`, `max_bucket_reso`, `bucket_reso_steps`, `bucket_no_upscale`.
- **`network`**: New group for network training settings (LoRA, LyCORIS, etc.).
  - `network_module`, `network_dim`, `network_alpha`, `network_weights`, `network_args`, etc.
  - Also includes `unet_lr`, `text_encoder_lr` (moved from global/training).
- **`metadata`**: New group for model metadata settings (ModelSpec).
  - `metadata_title`, `metadata_author`, `metadata_description`, etc.

### 2. Configuration Class Updates

Existing dataclasses were updated to include missing arguments found in `train_network.py`:

- **`DatasetConfig`**: Added `validation_split`, `validation_seed`, `weighted_captions` (from prompt utils).
- **`TrainingConfig`**: Added `initial_epoch`, `initial_step`, `skip_until_initial_step`, `validation_timesteps` (and validation schedule args).
- **`PerformanceConfig`**: Added `cpu_offload_checkpointing`, `fp8_base_unet`, `no_half_vae`.
- **`LoggingConfig`**: Added `live_plot_port`, `log_timestep_distribution_every_n_steps`.
- **`LossConfig`**: Added EDM2 loss weighting arguments.
- **`ModelConfig`**: Added `vae_conv2d_padding_mode`.

### 3. `scripts/train_network.py` Refactoring

- **Hydra Integration**: The script now uses `@hydra.main` and accepts a `TrainNetworkConfig` object.
- **ArgsAdapter**: An `ArgsAdapter` class was implemented to wrap the Hydra config object. This adapter mimics the behavior of `argparse.Namespace` (flattened attribute access), allowing legacy library functions (like `prepare_optimizer`, `load_target_model`) to work without modification.
- **Verification Removal**: `verify_training_args` and `prepare_dataset_args` calls were removed/skipped. `verify_training_args` logic (conflict checks) should ideally be migrated to dataclass `__post_init__` methods or a dedicated validator in the future.

## Verification Status

All arguments from the original `train_network.py` have been mapped to the new configuration structure.

- **External Argument Providers**:
  - `library.utils.sai_model_spec`: Mapped to `MetadataConfig`.
  - `library.data.prompt_utils`: `weighted_captions` mapped to `DatasetConfig`.

## Advice for Future Work

1.  **Refactor Library Functions**: Currently, `ArgsAdapter` bridges the gap between Hydra config and legacy functions. Future work should update library functions (e.g., in `library/training/optimizer.py`, `library/training/model_prep.py`) to accept specific config objects (e.g., `OptimizerConfig`, `PerformanceConfig`) directly, removing the need for the adapter.
2.  **Validation Logic**: The logic in `verify_training_args` (checking for conflicting arguments) was skipped. This logic should be reimplemented using Hydra's validation mechanisms or within the dataclasses.
3.  **Naming Consistency**: Some arguments have slightly different homes (e.g., `unet_lr` in `NetworkConfig` vs `TrainingConfig`). Reviewing these placements for semantic correctness across all scripts would be beneficial.
4.  **`BlueprintGenerator`**: This class was updated to handle `buckets` config separately. Ensure this pattern is consistent if more groups are extracted from `dataset` in the future.

## Naming Confusions / Redundancy

- **`unet_lr` / `text_encoder_lr`**: These are specific to network training (fine-tuning specific components) and were placed in `NetworkConfig`. In full fine-tuning (`sdxl_train.py`), they might be handled differently (e.g., block LRs).
- **`validation_timesteps`**: This is a string argument parsed as a list. Ideally, Hydra/OmegaConf supports lists natively. Future refactoring could change the type in `TrainingConfig` to `List[int]` and update the YAML to use list syntax `[50, 350, ...]`.

## Setup for `train_network.py`

The entry point configuration is `configs/train_network.yaml`. To run:

```bash
accelerate launch scripts/sd_peft.py
```

Arguments can be overridden via command line:

```bash
accelerate launch scripts/sd_peft.py network.network_dim=32 optimizer.learning_rate=1e-4
```
