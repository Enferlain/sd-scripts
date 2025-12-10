# Hydra Configuration Migration Status

This document outlines the progress of the migration from the legacy `argparse` and `.toml` configuration system to the new Hydra and `dataclasses` based system.

## What's Done: The Foundation

The first and most complex training script, `sdxl_train.py`, has been fully refactored.

### Key Changes:
1.  **New Configuration Structure:**
    *   A new `configs/` directory now holds all configuration in YAML files.
    *   The configuration is modular, broken down into logical groups like `dataset`, `optimizer`, and `training`.
    *   The `library/config/dataclasses/` directory contains Python `dataclass` definitions that provide a type-safe schema for the YAML files.

2.  **`sdxl_train.py` Refactoring:**
    *   The script no longer uses `argparse`.
    *   It is now decorated with `@hydra.main()` and receives a single, structured `cfg` object containing all configuration.
    *   All utility functions called by the script have been updated to accept the new `cfg` object.

3.  **Legacy Compatibility:**
    *   A compatibility "adapter" has been implemented within `sdxl_train.py`. This adapter converts the new nested Hydra configuration into a flat structure that the legacy `BlueprintGenerator` can understand.
    *   This ensures that the existing, complex dataset logic continues to work without requiring a full rewrite at this stage.

4.  **How to Use It:**
    *   **Defaults:** Running `accelerate launch scripts/sdxl_train.py` will use the default configuration defined in the `configs/` directory.
    *   **Overrides:** Any setting can be changed from the command line using a simple dot-path syntax (e.g., `optimizer.learning_rate=0.0001`).
    *   **Presets:** You can create reusable experiment presets (e.g., `my_lora.yaml`) and run them with `--config-name presets/my_lora`.

---

## Outstanding Work & Next Steps

This initial refactoring lays the groundwork. The following tasks remain to complete the migration.

### 1. Refactor Remaining Training Scripts
The following scripts still use the old `argparse` system and need to be migrated to Hydra:
*   `train_network.py`
*   `train_textual_inversion.py`
*   `fine_tune.py`
*   *(And any other scripts in the `scripts/` directory)*

### 2. Refine Configuration Groups
As you noted, the current `training` configuration group is very large because it's a direct 1-to-1 mapping of the old flat structure. To improve organization, we should break it down into smaller, more focused groups.

**Suggested New Groups:**
*   `saving:` (for `output_dir`, `save_every_n_steps`, `save_precision`, etc.)
*   `logging:` (for `logging_dir`, `log_with`, `wandb_api_key`, etc.)
*   `sampling:` (for `sample_every_n_steps`, `sample_sampler`, `sample_prompts`, etc.)
*   `buckets:` (for `min_bucket_reso`, `max_bucket_reso`, etc., which are currently under `dataset`)

This would involve creating new dataclasses and YAML files and updating the main `config.yaml`. The code in `sdxl_train.py` would then access these via `cfg.saving.output_dir`, for example.

### 3. Add Missing Script-Specific Arguments
Some arguments are specific to certain training types (like LoRA) and were not included in the `sdxl_train.py` refactor. As each script is migrated, these arguments must be added to the dataclasses and YAML files.

**Examples from `train_network.py`:**
*   `network_dim`
*   `network_alpha`
*   `network_module`
*   `network_args`
*   `network_train_unet_only`

A new configuration group, `network`, would be the ideal place for these.

### 4. Long-Term: Phase Out Legacy Utilities
The adapter pattern for `BlueprintGenerator` is a successful short-term solution. However, for a complete migration, the following long-term goals should be considered:
*   Refactor `BlueprintGenerator` and `ConfigSanitizer` to natively understand and work with the new Hydra `DictConfig` objects.
*   Once that is done, the adapter code in the training scripts can be removed.
*   Ultimately, the `prepare_dataset_args` function and the `ConfigSanitizer` class could be fully deprecated and removed, as their logic would be absorbed into the new dataclasses (with custom validators) and the refactored `BlueprintGenerator`.
