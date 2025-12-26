# Hydra Migration Progress (Status: CONSIDERED COMPLETED AS OF DEC 22 2025 ✅)



**Strategy:** Pure Hydra implementation. All scripts use `@hydra.main` with typed dataclass configs.

## Refactor Updates (Dec 2025)

- **Renamed `sdxl_training` -> `sdxl`**: To clarify distinction between generic training and SDXL-specific model settings.
- **Legacy Cleanup**: Removed `library.config.arguments` and all `argparse` dependencies from migrated scripts.
- **Dec 22 Cleanup**:
  - Refactored `huggingface_util.upload()` to accept `HuggingFaceConfig`
  - Removed `sdxl_data_utils.py`, `add_model_spec_arguments()`, dead `get_hidden_states(args)`
  - Fixed callers in `sd_textual_inversion.py` and `sd_peft.py`
  - Removed `ModelSpecMetadata.from_args()` legacy method from `sai_model_spec.py`
  - Renamed `generate_user_config_from_args()` → `generate_user_config_from_dataset()` in `config_util.py`
  - Renamed `args_set_seed()` → `set_seed_from_config()`, `prepare_deepspeed_args()` → `prepare_deepspeed_config()`
  - Renamed `args` params → `config` in textual inversion trainer methods
  - `sd_peft.py` and `sdxl_peft.py` fully migrated to Pure Hydra (all `args` removed).
  - `prepare_accelerator`, `deepspeed_utils` updated to use typed configs.
  - Removed `add_loss_weighting_arguments()`, `add_logging_arguments()`, `add_prompt_parsing_arguments()`
  - Fixed field location bugs in `sd_finetune.py`, `sd_textual_inversion.py`
  - Fixed dataclass naming: `sd_models:` → `model:` (6 dataclasses)
  - Removed `add_model_spec_arguments()` from `sai_model_spec.py` (superseded by MetadataConfig)
  - Removed `ModelSpecMetadata.from_args()` from `sai_model_spec.py` (use `from_config()` instead)
  - Renamed `generate_user_config_from_args()` → `generate_user_config_from_dataset()` in `config_util.py`
  - All `args` references removed. EDM2 loss weighting function calls corrected to use proper config sub-objects (`cfg.loss`, `cfg.training`, `cfg.output.saving.output_name`).
  - Renamed `args_set_seed()` → `set_seed_from_config()` in `torch_utils.py`
  - Renamed `prepare_deepspeed_args()` → `prepare_deepspeed_config()` in `deepspeed_utils.py`
  - Renamed `args` params → `config` in textual inversion trainer methods

## Script Status

| Script                      | Config Class                 | Status      |
| --------------------------- | ---------------------------- | ----------- |
| `sd_finetune.py`            | `FineTuneConfig`             | ✅ Complete |
| `sd_textual_inversion.py`   | `TextualInversionConfig`     | ✅ Complete |
| `sd_peft.py`                | `SDPeftConfig`               | ✅ Complete |
| `sdxl_finetune.py`          | `SDXLFineTuneConfig`         | ✅ Complete |
| `sdxl_textual_inversion.py` | `SDXLTextualInversionConfig` | ✅ Complete |
| `sdxl_peft.py`              | `SDXLPeftConfig`             | ✅ Complete |

## Naming Convention

**Pattern:** `{model}_{method}.py`

- `sd_` = SD 1.5/2.0
- `sdxl_` = SDXL
- `_finetune` = Full model fine-tuning
- `_textual_inversion` = Embedding training
- `_peft` = LoRA/LyCORIS/adapter training

## Library Migration Status

| Library                             | Status                                   |
| ----------------------------------- | ---------------------------------------- |
| `library/training/optimizer.py`     | ✅ Uses `OptimizerConfig`                |
| `library/training/model_prep.py`    | ✅ Uses config objects                   |
| `library/training/checkpointing.py` | ✅ Uses `SavingConfig`, `TrainingConfig` |
| `library/training/trainer_utils.py` | ✅ Uses `TrainingConfig`                 |
| `library/config/config_util.py`     | ✅ `RootConfig` Protocol defined         |
| `library/data/dataset.py`           | ✅ Uses `DatasetConfig`                  |
| `library/utils/huggingface_util.py` | ✅ Uses `HuggingFaceConfig`              |

## Next Steps

1. ~~**Config Audit** - Review for duplicate/misplaced settings across configs~~ ✅ Complete
2. **`sd_peft.py` Migration** - **COMPLETED** ✅
   - [x] Refactor `args` usage to `cfg`
   - [x] Fix ConfigAdapter usages
   - [x] Cleanup legacy argparse code
3. **Testing Infrastructure** - Add pytest tests for config instantiation
4. **Documentation** - Update README with new script names

## Migration Guide

1. **Do not pass `args`** - Use specific config objects
2. **Use typed imports** - `from library.config.dataclasses.sd_peft import TrainNetworkConfig`
3. **Add missing fields to dataclasses** in `library/config/dataclasses/`
