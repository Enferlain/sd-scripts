# Hydra Migration Progress (Status: Active)

**Strategy Update:** We are moving towards a "Pure Hydra" implementation. The use of temporary compatibility layers (like `ArgsAdapter`) is **deprecated**. All code must be refactored to consume the new component-specific configuration objects (`OptimizationConfig`, `DatasetConfig`, etc.) directly.

## Current Status Overview

| Script / Component                  | Status      | Notes                                                                                                        |
| :---------------------------------- | :---------- | :----------------------------------------------------------------------------------------------------------- |
| **Scripts**                         |             |                                                                                                              |
| `sdxl_train.py`                     | � Migrated  | Fully migrated. `ArgsAdapter` removed. Uses `SDXLFineTuningConfig`.                                          |
| `train_network.py`                  | � Migrated  | Fully migrated. `ArgsAdapter` removed. Function calls updated to strictly pass config objects.               |
| `sdxl_train_network.py`             | � Migrated  | Fully migrated. `ArgsAdapter` removed. Function calls updated to strictly pass config objects.               |
| `train_textual_inversion.py`        | 🔴 Legacy   | Still using `argparse`.                                                                                      |
| **Libraries**                       |             |                                                                                                              |
| `library/training/optimizer.py`     | 🟢 Migrated | Refactored to use `OptimizerConfig`.                                                                         |
| `library/training/model_prep.py`    | 🟢 Migrated | Refactored to use relevant config objects.                                                                   |
| `library/training/checkpointing.py` | � Migrated  | Refactored to accept `SavingConfig`, `TrainingConfig`, etc. Legacy `args` support removed for updated paths. |
| `library/utils/sai_model_spec.py`   | � Migrated  | Refactored to accept configs.                                                                                |

## Completed Action Items (The "No Adapter" Plan)

1.  **Refactor Checkpointing (`library/training/checkpointing.py`)**: ✅ Completed

    - Refactored `save_sd_model_on_epoch_end_or_stepwise`, `save_state_on_train_end`, etc. to accept specific config objects.

2.  **Refactor Model Spec (`library/utils/sai_model_spec.py`)**: ✅ Completed

    - Updated to support `TrainingConfig` and `MetadataConfig`.

3.  **Cleanup**: ✅ Completed
    - `ArgsAdapter` removed from main training scripts.
    - `library/utils/args_adapter.py` deleted.

## Next Steps

1.  **Testing Infrastructure**:

    - Configure `pytest`.
    - Create unit tests for refactored library functions (`optimizer`, `model_prep`, `checkpointing`).

2.  **Migrate Remaining Scripts**:
    - `train_textual_inversion.py`
    - `fine_tune.py`
    - (and their SDXL variants)

## Migration Guide for Contributors

When working on a script or library function:

1.  **Do not pass `args`**. If you see `args` in a function signature you are calling, **refactor that function first**.
2.  **Use Specific Configs**. Instead of passing the whole `cfg` object, pass only what is needed (e.g., `cfg.optimizer`, `cfg.training`).
3.  **Update Dataclasses**. If a standard argument is missing from a dataclass (e.g., `ValidationConfig`), add it to `library/config/dataclasses/`.
