# Project Roadmap & Future Ideas

## Architecture Overview

The codebase follows a strategy pattern where training loops live in scripts and model-specific operations are delegated to strategy classes.

```
Scripts (contain training loops):     Library Modules:
┌─────────────────┐                   ┌─────────────────────────────┐
│   sd_peft.py    │                   │ library/strategies/         │
│   (~860 lines)  │ ───imports───────►│   peft_strategy_base.py     │
└─────────────────┘                   │   peft_strategy_sd.py       │
┌─────────────────┐                   │   peft_strategy_sdxl.py     │
│  sdxl_peft.py   │ ───imports───────►└─────────────┬───────────────┘
│   (~860 lines)  │                                 │ uses
└─────────────────┘                   ┌─────────────┴───────────────┐
                                      │ library/training/           │
                                      │   trainer_utils.py          │
                                      │   checkpointing.py          │
                                      │   sd/sdxl_checkpointing.py  │
                                      └─────────────────────────────┘
```

**Module Separation:**

- `checkpointing.py` - Generic utilities, `sd_checkpointing.py` - SD-specific
- `model_prep.py` - Generic, `sd_model_prep.py` / `sdxl_model_prep.py` - Model-specific
- `sample_generation.py` - Generic, `sd_sample_generation.py` / `sdxl_sample_generation.py`
- `strategy_sd.py` - SD1.5/2.0 (uses `v2: bool`), `strategy_sdxl.py` - SDXL

---

## Testing Suite

### Current Status

**Infrastructure:** ✅ Complete (pytest, fixtures, coverage)

| Category                | Status         |
| ----------------------- | -------------- |
| **Unit Tests (Pure)**   | ✅ Complete    |
| **Unit Tests (Mocked)** | ✅ In Progress |
| **Integration Tests**   | 🔜 Future      |

**Completed:** 897+ unit tests across configuration, training, data, strategies, networks, losses, pipelines.

### Integration Tests (Remaining)

| Category               | Items                                        | Priority |
| ---------------------- | -------------------------------------------- | -------- |
| **Checkpoint I/O**     | `load_models_from_*`, `save_*_checkpoint`    | High     |
| **Network Classes**    | `LoRAAdapter.apply_to()`, `create_adapter()` | Medium   |
| **Sample Generation**  | `sample_images_common`, inference pipeline   | Medium   |
| **Full Training Loop** | Config → Trainer → Step                      | High     |

---

## Configuration Refactoring

### Active TODOs

- [ ] Config Validation Edge Cases: Test `prepare_config()` and `validate_config()` for dataset conflicts
- [ ] Work on validation in general to figure out a system for catching invalid configs, might need to be post testing

### Completed

- **Learning Rate Consolidation**: Unified `unet_lr`, `text_encoder_lr`, `learning_rate_te1/te2`, `block_lr` into `optimizer.learning_rates`.
- **Config Key Rename**: `cfg.network` → `cfg.peft` across all scripts and library modules
- **Network → Adapter Rename**: Renamed folder, classes, functions, variables, and config fields from `network` to `adapter` terminology
- **Legacy Cleanup**: Removed unused `@property` aliases and fallback logic from optimizer.py
- **Train Text Encoder Options**: Consolidated via `optimizer.learning_rates` usage (implicit vs explicit)
- **Schema 1 Refactor**: Unified configuration schema for PEFT/Fine-tuning scripts
- **Data Config Restructuring**: Merged `DatasetConfig` + `BucketsConfig` into `DataConfig` with 5 nested sub-configs (source, preprocessing, caption, bucketing, caching)
- **`sd_textual_inversion.py` Config Migration** - Completed: uses `cfg.*` pattern, `model_type` handling done via strategy

---

## Code Quality TODOs

- [ ] Timestep sampling needs proper reimplementation (currently hacked into training scripts)
- [ ] Clean integration for external `live_plotter`, possible rework at later time with dedicated logging setup
- [ ] **`edm2_loss_utils.py` Config Cleanup** (low priority, not critical component)
  - 15+ flat config fields with absurdly long names (`edm2_loss_weighting_importance_weighting_safety_override`)
  - Should extract to dedicated `Edm2LossConfig` sub-dataclass
  - Mutates config directly (`loss_config.debiased_estimation_loss = False`)
- [ ] **`training_plots.py`** - Functions access multiple sub-configs (`cfg.output.saving`, `cfg.output.logging`, `cfg.timestep`) - acceptable for orchestration functions but could be cleaner
- [x] ~~Dataset and bucketing decouple~~ (dataset.py split into 6 modules)
- [x] ~~Resolve duplicate `diffusers_xformers`~~ (moved to PerformanceConfig)
- [x] ~~Config passing pattern~~ (see DEVELOPMENT_GUIDE.md Section 5.D)
  - **Scripts/Strategies**: Use `cfg.*` directly (full root config access)
  - **Library Utilities**: Receive the **smallest container** with what they need:
    - Pass `PrecisionConfig` if only precision fields needed (not full `PerformanceConfig`)
    - Pass `LoggingConfig` if only logging fields needed (not full `OutputConfig`)
    - Different params can be at different depths (e.g., `precision_config, saving_config`)
- [x] ~~Apply config pattern to `sd_textual_inversion.py`~~ (completed)
- [x] get rid of lazy imports, move to top for transparency
- [x] ~~**PEFT Strategy Deduplication**~~: 4 methods moved to `peft_strategy_base.py` (`get_noise_scheduler`, `encode_images_to_latents`, `shift_scale_latents`, `post_process_loss`)
- [ ] **PEFT Strategy Internal Dedup**: `process_batch` and `process_val_batch` share ~45 lines of identical latent/text encoding setup - extract to helper method
- [ ] **Consolidate `init_ipex()` calls** (low priority) - During refactoring, `init_ipex()` was copied to all split-out library modules. Original pattern: only training scripts + `model_util.py` need it. Remove from other utility modules like `torch_utils.py`.

---

## Testability Improvements

- [ ] **Split `prepare_accelerator`** - Separate config computation from side effects

  - Currently mixes pure computation (logging_dir, log_with, plugins) with side effects (`os.makedirs`, `os.environ["WANDB_DIR"]`, `wandb.login`)
  - Suggested: Split into `compute_accelerator_config() -> AcceleratorConfig` (pure) and `prepare_accelerator(config)` (side effects)
  - Benefits: Easier to test config logic without network calls or filesystem changes

- [ ] **Explicit step 0 validation** - Clarify `calculate_val_loss_check` behavior at step 0
  - Current logic: `if global_step != 0 and ...` skips the check at step 0, implicitly returning `True`
  - This means validation always runs at step 0, but it's easy to miss in the code
  - Suggested: Add explicit early return `if global_step == 0: return True` with comment, or add `validate_at_start` config flag

---

## CI/CD Setup (Future)

- GitHub Actions workflow for pytest
- Coverage reporting and tracking
- Pre-commit hooks for running tests

---

## Future Ideas

- [ ] **Sample Generation Config Defaults** - Add global defaults to `SamplingConfig` for common sampling parameters:

  - `sample_width`, `sample_height` (default dimensions)
  - `sample_steps`, `sample_cfg_scale` (default inference settings)
  - `sample_negative_prompt` (global negative prompt)
  - These would serve as defaults that per-prompt overrides (in sample_prompts file) could supersede
  - Also: Add YAML support for `sample_prompts` for consistency with the rest of the config system (currently only .txt, .toml, .json)

- [x] **Hydra 1.2 Schema Migration** - Fixed deprecation warning about automatic schema matching:

  - Created `library/config/schemas.py` to centralize ConfigStore schema registration
  - Renamed schemas to `*_schema` suffix (e.g., `sd_peft_schema`) to avoid name collision with YAML files
  - Updated `sd_peft.yaml` and `sdxl_peft.yaml` to include schema in defaults list
  - Fixed stale `max_data_loader_n_workers` field in `performance/default.yaml`

- [ ] **Support for feather** - https://github.com/SuriyaaMM/feather

  - Feather is a high-performance emulation library that brings FP8 (E5M2 & E4M3) precision arithmetic to older GPU architectures (Ampere, Turing, Volta) that lack native hardware support. Currently only considered for inference

- [x] **Evaluate ty for type checking** - https://docs.astral.sh/ty/
  - ty is a fast Python type checker from Astral (ruff authors)
  - Could replace/complement basedpyright for CI type checking

---

## Future Architecture: Per-Model Directory Structure

**Goal:** Reorganize `library/models/` from flat files to per-model directories for better maintainability as more architectures are added.

**Recent changes:**

- [x] Moved `model_prep.py`, `sd_model_prep.py`, `sdxl_model_prep.py` from `training/` → `models/` (model loading belongs here)

**Key insights:**

- `text_encoder_util.py` is **SDXL-specific** (dual CLIP encoders) → should move to `sdxl/`
- `model_util.py` VAE functions are **SD/SDXL shared** (same 4-channel VAE architecture) but not generic for Flux (16-channel)
- No truly "universal" shared folder makes sense - different model families have different architectures
- Truly generic utilities (e.g., `is_safetensors()`) can stay in a `common.py` or move to `utils/`

Currently:

```
library/models/
├── model_util.py          # SD/SDXL VAE utils + is_safetensors
├── model_prep.py          # Generic module patching
├── sd_model_util.py       # SD1/2 conversion & loading
├── sd_model_prep.py       # SD model loading
├── sdxl_model_util.py     # SDXL conversion & loading
├── sdxl_model_prep.py     # SDXL model loading
├── sd_original_unet.py    # SD1/2 UNet architecture
├── sdxl_original_unet.py  # SDXL UNet architecture
├── text_encoder_util.py   # SDXL text encoder utils (misnamed!)
└── ...
```

Proposed future structure:

```
library/models/
├── sd/
│   ├── unet.py            # SD UNet architecture (from sd_original_unet.py)
│   ├── conversion.py      # SD checkpoint conversion (from sd_model_util.py)
│   └── loader.py          # SD model loading (from sd_model_prep.py)
├── sdxl/
│   ├── unet.py            # from sdxl_original_unet.py
│   ├── conversion.py      # from sdxl_model_util.py
│   ├── loader.py          # from sdxl_model_prep.py
│   └── text_encoder.py    # from text_encoder_util.py (SDXL-specific!)
├── flux/
│   ├── dit.py
│   ├── conversion.py
│   └── loader.py
├── vae.py                 # SD/SDXL shared VAE (same 4-ch architecture)
├── common.py              # Truly generic: is_safetensors(), shave_segments()
└── __init__.py
```

**Benefits:**

- Clear separation of concerns per model type
- Easier to add new architectures without bloating existing files
- Consistent structure makes navigation predictable
- No misleading "shared" folder - VAE/common utilities explicit about their scope
