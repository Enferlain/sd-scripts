# Project Roadmap & Future Ideas

## ✅ COMPLETED: `sd_peft.py` Refactoring (Phase 1-2)

> [!NOTE]
> Phase 1-2 of the PEFT refactoring is complete. `sd_peft.py` has been reduced from 927 lines to 50 lines.

**Completed Architecture:**

```
OLD (monolithic):                     NEW (modular):
┌─────────────────┐                   ┌─────────────────┐
│   sd_peft.py    │                   │   sd_peft.py    │
│   (927 lines)   │                   │   (50 lines)    │
│   SDPeftTrainer │                   │   thin wrapper  │
└────────┬────────┘                   └────────┬────────┘
         │ inherits                            │ imports
         ▼                                     ▼
┌─────────────────┐                   ┌─────────────────────────────┐
│  sdxl_peft.py   │                   │ library/strategies/         │
│  (254 lines)    │                   │   peft_strategy_base.py     │ (16 methods)
│  SDXLPeftTrainer│                   │   peft_strategy_sd.py       │ (21 methods)
└─────────────────┘                   └─────────────┬───────────────┘
                                                    │ imports
                                                    ▼
                                      ┌─────────────────────────────┐
                                      │ library/training/           │
                                      │   peft_trainer.py           │ (train function)
                                      │   peft_common.py            │ (8 utility funcs)
                                      └─────────────────────────────┘
```

**Completed Steps:**

| Step | Description                            | Status |
| ---- | -------------------------------------- | ------ |
| 1    | Extract `train()` to `peft_trainer.py` | ✅     |
| 2    | Create strategy ABC interfaces         | ✅     |
| 3    | Create SD strategy implementations     | ✅     |
| 4    | Extract logging/plotting utilities     | ✅     |
| 5    | Remove `SDPeftTrainer` class           | ✅     |

---

## ✅ COMPLETED: Phase 3 - SDXL Strategy & Script-Based Training Loops

**Design Decision:** Training loops live in the scripts (not shared in library). The strategy pattern handles model-specific operations.

**Completed Steps:**

| Step | Description                                              | Status  |
| ---- | -------------------------------------------------------- | ------- |
| 1    | Create SDXL strategy (`peft_strategy_sdxl.py`)           | ✅ Done |
| 2    | Move `train()` from `peft_trainer.py` into `sd_peft.py`  | ✅ Done |
| 3    | Extract `init_timestep_sampler()` to `peft_common.py`    | ✅ Done |
| 4    | Extract `create_training_metadata()` to `peft_common.py` | ✅ Done |
| 5    | Extract `setup_live_plotter()` to `peft_common.py`       | ✅ Done |
| 6    | Create `sdxl_peft.py` train() using SDXL strategy        | ✅ Done |
| 7    | Review/cleanup `peft_trainer.py`                         | ✅ Done |

---

## ✅ COMPLETED: Phase 4 - Training Loop Cleanup

**Goal:** Extract shared setup code from `sd_peft.py` and `sdxl_peft.py` into `peft_common.py`.

**Completed Extractions:**

| Function                            | Lines Saved (per script) | Description                         |
| ----------------------------------- | ------------------------ | ----------------------------------- |
| `prepare_datasets()`                | ~47                      | Dataset group creation & validation |
| `calculate_initial_step()`          | ~42                      | Resume step/epoch calculation       |
| `parse_dynamic_timestep_schedule()` | ~10                      | Dynamic timestep range parsing      |
| `register_network_state_hooks()`    | ~40                      | Network-only checkpointing hooks    |

**Total reduction:** ~140 lines per script (from 1173 to ~1035 lines)

**All extractions in `peft_common.py`:**

- `prepare_datasets()` - Dataset preparation, blueprint generation, validation
- `calculate_initial_step()` - Resume/initial step calculation
- `parse_dynamic_timestep_schedule()` - Dynamic timestep schedule parsing
- `register_network_state_hooks()` - Save/load hooks for network-only checkpointing
- `init_timestep_sampler()` - Handles 5+ sampler types (~100 lines)
- `create_training_metadata()` - Creates 65+ metadata keys (~230 lines)
- `setup_live_plotter()` + `get_plotter_settings()` - Live plotter setup (~140 lines)
- `generate_step_logs()` - Step logging for training progress
- `step_logging()`, `epoch_logging()` - Accelerator logging utilities

> [!NOTE]
> Smoke tests added for all extracted functions. Run `pytest tests/unit/test_peft_scripts_smoke.py -v` to verify.

---

## ✅ COMPLETED: SAI Model Spec Consolidation

**Issue:** Three duplicate `get_sai_model_spec` functions existed across the codebase.

**Solution:**

- Updated `peft_strategy_sd.py` and `peft_strategy_sdxl.py` to use `get_sai_model_spec_from_config()`
- Removed legacy functions from `checkpointing.py`:
  - `get_sai_model_spec()` (argparse-based, ~70 lines)
  - `get_sai_model_spec_dataclass()` (unused, ~60 lines)
- **Canonical function:** `library.utils.sai_model_spec.get_sai_model_spec_from_config()`

**Result:** ~135 lines of duplicate code removed from `checkpointing.py`.

---

## ✅ COMPLETED: HuggingFace Upload Fix

**Issue:** The Hydra migration broke HuggingFace upload functionality in checkpointing functions. The condition `if args.huggingface_repo_id is not None` was incorrectly changed to `if saving_config.resume is not None` and upload calls were stubbed with `pass`.

**Solution:**

- Added `hf_config: Optional[HuggingFaceConfig] = None` parameter to 9 checkpointing functions
- Restored proper upload logic for both model checkpoints and training state
- Affected files: `checkpointing.py`, `sdxl_checkpointing.py`

**Note:** Callers must now pass `hf_config=cfg.huggingface` to enable uploads.

---

## Testing Suite

### Current Status (2025-12-23)

**Infrastructure:** ✅ Complete (pytest, fixtures, coverage)

**Test Categories:**

| Category                | Description                                     | Status         |
| ----------------------- | ----------------------------------------------- | -------------- |
| **Unit Tests (Pure)**   | Test isolated functions with no/minimal mocking | ✅ Complete    |
| **Unit Tests (Mocked)** | Test functions with mocked dependencies         | ✅ In Progress |
| **Integration Tests**   | Test multiple components working together       | 🔜 Future      |

### Completed Unit Tests (897 tests)

- **Configuration** (28 tests) - validation, dataclasses, type safety
- **Performance & Checkpointing** (49 tests) - training utilities, checkpointing logic
- **Diffusion & Noise** (38 tests) - diffusion utilities, noise generation
- **Data Utilities** (56 tests) - dataset structures, image utils, dataset.py (register_image, cache_latents, cacheability checks, shuffle, get_image_size)
- **Model Utilities** (43 tests) - `model_util.py` (shave_segments, is_safetensors, config creation, renew paths, conv_attn, controlnet_map)
- **Network Utils** - LoRA state dicts, merging, block LR parsing (42 tests in test_networks_lora.py)
- **Format Utils** - JXL parsing, Safetensors I/O
- **Loss Functions** - SNR weighting, v-pred logic, EDM2 components
- **Pipelines** - Prompt attention parsing, token padding (SD + SDXL)
- **Strategy Classes** (77 tests) - `strategy_base.py`, `strategy_sd.py`, `strategy_sdxl.py` - singleton patterns, tokenizer loading, text encoding, NPZ caching
- **HuggingFace Utils** - API mocking for `exists_repo`, `list_dir`
- **Caching** (24 tests) - `caching.py` - latent cache validation, VAE encoding, text encoder output caching with heavy mocking
- **Trainer Utils** (27 tests) - `trainer_utils.py` - validation checks, LR logging, Accelerator preparation with heavy mocking

### Completed Unit Tests (with Heavy Mocking)

**Training Core:**

- ✅ `dataset.py` - 14 tests (bucketing + `__getitem__`: cached/disk latents, image loading, flip aug, batching)
- ✅ `training/checkpointing.py` - 23 tests (naming, metadata, utils). **BUG FIXED**: `v_parameterization` access
- ✅ `training/sdxl_checkpointing.py` - 16 tests (wrapper callbacks, SAI metadata). **BUG FIXED**: same
- ✅ `training/sample_generation.py` - 25 tests (prompt parsing, scheduler, prompts loading)
- ✅ `training/model_prep.py` - 7 tests (accelerator patching, unet modules)
- ✅ `training/sdxl_model_prep.py` - 10 tests (load_target_model, diffusers/ckpt handling)

**Model Utilities:**

- ✅ `model_util.py` - 64 tests (pure funcs + conversion utilities)
- ✅ `sdxl_model_util.py` - 27 tests (embeddings, conversion maps, state dict conversion)

**Optimization Modules:**

- ✅ `performance/custom_offloading_utils.py` - 53 tests (utils + Offloader classes)
- ✅ `performance/fp8_optimization_utils.py` - 22 tests (quantization + monkey patching)
- ✅ `performance/deepspeed_utils.py` - 17 tests (config, plugin creation)

**Network Utilities:**

- ✅ `networks/lora.py` - 42 tests (block LR, dims/alphas parsing)
- ✅ `networks/lora_diffusers.py` - 20 tests (conversion maps)
- ✅ `networks/lora_utils.py` - 8 tests (filtering, merging)

**Pipelines:**

- ✅ `pipelines/lpw_stable_diffusion.py` - 31 tests (prompt attention parsing, token padding)
- ✅ `pipelines/sdxl_lpw_stable_diffusion.py` - Same (shared test file)

**Strategies:**

- ✅ `strategies/strategy_base.py` - 74 tests (TokenizeStrategy, TextEncodingStrategy, caching)
- ✅ `strategies/strategy_sd.py` - 29 tests (SD1.5/2.0 tokenize, encoding, latent caching)
- ✅ `strategies/strategy_sdxl.py` - 36 tests (SDXL dual tokenizers, dual encoders, pool workaround)

---

### Integration Tests (Remaining)

These require real models, GPU access, or full component initialization:

| Category               | Items                                                      | Priority |
| ---------------------- | ---------------------------------------------------------- | -------- |
| **Checkpoint I/O**     | `load_models_from_*`, `save_*_checkpoint`                  | High     |
| **Network Classes**    | `LoRANetwork.apply_to()`, `create_network()`               | Medium   |
| **Sample Generation**  | `sample_images_common`, inference pipeline                 | Medium   |
| **Full Training Loop** | Config → Trainer → Step                                    | High     |
| **Pipelines**          | `StableDiffusionLongPromptWeightingPipeline` class methods | Low      |
| **Multi-GPU**          | DeepSpeed/FSDP distributed                                 | Low      |

**Model implementations (integration only):**

- `models/original_unet.py`, `models/sdxl_original_unet.py`

**Skipped (No Tests Needed):**

- `networks/oft.py`, `networks/dylora.py` (rarely used)

---

### CI/CD Setup (Future)

- GitHub Actions workflow for pytest
- Coverage reporting and tracking
- Pre-commit hooks for running tests

---

## Configuration Refactoring

- [ ] **Config Validation Edge Cases**: Test `prepare_config(cfg)` and `validate_config(cfg)` for dataset-related conflicts
- [x] ~~**Consolidate Learning Rate Configs**~~: Documented with improved comments. Future work: unify all TE LR fields into single list-based config
- [x] ~~**Type Safety**~~: Added documentation for `text_encoder_lr` Any type (OmegaConf limitation)
- [x] ~~**Dataclass Reorganization**~~: Consolidated `no_half_vae` to `PerformanceConfig` (removed from `SDXLConfig`)
- [x] ~~**Base/SD Separation**~~: Complete. Train loops intentionally remain in scripts (allows model-specific variations). All model-specific logic delegated via strategy pattern in `peft_strategy_*.py`.

> [!NOTE] > **Future LR Config Improvement**: Consider unifying `text_encoder_lr`, `learning_rate_te1/te2`, and `learning_rate_te` into a single list-based field (e.g., `text_encoder_lrs: List[float]`) that works for models with any number of text encoders.

---

## Testability Improvements

- **Split `prepare_accelerator`** - Separate config computation from side effects
- **Explicit step 0 validation** - Clarify `calculate_val_loss_check` behavior at step 0

---

## Code Quality TODOs

- Resolve duplicate settings in configs/dataclasses
- Timestep sampling needs proper reimplementation (currently hacked into training scripts)
- Clean integration for external `live_plotter`
- Dataset and bucketing decouple in code?

### Library Module Naming Clarification

> [!NOTE]
> The library modules **are properly separated by model type**, but naming can be confusing.

**Architecture Summary:**

```
┌─────────────────────────────────────────────────────────────────┐
│                 Truly Base Modules (Model-Agnostic)             │
├─────────────────────────────────────────────────────────────────┤
│ training/diffusion.py, optimizer.py, noise_utils.py             │
│ training/trainer_utils.py, losses/*, data/dataset.py            │
│ strategies/strategy_base.py, utils/*, performance/*             │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   SD1.5/2 Modules           │     │   SDXL Modules              │
├─────────────────────────────┤     ├─────────────────────────────┤
│ training/model_prep.py      │     │ training/sdxl_model_prep.py │
│ training/checkpointing.py   │     │ training/sdxl_checkpointing │
│ training/sample_generation  │     │ training/sdxl_sample_gen    │
│ strategies/strategy_sd.py   │     │ strategies/strategy_sdxl.py │
│ models/model_util.py        │     │ models/sdxl_model_util.py   │
│ models/original_unet.py     │     │ models/sdxl_original_unet   │
└─────────────────────────────┘     └─────────────────────────────┘
```

**Confusing Names:**

- `checkpointing.py` → Sounds generic, but is **SD1.5/2-specific** (SDXL uses `sdxl_checkpointing.py`)
- `model_prep.py` → **SD1.5/2-specific** (SDXL uses `sdxl_model_prep.py`)
- `sample_generation.py` → **SD1.5/2-specific** (SDXL uses `sdxl_sample_generation.py`)
- `strategy_sd.py` → Handles both SD1.5 AND SD2 via `v2: bool` flag

**Potential Improvement:** Rename to `sd_checkpointing.py`, `sd_model_prep.py`, `sd_sample_generation.py` for clarity.

### Investigation Findings (2025-12-24)

These modules contain **mixed generic + SD-specific code**:

| Module                 | Generic Functions (used by SDXL)                                                                                                                                                      | SD-Specific Functions                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `checkpointing.py`     | `precalculate_safetensors_hashes`, `get_git_revision_hash`, `model_hash`, `calculate_sha256`, `resume_from_local_or_hf_if_specified`, `save_and_remove_state_*`, `*_common` functions | `save_sd_model_on_*` (non-common)         |
| `model_prep.py`        | `replace_unet_modules`, `patch_accelerator_for_fp16_training`, `set_padding_mode_for_vae_conv2d_modules`                                                                              | `load_target_model`, `_load_target_model` |
| `sample_generation.py` | `sample_images_check`, `sample_images_common`                                                                                                                                         | `sample_images`                           |

**Conclusion:** Cannot simply rename to `sd_*` - need to **split** each module into:

- `<module>_utils.py` or keep in base (generic functions)
- `sd_<module>.py` (SD-specific functions)

**Verified:** SDXL strategy correctly uses `sdxl_sample_generation.sample_images` via the strategy pattern.

### ✅ COMPLETED: Checkpointing Module Split (2025-12-24)

- Created `sd_checkpointing.py` with SD-specific save functions
- `checkpointing.py` now contains only generic utilities + `*_common` functions
- Updated `sd_finetune.py` to import from `sd_checkpointing.py`

### ✅ COMPLETED: Model Prep Module Split (2025-12-24)

- Created `sd_model_prep.py` with SD-specific `load_target_model` functions
- `model_prep.py` now contains only generic utilities used by both SD and SDXL
- Updated SD scripts, strategies, and tools to import from `sd_model_prep.py`

### ✅ COMPLETED: Sample Generation Module Split (2025-12-24)

- Created `sd_sample_generation.py` with SD-specific `sample_images` wrapper
- `sample_generation.py` now contains only generic utilities
- Mirrors existing `sdxl_sample_generation.py` pattern

## **All 3 mixed modules now split:** `checkpointing`, `model_prep`, `sample_generation` ✅

## ✅ COMPLETED: Text Encoder Utility Consolidation

**Issue:** `get_hidden_states_sdxl()` and `pool_workaround()` were duplicated between `text_encoder_util.py` and `strategy_sdxl.py`.

**Solution:**

- Made `SdxlTextEncodingStrategy._pool_workaround()` delegate to `pool_workaround()` from `text_encoder_util.py`
- Made `SdxlTextEncodingStrategy._get_hidden_states_sdxl()` call the shared utility
- Strategy wrapper handles: deriving `max_token_length` from input shape, device movement

**Result:** ~60 lines of duplicate code removed from `strategy_sdxl.py`.
