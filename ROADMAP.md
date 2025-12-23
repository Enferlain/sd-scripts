# Project Roadmap & Future Ideas

## 🔴 HIGH PRIORITY: `sd_peft.py` Refactoring

> [!CAUTION]
> This is the most-used training script in the repository and has accumulated significant technical debt.

**Current State:**

- `sd_peft.py` is **2,326 lines** with a monolithic `SDPeftTrainer` god class (57 methods)
- The `train()` method alone is **~1,400 lines** (lines 907-2310)
- `sdxl_peft.py` **inherits from** `sd_peft.SDPeftTrainer` and overrides ~15 methods
- This creates fragile coupling unlike `*_finetune.py` scripts which use library imports only

**Problems:**
| Issue | Impact |
|-------|--------|
| God class pattern | Hard to understand, test, or modify safely |
| 1,400-line method | Unmaintainable - mixes setup, loop, logging, saving |
| Inheritance coupling | Changes to SD break SDXL; can't evolve independently |
| Nested functions | `save_model`, `load_model_hook` defined inside `train()` - untestable |

**Target Architecture:**

```
CURRENT (fragile):                    TARGET (modular):
┌─────────────────┐                   ┌─────────────────┐
│   sd_peft.py    │                   │   sd_peft.py    │
│   (2,326 lines) │                   │   (~500 lines)  │
│   SDPeftTrainer │                   │   thin wrapper  │
└────────┬────────┘                   └────────┬────────┘
         │ inherits                            │ imports
         ▼                                     ▼
┌─────────────────┐                   ┌─────────────────┐
│  sdxl_peft.py   │                   │ library/training│
│  (254 lines)    │                   │ /peft_utils.py  │
│  SDXLPeftTrainer│                   │ (shared logic)  │
└─────────────────┘                   └────────┬────────┘
                                               │ imports
                                               ▼
                                      ┌─────────────────┐
                                      │  sdxl_peft.py   │
                                      │  (~300 lines)   │
                                      │  thin wrapper   │
                                      └─────────────────┘
```

**Proposed Refactoring Steps:**

1. Extract `train()` setup phase → `library/training/peft_setup.py`
2. Extract training loop core → `library/training/peft_loop.py`
3. Extract logging/metrics → `library/training/peft_logging.py`
4. Extract model saving → already have `checkpointing.py`, extend for networks
5. Make both `sd_peft.py` and `sdxl_peft.py` thin orchestrators like `*_finetune.py`

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
- **Optimization & Checkpointing** (49 tests) - training utilities, checkpointing logic
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

- ✅ `optimizations/custom_offloading_utils.py` - 53 tests (utils + Offloader classes)
- ✅ `optimizations/fp8_optimization_utils.py` - 22 tests (quantization + monkey patching)
- ✅ `optimizations/deepspeed_utils.py` - 17 tests (config, plugin creation)

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
- [ ] **Consolidate Learning Rate Configs**: Unify `text_encoder_lr` (List/Any in NetworkConfig) and `learning_rate_te1/te2` (floats in SDXLConfig)
- [ ] **Type Safety**: Improve type definitions for `text_encoder_lr` to avoid `Any`
- [ ] **Dataclass Reorganization**: Audit duplicated/misplaced fields (e.g., `no_half_vae` in both PerformanceConfig and SDXLConfig)
- [ ] **Base/SD Separation**: Strip base-level code from sd_peft, sd_textual_inversion, sd_finetune - currently mixing base AND sd1/2

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
│ strategies/strategy_base.py, utils/*, optimizations/*           │
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

---

## Code Duplication (Future Consolidation)

Logic duplication between `text_encoder_util.py` and strategy classes:

- `get_hidden_states_sdxl()` in `text_encoder_util.py` (used by caching.py, sdxl_peft.py)
- `SdxlTextEncodingStrategy._get_hidden_states_sdxl()` in `strategy_sdxl.py`

Consider consolidating: make `caching.py` use the strategy, or move shared logic to common utility.
