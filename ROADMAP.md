# Project Roadmap & Future Ideas

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
- Investigate naming scheme and separation of concerns for backend modules vs training scripts
- Timestep sampling needs proper reimplementation (currently hacked into training scripts)
- Clean integration for external `live_plotter`
- Dataset and bucketing decouple in code?

---

## Code Duplication (Future Consolidation)

Logic duplication between `text_encoder_util.py` and strategy classes:

- `get_hidden_states_sdxl()` in `text_encoder_util.py` (used by caching.py, sdxl_peft.py)
- `SdxlTextEncodingStrategy._get_hidden_states_sdxl()` in `strategy_sdxl.py`

Consider consolidating: make `caching.py` use the strategy, or move shared logic to common utility.
