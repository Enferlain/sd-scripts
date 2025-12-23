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

### Next Phase: Unit Tests with Heavy Mocking

Legend: ✅ Done | 🔶 Partial (pure funcs done, classes need mocks) | ❌ No tests

**Training Core:**

- 🔶 `dataset.py` - Pure functions done. Remaining: `__getitem__`, `make_buckets` full flow
- ✅ `training/checkpointing.py` - 23 tests (naming, metadata, utils)
- ✅ `training/sample_generation.py` - 25 tests (prompt parsing, scheduler, prompts loading). Remaining: `sample_images_common`, `sample_images_inference`
- ✅ `training/model_prep.py` - 7 tests (accelerator patching, unet modules). Remaining: `load_target_model` with Accelerator

**Model Utilities:**

- 🔶 `model_util.py` - 49 tests (pure funcs + conversion utilities). Remaining: `load_checkpoint*`, `load_models_*`
- 🔶 `sdxl_model_util.py` - 27 tests (embeddings, conversion maps, state dict conversion). Remaining: `load_models_from_sdxl_checkpoint`, `save_stable_diffusion_checkpoint`
- ❌ `training/sdxl_model_prep.py` - `load_target_model`, `_load_target_model` (Accelerator + checkpoint mocks)
- ✅ `training/sdxl_checkpointing.py` - 16 tests (wrapper callbacks, SAI metadata, model passing). **BUG FIXED**: was accessing `training_config.v_parameterization` instead of `loss_config.v_parameterization`
- ✅ `training/checkpointing.py` - 23 tests. **BUG FIXED**: Same `v_parameterization` fix applied (uses LossConfig now)

**Optimization Modules:**

- 🔶 `optimizations/custom_offloading_utils.py` - 42 tests (to_device, to_cpu, weighs_to_device, wrapper). Remaining: `Offloader`, `ModelOffloader` (GPU streams, threads)
- 🔶 `optimizations/deepspeed_utils.py` - 17 tests (prepare_deepspeed_config, plugin creation, model wrapping). Remaining: full integration with real DeepSpeed
- 🔶 `optimizations/fp8_optimization_utils.py` - 15 tests (quantization logic). Remaining: `apply_fp8_monkey_patch`, integration

**Network Classes:**

- 🔶 `networks/lora.py` - 42 tests (block LR, dims/alphas, utils). Remaining: `LoRANetwork` class, `create_network()`
- 🔶 `networks/lora_diffusers.py` - 20 tests (conversion maps). Remaining: `LoRANetwork` class, `merge_lora_weights()`
- 🔶 `networks/lora_utils.py` - 8 tests (filtering, merging). Done for pure funcs
- ❌ `networks/oft.py` - `OFTNetwork` class (recursive module mocks)
- ❌ `networks/dylora.py` - `DyLoRANetwork` class (dynamic module switching)

**Models (Integration-level, skip for unit tests):**

- ❌ `models/original_unet.py` - Full UNet implementation (GPU/autograd - integration only)
- ❌ `models/sdxl_original_unet.py` - SDXL UNet (integration only)
- ❌ `models/sdxl_original_control_net.py` - ControlNet (integration only)

### Future: Integration Tests

These require real models/GPU and cannot use mocks:

- End-to-end config → training setup validation
- Checkpoint save/load cycles
- Multi-GPU scenarios (requires `requires_gpu` marker)
- Data loading with real filesystem caching

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

---

## Code Duplication (Future Consolidation)

Logic duplication between `text_encoder_util.py` and strategy classes:

- `get_hidden_states_sdxl()` in `text_encoder_util.py` (used by caching.py, sdxl_peft.py)
- `SdxlTextEncodingStrategy._get_hidden_states_sdxl()` in `strategy_sdxl.py`

Consider consolidating: make `caching.py` use the strategy, or move shared logic to common utility.
