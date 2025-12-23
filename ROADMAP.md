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

### Completed Unit Tests (690+ tests)

- **Configuration** (28 tests) - validation, dataclasses, type safety
- **Optimization & Checkpointing** (49 tests) - training utilities, checkpointing logic
- **Diffusion & Noise** (38 tests) - diffusion utilities, noise generation
- **Data Utilities** (56 tests) - dataset structures, image utils, dataset.py (register_image, cache_latents, cacheability checks, shuffle)
- **Network Utils** - LoRA state dicts, merging, block LR parsing, conversion maps
- **Format Utils** - JXL parsing, Safetensors I/O
- **Loss Functions** - SNR weighting, v-pred logic, EDM2 components
- **Pipelines** - Prompt attention parsing, token padding (SD + SDXL)
- **Strategy Classes** (77 tests) - `strategy_base.py`, `strategy_sd.py`, `strategy_sdxl.py` - singleton patterns, tokenizer loading, text encoding, NPZ caching
- **HuggingFace Utils** - API mocking for `exists_repo`, `list_dir`
- **Caching** (24 tests) - `caching.py` - latent cache validation, VAE encoding, text encoder output caching with heavy mocking
- **Trainer Utils** (27 tests) - `trainer_utils.py` - validation checks, LR logging, Accelerator preparation with heavy mocking

### Next Phase: Unit Tests with Heavy Mocking

These modules still require substantial mocked dependencies:

**Training Core:**

- `dataset.py` - `__getitem__`, `make_buckets` full flow - requires complex bucket/latent setup (partial coverage done)

**Model Utilities:**

- `model_util.py` / `sdxl_model_util.py` - Model loading/saving - requires architecture mocks
- `training/sdxl_model_prep.py` - `load_target_model` - require Accelerator and checkpoint loading
- `training/sdxl_checkpointing.py` - Save utilities - require full SDXL models

**Optimization Modules:**

- `optimizations/custom_offloading_utils.py` - Require GPU streams and thread pools
- `optimizations/deepspeed_utils.py` - Require DeepSpeed and distributed context
- `optimizations/fp8_optimization_utils.py` - Require full model state dicts

**Sample Generation:**

- `training/sample_generation.py` - `sample_images_common`, `sample_images_inference` - require full pipeline mocks

**Network Classes:**

- `networks/lora.py` - `LoRANetwork`, `create_network()` - require UNet/TextEncoder mocks
- `networks/oft.py` - `OFTNetwork` - require recursive module mocks
- `networks/dylora.py` - `DyLoRANetwork` - require dynamic module switching mocks
- `networks/lora_diffusers.py` - `LoRANetwork` class, `merge_lora_weights()` - require Diffusers model mocks

Detailed view:

- `dataset.py` - `__getitem__`, `make_buckets` full flow in dataset context - requires complex bucket/latent setup (partial coverage done)
- `data_structures.py` - `BucketManager.make_buckets()` now tested with mocked model_util
- `model_util.py` / `sdxl_model_util.py` - Model loading/saving - requires filesystem and model architecture mocks
- `networks/lora.py` - Network creation/injection - requires base model mocks
- `networks/oft.py` - OFT Network implementation - requires base model and recursive module mocks
- `networks/dylora.py` - DyLoRA Network implementation - requires base model and dynamic module switching mocks
- `networks/hypernetwork.py` - Hypernetwork implementation - requires base model mocks
- `optimizations/custom_offloading_utils.py` - `Offloader`, `ModelOffloader`, `swap_weight_devices_cuda` - require GPU streams and thread pools
- `optimizations/deepspeed_utils.py` - `prepare_deepspeed_plugin`, `prepare_deepspeed_model` - require DeepSpeed import and distributed context
- `optimizations/fp8_optimization_utils.py` - `optimize_state_dict_with_fp8`, `load_safetensors_with_fp8_optimization`, `apply_fp8_monkey_patch` - require full model state dicts
- `training/sample_generation.py` - `sample_images_common`, `sample_images_inference` - requires full pipeline (VAE, UNet, Tokenizer) mocks
- `models/original_unet.py` - `FlashAttentionFunction`, `TimestepEmbedding`, `Timesteps`, all `*Block2D` classes, `UNet2DConditionModel` - require GPU/autograd context
- `models/sdxl_original_unet.py` - `FlashAttentionFunction`, `GroupNorm32`, `ResnetBlock2D`, `CrossAttention`, `SdxlUNet2DConditionModel` - require SDXL architecture
- `models/sdxl_original_control_net.py` - `ControlNetConditioningEmbedding`, `SdxlControlNet.forward`, `SdxlControlledUNet` - require UNet and forward passes
- `training/sdxl_model_prep.py` - `load_target_model`, `_load_target_model` - require Accelerator and SDXL checkpoint loading
- `training/sdxl_checkpointing.py` - `save_sd_model_on_train_end`, `save_sd_model_on_epoch_end_or_stepwise` - require full SDXL models

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
