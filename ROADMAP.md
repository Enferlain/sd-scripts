# Project Roadmap & Future Ideas

## Testing Suite

### Current Status (2025-12-22)

**Infrastructure:** ✅ Complete (pytest, fixtures, coverage)

**Test Categories:**

| Category                | Description                                     | Status         |
| ----------------------- | ----------------------------------------------- | -------------- |
| **Unit Tests (Pure)**   | Test isolated functions with no/minimal mocking | ✅ Complete    |
| **Unit Tests (Mocked)** | Test functions with mocked dependencies         | ✅ In Progress |
| **Integration Tests**   | Test multiple components working together       | 🔜 Future      |

### Completed Unit Tests (660+ tests)

- **Configuration** (28 tests) - validation, dataclasses, type safety
- **Optimization & Checkpointing** (49 tests) - training utilities, checkpointing logic
- **Diffusion & Noise** (38 tests) - diffusion utilities, noise generation
- **Data Utilities** (34 tests) - dataset structures, image utils
- **Network Utils** - LoRA state dicts, merging, block LR parsing, conversion maps
- **Format Utils** - JXL parsing, Safetensors I/O
- **Loss Functions** - SNR weighting, v-pred logic, EDM2 components
- **Pipelines** - Prompt attention parsing, token padding (SD + SDXL)
- **HuggingFace Utils** - API mocking for `exists_repo`, `list_dir`

### Next Phase: Unit Tests with Heavy Mocking

These modules require substantial mocked dependencies (Accelerator, VAE, tokenizers):

**Strategy Classes:**

- `strategies/*` - Tokenization/Encoding orchestration - require tokenizer/model mocks

**Training Core:**

- `trainer_utils.py` - `init_trackers()`, `determine_grad_sync_context()` - need Accelerator mocks
- `caching.py` - `cache_batch_latents()`, `cache_batch_text_encoder_outputs()` - need VAE/encoder mocks
- `dataset.py` - `cache_latents()`, `register_image()`, `__getitem__` - requires filesystem and VAE mocks

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
