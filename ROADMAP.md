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
                                      │   peft_common.py (utilities)│
                                      │   checkpointing.py (generic)│
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
- [ ] **`sd_textual_inversion.py` Config Migration** (partial)
  - Renamed `config` → `cfg` throughout script
  - Updated config access patterns for new nested structure:
    - `cfg.dataset` → `cfg.data`
    - `cfg.saving/sampling/logging/huggingface/metadata` → `cfg.output.*`
    - `cfg.performance.xformers/sdpa/mem_eff_attn` → `cfg.performance.attention.*`
    - `cfg.loss.min_snr_gamma/debiased_estimation_loss/etc` → `cfg.loss.snr.*`
    - `cfg.masked_loss` → `cfg.loss.masked`
    - `training_config.gradient_checkpointing` → `cfg.performance.memory.gradient_checkpointing`
    - `training_config.full_fp16` → `cfg.performance.precision.full_fp16`
  - Added `tools/scan_config_patterns.py` utility for auditing config access
  - **Remaining**: `model_config.v2` needs to be derived from `model_type` or handled via strategy

### Completed

- **Learning Rate Consolidation**: Unified `unet_lr`, `text_encoder_lr`, `learning_rate_te1/te2`, `block_lr` into `optimizer.learning_rates`.
- **Config Key Rename**: `cfg.network` → `cfg.peft` across all scripts and library modules
- **Network → Adapter Rename**: Renamed folder, classes, functions, variables, and config fields from `network` to `adapter` terminology
- **Legacy Cleanup**: Removed unused `@property` aliases and fallback logic from optimizer.py
- **Train Text Encoder Options**: Consolidated via `optimizer.learning_rates` usage (implicit vs explicit)
- **Schema 1 Refactor**: Unified configuration schema for PEFT/Fine-tuning scripts
- **Data Config Restructuring**: Merged `DatasetConfig` + `BucketsConfig` into `DataConfig` with 5 nested sub-configs (source, preprocessing, caption, bucketing, caching)

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
- [ ] Apply config pattern to `sd_textual_inversion.py` (use `cfg.*` in `train()`, keep typed params in helper methods that are called externally)
- [ ] get rid of lazy imports, move to top for transparency
- [ ] **PEFT Strategy Deduplication**: 4 methods identical between `peft_strategy_sd.py` and `peft_strategy_sdxl.py` (`get_noise_scheduler`, `encode_images_to_latents`, `shift_scale_latents`, `post_process_loss`) - should move to shared base class
- [ ] **PEFT Strategy Internal Dedup**: `process_batch` and `process_val_batch` share ~45 lines of identical latent/text encoding setup - extract to helper method

---

## Testability Improvements

- **Split `prepare_accelerator`** - Separate config computation from side effects
- **Explicit step 0 validation** - Clarify `calculate_val_loss_check` behavior at step 0

---

## CI/CD Setup (Future)

- GitHub Actions workflow for pytest
- Coverage reporting and tracking
- Pre-commit hooks for running tests

---

## Future Ideas

- Library reorganization based on cleaner categories
- May extend to functions across scattered files
- **BLAKE3 Model Hashing**: Add optional fast model hashing using BLAKE3 (compatible with CivitAI AutoV3). Currently SHA256 is disabled due to ~1min overhead for 6GB models. BLAKE3 offers 4-8x speed improvement and multi-threading.
  - Add optional `blake3` dependency
  - Config option `metadata_hash_algorithm: "none" | "blake3" | "sha256"`
  - Compute hash after saving (stream from disk, no serialization overhead)
