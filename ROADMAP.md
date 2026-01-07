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

---

## Code Quality TODOs

- [ ] Timestep sampling needs proper reimplementation (currently hacked into training scripts)
- [ ] Clean integration for external `live_plotter`, possible rework at later time with dedicated logging setup
- [ ] **`edm2_loss_utils.py` Config Cleanup** (low priority, not critical component)
  - 15+ flat config fields with absurdly long names (`edm2_loss_weighting_importance_weighting_safety_override`)
  - Should extract to dedicated `Edm2LossConfig` sub-dataclass
  - Mutates config directly (`loss_config.debiased_estimation_loss = False`)
- [ ] **`training_plots.py`** - Functions access multiple sub-configs (`cfg.output.saving`, `cfg.output.logging`, `cfg.timestep`) - acceptable for orchestration functions but could be cleaner
- [ ] **Consolidate `init_ipex()` calls** (low priority) - During refactoring, `init_ipex()` was copied to all split-out library modules. Original pattern: only training scripts + `model_util.py` need it. Remove from other utility modules like `torch_utils.py`.
- [ ] **SD Data Pipeline Support** (low priority) - Update `peft_strategy_sd.py` to consume new batch format from `TrainingDataset`. Expects `batch["input_ids_list"]` / `batch["text_encoder_outputs_list"]` but new pipeline uses dict format. See AUDIT/AUDIT_PHASE_2.md.

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

## CI/CD Setup

- [x] GitHub Actions workflow for pytest (`.github/workflows/tests.yml` - multiple Python/PyTorch versions)
- [x] Coverage reporting and tracking (pytest-cov + Codecov)
- [ ] Pre-commit hooks for running tests

---

## Future Ideas

- [ ] **Sample Generation Config Defaults** - Add global defaults to `SamplingConfig` for common sampling parameters:

  - `sample_width`, `sample_height` (default dimensions)
  - `sample_steps`, `sample_cfg_scale` (default inference settings)
  - `sample_negative_prompt` (global negative prompt)
  - These would serve as defaults that per-prompt overrides (in sample_prompts file) could supersede
  - Also: Add YAML support for `sample_prompts` for consistency with the rest of the config system (currently only .txt, .toml, .json)

- [ ] **Support for feather** - https://github.com/SuriyaaMM/feather

  - Feather is a high-performance emulation library that brings FP8 (E5M2 & E4M3) precision arithmetic to older GPU architectures (Ampere, Turing, Volta) that lack native hardware support. Currently only considered for inference

- [ ] **Investigate 2022-2023 backend code**

  - After cecking sd_original_unet.py we found that it referenced bugs and had workaround for said bugs from 2022-2024. The model backend might be outdated or harming performance/code quality at large. A wider audit of the backend against diffusers or original code might be necessary down the line.

### Future Improvements

- [ ] Fix zero-dimension bucket edge case for images smaller than `bucket_reso_steps`
- [ ] Config-hash cache namespace - Auto-segregate caches by config hash (`resolution`, `bucket_steps`, `model_version`) to prevent cross-config issues. See `AUDIT/AUDIT_PHASE_6.md`.
- [ ] **Large-scale dataset manifest optimization** - Current JSON manifest grows ~2KB/entry (100k images = ~200MB JSON). Options:
  - Binary format (msgpack/pickle) for faster I/O
  - Incremental manifest updates instead of full rewrite
  - Lazy loading of manifest entries
  - Sharded manifests by bucket
  - Skip manifest creation if unchanged from previous run

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

---

## Data Pipeline Rework

**Status:** 🔄 In Progress

See `DATA_PIPELINE_PLAN.md` for design, `DATA_PIPELINE_CURRENT.md` for implementation checklist.

### Completed

- [x] Created `library/data/pipeline/` package
- [x] Core dataclasses: `CacheEntry`, `Bucket`, `EpochManifest`, `DatasetManifest`
- [x] Manifest I/O: `save_dataset_manifest()`, `load_dataset_manifest()`
- [x] Engine skeleton: `CachingStrategy` interface, `CachingEngine`
- [x] DataLoader: `TrainingDataset`, `create_training_dataloader()`
- [x] Epoch prep: `prepare_epoch()`, `prepare_validation_epoch()`

---

## Large Scale Caching Architecture

**Status:** 📋 Design Phase (for 1M+ image datasets)

The current per-image caching approach works well for <100k images but faces challenges at scale:

- **NTFS/filesystem overhead**: Millisecond-level per-file overhead at high file counts
- **Too many file opens**: Training I/O becomes bottleneck

### Current Architecture (Per-Image)

```
cache/
├── img_001_sdxl_latents.safetensors  # Latent + conditioning metadata
├── img_001_sdxl_te.safetensors       # TE outputs (optional, separate file)
└── ...
```

**Pros:** Simple, incremental updates, easy debugging
**Cons:** Doesn't scale beyond ~100k files on Windows/NTFS

### Proposed Future Architecture (Sharded)

```
cache/
├── latents/
│   └── {config_hash}/              # Bucket settings hash (invalidation namespace)
│       ├── bucket_1024x1024/
│       │   ├── shard_0000.safetensors  # ~10k images per shard
│       │   └── shard_0001.safetensors
│       └── bucket_768x1024/
│           └── shard_0000.safetensors
│
├── te_outputs/
│   └── {te_config_hash}/           # Tokenizer + encoder settings hash
│       ├── shard_0000.safetensors  # Keyed by caption hash (dedup!)
│       └── shard_0001.safetensors
│
└── # Tokens: on-the-fly (default) or epoch-level file (optional)
```

### Design Principles

1. **Config-Hash Namespacing**

   - Changing bucket settings writes to a _new_ namespace (new hash directory)
   - No in-place rewriting; old cache remains until explicitly deleted
   - Same pattern as how per-image caches invalidate when settings change

2. **Caption-Hash Deduplication for TE**

   - TE outputs keyed by caption hash, not image ID
   - Same caption → same encoding (dedup across images sharing captions)
   - Hash must include: tokenizer settings, max_length, clip_skip, encoder version

3. **Independent Invalidation**

   - Latent config hash: `bucket_steps`, `base_resolution`, `no_upscale`, etc.
   - TE config hash: `tokenizer_version`, `max_token_length`, `clip_skip`, etc.
   - Change buckets → only rebuild latent shards (TE remains valid)
   - Change captions → only rebuild TE shards (latents remain valid)

4. **Tokens: On-the-Fly Default**
   - Tokenization is ~0.5ms/sequence (negligible vs GPU time)
   - Epoch-level token file available as opt-in for specific workflows
   - Avoids coupling token cache to caption augmentation scheme

### Implementation Plan

1. **Abstract Cache Backend**

   ```python
   class CacheStore(ABC):
       def save(self, key: str, data: dict[str, Tensor]) -> None: ...
       def load(self, key: str) -> dict[str, Tensor]: ...
       def exists(self, key: str) -> bool: ...

   class PerImageCacheStore(CacheStore):  # Current implementation
       ...

   class ShardedCacheStore(CacheStore):   # Future implementation
       ...
   ```

2. **Training-facing API unchanged**

   - Same `CacheData` contract
   - Same `CachingStrategy` interface
   - Backend switch via config (`cache_backend: "per_image" | "sharded"`)

3. **Threshold-based recommendation**
   - Default: per-image for <100k images
   - Recommend sharded for 100k+ or Windows/NTFS users
   - Automatic detection possible (count files, check filesystem)

### Tasks

- [ ] Design config-hash computation for latent namespace
- [ ] Design caption-hash computation for TE dedup (include all relevant settings)
- [ ] Implement `ShardedCacheStore` with bucket-based sharding
- [ ] Implement shard-level `get_slice` for efficient batch loading
- [ ] Add `cache_backend` config option
- [ ] Migration utility: per-image → sharded conversion
