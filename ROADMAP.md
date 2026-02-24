# Project Roadmap & Future Ideas

## Architecture Overview

The codebase follows a Trainer + TrainingMode pattern where scripts are thin entry points, the Trainer orchestrates training phases, TrainingMode plugins handle mode-specific logic (PEFT vs fine-tune), and TrainingStrategy classes handle model-specific operations.

```
Scripts (thin entry points):          Library Modules:
┌─────────────────┐                   ┌─────────────────────────────┐
│   sd_peft.py    │                   │ library/strategies/         │
│   (~55 lines)   │ ───imports───────►│   base/training.py          │
└─────────────────┘                   │   sd/training.py            │
┌─────────────────┐                   │   sdxl/training.py          │
│  sdxl_peft.py   │ ───imports───────►└─────────────┬───────────────┘
│  (~55 lines)    │                                 │ uses
└─────────────────┘                   ┌─────────────┴───────────────┐
┌─────────────────┐                   │ library/training/           │
│ sdxl_finetune.py│ ───imports───────►│   runners/trainer.py        │
│  (~55 lines)    │                   │   modes/peft_mode.py        │
└─────────────────┘                   │   modes/finetune_mode.py    │
                                      │   phases/*.py               │
                                      │   checkpointing.py          │
                                      │   sample_generation.py      │
                                      └─────────────────────────────┘
```

**Module Separation (Post-Refactor):**

- `library/strategies/base/` - ABCs and shared logic (TrainingStrategy, TokenizeStrategy, etc.)
- `library/strategies/sd/` - SD1.5/2 implementations (SdTrainingStrategy, SdTokenizeStrategy)
- `library/strategies/sdxl/` - SDXL implementations (SdxlTrainingStrategy, SdxlTokenizeStrategy)
- `library/training/runners/` - Trainer orchestration
- `library/training/modes/` - TrainingMode plugins (PeftMode, FineTuneMode)
- `library/training/phases/` - Phase functions (setup, caching, model_prep, optimizer, training_loop)
- `library/training/` - Model-agnostic utilities (checkpointing.py, sample_generation.py)

---

## Testing Suite

### Current Status

**Infrastructure:** ✅ Complete (pytest, fixtures, coverage)

| Category                | Status         |
| ----------------------- | -------------- |
| **Unit Tests (Pure)**   | ✅ Complete    |
| **Unit Tests (Mocked)** | ✅ In Progress |
| **Integration Tests**   | 🔜 Future      |

**Completed:** 1017+ unit tests across configuration, training, data, strategies, networks, losses, pipelines, training modes.

### Integration Tests (Remaining)

| Category                   | Items                                         | Priority    |
| -------------------------- | --------------------------------------------- | ----------- |
| ~~**Checkpoint I/O**~~     | ~~`load_models_from_*`, `save_*_checkpoint`~~ | ~~High~~ ✅ |
| **Adapter Classes**        | `LoRAAdapter.apply_to()`, `create_adapter()`  | Medium      |
| **Sample Generation**      | `sample_images_common`, inference pipeline    | Medium      |
| ~~**Full Training Loop**~~ | ~~Config → Trainer → Step~~                   | ~~High~~ ✅ |

---

## Deferred Training Features

Features intentionally excluded from the Phase 2B `FineTuneMode` migration. Currently fail-fast with `NotImplementedError` in `FineTuneMode.build_optimizer_params()` to prevent silent behavior differences.

- [ ] **Block-level learning rates** — Per-UNet-block LR grouping (legacy `get_block_params_to_optimize()`). Should be implemented as a mode-agnostic optimizer-group feature, not mode-specific.
- [ ] **Pattern-based optimizer groups** — Regex/glob-based param grouping for fine-grained LR control. Same mode-agnostic approach as block LR.
- [ ] **Fused optimizer groups** — Multi-optimizer support with `fused_backward_pass` (per-parameter backward hooks). Complex multi-optimizer logic from legacy `sdxl_finetune.py`.
- [ ] **PEFT module/param breakdown** — Per-component (unet, TE) module and parameter counts in training diagnostics for PEFT mode. Requires an optional adapter protocol method (`get_diagnostics_components()`) that each adapter type implements to report its own per-component allocation. `PeftMode` already has the `hasattr` hook ready — just needs adapter-side implementations. Deferred because adapter internals vary (LoRA, LyCORIS, OFT) and LyCORIS is still external.

> [!IMPORTANT]
> Keep fused/block/pattern paths explicitly fail-fast (as planned), so they don't silently behave differently. Only remove the guards when proper implementations are added.

---

## Configuration Refactoring

### Active TODOs

- [ ] Config Validation Edge Cases: Test `prepare_config()` and `validate_config()` for dataset conflicts
- [ ] Work on validation in general to figure out a system for catching invalid configs, might need to be post testing
- [ ] **Sampling config error**: Add error in `config_validation.py` when both `sample_every_n_steps` and `sample_every_n_epochs` are set (epoch-based takes precedence, step-based silently ignored)

---

## Code Quality TODOs

- [ ] Timestep sampling needs proper reimplementation (currently hacked into training scripts)
- [ ] Clean integration for external `live_plotter`, possible rework at later time with dedicated logging setup
- [ ] **`edm2_loss_utils.py` Config Cleanup** (low priority, not critical component)
  - 15+ flat config fields with absurdly long names (`edm2_loss_weighting_importance_weighting_safety_override`)
  - Should extract to dedicated `Edm2LossConfig` sub-dataclass
  - Mutates config directly (`loss_config.debiased_estimation_loss = False`)
- [ ] **`training_plots.py`** - Functions access multiple sub-configs (`cfg.output.saving`, `cfg.output.logging`, `cfg.timestep`) - acceptable for orchestration functions but could be cleaner
- [ ] **Consolidate `init_ipex()` calls** (low priority) - During refactoring, `init_ipex()` was copied to all split-out library modules. Original pattern: only training scripts + `model_util.py` need it. Remove from other utility modules like `torch_utils.py`. **INIT_IPEX MIGHT BE USELESS POST TORCH 2.6.0**
- [ ] **SD Data Pipeline Support** (low priority) - Update `sd/training.py` to consume new batch format from `TrainingDataset`. Expects `batch["input_ids_list"]` / `batch["text_encoder_outputs_list"]` but new pipeline uses dict format. See AUDIT/AUDIT_PHASE_2.md.
- [ ] **Delete legacy training wrappers** (after legacy script deprecation) - Once `*_finetune.py` and `*_textual_inversion.py` scripts are migrated to new data pipeline, delete:
  - `library/training/sd_sample_generation.py`
  - `library/training/sdxl_sample_generation.py`
  - `library/training/sd_checkpointing.py`
  - `library/training/sdxl_checkpointing.py`
  - Strategies now call `sample_images_common()` directly; checkpointing logic can be inlined into strategies when legacy scripts are removed.
- [ ] **Refactor `register_adapter_state_hooks`** (low priority) - Return a structured object `{"epoch": int, "step": int}` instead of closure + side-effects for cleaner data flow. See AUDIT/2_AUDIT_RESUME_BEHAVIOR.md recommendation #3.
- [ ] **Remove `[DEBUG]` log statements in `sdxl/training.py`** (low priority) - Several `logger.info(f"[DEBUG] ...")` calls left in `_get_text_cond`. Either remove or change to `logger.debug()`. Alternatively, will work with logging config settings.

---

## Training Mode Extensibility

### Problem

All phase functions in `library/training/phases/` are typed as `trainer: Trainer`. The `TrainingMode` protocol and `PeftMode` implementation enable pluggable training modes without changing phase code.

### What Actually Differs (PEFT vs Full Fine-Tuning)

| Concern                     | PEFT (LoRA/Adapter)                          | Full Fine-Tuning                     |
| --------------------------- | -------------------------------------------- | ------------------------------------ |
| **Trainable params**        | `adapter.parameters()` + optionally TE       | `unet.parameters()` + optionally TE  |
| **Model prep**              | Create adapter, `apply_to()`, freeze base    | `requires_grad_(True)` on base model |
| **Checkpoint save**         | Adapter state dict                           | Full model state dict                |
| **`accelerator.prepare()`** | Wraps adapter                                | Wraps UNet directly                  |
| **Caching**                 | Identical                                    | Identical                            |
| **Training loop**           | Identical (strategy handles `process_batch`) | Identical                            |

### Phase Reusability

- `caching.py` — **100% shareable**
- `training_loop.py` — **~95% shareable** (checkpoint calls go through `trainer.save_checkpoint()`)
- `optimizer.py` — **~80% shareable** (param groups and `accelerator.prepare()` wrapping differ)
- `model_prep.py` — **PEFT-specific** (adapter creation is inherently a PEFT concept)

### Design Options (Open)

**Option A: One Trainer + `TrainingMode` plugin** (Lightning/Transformers style)

- Rename `PeftTrainer` → `Trainer`, add `trainer.mode: TrainingMode`
- `TrainingMode` protocol has ~4 methods: `prepare_models`, `trainable_params`, `prepare_with_accelerator`, `save_checkpoint`
- Phases call `trainer.mode.*` for divergent operations
- Two axes: `strategies` = model family (SDXL/SD/Flux), `mode` = training mode (PEFT/fine-tune)
- Pro: One class to understand, config-driven. Con: Extra indirection through `mode.*`

**Option B: Multiple trainer classes + `TrainerProtocol`**

- Keep `PeftTrainer`, add `FineTuneTrainer`, both satisfy a shared protocol
- Phases type against the protocol
- Pro: Each trainer is self-contained. Con: Large protocol surface, potential duplication

> [!IMPORTANT]
> Key constraint: this repo prioritizes readability and ease of modification over abstraction. Whatever pattern is chosen must feel intuitive when adding new features. Decision deferred until full fine-tuning support is actively being built.

### Phase 0: SDXL Decoupling (✅ Complete)

Removed all direct SDXL imports from `caching.py` and `training_loop.py`. Phase files now call strategy factory methods (`create_latent_caching_strategy`, `create_te_caching_strategy`, `tokenize_captions`, `encode_te_outputs_in_memory`) on `TrainingStrategy`.

- [x] Add 4 abstract methods to `TrainingStrategy` base class
- [x] Implement in `SdxlTrainingStrategy`
- [x] Add `NotImplementedError` stubs in `SdTrainingStrategy` (SD uses old pipeline)
- [x] Remove SDXL imports from `caching.py` and `training_loop.py`
- [ ] **Future:** Implement SD strategy methods when SD is migrated to new CachingEngine pipeline

### Phase 1: TrainingMode Extraction (✅ Complete)

Refactored `PeftTrainer` and phase files to use a pluggable `TrainingMode` protocol. Extracted PEFT-specific logic into `PeftMode`.

- [x] Define `TrainingMode` protocol
- [x] Extract `PeftMode` implementation
- [x] Update phases to delegate to mode hooks

### Phase 2A: Adapter-Neutral Shared Flow (✅ Complete)

Neutralized all remaining adapter-specific assumptions in shared code.

- [x] `PeftTrainer` → `Trainer` rename (hard cut)
- [x] 4 new mode hooks (`on_step_start`, `get_trainable_params`, `set_eval`, `set_train`)
- [x] `_primary_trainable` field + `trainable_model` property (distinct from `_grad_sync_handle` wrapper)
- [x] Strategy renames: `all_reduce_adapter` → `all_reduce_trainable`, `post_process_adapter` → `post_process_trainable`
- [x] `adapter` param → `trainable_model` in 6 base strategy methods
- [x] PEFT metadata guarded (keys omitted when `cfg.peft` absent)
- [x] `_grad_sync_handle` assertion before training loop
- [x] Tracker name `adapter_train` → `training`

### Phase 2B: FineTuneMode + SDXL Migration (Future)

- [x] **Guard `set_multiplier` in strategies** — `trainable_model.set_multiplier()` in `sdxl/training.py` and `sd/training.py` guarded with `hasattr(trainable_model, "set_multiplier")`. Non-adapter trainables skip differential output preservation silently.
- [x] Create `library/training/modes/finetune_mode.py`
- [x] Migrate `scripts/sdxl_finetune.py` to thin entrypoint
- [x] Add unit + integration tests

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

## Validation Refactoring

See `AUDIT/5_Strategy_Pattern_Boundaries.md` for full context.

- [ ] **Move validation loop to Trainer** - `SdxlTrainingStrategy.calculate_val_loss()` currently owns the entire validation loop (dataloader iteration, tqdm, RNG state). This is orchestration that belongs in the Trainer.
  - Refactor: Extract loop to `Trainer` (or `phases/validation.py`)
  - Reduce strategy method to `process_val_batch(batch)` for single-batch loss computation
- [ ] **Fix `calculate_val_loss` return type mismatch** - SD returns `tuple[float | None, float | None, dict | None]` (3 values) but base ABC declares `tuple[float | None, float | None]` (2 values). Resolve when validation is reworked.
- [ ] **Add missing ABC definitions** - `TrainingStrategy` ABC is missing `process_batch` and `calculate_val_loss`/`process_val_batch`. Trainer calls them dynamically, bypassing type safety.
- [ ] **Rename `_log_training_info`** - Currently initializes noise scheduler, plotters, trackers (setup concerns), not just logging. Rename to `_finalize_setup` or move initialization to proper phase.

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
  - Also: Add YAML support for `sample_prompts` for consistency with the rest of the config system (currently only txt, .toml, .json)

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
  - Maybe fp8 for te output storage, needs tests
- [ ] **Smarter resource tracking/management** - This helps with training and also with inference, for example falling back to tiled vae when it would hit resource contraints and such. See `docs_design/resource_monitor_plan.md`

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
