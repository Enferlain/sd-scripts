# Data Pipeline Implementation Plan

This document tracks implementation progress for the data pipeline rework.
See `DATA_PIPELINE_PLAN.md` for design and `DATA_PIPELINE_OLD.md` for legacy reference.

## Status: ✅ Phase 1-4 Complete, ✅ SDXL Integration Complete

**Last Updated:** 2026-01-09

- Phase 1 (scanning): Complete ✅
- Phase 2 (caching): Complete ✅
- Phase 3 (epoch prep): Complete ✅
- Phase 4 (dataloader): Complete ✅
- PEFT Strategy Integration: Complete ✅
- SDXL PEFT Script Integration: Complete ✅
- **Cache Path Simplification:** Complete ✅ - Paths set at manifest creation
- **Manifest Persistence:** Complete ✅ - Hash + file count validation, reuse across runs
- **Smoke Test:** Passed (training + sample generation with weighted prompts)
- **Unit Tests:** 947 passed, 6 skipped (skipped tests are for deprecated SD 1.x scripts)
- **Integration Tests:** 25 passed ✅
- **TE Dimension Bugs:** Fixed:
  - In-memory TE caching: Added `.squeeze(0)` when storing per-entry outputs
  - Sample generation: Removed premature `reshape()` in `_get_hidden_states_sdxl`
  - On-the-fly tokenization: SDXL training paths now reuse the shared CLIP-family token helper shape rules

---

## Migration Overview: Legacy vs New System

### What's Being Replaced

| Legacy Component            | Location                                     | Replacement                  | Status |
| --------------------------- | -------------------------------------------- | ---------------------------- | ------ |
| `BaseDataset`               | `library/data/_deprecated/dataset.py`        | `TrainingDataset`            | ✅     |
| `DreamBoothDataset`         | `library/data/_deprecated/dataset.py`        | `TrainingDataset`            | ✅     |
| `FineTuningDataset`         | `library/data/_deprecated/dataset.py`        | `TrainingDataset`            | ✅     |
| **VAE Dtype Configuration** | `library/data/structures.py`                 | `TrainingDataset`            | ✅     |
| `BucketManager`             | `library/data/_deprecated/bucket_manager.py` | `dataset_scanner.py`         | ✅     |
| `DatasetGroup`              | `library/data/_deprecated/dataset.py`        | `DatasetManifest`            | ✅     |
| Legacy caching              | Scattered in training scripts                | `CachingEngine`              | ✅     |
| On-the-fly bucketing        | `BaseDataset.__getitem__()`                  | Pre-computed `EpochManifest` | ✅     |

### Strategy Layer (Retained)

These strategies are **kept** but their usage differs:

| Strategy               | Location                                  | Legacy Usage                 | New Pipeline Usage                                                                                                      |
| ---------------------- | ----------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `SdxlTokenizeStrategy` | `library/strategies/sdxl/tokenization.py` | Called per-sample in dataset | Used by `tokenize_epoch_manifest()` for token caching; SDXL training-side fallback now shares the same CLIP-family helper path |
| `TextEncodingStrategy` | `library/strategies/base/contracts.py`    | Encode tokens → embeddings   | Still used when no cached TE outputs                                                                                    |
| `SdxlTrainingStrategy` | `library/strategies/sdxl/training.py`     | Training orchestration       | ✅ Updated to consume new batch format                                                                                  |

> **Note:** SDXL still has model-family tokenization concerns, but the active training-side token/chunk construction now reuses the same shared CLIP-family helper path as `SdxlTokenizeStrategy` instead of keeping a second implementation.

### Caching Strategy Layer (New)

These implement `CacheBackend` for the new pipeline:

| Strategy                          | Location                             | Purpose                         |
| --------------------------------- | ------------------------------------ | ------------------------------- |
| `SdxlLatentsPipelineStrategy`     | `library/strategies/sdxl/caching.py` | VAE encoding → latent caching   |
| `SdxlTextEncoderPipelineStrategy` | `library/strategies/sdxl/caching.py` | TE encoding → embedding caching |
| `SdLatentsPipelineStrategy`       | `library/strategies/sd/caching.py`   | SD1.5/2 VAE encoding            |

### Data Flow Comparison

```
LEGACY FLOW:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Script creates DatasetGroup with config                      │
│ 2. DatasetGroup → DreamBoothDataset/FineTuningDataset          │
│ 3. Dataset.__getitem__() called per sample:                     │
│    - Load image from disk                                       │
│    - Resize/bucket on-the-fly                                   │
│    - VAE encode (if not cached)                                 │
│    - Tokenize caption                                           │
│    - Return dict with latents, tokens, metadata                 │
│ 4. DataLoader batches and collates                              │
│ 5. Training loop processes batch                                │
└─────────────────────────────────────────────────────────────────┘

NEW FLOW:
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Scan (once, reuse across epochs)                       │
│   dataset_scanner.py → DatasetManifest (JSON)                   │
│   - Parallel directory scanning                                 │
│   - Bucket assignment                                           │
│   - Caption loading                                             │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 2: Cache (once, skip if cached)                           │
│   CachingEngine + CacheBackend → .safetensors files          │
│   - VAE latent encoding                                         │
│   - TE output encoding (optional)                               │
│   - Multi-GPU distributed                                       │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 3: Epoch Prep (per epoch, fast)                           │
│   prepare_epoch() → EpochManifest                               │
│   - Caption augmentation (shuffle, dropout)                     │
│   - Deterministic shuffle (seed + epoch)                        │
│   - Warmup ordering (largest batches first)                     │
│   - Token file generation (optional)                            │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 4: Training (per epoch)                                   │
│   TrainingDataset + DataLoader → batches                        │
│   - Load from .safetensors (disk → CPU → GPU)                   │
│   - Tokenize on-the-fly OR load from token file                 │
│   - Distributed sharding (rank/world_size)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Batch Format Comparison

| Key                    | Legacy Format                      | New Format                                                               |
| ---------------------- | ---------------------------------- | ------------------------------------------------------------------------ |
| `latents`              | `[B, 4, H, W]`                     | `[B, 4, H, W]` (unchanged)                                               |
| `input_ids`            | `[B, 77]`                          | `{"clip_l": [B, 77], "clip_g": [B, 77]}` or **None** (on-the-fly)        |
| `captions`             | `[str, ...]`                       | `[str, ...]` (unchanged)                                                 |
| `original_sizes`       | `[[H,W], ...]` or missing          | Via `batch["conditionings"][i].original_size_hw`                         |
| `crop_top_lefts`       | `[[T,L], ...]` or missing          | Via `batch["conditionings"][i].crop_top_left`                            |
| `target_sizes`         | Missing                            | Via `batch["conditionings"][i].target_size_hw`                           |
| `text_encoder_outputs` | `encoder_hidden_states1_list` etc. | `{"hidden_state1": [B,...], "hidden_state2": [B,...], "pool2": [B,...]}` |
| `loss_weights`         | `[float, ...]`                     | `[B]` tensor                                                             |
| `flippeds`             | Missing                            | `[bool, ...]`                                                            |
| `alpha_masks`          | Missing                            | `[B, H, W]` or None                                                      |

### Token Handling Modes

| Mode                     | tokens_path | streaming_tokens | Behavior                                                           |
| ------------------------ | ----------- | ---------------- | ------------------------------------------------------------------ |
| **On-the-fly (default)** | None        | -                | Tokenize from `batch["captions"]` using the shared CLIP-family helper path |
| **Cached (upfront)**     | Set         | False            | Load all tokens at TrainingDataset init                            |
| **Cached (streaming)**   | Set         | True             | Load batch tokens via `get_slice()`                                |

### Completed Integration

- [x] Wire `TrainingDataset` into `sdxl_peft.py` (replaced legacy DataLoader)
- [x] `calculate_val_loss_check` updated to accept int or dataloader
- [x] Resume support via `itertools.islice` (not `accelerator.skip_first_batches`)
- [x] `training_metadata.py` refactored for `DatasetManifest`
- [x] Config consolidation: moved TE caching fields to `DataConfig.caching`
- [x] Centralized config defaults in `prepare_config()` (learning rates, cache_dir)
- [x] Manifest saved to `cache_dir/dataset_manifest.json` after caching
- [x] Cache file naming includes resolution: `{id}_{w}x{h}_sdxl_latents.safetensors`
- [x] Tag parsing respects `keep_tokens_separator` (|||)
- [ ] Benchmark new vs legacy performance
- [ ] Remove `library/data/_deprecated/` after full validation

### Completed Config Integration

- [x] `class_tokens` param in `scan_directory()` - Fallback caption for reg images
- [x] `create_manifest_from_config()` - Handles all source types from `DataConfig`
- [x] `prior_loss_weight` param in `TrainingDataset` - Configurable reg image loss

### Test Plan

See `DATA_PIPELINE_TEST_PLAN.md` for:

- Unit test specifications (scanner, epoch prep, dataloader)
- Integration test design
- Open questions requiring audit

**Integration Smoke Tests** (`tests/integration/test_sdxl_peft_smoke.py`): 16 tests covering:

- ✅ Manifest creation from config
- ✅ Validation split logic
- ✅ SDXL latent caching + roundtrip verification
- ✅ Epoch preparation
- ✅ DataLoader batch format
- ✅ Training metadata generation
- ✅ Tag frequency computation
- ✅ Validation epoch preparation
- ✅ TE output caching (creates files, roundtrip)
- ✅ val_data_dir explicit directory
- ✅ Batch skipping via `itertools.islice`
- ✅ Config integration (cache_dir respected)

**Not yet tested:**

- Multi-GPU sharding
- Validation
- Regularization
- prefetch, dataloaders, full bf16/fp16
- subsets
- saving state
- resuming state
- alpha_mask (detection works, but mask extraction/usage not implemented)

### Audit Findings (from `AUDIT/AUDIT_PHASE_1.md`)

#### ⚠️ `accelerator.prepare()` Warning

The new `TrainingDataset` implements manual sharding via `rank`/`world_size`. **Do NOT pass the DataLoader to `accelerator.prepare()`** - it may attempt to shard again or fail on IterableDataset.

```python
# CORRECT - create dataloader directly, don't prepare
train_dataloader = create_training_dataloader(
    ..., rank=accelerator.process_index, world_size=accelerator.num_processes
)

# WRONG - don't do this
# train_dataloader = accelerator.prepare(train_dataloader)
```

#### 💡 Resume & Checkpointing (from `AUDIT/AUDIT_PHASE_4.md`)

**Epoch Manifest:** ✅ No need to persist - regenerated deterministically from `seed + epoch`.

**Fast Skip (TODO):** Add `start_batch_index` to `TrainingDataset`:

```python
def __init__(self, ..., start_batch_index: int = 0):
    self._start_index = start_batch_index
    # In __iter__: slice batches[start_batch_index:] - zero I/O for skipped batches
```

**Token File Reuse (TODO):** Update `tokenize_epoch_manifest` to check existing file:

1. Check if file exists
2. Validate `manifest_hash` in metadata
3. Skip tokenization if valid, regenerate if invalid/missing

**Storage Strategy:** Ephemeral + Cleanup

- Generate `epoch_manifest.json` and `tokens.safetensors` at epoch start
- Delete after epoch completes
- Regenerated automatically on resume if missing

**TODO (needs GPU testing):**

- [ ] Streaming tokens vs RAM loading (memory vs latency tradeoff)
- [ ] `pin_memory=True` effectiveness (requires CUDA device)

#### 🔒 Cache Invalidation (from `AUDIT/AUDIT_PHASE_6.md`)

- ✅ Bucket resolution changes trigger re-caching (`is_cache_valid` checks shape + metadata)
- ✅ Selective re-caching: only affected images are re-processed # TODO CONFIRM IF TRUE
- ✅ Metadata stored in `.safetensors` header for fast validation
- ✅ Config hash validation via `get_or_create_manifest()`

**Large-Scale Dataset Optimization (ROADMAP):** Current JSON manifest grows ~2KB/entry:

- Binary format (msgpack/pickle) for faster I/O
- Incremental manifest updates instead of full rewrite
- Lazy loading of manifest entries
- Sharded manifests by bucket
- Skip creation if unchanged from previous run

---

## Phase 1: Dataset Preparation

**Goal:** Scan directories, read captions, compute buckets, generate manifest.

## Phase 2: Caching

**Goal:** Fast VAE latent and text encoder output caching.

> [!NOTE] > `library/data` is model-agnostic. Strategies implement `CacheBackend` and get injected by training scripts.
> Flow: training script → creates strategy → passes to CachingEngine

## Phase 3: Epoch Preparation

**Goal:** Generate shuffled, batched epoch manifest.

## Phase 4: Training DataLoader

**Goal:** Fast batch iteration from pre-computed manifests.

- [x] Training script integration
  - [x] Replace current DataLoader in `sdxl_peft.py`
  - [ ] Replace current DataLoader in `sd_peft.py`
  - [ ] Benchmark vs current implementation

---

## Data Pipeline Files

| File                   | Status | Description                                                      |
| ---------------------- | ------ | ---------------------------------------------------------------- |
| `__init__.py`          | ✅     | Package exports (all structures and functions)                   |
| `structures.py`        | ✅     | CacheEntry, Bucket, BatchInfo, EpochManifest, DatasetManifest    |
| `manifest.py`          | ✅     | JSON I/O, `create_manifest`, config hash validation              |
| `scanners.py`          | ✅     | `scan_directory` (DreamBooth), `scan_metadata_file` (FineTuning) |
| `bucketing.py`         | ✅     | `make_bucket_resolutions`, `select_bucket` logic                 |
| `caching_engine.py`    | ✅     | CacheBackend interface, CachingEngine with multi-GPU         |
| `dataloader.py`        | ✅     | TrainingDataset, create_training_dataloader, distributed support |
| `epoch_preparation.py` | ✅     | prepare_epoch (warmup, shuffle), prepare_validation_epoch        |
| `caption_processor.py` | ✅     | Caption augmentation (dropout, shuffle, wildcards)               |
| `image_utils.py`       | ✅     | Image I/O utilities (size, alpha check, ID generation)           |
| `prompt_utils.py`      | ✅     | Prompt helpers for standalone scripts                            |

### Key Design Decisions (Skeleton Phase)

- **`BatchInfo`**: Full batch metadata stored in `EpochManifest.batches` instead of just image IDs
- **VAE-agnostic**: `DatasetManifest` stores `latent_channels`, `latent_scale_factor`, and `latent_dtype`
- **Warmup strategy**: Largest resolution batches first to establish CUDA memory allocation
- **`CacheEntry`**: Stores paths to cache files, not tensors — tensors loaded on-demand
- **Strategy delegation**: `CachingEngine` delegates model-specific work to `library/strategies/` classes

---

## Performance Targets

| Metric                | Current     | Target         | Status         |
| --------------------- | ----------- | -------------- | -------------- |
| Latent caching        | ~4 it/s     | 100+ it/s      | 🔜 Not started |
| Training data loading | ~4 it/s     | 200+ batches/s | 🔜 Not started |
| Epoch start time      | ~30s        | <2s            | 🔜 Not started |
| Memory predictability | Random OOMs | Zero OOMs      | 🔜 Not started |

---

## Notes

- SDXL is the primary focus; SD strategies work but less tested
- Cache validation detects: missing keys, shape mismatch, missing flip_aug, caption changes
- Crop coordinates computed for SDXL micro-conditioning (`get_crop_ltrb`)
- Alpha mask support: validation ready, but encoding/saving not yet implemented
- Async 4-stage pipeline from plan not yet implemented (current sync impl works)

---

## Architecture Notes: Model-Specific Caching

### Naming Convention

Following the per-model folder pattern established in Phase 1-2 refactoring:

| Component         | Files                                                                     | Purpose                           |
| ----------------- | ------------------------------------------------------------------------- | --------------------------------- |
| Checkpointing     | `training/sd_checkpointing.py`, `training/sdxl_checkpointing.py` (legacy) | Save/load training state          |
| Sample generation | `training/sample_generation.py` (generic)                                 | Generate samples                  |
| **Caching**       | `strategies/sd/caching.py`, `strategies/sdxl/caching.py`                  | Cache latents, TE outputs         |
| **Training**      | `strategies/sd/training.py`, `strategies/sdxl/training.py`                | Training orchestration (strategy) |

> **Note:** `sample_images` is now inlined into strategy classes (`SdTrainingStrategy.sample_images()`, etc.) which call `sample_images_common()` directly. The wrapper files (`sd_sample_generation.py`, `sdxl_sample_generation.py`) are kept only for backward compatibility with legacy `*_finetune.py` scripts.

### Integration Approach

The orchestration strategies (`sdxl/training.py`) have been updated to:

1.  **Accept our new data format** - `batch["conditionings"]` (list of `SdxlConditioning` objects) instead of flat keys
2.  **Extract values for UNet** - Training loop knows it's SDXL, casts `SdxlConditioning` for `get_size_embeddings()`
3.  **Use `CacheData.aux`** for TE outputs instead of `text_encoder_outputs*_list` keys

This keeps data pipeline model-agnostic while letting training strategies handle model-specific extraction.

### Composition Pattern (CacheData)

```
CacheData (model-agnostic, in structures.py)
├── latents, latents_flipped, alpha_mask
├── aux: dict[str, Tensor]  # For TE outputs
└── conditioning: ModelConditioning | None
         │
         └── SdxlConditioning (in pipeline_sdxl.py)
             ├── original_size_hw
             ├── crop_top_left
             └── target_size_hw
```

### TODO:

- [x] Add `target_size_hw` to `SdxlConditioning`
- [x] Extract `alpha_masks` to batch in dataloader
- [x] Add `flip_aug` parameter and `flippeds` to batch
- [x] Update `peft_strategy_sdxl.py` to use `batch["conditionings"]`
- [x] Renamed `pipeline_sdxl.py` → `sdxl_caching.py`
- [x] random_crop - Implemented in preprocess_image with padding percent option
- [ ] ThreadPool per batch - CPU optimization

---

### Future Research: TE Caching + Caption Augmentations

**Problem:** Currently, caption augmentations (shuffle, dropout, wildcards) require on-the-fly tokenization + TE encoding, sacrificing the speed gains of TE caching.

**Potential Solutions:**

1. **Per-Epoch TE Caching**: After `prepare_epoch()` applies augmentations, run TE encoding on the processed captions and cache to epoch-specific files. Clean up after epoch. Trade-off: adds TE encoding overhead at epoch boundaries.

2. **Embedding-Level Augmentations** (Research): Apply augmentations directly to cached TE outputs:
   - Caption dropout: Select between full TE output vs. empty caption TE output
   - Token dropout: Zero out specific token embeddings (requires per-token caching)
   - Shuffle/wildcards: Harder - attention patterns change with sequence order

**Status:** Not implemented. Per-epoch TE caching is more straightforward; embedding augmentations need more research into what's mathematically valid.
