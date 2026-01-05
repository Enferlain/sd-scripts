# Data Pipeline Implementation Plan

This document tracks implementation progress for the data pipeline rework.
See `DATA_PIPELINE_PLAN.md` for design and `DATA_PIPELINE_CURRENT.md` for legacy reference.

## Status: ✅ Phase 1-4 Data Loading Complete, ✅ PEFT Strategy Integration Complete

**Last Updated:** 2026-01-05

- Phase 1 (scanning): Complete ✅
- Phase 2 (caching): Complete ✅
- Phase 3 (epoch prep): Complete ✅
- Phase 4 (dataloader): Complete ✅ - all batch fields implemented
- **PEFT Strategy Integration: Complete ✅** - `peft_strategy_sdxl.py` uses new batch format

---

## Migration Overview: Legacy vs New System

### What's Being Replaced

| Legacy Component     | Location                                     | Replacement                  | Status   |
| -------------------- | -------------------------------------------- | ---------------------------- | -------- |
| `BaseDataset`        | `library/data/_deprecated/dataset.py`        | `TrainingDataset`            | ✅ Ready |
| `DreamBoothDataset`  | `library/data/_deprecated/dataset.py`        | `TrainingDataset`            | ✅ Ready |
| `FineTuningDataset`  | `library/data/_deprecated/dataset.py`        | `TrainingDataset`            | ✅ Ready |
| `BucketManager`      | `library/data/_deprecated/bucket_manager.py` | `dataset_scanner.py`         | ✅ Ready |
| `DatasetGroup`       | `library/data/_deprecated/dataset.py`        | `DatasetManifest`            | ✅ Ready |
| Legacy caching       | Scattered in training scripts                | `CachingEngine`              | ✅ Ready |
| On-the-fly bucketing | `BaseDataset.__getitem__()`                  | Pre-computed `EpochManifest` | ✅ Ready |

### Strategy Layer (Retained)

These strategies are **kept** but their usage differs:

| Strategy               | Location                                   | Legacy Usage                 | New Pipeline Usage                                                                                                      |
| ---------------------- | ------------------------------------------ | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `SdxlTokenizeStrategy` | `library/strategies/strategy_sdxl.py`      | Called per-sample in dataset | Used by `tokenize_epoch_manifest()` for token caching; **bypassed** by direct `tokenize_sdxl_captions()` for on-the-fly |
| `TextEncodingStrategy` | `library/strategies/strategy_base.py`      | Encode tokens → embeddings   | Still used when no cached TE outputs                                                                                    |
| `SdxlPeftStrategy`     | `library/strategies/peft_strategy_sdxl.py` | Training orchestration       | ✅ Updated to consume new batch format                                                                                  |

> **Note:** For on-the-fly tokenization, we call tokenizers directly via `tokenize_sdxl_captions()` rather than going through `SdxlTokenizeStrategy`. This avoids strategy overhead when tokenizers are already available.

### Caching Strategy Layer (New)

These implement `CachingStrategy` for the new pipeline:

| Strategy                          | Location                             | Purpose                         |
| --------------------------------- | ------------------------------------ | ------------------------------- |
| `SdxlLatentsPipelineStrategy`     | `library/strategies/sdxl_caching.py` | VAE encoding → latent caching   |
| `SdxlTextEncoderPipelineStrategy` | `library/strategies/sdxl_caching.py` | TE encoding → embedding caching |
| `SdLatentsPipelineStrategy`       | `library/strategies/sd_caching.py`   | SD1.5/2 VAE encoding            |

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
│   CachingEngine + CachingStrategy → .safetensors files          │
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
| **On-the-fly (default)** | None        | -                | Tokenize from `batch["captions"]` using `tokenize_sdxl_captions()` |
| **Cached (upfront)**     | Set         | False            | Load all tokens at TrainingDataset init                            |
| **Cached (streaming)**   | Set         | True             | Load batch tokens via `get_slice()`                                |

### Remaining Integration Work

- [ ] Wire `TrainingDataset` into `sdxl_peft.py` (replace legacy DataLoader)
- [ ] Remove `library/data/_deprecated/` after full validation
- [ ] Benchmark new vs legacy performance

### Completed Config Integration

- [x] `class_tokens` param in `scan_directory()` - Fallback caption for reg images
- [x] `create_manifest_from_config()` - Handles all source types from `DataConfig`
- [x] `prior_loss_weight` param in `TrainingDataset` - Configurable reg image loss

### Test Plan

See `DATA_PIPELINE_TEST_PLAN.md` for:

- Unit test specifications (scanner, epoch prep, dataloader)
- Integration test design
- Open questions requiring audit

---

## Phase 1: Dataset Preparation

**Goal:** Scan directories, read captions, compute buckets, generate manifest.

### Tasks

- [x] Create `dataset_scanner.py` module

  - [x] `scan_directory()` - Walk directory tree, find images
  - [x] `read_caption()` - Load caption from .txt/.caption files
  - [x] `select_bucket()` - Assign image to resolution bucket
  - [x] `create_manifest()` - Generate DatasetManifest from scan results

- [x] Handle metadata sources

  - [x] Directory-based (DreamBooth style)
  - [x] JSON metadata (FineTuning style)
  - [x] Error if captions missing (require_caption=True by default)

- [x] Bucket calculation
  - [x] Port bucket resolution logic from `BucketManager`
  - [x] Support configurable bucket parameters

---

## Phase 2: Caching

**Goal:** Fast VAE latent and text encoder output caching.

### Tasks

- [x] Create `CachingStrategy` interface
- [x] Create `CachingEngine` skeleton

- [x] Implement model-specific strategies (NEW code in `pipeline_*.py`)

  - [x] `SdLatentsPipelineStrategy` (`library/strategies/pipeline_sd.py`) - SD VAE encoding, scale factor 0.18215
  - [x] `SdxlLatentsPipelineStrategy` (`library/strategies/pipeline_sdxl.py`) - SDXL VAE encoding, scale factor 0.13025
  - [x] `SdxlTextEncoderPipelineStrategy` (`library/strategies/pipeline_sdxl.py`) - SDXL dual TE output caching

> [!NOTE] > `library/data` is model-agnostic. Strategies implement `CachingStrategy` and get injected by training scripts.
> Flow: training script → creates strategy → passes to CachingEngine

- [x] Implement caching loop

  - [x] Batched image loading (ThreadPoolExecutor)
  - [x] Batch grouping by bucket resolution
  - [x] Multi-GPU workload distribution
  - [x] Progress bar with tqdm

- [x] File format: `.safetensors` (memory-mapped, fast GPU transfer, metadata support)

- [x] Cache validation (`is_cache_valid()` on strategies)
  - [x] Check required keys exist (latents, hidden_states)
  - [x] Verify tensor shapes match bucket resolution
  - [x] Check flip_aug/alpha_mask presence if required
  - [x] Metadata matching (bucket_reso, caption_hash for TE)
  - [x] `skip_validity_check` option for fast path

---

## Phase 3: Epoch Preparation

**Goal:** Generate shuffled, batched epoch manifest.

### Tasks

- [x] Create `prepare_epoch()` function
- [x] Create `prepare_validation_epoch()` function
- [x] Implement warmup ordering (largest resolutions first)
- [x] Create `BatchInfo` dataclass for structured batch metadata

- [x] Caption processing

  - [x] Create `CaptionConfig` dataclass
  - [x] Port `process_caption()` from legacy
  - [x] Tag shuffle
  - [x] Caption dropout
  - [x] Tag dropout
  - [x] Wildcard resolution
  - [x] Token warmup
  - [x] Protected tags (immune to dropout)
  - [x] Keep tokens separator
  - [x] Unit tests (30 tests)

- [x] Integrate caption processing into `prepare_epoch()`

- [x] Tokenization integration

  - [x] `tokenize_epoch_manifest()` - batch tokenize to safetensors
  - [x] `load_epoch_tokens()` - load tokenized captions
  - [x] Safetensors storage (not JSON) for efficiency

- [x] Reproducibility fixes
  - [x] `stable_string_hash()` in hash_utils.py (64-bit blake2b)
  - [x] `BatchInfo.repeat_indices` for per-repeat tracking
  - [x] `BatchInfo.get_sample_key()` for unique sample identification
  - [x] Structured (img_id, repeat_idx) instead of string separator

---

## Phase 4: Training DataLoader

**Goal:** Fast batch iteration from pre-computed manifests.

### Tasks

- [x] Create `TrainingDataset` (IterableDataset)
- [x] Create `create_training_dataloader()` helper

- [x] Token file loading

  - [x] Add `tokens_path` param to TrainingDataset
  - [x] Offset-based batch slicing (sequential index mapping)
  - [x] Manifest hash validation
  - [ ] Streaming mode (`get_slice()` for memory efficiency) - deferred

- [x] Distributed training support

  - [x] Multi-GPU sharding (rank/world_size)
  - [x] Worker-level sharding (worker_id/num_workers)
  - [x] CPU tensor output with pin_memory
  - [x] Seed + epoch mixing for per-epoch shuffle variation

- [x] Additional data loading

  - [x] SDXL metadata (via `SdxlConditioning` in `conditionings` list)
  - [x] Flip augmentation (`flip_aug` param, loads `latents_flipped`)
  - [x] Loss weights (from `is_reg`)
  - [x] Alpha masks (from `CacheData.alpha_mask`)
  - [x] `target_size_hw` added to `SdxlConditioning`

- [ ] Training script integration
  - [ ] Replace current DataLoader in `sd_peft.py`
  - [ ] Replace current DataLoader in `sdxl_peft.py`
  - [ ] Benchmark vs current implementation

---

## Skeleton Files (Completed)

| File                   | Status | Description                                                   |
| ---------------------- | ------ | ------------------------------------------------------------- |
| `__init__.py`          | ✅     | Package exports (all dataclasses and functions)               |
| `dataclasses.py`       | ✅     | CacheEntry, Bucket, BatchInfo, EpochManifest, DatasetManifest |
| `manifest.py`          | ✅     | JSON I/O for manifests (dataset and epoch)                    |
| `caching_engine.py`    | ✅     | CachingStrategy interface, CachingEngine with multi-GPU       |
| `dataloader.py`        | ✅     | TrainingDataset (IterableDataset), create_training_dataloader |
| `epoch_preparation.py` | ✅     | prepare_epoch (warmup ordering), prepare_validation_epoch     |
| `dataset_scanner.py`   | ✅     | Phase 1: scan_directory, create_manifest, bucket logic        |

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

- Phase 1 (scanning) and Phase 2 (caching) are fully functional and tested
- SDXL is the primary focus; SD strategies work but less tested
- Cache validation detects: missing keys, shape mismatch, missing flip_aug, caption changes
- Crop coordinates computed for SDXL micro-conditioning (`get_crop_ltrb`)
- Alpha mask support: validation ready, but encoding/saving not yet implemented
- Async 4-stage pipeline from plan not yet implemented (current sync impl works)
- First integration target: `sdxl_peft.py` (most used script)

---

## Architecture Notes: Model-Specific Caching

### Naming Convention

Following the existing pattern for model-specific scripts:

| Component         | Files                                                  | Purpose                   |
| ----------------- | ------------------------------------------------------ | ------------------------- |
| Checkpointing     | `sd_checkpointing.py`, `sdxl_checkpointing.py`         | Save/load training state  |
| Sample generation | `sd_sample_generation.py`, `sdxl_sample_generation.py` | Generate samples          |
| **Caching**       | `pipeline_sdxl.py` (→ rename to `sdxl_caching.py`)     | Cache latents, TE outputs |

Our caching strategies (`SdxlLatentsPipelineStrategy`, `SdxlTextEncoderPipelineStrategy`) are the data-layer
equivalent of checkpointing/sampling - model-specific utilities that the training strategy coordinates.

### Integration Approach

The existing orchestration strategies (`peft_strategy_sdxl.py`) will be updated to:

1. **Accept our new data format** - `batch["conditionings"]` (list of `SdxlConditioning` objects) instead of flat keys
2. **Extract values for UNet** - Training loop knows it's SDXL, casts `SdxlConditioning` for `get_size_embeddings()`
3. **Use `CacheData.aux`** for TE outputs instead of `text_encoder_outputs*_list` keys

This keeps data pipeline model-agnostic while letting training strategies handle model-specific extraction.

### Composition Pattern (CacheData)

```
CacheData (model-agnostic, in dataclasses.py)
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
- [ ] ThreadPool per batch - CPU optimization
- [ ] color_aug, random_crop, face_crop_aug_range - These are on the fly probably
