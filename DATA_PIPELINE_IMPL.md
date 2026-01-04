# Data Pipeline Implementation Plan

This document tracks implementation progress for the data pipeline rework.
See `DATA_PIPELINE_PLAN.md` for design and `DATA_PIPELINE_CURRENT.md` for legacy reference.

## Status: 🔄 In Progress

**Last Updated:** 2026-01-04

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

- [ ] Implement model-specific strategies (NEW code in `strategy_*.py`)

  - [ ] `SdLatentsCachingStrategy` - SD VAE encoding, scale factor 0.18215
  - [ ] `SdxlLatentsCachingStrategy` - SDXL VAE encoding, scale factor 0.13025
  - [ ] `SdxlTextEncoderCachingStrategy` - SDXL dual TE output caching

> [!NOTE] > `library/data` is model-agnostic. Strategies implement `CachingStrategy` and get injected by training scripts.
> Flow: training script → creates strategy → passes to CachingEngine

- [x] Implement caching loop

  - [x] Batched image loading (ThreadPoolExecutor)
  - [x] Batch grouping by bucket resolution
  - [x] Multi-GPU workload distribution
  - [x] Progress bar with tqdm

- [ ] Optimize file format
  - [ ] Evaluate safetensors vs npz performance
  - [ ] Memory-mapped loading

---

## Phase 3: Epoch Preparation

**Goal:** Generate shuffled, batched epoch manifest.

### Tasks

- [x] Create `prepare_epoch()` function
- [x] Create `prepare_validation_epoch()` function
- [x] Implement warmup ordering (largest resolutions first)
- [x] Create `BatchInfo` dataclass for structured batch metadata

- [ ] Caption processing (Phase 3 implementation)

  - [ ] Caption dropout
  - [ ] Tag shuffle
  - [ ] Wildcard resolution
  - [ ] Token warmup

- [ ] Integration
  - [ ] Hook into training loop (per-epoch manifest generation)
  - [ ] Save/load epoch manifests for reproducibility

---

## Phase 4: Training DataLoader

**Goal:** Fast batch iteration from pre-computed manifests.

### Tasks

- [x] Create `TrainingDataset` (IterableDataset)
- [x] Create `create_training_dataloader()` helper

- [ ] Complete implementation

  - [ ] Efficient cache file loading
  - [ ] Prefetching and pinned memory
  - [ ] Caption processing (shuffle, dropout, warmup)

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

- Skeleton uses stub implementations - actual caching/loading logic not yet implemented
- **New implementations, not wrappers:** The strategies in `strategy_sd.py`/`strategy_sdxl.py` will get NEW classes implementing `CachingStrategy`. We're not wrapping the old `SdSdxlLatentsCachingStrategy` - that code is tightly coupled to `ImageInfo` and the legacy flow. Fresh code fitting our `CacheEntry`-based system.
- First integration target: `sdxl_peft.py` (most used script)
