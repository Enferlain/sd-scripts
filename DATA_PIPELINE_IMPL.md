# Data Pipeline Implementation Plan

This document tracks implementation progress for the data pipeline rework.
See `DATA_PIPELINE_PLAN.md` for design and `DATA_PIPELINE_CURRENT.md` for legacy reference.

## Status: 🔄 In Progress

**Last Updated:** 2026-01-04

---

## Phase 1: Dataset Preparation

**Goal:** Scan directories, read captions, compute buckets, generate manifest.

### Tasks

- [ ] Create `dataset_scanner.py` module

  - [ ] `scan_directory()` - Walk directory tree, find images
  - [ ] `read_caption()` - Load caption from .txt/.caption files
  - [ ] `compute_bucket()` - Assign image to resolution bucket
  - [ ] `create_manifest()` - Generate DatasetManifest from scan results

- [ ] Handle metadata sources

  - [ ] Directory-based (DreamBooth style)
  - [ ] JSON metadata (FineTuning style)
  - [ ] Fallback caption generation

- [ ] Bucket calculation
  - [ ] Port bucket resolution logic from `BucketManager`
  - [ ] Support configurable bucket parameters

---

## Phase 2: Caching

**Goal:** Fast VAE latent and text encoder output caching.

### Tasks

- [x] Create `CachingStrategy` interface
- [x] Create `CachingEngine` skeleton

- [ ] Implement model-specific strategies (NEW code in `strategy_*.py`)

  - [ ] `SdLatentsCachingAdapter` - SD VAE encoding, scale factor 0.18215
  - [ ] `SdxlLatentsCachingAdapter` - SDXL VAE encoding, scale factor 0.13025
  - [ ] `SdxlTextEncoderCachingAdapter` - SDXL dual TE output caching

- [ ] Implement caching loop

  - [ ] Batched image loading (async I/O)
  - [ ] Parallel VAE encoding
  - [ ] Async file writing
  - [ ] Progress bar with ETA

- [ ] Optimize file format
  - [ ] Evaluate safetensors vs npz performance
  - [ ] Memory-mapped loading

---

## Phase 3: Epoch Preparation

**Goal:** Generate shuffled, batched epoch manifest.

### Tasks

- [x] Create `prepare_epoch()` function
- [x] Create `prepare_validation_epoch()` function
- [x] Implement memory-aware bucket ordering

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

| File                   | Status | Description                                        |
| ---------------------- | ------ | -------------------------------------------------- |
| `__init__.py`          | ✅     | Package exports                                    |
| `dataclasses.py`       | ✅     | CacheEntry, Bucket, EpochManifest, DatasetManifest |
| `manifest.py`          | ✅     | JSON I/O for manifests                             |
| `caching_engine.py`    | ✅     | CachingStrategy interface, CachingEngine           |
| `dataloader.py`        | ✅     | TrainingDataset, create_training_dataloader        |
| `epoch_preparation.py` | ✅     | prepare_epoch, prepare_validation_epoch            |

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
