# Data Pipeline Implementation Plan

This document tracks implementation progress for the data pipeline rework.
See `DATA_PIPELINE_PLAN.md` for design and `DATA_PIPELINE_CURRENT.md` for legacy reference.

## Status: ✅ Phase 1-3 Complete, 🔄 Phase 4 In Progress

**Last Updated:** 2026-01-04

- Phase 1 (scanning): Complete ✅
- Phase 2 (caching): Complete ✅
- Phase 3 (epoch prep): Complete ✅
- **Phase 4 (dataloader): In Progress - token loading done**

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
  - [ ] Streaming mode (`get_slice()` for memory efficiency)

- [ ] Additional data loading

  - [ ] SDXL metadata (original_size, crop_ltrb)
  - [ ] Flip augmentation (load `latents_flipped`)
  - [ ] Loss weights and alpha masks

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
