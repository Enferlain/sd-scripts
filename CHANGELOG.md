# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2026-01-07]

### Added

- **Smoke Test Progress**: Live training confirmed through epoch 1, 50+ steps completed
- Large-scale dataset manifest optimization notes in `ROADMAP.md`
- **Cache Path Simplification**:
  - `cache_dir` and `config_hash` fields added to `DatasetManifest`
  - `latent_cache_path` and `te_cache_path` now set at manifest creation time
  - Added `get_or_create_manifest()` for manifest persistence and reuse
  - Added `compute_config_hash()` for config-based cache validation
- **Manifest Persistence**: Manifests can now be saved/loaded with cache paths preserved

### Changed

- **Simplified `CachingStrategy` Interface**:
  - Removed `get_cache_path()` - paths are now pre-set on `CacheEntry` at creation
  - Removed `set_cache_path()` - no longer needed
  - Added `get_entry_cache_path()` - reads pre-set path from entry
- `CachingEngine` no longer computes or sets cache paths - reads directly from entries
- `create_manifest()` now accepts optional `cache_dir` parameter to pre-set all entry paths

### Fixed

- **Config Access Fixes** (revealed by smoke test):
  - `init_timestep_sampler()`: Pass `cfg.timestep` instead of full `cfg`
  - `parse_dynamic_timestep_schedule()`: Pass `cfg.timestep` instead of full `cfg`
  - `prepare_edm2_loss_weighting()`: Pass `cfg.loss.edm2` instead of `cfg.loss`
  - `get_huber_threshold_if_needed()`: Updated signature to `(loss_config, huber_config, ...)` - now accepts `LossConfig` and `HuberConfig` separately
  - `cfg.loss.masked` → `cfg.loss.masked.masked_loss` (accessing nested config properly)
- **Device Placement**: `batch["loss_weights"].to(loss.device)` - tensor was on CPU
- **OmegaConf Compatibility**: Fixed `asdict(metadata_config)` to handle OmegaConf `DictConfig` objects in `model_metadata.py`
- **Learning Rate Defaults**: Centralized in `prepare_config()` - `unet` and `text_encoders` default to `base` if not set
- **Cache Dir Fallback**: `prepare_config()` now sets `cache_dir = train_data_dir` if not specified (with defensive `getattr`)
- **Test Updates**: `test_losses_loss.py` updated to use `LossConfig`/`HuberConfig` dataclasses instead of mock `args`
- **Test Fixtures**: Updated all pipeline test fixtures to set `latent_cache_path`/`te_cache_path` on entries

## [2026-01-06]

### Added

- **Data Pipeline: SDXL PEFT Script Integration Complete**

  - `scripts/sdxl_peft.py` now uses new data pipeline (`DatasetManifest`, `CachingEngine`, per-epoch DataLoader)
  - `create_manifest_from_config()` supports both `val_data_dir` and `validation_split` for validation data
  - `compute_tag_frequency()` helper for metadata generation from manifests
  - `training_metadata.py` refactored to accept `DatasetManifest` instead of `DatasetGroup`
  - Added `val_data_dir` field to `SourceConfig` for explicit validation directories

- **Data Pipeline: Comprehensive Audit Completed**

  - 6 audit documents in `AUDIT/` covering integration, batch format, validation, resume, performance, cache invalidation
  - `test_pipeline_benchmark.py` - Performance tests for `prepare_epoch` timing and DataLoader throughput
  - `test_pipeline_dataloader.py` - 8 tests for batch format, flip_aug, prior_loss_weight, streaming tokens, sharding
  - `test_epoch_preparation.py` - 7 tests for shuffle, warmup, repeats, caption processing, tokenization

- **Data Pipeline: Fast Skip Support** (design documented, TODO implementation)

  - `start_batch_index` parameter for O(1) resume without loading skipped batches
  - Token file reuse with `manifest_hash` validation

- **Integration Smoke Tests**: `test_sdxl_peft_smoke.py` - 16 tests covering:
  - Manifest creation, latent caching, dataloader, metadata, validation pipeline
  - NEW: Latent cache roundtrip (save → load → verify), TE caching, `val_data_dir`, batch skipping (`islice`), config integration

### Changed

- **Config Consolidation**: Moved `cache_text_encoder_outputs`, `cache_text_encoder_outputs_to_disk`, `disable_mmap_load_safetensors` from `PerformanceConfig.caching` to `DataConfig.caching`
- `calculate_val_loss_check()` now accepts either a DataLoader or an int (num_batches_per_epoch)
- `calculate_val_loss()` return type simplified from 3-tuple to 2-tuple (removed unused `logs` dict)
- Merged `test_dataset_scanner.py` into `test_pipeline_dataset_scanner.py` - now 26 tests
- Added `TestClassTokens` and `TestCreateManifestFromConfig` test classes
- Updated `DATA_PIPELINE_IMPL.md` with all audit findings and TODOs

### Fixed

- **Batch Skipping for Resume** - Replaced `accelerator.skip_first_batches()` with `itertools.islice()` for unprepared IterableDataset
- **`n_repeats` Aggregation** - Fixed bug where directories with multiple entries would lose repeat info (now uses `max()`)
- **`current_epoch`/`current_step` Type Mismatch** - Changed fallback from `torch.tensor(0)` to `types.SimpleNamespace(value=0)`
- **Masked Loss Fail-Fast** - `cfg.loss.masked=True` now raises `ValueError` if no masks in batch instead of silently proceeding unmasked
- Added type hints to `save_model()` and `remove_model()` inner functions in `sdxl_peft.py`
- Moved Hydra schema registration inside `if __name__ == "__main__"` block
- Removed unused `adapter_has_multiplier` variable
- **Config Consistency**: Fixed `cfg.sdxl.cache_text_encoder_outputs` → `cfg.data.caching.cache_text_encoder_outputs`
- **YAML Completeness**: Added `val_data_dir`, `subsets`, `cache_dir` to `configs/data/default.yaml`
- **LoaderConfig Expansion**: Added `prefetch_factor`, `pin_memory`, renamed `max_workers` → `num_workers`

## [2026-01-05]

### Added

- **Data Pipeline: Composition Pattern for Model-Agnostic DataLoader**

  - `ModelConditioning` ABC in `dataclasses.py` - Base class for model-specific conditioning
  - `CacheData` dataclass - Universal cache container with `latents`, `conditioning`, `aux`
  - `SdxlConditioning` in `pipeline_sdxl.py` - SDXL micro-conditioning (original_size_hw, crop_top_left, target_size_hw)
  - Dataloader now model-agnostic - passes `batch["conditionings"]` to training loop
  - Renamed `extra` → `aux` for TE outputs dict

- **Data Pipeline Phase 4: Complete Batch Fields**

  - `flip_aug` parameter - 50% random flip when enabled
  - `alpha_masks` extraction from `CacheData.alpha_mask`
  - `flippeds` list - tracks which samples used flipped latents
  - `target_size_hw` added to `SdxlConditioning`

- **PEFT Strategy Integration**

  - `SdxlPeftStrategy` now uses new pipeline batch format
  - Added `_extract_conditioning_tensors()` helper for SDXL micro-conditioning
  - `_get_text_cond()` updated for `batch["text_encoder_outputs"]` and `batch["input_ids"]["clip_l/g"]`
  - `call_unet()` now extracts conditioning from `batch["conditionings"]`

- **Dataset Scanner Enhancements**

  - `class_tokens` parameter - Fallback caption for images without caption files (DreamBooth reg)
  - `create_manifest_from_config()` - High-level function handling train_data_dir, reg_data_dir, in_json, subsets
  - `prior_loss_weight` parameter in `TrainingDataset` and `create_training_dataloader()` - Configurable loss weight for regularization images

### Changed

- `CachingStrategy.load_cache()` now returns `CacheData` instead of tuple
- `SdLatentsPipelineStrategy.load_cache()` updated for `CacheData` consistency
- Architecture notes added to `DATA_PIPELINE_IMPL.md` documenting naming conventions and integration approach
- Deprecated tests moved to `tests/_deprecated/` and excluded via `pyproject.toml`

### Fixed

- **Streaming Token Loading** - `streaming_tokens=True` now works correctly

  - Added `_load_tokens_streaming()` using `safetensors.get_slice()` for zero-copy batch loading
  - Memory-efficient: loads only batch tokens instead of entire file (~120MB savings for 100k images)

- **On-the-Fly Tokenization Fallback** - Token caching now optional

  - Added `tokenize_sdxl_captions()` helper for runtime tokenization
  - `_get_text_cond()` falls back to tokenizing from `batch["captions"]` if no `input_ids`

- **DataLoader num_workers** - Changed default from 0 to 4 to avoid blocking I/O
  - Clarified `__len__` returns per-rank count (correct for training loops)

## [2026-01-04]

### Added

- **Data Pipeline Phase 1: Dataset Scanner** (`library/data/pipeline/dataset_scanner.py`)

  - `scan_directory()` - Parallel directory scanning with ThreadPoolExecutor
  - `scan_metadata_file()` - JSON metadata support (FineTuning style)
  - `read_caption()` - Caption reading from .txt/.caption files
  - `make_bucket_resolutions()` - Bucket resolution generation (ported from BucketManager)
  - `select_bucket()` - Image-to-bucket assignment (ported from BucketManager)
  - `create_manifest()` - Generate `DatasetManifest` from scanned images
  - `require_caption` parameter - Error if captions missing (default: True)
  - Support for webp, jxl, tiff image formats

- **Data Pipeline Phase 2: Caching Engine** (`library/data/pipeline/caching_engine.py`)

  - `CachingStrategy` ABC - Interface for model-specific encoding
  - `CachingEngine` - High-performance caching orchestrator
  - Batch grouping by bucket resolution
  - Parallel image loading with ThreadPoolExecutor
  - tqdm progress bar with multi-GPU support
  - Modulo workload distribution across GPUs

- **Data Pipeline Phase 2: Model-Specific Strategies** (`library/strategies/pipeline_*.py`)

  - `SdLatentsPipelineStrategy` - SD 1.5/2.0 VAE latent caching (scale factor 0.18215)
  - `SdxlLatentsPipelineStrategy` - SDXL VAE latent caching (scale factor 0.13025)
  - `SdxlTextEncoderPipelineStrategy` - SDXL dual text encoder output caching
  - `get_crop_ltrb()` - SDXL micro-conditioning crop coordinate calculation
  - `is_cache_valid()` - Cache validation (keys, shapes, metadata, flip_aug)
  - All strategies use `.safetensors` format for fast loading and metadata support
  - Self-contained per-model files (no shared base classes for future flexibility)

- **VAE Dtype Configuration** (`library/data/pipeline/dataclasses.py`)

  - Added `latent_dtype` field to `DatasetManifest` ("fp16", "bf16", "fp32")
  - Added `latent_dtype` parameter to `Bucket.memory_per_image()` for accurate memory estimation

- **Tests**

  - `test_pipeline_dataset_scanner.py` - 22 tests (bucket, scanning, manifest, JSON metadata)
  - `test_pipeline_caching.py` - 6 tests (batching, multi-GPU split, file creation)
  - `test_pipeline_strategies.py` - 12 tests (SD/SDXL latent and TE strategies)
  - `test_sdxl_cache_roundtrip.py` - 13 tests (save/load roundtrip, metadata, validation)
  - `test_pipeline_integration.py` - 6 tests (end-to-end: scan → manifest → cache)
  - `test_pipeline_real_vae.py` - 3 tests (real SDXL VAE encoding validation)

- **Reorganized** `library/data/_deprecated/` - Moved old data scripts for cleaner separation

- **Data Pipeline Phase 3: Caption Processing** (`library/data/pipeline/caption_processor.py`)

  - `CaptionConfig` dataclass with all augmentation options
  - `process_caption()` function with full feature support:
    - Tag shuffle (randomize tag order)
    - Caption dropout (drop entire caption)
    - Tag dropout (drop individual tags)
    - Protected tags (immune to dropout)
    - Wildcard resolution (`{cat|dog}` → random choice)
    - Token warmup (gradual tag introduction)
    - Keep tokens separator for fixed prefix/suffix
  - Integrated into `prepare_epoch()` for per-batch caption processing
  - 30 unit tests covering all features

- **Epoch Tokenization Storage** (`library/data/pipeline/epoch_preparation.py`)

  - `tokenize_epoch_manifest()` - batch tokenize all captions to safetensors
  - `load_epoch_tokens()` - load tokenized captions from safetensors
  - Binary storage (~1.2MB) instead of JSON (~10-15MB) for 10k samples
  - Stores int64 tensors for HuggingFace tokenizer compatibility

- **Reproducibility Improvements** (`library/data/pipeline/`, `library/utils/hash_utils.py`)

  - Added `stable_string_hash()` using 64-bit blake2b (consistent across Python runs)
  - Added `BatchInfo.repeat_indices` for per-repeat disambiguation
  - Added `BatchInfo.get_sample_key()` for unique sample identification
  - Each image repeat now gets independent caption randomness

- **Data Pipeline Phase 4: Token Loading** (`library/data/pipeline/dataloader.py`)

  - `TrainingDataset` now accepts `tokens_path` for epoch token file
  - Offset-based batch slicing (sequential index mapping)
  - Manifest hash validation to ensure token file matches epoch
  - Support for both token file loading and legacy `BatchInfo.input_ids`

- **Distributed Training Support** (`library/data/pipeline/dataloader.py`)

  - Multi-GPU sharding via `rank`/`world_size` parameters
  - DataLoader worker sharding via `get_worker_info()`
  - Tensors yielded on CPU with `pin_memory=True` for efficient GPU transfer

### Fixed

- **Caption Hash Stability** (`library/strategies/pipeline_sdxl.py`)

  - Replaced Python's non-deterministic `hash()` with `stable_string_hash()`
  - TE cache validation now consistent across Python runs

- **Epoch Shuffle Reproducibility** (`library/data/pipeline/epoch_preparation.py`)

  - Fixed seed calculation to mix `seed + epoch` for per-epoch variation
  - Matches legacy behavior: same base seed, different shuffle each epoch

### Changed

- **Flexible Text Encoder Inputs** (`library/data/pipeline/dataclasses.py`)

  - Changed `BatchInfo.input_ids` from hardcoded `input_ids`/`input_ids_2` to flexible dict keyed by encoder name
  - Supports models with 1, 2, or 3+ text encoders (SD, SDXL, SD3, Flux)
