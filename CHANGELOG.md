# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2026-01-14]

### Added

- **Trainer Class Architecture (Phase 1 & 2)**:
  - Created `library/training/trainers/peft_trainer.py` with `PeftTrainer` class
  - Extracted setup logic (accelerator, manifests, models) from `sdxl_peft.py` into `PeftTrainer.setup()`
  - Created `StepOutput` dataclass for modular training loop data flow
  - Implemented internal event hook system (`_emit`) for future extensibility
  - Restored original inline code in `sdxl_peft.py` to maintain functionality during staged migration

## [2026-01-12]

### Changed

- **Model Directory Reorganization (Phase 1)**: Restructured `library/models/` into per-model folders

  - `library/models/sdxl/` now contains: `unet.py`, `conversion.py`, `loader.py`, `text_encoder.py`, `control_net.py`
  - `library/models/sd/` now contains: `vae.py` (shared VAE utilities)
  - Migrated from flat `sdxl_model_util.py`, `sdxl_original_unet.py` structure to organized hierarchy
  - All 971 unit tests passing after refactor

- **Strategy Consolidation (Phase 2)**: Restructured `library/strategies/` into per-model folders
  - Created `base/`, `sd/`, `sdxl/` subfolders with split modules (tokenization, encoding, caching, training)
  - Renamed `*PeftStrategy` → `*TrainingStrategy` (e.g., `SdxlPeftStrategy` → `SdxlTrainingStrategy`)
  - Inlined `sample_images` into strategy classes — strategies now call `sample_images_common` directly
  - Deleted legacy `*_old.py` backup files
  - Deprecated `SdSdxlLatentsCachingStrategy` (legacy npz format) — new pipeline uses safetensors

### Removed

- **Legacy Config Field**: Removed unused `cache_info` from `configs/data/default.yaml` and cleaned up schema mismatch

## [2026-01-10]

### Added

- **Resource Tracking**: New `ResourceTracker` utility for comprehensive GPU/CPU monitoring during training

  - Tracks PyTorch memory (allocated/reserved) with peak detection
  - **nvidia-smi integration**: Background thread polls every 500ms for true GPU memory peaks (matches nvitop)
  - CPU RAM tracking with before/after/peak values
  - Output includes both PyTorch internals and nvidia-smi values for complete picture

- **Enhanced Benchmark Reporting**: Major improvements to `run_benchmark.ps1`
  - **Multi-run support**: `-Fresh -Runs N` clears cache before each run for consistent caching benchmarks
  - **Timing statistics**: Shows avg/min/max when running multiple iterations
  - **CPU RAM tracking**: Before/after memory in report alongside GPU stats
  - **Distinct progress bars**: "Latent Caching (GPU 0)" and "TE Caching (GPU 0)" for clearer speed parsing
  - **Improved regex parsing**: Correctly extracts nvidia-smi GPU peaks and CPU RAM from resource tracker output

### Changed

- **Caching Progress Bars**: Added `cache_type` parameter to `CachingEngine.cache_dataset()` for labeled progress bars
- **Benchmark Config Extraction**: Report now shows actual YAML keys with proper grouping (training/caching/loader/performance)

### Fixed

- **Multi-run Benchmark Data**: Fixed issue where runs 2+ would overwrite first run's resource data (now preserves Run 1 for fresh caching metrics)
- **Resource Tracker Peak Detection**: Added `torch.cuda.synchronize()` calls for accurate peak memory capture
- **Markdown Table Formatting**: Fixed newline issues in configuration table generation

## [2026-01-09]

### Fixed

- **Image Preprocessing Distortion**: Fixed critical bug where images were squished to bucket resolution instead of properly cropped
  - `preprocess_image()` now resizes to `resized_size` (maintains aspect ratio) then crops to `target_size`
  - Affected files: `sdxl_caching.py`, `sd_caching.py`, `caching_engine.py`
  - Previously: 1556x2048 image → bucket 1536x2048 = **squished** (distorted)
- Now: 1556x2048 image → bucket 1536x2048 = **cropped 10px per side** (correct)
- **Integration Test Failures**: Fixed all 25 integration tests
  - Added `cache_dir` parameter to `create_manifest()` and `create_manifest_from_config()` calls in tests
  - Changed tests to use dynamic `EXPECTED_IMAGE_COUNT` from actual test directory contents
  - Fixed cache file glob patterns from `*_sdxl_latents.safetensors` to generic `*.safetensors`
  - Fixed `test_te_cache_roundtrip` to use `te_cache_path` instead of `latent_cache_path`
  - Fixed `test_val_data_dir_explicit` to use `IMAGE_EXTENSIONS` filter (not just `*.jpg`)
  - Fixed lint warnings in `peft_strategy_sdxl.py`.
- **Preprocessing Improvements**:
  - Implemented correct **resize-then-crop** logic to prevent aspect ratio distortion during caching.
  - Added `random_crop` (bool) and `random_crop_padding_percent` (float) to `PreprocessingConfig`.
  - Added `resize_interpolation` config option supporting:
    - Auto-selection (default): HAMMING for downscaling, LANCZOS for upscaling.
    - Explicit choices: `area` (cv2.INTER_AREA), `hamming`, `lanczos`, `bicubic`, `bilinear`.
- **VRAM Memory Fragmentation**: Fixed critical bug where CUDA reserved memory accumulated across bucket sizes during caching
  - Before: Peak VRAM grew to ~20GB (accumulating all buckets)
  - After: Peak VRAM bounded to ~7.5GB (largest bucket only)
  - Added `torch.cuda.empty_cache()` between bucket transitions in `CachingEngine.cache_dataset()`
  - Reduces peak VRAM by ~62% for multi-resolution datasets

### Added

- **Random Crop Support**: Added `random_crop` and `random_crop_padding_percent` params to caching strategies
  - Random crop uses configurable padding (default 5%) for more varied training crops
  - Center crop (default) is deterministic for reproducibility
- **Auto Interpolation**: Automatically select optimal resize interpolation
  - Uses HAMMING (similar to AREA) for downscaling (prevents aliasing)
  - Uses LANCZOS for upscaling (smooth edges)
- **Config Field**: Added `random_crop_padding_percent` to `PreprocessingConfig`
- **Benchmark Infrastructure**: Created comprehensive benchmarking setup for the new data pipeline
  - `run_benchmark.ps1`: PowerShell script to automate benchmark runs with cache clearing, timing, and GPU stats
  - `benchmark_sdxl.yaml`: Base config with 550 real images for accurate caching/loading measurements
  - `benchmark_sdxl_workers.yaml`: Tests DataLoader worker scaling
  - `benchmark_sdxl_large.yaml`: Extended run for steady-state throughput
  - `benchmark_sdxl_train_te.yaml`: Benchmarks TE training (no TE cache)
  - `benchmark_sdxl_offload.yaml`: Benchmarks TE offloading performance
- **Memory Debug Logging**: Added `DEBUG_CACHING_MEMORY=1` env var for per-batch memory tracking in `CachingEngine`
- **GPU Memory Profiling Script**: Added `scripts/profile_caching.py` for PyTorch memory snapshot analysis
- **TODO Tier 6**: Added critical performance section to `TODO_PRIORITIZED.md` documenting async pipeline optimization tasks

### Changed

- **Test Suite Cleanup**: Deleted stale `test_sdxl_train.py` dry run script (was for debugging)
- **Test Robustness**: Integration tests now adapt to the actual number of images in test assets

## [2026-01-08]

### Added

- **Text Encoder Offloading**: New `offload_text_encoders` option in `performance.memory` config

  - Keeps text encoders on CPU between forward passes to save VRAM
  - Allows caption augmentation (shuffle, dropout) unlike TE caching
  - Encoding happens on CPU, outputs moved to GPU for training
  - Mutually exclusive with `cache_text_encoder_outputs` (validation enforced)
  - Mutually exclusive with TE training (validation enforced)

- **Improved `is_text_encoder_not_needed_for_training()`**: Now returns True when TE caching is enabled and TEs aren't being trained, allowing proper memory cleanup
- **TE Offloading + Training Validation**: Added validation in `config_validation.py` to prevent training TEs while offloading to CPU (would cause major slowdown)
- **Granular Offloading TODO**: Added note for potential future per-TE offloading when using granular LRs like `[1e-5, 0]`

### Changed

- **Data Module Refactoring**: Restructured `library/data/` from flat `pipeline/` subfolder
  - Fixed circular imports in `manifest.py`, `caption_processor.py`, `scanners.py` (use direct module imports)
  - Merged `manifest_builder.py` into `manifest.py`
  - Moved `read_caption`, `_parse_tags`, `compute_tag_frequency` to `caption_processor.py`
  - Updated `__init__.py` to reflect new module structure
  - Updated all imports from `library.data.pipeline` → `library.data` across 5 files

### Fixed

- **Broken Imports**: Fixed references to deleted `dataset_scanner.py` in test files
- **Debug Logging**: Added INFO-level debug logs in `_get_text_cond` for TE device placement and trainability (marked for removal after testing)

## [2026-01-07]

### Added

- **Epoch Tokenization Option**:
  - New config `cache_tokens_per_epoch` in `CachingConfig` for pre-tokenizing captions per epoch
  - When enabled (and TE caching disabled), captions are tokenized once at epoch start to `.safetensors`
  - Reduces tokenizer overhead when using caption augmentations (shuffle, dropout, wildcards)
  - Token files are cleaned up after each epoch completes
- **Smoke Test Progress**: Live training confirmed through epoch 1, 50+ steps completed
- Large-scale dataset manifest optimization notes in `ROADMAP.md`
- **Cache Path Simplification**:
  - `cache_dir` and `config_hash` fields added to `DatasetManifest`
  - `latent_cache_path` and `te_cache_path` now set at manifest creation time
  - Added `get_or_create_manifest()` for manifest persistence and reuse
  - Added `compute_config_hash()` for config-based cache validation
- **Manifest Persistence**: Manifests can now be saved/loaded with cache paths preserved
- **Manifest Hash Validation**: `get_or_create_manifest()` validates config hash and image count before reusing cached manifests
- **Manifest Summary Section**: JSON manifest now includes `summary` with `total_images`, `total_captions`, `num_buckets`
- **Bucket Distribution**: JSON manifest includes `bucket_distribution` array with resolution and count per bucket (sorted by resolution)
- **Bucket Logging**: Restored legacy-style bucket distribution logging during manifest creation (resolution, count, mean AR error)

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
- **TE Dimension Mismatch Bugs**:
  - `sdxl_peft.py`: Added `.squeeze(0)` when storing per-entry TE outputs in memory cache (prevented 3D tensors after dataloader stacking)
  - `strategy_sdxl.py`: Removed premature `reshape()` in `_get_hidden_states_sdxl` that corrupted batch size calculation
  - `peft_strategy_sdxl.py`: Added `+2` to `max_token_length` in `tokenize_sdxl_captions()` to match legacy `SdxlTokenizeStrategy` chunking behavior

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

  - `SdxlTrainingStrategy` now uses new pipeline batch format
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
