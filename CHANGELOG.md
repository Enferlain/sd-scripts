# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2026-01-20]

### Fixed

- **Resume Behavior Bug** (Audits #2, #10): Fixed `global_step` reset bug where training always resumed from step 0 instead of the checkpoint step. Also fixed `load_model_hook` to properly sync `SimpleNamespace` state containers.
- **Text Encoder Offloading** (Audit #6): Restored missing `offload_text_encoders` logic for on-the-fly encoding — TEs now correctly moved to CPU when caching is disabled but offloading is enabled.
- **Lazy UNet Loading** (Audit #13): Restored `load_unet_lazily` support — UNet can now be deferred until after VAE/TE caching to save VRAM during caching phase.

### Changed

- **Accelerator Type Safety** (Audit #7): Implemented property pattern for `PeftTrainer.accelerator` — private `_accelerator` backing field with public property that asserts initialization. Eliminates `Accelerator | None` type warnings throughout codebase.
- **Removed Redundant Code** (Audits #1, #11): Removed duplicate `configure_precision()` call from optimizer phase, centralized training flags (`_train_unet`, `_train_text_encoder`) in model_prep phase, removed unused imports.

### Added

- **ROADMAP.md**: Added validation refactoring section covering: move validation loop to Trainer, add missing ABC definitions (`process_batch`, `process_val_batch`), rename `_log_training_info` to reflect its actual purpose.
- **Documentation**: Documented `sys.path.append` quirk in model_prep.py (adds `library/training/phases/` not `scripts/` like legacy), documented gradient checkpointing timing risk with DDP.

## [2026-01-18]

### Changed

- **Training Loop Extraction (Phase 6-7)**: Completed trainer-as-container refactor
  - **`sdxl_peft.py` simplified from ~780 lines to ~50 lines** — now just instantiates `PeftTrainer` and calls `trainer.train()`
  - Extracted training loop to `library/training/phases/training_loop.py` with signature `run_training_loop(trainer: PeftTrainer)`
  - Extended `phases/optimizer.py` with: val dataloader creation, `accelerator.prepare()`, gradient checkpointing, resume hooks
  - Added `PeftTrainer` helper methods: `_log_training_info()`, `_maybe_sample_at_start()`, `_finalize_training()`
  - Trainer now handles full lifecycle: setup → caching → model_prep → optimizer → training_loop → finalize
  - All 971 unit tests passing

## [2026-01-14]

### Added

- **Trainer Class Architecture (Phase 1-7)**:
  - Created `library/training/trainers/peft_trainer.py` with `PeftTrainer` class
  - Extracted setup logic from `sdxl_peft.py` into `PeftTrainer.setup()`
  - Created `StepOutput` dataclass for modular training loop data flow
  - **Phase 3**: Implemented `run_latent_caching()` and `run_te_caching()` in `library/training/phases/caching.py`
  - **Phase 4**: Implemented `create_adapter()` and `configure_precision()` in `library/training/phases/model_prep.py`
  - **Phase 5**: Added `calculate_max_train_steps()` in `library/training/phases/optimizer.py`
  - **Phase 6**: Added `save_checkpoint()` and `remove_checkpoint()` methods to `PeftTrainer`
  - **Phase 7**: Wired `PeftTrainer` methods to call phase functions with explicit params
  - Updated `sdxl_peft.py` to use extracted phase functions

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
