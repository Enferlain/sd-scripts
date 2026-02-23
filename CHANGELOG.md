# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2026-02-23]

### Added

- **Phase 2B: FineTuneMode (SDXL)** — Implemented `FineTuneMode` for full-model SDXL fine-tuning:
  - New `library/training/modes/finetune_mode.py` implementing all 13 `TrainingMode` protocol hooks
  - UNet unfreezing + optional per-TE training with individual learning rates
  - Full-model checkpoint saving delegated to strategy (no SDXL imports in mode)
  - EDM2 side-artifact saves via checkpoint name intent detection
  - Block LR fail-fast guard (deferred to mode-agnostic optimizer phase)
  - State hooks for epoch/step metadata only (accelerator handles full model state natively)
  - Exported `FineTuneMode` from `library.training.modes`
  - Migrated `scripts/sdxl_finetune.py` from 850-line monolith to ~55-line thin entrypoint using `Trainer` + `FineTuneMode`
  - Hardened `Trainer.remove_checkpoint()` to handle both file and directory (diffusers format) removal

### Changed

- **2B Architectural Correction: Strategy Delegation** — Separated mode lifecycle from model-family specifics:
  - Added `save_model_checkpoint()` to `CheckpointingStrategy` base class (generic signature, no mode-specific knobs)
  - Implemented `SdxlTrainingStrategy.save_model_checkpoint()` — moves full-model SD/Diffusers serialization out of mode
  - Implemented `SdxlTrainingStrategy.post_process_trainable()` — encapsulates TE1 last-layer/final_layer_norm freeze
  - `FineTuneMode.prepare_trainables()` now uses `strategies.is_train_unet()`/`get_text_encoders_train_flags()` + `post_process_trainable()`
  - `FineTuneMode.save_checkpoint()` reduced from 80+ lines to ~20 lines of delegation
  - Removed all `library.models.sdxl.conversion` imports from mode code
  - Fixed dual-TE DeepSpeed assumption in both `PeftMode` and `FineTuneMode` — dynamic kwargs replace hardcoded `text_encoder1`/`text_encoder2`
  - 27 unit tests verify strategy delegation, no-SDXL-leak assertions, and variable TE count (tested with 2 and 3 encoders)
- **Training Diagnostics Block** — centralized, mode-agnostic diagnostics emitted at training start:
  - Per-component module counts (parameterized leaf modules) and parameter counts with trainable/total breakdown
  - Modes decide what to show via `get_diagnostics_components()` — FineTuneMode shows all backbone, PeftMode shows adapter (with future hook for per-component breakdown via optional adapter method)
  - Context line: mode, strategy, precision, gradient checkpointing, xformers, deepspeed
  - Optimizer group summary: per-group LR and parameter count
  - New `benchmark_sdxl_finetune.yaml` and `test_finetune.yaml` configs for fine-tune benchmarking
  - Updated `run_benchmark.ps1` to route fine-tune configs to `scripts/sdxl_finetune.py`

### Fixed

- **DeepSpeed config truthiness** — `cfg.performance.deepspeed` is a `DeepSpeedConfig` dataclass (always truthy), but 4 checks treated it as a bool. Changed to `cfg.performance.deepspeed.deepspeed` in `peft_mode.py`, `finetune_mode.py` (2 spots), and `checkpointing.py`

## [2026-02-18]

### Changed

- **Phase 2A: Adapter-Neutral Shared Flow** — Neutralized all remaining adapter-specific assumptions across `trainer.py`, `training_loop.py`, and all strategy files:
  - Added 4 new `TrainingMode` protocol hooks: `on_step_start`, `get_trainable_params`, `set_eval`, `set_train`
  - Implemented all hooks in `PeftMode` delegating to adapter methods
  - Added `_primary_trainable` field and `trainable_model` property on `Trainer` (semantic model, set by mode); separate `_grad_sync_handle` field for `accelerator.accumulate()` target
  - Removed `_on_step_start_for_adapter` callback from `Trainer`
  - Replaced all `trainer.adapter` references in `training_loop.py` with `trainer.trainable_model` and mode hooks
  - Renamed `strategies.all_reduce_adapter()` → `all_reduce_trainable()` (hard cut)
  - Renamed `strategies.post_process_adapter()` → `post_process_trainable()` (hard cut)
  - Renamed `adapter` → `trainable_model` param in all strategy methods across base, SD, and SDXL (`on_step_start`, `on_validation_step_end`, `calculate_val_loss`, `process_batch`, `process_val_batch`, `get_noise_pred_and_target`)
  - Guarded `cfg.peft.*` metadata: PEFT keys omitted entirely when `cfg.peft` is absent
  - Added `assert trainer._grad_sync_handle is not None` and `assert trainer.trainable_model is not None` contract checks before training loop
  - Guarded `trainable_model.set_multiplier()` with `hasattr` in SD/SDXL differential output preservation (safe for non-adapter trainables)
  - Changed tracker name `"adapter_train"` → `"training"`
  - Updated `prepare_with_accelerator` docstring to document `_grad_sync_handle` + `_primary_trainable` contracts
  - Updated unit and integration test fixtures for new API
- **Runner Rename** — `PeftTrainer` → `Trainer` (`library/training/runners/peft_trainer.py` → `trainer.py`). Hard cut with no compatibility shim. All imports, exports, type references, test classes, and strategy docstrings updated across the codebase
- **Cleanup** — Removed redundant `Trainer = PeftTrainer` type aliases from `base.py` and `peft_mode.py`; updated stale docstring references in `peft_mode.py`, `sd/training.py`, `sdxl/training.py`
- **Test Rename** — `TestPeftTrainer` → `TestTrainer`, `save_checkpoint` parameter renamed `unwrapped_adapter` → `target_model` for protocol consistency

## [2026-02-17]

### Added

- **TrainingMode Protocol** (`library/training/modes/base.py`): New `TrainingMode` protocol with 9 hooks abstracting mode-specific training logic — `prepare_trainables`, `configure_trainable_precision`, `build_optimizer_params`, `prepare_with_accelerator`, `setup_gradient_training`, `register_state_hooks`, `on_epoch_start`, `on_step_end`, `save_checkpoint`
- **PeftMode Implementation** (`library/training/modes/peft_mode.py`): Concrete `TrainingMode` for PEFT/LoRA training, extracted from `PeftTrainer`, `model_prep`, `optimizer`, and `training_loop` phases — zero behavior change

### Changed

- **Phase 1: TrainingMode Extraction** — Extracted adapter-specific logic from phase files into `PeftMode`:
  - `model_prep.py`: `create_adapter()` moved to `PeftMode.prepare_trainables()`; adapter casting/freezing to `PeftMode.configure_trainable_precision()`
  - `optimizer.py`: Optimizer param building, accelerator preparation, gradient setup, and state hooks delegated to `trainer.mode.*` hooks
  - `training_loop.py`: `on_epoch_start` and `on_step_end` callbacks delegated to `trainer.mode.*` hooks
  - `peft_trainer.py`: `save_checkpoint()` delegates file operations to `trainer.mode.save_checkpoint()`; added `mode` as required constructor parameter
- **Test Updates** — Rewrote `test_model_prep.py` and `test_optimizer.py` to match mode delegation API

### Fixed

- **EDM2 Checkpoint Regression** — `save_checkpoint` was ignoring the passed `unwrapped_adapter` after mode extraction, always saving adapter weights. Now threads `target_model` through to `mode.save_checkpoint()` so EDM2 loss weight checkpoints save the correct model
- **Pre-existing Test Import**: Fixed `test_training_checkpointing.py` importing `build_minimum_adapter_metadata` from wrong module (`checkpointing` → `model_metadata`)
- **Stale Docstring** — Updated `PeftTrainer` class docstring to include `mode` parameter

## [2026-02-16]

### Added

- **Integration Tests: Checkpoint I/O** (`tests/integration/test_checkpoint_io.py`): 14 tests covering safetensors metadata roundtrip, epoch/step checkpoint creation with retention policies, state directory save/remove, and PEFT adapter state hooks — all with real file I/O
- **Integration Tests: Training Loop** (`tests/integration/test_training_loop_integration.py`): 10 tests covering step advancement across epochs, `max_train_steps` early stop, step-based checkpoint triggers, and epoch-based checkpoint triggers
- **Strategy Factory Methods** (`base/training.py`): 4 new abstract methods on `TrainingStrategy` — `create_latent_caching_strategy`, `create_te_caching_strategy`, `tokenize_captions`, `encode_te_outputs_in_memory` — enabling phase files to delegate model-specific caching without SDXL imports

### Changed

- **Phase 0: SDXL Decoupling** — Removed all direct SDXL imports from `caching.py` and `training_loop.py`. These files now call strategy factory methods instead of instantiating `SdxlLatentsPipelineStrategy`, `SdxlTextEncoderPipelineStrategy`, or `tokenize_sdxl_captions` directly

### Fixed

- **Training Loop `max_train_steps` Overshoot**: Added early-exit guard to the outer epoch loop so training stops cleanly when `max_train_steps` is reached, instead of processing 1 extra batch per remaining epoch
- **Tiny Image Bucketing**: Fixed zero-dimension bucket crash when images smaller than `bucket_reso_steps` are used with `no_upscale=True`. Now scales proportionally to preserve aspect ratio while meeting minimum size. (PR #91)

## [2026-02-15]

### Changed

- **Dependency Management Migration**: Migrated from `pip` + `requirements.txt` to `pyproject.toml` + `uv`
  - All runtime dependencies now declared in `project.dependencies` including `torch>=2.9,<2.11`, `torchvision`, and `xformers`
  - Default torch resolves from `pytorch-cu130` index (non-explicit fallback)
  - Torch CUDA extras: `torch-cu128`, `torch-cu129`, `torch-cu130` with per-index resolution
  - Dropped `torch-cu124` (incompatible with torch ≥2.9)
  - Torch version extras: `torch-v29`, `torch-v210` with matching `triton-windows` versions
  - ONNX extras: `onnx-cpu`, `onnx-gpu`
  - Dev dependency group: `pytest`, `pytest-asyncio`, `ruff`
  - `customized-optimizers` sourced from git via `tool.uv.sources`
  - Removed `[build-system]` and `[tool.setuptools]` sections (`package = false`)
  - Deleted `requirements.txt`
- **Developer Experience**:
  - Updated `AGENTS.md`, `RULES.md`, `DEVELOPMENT_GUIDE.md`, and benchmarks to use `uv run` and `uvx`
  - Recommended `rg` (ripgrep) for searching throughout the codebase for performance and reliability

## [2026-02-04]

### Fixed

- **CRITICAL: Double-Scaling Latent Bug**: Fixed `_prepare_latents()` double-scaling cached latents by VAE scale factor (0.13025²). Cached latents are pre-scaled during caching, but were incorrectly scaled again during training, causing ~100x lower loss and washed-out samples. Now only on-the-fly encoded latents are scaled.
- **Sampling Logic Clarification**: Clarified and deduplicated `sample_images_check` logic - epoch-based sampling takes precedence when configured; step-based only works when `sample_every_n_epochs` is not set
- **Sampling Pipeline Error**: Fixed `AttributeError: 'NoneType' object has no attribute 'encode_tokens_with_weights'` by adding missing `TextEncodingStrategy.set_strategy()` call in `peft_trainer.py` prepare phase
- **TensorBoard hparams Error**: Fixed `ValueError` in `add_hparams` by implementing `flatten_for_hparams()` to convert nested config dicts to dot-notation keys (TensorBoard requires flat scalar values)
- **optimizer_args Type Error**: Fixed `AttributeError` when `optimizer_args` is a string instead of dict (Hydra migration changed the type)
- **should_train_text_encoder Argument Error**: Fixed `ConfigAttributeError` by passing `cfg.optimizer.learning_rates` instead of `cfg.optimizer` to `should_train_text_encoder()`
- **Unicode Encoding Error**: Removed Japanese characters from print statement in `optimizer.py` that caused `UnicodeEncodeError` on Windows (cp1252 encoding)
- **test_checkpoint.yaml**: Fixed incorrect config key `num_train_epochs` → `max_train_epochs`
- **Sampling Memory Cleanup**: Added proper state clearing between samples to prevent VRAM accumulation:
  - Per-prompt cleanup (`gc.collect`, `torch.cuda.empty_cache`)
  - Model device restoration after sampling completes

### Added

- **Integration Test Configs**: Created 8 test configurations inheriting from `test_core.yaml` for systematic feature verification:
  - `test_checkpoint`, `test_resume`, `test_sampling`, `test_validation`, `test_text_encoder`, `test_memory_optim`, `test_advanced`, `test_logging`
- **run_benchmark.ps1**: Added test configs to benchmark script for resource tracking
- **sample_vae_dtype**: New config option to control VAE precision during sampling independently from training (e.g., use fp16 VAE for sampling even when training with fp32 VAE)

## [2026-02-03]

### Added

- **Phase Function Unit Tests**: Added unit test suite for extracted training phases
  - Created `tests/unit/training/phases/` with 35 tests covering `caching.py`, `model_prep.py`, `optimizer.py`, and `training_loop.py`
  - Shared fixtures in `conftest.py` with deeply mocked `PeftTrainer` for isolated testing
- **INTEGRATION_TESTING.md**: Created comprehensive integration testing checklist for manual verification of trainer features (checkpointing, sampling, validation, etc.)
- **Abstract Base Methods**: Added missing `process_batch` and `calculate_val_loss` abstract methods to `TrainingStrategy` base class for better type safety

### Changed

- **training loop**: Renamed `training/trainers` → `training/runners`

### Fixed

- **Type warnings in PeftTrainer**: Added assertions for optional types (`train_manifest`, `vae_dtype`, `weight_dtype`, `adapter`) and fixed `optimizer_args` dict→str conversion
- **Linting**: Ran `ruff check --fix` to auto-fix 66 issues across codebase

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
