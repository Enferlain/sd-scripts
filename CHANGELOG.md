# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2026-03-02]

### Changed

- **Logging Phase 2: Rank-aware logging** — Non-main processes in DDP now have root log level set to `WARNING` after accelerator init, suppressing duplicate `INFO` lines while preserving error/warning visibility on all ranks
- **Logging Phase 3: `log_every_n_steps`** — New `LoggingConfig` field to control tracker emission frequency (default: `1` = every step). `step_logging()` gated by interval; progress bar remains every step. Invalid values (`<= 0`) normalized to `1` with warning
- **Validation defaults now opt-in** — `run_at_start` and `run_at_end` default to `False` in both `ValidationConfig` and `configs/validation/default.yaml`

### Fixed

- **Stale `val_manifest.json` loading** — `get_or_create_manifest` loaded cached validation manifests even when `validation_split` was `0.0`. Now guards on `validation_split > 0` and cleans up stale files

## [2026-02-24]

### Added

- **Validation Scheduler** — Replaced untyped `calculate_val_loss_check()` with a typed `ValidationScheduler` in `library/training/phases/validation.py`:
  - `ValidationStepContext` dataclass replaces mixed `dataloader/int` input with typed context
  - `ValidationScheduler.should_run()` — single source of truth for validation trigger decisions
  - Supports `run_at_start`, `run_at_end`, `every_n_steps`, `every_n_epochs` with OR semantics
  - Default behavior: epoch-end validation when no cadence is set (preserves current behavior)
  - `validate_every_n_epochs` is now functionally active (was config-only, never checked)
- Added `run_at_start` and `run_at_end` fields to `ValidationConfig` dataclass and `configs/validation/default.yaml`
- Added cadence normalization: `validate_every_n_steps <= 0` and `validate_every_n_epochs <= 0` are normalized to `None` with warning
- Persisted `ss_run_validation_at_start` and `ss_run_validation_at_end` in training metadata
- Print `val_loss` and `avg` to console after each validation run (previously only logged to TensorBoard/W&B)
- 26 new unit tests (21 scheduler trigger matrix + 5 decoupling assertions)
- **Logging Phase 0: Correctness Patches** — 12 new unit tests in `tests/unit/logging/test_step_logging.py`:
  - LR key uniqueness (single group, TE+UNet, multi-TE, custom descriptions)
  - W&B run name retention through `init_trackers` kwargs merge

### Changed

- **`generate_step_logs` `lr_descriptions` is now required** — Removed dead fallback naming logic and unused `should_train_text_encoder` import. All callers already provide explicit descriptions
- **Removed import-time `setup_logging()` from all library modules** — Single bootstrap point now in `Trainer.__init__()` (line 214). Script entrypoints keep their own calls. Modules only declare `logger = logging.getLogger(__name__)`

- **Validation and sampling are now decoupled** — eval-mode block enters once if either trigger fires, but each action executes independently
- Removed internal schedule checks from `SdTrainingStrategy.calculate_val_loss()` and `SdxlTrainingStrategy.calculate_val_loss()` — caller now owns the scheduling decision
- Deleted `calculate_val_loss_check()` from `trainer_utils.py` — all scheduling now goes through `ValidationScheduler`
- Typed `Trainer.latent_strategy`/`te_strategy` as generic `CachingStrategy` instead of SDXL-specific concrete classes

### Fixed

- **Manifest cache hash didn't include `validation_split`/`validation_seed`** — switching between configs with different validation splits silently reused the cached manifest, resulting in no validation data
- **Device mismatch in `process_val_batch`** (SD and SDXL) — `total_loss` was initialized on CPU while loss tensors are on CUDA, causing `RuntimeError` during validation
- **Duplicate LR metric emission** — `generate_step_logs` used a `for...else` construct where the `else` block always executed (nothing `break`s), writing overlapping LR keys to trackers every step. Removed the `else` block
- **LR index math** — Fallback LR naming gave wrong labels when TE is trained (`textencoder` was never assigned; all groups got `unet`). Fixed index formula to `i - (1 if train_te else 0)` and adjusted group-naming threshold
- **`init_trackers` `wandb_run_name` overwrite** — `log_tracker_config` replaced `init_kwargs` entirely, dropping `wandb_run_name`. Now deep-merges on top of existing kwargs
- **Pre-existing test bugs** — 4 `TestInitTrackers` tests in `test_training_trainer_utils.py` passed the wrong type (full config instead of `LoggingConfig`) to `init_trackers`, silently passing due to `hasattr` guards

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
