# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2026-03-26]

### Changed

- **Training orchestration now reads in clearer shared phases without changing the active training behavior** — The shared runner/phase layer now makes epoch outcomes, eval-side execution, and startup/finalization sequencing more explicit.
  - Added `library/training/phases/orchestration_helpers.py` with small phase-shared helpers for monitored phase lifecycle and the common eval-side sampling/validation flow.
  - `library/training/phases/caching.py` now uses the shared monitored-phase helper instead of open-coded resource monitor lifecycle pairing.
  - `library/training/phases/training_loop.py` now uses explicit `EpochContext` / `EpochRunResult` dataclasses plus smaller helpers for epoch preparation, step execution, and conditional epoch finalization, so partial-epoch behavior and step-cap interruption are easier to follow.
  - `library/training/runners/trainer.py` now reads more like top-level orchestration: startup initialization, startup eval actions, and finalization are split into smaller named helpers instead of large mixed blocks.
  - Step-triggered eval actions in `training_loop.py` and startup eval actions in `trainer.py` now share the same eval-mode/sampling/validation orchestration path.
  - Updated trainer/phase unit tests and training-loop integration coverage to reflect the clearer orchestration boundaries.
- **EDM2 now has an active preset/example entry surface** — The feature is no longer only discoverable from nested defaults and source code.
  - Added `configs/presets/sdxl_peft_edm2.yaml` as a direct SDXL PEFT preset with EDM2 weighting, importance weighting, scheduler use, and visualization enabled.
  - Added `configs/examples/edm2_sdxl_peft.yaml` as a runnable example config that starts from `presets/sdxl_peft` and layers on EDM2-specific overrides.
  - `tests/unit/test_configs.py` now verifies that both the direct preset and the runnable example compose correctly through Hydra.

### Fixed

- **Accelerator logging directories are now built in a platform-correct way** — The pure accelerator config path no longer hardcodes POSIX separators when deriving the run log directory.
  - `library/training/trainer_utils.py` now uses `os.path.join(...)` when building the computed accelerator `project_dir`.
  - This restores the expected `tensorboard` default-path and `wandb` side-effect behavior on Windows and keeps `tests/unit/training/test_training_trainer_utils.py` green across platforms.

## [2026-03-25]

### Changed

- **EDM2 config wiring is now explicit in the active config pipeline** — The nested loss schema no longer leaves EDM2/SNR conflict handling stranded in an unused helper.
  - `library/config/config_validation.py` now normalizes EDM2 importance-weighting conflicts against `loss.snr.*` during `prepare_config()`, so the active nested config shape matches the intended auto-fix behavior.
  - `library/losses/edm2_loss_utils.py` now accepts the current nested SNR config explicitly when conflict handling is used directly.
  - `tests/unit/test_config_validation.py` and `tests/unit/losses/test_losses_edm2_loss.py` now cover the nested EDM2/SNR conflict path.
- **EDM2 config now reads as one coherent feature surface** — The active schema no longer exposes a long flat list of `edm2_loss_weighting_*` fields.
  - `library/config/dataclasses/loss.py` now keeps EDM2 under one `EDM2Config` root with nested `optimizer`, `importance`, and `visualization` sub-configs plus concise core fields like `enabled`, `num_channels`, and `initial_weights`.
  - `configs/_defaults/loss/default.yaml`, the active EDM2 utilities, strategy checks, and touched tests were updated to the new `loss.edm2.*` shape.
- **EDM2 runtime state now behaves like one subsystem instead of scattered trainer side fields** — The active training path now groups the optional loss-weighting sidecar more cleanly.
  - Added `library/losses/edm2_runtime.py` with a typed `EDM2LossRuntime` bundle.
  - `Trainer` now owns a single `edm2` runtime bundle instead of separate `_edm2_model`, `_edm2_optimizer`, `_edm2_lr_scheduler`, and EDM2-specific loss-tracking fields.
  - `library/training/phases/training_loop.py` now reads/writes EDM2 behavior through that runtime bundle for grad-sync accumulation, optimizer stepping, checkpoint side-saves, and scaled-loss tracking/logging.
  - Updated phase/integration trainer fixtures and trainer tests to reflect the bundled runtime shape.
- **Trainer-owned loss modification now has a generic seam** — EDM2 no longer leaks into the shared diffusion batch-processing contract.
  - Added `library/losses/loss_modifiers.py` with the generic `BatchLossOutput` / `LossModifierOutput` types, a `LossModifier` protocol, `NoOpLossModifier`, and `build_loss_modifier(...)`.
  - Added `library/losses/edm2_modifier.py` so EDM2 is a separate concrete implementation instead of the generic seam being wrapped around an EDM2-shaped runtime object.
  - `library/strategies/base/contracts.py` now points strategies at the shared `BatchLossOutput`, and the SD / SDXL diffusion strategies now return base per-sample loss plus timesteps without accepting an `edm2_model`.
  - `library/training/phases/training_loop.py` now applies the trainer-owned loss modifier after `process_batch()` returns, calls `optimizer_step()` through the generic seam, and reads optional modifier metrics through `LossModifierOutput.metrics`.
  - Step/epoch/final sidecar checkpoint saves, modifier LR logging, modifier metric accumulation, and modifier plotting now also flow through the generic loss-modifier hooks instead of the active path reaching into `trainer.edm2`.
  - `library/training/runners/trainer.py` now constructs the active modifier through `build_loss_modifier(...)`, and the last `trainer.edm2` / `edm2_runtime.py` compatibility layer has been removed from the active codebase.
- **EDM2 internals now live in a clearer package split** — The EDM2 package no longer keeps factory, plotting, and validation helpers mixed into one module.
  - Added `library/losses/edm2/factory.py`, `library/losses/edm2/plotting.py`, and `library/losses/edm2/validation.py`, while reducing `library/losses/edm2/edm2_loss_utils.py` to a thin compatibility facade.
  - `library/losses/loss_modifiers.py` now normalizes modifier metrics and enforces namespaced metric keys, so future modifiers have a clearer logging contract than a loose free-form dict.
  - Added regression coverage for metric normalization and EDM2 sidecar save/load hooks in `tests/unit/losses/test_losses_edm2_loss.py`.
- **Dormant EDM2 Laplace sampling now fails fast** — The unused config toggle no longer looks supported when no active training path wires it up.
  - `library/config/config_validation.py` now raises a clear error when `loss.edm2.laplace_timestep_sampling=true`.
  - Added regression coverage in `tests/unit/test_config_validation.py`.

## [2026-03-24]

### Changed

- **Accelerator setup now has an explicit pure-computation split** — `prepare_accelerator()` no longer mixes all config derivation and side-effect handling in one block.
  - `library/training/trainer_utils.py` now provides `compute_accelerator_config()` plus a typed `AcceleratorConfig` container for the derived accelerator constructor args and deferred wandb setup.
  - `prepare_accelerator()` now applies the wandb/env/filesystem side effects after config computation, then instantiates `Accelerator` from the computed setup.
  - `tests/unit/training/test_training_trainer_utils.py` now covers both the pure config derivation path and the retained wandb side effects.
- **Resume hook state flow is now explicit across training modes** — Checkpoint resume metadata no longer comes back through closure-bound containers.
  - `library/training/checkpointing.py` now defines a shared `ResumeState` dataclass plus small `train_state.json` read/write helpers.
  - PEFT and fine-tune modes now register their own accelerator hooks locally while reusing the shared metadata helpers, and `optimizer.py` restores counters from `resume_state.step` directly.
  - Updated optimizer, fine-tune mode, resume-logic, and checkpoint I/O tests to assert the explicit resume-state flow.
- **Legacy IPEX workaround code removed** — The old Intel Extension for PyTorch compatibility path is gone from the repo.
  - Deleted the unused `library/performance/ipex/` package.
  - Removed the leftover `init_ipex()` shim from `library.utils.device_utils` and dropped its no-op unit test.
- **Sampling config now supports inline/default generation parameters** — Sampling can now be configured from YAML without giving up prompt-file workflows.
  - `SamplingConfig` and `configs/_defaults/output/default.yaml` now expose `sample_prompt`, `sample_prompt_file`, `sample_negative_prompt`, `sample_width`, `sample_height`, `sample_steps`, `sample_cfg_scale`, and `sample_seed`.
  - `library/training/sample_generation.py` now uses prompt-file values first, then falls back to sampling-config defaults, and supports a single inline `sample_prompt` when no prompt file is configured.
  - Added unit coverage for inline prompt fallback, config-default resolution, and prompt-file override precedence in `tests/unit/training/test_training_sample_generation.py`.
- **Strategy/base naming cleanup landed in the active path** — The shared strategy contract file and cache-engine boundary now use the clearer names discussed in roadmap follow-ups.
  - `library/strategies/base/training.py` is now `library/strategies/base/contracts.py`, and current-facing docs now point to the renamed contract module.
  - The active cache-engine surface now uses `CacheBackend` naming in `library/data/caching_engine.py`, `Trainer`, dataloader construction, and the touched unit/integration tests.
- **Tokenizer cache config now lives with data caching settings** — The active config surface no longer treats tokenizer caching as part of model identity.
  - Moved `tokenizer_cache_dir` from `model.*` to `data.caching.*` in the dataclass schema and Hydra default fragments.
  - Updated the active SD / SDXL strategy construction paths, the active SDXL textual inversion script path, and related unit tests to read the cache location from `cfg.data.caching.tokenizer_cache_dir`.
- **Config validation now rejects ambiguous sampling cadence settings** — Active configs can no longer set both `sample_every_n_steps` and `sample_every_n_epochs` and silently fall through to epoch precedence.
  - `library/config/config_validation.py` now raises a clear error when both sampling cadence fields are set at once.
  - `tests/unit/test_config_validation.py` now covers that new guard, a dataset-side `cache_dir` prepare-config default, and the existing TE offload vs. TE-output caching conflict.
- **Config-validation edge-case coverage is broader and stricter** — The centralized validator now has regression coverage for more normalization paths and conflict branches, including one real FP8 validation fix.
  - `tests/unit/test_config_validation.py` now covers learning-rate inheritance in `prepare_config()`, tracker/resource-monitor normalization, validation cadence disabling, partial-config tolerance, more resource-monitor enum validation, textual-inversion mode requirements, TE training/caching conflicts, and dataset-group cacheability fallbacks.
  - `library/config/config_validation.py` now validates `fp8_base` / `fp8_base_unet` against `performance.precision.mixed_precision` correctly while still tolerating partial OmegaConf test configs that omit unrelated precision keys.

## [2026-03-23]

### Changed

- **Hydra config composition now has an active entry-config audit** — The current shipped entry configs are now checked against the nested dataclass schema instead of relying on a few spot checks.
  - `tests/unit/test_configs.py` now composes every active entry config under `configs/` (excluding `_defaults/` fragments and the example override file) so schema drift is caught by a single regression test.
  - The generic internal baseline config was updated to the current nested defaults layout and moved under `configs/_defaults/config/default.yaml` so ad hoc composition and config tooling still have a valid baseline without leaving a fake entry config at the top level.
  - Removed the empty SDXL placeholder namespace from the active schema/defaults path: the SDXL root dataclasses no longer expose a no-op `sdxl` field, the top-level SDXL config files no longer compose `sdxl: default`, and the dead placeholder files were deleted.
  - `configs/model/default.yaml` is now truly shared again: it no longer chooses a model family, `ModelConfig.model_type` is required in the schema, and the top-level SD / SDXL configs now set `model.model_type` explicitly so run identity cannot drift from the config entrypoint.
  - `configs/model/default.yaml` now still shows `model_type` explicitly, but as `null` instead of a misleading family default, and `config_validation.py` raises a clear error if it is left unset.
  - The duplicated SD / SDXL root dataclass definitions were removed from the active path entirely: top-level presets now compose through one shared `run_schema`, while `peft` and `textual_inversion` remain the only mode-specific nested config sections.
  - Active configs still declare `mode` explicitly (`peft`, `finetune`, or `textual_inversion`), and `config_validation.py` now enforces that `mode` matches the presence or absence of the `peft` / `textual_inversion` sections.
  - The old entrypoint-flavored dataset validators (`validate_sd_*`, `validate_sdxl_*`) were replaced in the active path with one generic `validate_dataset_groups(...)` helper that derives bucket-step rules from `model.model_type` and handles dataset-backed TE-cacheability checks without encoding script names in the API.
  - Config files are now organized more clearly by role: dataclass-backed default fragments live under `configs/_defaults/`, primary user entry presets under `configs/presets/`, example entry configs under `configs/examples/`, benchmark configs under `configs/benchmarks/`, and test-only configs under `configs/tests/`. Hydra defaults, script decorators, and moved config references were updated to match the new layout.
  - The `_defaults/` reorganization now keeps preset files readable: package ownership lives inside each `_defaults/*/default.yaml` via `# @package ...`, so top-level presets can use `- _defaults/optimizer: default` instead of the uglier Hydra remap form `- _defaults/optimizer@optimizer: default`.
  - Added missing Hydra `_self_` entries to the primary SD / SDXL top-level configs to remove composition-order warnings.
- **SDXL training-time text conditioning now routes through the strategy encoding/tokenization seam** — The SDXL diffusion path now follows the same contract shape SD already uses instead of reaching directly into model helpers from the training facet.
  - `library/strategies/sdxl/diffusion.py` now resolves live text conditioning through `tokenize()`, `tokenize_with_weights()`, `encode_tokens()`, `encode_tokens_with_weights()`, and `get_models_for_text_encoding()` rather than calling `encode_input_ids_sdxl(...)` directly.
  - This keeps diffusion ownership focused on orchestration while the SDXL tokenization/encoding facets continue to own model-family behavior.
  - SDXL weighted-caption support is now available on the live-encoding training path in the same way SD already handled it; TE-output caching policy remains a separate concern.
- **SDXL strategy tests now assert the hook-based text-conditioning path directly** — Updated unit coverage so the training facet checks the strategy seam rather than the old direct model-helper call path.
  - `tests/unit/strategies/test_strategies_sdxl.py` now verifies caption fallback, weighted-caption encoding, and cached-output short-circuit behavior via the active SDXL strategy methods.

## [2026-03-20]

### Changed

- **Sampling ownership was pulled back to the model strategies** — Shared sampling is generic again, while SD and SDXL now own their concrete pipeline construction and runtime setup directly.
  - `library/training/sample_generation.py` no longer chooses SD vs. SDXL pipelines, carries sampling-runtime state bundles, or relies on callback/builder-style indirection.
  - `library/strategies/sd/sampling.py` and `library/strategies/sdxl/sampling.py` now unwrap models, build their concrete pipelines locally, and restore runtime state around the shared sampling loop.
  - The deprecated SD / SDXL sampling wrappers were adjusted to keep working with the simpler shared helper shape.
- **Config validation structure tightened** — Model/profile-specific validation no longer has to accumulate in `config_validation.py`.
  - Kept the cleanup inside `library/config/config_validation.py` instead of introducing another validation module, while still separating generic config checks from smaller internal helper functions.
  - `validate_config()` now enforces known model-family bucket-step requirements directly from config (`sdxl` -> 32-step buckets, `sd1`/`sd15`/`sd2` -> 64-step buckets) instead of relying only on script-owned dataset-group validators.
  - `validate_config()` now rejects TE-output caching together with caption mutation settings that change text conditioning over time (`shuffle_caption`, `caption_dropout_rate`, `token_warmup_step`, `caption_tag_dropout_rate`).
  - The script-owned dataset validators (`validate_sd_peft()`, `validate_sdxl_peft()`, `validate_sd_textual_inversion()`, and `validate_sdxl_textual_inversion()`) remain in the same file for now rather than being split out.
- **SDXL strategy tests updated for the split concern-file layout** — The strategy unit tests now reflect the current SDXL architecture instead of the old `sdxl/training.py` monolith.
  - Updated `tests/unit/strategies/test_strategies_sdxl.py` to patch helpers from `encoding.py` and `diffusion.py` rather than `training.py`.
  - Replaced the stale `_get_text_cond(...)` test path with `_get_text_conds(...)` and added a direct concern-ownership assertion for the composed `SdxlTrainingStrategy`.
- **SD strategy now mirrors the split concern-file layout** — SD 1.5/2.0 strategy organization now follows the same contract-owned file layout as SDXL.
  - Split SD model loading, model preparation, checkpointing, sampling, denoiser calling, diffusion training, and validation into self-titled files under `library/strategies/sd/`.
  - Reduced `sd/training.py` to strategy assembly and init/wiring, with the mixin order matching the base `TrainingStrategy` facet order.
  - Updated `SdTextEncodingStrategy` for mixed-in use with `SdTokenizeStrategy`, added a training-facing `SdCachingStrategy`, and refreshed `tests/unit/strategies/test_strategies_sd.py` to assert the new concern ownership.

### Fixed

- **Post-refactor strategy leftovers found by benchmark runs** — Follow-up benchmark coverage flushed out the remaining mixed-in state and stale naming issues in the active path.
  - `SdxlTextEncodingStrategy` now tolerates mixed-in use when `_tokenizers` was never initialized directly, which fixed SDXL weighted-prompt sampling through the LPW pipeline.
  - `SdxlCheckpointingStrategy.save_model_checkpoint()` now unwraps `trainer.denoiser` instead of the removed `trainer.unet` field.
  - `SdTextEncodingStrategy` exposes `clip_skip` again for direct facet tests, and the SD strategy tests now call `process_batch(...)` with the active `denoiser=` seam rather than stale `unet=` kwargs.
- **Text-encoder benchmark config and validation feedback now line up** — The text-encoder benchmark config no longer re-enables TE caching through inherited disk-cache settings, and the resulting validation message is clearer.
  - `configs/test_text_encoder.yaml` now disables both `cache_text_encoder_outputs` and `cache_text_encoder_outputs_to_disk`.
  - The TE-training validation error now says plainly to disable TE output caching or set text-encoder LR to `0`.

## [2026-03-19]

### Changed

- **SDXL facet ownership cleanup started** — The SDXL strategy now composes its existing tokenization / encoding / caching facet files instead of re-implementing those contract methods inside `sdxl/training.py`.
  - `SdxlTrainingStrategy` now inherits `SdxlTokenizeStrategy`, `SdxlTextEncodingStrategy`, and `SdxlCachingStrategy`, while `sdxl/training.py` keeps only strategy assembly and SDXL-specific training behavior.
  - `sdxl/caching.py` now owns the training-facing caching facet methods while continuing to host the SDXL-specific cache/data-backend handlers used by the data pipeline.
  - `SdxlTextEncodingStrategy` now supports both standalone construction and mixed-in use with `SdxlTokenizeStrategy`, so active SDXL call sites do not need duplicate tokenizer ownership.
  - `sdxl/loading.py` now owns the `ModelLoadingStrategy` facet for SDXL, moving model loading and RamTorch/xformers setup out of `sdxl/training.py`.
  - `sdxl/model_preparation.py` now owns the `ModelPreparationStrategy` hooks for SDXL, including CLIP embedding prep, precision-cast decisions, and the TE1 freeze post-processing rule.
  - `sdxl/validation.py` now owns the SDXL validation facet, including the fixed-timestep validation-batch helper and validation-loss loop.
  - `sdxl/checkpointing.py` now owns the SDXL checkpointing facet, including metadata helpers and full-model save logic for both SD-format and Diffusers-format checkpoints.
  - `sdxl/sampling.py` now owns the SDXL sample-generation facet, leaving `sdxl/training.py` focused on the diffusion / UNet-calling core.
  - `sdxl/denoiser.py` now owns the SDXL denoiser-calling facet, including the UNet-call bridge and micro-conditioning tensor extraction.
  - `sdxl/diffusion.py` now owns the SDXL diffusion-training facet, including training-time text-conditioning resolution, noise/target assembly, and `process_batch(...)`.
  - The SDXL second-pass polish now mirrors the base contract more closely: the mixin order in `sdxl/training.py` follows the `TrainingStrategy` facet order, and the stale strategy-side `on_step_start(...)` validation hook has been removed now that step-start ownership lives on the training-mode layer.
- **Trainability policy moved out of the base strategy contract** — Generic LR-driven trainability decisions now live in explicit shared helpers instead of `ModelPreparationStrategy`.
  - Centralized `should_train_denoiser()`, `should_train_text_encoder()`, and per-TE flag resolution in the shared optimizer helper layer.
  - `FineTuneMode`, `PeftMode`, and shared optimizer setup now call the trainability helpers directly.
- **Model-preparation hooks are now explicit** — The remaining `ModelPreparationStrategy` hooks no longer rely on silent base defaults.
  - `cast_text_encoder()`, `cast_vae()`, `cast_denoiser()`, and `post_process_trainable()` are now explicit strategy methods instead of inherited default behavior.
  - SD and SDXL now provide the current behavior intentionally, making the remaining model-preparation seam more honest.
- **Base strategy facet ownership tightened** — The remaining training-local tokenization / text-encoding hooks now live on the existing concern facets instead of directly on `TrainingStrategy`.
  - Moved `tokenize_captions()` to `TokenizationStrategy`.
  - Moved `encode_te_outputs_in_memory()` and `get_models_for_text_encoding()` to `TextEncodingStrategy`.
  - Updated base-strategy tests to assert the new facet placement.
- **Helper / runner config boundary clarified** — Training helpers now use narrower config surfaces only where that meaningfully improves ownership, while trainer-facing orchestration helpers stay `cfg`-centric.
  - `prepare_latents()` now takes `CachingConfig` plus the explicit runtime values it needs, instead of the full root config / accelerator.
  - Generic LR-driven trainability policy stays in the shared optimizer helper layer.
  - `get_noise_scheduler()`, `log_training_diagnostics()`, and `create_training_metadata()` remain `cfg`-based where they are effectively trainer-facing orchestration helpers.

### Removed

- **Strategy contract no longer owns generic trainability and denoiser wrapping helpers** — The remaining non-model-specific helper seams were removed from the base strategy surface.
  - Removed the LR-query wrapper methods from `ModelPreparationStrategy`, leaving only the genuinely model-owned preparation hooks on that facet.
  - Removed `prepare_denoiser_with_accelerator(...)` from the base strategy contract; `FineTuneMode` and `PeftMode` now call `accelerator.prepare(...)` directly for denoiser wrapping in the non-DeepSpeed path.
  - Updated the base-strategy audit/tests to reflect that denoiser accelerator preparation is shared mode logic, not a model-family seam.

### Fixed

- **Validation contract cleanup for active strategies** — The live runner/strategy validation path now matches the current shared contract more closely.
  - Fixed the active step-validation call in `training_loop.py` to pass `epoch`, `batch`, and `train_text_encoder` in the correct order to `calculate_val_loss(...)`.
  - Removed the stale `self.on_step_start(...)` call from SD strategy validation after step-start ownership moved to the training-mode layer.
  - Aligned `SdTrainingStrategy.calculate_val_loss(...)` with the base validation contract by returning the same 2-tuple shape as SDXL and the shared runner path.

## [2026-03-18]

### Changed

- **Shared training contract naming now uses `denoiser`** — The active trainer/base strategy seam now uses denoiser-oriented terminology where the code is model-family-agnostic.
  - Renamed the active base hooks and trainer/model-prep state to `denoiser` terminology (`DenoiserCallingStrategy`, `is_train_denoiser`, `cast_denoiser`, `trainer.denoiser`, `trainer.denoiser_weight_dtype`, etc.).
  - Updated shared helpers like `sample_images_common()` and `append_lr_to_logs()` to use denoiser-oriented params/docs in the active path.
  - Added a `LearningRatesConfig.denoiser` alias over the existing `learning_rates.unet` field so shared code can move forward without forcing config churn yet.
  - Left concrete SD / SDXL internals, adapter APIs, and compatibility metadata/config keys on `unet` where they still refer to genuinely UNet-shaped or compatibility-sensitive surfaces.
- **TrainingStrategy now owns tokenization/text-encoding behavior directly** — `TrainingStrategy` now inherits the `TokenizationStrategy` and `TextEncodingStrategy` facets, while `SdTrainingStrategy` and `SdxlTrainingStrategy` implement `tokenize()`, `tokenize_with_weights()`, `encode_tokens()`, and `encode_tokens_with_weights()` directly.
  - Removed `_tokenize_strategy` / `_text_encoding_strategy` instance state from `TrainingStrategy`.
  - Removed `get_tokenize_strategy()`, `get_tokenizers()`, and `get_text_encoding_strategy()` from the active strategy contract.
  - SD and SDXL helper modules (`sd/tokenization.py`, `sd/encoding.py`, `sdxl/tokenization.py`, `sdxl/encoding.py`) now expose direct helper functions that the concrete training strategies call.
- **Concrete strategies are now born ready** — `SdTrainingStrategy(cfg)` and `SdxlTrainingStrategy(cfg)` now load their tokenizer/runtime state in the concrete strategy constructor instead of relying on a second lifecycle step.
  - Removed `TrainingStrategy.initialize(cfg)` from the base contract.
  - `Trainer.setup()` no longer activates strategy internals; it only consumes `strategy.tokenizers`.
  - Active SDXL scripts now instantiate ready strategy objects before passing them to `Trainer`.
- **Tokenizer ownership now matches the tokenization facet** — `tokenizers` now lives as a tokenization-facet contract with concrete storage on the SD / SDXL training strategies, instead of as a shared data field on the composed `TrainingStrategy` base.
- **Sampling now receives the training strategy itself** — `sample_images_common()` and the SDXL LPW sampling pipeline no longer receive separate tokenization/text-encoding runtime objects; SDXL sampling now uses the concrete `TrainingStrategy` instance as the prompt tokenization/encoding surface.
- **Strategy tests updated for direct facets** — Base, SD, SDXL, and resume-logic tests now exercise the direct strategy-facet API instead of the removed sub-strategy factories/instance state.

### Removed

- **Active strategy contract pruned further** — Deprecated-only hooks and misleading base decorations were removed from the live strategy surface.
  - Removed the unused `on_validation_step_end()` runtime hook from `TrainingRuntimeStrategy`.
  - Removed the unused `TrainingRuntimeStrategy.on_step_start()` hook and the shared-loop call site; active step-start behavior now lives on the training mode layer.
  - Removed `ValidationStrategy.validate_extra_config()` from the active strategy contract and from the SD / SDXL training strategies.
  - Removed `ModelPreparationStrategy.is_text_encoder_not_needed_for_training()`, which was only referenced by deprecated script paths.
  - Removed `SdxlTrainingStrategy.cache_text_encoder_outputs_if_needed()`, which was only referenced by deprecated PEFT script paths.
  - Removed the now-misleading `@dataclass` decoration from `TrainingStrategy`, `SdTrainingStrategy`, and `SdxlTrainingStrategy`.

## [2026-03-12]

### Removed

- **Legacy caching classes** — Deleted `LatentsCachingStrategy` and `TextEncoderOutputsCachingStrategy` base classes (`library/strategies/base/caching.py`), along with their concrete implementations `SdSdxlLatentsCachingStrategy` and `SdxlTextEncoderOutputsCachingStrategy`.
- **Singleton `set_strategy`/`get_strategy` methods** — Removed from `TokenizationStrategy` and `TextEncodingStrategy` in `library/strategies/base/contracts.py`. These patterns are replaced by instance ownership in `TrainingStrategy`.
- **Legacy factory methods** — Removed `get_latents_caching_strategy` from `SdTrainingStrategy` and `SdxlTrainingStrategy`, and `get_text_encoder_outputs_caching_strategy` from `SdxlTrainingStrategy`. The active pipeline uses `create_latent_caching_strategy` / `create_te_caching_strategy` instead.
- **Stale singleton calls** — Replaced broken `get_strategy()` calls in deprecated `dataset.py` with instance variables.
- **Deprecated script methods** — Removed `get_latents_caching_strategy` from `scripts/sdxl_textual_inversion.py`.
- **Obsolete tests** — Removed ~400 lines of tests for deleted classes and singleton patterns across all three strategy test files.

### Changed

- **CLIP-specific base defaults removed** — `prepare_text_encoder_grad_ckpt_workaround` and `prepare_text_encoder_fp8` no longer have CLIP-specific implementations in the generic `ModelPreparationStrategy` base. Both now raise `NotImplementedError`; SD and SDXL strategies provide the CLIP-specific overrides.

## [2026-03-11]

### Changed

- **Strategy internal state collapse** — `_tokenize_strategy` and `_text_encoding_strategy` are now owned as instance state by `TrainingStrategy`, no longer stored on `Trainer` or passed through runner code.
  - Added `TrainingStrategy.initialize(cfg)` lifecycle method and `tokenizers` property.
  - Removed the two pass-through params from `sample_images()`, `calculate_val_loss()`, and `process_batch()` across base, SD, and SDXL strategies.
  - `Trainer.setup()` now calls `strategies.initialize(cfg)` instead of manually creating and storing these objects.
  - Updated tests to set internal strategy state directly instead of passing params.

## [2026-03-10]

### Changed

- **Cache backend naming cleanup** — Training and data pipeline code now consistently refer to the active caching-engine implementation surface as cache backends rather than generic handlers or “strategies”.
  - `Trainer` fields are now `latent_cache_backend` / `te_cache_backend`.
  - Training phases and dataloader constructors use the new names to make the engine boundary clearer.
  - Integration/unit tests were updated to match the renamed arguments.

## [2026-03-08]

### Changed

- **Strategy boundary cleanup follow-up** — Shared strategy defaults no longer assume CLIP-specific text-encoder preparation in the generic base training contract.
  - `ModelPreparationStrategy` now leaves text-encoder grad-checkpoint / FP8 prep to concrete model-family strategies.
  - SD and SDXL training strategies now own their CLIP-specific text-encoder preparation behavior directly.
  - Added unit coverage in `tests/unit/strategies/test_strategies_base.py` for the stricter base contract behavior.
  - Deprecated TE-output caching hooks on the base training strategy now default to optional legacy behavior (`None` / move encoders to device), so active strategies no longer need no-op overrides just to satisfy deprecated SD script paths.
  - In `base/contracts.py`, `tokenize_captions()` now lives on `TokenizationStrategy`, while `get_text_encoding_strategy()`, `get_models_for_text_encoding()`, and `encode_te_outputs_in_memory()` stay on `TrainingStrategy`. The training-side `CachingStrategy` no longer owns text-encoding methods, so its scope is narrower and closer to its name.
  - Current CLIP-family token/chunk shaping and tokenizer loading now live under `library/models/sd/tokenizer.py`, and the SD / SDXL tokenization strategy classes delegate to that shared model-layer helper instead of keeping the behavior in `base/tokenization.py`.
  - The runtime tokenization and text-encoding contracts now live in `base/contracts.py` (`TokenizationStrategy`, `TextEncodingStrategy`), while `TrainingStrategy` keeps the provider/wiring hooks (`get_tokenize_strategy()`, `get_tokenizers()`, `get_text_encoding_strategy()`, etc.).
  - Removed the temporary `base/tokenization.py` and `base/encoding.py` shim modules after switching the remaining deprecated-dataset and unit-test callers to the canonical `base/contracts.py` contract and shared model-layer token helper APIs.
  - Removed the singleton-era latent / TE-output caching hooks from the main `base/contracts.py` caching contract, and stopped `Trainer.setup()` from registering legacy latent caching singletons that the manifest-based runner does not use.
  - SDXL sampling now receives tokenization and text-encoding strategies explicitly through `sample_images()` / `sample_images_common()` and the SDXL LPW pipeline, so the main `Trainer` no longer registers tokenization or text-encoding singletons for the active runner path.
  - SDXL training-side caption tokenization now routes through the shared CLIP-family helper path in `library/models/sd/tokenizer.py`, removing the duplicate `tokenize_sdxl_captions()` implementation from `sdxl/training.py` and aligning live caption fallback / in-memory TE caching with `SdxlTokenizeStrategy`.
  - Removed the temporary `[DEBUG]` info logging from `SdxlTrainingStrategy._get_text_cond()` now that the explicit strategy wiring path is in place.
- **SD new-pipeline strategy support** — SD strategy code now supports the active `CachingEngine` / `TrainingDataset` path instead of relying on “not migrated yet” compatibility stubs.
  - `SdTrainingStrategy` now creates real new-pipeline latent and text-encoder caching strategies, tokenizes captions for epoch-token caching, and computes in-memory TE outputs for the shared caching phase.
  - SD training/validation batch processing now accepts new batch keys (`input_ids`, `text_encoder_outputs`) while preserving the live-encoding fallback path.
  - Added `SdTextEncoderPipelineStrategy` and focused SD pipeline-strategy coverage in `tests/unit/data/test_pipeline_strategies.py` and `tests/unit/strategies/test_strategies_sd.py`.

## [2026-03-07]

### Changed

- **Strategy-owned epoch-token / TE-cache shape hooks** — Shared training phases no longer hardcode SDXL token encoder names or TE cache model packing.
  - Added `get_token_cache_encoder_names()` and `build_te_cache_model_bundle()` to the strategy caching surface.
  - `training_loop.py` now routes per-epoch token cache naming through `_maybe_cache_epoch_tokens(...)` and strategy-provided encoder names.
  - `caching.py` now routes disk TE caching through a strategy-provided model bundle instead of packing `(*text_encoders, *tokenizers)` in shared phase code.
- Added unit coverage in `tests/unit/training/phases/test_caching.py` to assert disk TE caching uses the strategy-provided bundle.
- **Text-encoder model-logic extraction cleanup** — SD text-encoder hidden-state logic now lives under `library/models/sd/text_encoder.py`, and SDXL text-encoder helpers now centralize token-ID encoding/weight application more consistently under `library/models/sdxl/text_encoder.py`.
  - `library/strategies/sd/encoding.py` now delegates SD hidden-state extraction and weight application to model helpers.
  - `library/strategies/sdxl/encoding.py` and `library/strategies/sdxl/training.py` now use shared SDXL model helpers for input-ID encoding instead of duplicating that logic in strategy code.
- **Phase 2 strategy-facet cleanup** — Shared defaults and runtime hooks no longer live as a grab-bag on the composed `TrainingStrategy` class.
  - Moved lazy UNet loading to `ModelLoadingStrategy`.
  - Added dedicated `ModelPreparationStrategy`, `DiffusionTrainingStrategy`, and `TrainingRuntimeStrategy` capability mixins for shared defaults used by model prep and the training loop.
  - Moved validation-loss ownership to `ValidationStrategy`.
  - Added focused base-strategy unit coverage for facet placement and helper behavior in `tests/unit/strategies/test_strategies_base.py`.
- **Shared training-mechanics extraction from `base/training.py`** — Generic scheduler/loss/runtime helpers now live in shared utility modules instead of the strategy base contract.
  - Moved noise-scheduler creation to `library/training/noise_utils.py`.
  - Moved shared loss post-processing assembly to `library/losses/loss_weighting.py`.
  - Moved gradient all-reduce and validation RNG save/restore helpers to `library/training/trainer_utils.py`.
  - Updated `Trainer`, `training_loop.py`, and concrete SD / SDXL strategy code to call the shared utilities directly.

## [2026-03-06]

### Changed

- **Epoch-end side-effect gating cleanup** — Removed the preserved compatibility path that entered eval mode whenever `save_every_n_epochs` was configured.
  - `compute_epoch_end_actions()` now sets `should_enter_eval_mode` only when a real epoch-end action is scheduled (`should_sample` or `should_save_epoch`).
  - `_finalize_epoch()` now calls `sample_images(...)` only when `epoch_end_actions.should_sample` is `True`.
  - Updated trigger unit coverage to assert no eval transition when neither epoch-end sampling nor epoch-end save fires.
- **Resource monitor startup caveat for DeepSpeed/ZeRO** — `emit_startup_component_memory(...)` now accepts DeepSpeed context and appends an explicit partitioning/offload caveat line when DeepSpeed is enabled.
  - `Trainer._log_training_info()` now passes `deepspeed_enabled` and `deepspeed_zero_stage` into the monitor startup estimate call.
- **Benchmark runner monitor override support** — `run_benchmark.ps1` now accepts `-ResourceMonitorMode` (`off|basic|sampled|deep`) so baseline/off vs monitored runs can be compared without editing configs.

### Added

- Added unit coverage for DeepSpeed startup caveat emission in `tests/unit/logging/test_resource_monitor.py`.
- Added integration coverage for real `BasicResourceMonitor` JSONL events through `run_training_loop()` hooks in `tests/integration/test_training_loop_integration.py`:
  - PEFT-like run smoke
  - Fine-tune-like run smoke

### Fixed

- Fixed stale integration-test import in `tests/integration/test_checkpoint_io.py`: `load_metadata_from_safetensors` now imports from `library.utils.model_metadata` (canonical module) instead of `library.training.checkpointing`.
- Fixed `prepare_config()` normalization regressions on partial/mock configs: logging/resource-monitor/validation cadence comparisons now guard numeric checks, preventing `TypeError` when test fixtures use `MagicMock` placeholders.
- Fixed training-loop epoch finalization on step-capped partial epochs: epoch-end sampling/saving now runs only after the epoch body fully consumes its remaining batches, avoiding premature epoch-end side effects when `max_train_steps` cuts an epoch short.
- Fixed resource monitor lifecycle cleanup on failures:
  - `training_epoch_*` phases now always emit `phase_end`, even if the epoch body raises before normal finalization.
  - `Trainer.train()` now guarantees `resource_monitor.end_session()` via top-level `finally`, so JSONL/session summaries are closed even when training aborts mid-run.

## [2026-03-05]

### Added

- Added run-level metadata fields to resource monitor JSONL events:
  - `run_id`
  - `config_name`
  - `git_sha`
  - `git_dirty` (tracked-file dirty state; untracked files ignored)
- `Trainer.setup()` now passes session/config/commit metadata into `create_resource_monitor(...)` so event streams can be grouped and compared across appended runs.

## [2026-03-04]

### Added

- Implemented real `sampled`/`deep` resource monitor mode behavior in `library/logging/resource_monitor.py`:
  - daemon sampler thread with bounded queue
  - queue pressure handling via `drop_oldest` / `drop_newest` / `block` policies
  - sampled GPU-used memory collection (NVML preferred, `torch.cuda.mem_get_info` fallback)
  - deep-window gated counter collection for `deep` mode using `torch.cuda.memory_stats()`
- Added JSONL event streaming for monitor events (`session_start`, `phase_start`, `phase_end`, `step_sample`, `session_end`) with:
  - stable fixed-key schema
  - configurable flush behavior (`auto`, `line`, `batch`)
  - forced flush at phase/session boundaries
  - relative path resolution under output dir for `resource_monitor.output_jsonl`
- Added resource monitor unit coverage in `tests/unit/logging/test_resource_monitor.py` for:
  - sampled factory routing and sampler lifecycle
  - queue drop policy behavior
  - JSONL schema emission and forced-flush behavior
- Added sampled resource-monitor presets:
  - `configs/test_peft_resource_sampled.yaml`
  - `configs/benchmark_finetune_resource_sampled.yaml`
- Updated `run_benchmark.ps1` aliases/routing to support sampled resource runs (`peft_resource_sampled`, `finetune_resource_sampled`) and dynamic monitor mode override (`basic` vs `sampled`)

### Changed

- `create_resource_monitor(...)` now returns `SampledResourceMonitor` for `mode=sampled|deep` instead of degrading to basic behavior with a warning
- Resource monitor event handling is now fault-tolerant by design (internal monitor failures are downgraded to warnings and do not interrupt training flow)
- Deep mode now emits meaningful allocator diagnostics (`alloc_retries`, `ooms`, `active/reserved/inactive_split` MB) into structured events instead of debug-only console output
- Deep-window semantics were tightened (step/time bounded from first optimization step) and now emit a one-time deep-window summary block when the window closes or session ends

## [2026-03-03]

### Added

- New trigger policy helper module `library/training/phases/triggers.py` with typed contexts and action decisions:
  - `StepTriggerContext` + `compute_step_actions()`
  - `EpochEndTriggerContext` + `compute_epoch_end_actions()`
  - typed action containers for eval/validation/sampling/save decisions
- New unit test coverage for trigger policy helpers in `tests/unit/training/phases/test_triggers.py`
- New config-driven resource monitor module `library/logging/resource_monitor.py`:
  - `NoOpResourceMonitor` for strict zero-overhead `off` mode
  - `BasicResourceMonitor` for session/phase summaries and optional periodic step snapshots
  - `create_resource_monitor(...)` factory for trainer integration
- New resource monitor unit tests in `tests/unit/logging/test_resource_monitor.py`
- New benchmark/test scenario configs:
  - `configs/test_peft_validation_run.yaml`
  - `configs/test_peft_resource_basic.yaml`
  - `configs/benchmark_finetune_resource_basic.yaml`

### Changed

- `training_loop.py` now delegates sampling/checkpoint trigger decisions through typed helper functions (matching the existing validation scheduler direction) while preserving current runtime behavior
- `training_loop.py` step/epoch orchestration was split into focused internal helpers (`_run_step_side_effects`, `_emit_step_tracking_logs`, `_update_live_timestep_outputs`, `_finalize_epoch`, checkpoint artifact helpers) to reduce branch density in `run_training_loop()` without changing behavior
- Added typed resource monitor config fields to `LoggingConfig` and `configs/output/default.yaml` (mode, rank scope, cadence, queue/flush/deep settings)
- Grouped resource monitor settings under nested `output.logging.resource_monitor` via new `ResourceMonitorConfig` dataclass (replacing flat `resource_monitor_*` keys)
- Added resource monitor config normalization + strict enum validation in `library/config/config_validation.py` (mode/rank/device scope, flush/drop policy, non-negative numeric bounds)
- Replaced `BENCHMARK_RESOURCES` branches in caching/training phases with monitor hooks:
  - `run_latent_caching` and `run_te_caching` now emit `phase_start/phase_end`
  - training loop now emits per-epoch `phase_start/phase_end` and per-optimization-step `step_end`
- `Trainer.setup()` now creates/starts a monitor instance from config, `_log_training_info()` emits startup component memory estimates, and `_finalize_training()` ends the session
- `run_benchmark.ps1` now resolves nested Hydra defaults when building config snapshots, captures wrapped resource monitor log blocks more reliably, includes session summary sections in markdown, and adds benchmark aliases for PEFT validation/resource and finetune-resource scenarios
- Updated `configs/test_core.yaml` and `configs/test_finetune.yaml` to use the shared `D:/Projects/sd-scripts/benchmark_cache` path for benchmark-oriented runs

### Fixed

- **Step checkpoint epoch semantics are now consistent with epoch-end saves** — step-triggered checkpoint calls in `training_loop.py` now pass `trainer._current_epoch_state.value` (1-based) instead of the loop-local zero-based `epoch` index
- Added integration assertions in `tests/integration/test_training_loop_integration.py` to verify step checkpoints carry 1-based current epoch values (including across epoch boundaries)
- Hardened integration test fixture setup by explicitly stubbing `_validation_scheduler.should_run=False` to prevent accidental validation-path activation during checkpoint trigger tests
- Fixed resource monitor startup estimates crash when diagnostics components are passed as a list (`AttributeError: 'list' object has no attribute 'items'`) by accepting both mappings and iterable `(name, module)` pairs

## [2026-03-02]

### Changed

- **Logging Phase 2: Rank-aware logging** — Non-main processes in DDP now have root log level set to `WARNING` after accelerator init, suppressing duplicate `INFO` lines while preserving error/warning visibility on all ranks
- **Logging Phase 3: `log_every_n_steps`** — New `LoggingConfig` field to control tracker emission frequency (default: `1` = every step). `step_logging()` gated by interval; progress bar remains every step. Invalid values (`<= 0`) normalized to `1` with warning
- **Validation defaults now opt-in** — `run_at_start` and `run_at_end` default to `False` in both `ValidationConfig` and `configs/validation/default.yaml`

### Fixed

- **Stale `val_manifest.json` loading** — `get_or_create_manifest` loaded cached validation manifests even when `validation_split` was `0.0`. Now guards on `validation_split > 0` and cleans up stale files
