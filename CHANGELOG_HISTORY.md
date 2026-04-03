# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2026-03-26]

### Changed

- **The active launcher path is now config-driven for PEFT/fine-tune** — The root entry surface no longer needs one near-duplicate script per active mode/model combination.
  - Added a canonical root `train.py` launcher that composes the active nested config schema, prepares/validates config, and routes the run through shared mode/strategy factories.
  - Added `library/training/modes/factory.py` for config-driven `mode -> TrainingMode` selection and `library/strategies/factory.py` for `model.model_type -> TrainingStrategy` selection.
  - `scripts/sdxl_peft.py` and `scripts/sdxl_finetune.py` are now thin compatibility wrappers around the shared launcher instead of open-coding the same setup flow.
  - `run_benchmark.ps1` now launches benchmark configs through `train.py --config-name=...` too, so benchmark runs follow the same active launcher surface instead of bypassing it through script-specific entrypoints.
  - The generic launcher currently supports the active Trainer/TrainingMode path for `peft` and `finetune`; `textual_inversion` is still called out as not yet migrated to that path and fails fast if requested through the new launcher.
  - The root launcher no longer invents a fake public default config: `train.py` now relies on an explicit `--config-name`, while the neutral internal `_defaults/default.yaml` baseline remains available for composition/tooling checks.
- **Conditioning ownership is now clearer in the active strategy path** — The base conditioning marker no longer lives in the data layer, and SDXL’s concrete conditioning payload now has its own strategy module.
  - Moved `ModelConditioning` from `library/data/structures.py` to `library/strategies/base/contracts.py` so strategy-owned conditioning types are defined with the rest of the strategy contract surface.
  - Added `library/strategies/sdxl/conditioning.py` for `SdxlConditioning`, and updated the active SDXL caching / denoiser paths to import that concrete conditioning type from its own strategy home.
  - `library/data/structures.py` now treats conditioning as a transported strategy payload via `CacheData.conditioning`, rather than owning the base type itself.
- **SD / SDXL text-conditioning acquisition now reads more clearly inside diffusion** — The `get_text_conds` path keeps the same behavior, but the source-selection flow is easier to follow.
  - `library/strategies/sd/diffusion.py` and `library/strategies/sdxl/diffusion.py` now split `get_text_conds` into small private helpers for cached-output loading, live encoding, and cached/live merge behavior.
  - Added focused SD tests for the cached-output and forced-live-reencode branches, while keeping the existing SDXL source-selection coverage in place.
- **Active timestep sampling now has a trainer-owned runtime seam** — The main training path no longer depends on strategy-owned `la_sampler` state just to support adaptive timestep policies.
  - Added `library/timesteps/runtime.py` with a `TimestepRuntime` that owns requested/effective mode resolution, adaptive sampler construction, dynamic timestep schedule state, timestep sampling, and adaptive-loss observation.
  - `Trainer` now creates and stores `self.timestep_runtime`, and the training loop advances schedule state and reports adaptive observations through that runtime instead of keeping separate trainer timestep fields or updating sampler state inside SD / SDXL strategies.
  - `library/training/diffusion.py` now always samples through a runtime path, and the old compatibility inputs (`la_sampler`, manual timestep overrides) are adapted into a temporary `TimestepRuntime` instead of keeping a second timestep-policy implementation alive.
  - SD / SDXL diffusion strategies no longer carry or update `self.la_sampler`, and live logging/plotter setup now reads timestep runtime state directly instead of inferring sampler identity from ad hoc strategy fields and config mutation.
  - The old compatibility utilities in `library/timesteps/timestep_utils.py` now delegate to the runtime too, so they no longer mutate config or maintain their own sampler/schedule logic.
  - The active sampler surface is now intentionally smaller: the retained modes are `uniform`, `shift`, `log_snr_uniform`, and a new general-purpose `adaptive_log_snr`, while the old one-off adaptive samplers (`mix_adaptive`, `tempered_adaptive`, `gaussian_mid_snr`, `snr_windowed`, and the stale v1 variant) were removed from the active codebase.
  - `library/config/config_validation.py` now fails fast on removed timestep sampler names and validates the new `timestep.adaptive_log_snr.*` config block.
  - Added runnable SDXL PEFT test configs for both `adaptive_log_snr` and `log_snr_uniform`, each layered on the existing `tests/test_core` benchmark-backed setup so the new sampler and its baseline are easy to smoke-test from `configs/tests/`.
  - Added focused runtime and diffusion coverage in `tests/unit/timesteps/test_runtime.py` and `tests/unit/training/test_training_diffusion.py`, and updated training-loop fixtures/integration coverage for the trainer-owned runtime path.
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
- **Validation-loop config now fails fast for runtime-breaking values** — Invalid validation settings no longer make it through to strategy runtime where they would fail later or produce divide-by-zero behavior.
  - `library/config/config_validation.py` now validates `validation.validation_split`, `validation.max_validation_steps`, and `validation.validation_timesteps`, including malformed literals and empty timestep lists.
  - `tests/unit/test_config_validation.py` now covers the new validation-loop config error paths.
  - The central validator now also rejects the previously ignored `data.source.val_data_dir` + `validation.validation_split` combination and fails fast when validation is scheduled but no validation data source is configured.

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
  - The generic internal baseline config was updated to the current nested defaults layout so ad hoc composition and config tooling still have a valid baseline without leaving a fake entry config at the top level.
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
  - All strategies use `.safetensors` format for fast loading and metadata support
  - Self-contained per-model files (no shared base classes for future flexibility)

- **VAE Dtype Configuration** (`library/data/pipeline/dataclasses.py`)

  - Added `latent_dtype` field to `DatasetManifest` ("fp16", "bf16", "fp32")
  - Added `latent_dtype` parameter to `Bucket.memory_per_image()` for accurate memory estimation

- **Unit Tests**

  - `test_pipeline_dataset_scanner.py` - 22 tests (bucket, scanning, manifest, JSON metadata)
  - `test_pipeline_caching.py` - 6 tests (batching, multi-GPU split, file creation)
  - `test_pipeline_strategies.py` - 12 tests (SD/SDXL latent and TE strategies)
  - `test_pipeline_integration.py` - 6 tests (end-to-end: scan → manifest → cache with real images)

- **Reorganized** `library/data/_deprecated/` - Moved old data scripts for cleaner separation

### Changed

- **Flexible Text Encoder Inputs** (`library/data/pipeline/dataclasses.py`)

  - Changed `BatchInfo.input_ids` from hardcoded `input_ids`/`input_ids_2` to flexible dict keyed by encoder name
  - Supports models with 1, 2, or 3+ text encoders (SD, SDXL, SD3, Flux)

## [2026-01-03]

### Fixed

- **Fixed ty and ruff errors**

  - Library/strategies/
  - Library/models/
  - Library/losses
  - Library/logging
  - Library/config
  - Library/adapters

- **Fixed TODOs and FIXMEs**

  - Addressed all todos and fixmes, left ones that were not needing fixes/for later to address

- **Fixed ImageInfo Circular Dependency**

  - Removed stale TODOs from `strategy_base.py`, `strategy_sd.py` referencing old `train_util.py` location
  - Added proper `from library.data.data_structures import ImageInfo` imports to strategy files
  - Added proper type hints for `batch` parameters (`list[ImageInfo]`) in caching methods

- **Fixed ImageInfo Type Hints** (`library/data/data_structures.py`)

  - `image_size`, `resized_size`, `bucket_reso`: Added `| None` (were initialized to `None` but typed without it)
  - `latents_crop_ltrb`: Fixed from `tuple[int, int]` to `tuple[int, int, int, int]` (LTRB = 4 values)
  - `alpha_mask`: Added `| npt.NDArray` (can be numpy array from image loading)

- **Fixed `get_hidden_states_sdxl` Signature** (`library/models/text_encoder_util.py`)

  - `max_token_length`: Changed `int` to `int | None` (function already handles `None` internally)
  - `text_encoder1`, `text_encoder2`: Added `| torch.nn.Module` for accelerator-wrapped models
  - `pool_workaround`: Added `| torch.nn.Module` parameter type for consistency

- **Fixed `np.savez` Type Warning** (`library/strategies/strategy_sdxl.py`)
  - Added assert for `text_encoder_outputs_npz is not None` before saving

### Changed

- **Applied ruff save fixes to all files**

- **Updated DATA_PIPELINE_PLAN.md**

  - Added Implementation Notes section with suggestions for config dataclass location, legacy folder naming, tokenizer validation, and ImageInfo integration

- **PEFT Strategy Internal Dedup** (`peft_strategy_base.py`)

  - Extracted `_prepare_latents()` helper to deduplicate latent encoding logic
  - Updated `process_batch` and `process_val_batch` in SD and SDXL strategies
  - Removed ~84 lines of duplicate code across 4 methods

- **Data Pipeline Skeleton** (`library/data/pipeline/`)
  - Created new pipeline architecture for high-performance data loading
  - Added `dataclasses.py`: `CacheEntry`, `Bucket`, `EpochManifest`, `DatasetManifest`
  - Added `manifest.py`: JSON I/O for manifests
  - Added `caching_engine.py`: `CachingStrategy` interface, `CachingEngine` for multi-GPU
  - Added `dataloader.py`: `TrainingDataset`, `create_training_dataloader()`
  - Added `epoch_preparation.py`: `prepare_epoch()`, `prepare_validation_epoch()`

## [2026-01-02]

### Added

- **CI/CD Pipeline (GitHub Actions)**

  - Expanded Python test matrix: 3.10, 3.11, 3.12, 3.13
  - Expanded PyTorch test matrix: 2.6.0, 2.9.0
  - Added dedicated `lint` job using ruff
  - Added coverage reporting with pytest-cov → Codecov upload
  - Added `uv` for fast package installation in CI
  - Updated workflow branches to `dev-upstream` and `hydra-config-refactor`

- **Project Configuration Consolidation**

  - Created `pyproject.toml` with all project metadata:
    - Ruff linting and formatting configuration
    - Pytest configuration (migrated from `pytest.ini`)
    - Coverage settings
  - Removed `ruff.toml`, `pytest.ini`, `setup.py` (now in `pyproject.toml`)

- **Custom Import Sorting Tool**

  - Added `tools/fix_imports.py` for custom import ordering:
    - Standard library imports → from imports
    - Third-party imports → from imports
    - First-party imports → from imports → multiline
  - Disabled isort in ruff (using custom ordering)

- **Dependabot Configuration**

  - Added `pip` package ecosystem to `.github/dependabot.yml` for Python dependency monitoring
  - Both GitHub Actions and pip dependencies now checked monthly

- **ty Type Checker Configuration**

  - Added `[tool.ty]` configuration to `pyproject.toml`
  - Set Python version to 3.10 with explicit venv path for type resolution
  - Excluded legacy directories (`feather`, `tools`, `data_processing`, `upscaling`)
  - Downgraded noisy rules to warnings for gradual adoption

- **absolufy-import ran**

  - Changed dataclasses to absolute imports, undid vendor absolute imports

### Changed

- **Ruff Modernization (UP rules)**

  - `Optional[X]` → `X | None` (modern union syntax)
  - `List[int]` → `list[int]` (builtin generics)
  - `.get(key, None)` → `.get(key)` (redundant None)
  - `.encode('utf-8')` → `.encode()` (default encoding)
  - Various SIM/C4 simplifications

### Fixed

- **Import Fixes After Ruff Cleanup**

  - Fixed `config_util.py` imports: `DreamBoothDataset`, `FineTuningDataset`, `ControlNetDataset`, `DatasetGroup` now import from correct source modules
  - Fixed `test_data_dataset.py`: `split_train_val` tests now import from `dataset_utils`

- **Fixed ty and ruff errors**

  - Library/utils
  - Library/training/

## [2026-01-01]

### Changed

- **Function Relocation**

  - Moved `swap_weight_devices()` from `library/utils/torch_utils.py` → `library/performance/custom_offloading_utils.py` (only used by offloading code)
  - Moved `is_safetensors()` from `library/models/model_util.py` → `library/utils/safetensors_utils.py`

- **Docstring Additions**

  - `model_prep.py`: Added docstrings with Args to most library scripts, utils folder remaining
  - `sdxl_model_util.py`: Added docstrings to 10+ functions including `get_timestep_embedding()`, `load_models_from_sdxl_checkpoint()`, `save_sdxl_checkpoint()`, conversion utilities
  - `training_metadata.py`: Added Args documentation to `create_training_metadata()`
  - `model_prep.py`: Fixed type hint for `padding_mode` parameter using `Literal["zeros", "reflect", "replicate", "circular"]`
  - All other library modules got docstrings now, and some small fixes like typos

- **Hydra 1.2 Schema Migration**

  - Created `library/config/schemas.py` with explicit `register_*()` functions (no side effects on import)
  - All 6 training scripts now call their specific register function before `@hydra.main`
  - Tests call `register_all()` to register all schemas at once
  - All 6 YAML configs updated to include `*_schema` in defaults list
  - Added `_self_` to `sd_textual_inversion.yaml` and `sdxl_textual_inversion.yaml` to fix composition order warning

### Fixed

- **Stale Config Fields**

  - Removed `max_data_loader_n_workers` and `persistent_data_loader_workers` from `performance/default.yaml` (moved to `data.loader` previously)

- **Test Fixes**

  - Updated `test_adapter_config_*` tests to use correct field name `adapter_module` instead of old `module`
  - Updated `test_swap_weight_devices_mock` mock path to `library.performance.custom_offloading_utils.torch`
  - Updated `test_sdxl_train_dry_run` config paths to new schema (`model.*`, `data.source.*`, `output.saving.*`)

## [2025-12-31]

### Changed

- **Dataclass Field Metadata Documentation**

  - Added comprehensive `metadata={"help": ...}` to all fields across config files:
    - `training.py`: All 11 fields in `TrainingConfig`
    - `loss.py`: All fields in `HuberConfig`, `SNRConfig`, `RegularizationConfig`, `LossConfig`
    - `performance.py`: All fields in `PrecisionConfig`, `MemoryConfig`, `AttentionConfig`, `CompilationConfig`, `DistributedConfig`
    - `output.py`: All fields in `SavingConfig`, `LoggingConfig`, `HuggingFaceConfig`, `SamplingConfig`, `MetadataConfig`
    - `sd_textual_inversion.py` / `sdxl_textual_inversion.py`: All fields in `TextualInversionSpecificConfig`
    - `peft.py`: All fields in `PeftConfig` with improved help text
    - `data.py`: Added missing `subsets` field help in `SourceConfig`
  - Added NOTE comments about TF32 enable/disable flags being mutually exclusive (could be consolidated)
  - Added TODO comment about `highvram` config field being unused (never sets `HIGH_VRAM` constant)

- **PeftConfig Field Naming Consistency**

  - Renamed config fields to use `adapter_` prefix consistently:
    - `weights` → `adapter_weights`
    - `module` → `adapter_module`
    - `args` → `adapter_args`
  - Updated all references in `sd_peft.py`, `sdxl_peft.py`, `training_metadata.py`
  - YAML config (`configs/peft/default.yaml`) already had correct naming

- **Library Reorganization - Model Prep Move**

  - Moved `model_prep.py`, `sd_model_prep.py`, `sdxl_model_prep.py` from `library/training/` → `library/models/` (model loading belongs with models, not training)
  - Updated test patches in `test_training_model_prep.py` and `test_training_sdxl_model_prep.py` to use new `library.models.*` paths

- **Library Reorganization - Optimizer Module Split**

  - Moved `optimizer.py` from `library/training/` → `library/optimizers/` and split into:
    - `optimizer_factory.py` - `get_optimizer()` factory function (431 lines)
    - `optimizer_utils.py` - `prepare_optimizer()` + helper functions (210 lines)
    - `scheduler.py` - `get_scheduler_fix()` + `get_dummy_scheduler()` (200 lines)
  - Moved `get_dummy_scheduler()` and `parse_string_to_type()` to `scheduler.py`
  - Fixed circular import by giving each file its own logger via `setup_logging()` pattern

- **Extracted `create_training_metadata()`**

  - Moved from `library/utils/model_metadata.py` → `library/training/training_metadata.py`
  - Training-specific `ss_*` metadata now in dedicated file

### Added

- **Configurable Hash Algorithm** (`output.saving.hash_algorithm`)

  - New config option to choose hash algorithm for model checksums
  - Options: `md5`, `sha1`, `sha256` (default), `sha512`, `blake3`
  - `blake3` is fastest (~3s for 6GB, saturates NVMe) but requires `pip install blake3`
  - `sha256` recommended for compatibility with A1111/ComfyUI/ModelSpec
  - Added `calculate_hash(filename, algorithm)` function in `hash_utils.py`

- **DataLoader Config Migration** (`data.loader`)

  - Moved `max_data_loader_n_workers` → `data.loader.max_workers`
  - Moved `persistent_data_loader_workers` → `data.loader.persistent_workers`
  - Removed unused `config_file` and `output_config` from `TrainingConfig`
  - Updated `prepare_deepspeed_config()` to accept `LoaderConfig` instead of `TrainingConfig`

- **ROADMAP Updates**

  - Added detailed explanations for `prepare_accelerator` split and `calculate_val_loss_check` TODOs
  - Marked lazy imports cleanup as complete

## [2025-12-30]

### Changed

- **Library Reorganization - Phase 1 Complete**

  - **Deleted `peft_common.py`** - All functions moved to specialized modules:

    - `prepare_datasets` → `library/data/dataset_setup.py`
    - `calculate_initial_step` → `library/training/trainer_utils.py`
    - `register_adapter_state_hooks` → `library/training/checkpointing.py`
    - `generate_step_logs`, `step_logging`, `epoch_logging` → `library/logging/step_logging.py`
    - `create_training_metadata` → `library/utils/model_metadata.py`
    - `resolve_adapter_kwargs` → `library/adapters/lora_utils.py`
    - `init_timestep_sampler`, `parse_dynamic_timestep_schedule` → `library/timestep/timestep_utils.py`
    - `setup_live_plotter` → `library/logging/training_plots.py`

  - **Refactored `common_utils.py`** (503 → 69 lines - 86% reduction):

    - Moved `swap_weight_devices`, `weighs_to_device`, `str_to_dtype` → `library/utils/torch_utils.py`
    - Moved `pil_resize`, `resize_image`, `get_cv2_interpolation`, `get_pil_interpolation`, `validate_interpolation_fn` → `library/data/image_utils.py`
    - Moved `GradualLatent`, `EulerAncestralDiscreteSchedulerGL` → `library/pipelines/gradual_latent.py` (NEW)
    - Remaining: `setup_logging`, `exists`, `default`, `fire_in_thread`

  - **Moved `init_trackers`** from `trainer_utils.py` → `library/logging/step_logging.py`

  - **Fixed circular imports**:

    - Removed unused `model_metadata` import from `checkpointing.py`
    - Added `setup_logging()` pattern to `dataset_setup.py`, `trainer_utils.py`, `checkpointing.py`
    - Applied duck-typing workaround in `model_metadata.py` for DreamBoothDataset detection

  - Updated smoke tests to import from new module locations
  - Updated documentation comments in `sd_peft.py`, `sdxl_peft.py`, `sd_peft_old.py`

- **Textual Inversion Config Pattern Refactoring**

  - Refactored `TextualInversionTrainer` methods to follow Strategies pattern (receive full `cfg`, use `cfg.*` access):
    - `load_target_model(cfg, weight_dtype, accelerator)`
    - `get_tokenize_strategy(cfg)`
    - `get_latents_caching_strategy(cfg)`
    - `get_text_encoding_strategy(cfg)`
  - Updated `SdxlTextualInversionTrainer` overrides to match new base class signatures
  - Fixed type hint: `assert_token_string` now correctly takes `tokenizers: List[Any]` instead of `CLIPTokenizer`
  - Fixed confusing `token_ids` reference to use `token_ids_list[-1]` for clarity

- **Config Validation Fix**

  - Fixed `should_train_text_encoder(cfg.optimizer)` → `should_train_text_encoder(cfg.optimizer.learning_rates)` in `config_validation.py`

### Fixed

- **Stale Test Fixtures**

  - Updated `test_config_validation.py` fixtures to use new config schema (`data.caching`, `loss.regularization`, `loss.snr`, `model.model_type`)
  - Updated `test_utils_torch.py` fixtures to use `PrecisionConfig` instead of `PerformanceConfig`
  - Updated `test_utils_sai_model_spec.py` to reference `model_metadata` instead of old `sai_model_spec` module
  - Updated `test_training_sdxl_checkpointing.py` patches and function names (`sai_model_spec` → `model_metadata`, `get_sai_model_spec_from_config` → `get_model_metadata_from_config`)
  - Removed stale `make_bucket_resolutions` tests from `test_models_model_util.py` (function was deleted)

- **Model Utility Refactoring**

  - Split `model_util.py` into shared utilities and SD-specific `sd_model_util.py`
  - Moved `make_bucket_resolutions` to `data_structures.py` (where `BucketManager` uses it)
  - Fixed circular import between `model_util.py` and `sd_model_util.py` by keeping helper functions in shared module
  - Added `__post_init__` to `OptimizerConfig` to handle legacy `use_8bit_adam` and `use_lion_optimizer` flags

- **Test Suite Fixes (13 tests fixed)**

  - `test_sd_peft_config.py`: Updated config key assertions (`buckets` → `data`, `dataset` → `model`)
  - `test_data_structures.py`: Updated `test_make_buckets_calls_model_util` to test real function (no longer mocks)
  - `test_utils_common.py`: Fixed mock paths (`common_utils` → `torch_utils`, `gradual_latent`)
  - `test_training_sample_generation.py`: Added missing `loss_config` parameter
  - `test_training_sdxl_checkpointing.py` (8 tests): Fixed to use `mock_sdxl_util` instead of importing real module
  - `test_models_bucket_resolutions.py`: Added new test file for `make_bucket_resolutions` function

### Removed

- Removed obsolete TODO/FIXME comments from textual inversion scripts (IDE type warnings, resolved issues)

## [2025-12-29]

### Changed

- **Configs sanitization continued:**

  - `trainer_utils.py`: Refactored `prepare_accelerator` to use `PrecisionConfig`, `CompilationConfig`, `DistributedConfig`, `DeepSpeedConfig` instead of parent `PerformanceConfig`
  - `trainer_utils.py`: Refactored `determine_grad_sync_context` to use `precision_config: PrecisionConfig` instead of `args`
  - `deepspeed_utils.py`: Fixed bugs where `prepare_deepspeed_plugin` accessed non-existent fields (e.g., `performance_config.mixed_precision` instead of `precision_config.mixed_precision`)
  - `deepspeed_utils.py`: Refactored `prepare_deepspeed_plugin` to use `DeepSpeedConfig`, `PrecisionConfig`, `TrainingConfig` instead of parent `PerformanceConfig`
  - `deepspeed_utils.py`: Refactored `prepare_deepspeed_config` to use `DeepSpeedConfig` instead of parent `PerformanceConfig`
  - `sd_model_prep.py`: Refactored `load_target_model` to use `MemoryConfig` instead of parent `PerformanceConfig`
  - `sdxl_model_prep.py`: Refactored `load_target_model` to use `ModelConfig`, `MemoryConfig`, `CachingConfig`, `PrecisionConfig` instead of full `SDXLFineTuneConfig`
  - `sdxl_model_prep.py`: Refactored `_load_target_model` to use `ModelConfig` instead of full config
  - `checkpointing.py`: Renamed `config` → `saving_config` for consistency; added `hf_config: HuggingFaceConfig` to `resume_from_local_or_hf_if_specified`
  - `sample_generation.py`: Added `loss_config: LossConfig` parameter; fixed `v_parameterization` access (was incorrectly using `training_config`, now correctly uses `loss_config`)
  - `optimizer.py`: Refactored `prepare_optimizer` to add explicit `learning_rates: LearningRatesConfig` parameter
  - `optimizer.py`: Refactored `get_optimizer` to add explicit `learning_rates: LearningRatesConfig` and `scheduler_config: SchedulerConfig` parameters
  - `optimizer.py`: Refactored `get_scheduler_fix` to use `scheduler_config: SchedulerConfig` and `optimizer_config: OptimizerConfig` instead of full params
  - `sd_textual_inversion.py`: Migrated from `*_config` aliases to direct `cfg.*` access pattern; fixed `prepare_accelerator` and `prepare_dtype` calls with correct sub-configs

### Fixed

- **Truncated `init_trackers` calls**

  - Fixed syntax errors in `sd_finetune.py`, `sdxl_finetune.py`, and `sd_textual_inversion.py` where `accelerator.init_trackers()` calls were malformed (double commas, missing `init_kwargs`, missing closing paren)

- **Null check for `hf_config`**

  - Fixed `resume_from_local_or_hf_if_specified` to check `hf_config is None` before accessing `resume_from_huggingface`

- **Test parameter naming**

  - Fixed `test_training_checkpointing.py` to use `adapter_args` instead of `network_args` and `ss_adapter_args` instead of `ss_network_args`

## [2025-12-28]

### Fixed

- **Epoch Variable Initialization**

  - Added `epoch = 0` initialization before training loops in `sd_finetune.py` and `sdxl_finetune.py` to prevent potential "referenced before assignment" errors when `num_train_epochs` is 0

- **noisy_latents dtype Handling**

  - Added `output_dtype` parameter to `get_noise_noisy_latents_and_timesteps()` in `diffusion.py`
  - Updated all callers (peft strategies, finetune scripts) to pass `output_dtype=weight_dtype`
  - Removed redundant `.to(weight_dtype)` casts from `call_unet` methods
  - Fixed incorrect function signature in `sd_textual_inversion.py` (pre-existing bug)

- **Optimizer Wrapper Guard**

  - Added guard in `optimizer.py` to raise clear error when `base_optimizer_type` is missing for ScheduleFreeWrapper/snoo_asgd optimizers

### Changed

- **Removed Legacy UI Workarounds**

  - Removed auto-adjustment of `first_cycle_max_steps` and `warmup_steps` based on `validation_split` in scheduler setup
  - Callers are now responsible for passing correct values

- **VAE Scale Factor Naming Cleanup**

  - Added `SD_VAE_LATENT_SCALE = 0.18215` constant to `constants.py`
  - Renamed `SDXL_VAE_SCALE_FACTOR` → `SDXL_VAE_LATENT_SCALE` for clarity
  - Renamed strategy field `vae_scale_factor` → `vae_latent_scale` in peft strategies and textual inversion trainers
  - lpw pipelines retain `vae_scale_factor` (spatial 8x, matches diffusers naming)

- **Model Metadata Module Rename**

  - Renamed `sai_model_spec.py` → `model_metadata.py` (now model-agnostic for Flux, Lumina, Hunyuan, etc.)
  - Renamed variable `sai_metadata` → `modelspec_metadata` across all files
  - Renamed strategy method `get_sai_model_spec()` → `get_model_metadata()`
  - Updated module imports throughout codebase

- **Naming Convention Cleanup**

  - Renamed `config` → `cfg` in `resume_from_local_or_hf_if_specified()` for consistency

- **Config Pattern Standardization in Library Utilities**

  - Established pattern: library utilities receive the **smallest container** that has what they need
  - `torch_utils.py`: Refactored `prepare_dtype`, `match_mixed_precision`, `set_torch_cuda_reduced_precision` to use `PrecisionConfig` instead of opaque `cfg`
  - `torch_utils.py`: Refactored `set_seed_from_config` to use `training_config: TrainingConfig`
  - `trainer_utils.py`: Fixed `init_trackers` to use `logging_config: LoggingConfig` directly
  - `sdxl_model_prep.py`: Fixed wrong `cfg.sd_models.*` → `cfg.model.*` and `cfg.performance.precision` type mismatch
  - `timestep_utils.py`: Refactored `parse_dynamic_timestep_schedule` and `init_timestep_sampler` to use `timestep_config: TimestepConfig`
  - `dataset_utils.py`: Refactored `load_arbitrary_dataset` to use `data_config: DataConfig` + `max_token_length: int`
  - Updated all corresponding tests to use new signatures

### Removed

- Stale TODO comments about TrainingConfig/v_parameterization in checkpointing modules
- Legacy `# TODO HYDRA` comment
- Hardcoded block_lr 23-value validation in `config_validation.py` (deferred for model-agnostic block/layer granular training)
- Unused `make_bucket_resolutions()` test code from `model_util.py`

## [2025-12-27]

### Changed

- **Optimizer Scheduler Config Nesting**

  - Created `SchedulerConfig` dataclass with 9 LR scheduler fields
  - Nested under `OptimizerConfig.scheduler`
  - Config access paths updated: `cfg.optimizer.lr_scheduler` → `cfg.optimizer.scheduler.lr_scheduler`, etc.
  - YAML updated: `configs/optimizer/default.yaml` now has `scheduler:` section

- **Learning Rate Consolidation**

  - Removed redundant `OptimizerConfig.learning_rate` field
  - `LearningRatesConfig.base` is now the canonical base LR (default: `2.0e-6`)
  - Config access paths updated: `cfg.optimizer.learning_rate` → `cfg.optimizer.learning_rates.base`

- **Learning Rate Parameter Passing Consolidation**

  - Refactored adapter `prepare_optimizer_params` methods (`lora.py`, `dylora.py`, `oft.py`) to accept `LearningRatesConfig` object instead of individual `text_encoder_lr`, `unet_lr`, and `learning_rate` float params
  - Removed legacy `text_encoder_lr` return value from `prepare_optimizer()` in `optimizer.py`
  - Updated `create_training_metadata()` to read `ss_text_encoder_lr` directly from `cfg.optimizer.learning_rates.text_encoders`
  - `cfg.optimizer.learning_rates` is now the single source of truth for all learning rates throughout the training pipeline

- **Model Config Restructuring**

  - Added `model_type` field to `ModelConfig` (`sd15 | sd2 | sdxl | flux`)
  - Removed `v2` boolean field - replaced with `model_type == "sd2"` checks
  - Configs are now self-documenting (model architecture visible at a glance)

- **Unified Validation Config**

  - Created top-level `ValidationConfig` with all 6 validation fields
  - Moved from `cfg.training.*` and `cfg.dataset.*` → `cfg.validation.*`
  - New file: `library/config/dataclasses/validation.py`, `configs/validation/default.yaml`
  - Added to all root config dataclasses and Hydra defaults

- **Loss Config Consolidation**

  - Created nested `LossConfig` with 5 sub-configs: `HuberConfig`, `SNRConfig`, `MaskedLossConfig`, `RegularizationConfig`, `EDM2Config`
  - Config access paths updated: `cfg.masked_loss.*` → `cfg.loss.masked.*`, `cfg.regularization.*` → `cfg.loss.regularization.*`
  - YAML configs consolidated: removed `configs/masked_loss/`, `configs/regularization/` → merged into `configs/loss/default.yaml`
  - Updated all 6 root config dataclasses and YAML files

- **Timestep Config Restructuring**

  - Replaced 22 flat `mix_adaptive_*` fields with 4 per-sampler nested dataclasses
  - New nested sections: `mix_adaptive`, `tempered_adaptive`, `gaussian_mid_snr`, `snr_windowed`
  - Config access paths updated: `cfg.timestep.mix_adaptive_bins` → `cfg.timestep.mix_adaptive.bins`, etc.
  - Each sampler type now has its own complete config section

- **Timestep and Logging Folder Reorganization**

  - Moved `library/timestep_samplers/` → `library/timestep/samplers/`
  - Created `library/timestep/timestep_utils.py` with `init_timestep_sampler`, `parse_dynamic_timestep_schedule`
  - Moved `tools/visualization/` → `library/logging/live_plotter/`
  - Created `library/logging/training_plots.py` with `save_timestep_distribution_plot`, `close_live_plotter`, `get_plotter_settings`, `setup_live_plotter`
  - Scripts (`sd_peft.py`, `sdxl_peft.py`) now import directly from new modules

- **Data Config Restructuring**

  - Merged `DatasetConfig` (~35 fields) + `BucketsConfig` (5 fields) into unified `DataConfig`
  - Created 5 nested sub-configs: `SourceConfig`, `PreprocessingConfig`, `CaptionConfig`, `BucketingConfig`, `CachingConfig`
  - New files: `library/config/dataclasses/data.py`, `configs/data/default.yaml`
  - Removed: `library/config/dataclasses/buckets.py`, `configs/dataset/`, `configs/buckets/`
  - Config access paths updated:
    - `cfg.dataset.train_data_dir` → `cfg.data.source.train_data_dir`
    - `cfg.dataset.cache_latents` → `cfg.data.caching.cache_latents`
    - `cfg.dataset.shuffle_caption` → `cfg.data.caption.shuffle_caption`
    - `cfg.dataset.flip_aug` → `cfg.data.preprocessing.flip_aug`
    - `cfg.buckets.enable_bucket` → `cfg.data.bucketing.enable_bucket`
  - Updated function signatures: `prepare_optimizer` (removed unused dataset_config param), `get_scheduler_fix` (now takes validation_split float), `load_arbitrary_dataset` (now takes root cfg)
  - `BlueprintGenerator` refactored to auto-search sub-configs for field values

- **DeepSpeedConfig Consolidation**

  - Moved `DeepSpeedConfig` from separate `deepspeed.py` into `performance.py` with other sub-configs
  - Deleted: `library/config/dataclasses/deepspeed.py`
  - Updated test imports to use new location
  - `peft_common.py` reduced from 974 → 650 lines

## [2025-12-26]

### Changed

- **Network → Adapter Terminology Rename**

  - **Folder:** `library/networks/` → `library/adapters/`
  - **Classes:** `LoRANetwork` → `LoRAAdapter`, `OFTNetwork` → `OFTAdapter`, `DyLoRANetwork` → `DyLoRAAdapter`
  - **Variable:** `network` → `adapter` across all scripts and library modules (~200+ occurrences)
  - **Functions:** `create_network` → `create_adapter`, `prepare_network` → `prepare_adapter`, etc.
  - **Config fields renamed:**
    - `dim` → `adapter_rank`
    - `alpha` → `adapter_alpha`
    - `dim_from_weights` → `adapter_rank_from_weights`
    - `neuron_dropout` kept (distinct from `rank_dropout`/`module_dropout`)
  - **Metadata keys:** `ss_network_*` → `ss_adapter_*` in safetensors metadata

- **Output Config Consolidation**

  - Created `OutputConfig` dataclass nesting 5 related configs: `SavingConfig`, `LoggingConfig`, `HuggingFaceConfig`, `SamplingConfig`, `MetadataConfig`
  - Config access paths updated: `cfg.saving.*` → `cfg.output.saving.*`, `cfg.logging.*` → `cfg.output.logging.*`, etc.
  - YAML configs consolidated: removed `configs/saving/`, `configs/logging/`, `configs/huggingface/`, `configs/sampling/`, `configs/metadata/` → single `configs/output/default.yaml`
  - Updated all 6 root config dataclasses and YAML files

- **Loss Config Consolidation**

  - Created nested `LossConfig` with 5 sub-configs: `HuberConfig`, `SNRConfig`, `MaskedLossConfig`, `RegularizationConfig`, `EDM2Config`
  - Config access paths updated: `cfg.masked_loss.*` → `cfg.loss.masked.*`, `cfg.regularization.*` → `cfg.loss.regularization.*`
  - YAML configs consolidated: removed `configs/masked_loss/`, `configs/regularization/` → merged into `configs/loss/default.yaml`
  - Updated all 6 root config dataclasses and YAML files

- **SD Fine-Tune Legacy Cleanup**

  - Removed `SDFineTuneSpecificConfig` dataclass and `fine_tune:` YAML section
  - `train_text_encoder` is now controlled via LR: set `optimizer.learning_rates.text_encoders` to enable TE training
  - `learning_rate_te` removed (use `optimizer.learning_rates.text_encoders` instead)

## [2025-12-25]

### Changed

- **Configuration Schema Refactor (Schema 1: Structured/Verbose)**

  - **Unified Learning Rate Configuration**: Consolidated all learning rate settings under `cfg.optimizer.learning_rates`:

    - `unet`: Dedicated field for UNet LR.
    - `text_encoders`: Dedicated field for Text Encoder LR(s). Supports separate LRs for SDXL via list `[lr_te1, lr_te2]`.
    - `blocks`: Per-block learning rates (moved from `cfg.sdxl.block_lr`).
    - Falls back to base `cfg.optimizer.learning_rate` when specific LRs are not set.
    - **Breaking**: Legacy fields (`unet_lr`, `text_encoder_lr` in PeftConfig, `learning_rate_te1/te2`, `block_lr` in SDXLConfig) removed.

  - **Config Key Rename: `cfg.network` → `cfg.peft`**:

    - Renamed config group from `network` to `peft` for semantic clarity.
    - All scripts and library modules updated to use `cfg.peft.*`.
    - YAML config moved from `configs/network/default.yaml` to `configs/peft/default.yaml`.

  - **PEFT Configuration Cleanup**:

    - Removed unused `@property` aliases from `PeftConfig` (they didn't work with Hydra/YAML).
    - Removed legacy fallback logic from `optimizer.py`.

  - **Script Migrations**:

    - **`sd_peft.py` / `sdxl_peft.py`**: Fully migrated to use `cfg.peft.*` and `cfg.optimizer.learning_rates.*`.
    - **`sdxl_finetune.py`**: Updated to use `cfg.optimizer.learning_rates.blocks`.
    - `validation.py` and tests updated accordingly.

  - **LR-Based Training Control**:

    - Removed `train_unet_only` and `train_text_encoder_only` boolean flags from `PeftConfig`.
    - Training control now inferred from learning rates: setting a component's LR to 0 disables its training.
    - Added `should_train_text_encoder()` and `should_train_unet()` helper functions to `optimizer.py`.
    - Updated `peft_strategy_base.py` methods `is_train_text_encoder()` and added `is_train_unet()` to delegate to LR helpers.
    - Updated validation, logging, and script logic to use LR-based detection.

  - **SDXLConfig Dissolution**:

    - Moved `cache_text_encoder_outputs`, `cache_text_encoder_outputs_to_disk`, `disable_mmap_load_safetensors` to `PerformanceConfig`.
    - Moved `fused_optimizer_groups` to `OptimizerConfig`.
    - Removed `train_text_encoder` (use LR-based control via `optimizer.learning_rates.text_encoders`).
    - `SDXLConfig` is now empty; kept for future SDXL-specific settings.

  - **PerformanceConfig Subcategories**:

    - Restructured `PerformanceConfig` with nested dataclasses for better organization:
      - `precision`: mixed_precision, full_fp16, full_bf16, fp8_base, fp8_base_unet, no_half_vae, cuda precision ops
      - `memory`: gradient_checkpointing, cpu_offload_checkpointing, lowram, highvram, ramtorch
      - `attention`: mem_eff_attn, xformers, sdpa, diffusers_xformers
      - `compilation`: torch_compile, dynamo_backend
      - `distributed`: ddp_timeout, ddp_gradient_as_bucket_view, ddp_static_graph
      - `caching`: cache_text_encoder_outputs, cache_text_encoder_outputs_to_disk, disable_mmap_load_safetensors
    - Updated all scripts, strategies, and validation to use nested paths (e.g., `cfg.performance.precision.mixed_precision`).

## [2025-12-24]

### Changed

- **Dataset Module Refactoring**

  - Split `library/data/dataset.py` (2351 → ~1060 lines) into modular files:
    - `dreambooth_dataset.py`: DreamBoothDataset class
    - `finetuning_dataset.py`: FineTuningDataset class
    - `controlnet_dataset.py`: ControlNetDataset class
    - `minimal_dataset.py`: MinimalDataset class
    - `dataset_group.py`: DatasetGroup class
    - `dataset_utils.py`: Utility functions (collator_class, load_arbitrary_dataset, split_train_val, debug_dataset, ImageLoadingDataset)
  - BaseDataset remains in `dataset.py` with backwards-compatible re-exports
  - No breaking changes: all existing imports continue to work

- **Config Consolidation: `diffusers_xformers`**

  - Moved `diffusers_xformers` from `SDXLConfig` and `SDFineTuneSpecificConfig` to `PerformanceConfig`
  - Updated `sd_finetune.py` and `sdxl_finetune.py` to use `cfg.performance.attention.diffusers_xformers`
  - Standardized `sd_finetune.py` to use `cfg` variable name (matching `sdxl_finetune.py`)

- **SAI Model Spec Consolidation**

  - Updated `peft_strategy_sd.py` and `peft_strategy_sdxl.py` to use `get_sai_model_spec_from_config()` instead of legacy argparse-based function
  - Removed ~135 lines of duplicate legacy code from `checkpointing.py`:
    - Removed `get_sai_model_spec()` (legacy, used argparse)
    - Removed `get_sai_model_spec_dataclass()` (legacy, unused)
  - Canonical function is now `library.utils.sai_model_spec.get_sai_model_spec_from_config()`

- **Text Encoder Utility Consolidation**

  - Consolidated duplicate `get_hidden_states_sdxl()` and `pool_workaround()` between `text_encoder_util.py` and `strategy_sdxl.py`
  - `SdxlTextEncodingStrategy` methods now delegate to shared utilities
  - Removed ~60 lines of duplicate code from `strategy_sdxl.py`

- **Dataclass Config Cleanup**

  - Consolidated `no_half_vae` to `PerformanceConfig` (canonical), removed from `SDXLConfig`
  - Updated `sdxl_finetune.py` to use `cfg.performance.precision.no_half_vae`
  - Improved `text_encoder_lr` documentation in `PeftConfig` (explains `Any` type, future unification plans)

- **Checkpointing Module Split**

  - Created `sd_checkpointing.py` with SD1.5/2-specific save functions
  - Removed SD-specific code from `checkpointing.py` (now generic-only)
  - Updated `sd_finetune.py` to import from `sd_checkpointing.py`
  - Mirrors existing pattern: `checkpointing.py` (generic) + `sdxl_checkpointing.py` (SDXL)

- **Model Prep Module Split**

  - Created `sd_model_prep.py` with SD1.5/2-specific model loading functions
  - `model_prep.py` now contains only generic utilities (`replace_unet_modules`, `patch_accelerator_for_fp16_training`, `set_padding_mode_for_vae_conv2d_modules`)
  - Updated SD scripts, strategies, and tools to import from `sd_model_prep.py`
  - Removed unused `load_target_model` import from `sdxl_peft.py`

- **Sample Generation Module Split**

  - Created `sd_sample_generation.py` with SD-specific `sample_images` wrapper
  - `sample_generation.py` now contains only generic utilities (`sample_images_common`, `sample_images_check`, etc.)
  - Mirrors existing `sdxl_sample_generation.py` pattern

- **Folder Rename: `optimizations/` → `performance/`**

  - Renamed `library/optimizations/` to `library/performance/` for consistency with `PerformanceConfig`
  - Updated all imports in scripts

- **Created `AGENTS.md`**
  - Agent instructions for working on the repository (venv location, test commands, project structure, conventions)

## [2025-12-23]

### Changed

- **PEFT Strategy-Based Refactoring (Phase 1-2)**

  - Reduced `sd_peft.py` from 927 lines to 50 lines (95% reduction)
  - Removed `SDPeftTrainer` class - all functionality extracted to modular components
  - Created `library/strategies/peft_strategy_base.py` (16 methods) - ABC interfaces for PEFT training
  - Created `library/strategies/peft_strategy_sd.py` (21 methods) - SD1.5/2 implementations
  - Created `library/training/peft_common.py` (6 functions) - shared logging/plotting utilities
  - Refactored `library/training/peft_trainer.py` to use strategy pattern and standalone functions
  - `train()` function now accepts `strategies: PeftTrainingStrategy` parameter

- **Phase 4: Training Loop Cleanup**

  - Extracted `prepare_datasets()` to `peft_common.py` (~47 lines saved per script)
  - Extracted `calculate_initial_step()` to `peft_common.py` (~42 lines saved per script)
  - Extracted `parse_dynamic_timestep_schedule()` to `peft_common.py` (~10 lines saved per script)
  - Extracted `register_adapter_state_hooks()` to `peft_common.py` (~40 lines saved per script)
  - Total: ~140 lines reduced from each PEFT script (1173 → 1035 lines)

### Fixed

- **HuggingFace Upload Bug in Checkpointing**

  - The Hydra migration accidentally replaced `if args.huggingface_repo_id is not None` with `if saving_config.resume is not None` (wrong!) and stubbed out upload calls with `pass`
  - Added `hf_config: Optional[HuggingFaceConfig] = None` parameter to 9 functions in `checkpointing.py` and `sdxl_checkpointing.py`
  - Restored proper upload logic for model checkpoints and training state
  - Functions affected: `save_sd_model_on_epoch_end_or_stepwise`, `save_sd_model_on_train_end`, `save_and_remove_state_*` variants

### Added

- **Test Suite Expansion (897 total tests)**

  - `test_optimizations_deepspeed.py` - Unit tests for DeepSpeed config/plugin preparation (17 tests)
  - `test_optimizations_offloading.py` - Unit tests for CPU offloading utilities (42 tests)
  - `test_training_sdxl_checkpointing.py` - Unit tests for SDXL checkpointing wrappers (16 tests)
  - `test_models_model_util.py` - Added tests for `renew_*_paths`, `get_model_version_str`, `conv_attn_to_linear`, `controlnet_conversion_map`, `reshape_weight_for_sd`, `linear_transformer_to_conv` (+21 tests, 49 total)
  - `test_models_sdxl_model_util.py` - Added tests for `convert_unet_state_dict`, bidirectional SDXL↔Diffusers conversion with roundtrip verification (+8 tests, 27 total)
  - `test_data_structures.py` - Added `BucketManager.make_buckets` test with mocked `model_util`
  - `test_dataset_bucketing.py` - Added unit tests for `dataset.py` bucketing logic (6 tests)
  - `test_training_sdxl_model_prep.py` - Added unit tests for `sdxl_model_prep.py` (10 tests)
  - `test_models_conversion.py` - Added unit tests for model conversion functions (15 tests: VAE attention/resnet paths, checkpoint assignment, VAE state dict conversion)
  - `test_fp8_optimization.py` - Added tests for `apply_fp8_monkey_patch` and `fp8_linear_forward_patch` (7 tests)
  - `test_optimizations_offloading.py` - Added tests for `Offloader` and `ModelOffloader` classes (11 tests)
  - `test_dataset_bucketing.py` - Added 8 tests for `BaseDataset.__getitem__` (cached/disk latents, image loading, flip aug, batching)
  - Audited ROADMAP "Heavy Mocking" section with accurate ✅/🔶/❌ status for each module

## [2025-12-22]

### Added

- **Centralized Config Validation Module** (`library/config/validation.py`)

  - `prepare_config(cfg)` - Auto-fixups: cache flags, optimizer shortcuts, backward compat
  - `validate_config(cfg)` - Cross-config errors: `adaptive_noise_scale`, `v_pred` conflicts, `full_fp16/bf16`, `fp8_base`, SDXL `block_lr`
  - Script-specific validators: `validate_sd_peft`, `validate_sdxl_peft`, `validate_sd_textual_inversion`, `validate_sdxl_textual_inversion`
  - Comprehensive unit tests in `tests/unit/test_validation.py` (24 tests)
  - `tests/unit/test_losses_loss_weighting.py` - Tests for SNR weighting formulas, v-prediction modes, and masking.
  - `tests/unit/test_utils_device.py` - Tests for device memory cleanup, synchronization, and device selection with strict hardware mocking.
  - `tests/unit/test_losses_edm2_loss.py` - Tests for EDM2 adaptive loss weighting components and configuration logic.
  - `tests/unit/test_utils_jpeg_xl.py` - Tests for JXL bitstream parsing, container structure using synthetic binary data, refactoring `JXLBitstream` for correctness.
  - `tests/unit/test_networks_lora_utils.py` - Tests for LoRA state dict filtering and weight merging (Linear & Conv2d) with mocked loading hooks.
  - `tests/unit/test_sai_model_spec.py` - Tests for ModelSpec metadata generation, config extraction, and resolution logic.
  - `tests/unit/test_utils_torch.py` - Added tests for `set_seed_from_config` mocking underlying seed setters.
  - `tests/unit/test_training_sample_generation.py` - Tests for parsing logic, scheduler factories, and sampling trigger checks in `sample_generation.py`.
  - `tests/unit/test_models_text_encoder_util.py` - Tests for CLIP pooling workaround (finding EOS tokens) and SDXL hidden state chunking/reshaping.
  - `tests/unit/test_config_util.py` - Tests for `BlueprintGenerator`, dataset type detection (DreamBooth/FineTune/ControlNet), validation split logic, and subdirectory parsing.
  - `tests/unit/test_constants.py` - Regression tests for critical constants: scheduler params, UNet/VAE architecture values, SDXL configs, and HuggingFace model IDs.
  - `tests/unit/test_sdxl_model_util.py` - Tests for `timestep_embedding`, `get_size_embeddings`, and `make_unet_conversion_map` (pure math functions).
  - `tests/unit/test_model_util.py` - Tests for `shave_segments`, `is_safetensors`, `create_unet_diffusers_config`, and `create_vae_diffusers_config`.
  - `tests/unit/test_caching.py` - Tests for latent cache validation (`is_disk_cached_latents_is_expected`) and text encoder output file I/O.
  - `tests/unit/test_original_unet.py` - Tests for `get_timestep_embedding`, `resize_like`, and `get_parameter_dtype/device` utilities from original_unet.py.
  - `tests/unit/test_fp8_optimization.py` - Tests for FP8 quantization: `calculate_fp8_maxval`, `quantize_fp8`, and `quantize_weight` with block/channel/tensor modes.
  - `tests/unit/test_networks_lora.py` - Tests for LoRA block LR parsing, dims/alphas calculation, block index resolution, and weight removal utilities.
  - `tests/unit/test_optimizers_adafactor_fused.py` - Tests for stochastic rounding (`copy_stochastic_`) and optimizer patching.
  - `tests/unit/test_networks_lora_diffusers.py` - Tests for UNet conversion map between Stability AI and Diffusers naming formats.
  - `tests/unit/test_pipelines_lpw.py` - Tests for prompt attention weight parsing and token/weight padding utilities.
  - `tests/unit/test_utils_huggingface.py` - Tests for HuggingFace Hub API (`exists_repo`, `list_dir`) with mocked network calls.
  - `tests/unit/test_strategies_base.py` - Tests for strategy base classes: singleton patterns, tokenizer loading, NPZ save/load, weighted input parsing, and cache validation (39 tests).
  - `tests/unit/test_strategies_sd.py` - Tests for SD 1.5/2.0 strategies: tokenization (v1/v2), text encoding with clip_skip, latent caching with VAE mocking (17 tests).
  - `tests/unit/test_strategies_sdxl.py` - Tests for SDXL strategies: dual tokenizers, pool workaround, dual text encoders, text encoder output caching (21 tests).
  - Updated `tests/unit/test_caching.py` - Added heavy mocking tests for `load_images_and_masks_for_caching`, `cache_batch_latents` (VAE), `cache_batch_text_encoder_outputs` (11 new tests, 24 total).
  - Updated `tests/unit/test_training_trainer_utils.py` - Added heavy mocking tests for `prepare_accelerator`, `init_trackers`, `determine_grad_sync_context` (11 new tests, 27 total).

### Fixed

- **Checkpointing v_parameterization Bug**

  - Fixed `checkpointing.py` and `sdxl_checkpointing.py` accessing `training_config.v_parameterization` which doesn't exist on `TrainingConfig` (the field is on `LossConfig`)
  - Replaced `training_config` parameter with `loss_config` in `save_sd_model_on_train_end` and `save_sd_model_on_epoch_end_or_stepwise` functions
  - Updated call sites in `sd_finetune.py` and `sdxl_finetune.py` to pass `cfg.loss` instead of `cfg.training`

- **Hydra Config Field Reference Bugs**

  - Fixed `sd_finetune.py` and `sd_textual_inversion.py` referencing non-existent config fields
  - Fixed `training_config.min_snr_gamma` → `config.loss.min_snr_gamma` (and similar loss fields)
  - Fixed `training_config.zero_terminal_snr` → `config.regularization.zero_terminal_snr`
  - Fixed undefined `sd_models_config` variable → `model_config`
  - Fixed `training_config.max_grad_norm` → `optimizer_config.max_grad_norm`

- **SDXL Fine-tune Config Fixes**

  - Fixed `prepare_deepspeed_model(cfg.performance)` → `prepare_deepspeed_model(cfg.training)`
  - Fixed `cfg.sdxl.fused_backward_pass` → `cfg.optimizer.fused_backward_pass`
  - Fixed `cfg.optimizer.fused_backward_pass` → `cfg.training.fused_backward_pass` (correction: check actual usage if needed, but assuming general fixes here)
  - Added missing `OmegaConf` import for tracker config serialization
  - Fixed `JXLBitstream` in `jpeg_xl_util.py`: `IndexError` on partial reads due to incorrect bit/byte offset tracking.
  - Fixed `sai_model_spec.py`: Missing `import time` causing NameError on timestamp generation.
  - Fixed `torch_utils.py` tests: Mocked `set_seed` to avoid dependency on global state/missing libraries in unit tests.

- **SD Peft Pure Hydra Migration**

  - **Completed `sd_peft.py` migration** to Pure Hydra (removed ~100 undefined `args` references)
  - Refactored `sd_peft.py` to use `cfg.*` paths correctly (e.g. `cfg.training`, `cfg.logging`)
  - Removed unused `import argparse` from `sdxl_peft.py`
  - Fixed method signatures: `generate_step_logs(args)` → `generate_step_logs(cfg)`, `save_timestep_distribution_plot(args)` → `save_timestep_distribution_plot(cfg)`
  - Fixed `cfg.la_sampler` → `self.la_sampler` (runtime object, not config)
  - Fixed call sites: `sample_images_check(cfg)` → `sample_images_check(cfg.sampling)`, `calculate_val_loss_check(cfg)` → `calculate_val_loss_check(cfg.training)`
  - Fixed `prepare_deepspeed_model(cfg)` → `prepare_deepspeed_model(cfg.training)`
  - Fixed `cfg.training.vae` → `cfg.model.vae`
  - Fixed `plot_edm2_loss_weighting_check()` and `plot_edm2_loss_weighting()` - added missing `cfg.training` and `cfg.output.saving.output_name` parameters
  - Removed stale TODO comment about `no_metadata` field (it exists in `SavingConfig`)

- **Textual Inversion Cleanup**

  - Renamed unused `args` parameters to `config` in `sd_textual_inversion.py` and `sdxl_textual_inversion.py` for consistency

- **Function Naming Cleanup**

  - Renamed `args_set_seed()` → `set_seed_from_config()` in `torch_utils.py`
  - Renamed `prepare_deepspeed_args()` → `prepare_deepspeed_config()` in `deepspeed_utils.py`

- **Library Type Safety**

  - Refactored `prepare_accelerator` in `trainer_utils.py` to accept typed configs (`PerformanceConfig`, `LoggingConfig`, `TrainingConfig`) instead of `DictConfig`
  - Refactored `init_trackers` in `trainer_utils.py` to accept any config type with `.logging` sub-config
  - Refactored `prepare_deepspeed_plugin` and `prepare_deepspeed_config` in `deepspeed_utils.py` to accept `PerformanceConfig` and `TrainingConfig`
  - Added missing `no_metadata` field to `SavingConfig` dataclass

- **Dataclass Naming Consistency**

  - Renamed `sd_models: ModelConfig` → `model: ModelConfig` in 6 dataclasses to match YAML config naming

### Changed

- **HuggingFace Upload Refactor**

  - Refactored `huggingface_util.upload()` to accept `HuggingFaceConfig` instead of `argparse.Namespace`
  - Updated callers in `sd_textual_inversion.py` and `sd_peft.py` to use `config.huggingface`

- **Config Validation Architecture**

  - Removed `__post_init__` methods from 5 dataclasses (`DatasetConfig`, `RegularizationConfig`, `SamplingConfig`, `OptimizerConfig`, `SDXLConfig`)
  - Trainer `validate_extra_config` methods now delegate to centralized module
  - All 6 training scripts call `prepare_config()` and `validate_config()` at entry point
  - Removed duplicate precision validation asserts from `sd_peft.py`, `sd_finetune.py`, `sdxl_finetune.py`

### Removed

- **Legacy Argparse Code**

  - Removed dead imports of `add_logging_arguments`, `add_prompt_parsing_arguments`, `add_loss_weighting_arguments` from scripts
  - Removed deprecated `add_loss_weighting_arguments()` from `loss_weighting.py`
  - Removed deprecated `add_logging_arguments()` from `common_utils.py`
  - Removed deprecated `add_prompt_parsing_arguments()` from `prompt_utils.py`
  - Removed unused `import argparse` from `loss_weighting.py` and `prompt_utils.py`
  - Removed `sdxl_data_utils.py` (unused, superseded by strategy pattern)
  - Removed dead `get_hidden_states(args)` from `text_encoder_util.py` (superseded by strategy)
  - Removed `add_model_spec_arguments()` from `sai_model_spec.py` (superseded by MetadataConfig)
  - Removed `ModelSpecMetadata.from_args()` from `sai_model_spec.py` (use `from_config()` instead)
  - Renamed `generate_user_config_from_args()` → `generate_user_config_from_dataset()` in `config_util.py`

## [2025-12-20]

### Added

- **Hydra Configuration System**

  - `configs/sd_finetune.yaml` - Hydra config for SD 1.5/2.0 fine-tuning
  - `configs/sd_textual_inversion.yaml` - Hydra config for SD 1.5/2.0 textual inversion
  - `configs/sd_peft.yaml` - Hydra config for SD 1.5/2.0 PEFT/LoRA training
  - `configs/sdxl_finetune.yaml` - Hydra config for SDXL fine-tuning
  - `configs/sdxl_textual_inversion.yaml` - Hydra config for SDXL textual inversion
  - `configs/sdxl_peft.yaml` - Hydra config for SDXL PEFT/LoRA training
  - `library/config/config_util.py` - Added `RootConfig` Protocol for type-safe config handling

- **Testing Infrastructure**

  - `tests/conftest.py` - Pytest fixtures for Hydra, configs, temporary directories, and mock objects
  - `tests/unit/test_configs.py` - 28 tests for configuration dataclasses (instantiation, defaults, Hydra composition, overrides)
  - `tests/unit/test_training_optimizer.py` - 26 tests for optimizer module (creation, detection, schedulers, utilities)
  - `tests/unit/test_training_checkpointing.py` - 23 tests for checkpointing module (naming, removal, metadata)
  - `tests/unit/test_training_diffusion.py` - 17 tests for diffusion utilities (timesteps, noisy latents)
  - `tests/unit/test_training_noise_utils.py` - 21 tests for noise utilities (SNR, pyramid noise, noise offset)
  - `tests/unit/test_data_image_utils.py` - 14 tests for image utilities (globbing, loading, cropping)
  - `tests/unit/test_utils_torch.py` - 12 tests for torch utilities (dtype preparation, mixed precision)
  - `tests/unit/test_training_trainer_utils.py` - 16 tests for trainer utilities (validation check, LR logging)
  - `tests/unit/test_data_prompt_utils.py` - 16 tests for prompt utilities (attention parsing, token padding)
  - Enhanced `pytest.ini` with test markers (`unit`, `integration`, `config`, `training`, `data`, `slow`, `requires_gpu`)
  - Restructured tests to `tests/unit/` directory to avoid import naming collisions
  - Added pytest-cov for code coverage reporting
  - `tests/unit/test_utils_safetensors.py` - Tests for `safetensors_utils` (I/O, metadata, header parsing, `mmap` logic, key finding).
  - `tests/unit/test_utils_common.py` - Tests for `common_utils` (`str_to_dtype`, `resize_image` with alpha/interpolation checks, `GradualLatent`, `swap_weight_devices`).
  - `tests/unit/test_data_image_utils.py` - Tests for `image_utils` (`glob_images` escaping, `load_image` mode conversion, `trim_and_resize` logic).
  - `tests/unit/test_losses_loss.py` - Comprehensive tests for loss functions (`stable_mse`, `stable_smooth_l1`, etc.) and Recorders (`LossRecorder`, `EMARecorder`).

### Changed

- **Script Naming Standardization** - Renamed to `{model}_{method}.py` pattern:

  - `fine_tune.py` → `sd_finetune.py`
  - `train_textual_inversion.py` → `sd_textual_inversion.py`
  - `train_network.py` → `sd_peft.py`
  - `sdxl_train.py` → `sdxl_finetune.py`
  - `sdxl_train_textual_inversion.py` → `sdxl_textual_inversion.py`
  - `sdxl_train_network.py` → `sdxl_peft.py`

- **Dataclass File Naming** - Matched to script names in `library/config/dataclasses/`

- **Library Refactoring**

  - `library/training/trainer_utils.py` - `calculate_val_loss_check()` now accepts `TrainingConfig` directly
  - `library/config/config_util.py` - Added `RootConfig` Protocol, removed stale circular import workaround
  - `library/data/dataset.py` - `load_arbitrary_dataset()` accepts `DatasetConfig` directly

### Removed

- **Legacy Code**

  - `ArgsAdapter` class removed from `sd_peft.py` (formerly `train_network.py`)
  - `ConfigAdapter` shim removed from migrated scripts
  - `setup_parser()` and argparse removed from all migrated scripts

### Fixed

- Type hints in `config_util.py` - replaced string hint `"RootConfig"` with proper Protocol class

- **Config Audit Cleanup**

  - Removed duplicate `logging_dir` from `PerformanceConfig` (canonical: `LoggingConfig`)
  - Removed duplicate `vae` from `TrainingConfig` (canonical: `ModelConfig`)
  - Removed duplicate HuggingFace fields from `SavingConfig` (canonical: `HuggingFaceConfig`)
  - Fixed `sdxl_peft.py` to use `ModelConfig` instead of `ModelLoadingConfig`
  - Added missing `BucketsConfig` to `SDXLPeftConfig`
  - Removed orphan `v_parameterization` from `sd_models/default.yaml`
  - Added missing `optimizer_schedulefree_wrapper` fields to `optimizer/default.yaml`
  - Removed unused `PeftConfig` import from `sd_textual_inversion.py`
  - Renamed `test_train_network_config.py` to `test_sd_peft_config.py`

- **SDXL Configuration Refactoring**

  - Renamed `sdxl_training` config group to `sdxl`
  - Renamed `configs/sdxl_training` directory to `configs/sdxl`
  - Renamed `SDXLTrainingConfig` to `SDXLConfig` in `library/config/dataclasses/sdxl.py`
  - Updates to `sdxl_finetune.py` and `sdxl_peft.py` to use explicit dataclass fields instead of inheritance
  - Fixed duplicate arguments and import errors in `sd_peft.py` and `sdxl_model_prep.py`
  - Removed legacy `library.config.arguments` usage from `sdxl_peft.py`, `sd_finetune.py`, and `sd_textual_inversion.py`
  - Fixed double `@dataclass` decorator in `library/config/dataclasses/model.py`

- **Model Config Refactor**

  - Renamed config group `sd_models` to `model`
  - Merged `SDModelsConfig` and `ModelLoadingConfig` into `ModelConfig` in `library/config/dataclasses/model.py`
  - Moved `v_parameterization` from `SDModelsConfig` to `LossConfig` (and updated scripts to use `config.loss.v_parameterization`)
  - Moved `vae` and `vae_conv2d_padding_mode` from `TrainingConfig`/`SDModelsConfig` to `ModelConfig`
  - Moved `use_ramtorch` and `direct_ramtorch` from `PeftConfig` to `PerformanceConfig`
  - Updated all scripts and YAMLs to reflect these changes

- **Type hint bug in `library/utils/torch_utils.py`**

  - `prepare_dtype()` function was incorrectly typed to accept `TrainingConfig` but accessed fields from both `PerformanceConfig` (mixed_precision) and `SavingConfig` (save_precision). Refactored to properly accept both config types as separate parameters.
