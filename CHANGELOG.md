# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Rules:
- Use proper sub titles "Added", "Changed", "Removed" and "Fixed"
- Keep proper track of days for where entries should go
- Be concise but mention all changes without necessarily detailing each one

## [2026-04-04]

### Added

- **The absorbed plain-optimizer surface now includes a first low-rank projection optimizer too** — The repo can now host a projector-backed optimizer without routing through the donor package at runtime.
  - Added `library/optimization/optimizers/galore.py` with a repo-owned `GaLore` implementation adapted from the vendor source.
  - Added `library/optimization/optimizers/utils/galore.py` with the repo-owned `GaLoreProjector` helper adapted from the vendor projector utility.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `GaLore` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `GaLore` construction and ranked 2D projector initialization through the shared factory path.
- **The absorbed plain-optimizer path now includes the first Shampoo-family entry too** — The repo can now host a preconditioner-based optimizer without routing through the donor package at runtime.
  - Added `library/optimization/optimizers/soap.py` with a repo-owned `SOAP` implementation adapted from the vendor source.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `SOAP` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `SOAP` construction and first-step preconditioner state initialization through the shared factory path.
- **The absorbed plain-optimizer path now includes its first optimizer-local LR-scheduling family too** — The repo can now host a donor optimizer with built-in warmup/warmdown behavior without silently double-scheduling it from the outside.
  - Added `library/optimization/optimizers/ranger21.py` with a repo-owned `Ranger21` implementation adapted from the vendor source while switching its helper imports onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `Ranger21` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `Ranger21` construction, first-step state initialization, and scheduler-orchestration behavior through the shared optimizer/scheduler entrypoints.
- **The absorbed plain-optimizer path now includes a full Shampoo-family optimizer too** — The repo can now host the heavier Shampoo-style preconditioner path without relying on vendor or `pytorch_optimizer` helper imports at runtime.
  - Added `library/optimization/optimizers/shampoo.py` with a repo-owned `ScalableShampoo` implementation adapted from the vendor source.
  - Added `library/optimization/optimizers/utils/shampoo.py` with the repo-owned Shampoo helper surface used by `ScalableShampoo`.
  - Updated `library/optimization/optimizers/soap.py` to reuse the repo-owned `merge_small_dims(...)` helper instead of importing it from `pytorch_optimizer`.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `ScalableShampoo` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `ScalableShampoo` construction and first-step preconditioner/graft initialization through the shared factory path.
- **The absorbed low-rank optimizer surface now includes the second projector-backed variant too** — The repo can now host both the baseline projector path and a richer projector-backed variant without falling back to the donor package.
  - Added `library/optimization/optimizers/fira.py` with a repo-owned `Fira` implementation adapted from the vendor source.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `Fira` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `Fira` construction and ranked 2D projector initialization through the shared factory path.
- **The absorbed plain-optimizer path now owns the baseline Compass family too** — The repo can now host the non-optional `Compass` variants without routing through the donor package or dragging the backend-heavy 8-bit/AO code into the first pass.
  - Added `library/optimization/optimizers/compass.py` with repo-owned `Compass`, `CompassPlus`, `CompassADOPT`, and `CompassADOPTMARS` implementations adapted from the vendor family file.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so the absorbed Compass family participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for Compass-family construction, `CompassPlus` lookahead-state initialization, and factored/previous-gradient state initialization in the ADOPT-style Compass variants.

### Changed

- **Scheduler orchestration now fails fast for `Ranger21` double-scheduling conflicts** — Runs using Ranger21’s built-in LR schedule no longer quietly stack a second external scheduler on top unless the config explicitly disables the optimizer-local scheduler.
  - Updated `library/optimization/scheduler.py` so `Ranger21` only accepts the effectively no-op external scheduler case (`lr_scheduler='constant'` with no warmup) while its internal LR schedule is active.
  - The shared scheduler path still allows normal external schedulers for `Ranger21` when `disable_lr_scheduler=True`.
- **The first Compass-family absorption pass also hardens a few donor edge cases while keeping the algorithm structure intact** — The repo-owned copy now avoids a couple of brittle donor assumptions instead of preserving them as hidden footguns.
  - Fixed the plain `Compass` step path so `weight_decouple` stays a boolean instead of accidentally becoming a one-tuple.
  - Updated `CompassPlus` to re-read per-group `betas` during its phase-3 update pass and to initialize diff-amp reset state without assuming gradients already exist during `reset()`.
  - Updated `CompassADOPT` and `CompassADOPTMARS` reset paths to derive factored-state shapes from parameter shapes instead of assuming pre-existing gradients.

## [2026-04-03]

### Added

- **The wrapper path now has its first fully repo-owned schedule-free wrapper too** — Wrapper absorption no longer stops at `SNOO_ASGD`, and the legacy schedule-free wrapper flag no longer depends on the external `schedulefree` package.
  - Added `library/optimization/wrappers/schedulefree.py` with a repo-owned `ScheduleFreeWrapper` implementation adapted from the vendor wrapper logic while switching to repo-local stochastic-copy utilities and the repo’s base-optimizer wrapping flow.
  - Updated `library/optimization/registry.py` so `ScheduleFreeWrapper` is modeled as a repo-owned wrapper with both train/eval-toggle and scheduler-on-base-optimizer capabilities.
  - Updated `library/optimization/optimizer_factory.py` so both the explicit `optimizer_type='ScheduleFreeWrapper'` path and the legacy `optimizer_schedulefree_wrapper=True` path now use the same repo-owned wrapper implementation.
  - Expanded wrapper-focused regression coverage so the absorbed wrapper continues to build through the shared wrapper path, schedule the base optimizer, and advertise schedule-free train/eval handling.

- **Repo-owned scheduler absorption now has a first real implementation slice** — Standalone scheduler classes no longer need to stay trapped behind `lr_scheduler_type` custom-import paths.
  - Added `library/optimization/schedulers/` with repo-owned `CosineAnnealingWarmRestarts` and `RexAnnealingWarmRestarts` implementations adapted from the vendor schedulers.
  - Updated `library/optimization/registry.py` so both warm-restart schedulers are modeled as registry-backed torch-style scheduler targets.
  - Updated `configs/smoke_test_customoptimizer.yaml` so the custom-optimizer smoke config now uses the repo-facing `RexAnnealingWarmRestarts` scheduler name directly instead of a vendor `lr_scheduler_type` path.
  - Added focused regression coverage for the new scheduler registrations and shared construction paths.
- **The shared registration path now hosts more than one shape of absorbed plain optimizer cleanly** — Adding repo-owned plain optimizers still does not require new factory special cases.
  - Added `library/optimization/optimizers/laprop.py` with a repo-owned `LaProp` implementation adapted from the vendor source.
  - Added `library/optimization/optimizers/adopt.py` with a repo-owned `ADOPT` implementation adapted from the vendor source.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so both optimizers participate in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for both registrations and both shared construction paths.
- **The optimization layer now hosts a repo-owned schedule-free leaf optimizer too** — Schedule-free support is no longer limited to third-party leaves and wrapper integrations.
  - Added `library/optimization/optimizers/adopt_schedulefree.py` with a repo-owned `ADOPTScheduleFree` implementation adapted from the vendor source.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `ADOPTScheduleFree` is modeled as a repo-owned schedule-free leaf with train/eval-toggle and no-external-scheduler capabilities.
  - Added focused regression coverage for the new registration metadata, shared construction path, train/eval handling, and dummy-scheduler routing.
- **The repo-owned schedule-free leaf surface now covers the rest of the non-AO ADOPT/FADOPT family too** — The big donor `schedulefree.py` file is no longer represented by a single absorbed leaf plus several forgotten siblings.
  - Expanded `library/optimization/optimizers/adopt_schedulefree.py` to absorb `ADOPTEMAMixScheduleFree`, `ADOPTNesterovScheduleFree`, `ADOPTMARSScheduleFree`, `FADOPTScheduleFree`, `FADOPTEMAMixScheduleFree`, `FADOPTNesterovScheduleFree`, and `FADOPTMARSScheduleFree`.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so the full non-AO schedule-free ADOPT/FADOPT family participates in the shared repo-owned registration path with train/eval-toggle and no-external-scheduler capabilities.
  - Expanded absorbed-integration coverage so the new schedule-free leaves construct through the shared factory path and continue to use the dummy-scheduler route consistently.
- **The AO schedule-free ADOPT leaf now has its own low-bit-specific home too** — The remaining low-bit schedule-free variant no longer has to share a file with the non-AO family or stay lost in the donor module.
  - Added `library/optimization/optimizers/adopt_schedulefree_ao.py` with a repo-facing `ADOPTAOScheduleFree` implementation separated from the non-AO schedule-free family.
  - Added small repo-owned helper seams needed by the AO path under `library/optimization/optimizers/utils/`, including `CLIP_TYPE` / `STATE_PRECISION`, beta warmup scheduling, spam clipping helpers, cosine-decay helpers, stable-spam tensor helpers, and a compiled paper-OrthoGrad wrapper.
  - Updated `library/optimization/optimizers/__init__.py`, `library/optimization/registry.py`, and the absorbed optimizer tests so `ADOPTAOScheduleFree` participates in the shared registry path with TorchAO-backed capability modeling and the same dummy-scheduler schedule-free behavior.
- **The absorbed plain-optimizer surface is still scaling cleanly without new factory branches** — The repo-owned optimizer path can now host more varied non-Adam-family optimizers too.
  - Added `library/optimization/optimizers/lpf_adamw.py` with a repo-owned `LPFAdamW` implementation adapted from the vendor source.
  - Added `library/optimization/optimizers/sgd_sai.py` with a repo-owned `SGDSaI` implementation adapted from the vendor source.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so both optimizers participate in the shared repo-owned registration path.
  - Added focused regression coverage for both new registrations and both shared construction paths.
- **The repo-owned adaptive-optimizer set continues to grow without changing the factory shape** — The next absorbed pair still fit the existing registration model with only light repo-owned adaptation.
  - Added `library/optimization/optimizers/adai.py` with a repo-owned `Adai` implementation adapted from the vendor source.
  - Added `library/optimization/optimizers/vsgd.py` with a repo-owned `VSGD` implementation adapted from the vendor source, including an explicit `stochastic_fp` default on the repo-owned path.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so both optimizers participate in the shared repo-owned registration path.
  - Added focused regression coverage for both new registrations and both shared construction paths.
- **The absorbed plain-optimizer set now includes the first row/column-scaled SGD variant too** — The shared registration path can host small donor helper adaptations without growing a new utility dependency pile.
  - Added `library/optimization/optimizers/racs.py` with a repo-owned `RACS` implementation adapted from the vendor source.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `RACS` participates in the shared repo-owned registration path.
  - Added focused regression coverage for the new registration metadata and shared construction path.
- **Shared optimizer helpers and absorbed test coverage now have clearer homes** — The optimization package and its tests should stay easier to grow from here instead of accumulating more root-level helpers and one giant optimizer test file.
  - Moved shared optimizer-helper modules under `library/optimization/optimizers/utils/`, including the shared stochastic-rounding helper and the reusable math helpers.
  - Updated the absorbed optimizer implementations plus `library/optimization/adafactor_fused.py` to import from the new `optimizers/utils/` package.
  - Split absorbed optimizer and scheduler registration/construction coverage into `tests/unit/optimizers/test_registry.py` and `tests/unit/optimizers/test_absorbed_integrations.py`.
  - Kept `tests/unit/training/test_training_optimizer.py` focused on shared optimizer/scheduler orchestration, compatibility flags, and config-driven behavior instead of growing it with every absorbed implementation.
- **The absorbed plain-optimizer surface now covers additional large-batch and subspace-style variants too** — The current registration path can keep scaling to richer optimizer behaviors without forcing new factory branches.
  - Added `library/optimization/optimizers/alice.py` with a repo-owned `Alice` implementation adapted from the vendor source.
  - Added `library/optimization/optimizers/lamb.py` with a repo-owned `Lamb` implementation adapted from the vendor source.
  - Added `library/optimization/optimizers/utils/norms.py` with a repo-owned global-gradient-norm helper and updated `Adan` to reuse it instead of importing from `pytorch_optimizer` internals.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so both optimizers participate in the shared repo-owned registration path.
  - Added absorbed-registry and shared-construction coverage for both optimizers under `tests/unit/optimizers/`.
- **The `adopt.py` family is now absorbed as a whole instead of leaving sibling variants behind** — The repo-owned optimization layer can now host the MARS-corrected ADOPT variants without splitting them into a separate donor-driven utility pile.
  - Extended `library/optimization/optimizers/adopt.py` with repo-owned `ADOPTMARS` and `FADOPTMARS` implementations adapted from the vendor source, keeping the whole family together with `ADOPT`.
  - Added `library/optimization/optimizers/utils/clipping.py` with shared `agc(...)`, `adaptive_eps(...)`, and `NORM_TYPE` helpers for optimizer families that need clipping/adaptive-epsilon behavior.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so both new ADOPT-family variants participate in the shared repo-owned registration path.
  - Added absorbed-registry and shared-construction coverage for both new variants under `tests/unit/optimizers/`.

### Changed

- **The TorchAO AdamW low-bit family is now absorbed as a real repo-owned optimizer family** — The optimization layer no longer needs `lr_scheduler_type`-style custom imports or donor package wiring to expose the AO AdamW variants we want to keep.
  - Added `library/optimization/optimizers/adamw_low_bit.py` with `AdamW8bitAO`, `AdamW4bitAO`, and `AdamWfp8AO` as the absorbed low-bit AdamW family built on top of the installed `torchao.optim.adam` classes.
  - Registered the AO family in `library/optimization/registry.py` with an explicit `torchao` backend, and taught the shared optimizer factory to surface missing-TorchAO imports through the same backend-aware error path as the other optional optimizer stacks.
  - Added focused registry and construction coverage so the AO family is exercised through the shared optimization-layer entrypoints rather than only existing as copied implementation files.
- **The `rmsprop.py` family is now absorbed as a whole too** — The repo-owned optimization layer can now host the base RMSProp variant plus its ADOPT-style siblings without falling back to donor utility imports.
  - Added `library/optimization/optimizers/rmsprop.py` with repo-owned `RMSProp`, `RMSPropADOPT`, and `RMSPropADOPTMARS` implementations adapted from the vendor family file.
  - Added `library/optimization/optimizers/utils/second_moment.py` with the shared factored-second-moment helpers used by the RMSProp family and future adaptive optimizers that want the same storage/reconstruction behavior.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so all three RMSProp-family variants participate in the shared repo-owned registration path.
  - Added focused absorbed-registry and shared-construction coverage for the full RMSProp family under `tests/unit/optimizers/`.
- **The `ademamix.py` family is now absorbed behind repo-owned helper seams instead of donor utility imports** — The repo-owned optimization layer can now host the AdEMAMix family without inheriting the vendor package’s giant mixed utility module.
  - Added `library/optimization/optimizers/ademamix.py` with repo-owned `AdEMAMix`, `SimplifiedAdEMAMix`, and `SimplifiedAdEMAMixExM` implementations adapted from the vendor family file.
  - Added repo-owned helper modules for shared AdEMAMix-family needs: `utils/types.py` for update-strategy typing, `utils/orthograd.py` for Newton-Schulz / orthograd helpers, `utils/stable_spam.py` for Stable-SPAM clipping, and `utils/update.py` for reusable update post-processing.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so the full AdEMAMix family participates in the shared repo-owned registration path.
  - Added focused absorbed-registry and shared-construction coverage for the AdEMAMix family under `tests/unit/optimizers/`.
- **The `fcompass.py` family now lives on the repo-owned optimization path too** — The Fisher/Compass variants no longer need to stay in the vendor tree to participate in the shared optimizer registry.
  - Added `library/optimization/optimizers/fcompass.py` with repo-owned `FCompass`, `FCompassADOPT`, `FCompassADOPTMARS`, and `FCompassPlus` implementations adapted from the donor family file.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so the full FCompass family participates in the shared repo-owned registration path.
  - Added focused absorbed-registry and shared-construction coverage for the FCompass family under `tests/unit/optimizers/`.

## [2026-04-02]

### Changed

- **Optimization layer Phase 1 now has shared argument parsing and a first typed parameter-group seam** — The renamed `library/optimization/` package now owns the common optimizer-argument parsing path and a small shared group payload instead of keeping those details duplicated across fine-tune and PEFT setup.
  - Added `library/optimization/arguments.py` with shared `key=value` parsing for optimizer-style argument lists.
  - Added `library/optimization/types.py` with a typed `ParameterGroup` plus compatibility helpers that still materialize the existing optimizer dict payload.
  - Updated `library/training/modes/finetune_mode.py` so full fine-tune now builds typed parameter groups through the shared helper instead of open-coding raw group dicts.
  - Updated `library/optimization/optimizer_factory.py` and `library/optimization/optimizer_utils.py` to reuse the shared parsing/materialization helpers while preserving current runtime behavior.
  - Added focused optimizer and fine-tune mode regression coverage for the new shared parsing/group helpers.
- **Built-in optimizer and scheduler identities now have a first repo-owned registry skeleton** — The optimization layer can start making orchestration decisions from declared metadata instead of relying only on string suffix checks.
  - Added `library/optimization/registry.py` with built-in optimizer/scheduler registrations, effective optimizer-name resolution, and initial capability flags for schedule-free and wrapper behaviors.
  - Updated optimizer construction, schedule-free detection, and scheduler setup to consult the registry metadata where available while keeping current fallback behavior intact.
  - Added focused regression coverage for registry resolution and the built-in schedule-free/wrapper capability flags.
- **A small set of built-in optimizers now construct through registry target metadata instead of only through the legacy branch chain** — The factory can start migrating toward registration-driven construction without forcing a broad behavior change all at once.
  - Added target metadata for `AdamW`, `Lion`, `SGDNesterov`, and the built-in schedule-free optimizers in `library/optimization/registry.py`.
  - Updated `library/optimization/optimizer_factory.py` so those built-ins instantiate through the repo-owned registrations first, while all unported and special-case optimizers still fall back to the existing conditional logic.
  - Added focused coverage to keep the built-in target metadata and `SGDNesterov` default-momentum behavior stable during the migration.
- **The scheduler layer now has its first registry-backed construction path too** — Scheduler metadata is starting to drive real construction behavior instead of serving only as a name map.
  - Added target metadata for the built-in `CosineAnnealingLR` registration in `library/optimization/registry.py`.
  - Updated `library/optimization/scheduler.py` so the `CosineAnnealingLR` path now instantiates through registry target metadata first, while all other scheduler families still use the existing logic.
  - Added focused regression coverage to keep the built-in scheduler target metadata and the registry-backed `CosineAnnealingLR` construction path stable during the migration.
- **Built-in scheduler construction now runs through one shared registry-driven dispatch path** — Scheduler setup no longer needs a long flat branch chain for the standard families.
  - Expanded `SchedulerRegistration` so built-ins now declare whether they route through the `transformers`, `diffusers`, `torch`, or optimizer-embedded scheduler path.
  - Refactored `library/optimization/scheduler.py` into shared internal builders for custom schedulers, registered torch schedulers, diffusers schedulers, optimizer-embedded schedulers, and the standard transformers scheduler family while keeping `get_scheduler_fix(...)` as the single public entrypoint.
  - Added broader regression coverage for registry-backed `constant_with_warmup`, `piecewise_constant`, and custom `lr_scheduler_type` construction alongside the existing `CosineAnnealingLR` checks.
- **Built-in optimizer construction now routes through shared backend-aware builders instead of one long conditional chain** — The optimizer side is starting to match the cleaner scheduler dispatch structure.
  - Expanded `OptimizerRegistration` so built-ins now declare a construction backend such as `torch`, `bitsandbytes`, `dadaptation`, `prodigy`, `transformers`, or `schedulefree`.
  - Refactored `library/optimization/optimizer_factory.py` into shared helpers for Nesterov defaults, adaptive-family warnings, Adafactor preprocessing, registry-backed class loading, and the arbitrary fully-qualified fallback path while preserving the existing public `get_optimizer(...)` entrypoint.
  - Moved the built-in bitsandbytes, D-Adaptation, Prodigy, Adafactor, schedule-free, and core torch optimizer families onto the shared registration-backed path, while keeping wrapper and arbitrary custom optimizers on the fallback path.
  - Added regression coverage for registry-modeled Adafactor behavior and the fully-qualified optimizer fallback path.
- **Optimization-layer helper logic is a bit more centralized ahead of the grouping work** — The next refactor stage no longer needs to keep re-encoding the same import and wrapper-name conventions.
  - Added `library/optimization/loading.py` with a shared target loader used by both optimizer and scheduler construction.
  - Added shared optimizer-name helpers in `library/optimization/registry.py` for schedule-free and wrapper-style name detection instead of repeating that string logic across modules.
  - Refactored `library/optimization/optimizer_utils.py` so orthograd target resolution, optimizer-signature inspection, and adapter optimizer-param preparation now live in smaller helpers instead of one large `prepare_optimizer(...)` branch pile.
- **The legacy schedule-free wrapper config now drives a real wrapper path instead of acting like dead compatibility baggage** — Wrapper-style optimizer behavior no longer depends entirely on fully-qualified fallback paths.
  - Updated `library/optimization/optimizer_factory.py` so `optimizer_schedulefree_wrapper=true` now wraps the constructed base optimizer with `schedulefree.ScheduleFreeWrapper`, exposes the wrapped base optimizer consistently, and preserves the existing optimizer entrypoint.
  - Updated optimizer/scheduler detection helpers so schedule-free wrapper config now participates in wrapper detection, schedule-free train/eval handling, and scheduler-to-base-optimizer routing.
  - Added focused regression coverage for the wrapper config, including base-optimizer exposure and scheduler application against the wrapped base optimizer.
- **Wrapper absorption now has a first explicit repo-owned construction seam** — Registered wrappers no longer have to pretend they share the same constructor shape as plain optimizers.
  - Expanded `OptimizerRegistration` with wrapper-style metadata and concrete wrapper targets for `ScheduleFreeWrapper` and the first absorbed repo-owned `SNOOASGD` wrapper.
  - Added `library/optimization/wrappers.py` with a stable wrapped-optimizer proxy plus a repo-owned `SNOOASGD` implementation adapted from the vendor source.
  - Updated `library/optimization/optimizer_factory.py` so registered wrappers now build a real base optimizer first, support namespaced `base_optimizer.*` arguments for base-optimizer options, and then apply the wrapper through one shared construction path.
  - Added focused regression coverage for explicit `ScheduleFreeWrapper` / `snoo_asgd` optimizer types, including scheduler routing onto the wrapped base optimizer.
- **The first plain absorbed optimizer now lives inside the repo-owned optimization layer** — Plain optimizer absorption no longer depends entirely on vendor fallback paths or external class names.
  - Added repo-owned `AdaBelief` and `Adan` implementations under `library/optimization/optimizers/`, adapted from the vendor source while keeping their `pytorch_optimizer` base helpers.
  - Added `library/optimization/optimizers/stochastic.py` so stochastic-copy behavior now has a shared home used by both absorbed optimizers and the fused Adafactor path.
  - Updated `library/optimization/registry.py` so `AdaBelief` and `Adan` are modeled as repo-backed optimizer registrations instead of vendor-only candidates.
  - Added focused regression coverage for repo-owned AdaBelief / Adan construction and the new repo-backed registration metadata.
- **Absorbed optimization implementations now have dedicated subpackages instead of crowding the package root** — The growing set of repo-owned optimizers and wrappers now has a clearer home.
  - Moved repo-owned optimizer implementations into `library/optimization/optimizers/`.
  - Turned `library/optimization/wrappers/` into the module path for wrapper implementations while keeping the orchestration layer imports stable.
- **The first optimizer-local augmentation now lives on the shared registration path too** — The optimization layer can now host repo-owned optimizer variants that still depend on a third-party backend.
  - Added `library/optimization/optimizers/adamw_8bit_kahan.py` with a repo-owned `AdamW8bitKahan` implementation adapted from the vendor source.
  - Updated `library/optimization/registry.py` so `AdamW8bitKahan` is modeled as a repo-owned optimizer target with a `bitsandbytes` backend dependency instead of living only as a future note.
  - Added focused regression coverage for the new registration metadata and the shared construction path for `AdamW8bitKahan`.

### Fixed

- **VAE cache-signature generation now ignores non-config mock objects instead of trying to serialize them** — Unit tests and other partial config callers no longer fail just because an unspecced mock leaks into a cache-signature field.
  - Hardened `library/data/caching_engine.py` so cache-signature inputs only accept stable string/path-like config values.
  - Cleaned the last stale `library.optimizers` import path from the deprecated SDXL fine-tune script after the package rename.

## [2026-04-01]

### Added

- **SDXL can now use the active rectified-flow path for both training and sample generation** — RF is no longer an SD3-only consumer of the current objective/runtime seam.
  - Updated `library/strategies/sdxl/diffusion.py` so SDXL now branches on the active objective runtime, keeps its DDPM path intact, and adds an RF branch that consumes `RectifiedFlowObjectiveRuntime` batch state directly.
  - SDXL RF training now uses the direct velocity target `noise - latents`, with RF loss weighting flowing through from the shared RF runtime instead of borrowing DDPM target semantics.
  - Updated `library/config/config_validation.py` so `model.model_type=sdxl` now accepts `objective.path='rectified_flow'` with `objective.prediction='flow'`.
  - Added focused unit coverage for the SDXL RF target, RF training-branch behavior, RF checkpoint metadata, and the new SDXL validation allowance.

### Changed

- **SDXL checkpoint metadata and sample generation now follow the active objective path instead of assuming DDPM everywhere** — The SDXL family no longer serializes DDPM prediction metadata or routes sample generation through DDPM-only schedulers when the run is using RF.
  - Updated `library/strategies/sdxl/checkpointing.py` so RF-backed SDXL checkpoints omit DDPM `prediction_type` model-spec metadata, while DDPM-backed SDXL checkpoints keep the existing behavior.
  - Updated `library/strategies/sdxl/sampling.py` so SDXL now keeps the existing DDPM pipeline path for DDPM runs, while RF runs route through a thin backend adapter into the SDXL pipeline layer instead of keeping the RF denoising loop in strategy code.
  - Added `flow_text2img(...)` to `library/pipelines/sdxl_lpw_stable_diffusion.py` so SDXL RF sampling now lives alongside the rest of the SDXL pipeline behavior, including pipeline-owned progress reporting and finite-value checks around RF sampling/decode.
- **Sample generation now routes through an explicit backend seam instead of assuming every model family uses the same latent-returning local pipeline contract** — The common sampling layer can now host repo-local pipelines and custom/backend-owned executors without baking DDPM scheduler setup and latent decoding into every path.
  - Added a normalized `SamplingRequest` plus `SamplingBackend` / `LocalPipelineSamplingBackend` seam in `library/training/sample_generation.py`.
  - Moved prompt/default resolution, width/height normalization, seed setup, save naming, and tracker logging behind the shared orchestration layer instead of duplicating those concerns in family strategies.
  - Reduced SD3 and the SDXL RF branch so they now plug into `sample_images_common(...)` through model-family-specific backend objects, while the existing SD / SDXL DDPM local pipelines still flow through the same entrypoint via the local-pipeline backend adapter.
  - Kept `get_my_scheduler(...)` as the current local DDPM scheduler helper, but stopped making the common sampling loop assume that every backend wants repo-owned scheduler replacement or repo-owned latent decoding.

### Fixed

- **Latent caches now invalidate when the active VAE changes** — Reusing an existing manifest no longer causes stale latent caches to survive a VAE swap.
  - Added `build_vae_cache_signature(...)` to `library/data/caching_engine.py` and threaded the resulting `vae_signature` through the active SD / SDXL / SD3 latent-caching strategies.
  - Cache metadata now records the active VAE source plus effective padding mode, so existing latent caches are treated as invalid when those inputs change.
  - Added focused regression coverage in `tests/unit/data/test_sdxl_cache_roundtrip.py` for VAE-signature mismatch invalidation.
- **Objective-validation warnings now match the active math path more closely** — RF configs no longer inherit DDPM-specific warning noise just because they still carry legacy compatibility mirrors.
  - Updated `library/config/config_validation.py` so the `zero_terminal_snr` warning only fires on the DDPM path, where that scheduler-shaping setting actually applies.
  - Kept `loss.v_parameterization` synchronized as a legacy DDPM compatibility mirror without warning for RF configs that explicitly set `objective.prediction='flow'`.
  - Added focused config-validation coverage so RF configs stay free of the DDPM-only warning path.

## [2026-03-31]

### Changed

- **Objective config now separates training-path choice from prediction-target choice, and the active schema no longer infers either axis through `auto`** — The current math layer now declares the two axes explicitly instead of overloading one mixed `objective.target` field.
  - Replaced `objective.target` with explicit `objective.path` and `objective.prediction` fields in the shared run schema and default config fragments.
  - Removed the active `auto` resolution path from config preparation and objective selection, so the runtime no longer infers DDPM vs RF or epsilon vs v-pred behind the scenes.
  - Updated DDPM diffusion, sampling, loading, and checkpoint metadata paths to read `cfg.objective.prediction` directly, while `build_objective(...)` now selects the owner from `cfg.objective.path`.
  - Kept `loss.v_parameterization` only as a synchronized legacy compatibility mirror for `objective.prediction == "v_prediction"`, and updated validation messages to point at the explicit objective fields.
  - Tightened the supported combination matrix too: DDPM now allows only `epsilon` / `v_prediction`, while rectified flow now requires the explicit RF-native `flow` prediction label.
  - The active SD3/RF diffusion path now validates that explicit `flow` prediction contract instead of silently ignoring the prediction field.
- **Prediction-target metadata now stops at the DDPM family boundary** — The active repo no longer serializes DDPM-style `epsilon`/`v` model-spec metadata for SD3 checkpoints just because the shared config still carries `v_parameterization`.
  - Updated `library/utils/model_metadata.py` so active callers now choose `modelspec.prediction_type` explicitly instead of relying on model-family checks inside the shared helper.
  - SD / SDXL checkpoint strategies now pass their DDPM `epsilon`/`v` choice explicitly, while SD3 passes `None` to omit the field.
  - Added focused unit coverage to keep SD3 model-spec metadata from inheriting the DDPM `prediction_type` axis.
  - Tightened the `v_parameterization` config help text and design notes so they describe a DDPM prediction-target choice rather than a generic loss toggle.
- **DDPM prediction-target behavior now has one objective-owned mapping instead of repeated boolean branches** — The active DDPM path no longer re-decides `epsilon` vs `v_prediction` separately in each diffusion strategy and sampling entrypoint.
  - Added a DDPM-owned prediction-type resolver and training-target builder in `library/objectives/ddpm.py`.
  - SD / SDXL diffusion strategies now build the DDPM training target through that objective seam instead of branching on `cfg.loss.v_parameterization` locally.
  - Sample-time scheduler setup now accepts an explicit DDPM prediction type, and the active SD / SDXL sampling paths resolve it through the same objective-owned mapping.
- **Zero-terminal-SNR is now described as DDPM scheduler shaping instead of generic loss regularization** — The active config/help/docs now point at the path/state-construction role the setting actually has.
  - Updated the typed config help text and validation warning for `loss.regularization.zero_terminal_snr` so they describe DDPM scheduler shaping for noisy-state construction.
  - Refreshed the active design notes to call out `zero_terminal_snr` as a scheduler/state-construction option instead of describing it like ordinary loss regularization.
- **Loss-weighting ownership now distinguishes DDPM post-loss math from generic masking** — The active code no longer keeps DDPM-only SNR/v-pred weighting and generic mask application in the same helper module.
  - Moved Min-SNR weighting, debiased-estimation weighting, v-pred scaling, and the shared DDPM post-processing order into `library/objectives/ddpm.py`.
  - Added `library/losses/masking.py` so mask application stays with generic loss behavior instead of living in an objective-shaped weighting module.
  - Updated SD / SDXL diffusion strategies to call `post_process_ddpm_loss(...)` from the DDPM objective seam, while SD3 now imports only the shared masking helper.
  - Refreshed the focused unit coverage around DDPM weighting and masking behavior.
- **Huber threshold scheduling now has its own small loss helper seam** — The active strategies no longer import a scheduler/timestep-aware threshold helper from the same module that owns the raw loss functions.
  - Added `library/losses/huber.py` with `get_huber_threshold_if_needed(...)` and updated SD / SDXL / SD3 diffusion strategies to import the helper from there.
  - Reduced `library/losses/loss.py` back to the actual loss primitives and `conditional_loss(...)` dispatch path.
  - Updated the focused unit coverage so the Huber-threshold tests follow the new helper location.
- **Objective runtime ownership now reaches the trainer/strategy seam instead of stopping at runtime construction** — The active trainer no longer keeps a DDPM-shaped `noise_scheduler + timestep_runtime + loss_modifier` field split after objective selection.
  - `Trainer` now stores one `objective_runtime` bundle directly, and the shared training/validation loops pass that bundle through to strategy calls instead of threading a raw scheduler as a fake universal dependency.
  - `ObjectiveRuntime` now carries only the genuinely shared runtime fields (`num_train_timesteps`, optional `alphas_cumprod`, optional `timestep_runtime`, and `loss_modifier`), while DDPM and RF each expose typed runtime subclasses for their path-specific state.
  - SD / SDXL diffusion and validation now require a DDPM runtime explicitly when they need scheduler math, while SD3 / RF now accepts a rectified-flow runtime without inheriting a DDPM scheduler by accident.
  - Live timestep plotting and Huber-threshold scheduling now read objective-owned runtime metadata instead of assuming every active path has a raw Diffusers scheduler object.
  - `RectifiedFlowObjective.build_runtime()` is now RF-native and fails fast on DDPM-only EDM2 weighting instead of silently inheriting DDPM runtime assembly.
  - Followed up on the first runtime pass too: the shared base runtime no longer carries the DDPM-only `alphas_cumprod` field, the batch-feedback hook is now named `update_from_batch(...)` instead of `observe(...)`, and the DDPM family strategies now read their typed runtime via local casts instead of runtime assertion helpers.
- **RF runtime ownership now includes RF batch-state assembly instead of stopping at runtime identity** — The active SD3 strategy no longer owns RF timestep/sigma/model-input construction directly.
  - Added `RectifiedFlowObjectiveRuntime.build_training_batch_state(...)` plus a small `RectifiedFlowBatchState` container in `library/objectives/rectified_flow.py`.
  - RF runtime construction now stores the active RF timestep config and RF loss-weighting scheme so the runtime can assemble `timesteps`, `sigmas`, noisy model input, and loss weighting from clean latents.
  - `library/strategies/sd3/diffusion.py` now consumes that RF-owned batch state instead of rebuilding RF timestep state locally, which makes the SD3 strategy read more like denoiser orchestration plus loss application.
  - Added focused unit coverage for the runtime-owned RF batch-state builder in `tests/unit/training/test_training_flow.py`.
- **The active SD3 RF target now follows the paper-style direct velocity contract instead of supervising a projected clean latent inside the shared RF runtime** — Generic RF batch-state assembly now stops at path construction, while the SD3 strategy owns the family-specific target semantics directly.
  - Removed the generic `target` field from `RectifiedFlowBatchState`, so the shared RF runtime now owns only noise, interpolated model input, timesteps, sigmas, and RF loss weighting.
  - Added an SD3-local `build_sd3_flow_target(...)` helper in `library/strategies/sd3/diffusion.py` and switched the active SD3 training target to `noise - latents`, matching the direct RF velocity target described in the SD3 paper.
  - Removed the old SD3 training-side projection `model_pred = model_pred * (-sigmas) + noisy_model_input` from the active SD3 loss path, since that projected-`x0` supervision was an older donor-specific parameterization rather than the paper-native RF target.
  - Added focused unit coverage so the RF runtime test now checks only shared batch-state fields and the SD3 strategy test asserts the paper-style target direction explicitly.

### Fixed

- **SDXL conditioning regression introduced by the objective-runtime refactor** — The SDXL training and validation paths now call the shared conditioning facet using the same keyword argument contract as the other families.
  - Updated `library/strategies/sdxl/diffusion.py` and `library/strategies/sdxl/validation.py` to pass `batch`, `text_encoders`, `accelerator`, `cfg`, and `weight_dtype` by name when resolving conditioning.
  - Added focused regression coverage in `tests/unit/strategies/test_strategies_sdxl.py` so the SDXL batch-processing and validation paths fail loudly if they drift back to the old positional call shape.

### Removed

- **Removed the old mixed DDPM weighting helper module** — The repo no longer keeps DDPM-only weighting logic in a generic loss helper file.
  - Deleted `library/losses/loss_weighting.py` after moving DDPM post-loss weighting into `library/objectives/ddpm.py` and generic masking into `library/losses/masking.py`.

## [2026-03-30]

### Changed

- **RF timestep-density and RF loss-weighting are now modeled as separate config concepts** — The active RF path no longer overloads one `weighting_scheme` field to mean both density sampling and post-loss weighting.
  - Replaced the old overloaded `timestep.weighting_scheme` with a dedicated `timestep.rf_loss_weighting_scheme`, but folded RF timestep-density selection back into the main `timestep.timestep_sampling` surface instead of keeping a second parallel sampling selector.
  - Updated `library/objectives/rectified_flow.py` so RF now reads training-time sampling from `timestep_sampling`, treats `logit_normal` as the canonical logit-family sampler, and keeps `cosine_shaped` as the remaining RF-local density option.
  - Renamed the vague RF density label `mode` to `cosine_shaped`, and `mode_scale` to `cosine_shape_scale`, including the shared RF metadata field.
  - Updated config validation to enforce the currently implemented timestep-sampling values by active model family, and updated RF metadata keys to record the shared `ss_timestep_sampling` field instead of an RF-only density-selector name.
  - Renamed the remaining script-era timestep knobs `sigmoid_scale` and `discrete_flow_shift` to `logit_scale` and `training_shift` so the active config surface describes training-time behavior more directly.
  - Collapsed the redundant `shift` RF sampler into `logit_normal + training_shift`, removed the now-redundant `logit_scale` knob, and added a shared timestep-density helper documenting the equivalence in code.
  - Followed up on the shared sampler naming so the extracted helper now lives in `library/timesteps/continuous_sampling.py`, uses `sample_continuous_timesteps(...)`, and keeps the intermediate RF variable as `u` instead of introducing a misleading `density`/`timestep_values` term.
- **Objective/runtime ownership now has a first explicit home outside trainer and model-family helpers** — The active runtime no longer wires DDPM scheduler setup, timestep runtime construction, RF metadata ownership, and post-loss modifier assembly as unrelated pieces.
  - Added `library/objectives/` with explicit objective owners plus a small runtime bundle/factory surface.
  - Trainer runtime initialization now resolves one objective owner, builds its runtime bundle, and reads checkpoint metadata hooks from that same objective seam.
  - Moved DDPM scheduler construction into `library/objectives/ddpm.py`, while `library/training/noise_utils.py` now keeps only the reusable noise-regularization helpers.
  - Moved the shared RF training helpers from `library/training/flow.py` into `library/objectives/rectified_flow.py`, and updated SD / SDXL / SD3 diffusion code to import their objective-owned helpers from the new package.
  - Added focused test updates for the new objective-owned DDPM scheduler / batch-input helper path and the RF metadata / helper imports.
- **RF config ownership now points at timestep/sampling surfaces instead of pretending to be model identity** — The first SD3/RF port no longer reads its weighting and flow-shift settings only from ad hoc `cfg.model` fields.
  - Added typed RF timestep settings to `library/config/dataclasses/timestep.py` and `configs/_defaults/timestep/default.yaml`: shared `timestep_sampling`, `rf_loss_weighting_scheme`, `logit_mean`, `logit_std`, and `cosine_shape_scale`.
  - Added `sample_flow_shift` to `library/config/dataclasses/output.py` under `output.sampling`, with the shared output defaults YAML exposing the new field too.
  - Updated the active SD3 diffusion, sampling, and checkpoint metadata paths to read the new typed config homes directly.
  - Added focused unit coverage for the new config ownership in the SD3 strategy tests plus default-value assertions in the config tests.
- **RF objective metadata no longer lives in SD3 checkpoint strategy code** — The shared training metadata builder now owns the current RF metadata fields, while SD3 checkpointing keeps only genuinely SD3-specific metadata.
  - Added `append_objective_metadata(...)` to `library/training/training_metadata.py` and called it from `create_training_metadata(...)`.
  - Moved `ss_timestep_sampling`, `ss_rf_loss_weighting_scheme`, `ss_logit_mean`, `ss_logit_std`, and `ss_cosine_shape_scale` out of `library/strategies/sd3/checkpointing.py`.
  - Kept `ss_apply_lg_attn_mask` and `ss_apply_t5_attn_mask` in the SD3 checkpoint strategy because those remain family-specific encoding/runtime flags.
- **The sampler call interface is now slimmer and more honest** — The runtime no longer threads shift-only knobs through every sampler-backed path.
  - `log_snr_uniform` and `adaptive_log_snr` no longer accept or discard the shift-only time-sampling knobs, and the runtime only passes the shared sampling inputs that sampler-backed modes actually use.

## [2026-03-29]

### Changed

- **RF training math has its first shared runtime home outside SD3 strategy code** — The flow-matching helper functions used by SD3 training no longer live in the model-family strategy module.
  - Added `library/training/flow.py` with shared flow-matching timestep-density, loss-weighting, and noisy-input construction helpers.
  - Reduced `library/strategies/sd3/diffusion.py` so it now uses those training-side helpers instead of owning the RF math locally.
  - Added focused unit coverage in `tests/unit/training/test_training_flow.py` for the extracted helper surface.
- **Discrete-flow sampling math now has a pipeline-side home outside SD3 strategy code** — The reusable sigma-schedule helpers used by SD3 sampling no longer live under the model-family strategy package.
  - Added `library/pipelines/flow.py` with `DiscreteFlowModelSampling`, `get_discrete_flow_sigmas(...)`, and `starts_at_max_denoise(...)`.
  - Reduced `library/strategies/sd3/sampling.py` so it keeps SD3 prompt encoding and sampling orchestration, but imports the discrete-flow runtime math from the pipeline-side helper module.
  - Added focused unit coverage in `tests/unit/test_pipelines_flow.py` for the extracted discrete-flow helper surface.
- **CLIP-family model-preparation workarounds now have one explicit shared home** — The repo no longer keeps the same CLIP embedding prep helpers duplicated across SD and SDXL, and SD3 now reuses that shared CLIP branch while keeping its T5-specific behavior local.
  - Added `library/strategies/shared/clip/model_preparation.py` for the shared CLIP text-encoder gradient-checkpointing workaround and FP8 embedding restore helper.
  - Reduced `library/strategies/sd/model_preparation.py` and `library/strategies/sdxl/model_preparation.py` to family-specific policy around those shared CLIP helpers instead of each owning identical embedding prep code.
  - Updated `library/strategies/sd3/model_preparation.py` so CLIP-L / CLIP-G reuse the shared helper path while T5-XXL keeps its explicit local grad-checkpointing and FP8 behavior.
  - Added focused unit coverage for the new shared CLIP model-preparation helpers.

## [2026-03-28]

### Changed

- **CLIP tokenization sharing now has an explicit split between component bootstrap and strategy behavior** — The repo no longer keeps identical CLIP tokenizer loading logic duplicated across SD / SDXL / SD3 strategy files, and the shared CLIP prompt-tokenization behavior now has one home.
  - Added `library/models/sd/tokenizer.py` as the shared Hugging Face tokenizer loading/bootstrap helper used by SD, SDXL, and the CLIP side of SD3.
  - Added `library/strategies/shared/clip/tokenization.py` for shared CLIP-family prompt-weight parsing, long-prompt chunking, and caption tokenization behavior.
  - Reduced `library/strategies/sd/tokenization.py` and `library/strategies/sdxl/tokenization.py` to family-specific assembly on top of those shared seams, while `library/strategies/sd3/tokenization.py` now reuses the shared loader without forcing SD3’s CLIP+T5 behavior into the CLIP-only helper layer.
- **Weighted prompt support is now modeled as an explicit optional strategy capability** — The base strategy contracts no longer imply that every active model family must support weighted tokenization/encoding just because SD and SDXL do.
  - Added `WeightedPromptStrategy` in `library/strategies/base/features.py` for the paired `tokenize_with_weights()` and `encode_tokens_with_weights()` capability.
  - `SdTrainingStrategy` and `SdxlTrainingStrategy` now opt into that capability explicitly, while SD3 no longer carries placeholder weighted-prompt methods that only raise `NotImplementedError`.
  - Updated the base strategy tests so weighted prompt support is asserted on the optional capability surface rather than on the required tokenization/text-encoding facets.
- **Initial SD3 model-family groundwork is now present in the active library/strategy path** — The repo now has a first real SD3 implementation surface wired into the shared strategy system instead of only carrying SD / SDXL.
  - Added `library/models/sd3/` component code for SD3 checkpoint conversion, MMDiT construction/loading, CLIP-L / CLIP-G / T5-XXL loading, and SD3 VAE handling.
  - Added `library/strategies/sd3/` facet files for SD3 tokenization, text encoding, caching, denoiser calling, diffusion training, validation, checkpointing, sample generation, model preparation, and strategy assembly.
  - Registered `Sd3TrainingStrategy` in the shared strategy factory so the config-driven launcher can resolve the SD3 family through the same `model.model_type -> TrainingStrategy` path used by the other active families.
  - SD3 loading now supports the current unified-checkpoint path plus optional sidecar text encoders, scaled positional embeddings, block-swap setup, and the active partial FP8 handling used by the current strategy implementation.
  - SD3 training/runtime behavior now includes CLIP-L + CLIP-G + T5 tokenization/encoding, SD3-specific cache payload handling, flow-matching diffusion/loss-weighting helpers, SD3 validation loss execution, SD3 metadata population, safetensors full-model checkpoint saving, and direct SD3 sample generation.
  - Added named SD3-local text payload dataclasses in `library/strategies/sd3/encoding.py`, and rewired SD3 tokenization / encoding / diffusion / denoiser / sampling / caching internals to use those named payloads instead of positional six-item tensor lists.
  - The current SD3 path is still intentionally partial: weighted captions / prompt weighting are not implemented yet, T5-XXL FP8 preparation still fails fast, and full-model checkpoint saving currently supports only `save_model_as='safetensors'`.
- **Conditioning is now a real strategy facet instead of staying half-buried in diffusion helpers** — The shared strategy surface now has one explicit high-level conditioning seam, while each family keeps its own cache/live/merge implementation details local.
  - Added `ConditioningStrategy.resolve_conditioning(...)` to `library/strategies/base/contracts.py` as the small trainer/runtime-facing contract for family conditioning resolution.
  - Added real family facets in `library/strategies/sd/conditioning.py`, `library/strategies/sdxl/conditioning.py`, and `library/strategies/sd3/conditioning.py` to own cached-conditioning loading, live encoding fallback, and merge policy.
  - Reduced `sd/diffusion.py`, `sdxl/diffusion.py`, and `sd3/diffusion.py` so they now consume `resolve_conditioning(...)` instead of carrying their own private `_get_text_conds()` flow, and updated validation paths to use that same seam.
  - Kept family payload shapes local: SD still returns its simple text tensor list, SDXL still resolves the current text tuple alongside its existing `SdxlConditioning` metadata path, and SD3 still uses `Sd3TextConditioning` from `sd3/encoding.py`.
- **The generic launcher no longer hides behind compatibility wrappers** — The active entry surface is now just the root launcher we already expect users and benchmarks to call.
  - Inlined the shared launch body into `train.py` and removed `library/training/launcher.py`.
  - Removed the `scripts/sdxl_peft.py` and `scripts/sdxl_finetune.py` compatibility wrappers instead of keeping duplicate Hydra entrypoints around after the root launcher was already working.
  - Updated launcher tests to exercise `train.py` directly.
- **SD / SDXL CLIP behavior now lives with strategy tokenization/encoding instead of under `library/models/sd`** — The active ownership split is now more consistent with the strategy contracts and the design note for model vs behavior code.
  - Moved the current SD text-encoding helpers into `library/strategies/sd/encoding.py` and the SDXL text-encoding helpers into `library/strategies/sdxl/encoding.py`.
  - Moved the CLIP-family tokenization helpers out of `library/models/sd/tokenizer.py` into the SD / SDXL strategy tokenization modules directly; for now the identical helper logic is duplicated locally instead of introducing a new shared/base strategy module.
  - Deleted the old model-layer helper files `library/models/sd/text_encoder.py`, `library/models/sd/tokenizer.py`, and `library/models/sdxl/text_encoder.py`.
  - Updated the active SD / SDXL strategy, caching, pipeline, and tests/imports to use the new strategy-owned locations.
- **Prompt-attention parsing now has one canonical implementation again** — The repo no longer carries separate LPW-pipeline copies of the same parser.
  - Kept `library/data/prompt_utils.py::parse_prompt_attention()` as the canonical implementation and upgraded it to the best typed/docstring variant from the duplicated copies.
  - `library/pipelines/lpw_stable_diffusion.py` and `library/pipelines/sdxl_lpw_stable_diffusion.py` now import that canonical parser instead of defining their own local copies.
