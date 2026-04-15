# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Rules:
- Use proper sub titles "Added", "Changed", "Removed" and "Fixed"
- Keep proper track of days for where entries should go
- Be concise but mention all changes without necessarily detailing each one

## [2026-04-13]

### Changed

- **Base fine-tune grouping now supports simple named parameter-group LR overrides on top of component/default learning rates** — The optimizer config can now express user-facing override groups under `optimizer.learning_rates.groups`, matched against live named parameters, and `base: null` now means “only explicitly named things train.”
  - Updated `library/config/dataclasses/optimizer.py`, `configs/_defaults/optimizer/default.yaml`, `library/optimization/grouping.py`, and `library/training/modes/finetune_mode.py` with inline `optimizer.learning_rates.groups` support, optional `optimizer.learning_rates.groups_file` loading from YAML, live named-parameter matching (glob by default, `re:` for regex), and fine-tune grouping/trainability resolution that layers named overrides over denoiser/text-encoder defaults.
  - Updated `library/optimization/optimizer_utils.py` so base/component trainability helpers treat `base: null` as “no fallback training” instead of implicitly enabling denoiser/text-encoder training.
  - Updated `library/training/modes/peft_mode.py` to fail fast if inline or file-backed fine-tune grouping overrides are configured, since adapter-specific grouping is still deferred to the later adapter rework.
  - Added focused coverage in `tests/unit/training/test_training_optimizer.py` and `tests/unit/training/modes/test_finetune_mode.py` for `base: null` semantics, `groups_file` loading, group-driven component activation, and named-group overrides splitting matched denoiser parameters away from the component remainder.
- **The repo now has a standalone named-parameter inspection tool for loaded model runtimes** — You can dump live parameter names grouped by explicit family component names (`clip_l`, `clip_g`, `t5xxl`, `vae`, `unet`, `mmdit`) instead of inferring them from state-dict structure.
  - Added `library/models/parameter_dump.py` with reusable helpers for family-aware component mapping and compact YAML rendering.
  - Added `tools/model_management/dump_named_parameters.py` to load a model through the active strategy path and emit one-line-per-parameter YAML, with optional component filtering and `--trainable-only`.
  - Added focused unit coverage in `tests/unit/models/test_parameter_dump.py` for SD3 component naming and formatter output shape.

## [2026-04-12]

### Added

- **The optimization-layer architecture work now has a persistent OpenSpec change scaffold** — The repo can keep the logical-grouping redesign proposal, design notes, and task breakdown alongside the implementation work instead of relying only on transient conversation context.
  - Added `openspec/changes/optimization-logical-grouping-foundation/` with proposal, design, spec, and task artifacts covering the base-path logical grouping foundation.

### Changed

- **The first base-path optimization-plan foundation is now in the active codepath** — Fine-tune mode can now emit stable logical-group metadata without requiring trainer-facing code to reconstruct meaning entirely from runtime `optimizer.param_groups`.
  - Updated `library/optimization/types.py` with `LogicalParameterGroup`, `OptimizationPlan`, `OptimizerBuildResult`, and a shared logical-group builder while preserving legacy execution-group materialization.
  - Updated `library/training/modes/finetune_mode.py`, `library/training/modes/base.py`, and `library/training/phases/optimizer.py` so the fine-tune optimizer build seam can return a plan-aware result, the optimizer phase stores plan metadata on the trainer, and the legacy tuple path remains available for compatibility-oriented callers.
  - Updated `library/training/runners/trainer.py`, `library/training/trainer_utils.py`, `library/training/phases/training_loop.py`, and `library/logging/step_logging.py` so startup diagnostics and step LR logging can prefer logical-group metadata from the stored optimization plan instead of only reading raw runtime optimizer groups plus a parallel string list, and the plan-backed path no longer requires callers to keep passing populated legacy `lr_descriptions` lists just to label groups correctly.
  - Updated the shared stochastic-rounding helper imports so wrappers now use the optimizer-utils implementation directly, and `library/optimization/stochastic.py` remains only as a compatibility re-export instead of carrying a second copy of the function body.
  - Added/updated focused unit coverage in `tests/unit/training/test_training_optimizer.py`, `tests/unit/training/modes/test_finetune_mode.py`, `tests/unit/training/phases/test_optimizer.py`, `tests/unit/training/phases/test_training_loop.py`, `tests/unit/training/test_training_trainer_utils.py`, and `tests/unit/logging/test_step_logging.py` for the new logical-group/plan types and the base-path plan-aware logging contract.
- **Base-path grouping policy now has a shared optimization-layer home instead of living inline in `FineTuneMode`** — The plan foundation is now backed by a reusable grouping helper so the mode keeps trainable selection ownership without also owning denoiser/text-encoder group assembly details.
  - Added `library/optimization/grouping.py` with a shared `build_finetune_grouping(...)` helper and `GroupingResult` payload for base-path execution/logical-group construction.
  - Updated `library/training/modes/finetune_mode.py` so fine-tune optimizer preparation uses the shared grouping helper before creating the `OptimizationPlan`.
  - Added focused coverage in `tests/unit/training/test_training_optimizer.py` and `tests/unit/training/modes/test_finetune_mode.py` for shared grouping order, learning-rate behavior, and the fine-tune integration seam.
- **The optimization plan now models execution groups explicitly instead of only carrying a generic parameter-group payload** — The base-path plan shape now names the execution side directly while preserving compatibility for older callers that still expect `parameter_groups`.
  - Updated `library/optimization/types.py` so `OptimizationPlan` stores `execution_groups`, keeps a compatibility `parameter_groups` accessor, and still materializes legacy optimizer dicts through the shared boundary.
  - Updated `library/optimization/grouping.py` and `library/training/modes/finetune_mode.py` so the shared fine-tune grouping flow returns and consumes execution-group-oriented data.
  - Added focused coverage in `tests/unit/training/test_training_optimizer.py` and `tests/unit/training/modes/test_finetune_mode.py` for execution-group compatibility accessors and the base-path fine-tune integration.
- **Scheduler ownership is now starting to move into explicit optimization-plan runtime metadata instead of staying purely heuristic inside the scheduler entrypoint** — Plan-aware paths can now carry scheduler mode/target information while legacy callers still keep the older fallback behavior.
  - Updated `library/optimization/types.py`, `library/optimization/scheduler.py`, and `library/training/phases/optimizer.py` with explicit scheduler/runtime metadata, a shared resolver, and plan-aware scheduler construction.
  - Added focused coverage in `tests/unit/training/test_training_optimizer.py` and `tests/unit/training/phases/test_optimizer.py` for plan-owned scheduler target selection and optimizer-phase metadata population.
- **Plan-aware optimizer runtime transitions no longer pretend the legacy callback pair is part of the primary contract** — The fine-tune/base path now relies on plan-owned runtime metadata directly, while callback-pair handling stays confined to the legacy tuple boundary.
  - Updated `library/optimization/types.py`, `library/training/modes/base.py`, `library/training/modes/finetune_mode.py`, and `library/training/phases/optimizer.py` so `OptimizerBuildResult` no longer carries train/eval callbacks, fine-tune stops constructing them, and the optimizer phase clears legacy callback state on plan-aware results.
  - Updated `library/optimization/optimizer_utils.py` and `library/training/runners/trainer.py` to document the remaining callback pair as compatibility-only state for older tuple-based callers.
  - Added focused coverage in `tests/unit/training/modes/test_finetune_mode.py`, `tests/unit/training/phases/test_optimizer.py`, and `tests/unit/training/test_training_trainer.py` to confirm plan-aware runtime transitions do not depend on callback-pair plumbing.

## [2026-04-11]

### Changed

- **The Torch/TorchAO compatibility story is now expressed through the Torch version profiles instead of a broad unconditional TorchAO dependency** — The base environment no longer claims one `torchao` line works across every supported Torch version, and the project now has an explicit `torch-v211` profile too.
  - Updated `pyproject.toml` to widen the core Torch range to `<2.12`, remove the unconditional base `torchao` dependency, add a `torch-v211` profile, and bundle TorchAO version ranges with the `torch-v29`, `torch-v210`, and `torch-v211` extras according to the supported compatibility table.
- **The CUDA wheel extras now mirror the mainstream PyTorch installer lanes more closely** — The dependency matrix no longer carries the awkward `cu129` lane, and the extra/conflict layout now cleanly models “pick one CUDA lane and one Torch version lane.”
  - Updated `pyproject.toml` to replace `torch-cu129` with `torch-cu126`, add the `pytorch-cu126` index/source mapping, and remove the stale cross-conflict rules between CUDA-lane extras and Torch-version extras.
- **The repo no longer declares the old git-installed `customized-optimizers` package as a dependency** — The absorbed optimizer surface now lives in-repo, and the lingering smoke config has been retargeted to the repo-owned optimizer name instead of the old vendor package path.
  - Updated `pyproject.toml` to remove the `customized-optimizers` dependency and its `tool.uv.sources` git entry.
  - Updated `smoke_test_customoptimizer.yaml` so the smoke config uses `SimplifiedAdEMAMixExM` directly and the current `use_orthograd` argument spelling.
- **All PyTorch CUDA package indexes are now consistently explicit in the uv source map** — The `cu130` index no longer behaves differently from `cu126`/`cu128` during base dependency resolution.
  - Updated `pyproject.toml` so `[[tool.uv.index]] pytorch-cu130` also uses `explicit = true`, which keeps Torch/TorchVision resolution behind the intended extra-gated source mapping.
- **The PyTorch index layout now preserves the old “cu130 as default fallback, cu126/cu128 as explicit override lanes” behavior** — This avoids conflicting Torch index assignments when a non-default CUDA extra is selected while still keeping plain `uv run ...` aligned with the normal cu130 workflow.
  - Updated `pyproject.toml` so `pytorch-cu130` is no longer `explicit = true`, and `tool.uv.sources` again only binds CUDA-specific Torch/TorchVision sources through the explicit `torch-cu*` extras.
- **The experimental WiwiOpt copy now tracks the newer V1.3 algorithm shape instead of the older absorbed variant** — The repo-owned experimental copy now carries the newer factorized-variance and PAST-capable update path while keeping the shared stochastic-rounding and Windows compile-bootstrap integrations.
  - Updated `wiwiopt.py` with the `WiwiOptV1.py` algorithm changes, including CAME-style factorized variance tracking, `weight_decay_rate`, and configurable `egd_method` support.
  - Updated `registry.py` so the `WiwiOpt` registration points at the experimental package path used after the optimizer package reorganization.
  - Updated `test_absorbed_integrations.py` to reflect the new first-step WiwiOpt state layout (`exp_avg_sq_row` / `exp_avg_sq_col`) instead of the older `polyak` state.

### Fixed

- **Repo-owned TorchAO optimizer and wrapper integration now handles several shared factory/runtime edge cases more cleanly** — Recent benchmark and smoke-test failures no longer fall through into confusing runtime crashes or signature-probing warnings.
  - Updated `library/optimization/optimizer_utils.py` so orthograd signature probing resolves registered repo-owned optimizers and wrapper base optimizers through the registry instead of incorrectly probing `torch.optim` or treating short names like full import paths.
  - Updated `library/optimization/optimizers/adopt/adopt_schedulefree_ao.py` and `library/optimization/optimizers/compass/compass.py` so repo-owned TorchAO optimizers normalize per-group learning rates to tensors when parameter groups are added or state is reloaded, which avoids step-time crashes when grouped params supply float LRs.
  - Updated `library/optimization/optimizer_factory.py` so wrapper-style optimizer kwargs are split more accurately between wrapper-owned args and base-optimizer args, including support for bare base-optimizer kwargs in `CPUOffloadOptimizer` configs.
  - Updated `library/optimization/arguments.py` so lowercase config-style literals such as `true`, `false`, and `null` are parsed into proper Python values instead of leaking through as strings.
  - Updated `library/optimization/wrappers/cpu_offload.py` so `CPUOffloadOptimizer` fails fast with a clear repo-owned error when paired with bitsandbytes optimizers, which are incompatible with TorchAO CPU offload's CPU-side optimizer stepping model.
  - Added/updated focused regression coverage in `tests/unit/training/test_training_optimizer.py` and `tests/unit/optimizers/test_absorbed_integrations.py` for the signature lookup, TorchAO tensor-LR normalization, CPU offload wrapper kwarg routing, lowercase literal parsing, and bitsandbytes incompatibility guard.

## [2026-04-10]

### Added

- **The last top-level donor optimizer leaves are now repo-owned too** — The repo no longer needs to leave `adammini.py` or `clybius_experiments.py` behind in the authoritative vendor tree just because they were awkward fits for the shared optimizer layer.
  - Added `[adammini.py` with a repo-owned `AdamMini` implementation adapted from the authoritative vendor tree while preserving its name-aware grouping logic for attention/embed-specific update paths.
  - Added `momentus_caution.py` and `remaster.py` by splitting the donor `clybius_experiments.py` file into one-optimizer-per-file repo-owned leaves while keeping the key donor comments/docstrings.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `AdamMini`, `MomentusCaution`, and `REMASTER` participate in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `AdamMini` named-group construction plus first-step specialized state initialization, and for `MomentusCaution` / `REMASTER` construction plus first-step running-state initialization through the shared factory path.

### Changed

- **The shared optimizer group materialization now preserves parameter names for name-aware repo-owned optimizers** — Optimizers adapted from donor code no longer have to bypass the factory path just because they need module parameter names to rebuild specialized group structure.
  - Updated `library/optimization/types.py` so `build_module_parameter_group(...)` preserves a materialized `named_params` payload alongside the flat `params` list.
  - Updated `adammini.py` so the repo-owned `AdamMini` can reconstruct donor-style embed/QK/attention grouping from shared factory param groups instead of requiring a raw `nn.Module`.
- **The AdamMini adaptation also hardens a donor broadcasting edge on newer PyTorch builds** — The repo-owned copy now avoids the donor’s in-place broadcast multiply in attention update paths, which can raise on current PyTorch instead of applying the intended scaling.
  - Updated `adammini.py` so the attention-projection and grouped-attention update branches use equivalent non-in-place scaling math that keeps the donor behavior without depending on in-place broadcast semantics.
- **The optimization wrapper layer now exposes TorchAO CPU optimizer offload through the shared factory path** — The repo no longer has to leave the old vendor `low_bit_optim/cpu_offload.py` script dangling just to access that feature.
  - Added `cpu_offload.py` with a repo-owned `CPUOffloadOptimizerWrapper` that wraps upstream `torchao.optim.CPUOffloadOptimizer` through the existing `base_optimizer_type=...` wrapper config pattern.
  - Updated `library/optimization/registry.py`, `library/optimization/optimizer_factory.py`, `library/optimization/optimizer_utils.py`, and `library/optimization/wrappers/__init__.py` so `CPUOffloadOptimizer` is a registered wrapper, preserves base-optimizer kwargs during wrapper construction, and participates in wrapper detection/signature handling cleanly.
  - Added focused regression coverage for `CPUOffloadOptimizer` registration metadata, wrapper construction with namespaced base args, scheduler integration, and fast-fail behavior when no CUDA/XPU runtime is available.
- **The older repo-owned optimization surface now has package-alignment guardrails too** — The earlier in-repo optimizers and schedulers were already on the shared registry/factory path, and the test suite now checks that they stay aligned with the public package exports as the package is cleaned up.
  - Updated `tests/unit/optimizers/test_registry.py` with coverage that iterates registered optimizer and scheduler targets under `library.optimization.optimizers.*` / `library.optimization.schedulers.*` and asserts the package-level exports resolve to the same underlying classes.
  - Updated `library/optimization/optimizers/README.md` and `ROADMAP.md` to note that the pre-existing repo-owned optimizer/scheduler files were audited and already participate in the unified optimization layer.
- **State-storage dtype normalization is now shared across the absorbed offload-aware optimizers** — The repo no longer keeps duplicate string-to-dtype normalization helpers in each optimizer file that stages state onto a configurable storage dtype.
  - Added `library/optimization/optimizers/utils/state.py` with a shared `resolve_state_storage_dtype(...)` helper and re-exported it through `library/optimization/optimizers/utils/__init__.py`.
  - Updated `bcos.py`, `oagopt.py`, `ocgopt.py`, and `projective_adam.py` to use the shared helper instead of carrying local copies.
- **The common FFT low-pass gradient helper is now shared where the implementations were already aligned** — Several of the orthogonalized/spectral optimizer files no longer each carry the same `filter_grad(...)` body locally.
  - Added `library/optimization/optimizers/utils/frequency.py` with a shared `filter_grad(...)` helper and re-exported it through `library/optimization/optimizers/utils/__init__.py`.
  - Updated `fftdescent.py]`, `oagopt.py`, `ocgopt.py`, `scgopt.py`, `abmog.py`, and `singstate.py` to use the shared helper, while leaving `TALON` on its local FFT variant because its normalization path differs.
- **Windows torch.compile optimizer helpers now bootstrap partial MSVC/Triton envs instead of dropping or crashing the compiled path** — The compiled spectral/orthogonal helper paths now complete missing Visual Studio / Windows SDK env vars before Triton probes them, so Windows shells with a real toolchain but incomplete env hydration can still compile successfully.
  - Added `library/optimization/optimizers/utils/compile_env.py` with shared Windows compiler-environment bootstrap helpers that fill in missing `VCToolsVersion`, `WindowsSDKVersion`, `WindowsSDKVer`, and `CC` values from the installed MSVC / Windows SDK layout.
  - Updated `abmog.py`, `singstate.py`, and `talon.py` to use the shared compiled-helper availability check instead of carrying their own Windows-sensitive detection.
  - Updated `fftdescent.py`, `oagopt.py`, `ocgopt.py`, `scgopt.py`, `projective_adam.py`, and `wiwiopt.py` so their compile-enabled helper paths bootstrap the Windows toolchain environment before the first `torch.compile` invocation.

## [2026-04-09]

### Added

- **The absorbed plain-optimizer path now covers BCOS too** — The repo can now host the donor BCOS optimizer without routing through the vendor package at runtime.
  - Added `library/optimization/optimizers/bcos.py` with a repo-owned `BCOS` implementation adapted from the donor source while switching stochastic-copy usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `BCOS` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `BCOS` construction plus first-step CPU/offloaded-state initialization through the shared factory path.
- **The absorbed plain-optimizer path now covers ProjectiveAdam too** — The repo can now host the donor projection-based Adam variant without routing through the vendor package at runtime.
  - Added `library/optimization/optimizers/projective_adam.py` with a repo-owned `ProjectiveAdam` implementation adapted from the donor source while switching stochastic-copy usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `ProjectiveAdam` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `ProjectiveAdam` construction plus first-step projection / NorMuon state initialization through the shared factory path.
- **The absorbed plain-optimizer path now covers WiwiOpt too** — The repo can now host the donor WiwiOpt optimizer without routing through the vendor package at runtime.
  - Added `library/optimization/optimizers/wiwiopt.py` with a repo-owned `WiwiOpt` implementation adapted from the donor source while switching stochastic-copy usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `WiwiOpt` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `WiwiOpt` construction plus first-step dynamic-LR / Oja / NorMuon state initialization through the shared factory path.
- **The absorbed plain-optimizer path now covers OAGOpt, OCGOpt, and SCGOpt too** — The repo can now host the donor orthogonalized/centralized variants without routing through the vendor package at runtime.
  - Added `library/optimization/optimizers/oagopt.py`, `library/optimization/optimizers/ocgopt.py`, and `library/optimization/optimizers/scgopt.py` with repo-owned implementations adapted from the donor sources while switching stochastic-copy usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `OAGOpt`, `OCGOpt`, and `SCGOpt` participate in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `OAGOpt`, `OCGOpt`, and `SCGOpt` construction plus first-step state initialization through the shared factory path.
- **The absorbed plain-optimizer path now covers FFTDescent and the plain FishMonger variant too** — The repo can now host two more donor optimizers without routing through the vendor package at runtime.
  - Added `library/optimization/optimizers/fftdescent.py` and `library/optimization/optimizers/fishmonger.py` with repo-owned implementations adapted from the donor sources while switching shared helper usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `FFTDescent` and `FishMonger` participate in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `FFTDescent` and `FishMonger` construction plus first-step state initialization through the shared factory path.
- **The FishMonger absorption pass now covers the vendored 8-bit sibling too** — The repo no longer leaves the bitsandbytes-backed FishMonger variant stranded behind the donor file.
  - Updated `library/optimization/optimizers/fishmonger.py` to add the repo-owned `FishMonger8BitBNB` implementation adapted from the vendored backend source while switching helper usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `FishMonger8BitBNB` participates in the shared optimizer registration path with bitsandbytes-backed loading semantics.
  - Added focused regression coverage for `FishMonger8BitBNB` construction plus first-step quantized-state initialization through the shared factory path.
- **The absorbed Compass family now covers the vendored AO and bitsandbytes siblings too** — The repo no longer leaves the heaviest remaining `compass.py` variants stranded behind the donor file.
  - Updated `library/optimization/optimizers/compass.py` to add repo-owned `Compass8BitBNB`, `_CompassBase`, `single_param_compass(...)`, and `CompassAO` implementations adapted from the vendored backend source while switching helper usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `Compass8BitBNB` and `CompassAO` participate in the shared optimizer registration path with bitsandbytes / torchao backend metadata.
  - Added focused regression coverage for `Compass8BitBNB` construction/runtime support handling plus `CompassAO` parameter-precision and quantized-state initialization through the shared factory path.
- **The absorbed standalone optimizer pass now covers ABMOG, SingState, and TALON too** — The repo can now host three more donor optimizers directly without routing back through the vendor package.
  - Added `abmog.py`, `singstate.py`, and `talon.py` with repo-owned implementations adapted from the authoritative vendor tree while keeping comments/docstrings and switching shared helper usage onto the repo-owned optimization utilities where appropriate.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `ABMOG`, `SingState`, and `TALON` participate in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `ABMOG`, `SingState`, and `TALON` construction plus first-step state initialization through the shared factory path.
- **The absorbed standalone optimizer pass now covers Glyph too** — The repo can now host the donor Glyph optimizer directly without routing back through the vendor package.
  - Added `glyph.py` with a repo-owned `Glyph` implementation adapted from the authoritative vendor tree while reusing the repo-owned Newton-Schulz helper and stochastic-copy utility.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `Glyph` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `Glyph` construction plus first-step EMA / squared-EMA / previous-gradient state initialization through the shared factory path.
- **The absorbed standalone optimizer pass now covers FARMSCrop and FARMSCropV2 too** — The repo can now host the donor FARMSCrop pair directly without routing back through the vendor package.
  - Added `farmscrop.py` and `farmscrop_v2.py` with repo-owned implementations adapted from the authoritative vendor tree while switching helper usage onto the repo-owned adaptive-epsilon and stochastic-copy utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `FARMSCrop` and `FARMSCropV2` participate in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `FARMSCrop` and `FARMSCropV2` construction plus first-step FIM / momentum / diff-history state initialization through the shared factory path.
- **The absorbed FMARS family pass now covers the plain FMARSCrop pair too** — The repo can now host the vendor file’s plain `FMARSCrop` and `FMARSCropV2` paths directly without routing back through the donor package.
  - Added `fmarscrop.py` and `fmarscrop_v2.py` with repo-owned implementations adapted from the authoritative vendor tree while switching helper usage onto the repo-owned adaptive-epsilon, AGC, and stochastic-copy utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `FMARSCrop` and `FMARSCropV2` participate in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `FMARSCrop` and `FMARSCropV2` construction plus first-step MARS/FIM/momentum state initialization through the shared factory path.
- **The heavier FMARS family follow-up now covers FMARSCropV2ExMachina too** — The repo no longer leaves that donor variant stranded behind the vendor file while the rest of the plain FMARS pair is repo-owned.
  - Added `fmarscrop_v2_exmachina.py` with a repo-owned `FMARSCropV2ExMachina` implementation adapted from the authoritative vendor tree while preserving the fuller donor docstring and key inline algorithm comments.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `FMARSCropV2ExMachina` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `FMARSCropV2ExMachina` construction plus first-step update-strategy / diff-history state initialization through the shared factory path.
- **The remaining FMARS family pass now covers FMARSCropV3 and FMARSCropV3ExMachina too** — The repo no longer leaves the final donor `fmarscrop.py` variants stranded behind the vendor file.
  - Added `fmarscrop_v3.py` and `fmarscrop_v3_exmachina.py` with repo-owned implementations adapted from the authoritative vendor tree while preserving the fuller donor docstrings and key inline algorithm comments.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `FMARSCropV3` and `FMARSCropV3ExMachina` participate in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `FMARSCropV3` and `FMARSCropV3ExMachina` construction plus first-step FIM / momentum / diff-history state initialization through the shared factory path.

### Changed

- **The BCOS absorption pass also hardens the donor offload path while keeping the algorithm shape intact** — The repo-owned copy no longer assumes CUDA is always available for CPU parameters or accidentally drops back to the stored CPU state tensors mid-step.
  - Updated `library/optimization/optimizers/bcos.py` so CPU-parameter execution falls back to CPU compute cleanly when CUDA is unavailable, while still preserving staged CUDA execution when it exists.
  - Updated `library/optimization/optimizers/bcos.py` so staged momentum/variance tensors stay on the active compute device until they are explicitly synchronized back to the configured state-storage device.
- **The ProjectiveAdam absorption pass also hardens the donor offload path while keeping the algorithm shape intact** — The repo-owned copy no longer assumes CUDA is always available for CPU parameters and no longer routes float32 state writes through the stochastic-copy path by mistake.
  - Updated `library/optimization/optimizers/projective_adam.py` so CPU-parameter execution falls back to CPU compute/sync behavior cleanly when CUDA is unavailable, while still preserving staged CUDA execution when it exists.
  - Updated `library/optimization/optimizers/projective_adam.py` so float32 state buffers copy back directly instead of always passing through the donor’s stochastic-rounding write path.
- **The WiwiOpt absorption pass also hardens a donor scalar/state edge while keeping the algorithm shape intact** — The repo-owned copy no longer assumes every parameter tensor has at least one dimension when it builds row-wise summary state.
  - Updated `library/optimization/optimizers/wiwiopt.py` so summary, RMS, and norm helper paths handle scalar parameters cleanly instead of assuming `dim=-1` is always valid.
- **The Compass backend absorption pass also hardens low-bit runtime behavior while keeping the donor family together in one module** — The repo-owned copy now fails fast on unsupported backend combinations instead of falling through to backend-specific crashes.
  - Updated `library/optimization/optimizers/compass.py` so `Compass8BitBNB` explicitly rejects CPU execution and invalid bitsandbytes block sizes before touching blockwise quantization.
  - Updated `library/optimization/optimizers/compass.py` so `CompassAO` keeps the full family in the same module as `Compass` while rejecting quantized-state requests for CPU parameters with a clear runtime error.
- **The new standalone absorption batch also hardens compiled spectral-helper selection for CPU-first environments** — The repo-owned copies no longer assume the donor `torch.compile` fast path is always usable when `nvcc` is unavailable or not executable.
  - Updated `library/optimization/optimizers/abmog.py`, `singstate.py`, and `talon.py` so compiled spectral helpers are only selected when CUDA and an executable `nvcc` are actually available; otherwise they fall back to the plain helper implementation.
  - Updated `abmog.py` so CPU-parameter execution no longer assumes a CUDA compute device exists before staging state/offload work.

## [2026-04-05]

### Added

- **The absorbed plain-optimizer path now covers SCORN too** — The repo can now host another orthogonalized/focus-style optimizer without routing through the donor package at runtime.
  - Added `library/optimization/optimizers/scorn.py` with a repo-owned `SCORN` implementation adapted from the vendor source while switching shared helper usage onto the repo-owned optimization utilities.
  - Added the repo-owned `orthograd_atan(...)` helper under `library/optimization/optimizers/utils/orthograd.py` and re-exported it through `library/optimization/optimizers/utils/__init__.py` so the optimizer no longer needs to reach back into donor utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `SCORN` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `SCORN` construction plus first-step focus/reset bookkeeping through the shared factory path.
- **The absorbed plain-optimizer path now covers SCORNMachina too** — The repo can now host the heavier offloaded-state SCORN variant without routing through the donor package at runtime.
  - Added `library/optimization/optimizers/scornmachina.py` with a repo-owned `SCORNMachina` implementation adapted from the vendor source while switching shared helper usage onto the repo-owned optimization utilities.
  - Added `library/optimization/optimizers/utils/adagc.py` with repo-owned AdaGC helper functions used by the `SCORNMachina` path, and re-exported them through `library/optimization/optimizers/utils/__init__.py`.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `SCORNMachina` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `SCORNMachina` construction plus first-step offloaded-state / AdaGC bookkeeping through the shared factory path.
- **The absorbed plain-optimizer path now covers CAME too** — The repo can now host the confidence-guided memory-efficient optimizer without routing through the donor package at runtime.
  - Added `library/optimization/optimizers/came.py` with a repo-owned `CAME` implementation adapted from the vendor source while switching shared helper usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `CAME` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `CAME` construction plus first-step factored-state / AMSBound initialization through the shared factory path.
- **The absorbed plain-optimizer path now covers CStableAdamW too** — The repo can now host the donor stable-AdamW variant with optional Stable-SPAM clipping and ADOPT behavior without routing through the donor package at runtime.
  - Added `library/optimization/optimizers/cstableadamw.py` with a repo-owned `CStableAdamW` implementation adapted from the vendor source while switching Stable-SPAM helper usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `CStableAdamW` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `CStableAdamW` construction plus first-step ADOPT / Stable-SPAM state initialization through the shared factory path.
- **The absorbed plain-optimizer path now covers GrokFastAdamW too** — The repo can now host the donor grokfast optimizer without routing through the donor package at runtime.
  - Added `library/optimization/optimizers/grokfast.py` with repo-owned `GrokFastAdamW`, `gradfilter_ma(...)`, and `gradfilter_ema(...)` implementations adapted from the vendor source while switching stochastic-copy usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `GrokFastAdamW` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `GrokFastAdamW` construction plus post-warmup grok-EMA state tracking through the shared factory path.

### Changed

- **The SCORN absorption pass also hardens two donor crash edges while keeping the algorithm shape intact** — The repo-owned copy no longer carries over a pair of brittle donor assumptions around initialization and norm setup.
  - Updated `library/optimization/optimizers/scorn.py` so `init()` tolerates the donor’s missing `scale` group key instead of crashing when the method is exercised.
  - Updated `library/optimization/optimizers/scorn.py` so the LMO norm helper is always built before use, instead of only on the `spectral_update_scale > 0` branch.
- **The SCORNMachina absorption pass also hardens CPU-only/offloaded execution edges while keeping the algorithm shape intact** — The repo-owned copy no longer assumes CUDA is always available when it queues compute-device work for CPU parameters.
  - Updated `library/optimization/optimizers/scornmachina.py` so offloaded-state execution falls back to CPU compute/sync behavior cleanly when CUDA is unavailable, while still preserving the asynchronous path when CUDA exists.
- **The CAME absorption pass also hardens CPU-only/offloaded execution edges while keeping the algorithm shape intact** — The repo-owned copy no longer assumes CPU parameters can always stage work onto a CUDA device when using offloaded optimizer state.
  - Updated `library/optimization/optimizers/came.py` so the offloaded-state path falls back to CPU compute/sync behavior cleanly when CUDA is unavailable, while preserving the asynchronous path when CUDA exists.
- **The GrokFastAdamW absorption pass also fixes a donor denominator typo while keeping the algorithm shape intact** — The repo-owned copy no longer carries the donor’s invalid tensor API call in the Adam denominator path.
  - Updated `library/optimization/optimizers/grokfast.py` so the denominator adds `eps` correctly instead of calling `Tensor.add_(min=...)`, which would raise at runtime.

## [2026-04-04]

### Added

- **Adapter-layer direction now has a dedicated design note** — The repo now has a written proposal for how the adapter surface should evolve without freezing the public architecture around the current Kohya/LyCORIS-shaped runtime seam.
  - Added `docs_design/adapter_layer_direction.md` covering the current implicit adapter contract, the desired trainer/runtime/target/backend split, a power-user-friendly config direction, and a staged plan for absorbing LyCORIS behind a repo-owned facade.
- **The absorbed plain-optimizer path now covers SCION too** — The repo can now host the norm-constrained LMO optimizer without routing through the donor package at runtime.
  - Added `library/optimization/optimizers/scion.py` with a repo-owned `SCION` implementation adapted from the vendor source while switching its helper imports onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `SCION` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `SCION` construction plus manual initialization / first-step state handling through the shared factory path.
- **The absorbed plain-optimizer path now covers StableSPAM too** — The repo can now host the stable-spam optimizer without routing through the donor package at runtime.
  - Added `library/optimization/optimizers/spam.py` with a repo-owned `StableSPAM` implementation adapted from the vendor source while switching its helper imports onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `StableSPAM` participates in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for `StableSPAM` construction and the float32 projection-buffer reset path when `update_proj_gap` triggers through the shared factory path.
- **The absorbed plain-optimizer path now covers three more self-contained sign/muon variants too** — The repo can now host another small cluster of donor optimizers without routing through the vendor package at runtime.
  - Added `library/optimization/optimizers/dehaze.py`, `library/optimization/optimizers/gooddog.py`, and `library/optimization/optimizers/mythical.py` with repo-owned implementations adapted from the vendor sources while switching shared helper usage onto the repo-owned optimization utilities.
  - Updated `library/optimization/optimizers/__init__.py` and `library/optimization/registry.py` so `Dehaze`, `GOODDOG`, and `Mythical` participate in the shared repo-owned optimizer registration path.
  - Added focused regression coverage for construction plus first-step state initialization of all three optimizers through the shared factory path.
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

- **The optimizer absorption queue docs now reflect the post-SCION backlog** — The local optimizer README no longer points at SCION as the next untouched serious target after it moved onto the repo-owned path.
  - Updated `library/optimization/optimizers/README.md` counts and remaining-target notes after absorbing `scion.py`.
- **The StableSPAM absorption pass also hardens one donor state-reset edge case while keeping the algorithm shape intact** — The repo-owned copy now preserves the intended projection-buffer reset behavior for normal float32 params instead of only updating the replacement tensors on the bf16 copy-back path.
  - Updated `library/optimization/optimizers/spam.py` so `update_proj_gap` resets write the replacement `exp_avg` and `exp_avg_sq` tensors back into optimizer state before continuing the step.
- **The Dehaze absorption pass also hardens one donor adaptive-muon edge case while keeping the algorithm shape intact** — The repo-owned copy now initializes and reuses the normalized-gradient tensor instead of referencing an undefined local inside the adaptive-muon branch.
  - Updated `library/optimization/optimizers/dehaze.py` so adaptive-muon normalization always starts from the current gradient and writes the normalized result back before the denominator stages.
- **Scheduler orchestration now fails fast for `Ranger21` double-scheduling conflicts** — Runs using Ranger21’s built-in LR schedule no longer quietly stack a second external scheduler on top unless the config explicitly disables the optimizer-local scheduler.
  - Updated `library/optimization/scheduler.py` so `Ranger21` only accepts the effectively no-op external scheduler case (`lr_scheduler='constant'` with no warmup) while its internal LR schedule is active.
  - The shared scheduler path still allows normal external schedulers for `Ranger21` when `disable_lr_scheduler=True`.
- **The first Compass-family absorption pass also hardens a few donor edge cases while keeping the algorithm structure intact** — The repo-owned copy now avoids a couple of brittle donor assumptions instead of preserving them as hidden footguns.
  - Fixed the plain `Compass` step path so `weight_decouple` stays a boolean instead of accidentally becoming a one-tuple.
  - Updated `CompassPlus` to re-read per-group `betas` during its phase-3 update pass and to initialize diff-amp reset state without assuming gradients already exist during `reset()`.
  - Updated `CompassADOPT` and `CompassADOPTMARS` reset paths to derive factored-state shapes from parameter shapes instead of assuming pre-existing gradients.
- **Internal contributor docs now describe the active launcher/config architecture more accurately** — The repo docs no longer describe the old script-per-mode setup as the current design.
  - Updated `AGENTS.md` project structure, config-system notes, and common task examples around `train.py`, `RunConfig`, and the current `configs/` layout.
  - Updated `DEVELOPMENT_GUIDE.md` launcher, type-checking, config-grouping, config-passing, and testing guidance to match the active schema-driven architecture.
  - Pruned `ROADMAP.md` so completed architecture notes stay in the changelog while the roadmap focuses on active follow-up work.

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
