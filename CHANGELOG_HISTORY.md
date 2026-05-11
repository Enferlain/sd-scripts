# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2026-04-29]

### Changed

- **The active adapter training surface is now named and wired as `adapter` instead of the old root-level PEFT mode/config surface** — The canonical launcher/config path now uses `mode: adapter` plus `adapter.peft.<method>`, while the remaining `peft` concepts stay scoped to the current adapter family instead of acting as the top-level training mode.
  - Added `AdapterConfig`, renamed the active training-mode implementation to `AdapterMode`, and updated the mode factory / trainer wiring to resolve the adapter path as the current non-finetune training mode.
  - Moved the active config surface from root `peft` to `adapter.peft`, switched PEFT method selection to branch presence, and updated validation/compatibility shims so old root-level PEFT config now fails with migration-oriented errors instead of silently remaining the primary surface.
  - Updated active presets, smoke configs, benchmark configs, focused tests, and current-facing docs so the repo now describes the adapter shell as the current training/runtime concept and PEFT as the currently implemented adapter family beneath it.

- **Benchmark run results now report the active adapter-shaped config surface instead of the old PEFT root shape** — The markdown/JSON run report path now reflects the current adapter config keys and stays aligned with the resource-monitor output used by the benchmark/reporting flow.
  - Updated `library/logging/run_report.py` so key-config summaries now report `adapter.peft.lora` / `adapter.peft.loha` rather than the removed root `peft.method` field.
  - Updated focused run-report coverage so benchmark report fixtures compose under `mode: adapter` with nested `adapter.peft` config and continue validating the resource-monitor-backed markdown/JSON output.

## [2026-04-26]

### Changed

- **Optimization and adapter targeting now share one repo-owned target-provenance model across component, module, and parameter selection** — The active fine-tune and PEFT paths still keep their different lifecycle sequencing, but they now carry the same selector-stable target identity and richer provenance through optimization-owned refs instead of splitting long-term target vocabulary between grouping code and adapter runtime internals.
  - Added `library/optimization/targets.py` with shared component/module/parameter target refs, helper constructors, canonical component-qualified selectors, module-type capture, and owner-module provenance for parameter targets.
  - Updated `library/optimization/grouping.py` so fine-tune parameter collection now builds shared parameter target refs while preserving existing selector strings and grouping behavior.
  - Updated `library/adapters/runtime/targets.py` so adapter-facing resolved targets wrap shared optimization target refs while keeping compatibility fields like `component`, `component_key`, `path`, and `module` available to current runtimes.
  - Updated repo-owned adapter trainable refs so LoHa now preserves real source module-target provenance, while the legacy built-in LoRA wrapper keeps that provenance explicitly unset where exact module identity cannot be recovered from the compatibility runtime.
  - Added focused unit coverage for target-ref construction, owner-module provenance, selector stability, adapter module-type provenance, and the unchanged fine-tune / adapter grouping behavior, including PEFT mode and parameter-dump helper verification.

## [2026-04-23]

### Changed

- **Adapter method config ownership now lives in method registration instead of a transitional bridge module** — The active PEFT config translation path now treats method registration as the source of truth for method-local config identity and runtime-settings translation, so adding a new repo-owned method no longer requires updating hidden parallel config maps beside the registry.
  - Added repo-owned method-local config translators under `library/adapters/methods/peft/lora/config.py` and `library/adapters/methods/peft/loha/config.py`.
  - Extended `AdapterMethodRegistration` with method-config ownership metadata and replaced `library/adapters/config_bridge.py` with `library/adapters/method_configs.py`, which now builds runtime specs and validation views from registration-owned config bindings instead of `_METHOD_CONFIG_NAMES` / `_METHOD_CONFIG_TYPES` tables.
  - Updated PEFT mode, training metadata, validation, and runtime-registry tests to use the registration-owned method-config system.

- **The forward PEFT config surface is now intent-shaped instead of loader-shaped** — Active adapter config now centers on `peft.method`, method-local config subtrees, and explicit continuation intent, while old loader-mechanics fields remain only as a thin normalization shim during the transition.
  - Added typed repo-owned `PeftLohaConfig` plus forward `peft.continue_from` / `peft.continue_mode` config fields, and moved `training_comment` into generic output metadata ownership.
  - Added adapter config translation helpers so `PeftMode` now resolves the selected method, normalizes method-local settings into `AdapterRuntimeSpec`, and chooses strict continuation vs `initialize_from_artifact` flow without passing raw config bags into adapter runtimes.
  - Updated centralized config validation to reject ambiguous active/inactive method subtrees and to fail fast when strict continuation is combined with config-defined method settings.
  - Updated active smoke/benchmark/example configs plus focused config/mode/metadata tests to use the new forward PEFT surface by default while keeping the old fields only as compatibility shims.

- **`loha` is now a fully repo-owned method implementation instead of a repo-owned runtime around vendored LyCORIS algorithm code** — The `peft/loha` method folder now owns the LoHa module behavior, state-dict reconstruction, and export/merge handling in repo code, while the runtime layer remains a thin orchestrator over the repo-owned implementation.
  - Added repo-owned `LohaModule` and state-dict helpers under `library/adapters/methods/peft/loha/` and removed the runtime dependency on vendored `library.vendor.lycoris.lycoris.modules.loha.LohaModule`.
  - Updated the `loha` runtime to build, load, save, and merge through the repo-owned method implementation instead of vendored module-class helpers.
  - Extended focused runtime-registry coverage so the `loha` adapter tests now assert that built modules come from `library.adapters.methods.peft.loha.module` rather than a vendored LyCORIS runtime class.
  - Added a repo-owned `LohaConfig` construction path plus direct `LohaModule` unit coverage for rank validation, zero-diff behavior, merged-weight vs forward consistency, and Tucker export/reconstruction round-trips.

## [2026-04-22]

### Changed

- **Adapter training now has an optimization-owned module-targeting path plus a first repo-native absorbed `loha` runtime** — PEFT adapter runs now resolve concrete module targets before runtime construction, and the new `loha` method plugs into the same repo-owned build, trainable-ref, loaded-runtime merge, and export seams instead of depending on the vendor LyCORIS wrapper as the repo contract.
  - Added module-resolved adapter target selection in `library/adapters/runtime/targets.py` and `library/optimization/grouping.py`, including stable `component_key` plus component-qualified target paths for downstream adapter runtimes.
  - Added the repo-owned `library/adapters/methods/peft/loha/runtime.py` path and registered `loha` in `library/adapters/registry.py`, while keeping grouping parameter-native through repo-owned `AdapterTrainableParameterRef` exposure.
  - Added focused coverage in `tests/unit/adapters/test_runtime_registry.py`, `tests/unit/training/modes/test_peft_mode.py`, and `tests/unit/training/test_training_optimizer.py` for module-target resolution, `loha` runtime registration, trainable-ref provenance, export/load round-trips, loaded-runtime merge behavior, and PEFT mode orchestration.

- **The built-in PEFT method implementations now live with their method packages instead of at the flat adapter root** — The actual built-in LoRA, DyLoRA, and OFT implementation code now lives under `library/adapters/methods/peft/.../`, while the old `library/adapters/lora.py`, `dylora.py`, and `oft.py` files are reduced to compatibility shims so existing config strings, tools, and tests keep working during the broader cleanup.

## [2026-04-21]

### Changed

- **The adapter-system rework now explicitly validates and documents that optimization still owns targeting/grouping across the corrected loaded-runtime merge path** — The final migration slice now proves that from-weights adapter construction still consumes optimization-owned resolved targets, grouping still requires the repo-owned trainable-ref contract, and the written design/docs coverage matches the runtime-layer loaded-runtime plus merge-request boundary.
  - Added focused validation coverage in `tests/unit/training/modes/test_peft_mode.py` for the `adapter_rank_from_weights` path using the same optimization-owned resolved targets as the main adapter build.
  - Added focused validation coverage in `tests/unit/training/test_training_optimizer.py` to prove adapter grouping still requires the repo-owned `describe_trainable_parameter_refs()` contract instead of falling back to adapter-method internals.
  - Updated `openspec/changes/adapter-system-rework/design.md`, `docs_design/adapter_system_overview.md`, and `ROADMAP.md` so the current architecture is described in repo-owned loaded-runtime / merge-request terms rather than the earlier transitional shared-merge wording.

## [2026-04-20]

### Changed

- **The built-in adapter path now uses the same repo-owned loaded-runtime and merge seams across training, from-weights, and base-weight merge flows** — The remaining built-in LoRA-shaped path no longer makes `PeftMode` call built-in merge signatures directly, and base-weight merge now rides the same optimization-owned resolved-target handoff used by the main adapter training build.
  - Updated `library/adapters/runtime/context.py` and `library/adapters/runtime/build.py` so from-weights construction normalizes onto a repo-owned `LoadedAdapterRuntime`, and merge now flows through a repo-owned `AdapterMergeRequest` owned by that loaded runtime.
  - Updated the built-in method wrappers under `library/adapters/methods/` so they translate the repo-owned loaded-runtime `merge_into(...)` capability to older built-in `merge_to(...)` signatures internally instead of exposing those signatures as the repo contract.
  - Updated `library/training/modes/peft_mode.py` so base-weight merge uses the same optimization-owned resolved targets as the main training build, while `adapter_rank_from_weights` and merge both consume the normalized loaded-runtime result.
  - Added focused coverage in `tests/unit/adapters/test_runtime_registry.py` and `tests/unit/training/modes/test_peft_mode.py` for the new loaded-runtime normalization, merge seam delegation, invalid wrapper results, and the base-weight merge handoff.

- **The adapter-system rework now has an explicit persistence split between training checkpoint state and adapter export-style save/load** — `PeftMode` remains the training-side orchestration owner for adapter persistence flows, while adapter runtime objects now participate through repo-owned helpers that distinguish accelerator checkpoint state from adapter-format export/load operations.
  - Expanded `library/adapters/shared/state_io.py` into a repo-owned adapter persistence seam with explicit export request objects plus a dedicated training-checkpoint hook helper, and re-exported that seam from `library/adapters/shared/__init__.py` and `library/adapters/__init__.py`.
  - Updated `library/training/modes/peft_mode.py` so adapter weight loads, adapter-format checkpoint writes, and training checkpoint hook registration now route through the new adapter persistence helpers instead of open-coding the flow inside the mode.
  - Updated `library/training/checkpointing.py` so the deprecated `register_adapter_state_hooks(...)` compatibility helper delegates to the new adapter-owned checkpoint-state seam instead of carrying a second copy of the filtering logic.
  - Added focused coverage in `tests/unit/adapters/test_state_io.py` and `tests/unit/training/modes/test_peft_mode.py` for export-style adapter persistence delegation, adapter-only checkpoint hook registration, and the explicit `PeftMode` handoff to the repo-owned persistence seam.

## [2026-04-19]

### Changed

- **The adapter-system config rework now has its first real method-specific surface under the existing PEFT shell** — The active adapter path now treats nested `peft.lora` config as the live LoRA authority without requiring a top-level mode rename yet, while the top-level `peft` section stays focused on PEFT-shell orchestration concerns instead of carrying a second flat LoRA schema.
  - Added `PeftLoraConfig` in `library/config/dataclasses/peft.py` as the single LoRA method-config authority under the existing PEFT shell, while keeping only true PEFT-shell/orchestration fields flat at the top level.
  - Updated `configs/_defaults/peft/default.yaml` plus the active smoke/example/benchmark PEFT configs so LoRA settings now live under `peft.lora` instead of a second flat LoRA-shaped surface.
  - Updated `library/adapters/lora_utils.py`, `library/training/modes/peft_mode.py`, and `library/training/training_metadata.py` so the active LoRA path and PEFT metadata now read the nested method surface directly, while unsupported legacy optimizer-policy knobs remain explicit under `peft.lora` and still fail fast.
  - Added focused coverage in `tests/unit/test_configs.py`, `tests/unit/training/modes/test_peft_mode.py`, and `tests/test_sd_peft_config.py` for the new nested `peft.lora` surface, flat-field compatibility fallback, and mode-side precedence behavior.

- **PEFT optimizer setup now consumes a repo-owned adapter trainable-ref contract and optimization-owned grouping plan instead of adapter-owned optimizer hooks** — The active adapter training path no longer relies on compatibility-era `prepare_optimizer_params(...)` as its optimizer boundary, and `PeftMode` now builds a typed optimization plan like the fine-tune path.
  - Added `AdapterTrainableParameterRef` and provider helpers under `library/adapters/shared/` so adapter runtimes can expose trainable parameter provenance through a repo-owned contract.
  - Updated the built-in adapter runtime wrappers under `library/adapters/methods/` to attach repo-owned trainable-ref describers that only expose trainable identity plus original-target provenance, without smuggling built-in LoRA-specific LR math or grouping hints through the runtime boundary.
  - Updated `library/optimization/grouping.py` and `library/training/modes/peft_mode.py` so adapter grouping is resolved in optimization-owned code, while legacy built-in optimizer-policy knobs that are not yet real repo concepts now fail fast instead of leaking through the adapter runtime handoff.
  - Added focused coverage in `tests/unit/adapters/test_runtime_registry.py`, `tests/unit/training/modes/test_peft_mode.py`, and `tests/unit/training/test_training_optimizer.py` for the new adapter trainable-ref contract and optimization-owned PEFT grouping behavior.

## [2026-04-18]

### Changed

- **The first repo-owned adapter runtime seam now consumes an explicit build request plus optimization-owned resolved original-model targets** — `PeftMode` no longer creates built-in adapters by calling raw compatibility-era positional constructors directly, and the migration path now carries public component-root target provenance like `clip_l` / `clip_g` / `unet` into the runtime boundary without making the runtime filter the model context itself.
  - Updated `library/adapters/runtime/` with `AdapterBuildRequest`, `AdapterRuntimeSpec`, and public component-root target builders that the optimization layer now owns for the current PEFT migration slice.
  - Updated `library/optimization/grouping.py` with an optimization-owned `resolve_adapter_target_selection(...)` helper so adapter target policy is resolved outside `PeftMode` before adapter instantiation.
  - Updated the built-in method wrappers under `library/adapters/methods/` and `library/training/modes/peft_mode.py` so adapter construction routes through the repo-owned runtime request path while `PeftMode` keeps training-side ownership and base-weight merge behavior.
  - Updated the current built-in adapter implementations to tolerate unselected components during migration, while the runtime wrappers now preserve the full model context and attach resolved-target provenance onto the constructed adapter instead of stripping untargeted components out during construction.
  - Added focused coverage in `tests/unit/adapters/test_runtime_registry.py`, `tests/unit/training/modes/test_peft_mode.py`, and `tests/unit/training/test_training_optimizer.py` for component-root target construction, optimization-owned PEFT target resolution, the real runtime-wrapper boundary, and the new `PeftMode` build-request handoff.

- **The adapter-system rework now has its first repo-owned runtime scaffolding slice in place** — The repo can start moving built-in adapter training onto a new `library/adapters/` framework surface without touching LyCORIS yet or preserving the old PEFT module-import shape as the architectural center.
  - Added `library/adapters/registry.py`, `library/adapters/types.py`, `library/adapters/runtime/`, and `library/adapters/shared/` as the first package structure for the new adapter-system architecture.
  - Added built-in adapter-type wrapper packages under `library/adapters/methods/` for `lora`, `dylora`, and `oft`, with a lazy registry/build path that keeps the new runtime surface cheap to import.
  - Added focused coverage in `tests/unit/adapters/test_runtime_registry.py` for adapter-type discovery, legacy-module resolution, and the first repo-owned build entrypoints.

## [2026-04-17]

### Changed

- **Text-encoder training validation now treats explicit TE-targeting optimizer groups as real TE training, not just positive baseline TE LRs** — Configs can no longer sneak TE subset training past TE-cache/offload guards by setting `text_encoders: 0` and only using positive `clip_*` / `text_encoder*` groups.
  - Updated `library/config/config_validation.py` so TE caching/offload conflicts consider both baseline TE LRs and positive explicit groups that target known TE selector namespaces for the active model family, with clearer error text about removing TE-targeting groups when the TE path must stay frozen.
  - Updated `library/optimization/optimizer_utils.py` to keep the baseline TE trainability helpers/documentation aligned with the current `null`-means-inherit / `0`-means-frozen semantics.
  - Added focused regression coverage in `tests/unit/test_config_validation.py`, `tests/unit/training/modes/test_finetune_mode.py`, and `tests/unit/training/test_training_optimizer.py` for grouped TE validation conflicts and explicit zero-LR frozen-component behavior.

## [2026-04-16]

### Changed

- **Grouped optimizer construction now treats explicit group learning rates as the authoritative runtime source** — Fine-tune group-only configs no longer rely on `optimizer.learning_rates.base` reaching backend constructors when every execution group already defines its own `lr`, and the user-facing `null` versus `0` semantics are now documented more clearly.
  - Updated `library/optimization/optimizer_factory.py` so optimizer construction omits constructor `lr` when `base` is `null` but every materialized optimizer param group already defines an explicit `lr`, and now raises repo-owned `ValueError`s when `base: null` leaves any optimizer group without an explicit LR.
  - Updated `library/optimization/optimizer_utils.py`, `library/config/dataclasses/optimizer.py`, `library/config/config_validation.py`, and `configs/_defaults/optimizer/default.yaml` so helper docs, config help text, defaults comments, and validation guidance consistently describe `null` as inherit/no-fallback and `0` as a frozen baseline path.
  - Added focused regression coverage in `tests/unit/training/test_training_optimizer.py`, `tests/unit/test_config_validation.py`, and `tests/unit/training/modes/test_finetune_mode.py` for group-only optimizer construction, bitsandbytes grouped LR handling, repo-owned error reporting, and the clarified frozen-versus-inherited LR semantics.

- **Parameter selector names now use model-facing component prefixes across dumps and fine-grained optimizer matching** — The public selector surface is now `component.local_name` instead of the old training-internal placeholders like `denoiser.*` and `text_encoder1.*`.
  - Updated `library/models/parameter_dump.py` and `tools/model_management/dump_named_parameters.py` so parameter-oriented YAML emits component-qualified selector names such as `unet.*`, `clip_l.*`, and `mmdit.*` while still grouping entries under the same top-level component sections.
  - Updated `library/optimization/grouping.py` and `library/training/modes/finetune_mode.py` so inline/file-backed `optimizer.learning_rates.groups` patterns are matched against the same component-qualified selector names shown by the inspection dumps, while keeping training-internal component bookkeeping internal.
  - Removed the obsolete `optimizer.learning_rates.blocks` field from `library/config/dataclasses/optimizer.py`, `configs/_defaults/optimizer/default.yaml`, and stale validation comments/tests.
  - Added focused regression coverage in `tests/unit/models/test_parameter_dump.py`, `tests/unit/tools/test_dump_named_parameters.py`, `tests/unit/training/test_training_optimizer.py`, `tests/unit/training/modes/test_finetune_mode.py`, and `tests/unit/test_config_validation.py` for the normalized selector namespace and schema cleanup.

- **Execution-group metadata no longer serializes live duplicate parameter references through the generic optimizer payload** — The optimization plan now keeps parameter-name context as sidecar metadata instead of stuffing `(name, Parameter)` tuples into runtime param groups, avoiding the extra denoiser-sized allocation that `accelerate.prepare(optimizer)` could trigger when it device-moved arbitrary param-group state.
  - Updated `library/optimization/types.py` so `ParameterGroup` distinguishes runtime `options` from non-serialized `metadata`, and so `materialize_parameter_groups(...)` only includes explicitly requested metadata keys.
  - Updated `library/optimization/grouping.py` so fine-tune execution groups keep only string `param_names` metadata for optimizer-specific consumers instead of live `named_params` tuples.
  - Updated `library/optimization/optimizer_factory.py` and `library/optimization/optimizers/adammini.py` so `AdamMini` opts into the safe `param_names` metadata path, including fully qualified `AdamMini` config targets, while the generic runtime payload remains minimal.
  - Added focused regression coverage in `tests/unit/training/test_training_optimizer.py` for metadata-free generic materialization plus selective safe metadata inclusion.

## [2026-04-15]

### Changed

- **The model parameter dump tooling was rebuilt as a real model inspection tool** — `tools/model_management/dump_named_parameters.py` now uses the repo's existing strategy loading path to inspect real loaded runtime modules and emit deterministic YAML views for parameters, modules, or full state.
  - Replaced the old Hydra/config-driven training-pipeline surface with a direct inspection CLI: `--model-type`, `--model-path`, optional sidecars, and `--view parameters|modules|summary|state`.
  - Kept `library/models/parameter_dump.py` limited to the accepted shared surface: top-level component-name metadata plus pure YAML formatting helpers, without any dump-specific loader/package APIs.
  - Made the default output parameter-oriented and grouped under existing `NAMED_PARAMETER_COMPONENT_NAMES`, with module inspection and buffer/state inspection sharing the same component ordering.
  - Added a compact `summary` inspection view that prints a cheat-sheet mapping of `component -> composite module type -> useful child module types`, derived generically from the loaded runtime tree without dumping raw parameter names.
  - Hardened the SD3 Hugging Face text-encoder loading path so meta-initialized CLIP/T5 modules are materialized with `to_empty()` before state-dict assignment, avoiding the `Cannot copy out of meta tensor` failure that the inspection tool exposed on unified SD3 checkpoints without changing the underlying assign-based loading path.
  - Added focused unit coverage for tool-side loading orchestration, component grouping, and the parameter/module/state renderers.

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
