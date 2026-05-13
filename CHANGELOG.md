# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Rules:
- Use proper sub titles "Added", "Changed", "Removed" and "Fixed"
- Keep proper track of days for where entries should go
- Be concise but mention all changes without necessarily detailing each one

## [2026-05-13]

### Added

- **Benchmark report files now register as logging artifacts after they are written** — the training observer now records produced artifact paths with kind/format metadata, and trainer finalization registers both Markdown and JSON benchmark reports through that observer hook while keeping report generation itself local and backend-agnostic.
- **Training observers now receive explicit run lifecycle calls** — observer startup records the tracker run name and sanitized tracker config before metrics/artifacts are emitted, and trainer cleanup finishes the observer run after final report artifacts are registered so future logging sinks can bracket complete runs cleanly.
- **Benchmark report payload construction now has an explicit run-context boundary** — `write_run_report()` remains trainer-facing, while payload assembly now consumes a `RunReportContext` and declared non-adapter key-config policy instead of scattering direct trainer/config lookups through the report builder.

## [2026-05-12]

### Changed

- **Observability, optimizer grouping, and adapter target resolution now consume family-declared loaded components instead of reconstructing the old SD-shaped trio** — startup diagnostics and fine-tune mode now project labels/order from `trainer.loaded_components`, fine-tune selector/grouping paths now derive parameter-target provenance and public selector prefixes from declared components, and adapter runtime target expansion now resolves component identity from the same loaded-component surface through stricter loaded-components-only helper APIs.
  - Added focused coverage for declared-order diagnostics, selector-qualified fine-tune matching, adapter target resolution, and non-slot component shapes with custom keys/order so the migration is locked to the component contract rather than the legacy `text_encoderN`/`vae`/`denoiser` reconstruction path.
  - Fixed the follow-up regressions where adapter optimizer grouping still derived text-encoder LR policy from legacy `text_encoderN` key parsing alone and fine-tune gradient clipping still gathered params through hardcoded `denoiser` / `text_encoderN` buckets instead of the resolved component-key sets.
- **The model-layer README now documents the loaded-component contract conventions explicitly** — `library/models/README.md` now records that shared top-level component helpers live in `library/models/components.py`, family package `__init__.py` files own ordered `LOADED_MODEL_COMPONENT_SPECS` declarations, and generic runtime code should consume roles/capabilities instead of drifting back into family-name branches.

## [2026-05-11]

### Changed

- **The active strategy/trainer root contract now loads family-declared components instead of unpacking the old SD-shaped tuple** — `ModelLoadingStrategy.load_target_model()` now returns ordered `LoadedModelComponent` values, trainer setup stores `loaded_components` as the primary top-level model state, and the startup resource breakdown consumes that declared component surface directly instead of rebuilding it from `text_encoders`, `vae`, and `denoiser`.
  - Moved the shared component helpers into `library/models/components.py`, kept `library.models` as a thin public re-export surface, updated the parameter-dump tool to load/render top-level components through the new contract, and added focused contract/test coverage for the updated strategy, trainer, model-prep, and inspection paths.
- **The new family-declared loaded-component contract now has its initial repo-owned declaration seam** — `library.models` now exposes `LoadedModelComponentSpec` plus component-spec resolution, the current SD / SDXL / SD3 model-family packages declare ordered top-level components through that seam, and the legacy name-only helper is derived from the new declarations so the broader runtime migration can proceed incrementally from one source of truth.
  - Added focused model/declaration coverage proving the current families expose the expected ordered component specs and that the legacy component-name helper still derives the same public labels from those declarations.
- **Shared public model-component naming now lives under `library.models` instead of the parameter-dump module** — The repo-owned `NamedParameterComponentNames`, component-name resolution, selector-name construction, and loaded-component grouping helpers now sit on the model package surface, so startup summaries, optimizer grouping, adapter targeting, config validation, and the dump tool no longer depend on a file whose real purpose is YAML inspection formatting.
  - Updated the model-family package exports plus the affected runtime, logging, optimization, and dump-tool consumers to import the extracted seam from `library.models`, while keeping `library/models/parameter_dump.py` focused on dump rendering and preserving existing public component labels/selector behavior.

## [2026-05-10]

### Changed

- **Adapter trainable-ref validation now fails at the shared contract layer instead of only during reporting** — Missing `adapter_module_path` provenance is now rejected as soon as repo-owned adapter trainable refs are read, so malformed adapter runtimes fail closer to the source instead of surfacing later in startup reporting.
  - Centralized trainable-ref validation under `library/adapters/shared/trainables.py`, removed the reporting-only check from adapter component-row assembly, updated the remaining deprecated PEFT runtimes to populate `adapter_module_path`, and tightened the affected adapter/optimizer tests to match the stronger repo-owned contract.
- **Startup resource-table rendering now has a small defensive empty-row guard** — The width-calculation path in the startup resource breakdown now safely returns before `max(...)` if it ever encounters an empty table-row list, keeping the startup monitor logic robust against future caller or formatting changes.
- **Progress-safe lifecycle output now has a trainer-owned fallback seam instead of repeated loop-local branching** — The main training loop now routes epoch banners and prepared-epoch status lines through shared trainer helpers, so the console-vs-logger fallback policy lives in one place and the `log_external()` stacklevel behavior is documented more explicitly at the console transport boundary.
  - Added focused attribution coverage to prove that progress-safe lifecycle logs still resolve to the real caller `file:line` through the trainer-helper and console transport layers.
- **Startup resource rows now follow the same producer-owned component order as the `components` table** — The startup weight-residency block no longer shows a different component order than the main diagnostics table when both are rendered during bootup.
- **Logging callers now import the real observability module homes directly instead of going through compatibility wrappers** — Deprecated scripts and focused unit tests now import metrics/report helpers from `library.logging.metrics` and `library.logging.reports`, removing the need for the temporary `step_logging.py` and `run_report.py` shim modules.
- **Fine-tune full-model save-start messages now use the same tagged checkpoint logger style as the rest of training lifecycle output** — SDXL and SD3 full-model checkpointing no longer drop a bare `accelerator.print(...)` line for save-start status, keeping checkpoint lifecycle output visually consistent before the final `[checkpoint] checkpoint saved` confirmation.
- **Double-`Ctrl+C` interrupt handling is now documented at the code seam where it matters** — The shared interrupt guard now explains its warning-only first press, 5-second second-press window, possible signal-delivery delay during long C/CUDA work, and the fact that real cleanup still happens in the trainer’s unconditional `finally` path.

### Fixed

- Removed stray blank checkpoint log lines from SDXL/SD3 full-model checkpoint saves so closeout logging stays consistent with adapter saves.

## [2026-05-09]

### Added

- **Training interruption handling now has a shared double-`Ctrl+C` guard with a short retry window** — Active launcher runs can now ignore accidental first interrupts, warn clearly, and only raise `KeyboardInterrupt` when the second `Ctrl+C` arrives within a 5-second cooldown window.
  - Added `library/training/interrupts.py` plus focused launcher/trainer coverage for the warning-only first press, cooldown reset behavior, real second interrupt, and progress-bar cleanup on interrupted exits.

### Changed

- **Canonical lifecycle output now stays clean around the live progress bar without giving up the normal logger style** — Epoch banners, prepared-epoch status, and resource-monitor lines emitted during active training now use `tqdm` external-write mode so they do not collide with the live bar while still preserving timestamped `file:line` formatting.
  - The data-layer `prepare_epoch()` helper no longer owns lifecycle status emission; the training loop now logs the canonical `[epoch] prepared epoch ...` line itself so progress/UI output stays in orchestration code.
- **Trainer shutdown now tears down the live progress bar before final save/session logging** — Final checkpoint and resource-summary lines no longer render beside a stale `100%` bar at the end of a successful run.
  - `_finalize_training()` now closes and clears the active progress bar before final save/checkpoint work, `Trainer.train()` also performs unconditional crash/interrupt cleanup in `finally`, and the final save confirmation now reads `[checkpoint] checkpoint saved` to stay consistent with the tagged lifecycle style.
- **Startup resource memory now uses one shared loaded-weight view for both fine-tune and adapter runs** — The bootup resource block now reports loaded/trainable/frozen weight residency from the real loaded model components instead of reusing the adapter-centric diagnostics rows.
  - The startup resource block now appears under `Resource startup breakdown`, groups `loaded model weights` separately from estimated `training state`, renders the loaded-weight section as an aligned terminal-friendly table, prints the dense startup block directly instead of squeezing it through the multiline `INFO` logger gutter, and leaves a visual blank line between the main startup summary and the resource block.
- **Startup `components` now use the same table-oriented presentation style as the resource block** — Dense component diagnostics no longer repeat inline prose labels per row, making both fine-tune and adapter startup output easier to scan in a normal terminal width.
  - The `components` table keeps `params` left-aligned as a ratio field, uses `trainable` as the final status column, preserves the older inner ratio padding for values like `0/  99`, exposes the extra `adapter modules` column in adapter mode when present, and now preserves producer/source order instead of imposing alphabetical sorting in shared reporting code.

## [2026-05-07]

### Added

- **Training observability now has explicit module homes under `library/logging/`** — The repo now has first-pass `console.py`, `metrics.py`, `summaries.py`, and `reports.py` ownership buckets alongside the existing `resource_monitor.py`, with compatibility wrappers left in `step_logging.py` and `run_report.py` for existing imports.
  - Added shared startup-summary dataclasses/rendering, a repo-facing `TrainingObserver` seam plus a narrower backend `MetricsSink` contract, and a `MainProcessConsole` helper for canonical human-facing training output.

### Changed

- **Startup diagnostics now flow through one shared observability summary path instead of ad hoc `accelerator.print(...)` formatting** — Trainer startup now builds structured rows, renders sectioned startup blocks, and reuses the same summary facts for benchmark-report memory estimates.
  - The preferred console split is now a sectioned startup block for dense run/configuration output plus tagged lifecycle lines such as `[epoch] ...` and `[checkpoint] ...` for standalone status messages.
  - The startup summary block now renders directly through the repo console layer instead of as a multiline `INFO` log record, preserving the full terminal width for dense component rows while leaving normal one-line logs on the timestamped `file:line` logger path.
- **Adapter diagnostics now derive from repo-owned trainable-ref provenance and default to public component labels** — Adapter-mode startup and report breakdowns now group by stable internal component keys while showing public model-family labels like `unet`, `clip_l`, and `clip_g` in user-facing output.
  - Adapter-side reporting facts now live under `library/adapters/shared/reporting.py`, with repo-owned trainable refs carrying explicit adapter-module provenance so logging can consume both original component-module counts and adapter-module counts without owning PEFT-specific inference.
  - Resource startup memory estimates now accept structured diagnostic rows directly, so adapter and fine-tune paths can share the same reporting flow without rebuilding component stats separately.
  - Adapter runs now surface the active method inside the `training run` startup section as `method: ...`, sourced from `AdapterMode`/trainer state instead of a stray standalone print.
- **Canonical training lifecycle output now uses progress-bar-safe emission without falling back to raw text formatting** — The epoch banner, prepared-epoch status line, and resource-monitor summaries emitted while training is live now write through `tqdm` external-write mode so they no longer collide with the active progress bar.
  - The progress-safe path keeps normal logger formatting for those lines instead of introducing a separate raw-text transport, so timestamped `file:line` output remains intact where the logger already owns presentation.

## [2026-05-06]

### Changed

- **Repo-owned PEFT methods now share an autocast-first precision rule for their hot forward paths** — The common adapter wrappers now rely on a shared PEFT precision helper so normal mixed-precision training avoids unnecessary input/output dtype shuffling, while no-autocast mismatch handling stays explicit and narrow.
  - Applied the shared cast/restore helpers across repo-owned LoRA, VeRA, GLoRA, LoCon, DyLoRA, BOFT, OFT, IA3, and LoKr module paths, and added focused shared precision coverage for the new helper behavior under CPU autocast.

### Fixed

- **Repo-owned LoRA rank dropout now masks the actual rank axis for Linear outputs instead of assuming channel-first layout** — `LoraModule._apply_rank_dropout()` now applies rank dropout on the last axis for Linear activations such as `[batch, seq, rank]`, while preserving the existing channel-axis behavior for Conv outputs.
  - Added focused unit coverage for Linear-shaped rank-dropout broadcasting so sequence-shaped LoRA activations do not silently mask the wrong dimension.

## [2026-05-04]

### Added

- **The strategy layer now exposes scoped runtime facts through `StrategyContext` during denoiser forward** — Active diffusion families can now publish phase, model family, global step, timesteps, and selected sample indices for the duration of denoiser execution, giving lower library layers one shared read path instead of new per-method parameter threading.
  - Added `library/strategies/base/context.py` plus focused unit coverage for scoped publication, nested restoration, absence-safe reads, and read-only context records.
  - Added focused SD, SDXL, and SD3 strategy coverage proving denoiser-forward context is visible only during the intended train/validation forward scopes, including indexed prior-preservation passes.

### Changed

- **The denoiser strategy seam now carries explicit runtime facts without introducing a separate request/helper layer** — `DenoiserCallingStrategy` now defines the denoiser-forward runtime facts directly in its signature, while SD, SDXL, and SD3 denoiser facets publish `StrategyContext` locally inside their family-owned denoiser execution paths.
  - Migrated SD, SDXL, and SD3 diffusion/denoiser paths off the temporary request/helper shape, kept `contracts.py` definition-oriented, and threaded `global_step` through SD and SD3 validation denoiser paths so train/validation contexts stay aligned.

### Fixed

- **Repo-owned TLora now consumes published strategy timesteps through the PEFT runtime boundary instead of staying method-local-only** — The TLora runtime now reads `StrategyContext.denoiser.timesteps`, derives timestep masks per forward, applies them to TLora modules in scope, and clears the transient mask state after execution.
  - Added receiver-side registry/runtime coverage proving TLora output changes with published timesteps and that temporary mask state does not leak after the forward returns.

## [2026-05-03]

### Added

- **VeRA is now available as a repo-owned PEFT adapter method under `adapter.peft.vera`** — The active adapter runtime can now build, train, export, load, and merge a Hugging Face PEFT VeRA-style method through the same repo-owned resolved-target/runtime surface as the other modernized adapter methods.
  - Added repo-owned VeRA config/runtime/state-dict ownership under `library/adapters/methods/peft/vera/`, with a runtime-owned shared projection bank (`vera_A` / `vera_B`), per-target learned `vera_lambda_b` / `vera_lambda_d` vectors, repo-owned trainable-ref provenance, and loaded-runtime merge behavior.
  - Kept the repo-owned VeRA slice intentionally narrow and explicit while still covering the remaining method-local gaps: plain `nn.Linear` plus Transformers `Conv1D`-style linear wrappers, deterministic projection init via `projection_prng_key`, and metadata-backed `save_projection=False` reconstruction without adding empty sentinel state-dict keys.
  - Added focused module/runtime/config coverage for zero-delta initialization, merged-weight consistency, registry-owned runtime construction, repo-owned trainable refs, runtime export/load/merge round-trips, and centralized config validation for the new method branch.
- **LoRA is now a repo-owned PEFT runtime instead of the remaining legacy wrapper path** — The active adapter runtime can now build, train, export, load, and merge LoRA modules from optimization-owned resolved targets through repo-owned method code instead of delegating steady-state behavior to the older built-in adapter implementation.
  - Added repo-owned LoRA module/runtime/state-dict ownership under `library/adapters/methods/peft/lora/`, with Linear and Conv1d/2d/3d support, trainable-ref provenance, repo-owned save/load helpers, and loaded-runtime merge behavior.
  - Added focused module/runtime/config coverage for merged-weight consistency, mixed-dtype forward behavior, registry-owned runtime construction, repo-owned trainable refs, and config validation around the narrowed method-local LoRA surface.
- **TLora is now available as a repo-owned PEFT adapter method under `adapter.peft.tlora`** — The active adapter runtime can now build, train, export, load, and merge TLora modules through the same repo-owned method surface as the other absorbed PEFT methods instead of leaving TLora only in the vendored LyCORIS layer.
  - Added repo-owned TLora config/runtime/state-dict ownership under `library/adapters/methods/peft/tlora/`, including SVD-based orthogonal initialization, learnable singular values, optional scalar mode, optional bypass mode, trainable-ref provenance, and repo-owned save/load/merge behavior.
  - Added `adapter.peft.tlora` typed config plus `configs/_defaults/adapter/peft/tlora.yaml`, with focused validation for missing/non-positive rank, invalid mask bounds, invalid mask schedule values, unsupported singular-vector choices, and invalid dropout probabilities.
  - Added focused module/runtime/config coverage for TLora initialization, mixed-dtype batched-mask forward behavior, export/load round-trips, registry-owned config translation, and centralized config validation.

### Changed

- **The active LoRA method surface now stays method-local and rejects old optimizer-policy compatibility knobs** — The repo-owned LoRA config translation now treats rank, alpha, dropout, and Conv rank/alpha as the active method contract while leaving block-rank, LoRA+, and related grouping-policy fields as explicit rejected legacy inputs instead of silently carrying them forward.
  - Updated `library/adapters/methods/peft/lora/config.py` to fail fast on invalid ranks, invalid dropout probabilities, and `conv_alpha` without `conv_rank`, while rejecting legacy optimizer-policy fields through method-local validation.
  - Updated the repo-owned LoRA runtime and registry coverage so trainable parameter refs now preserve resolved-target provenance instead of exposing the older wrapper behavior where source target identity had to stay unset.

### Removed

- **Obsolete legacy LoRA implementation files are removed from the active PEFT package** — The old built-in LoRA implementation sources and their legacy-only tests no longer live beside the repo-owned LoRA runtime now that the method package owns the active LoRA training/runtime path directly.
  - Removed `library/adapters/methods/peft/lora/impl.py`, `lora.py`, and `lora_diffusers.py`, plus the legacy-only unit coverage in `tests/unit/adapters/test_adapters_lora.py` and `tests/unit/adapters/test_adapters_lora_diffusers.py`.
- **LoRA no longer exposes legacy optimizer-policy knobs on the active method config surface** — The forward `adapter.peft.lora` dataclass/default YAML now only carries method-local LoRA behavior, instead of keeping rejected block-rank and LoRA+ compatibility fields visible in the active config schema.
  - Removed the legacy policy fields from `library/adapters/methods/peft/lora/config.py`, `library/config/dataclasses/peft.py` via the imported method config, `configs/_defaults/adapter/peft/lora.yaml`, and the LoRA-specific training-mode/config-validation tests that only existed to reject those fields after config construction.
- **The PEFT family config no longer carries adapter-shell migration shims as active structured fields** — The active `adapter.peft` surface now relies on branch presence plus `continue_from` / `continue_mode`, instead of keeping the old `method`, `adapter_module`, `adapter_args`, `adapter_weights`, `adapter_rank_from_weights`, `base_weights`, `base_weights_multiplier`, and `training_comment` aliases alive inside the structured PEFT config.
  - Removed the adapter-level shim fields from `library/config/dataclasses/peft.py` and `configs/_defaults/adapter/peft/default.yaml`, simplified `library/adapters/methods/peft/config_resolution.py` to branch-presence resolution only, dropped the legacy continuation/base-merge normalization from `library/config/config_validation.py` and `library/training/modes/adapter_mode.py`, removed the `base_weights` / `base_weights_multiplier` pre-merge path that only existed behind those shims, and updated the affected config/mode/smoke-test fixtures accordingly.

### Fixed

- **Repo-owned TLora now documents the vendor-only timestep-mask contract explicitly instead of quietly widening the architecture boundary** — The method package keeps the TLora mask helpers and validation local, but the active repo training path does not yet wire the vendor-described timestep scheduling into shared strategy/denoiser layers.
  - Added explicit notes in the TLora runtime/default config and PEFT design notes pointing at `library/vendor/lycoris/lycoris/modules/tlora.py`, `library/vendor/lycoris/docs/Algo-Details.md`, and `library/vendor/lycoris/docs/Network-Args.md` as the reference surface for future timestep-mask integration work.
  - Kept the repo-owned TLora module fail-fast behavior for batched-mask merged-weight requests so manual or future integration work still cannot silently collapse per-sample mask intent into invalid merged-weight math.

## [2026-05-02]

### Added

- **ABBA is now available as a repo-owned PEFT adapter method under `adapter.peft.abba`** — The active adapter runtime can now build, train, export, load, and merge ABBA modules through the same repo-owned method surface as the other absorbed PEFT methods instead of leaving ABBA only in the vendored LyCORIS layer.
  - Added repo-owned ABBA config/runtime/state-dict ownership under `library/adapters/methods/peft/abba/`, with the LyCORIS-style split rank surface, optional plain/rank/module dropout, optional scalar mode, optional weight decomposition, optional bypass mode, trainable-ref provenance, and repo-owned save/load/merge behavior.
  - Added `adapter.peft.abba` typed config plus `configs/_defaults/adapter/peft/abba.yaml`, with focused validation for missing/too-small rank, invalid dropout probabilities, and invalid bypass/decompose combinations.
  - Added focused module/runtime/config coverage for ABBA initialization, merged-weight consistency, mixed-dtype forward behavior, stale-cache regression coverage, convolution bypass bias behavior, export/load round-trips, registry-owned config translation, and centralized config validation.
- **IA3 is now available as a repo-owned PEFT adapter method under `adapter.peft.ia3`** — The active adapter runtime can now build, train, export, load, and merge IA3 modules through the same repo-owned method surface as the other PEFT methods instead of leaving IA3 only as a vendored LyCORIS path.
  - Added repo-owned IA3 config/runtime/state-dict ownership under `library/adapters/methods/peft/ia3/`, with explicit input-vs-output axis selection through `train_on_input`, optional module dropout, optional bypass mode, trainable-ref provenance, and repo-owned save/load/merge behavior.
  - Added `adapter.peft.ia3` typed config plus `configs/_defaults/adapter/peft/ia3.yaml`, with focused validation for invalid module-dropout probabilities.
  - Added focused module/runtime/config coverage for IA3 initialization, merged-weight consistency, mixed-dtype forward behavior, input-axis bypass diff behavior, export/load round-trips, registry-owned config translation, and centralized config validation.

### Fixed

- **Repo-owned ABBA now fixes the main vendor-path correctness hazards while keeping the actual Hadamard-product method shape intact** — The repo-owned ABBA path now treats its optimized linear bypass math as a fresh view of current parameters instead of a stale cache, and its merge/bypass behavior matches the intended diff semantics for both linear and convolution targets.
  - Fixed ABBA linear bypass behavior so Khatri-Rao factors are rebuilt from the current trainable weights instead of being cached once and silently going stale as training updates the factors.
  - Fixed ABBA convolution bypass diff behavior so the adapter delta path no longer adds the original bias a second time before the main forward adds the base-module output.
  - Fixed the standard ABBA merged-weight path so the adapter contribution is scaled by `multiplier` exactly once instead of being multiplied once in diff construction and again during merge.
  - Fixed ABBA export behavior so scalar mode no longer depends on a fragile `sqrt(scalar)` symmetry bake; the repo-owned artifact now folds scalar into one exported factor directly and reloads with identity scalar state.
- **Repo-owned IA3 now fixes the main vendor-path correctness hazards while keeping the actual scaling method intact** — The repo-owned IA3 path now treats input-vs-output scaling as first-class state instead of incidental tensor shape, and its merged-weight path matches the real activation-space behavior for biasful output-side targets.
  - Fixed IA3 merged-weight behavior so output-side scaling transforms bias consistently instead of scaling only the weight matrix.
  - Fixed IA3 state-dict reconstruction so the saved `on_input` flag participates in module rebuilds and vendor-style convolution weight layouts are normalized cleanly on load.
  - Fixed the initial repo-owned IA3 runtime shape so leaving `adapter.peft.ia3.train_on_input` unset now auto-selects the axis per target using the current IA3 target-name patterns (`k_proj` / `v_proj` / `to_k` / `to_v` on output, `mlp.fc2` / `ff.net.2` on input) instead of forcing one global default across every target.

## [2026-05-01]

### Added

- **OFT is now available as a repo-owned PEFT adapter method under `adapter.peft.oft`** — The active adapter runtime can now build, train, export, load, and merge OFT modules through the same repo-owned method surface as LoHa, LoCon, and LoKr instead of treating the old built-in OFT path as a supported contract.
  - Added repo-owned OFT config/runtime/state-dict ownership under `library/adapters/methods/peft/oft/`, including LyCORIS-style factorized orthogonal blocks, optional learned rescaling, bypass mode, module dropout, plain dropout, rank dropout, trainable-ref provenance, and repo-owned save/load/merge behavior.
  - Added `adapter.peft.oft` typed config plus `configs/_defaults/adapter/peft/oft.yaml`, with focused validation for missing/non-positive factors, negative constraints, and invalid dropout probabilities.
  - Added focused module/runtime/config coverage for OFT initialization, merged-weight consistency, mixed-dtype forward behavior, export/load round-trips, registry-owned config translation, and centralized config validation.
- **BOFT is now available as a repo-owned PEFT adapter method under `adapter.peft.boft`** — The active adapter runtime can now build, train, export, load, and merge BOFT modules through the same repo-owned method surface as LoHa, LoCon, LoKr, and OFT instead of relying on the vendored LyCORIS implementation as the repo contract.
  - Added repo-owned BOFT config/runtime/state-dict ownership under `library/adapters/methods/peft/boft/`, including LyCORIS-style butterfly factorization, optional partial butterfly depth through `num_stages`, optional learned rescaling, bypass mode, module dropout, plain multiplicative dropout on butterfly transforms, trainable-ref provenance, and repo-owned save/load/merge behavior.
  - Added `adapter.peft.boft` typed config plus `configs/_defaults/adapter/peft/boft.yaml`, with focused validation for missing/non-positive factors, negative constraints, and invalid dropout probabilities.
  - Added focused module/runtime/config coverage for BOFT initialization, merged-weight consistency, mixed-dtype forward behavior, vendor-parity bypass diff behavior, compact export/load round-trips, registry-owned config translation, and centralized config validation.
- **DyLoRA is now available as a repo-owned PEFT adapter method under `adapter.peft.dylora`** — The active adapter runtime can now build, train, export, load, and merge DyLoRA modules through the same repo-owned method surface as the other absorbed PEFT methods instead of routing `library.adapters.dylora` through the old legacy bridge.
  - Added repo-owned DyLoRA config/runtime/state-dict ownership under `library/adapters/methods/peft/dylora/`, with LoRA-shaped weights, explicit `block_size` layout ownership, module dropout, optional bypass mode, trainable-ref provenance, and repo-owned save/load/merge behavior.
  - Added `adapter.peft.dylora` typed config plus `configs/_defaults/adapter/peft/dylora.yaml`, with focused validation for missing/non-positive ranks, invalid `block_size`, and invalid dropout probabilities.
  - Added focused module/runtime/config coverage for DyLoRA initialization, merged-weight consistency, mixed-dtype forward behavior, gradient isolation to the sampled block, registry-owned config translation, and centralized config validation.
- **GLoRA is now available as a repo-owned PEFT adapter method under `adapter.peft.glora`** — The active adapter runtime can now build, train, export, load, and merge GLoRA modules through the same repo-owned method surface as the other absorbed PEFT methods instead of leaving `library.adapters.glora` as a vendor-only path.
  - Added repo-owned GLoRA config/runtime/state-dict ownership under `library/adapters/methods/peft/glora/`, including the real branch-A/branch-B method shape, optional Tucker factorization on the B branch for convolution targets, scalar mode, orthogonalization, bypass mode, trainable-ref provenance, and repo-owned save/load/merge behavior.
  - Added `adapter.peft.glora` typed config plus `configs/_defaults/adapter/peft/glora.yaml`, with focused validation for missing/non-positive ranks and invalid dropout probabilities.
  - Added focused module/runtime/config coverage for GLoRA initialization, merged-weight consistency, bypass multiplier behavior, mixed-dtype forward behavior, Tucker reachability, registry-owned config translation, and centralized config validation.

### Changed

- **The active PEFT registry now promotes OFT as a normal repo-owned method instead of keeping the old built-in path in the supported adapter list** — `library.adapters.oft` now resolves through the repo-owned OFT registration, and the stale `oft_deprecated` package has been removed instead of remaining as on-disk cleanup debt.
  - Repo-owned OFT now stores only the independent upper-triangle values for each skew-symmetric block in its native trainable/exported parameter layout, reducing OFT parameter/state size while keeping the same effective Diag-OFT transform math.
- **The active PEFT registry now promotes BOFT as a normal repo-owned method instead of keeping it as vendored LyCORIS-only behavior** — `library.adapters.boft` now resolves through the repo-owned BOFT registration, and the BOFT path follows the same method-local config/runtime/state-dict ownership model as the other absorbed repo-owned PEFT methods.
  - Repo-owned BOFT stores only the independent upper-triangle values for each butterfly-stage skew block in its native trainable/exported parameter layout, avoiding the older full-square vendor storage while preserving the same effective transform math.
- **The active PEFT registry now promotes DyLoRA as a normal repo-owned method instead of keeping `library.adapters.dylora` on the legacy bridge path** — The active registry now resolves DyLoRA through the repo-owned method package, and the method is treated as a LoRA-shaped training-policy method with explicit block-size ownership instead of as compatibility debt.
- **The active PEFT registry now promotes GLoRA as a normal repo-owned method instead of leaving it as a vendor-only path** — `library.adapters.glora` now resolves through the repo-owned GLoRA registration, and the active method surface owns the method-local config/runtime/state-dict behavior directly rather than depending on vendored wrapper behavior.

### Fixed

- **Repo-owned OFT now fixes broken vendor/legacy bypass behavior while preserving the intended Diag-OFT algorithm shape** — The absorbed OFT path no longer relies on raw mixed-dtype module forwards in bypass/dropout paths, and its bypass diff/rescale behavior now uses shape-safe logic instead of the buggy vendor/legacy implementation.
  - Removed the stale output-shaped plain-dropout mask from the OFT bypass path so `bypass_forward_diff()` no longer hits invalid batch-vs-block broadcasting during training and keeps plain dropout scoped to the OFT block transforms themselves.
  - Fixed OFT state-dict reconstruction so saved block layouts are reloaded exactly instead of being reinterpreted through the generic factorization hint, and fixed `apply_max_norm()` so it actually clamps large effective OFT norms instead of silently no-oping.
  - Fixed the standard OFT merged-weight path so biasful target modules now transform and merge bias consistently with the output-space orthogonal transform instead of leaving the original bias untouched.
- **Repo-owned BOFT now owns its runtime math with repo-tested mixed-dtype, dropout, and checkpoint behavior instead of inheriting those details opaquely from vendored LyCORIS code** — The absorbed BOFT path now has explicit repo coverage for merged-weight parity, bypass diff behavior, compact/full export compatibility, and partial-checkpoint reporting through the active adapter runtime.
  - Added focused BOFT coverage for partial-stage exports/load round-trips and direct/runtime validation of invalid `num_stages` settings.
  - Fixed the standard BOFT merged-weight path so biasful target modules now transform and merge bias consistently with the output-space butterfly operation instead of leaving the original bias untouched.
- **Repo-owned DyLoRA now fixes several vendor-path correctness hazards while keeping the useful dynamic-prefix training idea** — The absorbed DyLoRA path no longer depends on vendor `.data` concatenation, disabled state-dict loading, or the broken `scale` / `gamma` bypass branch.
  - Added explicit block-size-aware export/load ownership so DyLoRA artifacts round-trip the dynamic training layout exactly instead of dropping that information on save.
  - Rebuilt DyLoRA dynamic-prefix training so the sampled block receives gradients while the effective update still uses the prefix through that block, and kept full-rank export/merge on standard LoRA-shaped weights.
- **Repo-owned GLoRA now fixes the main vendor-path correctness hazards while keeping the actual GLoRA branch math intact** — The absorbed GLoRA path no longer leaves Tucker effectively unreachable on convolution targets, and bypass mode now applies multiplier/scale handling consistently instead of inheriting the vendor mismatch.
  - Preserved the unusual LyCORIS non-bypass plain-dropout behavior as an explicit method-local choice for now, rather than silently normalizing it into generic LoRA-family assumptions.

## [2026-04-30]

### Added

- **LoKr is now available as a repo-owned PEFT adapter method under `adapter.peft.lokr`** — The active adapter runtime can now build, train, export, load, and merge LoKr modules without depending on the vendored LyCORIS runtime as the repo contract.
  - Added repo-owned `LokrModule`, runtime, config translation, and state-dict helpers under `library/adapters/methods/peft/lokr/`, including Kronecker factor reconstruction, optional Tucker factorization, scalar mode, DoRA-style weight decomposition, rank/module dropout, RS-LoRA scaling, full-matrix/decomposed factor layouts, and trainable-ref provenance.
  - Registered `lokr` in the adapter-method registry and shared PEFT config schema, with defaults in `configs/_defaults/adapter/peft/lokr.yaml` and fail-fast validation for missing/non-positive rank, unsupported plain dropout, invalid factor, invalid init mode, and bypass/decompose conflicts.
  - Added focused module/runtime/config coverage for LoKr initialization, merged-weight consistency, decomposed export/load round-trips, runtime save/load/merge behavior, registry-owned config translation, Hydra composition, and centralized config validation.
- **LoCon is now available as a repo-owned PEFT adapter method under `adapter.peft.locon`** — The active adapter runtime can now build, train, export, load, and merge LoCon modules without depending on the vendored LyCORIS runtime as the repo contract.
  - Added repo-owned `LoconModule`, runtime, config translation, and state-dict helpers under `library/adapters/methods/peft/locon/`, including classic LoCon down/up factorization, optional Tucker convolution handling, scalar mode, DoRA-style weight decomposition, rank/module dropout, optional plain-dropout compatibility, orthogonalized training weights, RS-LoRA scaling, and trainable-ref provenance.
  - Registered `locon` in the adapter-method registry and shared PEFT config schema, with defaults in `configs/_defaults/adapter/peft/locon.yaml` and fail-fast validation for missing/non-positive rank, invalid init mode, and bypass/decompose conflicts.
  - Added focused LoCon module/runtime/config coverage and updated PEFT method notes so future absorbed LyCORIS work can treat LoCon as the third concrete method reference point.

### Changed

- **Adapter defaults now split shared PEFT shell settings from method-local YAML under `configs/_defaults/adapter/peft/`** — The base adapter defaults no longer inline every method surface in one file. Shared PEFT shell settings now live in `configs/_defaults/adapter/peft/default.yaml`, while method-local defaults live in `configs/_defaults/adapter/peft/lora.yaml` and `configs/_defaults/adapter/peft/loha.yaml`, with `configs/_defaults/adapter/default.yaml` composing LoRA as the current default active adapter config.
- **PEFT-family method resolution no longer lives under the root adapter package** — PEFT method branch selection, legacy PEFT config normalization, and runtime-spec translation now live under `library.adapters.methods.peft`, while the root adapter layer stays focused on generic runtime lookup/build concerns.

### Fixed
- **Repo-owned LoKr now preserves explicit factor orientation and safely handles mixed input/weight dtypes across all forward paths** — LoKr no longer silently reorders user-specified Kronecker factors, and its standard, module-dropout, and bypass execution paths now cast computation to the adapter/base weight dtype before restoring the caller-visible output dtype.
  - Updated `library/adapters/methods/peft/lokr/module.py` so `factorization(..., factor=N)` preserves the requested divisor position, mixed-dtype forward calls no longer fail through raw `F.linear`/`F.conv*` dtype mismatches, and bypass mode uses the same dtype-safe base-op path as the main forward path.
  - Added focused LoKr regression coverage in `tests/unit/adapters/test_lokr_module.py` for explicit-factor orientation plus mixed-dtype standard and bypass forwards.
- **Repo-owned LoHa config/runtime now fails clearly on invalid rank and conflicting bypass/DoRA settings instead of falling through to runtime-side surprises** — The active adapter path now rejects unset or non-positive LoHa rank unless the method-local config explicitly defines a default, and it rejects `bypass_mode=True` together with `weight_decompose=True` before those settings can silently skip the intended decomposition path.
  - Updated `library/adapters/methods/peft/loha/config.py` so runtime settings require an explicit `adapter.peft.loha.rank`, reject non-positive configured ranks, and reject the invalid bypass/decompose combination at method-config translation time.
  - Updated `library/adapters/methods/peft/loha/runtime.py` and `library/adapters/methods/peft/loha/module.py` so the repo-owned LoHa runtime no longer relies on lower-level constructor defaults for rank and raises clear `ValueError`s for invalid direct construction paths.
  - Added focused validation/runtime coverage in `tests/unit/test_config_validation.py`, `tests/unit/adapters/test_loha_module.py`, and `tests/unit/adapters/test_runtime_registry.py` for missing/non-positive LoHa rank and bypass/decompose rejection.
- **Active adapter validation now delegates through method-owned translation instead of hardcoding LoHa rules inside shared PEFT validation** — Shared PEFT validation now exercises the active adapter method’s own config binding/runtime-settings builder, which lets partial configs inherit dataclass defaults generically while keeping method-specific validation rules in the method-local config modules.
  - Updated `library/adapters/method_configs.py` to materialize active method config objects with their dataclass defaults before building runtime settings, so generic validation can validate partial method branches without hardcoded adapter-name checks.
  - Updated `library/config/config_validation.py` to validate the active adapter branch through `build_adapter_runtime_spec(...)` rather than embedding LoHa-specific checks in the shared PEFT validator.
  - Updated `library/adapters/methods/peft/loha/config.py` and `library/adapters/methods/peft/loha/runtime.py` so LoHa now requires `adapter.peft.loha.rank` explicitly unless the method-local config itself defines a default, rather than silently falling back to the lower-level module constructor default.
- **Repo-owned LoHa now treats plain `dropout` as unsupported while pinning initialization and behavior semantics with broader unit coverage** — The repo-owned forward config now rejects plain LoHa `dropout` in favor of the already-defined `rank_dropout` and `module_dropout` knobs, while the method config also exposes explicit initialization modes so the repo can choose between LyCORIS-style legacy init, zero-delta He init, and random nonzero He init.
  - Updated `library/adapters/methods/peft/loha/config.py` and `library/adapters/methods/peft/loha/runtime.py` to reject plain LoHa dropout in both method-config translation and direct runtime settings, rather than accepting a knob whose semantics are intentionally disabled.
  - Updated `library/adapters/methods/peft/loha/module.py` and `tests/unit/adapters/test_loha_module.py` to make the repo-owned init policy explicit and selectable: `lycoris_legacy`, `zero_delta_he`, and `random_nonzero`, with coverage for both zero-effect and active-start behaviors.
  - Broadened LoHa coverage in `tests/unit/adapters/test_loha_module.py`, `tests/unit/adapters/test_runtime_registry.py`, and `tests/unit/test_config_validation.py` for disabled plain dropout, init-mode validation, default/scalar init behavior, rank-dropout scaling, RS-LoRA scaling, DoRA output-side differences, module dropout, bypass-mode forward behavior, and Conv1d/Conv3d merged-weight consistency.
