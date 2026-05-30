**Discovery**

The biggest discovery is that the epoch problem is more than one banner. In [library/training/phases/training_loop.py](/mnt/d/Projects/sd-scripts/library/training/phases/training_loop.py:658), `run_training_loop()` sets `trainer._current_epoch_state.value = epoch + 1` and prints `Epoch 1/N`. That same one-based value then feeds step/epoch side effects and scheduling through [_run_step_side_effects()](/mnt/d/Projects/sd-scripts/library/training/phases/training_loop.py:126), [_finalize_epoch()](/mnt/d/Projects/sd-scripts/library/training/phases/training_loop.py:325), checkpoint saves in [_save_step_checkpoint_artifacts()](/mnt/d/Projects/sd-scripts/library/training/phases/training_loop.py:83), and validation cadence via [ValidationScheduler.should_run()](/mnt/d/Projects/sd-scripts/library/training/phases/validation.py:53). So changing the stored epoch state to zero-based would not be a cosmetic cleanup; it would change existing runtime semantics and cadence behavior.

Startup sampling/validation is still outside trace ownership exactly where you called out. [_run_startup_eval_actions()](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:770) just computes booleans and calls [run_sampling_and_validation()](/mnt/d/Projects/sd-scripts/library/training/phases/orchestration_helpers.py:60) inside eval mode, but there are no `monitored_phase(...)` wrappers or runtime-trace events around either startup sampling or startup validation. If startup regressions matter, those need explicit trace boundaries rather than being left inside one opaque startup window.

The long metadata span is very likely real work, not just awkward labeling. [_initialize_training_metadata()](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:636) calls [build_training_metadata_bundle()](/mnt/d/Projects/sd-scripts/library/metadata/emitters/run.py:141) and then `strategies.update_metadata(...)`. For SD and SDXL the strategy appenders are trivial, so the weight is mostly inside [_build_training_ss_metadata()](/mnt/d/Projects/sd-scripts/library/metadata/emitters/run.py:196), especially [_append_dataset_compatibility_metadata()](/mnt/d/Projects/sd-scripts/library/metadata/emitters/run.py:294), which computes full tag frequency over the manifest via [compute_tag_frequency()](/mnt/d/Projects/sd-scripts/library/data/caption_processor.py:263), and [_append_model_source_compatibility_metadata()](/mnt/d/Projects/sd-scripts/library/metadata/emitters/run.py:344), which hashes base model and VAE files when paths exist. That is exactly the kind of work that can create a noticeable post-cache startup stall.

**Best Direction**

For epochs, the safest high-quality direction is: keep orchestration/cadence/checkpoint semantics as they are for now, but stop using that one-based state as if it were runtime identity. In practice that means zero-based stays canonical for trace/resource/runtime identifiers, and presentation gets its own explicit display value instead of leaking into trace-like contexts. I would not recommend flipping `_current_epoch_state.value` itself to zero-based in the same pass unless we want a broader contract migration with scheduler/checkpoint fallout.

For startup tracing, I’d add separate startup trace phases for sampling and validation rather than one generic startup-eval bucket. That gives you actual regression visibility and avoids conflating image generation time with validation loss time. Since [run_sampling_and_validation()](/mnt/d/Projects/sd-scripts/library/training/phases/orchestration_helpers.py:60) is reused outside startup, I’d keep the helper generic and add startup-owned wrapping at the trainer layer instead of stuffing startup-only semantics into the shared helper.

For the metadata stall, I’d split it into at least:
- base training metadata assembly
- dataset-derived compatibility metadata
- model/VAE hash metadata
- strategy-specific metadata appenders

I’d also seriously consider whether tag-frequency and model-hash compatibility fields belong in startup at all, or whether they should be deferred to checkpoint/report generation when they are actually needed.

**Polish Recommendations**

- Rename `time_to_training_finished_s` to `time_to_run_ended_s` or `time_to_run_finished_s`. The markdown already moved to “Run Ended”; the JSON should match.
- Keep runtime trace out of first-class metadata facts for now. [AnalyticsSnapshotFacts](/mnt/d/Projects/sd-scripts/library/metadata/dataclasses/observability.py:99) is a good holding place until the trace schema stabilizes.
- Normalize Python-side naming to `run_identifier`. Right now [resource_monitor.py](/mnt/d/Projects/sd-scripts/library/logging/resource_monitor.py:153) still takes `run_id`, while metadata/report code prefers `run_identifier`. I’d make `run_identifier` the code-level source of truth and only preserve `run_id` in JSONL if compatibility matters.
  Marker: [ADDRESSED 2026-05-28]
  Current state:
  [library/logging/resource_monitor.py](/mnt/d/Projects/sd-scripts/library/logging/resource_monitor.py:187)
  now accepts `run_identifier`, stores `_run_identifier`, emits raw
  resource-monitor events with a `run_identifier` field, and reports filter the
  JSONL stream by that same name in
  [library/logging/reports.py](/mnt/d/Projects/sd-scripts/library/logging/reports.py:227).
- Consolidate Hydra context helpers. [metrics.resolve_hydra_config_name()](/mnt/d/Projects/sd-scripts/library/logging/metrics.py:268) and [reports._resolve_hydra_context()](/mnt/d/Projects/sd-scripts/library/logging/reports.py:126) should become one shared helper.
  Marker: [ADDRESSED 2026-05-29]
  Current state:
  the duplication is gone. Hydra runtime context now resolves through
  `resolve_hydra_runtime_context()` in
  [library/utils/common_utils.py](/mnt/d/Projects/sd-scripts/library/utils/common_utils.py:6),
  and both
  [library/logging/reports.py](/mnt/d/Projects/sd-scripts/library/logging/reports.py:323)
  and
  [library/training/runners/trainer.py](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:309)
  read from that shared helper directly instead of maintaining local
  report/metrics variants.
- Keep the `TrackerSink` vs `TrainingObserver` split. The current code still reads coherently: [TrackerSink](/mnt/d/Projects/sd-scripts/library/logging/metrics.py:75) is tracker transport only, while [TrainingObserver](/mnt/d/Projects/sd-scripts/library/logging/metrics.py:87) owns richer repo observability. I’d document that boundary more explicitly, not redesign it.
  Marker: [ADDRESSED 2026-05-30]
  Current state:
  the boundary is now more explicit in code. The trainer-facing step-metrics
  path routes through `TrainingObserver.log_metrics(...)`, while the narrower
  backend transport names now read as tracker-specific:
  `TrackerSink` and `AccelerateTrackerSink` in
  [library/logging/metrics.py](/mnt/d/Projects/sd-scripts/library/logging/metrics.py:75).
  [LoggingTrainingObserver](/mnt/d/Projects/sd-scripts/library/logging/metrics.py:134)
  remains the repo-facing observability owner for lifecycle/artifact/startup
  summary work, and it delegates flat scalar tracker emission through the
  attached tracker sink instead of leaving trainer code to choose between
  observer and raw tracker-helper paths.
- Add failure-path coverage around startup-phase exceptions, startup-eval exceptions, and failures after progress-bar creation but before epoch completion. The cleanup path in [Trainer.train()](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:226) is good, but the trace/report tests are still thin there.
  Marker: [ADDRESSED 2026-05-30]
  Current state:
  the missing cases are now covered. Integration coverage in
  [tests/integration/test_training_loop_integration.py](/mnt/d/Projects/sd-scripts/tests/integration/test_training_loop_integration.py:425)
  still exercises failure before the first synced step, and it now also covers
  failure after the progress bar has started but before epoch completion.
  Trainer tests in
  [tests/unit/training/test_training_trainer.py](/mnt/d/Projects/sd-scripts/tests/unit/training/test_training_trainer.py:207)
  already verified monitor shutdown and report writing on mid-run failure, and
  they now also verify startup-phase failure and startup-eval failure with
  benchmark-report/resource-monitor/runtime-trace cleanup.
- Keep the runtime trace single-threaded assumption for now. It’s fine as long as trainer-owned code is the only emitter.

**Resource Investigation**

The current monitor already captures more than just a timer:
- allocated/reserved/peak allocated GPU memory and CPU RSS in [_collect_snapshot()](/mnt/d/Projects/sd-scripts/library/logging/resource_monitor.py:238)
- sampled `gpu_used_mb` plus `collection_ms` in [_collect_sample_metrics()](/mnt/d/Projects/sd-scripts/library/logging/resource_monitor.py:924)
- deep allocator counters in deep mode via the resource monitor event payload

What it still does not answer well is per-device detail, virtual memory, allocation origin, or “what exactly caused this startup phase to bloat.” That still feels like separate investigation work under `sd-scripts-83b`, not something to blur into the immediate cleanup pass.

## Additional Addressed Follow-up

Marker: [ADDRESSED 2026-05-28]

The items below were also resolved during the same logging / trace cleanup
slice, even though they were not originally grouped under the staged-review
tomorrow section.

### [ADDRESSED] Runtime Epoch Identity vs Display Counter Split

Marker: [ADDRESSED]

The code now makes the intended epoch split explicit:

- zero-based runtime identity is represented separately as `epoch_index`
- one-based user-facing progress remains presentation-only as `display_epoch`

That means `training.epoch.<n>` and related runtime/trace identity stay
zero-based without forcing the console `Epoch 1/N` progress banner to change.

### [ADDRESSED] Runtime Milestone Naming

Marker: [ADDRESSED]

The machine-readable runtime trace payload no longer uses the older
`time_to_training_finished_s` wording.

It now uses `time_to_run_ended_s`, which matches the report language more
closely and removes the old JSON/markdown naming mismatch.

### [ADDRESSED] Trace-Only Bootstrap Phase Reporting

Marker: [ADDRESSED]

Benchmark/report output now surfaces bootstrap-only trace spans such as
`startup.accelerator` more explicitly instead of leaving them as implicitly
“missing” from the resource-phase view.

This does not make them resource-monitor phases; it makes the report honest
about the fact that they are runtime-trace-only because they begin before the
resource monitor starts.

## Deferred Trace-Tag Follow-up

This note is intentionally sequenced after the broader startup / eval /
observability sections are reviewed overall.

Marker: [PARTIALLY DEFERRED TO VALIDATION / SAMPLING REDESIGN]

The repo should not rush into adding many new phase tags before the surrounding
ownership and lifecycle seams are inspected end-to-end. In particular:

- startup sampling / validation should be reviewed as part of the broader
  startup-flow trace and logging pass
- training-time sampling / validation should be reviewed as part of the broader
  eval-side-effect ownership pass
- validation-start / sample-start human-facing lifecycle output should be
  reviewed as part of the broader “training-owned vs strategy-owned logging”
  cleanup

Based on the planned redesign of validation, and possibly sampling, the
sampling/validation-specific parts of this section should be treated as
redesign-gated rather than near-term cleanup work. Their logging, metadata, and
trace shape should be decided together with that redesign instead of being
polished in place first.

Only after those sections are looked at overall should the next trace-tag pass
land.

### [REDESIGN-GATED] Most Important Missing Tags

The clearest missing coverage today is eval work.

[Trainer._run_startup_eval_actions()](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:770)
currently runs sampling / validation outside `monitored_phase(...)`, so the
runtime trace sees the enclosing startup spans but not the startup eval work as
its own timed phases.

If startup-regression visibility still matters after the broader startup review,
the most useful additions would be:

- `startup.eval.sample`
- `startup.eval.validation`
- `startup.eval.cleanup`

`startup.eval.cleanup` is worth keeping separate because
[_cleanup_after_startup_eval()](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:761)
does explicit cleanup and synchronization work (`gc.collect()`,
`torch.cuda.empty_cache()`, `torch.cuda.synchronize()`) that can itself look
like a mysterious post-eval stall.

Marker: [DEFERRED TO VALIDATION / SAMPLING REDESIGN]

These startup-eval tags should not be treated as immediate cleanup tasks while
validation/sampling structure is still expected to change.

### [REDESIGN-GATED] Training-Time Eval Gaps

The same kind of gap exists during normal training.

Step-triggered and epoch-end eval work routes through
[run_sampling_and_validation()](/mnt/d/Projects/sd-scripts/library/training/phases/orchestration_helpers.py:60),
but that helper is still not wrapped in a trace phase of its own. As a result,
expensive sample generation or validation loss calculation currently disappears
into the surrounding `training.epoch.<n>` span.

If that broader eval pass still concludes these are the right boundaries, the
next useful tags would be:

- `training.eval.sample`
- `training.eval.validation`

These would most naturally be applied from training-owned orchestration code
such as:

- [_run_step_side_effects()](/mnt/d/Projects/sd-scripts/library/training/phases/training_loop.py:126)
- [_finalize_epoch()](/mnt/d/Projects/sd-scripts/library/training/phases/training_loop.py:325)

Marker: [DEFERRED TO VALIDATION / SAMPLING REDESIGN]

These training-eval tags should be revisited only when the validation /
sampling redesign settles the intended orchestration shape.

### Metadata Span Follow-up

The current `startup.metadata` phase is still likely too coarse if the goal is
to understand pre-training hangs precisely.

The likely expensive work inside
[_initialize_training_metadata()](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:636)
does not appear to be mainly strategy-side for SD / SDXL. The heavier likely
contributors are inside
[_build_training_ss_metadata()](/mnt/d/Projects/sd-scripts/library/metadata/emitters/run.py:196),
especially:

- dataset-derived compatibility metadata in
  [_append_dataset_compatibility_metadata()](/mnt/d/Projects/sd-scripts/library/metadata/emitters/run.py:294)
- model / VAE hashing in
  [_append_model_source_compatibility_metadata()](/mnt/d/Projects/sd-scripts/library/metadata/emitters/run.py:344)

Current state:
`_build_training_ss_metadata()` is the central aggregator for the heavy work.
It appends a large static metadata map, then performs three potentially
expensive follow-ups in sequence:

- adapter compatibility projection
- dataset compatibility projection
- model / VAE source hashing

The dataset path is not just a couple of fields: it computes full tag frequency
over the manifest, walks every manifest entry to build dataset-dir summaries,
builds bucket-resolution summaries, and JSON-serializes all of those
structures. The model-source path hashes the base model and VAE when those
paths exist on disk, using both the legacy short hash and the configured
full-hash algorithm.

If the broader metadata/startup review still wants more visibility here, a
later split could look like:

- `startup.metadata.base`
- `startup.metadata.dataset`
- `startup.metadata.model_source`
- `startup.metadata.strategy`

This should remain a follow-up to the broader startup/metadata review, not an
immediate blind expansion of the tag list.

### [REDESIGN-GATED] Existing Lifecycle Output That Is Not Yet Canonicalized

Some user-facing lifecycle output already exists, but it is still not expressed
through the structured trace/tag vocabulary.

- Sampling currently logs `[sample] generating sample images at step: ...` from
  [sample_generation.py](/mnt/d/Projects/sd-scripts/library/training/sample_generation.py:459).
  This is training-owned, which is good, but it is still a free-form logging
  convention rather than a canonical phase tag.
- Validation still prints `Validating...` from model-family strategy code, for
  example:
  [sd/validation.py](/mnt/d/Projects/sd-scripts/library/strategies/sd/validation.py:99),
  [sdxl/validation.py](/mnt/d/Projects/sd-scripts/library/strategies/sdxl/validation.py:145),
  and
  [sd3/validation.py](/mnt/d/Projects/sd-scripts/library/strategies/sd3/validation.py:104).
  That is exactly the kind of lifecycle output that should likely move up into
  training-owned orchestration once the broader eval/logging review is done.

Marker: [DEFERRED TO VALIDATION / SAMPLING REDESIGN]

The sampling/validation lifecycle-output cleanup described here depends on the
planned redesign and should be handled there, not as a standalone polish pass
on the current structure.

### Lower-Priority Untagged Areas

There are also a few untagged end-of-run or helper spans that may matter later,
but they appear lower priority than startup/eval:

- [_initialize_training_runtime()](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:665)
  is grouped under `startup.runtime` and may eventually want finer visibility
  if live-plotter or objective-runtime setup becomes slow.
- [_save_final_state_if_enabled()](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:788)
  and
  [_save_final_checkpoint_artifacts()](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:796)
  are still not broken out as distinct finalization spans.

Current state:
`startup.runtime` is currently a small but mixed bucket. Today it includes:

- `objective.build_runtime(...)`
- loss-modifier runtime attachment
- main-process live-plotter setup via `setup_live_plotter(...)`

The finalization path is also more specific now than the old note implies:

- `_save_final_state_if_enabled()` conditionally persists accelerator state
- `_save_final_checkpoint_artifacts()` unwraps the trainable model, saves the
  final checkpoint through the canonical checkpoint path, and may also emit a
  loss-modifier sidecar checkpoint

So if final shutdown timing ever matters, those two finalization helpers are
good candidate boundaries rather than one generic “train end” bucket.

## Remaining Investigation Notes

These are still open, but the current code now gives a little more shape to
what the next pass would actually be inspecting.

### Resource Visibility Limits

The resource monitor already supports a bit more than the earliest note
suggested:

- allocator/RSS snapshots from
  [_collect_snapshot()](/mnt/d/Projects/sd-scripts/library/logging/resource_monitor.py:254)
- sampled used-memory collection via NVML or `torch.cuda.mem_get_info()` in
  [resource_monitor.py](/mnt/d/Projects/sd-scripts/library/logging/resource_monitor.py:907)
- `device_scope` support for current-device vs `all_visible` aggregate reads

But the current surfaces are still aggregate-oriented:

- there are no per-device rows in the emitted phase summaries or report payloads
- CPU virtual memory is not recorded, only RSS
- allocation provenance is not tracked; deep mode exposes allocator counters,
  not causal ownership of allocations

That means `sd-scripts-83b` still looks like a separate observability expansion
track rather than a quick polish pass on the current monitor.

Those are useful to remember, but they should come after the broader startup /
eval / logging ownership review rather than compete with it.

## Tomorrow Follow-up From Staged Review

Review of the currently staged observability changes did not find any blocking
behavioral regressions, but it did surface a few follow-up items worth
addressing soon.

Marker: [ADDRESSED 2026-05-27]

The three concrete cleanup items below have now been addressed in the
subsequent cleanup pass. They remain here as a record of what was followed up
and resolved.

### [ADDRESSED] Trace / Monitor Phase-End Consistency

The current `monitored_phase(...)` cleanup path ends the resource-monitor phase
first and only then ends the runtime-trace phase.

That means a rare `runtime_trace.phase_end(...)` failure could leave the monitor
showing a phase as closed while the trace still considers it open. This looks
like a low-risk defensive-path issue rather than a normal runtime problem, but
it is worth tightening or at least documenting deliberately so the two
observability surfaces do not silently drift apart.

Marker: [ADDRESSED]

`RuntimeTrace.phase_end(...)` now keeps the phase open until
end-recording work finishes successfully, so unexpected trace-side recording
failures no longer silently discard the open phase from diagnostics.

### [ADDRESSED] Checkpoint Saved Event Semantics

The new centralized checkpoint lifecycle logging is the right ownership
direction, but the exact event/phase boundary still needs a small semantic
cleanup.

- `EVENT_CHECKPOINT_SAVED` currently fires after the enclosing
  `PHASE_CHECKPOINT_SAVE` span has ended
- one lifecycle log path still uses the event tag instead of the phase tag

That is not a functional bug, but it does make the checkpoint trace semantics a
bit muddy. Tomorrow's follow-up should decide whether the saved event belongs
inside the save phase, or whether the current ordering is intentional and just
needs clearer naming / documentation.

Marker: [ADDRESSED]

`EVENT_CHECKPOINT_SAVED` now records inside the enclosing
`PHASE_CHECKPOINT_SAVE` span, and the extra lifecycle log that reused the event
tag as a phase-like status line has been removed.

### [ADDRESSED] Epoch Phase Tag Matcher Precision

Marker: [ADDRESSED]

`is_training_epoch_phase(...)` no longer
matches by prefix alone and now accepts only numeric
`training.epoch.<n>` tags.

That removes the risk that a future tag beginning with `training.epoch.` could
be treated as a real epoch span when it was actually some other kind of phase.
