## Context

Step metrics currently flow through `library/training/phases/training_loop.py` into `library/logging/metrics.py`. The main producer is `generate_step_logs()`, which builds a flat metric dictionary containing loss, learning-rate, max-norm, validation, loss-modifier, and sampler/timestep metrics. The current tracker bridge then routes that dictionary to Accelerate-backed TensorBoard, W&B, and any additional trackers.

The implementation works, but the contract is implicit. Tests strongly cover LR key naming and tracker initialization, while several metric categories and tracker-routing details are only protected by incidental behavior. This is risky because future logging sinks, dashboards, and run/metadata systems will need stable step metric semantics.

## Goals / Non-Goals

**Goals:**

- Make the step metric event contract explicit and test-covered.
- Preserve existing metric key names unless there is an intentional compatibility decision.
- Make LR label mismatch behavior fail with a clear `ValueError`.
- Cover tracker routing behavior for TensorBoard, W&B, and other Accelerate trackers.
- Keep the implementation small enough to land independently from the metadata system while avoiding a throwaway loose-dict internal API.

**Non-Goals:**

- Do not introduce the global metadata system in this change.
- Do not add adapter config/report rows or PEFT-specific reporting logic.
- Do not redesign resource monitoring, benchmark reports, or dashboard/live plot UI.
- Do not replace Accelerate trackers or change external tracker dependencies.
- Do not introduce a persisted metrics schema or run warehouse in this change.

## Decisions

### Introduce a typed internal step event now

The logging concern should not keep growing around an untyped dictionary if the known direction is structured metric events. This change introduces a typed internal step metric event that captures current metric categories and renders to the existing flat tracker dictionary at the boundary.

The public compatibility behavior remains: `generate_step_logs()` returns the flat dictionary existing training-loop and tracker callers expect. The important difference is ownership: the flat dict is now an output format, not the internal representation future logging work must keep patching directly.

Alternative considered: keep the flat dictionary as the internal runtime API and add coverage around it. That would reduce code churn now, but it knowingly builds on the wrong seam and makes the later typed-event migration more expensive.

### Add small helpers only where they remove ambiguity

The event builder and renderer should stay small and local to `library/logging/metrics.py`. LR metric construction should validate label count and centralize DAdapt/Prodigy derived LR behavior before the event is rendered.

Alternative considered: leave the function monolithic and only add tests. That improves confidence but leaves the mismatch failure mode unclear and keeps duplicated LR logic harder to reason about.

### Preserve metric key compatibility

The change should keep current keys such as `loss/current`, `lr/<name>`, `max_norm/keys_scaled`, `sampler/entropy_ratio`, and `sampler_timestep_hist/bin_<n>`. If any key looks undesirable, record it as a future compatibility discussion rather than changing it silently.

Alternative considered: rename metric namespaces during hardening. That would be tidier in places, but it would be user-visible and should wait for a versioned metrics/schema decision.

### Copy backend-specific tracker payloads

The tracker bridge should treat `global_step` and `epoch` as W&B-specific payload fields and copy the payload for that backend. TensorBoard and other trackers receive the caller-selected `step` argument without inheriting W&B-only keys through shared-dictionary mutation.

Alternative considered: preserve shared-dictionary mutation because it is current behavior. That makes backend ordering observable and lets W&B-only fields leak into later trackers, so this change intentionally locks in copy-per-backend behavior instead.

## Risks / Trade-offs

- Existing behavior may encode quirks that are not ideal, such as W&B-specific payload mutation. Mitigation: define the intended backend-copy behavior in tests as part of this change.
- The typed event is not yet a durable persisted schema. Mitigation: keep it narrowly focused on runtime step observability and render to the existing flat keys for compatibility.
- LR mismatch validation may expose previously hidden caller bugs. Mitigation: raise a clear `ValueError` with label and LR counts, and add call-site tests where useful.
- Sampler metric tests may require light fake objects with tensors. Mitigation: use minimal stubs and CPU tensors to avoid model/training overhead.

## Migration Plan

1. Add a typed step-event builder and flat-dict renderer for current metric categories.
2. Add tracker-routing tests for TensorBoard, W&B, and generic trackers.
3. Add LR label mismatch tests that define the desired `ValueError`.
4. Extract or update LR helper logic as needed to satisfy the contract.
5. Run focused logging and training-loop tests.

Rollback is straightforward: metric hardening is limited to `library/logging/metrics.py` and tests unless a narrow training-loop call adjustment is required.

## Open Questions

- Should old helper functions such as `append_lr_to_logs_with_names()` remain long-term, or should they be deprecated after the step-event path has equivalent tested helper behavior?
