## Context

The current resource stack has a good foundation:

- `phase_tags.py` names the runtime windows.
- `runtime_trace.py` records when those windows happen.
- `resource_monitor/` records observed counters at sessions, phases, and
  samples.
- `summaries.py` estimates known component memory for startup reporting.
- `reports.py` combines those pieces into benchmark output.

The missing part is not a separate provenance feature. The missing part is
phase resource accounting: the ability to say, within the same phase timeline,
which known runtime work accounts for which resource changes.

## Goals / Non-Goals

**Goals:**

- Build one phase-based resource accounting path instead of multiple parallel
  report surfaces.
- Attach ownership to the same phases/events that already define runtime work.
- Prefer explicit owner scopes around code that knows what it is doing over
  post-hoc guesses from aggregate counters.
- Keep the default path low overhead.
- Use plain report language such as `observed`, `estimated`, `accounted`, and
  `accounting gap`.
- Preserve startup/component estimates as inputs to the accounting view, not as
  a separate competing report.

**Non-Goals:**

- Building a standalone `Resource Attribution` report section.
- Introducing user-facing vocabulary such as evidence levels, candidate owners,
  or unknown remainder.
- Moving ownership inference into metadata.
- Promising per-allocation CUDA ownership.
- Adding hot-path per-step ownership tracing by default.

## Decisions

### Decision: Phases/events are the accounting timeline

Resource accounting SHALL use the existing phase/event vocabulary as its primary
runtime identity. If a phase is too broad to explain resource use, the system
should add a narrower phase/event or owner scope inside that runtime code path.

Why:

- the existing system already ties timing and resource counters to phases
- users and developers can reason from code paths they recognize
- it avoids a second naming system beside the runtime trace

### Decision: Add declared owner scopes, not post-hoc candidate claims

Code that knows it is performing resource-relevant work should be able to enter
an owner scope. The initial owner labels are:

- `model_weights`
- `trainable_parameters`
- `gradients`
- `optimizer_state`
- `cache_state`
- `checkpoint_serialization`
- `runtime_workspace`
- `dataloader_workers`
- `allocator_runtime`

An accounting gap is not an owner label. It is a diagnostic result indicating
that observed resource state or movement exceeded what declared scopes and
structural accounting could explain.

Broad labels such as `runtime_workspace` and `allocator_runtime` must only be
used when instrumentation can identify that work directly; they are not
fallback buckets for accounting gaps. `dataloader_workers` requires
worker-process resource measurement and must not be inferred from parent-process
RSS alone.

The first implementation should be small and explicit. Instrumentation targets
should be approached in this order:

1. cache phases
2. checkpoint save
3. model/component loading or registration
4. optimizer setup
5. first training step boundaries
6. accelerator preparation

Why:

- those code paths already know what operation is happening
- a declared scope can be checked against before/after snapshots
- reports can show direct accounting language instead of speculative labels
- broad mixed operations such as accelerator preparation should follow narrower
  operations that can prove the accounting model first

### Decision: Separate structural accounting from scope accounting

Phase accounting SHALL distinguish two sources of accounting information:

- **structural accounting** describes known object sizes, such as parameters,
  trainable parameters, gradients, and optimizer tensors
- **scope accounting** describes observed resource movement during a declared
  runtime operation

An owner scope proves that resource movement occurred during code-declared work.
It does not automatically prove that every persistent byte after the scope is
owned by that operation. Reports and accounting records must preserve that
distinction.

Why:

- structural facts can account for persistent known objects
- operation scopes reveal when resource movement occurs
- combining them without distinguishing their meanings would overstate what the
  runtime observation proves

### Decision: Owner scopes belong to the resource monitor

The first owner-scope API SHALL be a `resource_monitor` context manager that
emits structured accounting-scope start/end events. It may share phase/event
identifiers with runtime trace, but ownership logic SHALL NOT move into
`runtime_trace.py`.

Conceptually:

```python
with resource_monitor.owner_scope(
    owner="cache_state",
    operation="cache.latents.materialize",
):
    ...
```

The resource monitor can capture before/end/peak counters for the scope when
accounting is enabled. Runtime trace remains responsible for timing only.

### Decision: Reports should upgrade the existing phase resource summary

The output target is an improved phase resource view, not a new parallel
section. A future report row should be able to show:

- observed start/end/peak counters
- accounted resource owners for that phase
- estimated structural inputs when relevant
- transient peak/workspace where visible
- accounting gap when observed resource change is not fully accounted for

Why:

- this matches the actual user question: what is using resources during this
  phase
- it keeps the report readable and avoids internal implementation vocabulary

### Decision: Metadata stays downstream

Metadata may eventually store resource accounting records after the runtime
contract settles. It should not become the layer that decides ownership.

Why:

- runtime code owns live state and source truth
- metadata should store accepted facts, not create them

## Migration Plan

1. Remove the experimental standalone `resource_attribution` report surface and
   config knob.
2. Define the first accounting record shape in `library/logging/resource_monitor/`
   around phase/event identity and owner scopes.
3. Add explicit owner scopes around a narrow set of code paths.
4. Combine observed counters, startup estimates, and owner scopes into phase
   accounting rows.
5. Upgrade the existing report output once the accounting rows are useful.
6. Revisit metadata filing only after report/debug output proves stable.

## Open Questions

- How should first-step training state be measured without adding default
  per-step overhead?
- Which accounting rows need host-memory ownership, and which should stay GPU
  focused until there is better evidence?
