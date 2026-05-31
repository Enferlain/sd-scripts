## Context

The current resource monitor already exposes useful aggregate counters:

- allocator snapshots (`gpu_allocated_mb`, `gpu_reserved_mb`,
  `gpu_peak_allocated_mb`)
- sampled `gpu_used_mb`
- CPU RSS
- deep-mode allocator counters

Those facts are emitted through three main surfaces:

- raw JSONL/resource-monitor events
- metadata-facing `ResourceMonitorFacts`
- benchmark report / analytics snapshot payloads

The investigation in
[docs_design/resource_visibility_capability_map.md](/mnt/d/Projects/sd-scripts/docs_design/resource_visibility_capability_map.md)
showed that the next gaps are mostly non-console:

- per-device visibility is missing even though `all_visible` collection already
  exists in aggregate form
- CPU virtual memory is missing even though process memory is already read for
  RSS
- richer report/debug summaries are possible, but do not all belong in live
  metadata
- allocation provenance / causality is not a field-growth problem and should
  remain a separate later design step

## Goals / Non-Goals

**Goals:**

- add richer live-capable resource facts where collection is already cheap and
  compatible with the current monitor architecture
- keep raw events, metadata facts, and report payloads aligned as the live
  schema grows
- separate live-capable extensions from report/debug-only expansions
- begin implementation with a small slice that improves non-console resource
  visibility immediately

**Non-Goals:**

- do not solve allocation provenance / causality in this change
- do not redesign phase-tag semantics here; that belongs under `sd-scripts-cuc`
- do not treat console phase-summary behavior as the center of this work
- do not commit up front to every richer report surface before the live schema
  stabilizes

## Decisions

### Decision: Stage the work by data-model difficulty

This change will proceed in three layers:

1. live-capable schema growth
2. report/debug summary growth
3. later provenance-oriented design

Why:

- the monitor already knows how to collect some richer facts cheaply
- report/debug summaries are easier to build once the live schema is stable
- provenance asks are qualitatively different and would otherwise distort the
  first implementation slice

Alternative considered:

- implement richer reports first without changing the live schema

Why not:

- it would force more derived/report-local logic before the underlying resource
  facts are expressive enough

### Decision: Start with host-memory breadth as the first implementation slice

The first implementation slice should add CPU virtual memory alongside RSS.

Why:

- it is cheap: the code already calls `psutil.Process.memory_info()`
- it is non-console and immediately useful
- it exercises the full pipeline:
  - resource-monitor raw events
  - metadata facts/emitter
  - report payload / markdown / analytics snapshot
- it avoids prematurely locking in a per-device nested schema before we are
  ready to review that shape

Alternative considered:

- start with per-device GPU fields first

Why not first:

- that likely needs a new nested payload shape and a broader report update
- it is still a good next slice, but not the smallest “get started” step

### Decision: Keep aggregate fields while adding richer ones

New live fields should be additive rather than replacing the current aggregate
fields immediately.

Why:

- the current report/test surfaces already depend on the aggregate schema
- additive growth keeps the early implementation slice lower-risk

Alternative considered:

- replace aggregate-only fields with a fully nested per-device-first schema

Why not now:

- it is a bigger migration than needed for the first implementation pass

### Decision: Treat provenance as explicit future work

Allocation causality and component-level live ownership remain out of scope for
this change.

Why:

- current counters do not encode ownership
- faking provenance from aggregate counters would create misleading data

Alternative considered:

- infer provenance heuristically from phase boundaries alone

Why not:

- phase-local timing narrows the window but does not establish ownership

## Risks / Trade-offs

- **[Schema growth across several surfaces]** → keep the first slice narrow and
  additive, with focused tests for raw events, metadata facts, and report
  payloads.
- **[Per-device schema may become awkward later]** → start with CPU VMS first,
  then revisit per-device field shape with the capability map in hand.
- **[Report/debug additions may overtake live schema discipline]** → defer
  richer derived summaries until the live fields they depend on are stable.
- **[Users may overread richer counters as provenance]** → keep provenance and
  causality explicitly out of scope in this change.

## Migration Plan

1. Add `cpu_vms_mb` to the live resource event/data model.
2. Thread that field through metadata filing and report payload generation.
3. Update report rendering and targeted tests.
4. Land follow-up tasks for per-device GPU visibility and richer report/debug
   summaries after the first slice is stable.

Rollback is straightforward:

- the first slice is additive
- reverting the new field paths restores the old aggregate-only behavior
