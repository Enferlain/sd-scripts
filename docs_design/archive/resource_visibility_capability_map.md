# Resource Visibility Capability Map

> Archived: superseded by `docs/resource_intelligence.md` and the governing
> `resource-intelligence-system` OpenSpec change.

This note captures the current non-console resource-observability surface and
the remaining implementation questions for `sd-scripts-83b`.

The goal is not to propose a full design up front. It is to answer four
practical questions from the current code:

1. what facts are already collected but only exposed in aggregate form
2. which missing facts are cheap enough for live JSONL / metadata emission
3. which facts belong only in reports / debug snapshots
4. which asks require a new provenance model rather than one more field

## Scope

This document is about underlying resource facts and surfaces, not console
verbosity.

Relevant active surfaces:

- `library/logging/resource_monitor.py`
- `library/logging/reports.py`
- `library/logging/summaries.py`
- `library/metadata/dataclasses/observability.py`
- `library/metadata/emitters/observability.py`

Adjacent but separate topics:

- phase-tag vs resource-phase coupling belongs under `sd-scripts-cuc`
- `output.saving.no_metadata` behavior belongs under `sd-scripts-due`
- validation / sampling trace work is redesign-gated elsewhere

## Current Capability Surface

### Collection Modes

`ResourceMonitorConfig` currently exposes:

- `off`
- `basic`
- `sampled`
- `deep`

From `library/logging/resource_monitor.py`, the intended layering is:

- `basic`: cheap allocator + RSS counters on the hot path
- `sampled`: adds a daemon sampler thread for `gpu_used_mb` and `cpu_rss_mb`
- `deep`: sampled mode plus bounded `torch.cuda.memory_stats()` collection

### Raw Event Schema

`BasicResourceMonitor._build_event(...)` emits a JSONL/metadata event payload
with these resource fields:

- `gpu_allocated_mb`
- `gpu_reserved_mb`
- `gpu_peak_allocated_mb`
- `gpu_used_mb`
- `cpu_rss_mb`
- `steps_per_sec`
- `samples_per_sec`
- `dropped_samples`
- `collection_ms`
- `deep_alloc_retries`
- `deep_ooms`
- `deep_active_mb`
- `deep_reserved_mb`
- `deep_inactive_split_mb`
- `deep_window_active`

This same aggregate schema is mirrored into metadata-facing
`ResourceMonitorFacts`.

### Report / Snapshot Surface

`library/logging/reports.py` currently preserves resource-monitor data in three
shapes:

1. session summary
   - `session_start`
   - `session_end`
   - `gpu_used_peak_session_mb`

2. paired phase rows
   - duration
   - allocated/reserved start and end
   - peak allocated
   - peak sampled `gpu_used_mb`
   - CPU RSS start and end

3. analytics snapshot payload
   - the benchmark-report snapshot currently carries the same summarized
     `resource_monitor` block that the JSON report writes

### Existing Richer Breakdown Surface

`emit_startup_component_memory(...)` already provides one richer non-console
resource breakdown:

- loaded component parameter bytes
- estimated trainable bytes
- estimated frozen bytes
- estimated gradients
- estimated optimizer state

Important limitation:

- this is a component-level parameter estimate, not live allocator provenance
- it is currently startup-only

## Question 1: What Is Already Collected But Only Aggregated?

### GPU Used Memory Across Multiple Devices

`SampledResourceMonitor` already supports:

- `device_scope=local`
- `device_scope=all_visible`

But for `all_visible`, both NVML and `torch.cuda.mem_get_info()` paths sum used
bytes across visible devices into one aggregate `gpu_used_mb` number.

Current consequence:

- we can answer “how much GPU memory is used across all visible devices”
- we cannot answer “which device is using how much”

### Phase Resource Rows

Reports already derive phase rows from raw `phase_start` / `phase_end` events
and sampled `step_sample` events.

But those rows collapse all resource data into one record per phase:

- one start/end allocator snapshot
- one sampled GPU-used peak
- one CPU RSS start/end pair

Current consequence:

- we can answer “what happened during this phase overall”
- we cannot answer “how did device/resource usage evolve inside the phase”

### Deep Allocator Counters

Deep mode already collects global allocator counters from
`torch.cuda.memory_stats()`:

- alloc retries
- OOMs
- active bytes
- reserved bytes
- inactive split bytes

But these are still emitted as aggregate counters for the active device context,
not attributed by component, call site, or subphase.

Current consequence:

- we can answer “allocator pressure increased”
- we cannot answer “what part of startup/training caused it”

### Host Memory

CPU RSS is already collected in both snapshot and sampled paths.

Current consequence:

- we can answer “how much resident process memory was used”
- we cannot answer broader host-memory questions like virtual memory size,
  committed mappings, or whether host-memory growth came from one subsystem or
  another

## Question 2: Which Missing Facts Look Cheap Enough For Live JSONL / Metadata?

These are the strongest candidates for live-capable expansion because they are
close to data the code already touches today.

### Per-Device GPU Used Rows

Why it looks cheap:

- the sampled monitor already iterates devices when `device_scope=all_visible`
- instead of summing, it could also retain per-device values

Likely shape:

- add a per-device structure alongside aggregate `gpu_used_mb`
- keep aggregate fields for compatibility / report totals

Likely surfaces:

- JSONL event payload
- `ResourceMonitorFacts`
- report payload

### Per-Device Allocated / Reserved / Peak Allocated

Why it looks plausible:

- current snapshot path already reads allocator stats
- expanding to per-device reads is conceptually similar to the `all_visible`
  sampled used-memory path

Caveat:

- this is still somewhat more intrusive than sampled used memory because the
  current snapshot path assumes a single active/local device

Likely bucket:

- still live-capable, but a little riskier than per-device `gpu_used_mb`

### CPU Virtual Memory

Why it looks cheap:

- the code already calls `self._process.memory_info()`
- RSS comes from that call now
- adding VMS (and possibly other fields if consistently available) is likely a
  low-cost extension of the same process-memory read

Likely surfaces:

- JSONL
- `ResourceMonitorFacts`
- session / phase report summaries

### Explicit Aggregation Metadata

This is not a new resource fact so much as a clarity improvement:

- whether a value is local-device, all-visible aggregate, or per-device
- whether a peak came from sampled events or allocator snapshots

Why it looks cheap:

- the code already carries `device_scope`
- the phase-row builder already knows which derived values are sampled

Likely value:

- reduces ambiguity before adding more raw counters

## Question 3: Which Facts Belong Only In Reports / Debug Snapshots?

These are useful, but they do not look like good default live-event fields.

### Richer Startup / Finalization Breakdowns

Examples:

- startup component memory rows beyond the current startup estimate block
- breakdowns for optimizer-state estimates by component
- shutdown/finalization-specific resource summaries

Why report/debug-only:

- these are derived summary products
- they are more useful as read-friendly aggregates than as per-event fields

## Implemented Surface Split

The current implementation now has an explicit split between live resource facts
and report/debug-only summaries.

### Live Metadata / JSONL Facts

These fields are emitted directly by the resource monitor and mirrored into
`ResourceMonitorFacts`:

- `gpu_allocated_mb`
- `gpu_allocated_by_device_mb`
- `gpu_reserved_mb`
- `gpu_reserved_by_device_mb`
- `gpu_peak_allocated_mb`
- `gpu_peak_allocated_by_device_mb`
- `gpu_used_mb`
- `gpu_used_by_device_mb`
- `cpu_rss_mb`
- `cpu_vms_mb`

These belong to always-on event/schema surfaces because they are raw or
near-raw process/runtime observations.

### Report / Debug-only Summaries

These are derived in `library/logging/reports.py` from the live schema above
instead of being emitted as new resource-monitor event fields:

- `session_gpu_used_peak_by_device_rows`
- `phase_device_rows`
- markdown sections such as:
  - `Per-Device GPU Session Peaks`
  - `Per-Device GPU Phase Details`

These belong to report/debug surfaces because they are explanatory reshapes of
live data rather than new first-class runtime observations.

### Per-Phase Delta Narratives

Examples:

- “GPU allocated grew by X during this phase”
- “reserved memory stayed flat but sampled GPU used peaked at Y”
- “CPU RSS climbed while GPU allocator stayed flat”

Why report/debug-only:

- these are interpretation layers over existing measurements
- they do not need to be emitted on every raw event

### Deep-Window Summary Records

Deep mode already has a window concept and summary logging.

Potential extensions:

- peak inactive-split by device
- allocator-pressure summary across the deep window
- step-window aggregates rather than one event per sample

Why report/debug-only:

- deep mode is already the heavier path
- window summaries are naturally debug artifacts, not baseline live metadata

### Historical Time-Series Exports

Examples:

- sampled device-memory curves
- host-memory curves
- per-phase subseries

Why report/debug-only:

- they are bulky
- they are more analytics/debug material than first-class lifecycle metadata

## Question 4: Which Asks Need A New Provenance Model?

These are the areas where “just add another field” is the wrong instinct.

### Allocation Provenance / Causality

Examples:

- which component caused the allocation jump
- whether growth came from weights, optimizer state, activations, cache, or
  temporary workspace
- which startup helper or subphase actually caused the memory increase

Why this needs a new model:

- current data sources (`memory_allocated`, `memory_reserved`, `mem_get_info`,
  `memory_stats`) expose counters, not ownership
- counters alone do not encode causal attribution
- even per-phase timing only narrows the window; it does not explain the owner

Implication:

- this likely needs either explicit phase/component attribution logic, new
  local instrumentation around known operations, or a separate provenance/debug
  concept rather than more fields on `ResourceMonitorFacts`

### Component-Level Live Attribution

Examples:

- live per-component GPU footprint during startup/training
- live optimizer-state attribution by component

Why this needs design:

- startup parameter-byte estimates already exist, but they are static estimates
- turning that into live runtime attribution requires explicit modeling of what
  “belongs” to a component once tensors, optimizer states, and temporary
  workspaces are in play

### Cross-Device Ownership Semantics

Examples:

- how to attribute all-visible totals back to devices and then back to runtime
  owners
- whether distributed/deepspeed/offload paths should preserve one schema or use
  mode-specific resource records

Why this needs design:

- a flat per-device map is easy
- a stable ownership model across local, distributed, and offloaded paths is
  not just a field addition

## Recommended Buckets For Implementation

### Bucket A: Live-Capable Extensions

Strongest candidates:

- per-device `gpu_used_mb`
- per-device allocator snapshots (`allocated`, `reserved`, `peak_allocated`)
- CPU virtual memory fields
- clearer aggregation/scope metadata

These are the best first implementation candidates because they build on data
the monitor already knows how to access.

### Bucket B: Report / Debug-Only Expansions

Strongest candidates:

- richer phase delta summaries
- richer startup/finalization breakdown sections
- deep-window summary artifacts
- optional sampled time-series exports

These should likely be implemented at the report/snapshot layer rather than as
always-on live metadata.

### Bucket C: Needs A Provenance Design

Strongest candidates:

- allocation causality
- component-level live attribution
- precise “where did this memory come from?” answers

These should not be attempted as ad hoc field growth on the current event
schema.

## Suggested Implementation Order

If this investigation turns into code work, the cleanest next order looks like:

1. add explicit per-device capability to the resource data model
2. decide which per-device fields should appear in JSONL and metadata versus
   only in report payloads
3. add CPU virtual memory if it proves reliable across supported platforms
4. improve report/debug summaries using the richer live data
5. only then open a separate design pass for provenance / causality

## Short Takeaway

The current resource system is already reasonably strong at:

- aggregate allocator state
- aggregate sampled used memory
- CPU RSS
- session / phase peaks
- startup parameter-memory estimates

It is still weak at:

- per-device visibility
- host-memory breadth beyond RSS
- structured “what changed during this phase” summaries
- phase resource accounting that can explain known resource owners inside the
  existing phase/event timeline

That makes the next implementation boundary fairly clear:

- first expand the resource facts we can measure cheaply
- then improve report/debug presentation of those facts
- then add declared owner scopes for code paths that know what resource-relevant
  work is happening

## Follow-up Status

The original investigation issue for this note (`sd-scripts-83b`) is complete.

The remaining work is now governed more explicitly:

- report/debug interpretation upgrades were taken as direct implementation work
  and now live in the benchmark-report debug/interpretation layer
- the complete resource-system direction now lives under
  `openspec/changes/resource-intelligence-system/`, including observations,
  structural facts, profiles, evidence-constrained accounting, metadata-backed
  queries, and projected artifacts
- the historical phase-resource-accounting direction is archived and no longer
  governs implementation
- broader host-memory and distributed-semantics evaluation remains a separate
  follow-up evaluated against the governing resource-intelligence design
