## Why

The resource monitor can now report richer counters and more honest
interpretation layers, but it still cannot answer which runtime owner or helper
caused a memory jump without overclaiming certainty. This is the right time to
define a provenance/causality model before ad hoc report fields or debug hooks
turn aggregate counters into fake attribution.

## What Changes

- Define a repo-owned resource-allocation provenance capability that separates
  raw observational facts from best-effort attribution and stronger causal
  claims.
- Establish a future attribution model based on explicit instrumentation around
  known runtime operations instead of inferring ownership from aggregate deltas
  alone.
- Keep `ResourceMonitorFacts` and the current live resource event schema
  observational; provenance records remain a distinct debug/report-oriented
  surface.
- Define how future provenance surfaces should report evidence level, caveats,
  and unattributed/unknown remainder instead of forcing every delta into a named
  owner bucket.
- Clarify that startup component estimates, runtime trace tags, and deep-mode
  allocator counters are supporting evidence for attribution work, not
  interchangeable with live ownership facts.

## Capabilities

### New Capabilities
- `resource-allocation-provenance`: define the future provenance and causality
  model for resource ownership claims, attribution windows, evidence levels, and
  unknown/unattributed outcomes.

### Modified Capabilities
- `resource-visibility`: resource visibility must continue to distinguish live
  observational facts from report/debug interpretations and any future
  attribution or provenance claims.

## Impact

- Design references under `docs_design/`, especially
  `resource_visibility_capability_map.md`
- `library/logging/resource_monitor/`
- `library/logging/reports.py`
- `library/logging/runtime_trace.py`
- `library/logging/summaries.py`
- future observability metadata/debug payload contracts if attribution records
  are persisted beyond report-only output
