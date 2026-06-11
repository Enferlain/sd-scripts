## Why

The current resource monitor captures useful training-run counters, but its
resource event dictionary, JSONL artifact, metadata facts, and report parsing
are evolving as parallel surfaces. That shape will become increasingly
conflicted and misleading as the project adds richer profiling and accounting.

The repository needs one coherent resource-intelligence system designed around
the actual goal: record what resources a training run used, preserve what is
known about resource-bearing runtime state, build reusable run profiles, and
account for resource use without confusing observation, estimation, and
attribution.

## What Changes

- Define one end-to-end resource-intelligence model covering:
  - measured resource observations
  - known structural resource facts
  - derived run and workload profiles
  - resource-accounting statements and unresolved accounting gaps
- Keep the current small trainer-facing resource-monitor lifecycle API as the
  runtime integration spine while making its internal responsibilities
  extensible.
- Establish resource-domain runtime code as the owner of collection, live
  state, derivation, and accounting decisions.
- Establish `library/metadata/` as the owner of accepted durable fact schemas,
  validation, identities, relationships, storage, and projections.
- Replace parallel hand-authored JSONL and metadata schemas with one typed fact
  flow; JSONL and other artifacts become projections/exports of accepted facts.
- Define queryable resource-run data as the input to reports, profiles,
  comparisons, accounting, and future recommendations instead of requiring
  reports to parse a separate canonical JSONL schema.
- Preserve explicit semantics for measured, estimated, derived, and accounted
  values so reports never present inference as raw observation.
- Design collection cost, sampling, retention, and failure behavior as part of
  the system contract rather than adding counters without operational policy.
- Supersede the current `phase-resource-accounting` direction as the governing
  resource-system design; phase accounting may remain one view within the
  broader system rather than the architecture itself.

## Capabilities

### New Capabilities

- `resource-intelligence`: Defines the complete resource fact model, metadata
  integration, profiling, accounting, querying, and export behavior.

### Modified Capabilities

- `resource-visibility`: Expands resource visibility from additive live fields
  and report-local summaries into one canonical typed resource-fact flow with
  explicit semantic and durability rules.
- `training-observability`: Requires training observability to expose stable
  lifecycle context for resource intelligence without making logging or runtime
  tracing own resource interpretation.

## Impact

- `library/logging/resource_monitor/`
- `library/logging/reports.py`
- `library/logging/runtime_trace.py`
- `library/logging/phase_tags.py`
- `library/logging/summaries.py`
- `library/metadata/`
- `library/training/runners/trainer.py`
- `library/training/phases/`
- resource-monitor and benchmark-report configuration
- resource, metadata, report, and training-observability tests
- existing resource JSONL consumers and artifacts
