## Why

Training-emitted signals are currently useful but fragmented: human console output, tracker metrics, startup diagnostics, benchmark reports, and resource monitoring are spread across different layers with different shapes and ownership rules. The repo is also expected to grow toward a custom dashboard/experiment system, so now is the right time to define a repo-owned observability architecture instead of growing more one-off reporting paths.

## What Changes

- Define a repo-owned training observability capability that separates signal production from presentation and sink/back-end routing.
- Establish a concrete observability package direction inside `library/logging/` with distinct ownership for console output, metrics, summaries, reports, and resource monitoring.
- Preserve the current stdlib `logging` + Rich console stack, including timestamped and source-rich runtime output, instead of changing logging frameworks.
- Introduce shared observability contracts for startup summaries, tracker metrics, and run/report payloads so console output, reports, and future custom dashboards can consume the same structured data.
- Treat early config/setup logs and runtime/trainer logs as one user-facing console surface and align their presentation rather than leaving them as unrelated styles.
- Keep resource monitoring as a distinct runtime/resource concern while aligning it with the broader observability architecture through stable event and summary interfaces.
- Define adapter diagnostics as an observability concern that derives from repo-owned adapter provenance rather than ad hoc method-specific display code.
- Establish migration guidance for existing logging paths so trainer/mode/phase code emits structured facts while observability-owned code formats and routes them, using sectioned startup blocks plus tagged lifecycle lines as the preferred human-facing console style.

## Capabilities

### New Capabilities
- `training-observability`: Repo-owned observability contracts and sinks for training console output, tracker metrics, startup diagnostics, reports, and future dashboard-oriented integrations.

### Modified Capabilities

## Impact

- Affected code: `library/logging/`, `library/training/`, selected mode/phase/trainer seams, and adapter provenance helpers.
- Affected systems: console lifecycle output, tracker emission, startup diagnostics, benchmark reports, and future custom dashboard/back-end work.
- Expected package direction: observability-specific responsibilities should converge under `library/logging/` instead of staying split between training helpers and sink-specific files.
- Expected console direction: one repo-owned console surface using stdlib `logging` + Rich, with sectioned startup summaries and tagged lifecycle lines for canonical training UX.
- Dependencies: existing Accelerate tracker integration remains in place initially; future custom observability back-ends should be able to plug into the new contracts without redefining training-layer behavior.
