## Why

The current resource monitor is good at aggregate session and phase counters,
but it is still weak at per-device visibility, host-memory breadth, and
resource surfaces that can explain what changed without relying on console-only
output. We now have a concrete capability map, so this is the right time to
turn that investigation into a staged implementation plan.

## What Changes

- Expand the resource-monitor event/report surface beyond aggregate-only GPU and
  host-memory readings.
- Add structured support for richer live resource facts where the measurements
  are already cheap and reliable.
- Preserve the distinction between live resource facts, report/debug summaries,
  and future provenance/causality work that needs a separate design step.
- Start with a small live-capable slice so the broader resource-visibility work
  moves from notes into code.

## Capabilities

### New Capabilities
- `resource-visibility`: define the live/report/debug resource-observability
  contract for per-device readings, host-memory breadth, and richer report
  surfaces.

### Modified Capabilities
- `training-observability`: resource-monitor observability/reporting behavior
  will grow to include richer resource facts while keeping console concerns
  separate from non-console visibility work.

## Impact

- `library/logging/resource_monitor.py`
- `library/logging/reports.py`
- `library/logging/summaries.py`
- `library/metadata/dataclasses/observability.py`
- `library/metadata/emitters/observability.py`
- resource-monitor/report tests
- design references under `docs_design/`
