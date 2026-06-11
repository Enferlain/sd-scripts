## Why

The current resource system tracks when runtime phases happen and how visible
resource counters change across those phases. It also has useful startup
component estimates. What it does not yet have is phase-level resource
accounting: a single view that explains, for each defined runtime window, what
known code-owned work is using resources and how much of the observed state can
be accounted for.

The previous provenance framing was too separate from the existing system. This
change resets the direction around the resource monitor and runtime phase
vocabulary that already exist.

## What Changes

- Define phase resource accounting as the next layer of the existing resource
  monitor/report system.
- Treat phases/events as the primary runtime identity for resource accounting.
- Introduce declared resource-owner scopes for code paths that know what work is
  happening, such as model loading, cache materialization, optimizer setup,
  accelerator preparation, and checkpoint serialization.
- Distinguish structural accounting of known object sizes from scope accounting
  of resource movement during declared operations.
- Treat incomplete accounting as an accounting coverage gap, not as a resource
  owner.
- Keep `resource_monitor` events as the source of observed counters while adding
  accounting records that attach to the same phase/event timeline.
- Upgrade report output toward one phase resource accounting view instead of a
  separate attribution/provenance section.
- Keep metadata downstream: it may store accounting facts later, but it should
  not infer ownership.

## Capabilities

### New Capabilities

- `phase-resource-accounting`: define how runtime phases, resource snapshots,
  and declared owner scopes combine into phase-level resource accounting.

### Modified Capabilities

- `resource-visibility`: resource reports should evolve from phase counters plus
  estimates toward a single phase accounting view.

## Impact

- `library/logging/resource_monitor/`
- `library/logging/reports.py`
- `library/logging/runtime_trace.py`
- `library/logging/phase_tags.py`
- `library/logging/summaries.py`
- `docs_design/resource_visibility_capability_map.md`
- future metadata filing only after the runtime/report contract is useful
