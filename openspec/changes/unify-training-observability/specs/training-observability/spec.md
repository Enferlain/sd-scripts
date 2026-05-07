## ADDED Requirements

### Requirement: Training observability uses repo-owned structured signals
The training system SHALL expose repo-owned structured observability signals for human console summaries, tracker metrics, and report-generation paths instead of requiring each sink to reconstruct those views independently from raw runtime objects.

#### Scenario: Startup diagnostics are consumed by more than one sink
- **WHEN** startup diagnostics are emitted for a training run
- **THEN** the system MUST produce structured summary data before sink-specific formatting
- **AND** console output and report-generation paths MUST be able to consume that shared structured summary data

#### Scenario: Training layers do not define sink-local formats
- **WHEN** trainer, mode, or phase code emits observability facts
- **THEN** those layers MUST NOT be forced to own final sink-specific formatting rules for every consumer

### Requirement: Observability ownership is organized by concern
The training observability architecture SHALL define distinct ownership buckets for console output, metrics, summaries, reports, and resource monitoring instead of treating all emitted signals as one undifferentiated logging surface.

#### Scenario: Routing a human lifecycle message
- **WHEN** the system emits a human-facing lifecycle message
- **THEN** that message MUST route through an observability concern that owns console-oriented behavior such as rank-aware and progress-safe output

#### Scenario: Routing tracker metrics
- **WHEN** the system emits per-step scalar metrics
- **THEN** that emission MUST route through an observability concern that owns metric sink/back-end behavior rather than through console/report-only paths

### Requirement: Console observability preserves the repo logging stack and presents one user-facing surface
The training observability architecture SHALL preserve stdlib `logging` with Rich-backed console handling as the canonical repo console stack, and it SHALL treat early config/setup logging plus runtime/trainer logging as one user-facing console surface.

#### Scenario: Preserving the logging stack
- **WHEN** the repo implements the observability architecture
- **THEN** it MUST NOT require replacing the current stdlib `logging` plus Rich-backed console stack with a different logging framework as part of this change
- **AND** timestamped, level-aware, source-rich runtime output MUST remain compatible with the canonical console experience

#### Scenario: Early and runtime logs share one console surface
- **WHEN** the system emits repo-owned logs before trainer setup or during runtime
- **THEN** those messages MUST be treated as part of one user-facing console surface
- **AND** the architecture MUST aim for compatible presentation rather than treating pre-trainer and runtime logging as unrelated styles

#### Scenario: Canonical startup and lifecycle presentation
- **WHEN** the system renders canonical startup diagnostics or lifecycle messages
- **THEN** dense startup information MUST be representable as structured sectioned blocks
- **AND** standalone lifecycle messages MUST be representable as concise tagged lines
- **AND** ad hoc direct `accelerator.print(...)` formatting MUST NOT remain the default long-term path for canonical training UX

### Requirement: Observability supports swappable sinks and future repo-owned back-ends
The observability architecture SHALL support multiple sinks and future repo-owned experiment/dashboard back-ends without requiring training orchestration code to change its semantic event production for each backend.

#### Scenario: Existing tracker integration remains one backend
- **WHEN** the training system emits per-step metrics
- **THEN** the current tracker integration MAY remain as one backend
- **AND** the emitted metric facts MUST be expressible through a repo-owned observability seam rather than being defined only by one third-party tracker implementation

#### Scenario: Adding a future repo-owned backend
- **WHEN** a future repo-owned experiment or dashboard backend is introduced
- **THEN** it MUST be able to consume the same repo-owned observability facts without redefining training-layer behavior for each signal category

#### Scenario: Existing tracker backend remains usable
- **WHEN** the repo routes metrics to the current Accelerate-backed tracker system
- **THEN** that integration MUST remain representable as one observability sink/backend rather than the only definition of tracker semantics

#### Scenario: Backend sink stays narrower than the repo-facing observer
- **WHEN** the repo defines a swappable backend sink for trackers or future experiment backends
- **THEN** that sink MUST support run initialization, metric persistence, and run finalization as the default minimum contract
- **AND** the sink MUST NOT be required by default to own console output, startup-summary rendering, or resource-monitor runtime behavior

### Requirement: Adapter diagnostics derive from repo-owned adapter provenance
Adapter-oriented diagnostics SHALL derive from repo-owned adapter provenance such as resolved target and trainable-ref contracts rather than depending on method-specific runtime internals or per-method display code as the default path.

#### Scenario: Building adapter component diagnostics
- **WHEN** adapter training emits startup diagnostics or report summaries
- **THEN** the adapter breakdown MUST be derivable from repo-owned adapter provenance
- **AND** it MUST NOT require hardcoded branches over concrete adapter method internals as the default shared behavior

#### Scenario: Preserving internal keys and public labels
- **WHEN** adapter diagnostics group trainable state by component
- **THEN** the observability path MUST preserve both the internal component key used by training code and the public component label carried by adapter provenance

#### Scenario: Human-facing diagnostics prefer public labels
- **WHEN** the system renders human-facing adapter or startup component diagnostics
- **THEN** public component labels MUST be the default display labels
- **AND** normalized internal component keys MUST remain available as structured observability data rather than becoming the default user-facing names

### Requirement: Resource monitoring remains a distinct observability sub-concern
Runtime resource sampling, phase summaries, and resource-event persistence SHALL remain a distinct observability sub-concern even as the broader observability architecture is unified.

#### Scenario: Resource monitoring contributes to reports
- **WHEN** run reports or summaries include resource information
- **THEN** those views MUST be able to consume resource-monitor outputs through a stable interface
- **AND** the resource-monitor implementation MAY continue to own its runtime sampling and persistence behavior

#### Scenario: Non-resource diagnostics do not inherit resource-monitor semantics
- **WHEN** the system formats startup trainability diagnostics or tracker metrics
- **THEN** those concerns MUST NOT be forced into the resource-monitor runtime/config model solely to share output plumbing

### Requirement: Training and observability ownership stay separate
The observability architecture SHALL keep training-side ownership of timing and fact production separate from observability-side ownership of formatting, routing, buffering, and persistence.

#### Scenario: Training phase emits a fact
- **WHEN** a training phase, mode, or strategy knows that an event or summary fact has occurred
- **THEN** that layer MAY decide when to emit it
- **AND** observability-owned code MUST decide how the emitted fact is formatted, routed, or persisted

#### Scenario: Observability does not become orchestration owner
- **WHEN** observability code is introduced for training signals
- **THEN** it MUST NOT take ownership of training orchestration decisions that belong to phases, modes, or strategies

### Requirement: First-pass implementation lands in the target observability modules
The first-pass implementation SHALL move the affected observability concerns directly into the target `library/logging/` ownership modules rather than introducing temporary stopgap modules for concerns whose destination is already known.

#### Scenario: Startup diagnostics move to summaries/console
- **WHEN** startup diagnostics are refactored in the first pass
- **THEN** their structured rendering and console ownership MUST land in `library/logging/summaries.py` and `library/logging/console.py` rather than in new transitional training-layer helpers

#### Scenario: Tracker and report paths move to their target modules
- **WHEN** tracker emission and report composition are refactored in the first pass
- **THEN** tracker routing MUST move toward `library/logging/metrics.py` and backend adapters under `library/logging/sinks/`
- **AND** report composition MUST move toward `library/logging/reports.py`
- **AND** `library/logging/resource_monitor.py` MAY remain in place as its own subsystem while integrating through the new observability seams
