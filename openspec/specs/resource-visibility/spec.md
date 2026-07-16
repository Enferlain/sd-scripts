# resource-visibility Specification

## Purpose
TBD - created by archiving change expand-resource-visibility. Update Purpose after archive.
## Requirements
### Requirement: Resource monitor records broader host-memory facts
The resource-intelligence observability surface SHALL record useful host-memory
facts when those facts have a defined question, reliable collection behavior,
and explicit retention policy.

#### Scenario: Capturing host-memory facts in accepted resource observations
- **WHEN** the resource monitor captures supported host-memory facts
- **THEN** those facts MUST be representable in the canonical typed resource-fact flow
- **AND** metadata storage, JSONL projections, and resource-run views MUST derive from that accepted representation
- **AND** the system MUST NOT add host-memory fields solely because a backend exposes them

### Requirement: Resource visibility can grow beyond aggregate-only GPU views
The resource-intelligence surface SHALL support aggregate and richer scoped GPU
views while preserving their measurement source and scope.

#### Scenario: Preserving aggregate and richer readings
- **WHEN** the system records aggregate and per-device GPU readings
- **THEN** each reading MUST preserve its device/resource scope and measurement source
- **AND** aggregate fields MAY remain available as compatibility projections
- **AND** richer fields MUST NOT be interpreted as ownership solely from their device scope

### Requirement: Report/debug summaries remain distinct from live resource facts
The resource-intelligence system SHALL distinguish live observations, durable
derived profiles, accounting statements, and presentation-only report output.

#### Scenario: Derived summary becomes a reusable profile
- **WHEN** a derived resource summary is useful for comparison, regression detection, capacity planning, or recommendations
- **THEN** it MUST be representable as a versioned resource profile
- **AND** it MUST NOT exist only as report formatting

#### Scenario: Presentation-only summary remains report-local
- **WHEN** a resource summary is only explanatory formatting over accepted facts, profiles, or accounting statements
- **THEN** the system MAY keep that summary in report/debug output
- **AND** it MUST NOT require a new canonical resource fact type

