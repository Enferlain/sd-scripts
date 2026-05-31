## ADDED Requirements

### Requirement: Resource monitor records broader host-memory facts
The resource-monitor observability surface SHALL record host-memory facts beyond
RSS when those facts are already cheap and reliable to collect from the current
process runtime.

#### Scenario: Capturing host-memory facts in live events
- **WHEN** the resource monitor emits a session, phase, or step event
- **THEN** the live event schema MUST be able to carry CPU virtual memory data
- **AND** that addition MUST remain aligned across raw event payloads,
  metadata-facing observability facts, and report payloads

### Requirement: Resource visibility can grow beyond aggregate-only GPU views
The resource-observability surface SHALL support richer GPU visibility than one
aggregate number when the monitor already has enough local information to do so.

#### Scenario: Preserving aggregate and richer readings
- **WHEN** the system adds richer resource fields such as per-device readings
- **THEN** it MUST be able to preserve the existing aggregate-oriented view for
  compatibility and summary purposes
- **AND** richer fields MUST be additive rather than requiring an immediate
  replacement of the existing aggregate report/event surfaces

### Requirement: Report/debug summaries remain distinct from live resource facts
The observability system SHALL distinguish between always-on live resource facts
and richer report/debug summaries derived from those facts.

#### Scenario: Derived summary remains report-local
- **WHEN** a resource summary is primarily explanatory or presentation-oriented
- **THEN** the system MUST be able to keep that summary in report/debug
  surfaces without requiring it to become a first-class live event field
