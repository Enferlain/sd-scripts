## MODIFIED Requirements

### Requirement: Report/debug summaries remain distinct from live resource facts

The observability system SHALL distinguish between always-on live resource facts
and richer report/debug summaries derived from those facts.

#### Scenario: Derived summary remains report-local
- **WHEN** a resource summary is explanatory, presentation-oriented, or
  accounting-oriented
- **THEN** the system MUST be able to keep that summary in report/debug surfaces
  without requiring it to become a first-class live event field

#### Scenario: Accounting does not rewrite observed counters
- **WHEN** a report adds resource accounting for a phase
- **THEN** the report MUST preserve the original observed resource counters
- **AND** accounting rows MUST be presented as accounting output, not as raw
  observed counters
