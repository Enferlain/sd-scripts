## MODIFIED Requirements

### Requirement: Report/debug summaries remain distinct from live resource facts
The observability system SHALL distinguish between always-on live resource facts
and richer report/debug summaries derived from those facts, including future
attribution or provenance claims.

#### Scenario: Derived summary remains report-local
- **WHEN** a resource summary is primarily explanatory, presentation-oriented,
  or attribution-oriented
- **THEN** the system MUST be able to keep that summary in report/debug
  surfaces without requiring it to become a first-class live event field

#### Scenario: Attribution does not change observational facts
- **WHEN** a future report or debug surface adds attribution or provenance
  claims about a resource change
- **THEN** those claims MUST remain distinct from the live observational event
  schema
- **AND** the system MUST preserve the original observational counters even when
  richer attribution output is present
