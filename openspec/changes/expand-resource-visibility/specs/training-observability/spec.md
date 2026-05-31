## ADDED Requirements

### Requirement: Resource-monitor report surfaces preserve additive live schema growth
Training observability SHALL allow the resource-monitor event/report path to
grow with richer additive live fields without forcing console behavior or phase
ownership changes to lead that work.

#### Scenario: Non-console resource field expansion
- **WHEN** the repo adds richer live resource facts to resource-monitor events
- **THEN** the benchmark report and analytics snapshot surfaces MUST be able to
  preserve those fields or derived summaries where appropriate
- **AND** that expansion MUST NOT depend on changing console phase-summary
  behavior as the primary implementation path
