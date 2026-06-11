## MODIFIED Requirements

### Requirement: Resource monitoring remains a distinct observability sub-concern
Runtime resource collection and resource-domain interpretation SHALL remain a
distinct observability sub-concern while integrating with the repo-owned
metadata backbone and shared training lifecycle context.

#### Scenario: Resource facts become durable
- **WHEN** resource-domain code produces accepted observations, structural facts, profiles, or accounting statements
- **THEN** those facts MUST be able to flow through the metadata runtime/backend boundary
- **AND** metadata MUST own validation, identity, relationship, storage, and projection behavior
- **AND** metadata MUST NOT take ownership of resource collection or interpretation decisions

#### Scenario: Resource monitoring contributes to reports
- **WHEN** run reports or summaries include resource information
- **THEN** those views MUST consume a stable resource-run interface over accepted facts
- **AND** report generation MUST NOT require JSONL to remain the canonical resource database

#### Scenario: Training provides lifecycle context
- **WHEN** training code knows that a relevant session, phase, step, component, or artifact boundary has occurred
- **THEN** it MAY expose that context through stable observability/resource-monitor seams
- **AND** it MUST NOT be required to own collector, persistence, profile, accounting, or report formatting behavior

#### Scenario: Non-resource diagnostics do not inherit resource semantics
- **WHEN** the system formats startup trainability diagnostics or tracker metrics
- **THEN** those concerns MUST NOT be forced into the resource-intelligence runtime/config model solely to share output plumbing
