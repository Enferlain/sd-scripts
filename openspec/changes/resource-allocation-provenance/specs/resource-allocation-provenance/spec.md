## ADDED Requirements

### Requirement: Resource attribution distinguishes observations from claims
The system SHALL distinguish between raw resource observations,
best-effort attribution, and stronger causal explanations when presenting
resource ownership information.

#### Scenario: Presenting an attribution result
- **WHEN** a report, debug payload, or future metadata surface presents a
  resource ownership claim
- **THEN** it MUST preserve the evidence level of that claim
- **AND** it MUST NOT present a best-effort attribution as if it were a raw
  observed resource fact

### Requirement: Attribution records preserve evidence and caveats
The system SHALL preserve the basis and limitations of a resource attribution
record instead of only presenting a named owner bucket.

#### Scenario: Recording an instrumented attribution window
- **WHEN** the repo records future attribution for a named runtime operation or
  attribution window
- **THEN** the record MUST be able to preserve the triggering tag or operation
  identity, before/after observational context, and candidate owner categories
- **AND** it MUST be able to carry caveats or explanatory notes describing the
  limits of the claim

### Requirement: Unknown remainder is allowed
The system SHALL allow unattributed or unknown remainder rather than forcing all
resource deltas into named owners.

#### Scenario: Evidence is insufficient for full ownership
- **WHEN** the repo cannot honestly attribute all or part of a resource change
  to a supported owner category
- **THEN** the result MUST be able to record that remainder as unknown or
  unattributed
- **AND** it MUST NOT fabricate a named owner solely to make the accounting sum
  to the observed delta
