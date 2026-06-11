## ADDED Requirements

### Requirement: Phase resource accounting uses runtime phases as identity

The system SHALL use existing runtime phases/events as the primary identity for
resource accounting.

#### Scenario: Reporting phase accounting
- **WHEN** a report presents resource accounting for a run
- **THEN** accounting entries MUST be tied to phase/event identifiers already
  used by runtime trace and resource monitor output
- **AND** the report MUST NOT introduce a separate user-facing attribution
  timeline for the same work

### Requirement: Ownership comes from declared runtime scopes

The system SHALL prefer explicit runtime owner scopes over post-hoc owner guesses
from aggregate counter deltas.

#### Scenario: Known resource-relevant code path runs
- **WHEN** runtime code performs known resource-relevant work such as model
  loading, cache materialization, optimizer setup, accelerator preparation, or
  checkpoint save
- **THEN** that work SHOULD be representable as a declared owner scope attached
  to the active phase/event context

### Requirement: Accounting output distinguishes owners from accounting gaps

The system SHALL allow phase resource accounting to show what is accounted for
and where accounting coverage is incomplete.

#### Scenario: Phase accounting is incomplete
- **WHEN** observed resource changes exceed what known owner scopes can account
  for
- **THEN** the phase accounting output MUST preserve the difference as an
  accounting gap
- **AND** it MUST NOT force that amount into a named owner
- **AND** an accounting gap MUST NOT be represented as an owner label

### Requirement: Structural and scope accounting remain distinct

The system SHALL distinguish structural accounting of known object sizes from
scope accounting of observed resource movement during declared operations.

#### Scenario: Declared owner scope records resource movement
- **WHEN** an owner scope records a resource change during a known operation
- **THEN** the accounting output MUST identify that result as scope accounting
- **AND** it MUST NOT treat the result as proof that all persistent resource
  state after the operation belongs to that owner

### Requirement: Resource monitor owns accounting scopes

The resource monitor SHALL own the runtime API for declared resource-owner
scopes.

#### Scenario: Runtime code declares known resource-relevant work
- **WHEN** runtime code enters a declared resource-owner scope
- **THEN** the resource monitor MUST be able to emit structured accounting-scope
  start/end events tied to the active phase/event context
- **AND** runtime trace MUST remain responsible for timing rather than ownership
  accounting

### Requirement: Default resource monitoring stays low overhead

The system SHALL keep normal resource monitoring low overhead.

#### Scenario: Running without deeper diagnostics
- **WHEN** a normal training run uses resource monitoring
- **THEN** phase timing and resource counter collection MUST remain the default
  behavior
- **AND** heavier ownership diagnostics MUST require explicit configuration or a
  bounded diagnostic path
