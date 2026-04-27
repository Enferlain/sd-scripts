## MODIFIED Requirements

### Requirement: Repo-owned adapter target provenance for module-resolved methods
Module-resolved adapter runtimes SHALL receive repo-owned target provenance
that ties each resolved target back to optimization-owned policy decisions and
the shared optimization target-ref model.

#### Scenario: Passing module-resolved targets into adapter runtime
- **WHEN** optimization constructs the adapter build request
- **THEN** each resolved target MUST include the concrete module object and
  stable provenance needed to identify its component and path within the model
- **AND** each module-resolved target MUST expose or wrap a shared optimization
  target ref with kind `module`
- **AND** the shared target ref MUST include the target module type

#### Scenario: Preserving provenance through runtime realization
- **WHEN** an adapter runtime realizes adapter state against resolved targets
- **THEN** it MUST retain enough provenance to tie returned trainable parameter
  refs back to the original resolved module targets
- **AND** repo-owned adapter trainable refs MUST preserve source target
  provenance when the adapter runtime can identify the source target

## ADDED Requirements

### Requirement: Adapter target refs remain compatible during migration
Adapter runtime callers SHALL be able to consume existing adapter target fields
while shared target refs are introduced.

#### Scenario: Existing adapter runtime reads target fields
- **WHEN** an adapter runtime reads `component`, `component_key`, `path`, or
  `module` from an adapter resolved target
- **THEN** those fields MUST remain available during the shared target-ref
  migration
- **AND** new target-ref-backed behavior MUST NOT require adapter runtimes to
  inspect optimization grouping internals
