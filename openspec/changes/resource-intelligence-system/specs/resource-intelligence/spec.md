## ADDED Requirements

### Requirement: Resource intelligence preserves fact semantics
The system SHALL represent measured observations, structural facts, derived
profiles, and accounting statements as distinct resource fact classes.

#### Scenario: Consumer reads resource values with the same unit
- **WHEN** a consumer reads measured, estimated, derived, or accounted memory values
- **THEN** each value MUST preserve its semantic class
- **AND** the consumer MUST be able to distinguish those values without inferring semantics from field names

### Requirement: Resource facts use the metadata backbone
The system SHALL route accepted durable resource facts through the repo-owned
metadata schema, validation, identity, relationship, backend, storage, and
projection boundaries.

#### Scenario: Runtime produces a durable resource fact
- **WHEN** resource-domain runtime code produces a resource fact selected for durable retention
- **THEN** the fact MUST be representable as an accepted typed metadata item
- **AND** durable ingestion MUST remain behind the metadata runtime/backend boundary
- **AND** resource-domain code MUST NOT call metadata backends or storage directly

#### Scenario: Resource fact requires domain interpretation
- **WHEN** a profile or accounting fact requires resource-domain interpretation
- **THEN** resource-domain code MUST produce that interpretation
- **AND** metadata code MUST NOT infer the interpretation from raw counters

### Requirement: Resource telemetry ingestion is bounded and observable
The system SHALL support bounded ingestion behavior for sampled or
high-frequency resource observations.

#### Scenario: Sample production exceeds ingestion capacity
- **WHEN** sampled resource observations are produced faster than they can be ingested
- **THEN** the system MUST apply a declared buffering, batching, retention, or drop policy
- **AND** dropped or degraded telemetry MUST itself be observable
- **AND** resource ingestion MUST NOT block training indefinitely

### Requirement: Resource facts preserve relational identity
The system SHALL preserve sufficient identity and relationships to associate
resource facts with their relevant runtime and structural scopes.

#### Scenario: Recording a device observation during training
- **WHEN** the system records a device-scoped observation
- **THEN** the fact MUST be relatable to the run and device
- **AND** it MUST preserve available process, rank, phase, step, host, and collector context without forcing unavailable identities

#### Scenario: Recording a structural component fact
- **WHEN** the system records known component or optimization-state size
- **THEN** the fact MUST be relatable to the owning component or group
- **AND** it MUST remain distinguishable from a live process or device observation

### Requirement: Co-collected observations preserve frame identity
The system SHALL preserve shared collection and runtime context for resource
measurements collected at the same boundary without conflating their individual
measurement semantics.

#### Scenario: Collector produces several measurements together
- **WHEN** one lifecycle boundary or sampler collection produces several resource measurements
- **THEN** the measurements MUST be retainable as one observation frame or equivalent accepted batch
- **AND** each measurement MUST remain individually identifiable by resource kind, measurement kind, source, scope, value, and unit
- **AND** profiles and accounting statements MUST be able to reference the individual measurements they use
- **AND** ingestion and retention policy MUST be able to operate on the shared frame without requiring duplicated context for every measurement

#### Scenario: Resolving measurements from a stored frame
- **WHEN** a resource-run view resolves the measurements contained by an accepted observation frame
- **THEN** it MUST verify that the frame belongs to the viewed run
- **AND** it MUST preserve the frame-declared measurement-identifier order independently of backend edge iteration order

### Requirement: Resource profiles are durable derived products
The system SHALL support versioned resource profiles that describe reusable run
or workload resource behavior.

#### Scenario: Building a run resource profile
- **WHEN** resource-domain code derives peaks, ranges, phase signatures, or comparison features from accepted facts
- **THEN** the resulting profile MUST preserve its derivation version
- **AND** it MUST retain links or references to the source facts or source run
- **AND** metadata validation MUST reject invalid required source relationships
- **AND** it MUST be storable and queryable independently of report formatting

### Requirement: Resource accounting is evidence-constrained
The system SHALL support first-class resource accounting statements without
presenting unsupported causality or ownership claims.

#### Scenario: Accounting for a known structural resource
- **WHEN** the system accounts for resource use using a known structural fact
- **THEN** the accounting statement MUST identify its basis as structural
- **AND** it MUST identify the owner, resource scope, quantity, and validity boundary
- **AND** it MUST preserve its derivation method/version and supporting source-fact relationships

#### Scenario: Accounting remains incomplete
- **WHEN** accepted accounting statements do not explain the relevant observed resource quantity
- **THEN** the system MUST preserve an explicit accounting gap
- **AND** the gap MUST NOT be represented as a resource owner
- **AND** reports and exports MUST render the gap distinguishably from accounted owners

#### Scenario: Operation-local resource movement is observed
- **WHEN** resource movement is measured during a known operation
- **THEN** an accounting statement MUST NOT claim persistent ownership solely from that operation-local observation

#### Scenario: Runtime declares a known owner scope
- **WHEN** runtime code declares a resource-relevant owner or operation scope
- **THEN** the scope MAY provide bounded operation/window evidence
- **AND** it MUST remain distinguishable from structural accounting
- **AND** heavier scope diagnostics MUST require explicit configuration or a bounded diagnostic window

### Requirement: Reports and analyses consume a resource-run view
The system SHALL provide a queryable resource-run view that preserves original
facts, profiles, and accounting statements for reports and analyses.

#### Scenario: Generating a resource report
- **WHEN** a report requests resource information for a run
- **THEN** it MUST be able to consume the resource-run view without parsing a canonical JSONL artifact
- **AND** the resource-run view MUST consume metadata-owned public snapshot, query, or projection APIs rather than raw storage
- **AND** it MUST be able to distinguish observed facts from profiles and accounting statements

#### Scenario: Metadata and compatibility JSONL are both available
- **WHEN** a resource-run view and a compatibility JSONL artifact both contain resource information
- **THEN** report generation MUST prefer the accepted facts exposed by the resource-run view
- **AND** JSONL parsing MAY remain only as a declared fallback when no resource-run view or projected observation frames are available
- **AND** the selected input source MUST remain observable in the report payload during migration

#### Scenario: Resolving declared source evidence
- **WHEN** a profile or accounting record declares accepted source-fact references
- **THEN** the resource-run view MUST resolve available referenced records through metadata relationships
- **AND** it MUST NOT discard evidence solely because it belongs to another run or is an artifact or other non-resource metadata record
- **AND** resolved evidence MUST NOT be treated as a resource fact owned by the viewed run

### Requirement: Resource artifacts are projections
The system SHALL treat JSONL and other resource artifacts as projections or
exports of accepted resource facts rather than independent canonical schemas.

#### Scenario: Exporting resource JSONL
- **WHEN** the system writes a resource JSONL artifact
- **THEN** the artifact MUST be generated from accepted typed resource facts or their canonical projection
- **AND** the artifact MUST NOT require a separately authored resource schema

#### Scenario: Canonical frame production succeeds
- **WHEN** an identified resource-monitor boundary produces an accepted observation frame
- **THEN** metadata ingestion MUST file the canonical frame without also filing the bundled compatibility fact
- **AND** a compatibility fact MAY remain only as a fallback when no canonical frame can be produced

#### Scenario: Producing any resource export
- **WHEN** the system shapes a JSONL, report, profile, or accounting export
- **THEN** metadata projection code MUST own the export schema
- **AND** resource-domain runtime code MUST NOT independently author that schema

#### Scenario: Exporting profiles or accounting results
- **WHEN** the system exports accepted profiles, accounting statements, or accounting gaps
- **THEN** the export MUST preserve its export schema version and source run
- **AND** profile and accounting derivation versions and source-fact references MUST remain intact
- **AND** accounting gaps MUST remain separate from owner-bearing accounting statements
- **AND** projection code MUST NOT derive new profile or accounting claims while formatting the export

#### Scenario: Registering a produced resource artifact
- **WHEN** a resource JSONL, report, profile, or accounting artifact is produced for an identified run
- **THEN** observability metadata MUST register the artifact with its export schema identity
- **AND** the metadata graph MUST link the artifact identity to its source run
- **AND** an unavailable run identity MUST be omitted rather than replaced with a placeholder

### Requirement: Runtime integration remains low-coupling
The system SHALL keep training orchestration insulated from collector,
metadata-storage, profile, accounting, and export implementation details.

#### Scenario: Training lifecycle emits resource context
- **WHEN** trainer or phase code reaches an existing resource lifecycle boundary
- **THEN** it MUST be able to notify the resource-monitor facade through a small stable API
- **AND** it MUST NOT need to call individual collectors, metadata emitters, or artifact writers

### Requirement: Collection capabilities carry operational policy
The system SHALL associate resource collection capabilities with explicit cost,
scope, availability, cadence, and degraded-behavior policy.

#### Scenario: Enabling normal resource monitoring
- **WHEN** a run enables a low-cost resource-monitoring configuration
- **THEN** the system MUST select only collection capabilities declared safe for that configuration
- **AND** unavailable optional collectors MUST degrade without failing the training run

#### Scenario: Enabling expensive diagnostics
- **WHEN** a collector is declared expensive or intrusive
- **THEN** it MUST require explicit diagnostic configuration or a bounded diagnostic window

#### Scenario: Recording a collection cycle
- **WHEN** one lifecycle or sampling boundary coordinates multiple collection capabilities
- **THEN** the retained observations MUST preserve the concrete source or provider for each measurement
- **AND** later lifecycle boundaries MUST NOT re-record stale sampled or diagnostic values as newly co-collected measurements
- **AND** compatibility exports MAY join prior observations only as a projection behavior

#### Scenario: Collector fallback or failure
- **WHEN** a preferred optional collector is unavailable and a fallback collector is used
- **THEN** the retained measurements MUST identify the fallback source and degraded quality
- **AND** the system MUST NOT fail the training run solely because the preferred optional collector is unavailable
- **AND** a collector failure that produces no measurements MUST emit degradation/status evidence rather than an empty observation frame

#### Scenario: Collector budgets and diagnostic windows
- **WHEN** collection exceeds a configured warning budget
- **THEN** the budget breach MUST be recorded as degraded operational evidence unless the collector contract supports hard cancellation
- **AND** side-effecting operations such as CUDA allocator peak resets MUST be represented as explicit diagnostic-window behavior
