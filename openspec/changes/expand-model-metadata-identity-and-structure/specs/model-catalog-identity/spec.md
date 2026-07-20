## ADDED Requirements

### Requirement: Catalog identity separates logical models from representations and realizations
The metadata system SHALL represent portable catalog lineage, immutable model revision, source representation, source selection, run realization, and produced artifact identities as distinct accepted identities with explicit relationships.

#### Scenario: Same artifact is renamed
- **WHEN** two local paths have the same verified complete-file digest
- **THEN** they MUST resolve to the same source-representation identity under the same identity-policy version
- **AND** the paths MUST remain aliases or observations rather than becoming the identity

#### Scenario: One model has multiple representations
- **WHEN** a checkpoint and a converted repository representation are related by accepted conversion evidence
- **THEN** the system MUST preserve distinct representation identities
- **AND** it MUST relate them to the applicable catalog model/revision without claiming byte equality

#### Scenario: One representation contains multiple selectable states
- **WHEN** the same checkpoint representation can supply distinct selected states such as EMA and non-EMA weights
- **THEN** both loads MUST retain the same source-representation identity
- **AND** each selected state MUST receive distinct versioned source-selection evidence and resulting realization composition
- **AND** exact representation equality MUST NOT by itself establish selected-state or revision equality

#### Scenario: Different representations contain the same selected state
- **WHEN** two source representations have different exact digests and an accepted state policy relates their selected states
- **THEN** the source-representation identities MUST remain distinct
- **AND** the source selections MAY relate to the same immutable model revision only under the declared policy/evidence

#### Scenario: Same representation receives conflicting publisher claims
- **WHEN** two locations or publishers expose the same exact representation bytes with incompatible catalog-lineage claims
- **THEN** the system MUST preserve one content-addressed source-representation identity and separate observation/claim records
- **AND** byte equality MUST NOT resolve the logical-lineage conflict automatically

#### Scenario: Model is loaded in another run
- **WHEN** a previously cataloged source model is loaded in a new run
- **THEN** the new run realization MUST have a new run-scoped identity
- **AND** it MUST reference the existing portable catalog/revision/source identities

### Requirement: Portable identity is carried or deterministically derived at the correct identity layer
The catalog SHALL preserve carried portable identity claims with trust evidence and SHALL derive a versioned deterministic source-representation identity for unseen serialized sources that lack matching exact representation evidence.

#### Scenario: Kuro artifact carries portable identity
- **WHEN** a source artifact contains a structurally valid repo-owned portable model/revision identity and issuer/schema evidence
- **THEN** the catalog MUST preserve the portable claim with its authenticated, trusted-local, or unsigned/self-asserted trust class
- **AND** it MUST record the evidence used to accept or limit the claim

#### Scenario: Unsigned carried claim has no corroborating evidence
- **WHEN** a structurally valid carried identity is unsigned/self-asserted and lacks consistent previously accepted strong evidence
- **THEN** the catalog MUST preserve it as a portable claim
- **AND** it MUST NOT independently merge or redirect unrelated representations or logical identities

#### Scenario: Carried identity is invalid or unsupported
- **WHEN** a carried identity has an invalid namespace, identity kind, scheme version, bounded payload, or unsupported issuer schema
- **THEN** the system MUST preserve it only as unverified namespaced source evidence when policy permits
- **AND** it MUST NOT use the claim to select, assign, alias, or merge a catalog identity

#### Scenario: Structurally valid carried identity conflicts with exact evidence
- **WHEN** a structurally valid carried identity is incompatible with accepted exact evidence or lineage kind
- **THEN** the catalog MUST produce explicit conflict evidence or fail resolution
- **AND** structural schema validation MUST NOT be treated as proof of issuer authenticity

#### Scenario: Unseen local file has no portable identity
- **WHEN** a local model file has no accepted portable identity or matching strong evidence
- **THEN** the resolver MUST calculate a streaming cryptographic digest over the complete file
- **AND** it MUST derive and register a deterministic source-representation identity under an explicit scheme/version
- **AND** any provisional catalog model/revision assignment MUST be a separate evidence-backed record and relationship

#### Scenario: Unseen local directory has no portable identity
- **WHEN** a local model directory has no accepted portable identity or matching strong evidence
- **THEN** the resolver MUST build a canonical versioned selected-file manifest using relative paths and complete-file digests
- **AND** the resulting deterministic source-representation identity MUST be independent of the directory's local absolute path
- **AND** any provisional catalog model/revision assignment MUST remain distinct from the manifest identity

#### Scenario: Source selection is resolved
- **WHEN** a loader selects a state, namespace, component subset, variant, or extraction from a source representation
- **THEN** it MUST record a versioned source-selection identity/evidence distinct from the representation identity
- **AND** provisional revision/catalog assignment MUST be rooted in the selection evidence rather than treating the whole container digest as selected-state identity

#### Scenario: Remote repository source is resolved
- **WHEN** a mutable remote reference is used to obtain model files
- **THEN** catalog evidence MUST preserve the resolved immutable revision and selected file identities/digests
- **AND** the mutable branch or tag alone MUST NOT be treated as strong identity evidence

#### Scenario: Cached digest is considered for a local path
- **WHEN** the catalog has a prior digest for a path whose current content is not covered by accepted immutable evidence
- **THEN** path, size, modification time, or file identity MAY be used as cache-validation evidence
- **AND** those attributes MUST NOT be treated as cryptographic identity
- **AND** the resolver MUST rehash when the accepted policy cannot prove the current content matches the cached digest

#### Scenario: Directory manifest is normalized
- **WHEN** a selected directory/repository representation is hashed
- **THEN** its policy MUST define logical path separators, Unicode normalization, case comparison, selected symlink handling, duplicate normalized paths, special-file rejection, and selected-file membership
- **AND** mutation of a selected file or selected-file set during hashing MUST invalidate the attempt rather than produce a mixed-time identity

### Requirement: Catalog registration has one durable authority
The central metadata subsystem SHALL provide one authoritative catalog resolution and registration boundary backed by persistent accepted records and relationships.

#### Scenario: Catalog is reopened in another process
- **WHEN** a persistent catalog is closed and reopened for a later run
- **THEN** the same exact representation evidence under the same policy MUST resolve to the same accepted source-representation identity
- **AND** the same accepted carried or provisional selection assignment evidence MUST resolve to its separately recorded logical identities
- **AND** the run-scoped metadata runtime MUST NOT mint an unrelated identity

#### Scenario: Lookup index is rebuilt
- **WHEN** a catalog lookup index is absent or rebuilt
- **THEN** it MUST be derivable from accepted catalog records and relationships
- **AND** it MUST NOT become a second canonical identity source

#### Scenario: Derived index is inconsistent
- **WHEN** catalog-open or rebuild validation finds an index entry inconsistent with canonical accepted records
- **THEN** the index MUST be rejected, discarded, or rebuilt before it serves resolution
- **AND** the inconsistency MUST NOT rewrite canonical records

#### Scenario: Workflow is explicitly ephemeral
- **WHEN** a caller uses an in-memory catalog without persistent backing
- **THEN** the resulting capability MUST be identified as ephemeral
- **AND** it MUST NOT claim cross-run catalog resolution guarantees

#### Scenario: Persistent catalog initialization fails
- **WHEN** an active persistent catalog cannot be opened or migrated
- **THEN** catalog initialization MUST fail observably
- **AND** it MUST NOT silently fall back to ephemeral assignment

#### Scenario: Same evidence is registered concurrently
- **WHEN** concurrent transactions register the same normalized exact representation evidence and source-representation identity under one policy version
- **THEN** registration MUST be idempotent and return the same accepted mapping

#### Scenario: Conflicting targets are registered concurrently
- **WHEN** concurrent transactions register the same normalized exact representation evidence for different source-representation identities
- **THEN** the persistent authority MUST enforce an atomic uniqueness decision and surface a conflict
- **AND** neither caller MUST resolve successfully to an arbitrary insertion winner

### Requirement: Recognition policy is progressive and versioned
Catalog recognition SHALL record the evidence, strength, decision, and policy version used to reuse, assign, alias, or reject an identity.

#### Scenario: Previously verified exact evidence is available
- **WHEN** an exact digest already has an accepted non-conflicting source-representation mapping
- **THEN** the resolver MUST reuse that representation mapping without recalculating optional expensive fingerprints
- **AND** it MUST resolve source-selection and logical catalog/revision relationships independently

#### Scenario: Only weak evidence matches
- **WHEN** only a name, path, architecture, parameter count, or structural similarity matches an existing entry
- **THEN** the catalog MAY return candidates
- **AND** it MUST NOT automatically merge or alias the entries

#### Scenario: Strong evidence conflicts
- **WHEN** available strong evidence maps incompatibly to more than one accepted identity
- **THEN** resolution MUST fail or produce explicit conflict evidence
- **AND** it MUST NOT choose a result by record insertion order

#### Scenario: Strong-evidence conflict is classified
- **WHEN** the same normalized exact evidence maps to different source-representation identities or one portable logical identity makes incompatible representation/selection/lineage-kind claims
- **THEN** the catalog MUST classify the result as a strong-evidence conflict
- **AND** it MUST retain the evidence and policy version for repair

#### Scenario: Recognition rules change
- **WHEN** matching or normalization rules are revised
- **THEN** new decisions MUST record the new policy version
- **AND** prior accepted mappings MUST remain auditable rather than being silently rewritten

#### Scenario: Conflict is manually resolved
- **WHEN** an authorized catalog repair resolves an alias or conflict
- **THEN** it MUST append actor, reason, policy, and resolution/supersession evidence
- **AND** it MUST NOT edit or delete the conflicting historical facts

### Requirement: Catalog preserves typed relationships, immutable revisions, and lineage
The metadata graph SHALL keep observation/display aliases, identity redirects, `represents`, `derived_from`, and `equivalent_under_policy` relationships distinct while representing a model revision as immutable selected logical state within a catalog lineage.

#### Scenario: Release label names a revision
- **WHEN** a publisher or local producer assigns a version/release name to a model state
- **THEN** the name MUST be recorded as a claim or label attached to the immutable revision
- **AND** the release event/name MUST NOT replace the revision identity

#### Scenario: Saved training output descends from a source model
- **WHEN** training produces a reusable modified model artifact
- **THEN** the output MUST receive a model revision/artifact identity
- **AND** it MUST retain a lineage relationship to the source revision and producing run

#### Scenario: In-memory transformation is not saved
- **WHEN** a run casts, moves, offloads, wraps, or otherwise changes a live model only for execution
- **THEN** the source model/revision identity MUST remain unchanged
- **AND** applicable effects MAY be recorded as run-scoped realization observations

#### Scenario: Alternate representation preserves state semantics
- **WHEN** an accepted conversion policy establishes an alternate serialization of the same revision
- **THEN** the system MUST retain separate representation identity and conversion evidence
- **AND** it MAY relate both representations to the same revision only under that declared policy

#### Scenario: Alias evidence is weak
- **WHEN** an alias or merge candidate is supported only by a name, path, architecture, size, or structural similarity
- **THEN** the catalog MUST retain it as a candidate only
- **AND** it MUST NOT apply an identity redirect, representation relationship, or equivalence assertion automatically

#### Scenario: Modification semantics are uncertain
- **WHEN** a persisted transformation cannot be safely classified as alternate representation or descendant revision
- **THEN** the system MUST retain the transformation and unresolved relation evidence
- **AND** it MUST NOT assert equivalence or unrelatedness automatically

### Requirement: Produced artifacts can carry portable catalog identity
Repo-owned model artifact projections SHALL be capable of carrying portable catalog, revision, representation, and lineage references without making projection keys canonical internal facts.

#### Scenario: Kuro model artifact is exported
- **WHEN** a produced model artifact has accepted portable identity and lineage facts
- **THEN** the Kuro projection MUST be able to render those facts in its versioned external representation
- **AND** another installation MUST be able to submit them as catalog evidence

#### Scenario: Compatibility metadata contains an unverified identifier
- **WHEN** external metadata contains an identifier outside a trusted/validated portable identity schema
- **THEN** the system MAY preserve it as namespaced source evidence
- **AND** it MUST NOT automatically treat it as verified catalog identity
