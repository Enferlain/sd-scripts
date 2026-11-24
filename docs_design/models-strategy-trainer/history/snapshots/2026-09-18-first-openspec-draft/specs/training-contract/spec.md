## Purpose

Define how the Trainer's acceptance contract guides strategy authoring and establishes executable runs with explicit ownership and extensibility.

## ADDED Requirements

### Requirement: Contract guides authoring and acceptance
The system SHALL expose the vocabulary, supported capabilities, execution ownership and rules of the active training contract before strategy authoring. Establishment SHALL validate the complete authored definition under that contract and derive lifecycle obligations. Ordinary authors SHALL NOT have to reconstruct compatibility rules already owned by the contract or selected library implementations.

#### Scenario: Unsupported combination
- **WHEN** authored selections or wiring violate the active contract
- **THEN** establishment rejects the definition with an explanation of the violated obligation
- **AND** no Trainer-acceptable arrangement is produced

#### Scenario: Evidence becomes available later
- **WHEN** an accepted definition permits deferred materialization
- **THEN** the arrangement retains the obligations requiring runtime evidence
- **AND** dependent use remains unavailable until those obligations are satisfied

### Requirement: Accepted execution survives authoring
The accepted arrangement SHALL retain the behavior, dependencies and configuration necessary to execute the authored training meaning without an ordinary runtime dependency on the authored strategy object. Model and objective behavior SHALL remain selected implementations rather than becoming family-specific Trainer logic.

#### Scenario: Authoring object released
- **WHEN** establishment succeeds and the authored strategy object is released
- **THEN** the accepted arrangement can still be materialized, prepared and executed
- **AND** ordinary steps do not call back into that object to recover static choices

### Requirement: Contract separates core and selected capabilities
The contract SHALL distinguish core engine requirements, selected pipeline capabilities and reusable feature contracts. It SHALL NOT require universal denoiser, text-encoder, VAE, timestep or single-loss fields merely because current maintained paths use them.

#### Scenario: A run does not use latent encoding
- **WHEN** an otherwise conforming definition trains without a latent representation feature
- **THEN** acceptance does not require a VAE participant or placeholder latent-encoding implementation

### Requirement: Custom execution declares its authority
The contract SHALL support maintained operations, custom structured operations and explicitly authorized imperative regions. A region requesting authority beyond the standard execution profile SHALL require an accepted contract extension/profile that defines its authority and obligations.

#### Scenario: Imperative optimization region
- **WHEN** a research definition requests control over backward or optimizer advancement
- **THEN** acceptance checks that control against an explicit supported ownership contract
- **AND** rejects it if no such contract is available
- **AND** does not implicitly give the strategy object whole-Trainer mutation access
