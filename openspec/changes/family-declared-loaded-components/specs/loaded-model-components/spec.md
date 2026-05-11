## ADDED Requirements

### Requirement: Training-capable model families declare top-level loaded components
The training/runtime architecture SHALL treat family-declared top-level loaded components as the primary repo-owned representation of a loaded model.

#### Scenario: Family declares ordered components
- **WHEN** a model family participates in the active strategy/trainer path
- **THEN** that family MUST declare its top-level loaded components in family-owned order
- **AND** each declared component MUST expose a stable component key, a public component label, and the live loaded module object when present

#### Scenario: Family has multiple components with shared semantics
- **WHEN** a model family exposes multiple top-level components that share a semantic role
- **THEN** the loaded-component contract MUST preserve them as distinct declared components
- **AND** it MUST NOT collapse them into one universal shared slot

### Requirement: Loaded components declare generic semantics explicitly
Loaded components SHALL declare the roles and/or capabilities that generic repo code needs instead of relying on family-name or slot-name assumptions.

#### Scenario: Generic code needs component semantics
- **WHEN** trainer, logging, optimization, metadata, or tooling code needs to know how to treat a component
- **THEN** that code MUST consume declared roles and/or capabilities from the loaded-component contract
- **AND** it MUST NOT infer generic behavior by hardcoding family-native component names

#### Scenario: Components opt into generic participation
- **WHEN** a generic consumer needs to know whether a top-level component is relevant for behaviors such as reporting, targeting, or metadata
- **THEN** that participation MUST come from family-declared component semantics
- **AND** it MUST NOT be reconstructed from the old `text_encoders/vae/denoiser` slot assumption

### Requirement: Model loading returns the loaded-component surface
The active model-loading strategy contract SHALL return a family-declared loaded-component surface instead of the fixed `text_encoders/vae/denoiser` tuple.

#### Scenario: Loading a training model
- **WHEN** `ModelLoadingStrategy.load_target_model()` loads a training-capable model family
- **THEN** it MUST return the model version together with the family-declared loaded-component surface
- **AND** the loaded-component surface MUST be the authoritative top-level component representation for downstream runtime code

#### Scenario: Trainer stores the primary loaded-component state
- **WHEN** trainer setup receives loaded model state from the strategy layer
- **THEN** the trainer MUST treat the loaded-component surface as its primary top-level model representation
- **AND** generic runtime code MUST NOT depend on the old tuple fields as the long-term contract

### Requirement: Top-level components remain the highest generic control surface
The repo SHALL treat top-level model components as the highest generic control surface for training/runtime structure.

#### Scenario: Expanding to lower-level targets
- **WHEN** optimization or adapter code needs module-level or parameter-level targeting
- **THEN** that lower-level expansion MUST start from declared top-level components
- **AND** the loaded-component contract itself MUST remain top-level rather than becoming a module-level registry

#### Scenario: Avoiding slot-based control surfaces
- **WHEN** a future family exposes top-level components that do not fit the current diffusion tuple
- **THEN** generic runtime code MUST still be able to reason about the family at the component level
- **AND** the repo MUST NOT require adding new universal slot fields to preserve that behavior
