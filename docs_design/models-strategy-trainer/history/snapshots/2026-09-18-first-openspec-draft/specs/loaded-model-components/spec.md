## MODIFIED Requirements

### Requirement: Model loading returns the loaded-component surface
The model-loading boundary SHALL produce the model version and family-declared loaded-component candidates rather than a fixed `text_encoders/vae/denoiser` tuple. The accepted run arrangement's authority SHALL validate and publish their participant bindings before dependent runtime use.

#### Scenario: Loading a training model
- **WHEN** a model loader produces a training-capable model family's components
- **THEN** it MUST preserve the declared component keys, order and public labels alongside candidate loaded values
- **AND** those candidates MUST enter runtime use through accepted participant bindings

#### Scenario: Trainer stores the primary loaded-component state
- **WHEN** Trainer setup coordinates model materialization
- **THEN** current bindings MUST be owned by the accepted run arrangement's authority
- **AND** Trainer MUST NOT retain a competing authoritative component collection or depend on fixed family slots

### Requirement: Top-level components remain the highest generic control surface
Family-declared top-level components SHALL remain the model-loading and model-substructure boundary. The run participant surface SHALL also support independently managed participants outside one loaded model, without requiring their anatomy to fit family component slots.

#### Scenario: Expanding to lower-level targets
- **WHEN** optimization or adapter code needs module-level or parameter-level targets within a loaded model component
- **THEN** expansion MUST preserve that declared component's qualified identity and provenance
- **AND** the loaded-component declaration MUST remain top-level rather than becoming a module registry

#### Scenario: Avoiding slot-based control surfaces
- **WHEN** a future family exposes components outside the current diffusion tuple
- **THEN** generic runtime code MUST consume accepted participant bindings without adding universal family slot fields

#### Scenario: Additional run participant
- **WHEN** a run declares an independently managed adapter or teacher alongside a loaded model
- **THEN** the arrangement MUST represent that participant without redefining the source model's family component inventory
