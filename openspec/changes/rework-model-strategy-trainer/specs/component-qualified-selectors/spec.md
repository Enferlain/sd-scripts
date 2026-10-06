## MODIFIED Requirements

### Requirement: Parameter dumps use component-qualified selector names
For parameters with genuine family-declared model-component provenance, the
inspection tool SHALL expose selector names in the form `component.local_name`,
where `component` is the declared public component prefix and `local_name` is
the real runtime parameter path within that component. This component-qualified
form SHALL NOT impose component ancestry on accepted participant-owned
parameters without that provenance. Where such targets are exposed for
inspection, their selectors SHALL preserve the accepted participant/substructure
meaning without fabricating a component prefix.

#### Scenario: Dumping SDXL parameter selectors
- **WHEN** the inspection tool dumps parameter selectors for an SDXL runtime
- **THEN** the output SHALL use names such as `unet.input_blocks.0.0.weight` and `clip_l.text_model.embeddings.token_embedding.weight`

#### Scenario: Dumping SD3 parameter selectors
- **WHEN** the inspection tool dumps parameter selectors for an SD3 runtime
- **THEN** the output SHALL use names such as `mmdit.<path>`, `clip_l.<path>`, `clip_g.<path>`, `t5xxl.<path>`, and `vae.<path>` rather than training-internal placeholders

#### Scenario: Inspecting a standalone participant parameter
- **WHEN** inspection exposes a learned parameter owned by an accepted standalone participant without family-declared component provenance
- **THEN** its selector MUST preserve the accepted participant/substructure meaning
- **AND** inspection MUST NOT invent or borrow a model-component prefix to expose that parameter

### Requirement: Fine-grained optimizer matching uses the same selector namespace
For component-backed parameters, fine-grained optimizer group matching SHALL
evaluate `groups` and `groups_file` patterns against the same component-qualified
selector names exposed by the inspection dump. Where accepted grouping supports
participant-owned parameters without component provenance, matching SHALL use
their accepted participant/substructure selector meaning consistently with
inspection, without requiring synthetic component ancestry. Existing
component-backed selectors and learning-rate matching behavior SHALL remain
unchanged.

#### Scenario: Regex targets a specific component
- **WHEN** a learning-rate group pattern matches `^unet\\.` or `^mmdit\\.`
- **THEN** only parameters in that matching component SHALL be selected

#### Scenario: Regex targets a specific text encoder
- **WHEN** a learning-rate group pattern matches `^clip_l\\.` or `^t5xxl\\.`
- **THEN** only parameters in that matching component SHALL be selected

#### Scenario: Matching an accepted standalone parameter selector
- **WHEN** accepted fine-grained grouping supports a standalone participant parameter exposed for inspection
- **THEN** matching MUST use the same accepted participant/substructure selector meaning
- **AND** absence of a family component key MUST NOT require relabeling that parameter as component-backed

### Requirement: Component-qualified selectors identify target refs
Shared module and parameter target refs with genuine model-component provenance
SHALL use the component-qualified selector namespace. Shared refs for accepted
participant-owned substructure without that provenance SHALL instead preserve
the accepted participant/substructure selector meaning. Such refs SHALL NOT
require an invented or borrowed public component prefix. These applicability
rules SHALL preserve existing component-backed selector behavior without
prescribing the concrete standalone selector syntax.

#### Scenario: Parameter target selector matches inspection output
- **WHEN** a parameter target ref is built for a model parameter with genuine declared component provenance
- **THEN** its selector MUST match the component-qualified parameter name that
  the inspection tool exposes for that parameter

#### Scenario: Module target selector uses the same component prefix
- **WHEN** a module target ref is built for a model module with genuine declared component provenance
- **THEN** its selector MUST use the same public component prefix used by
  parameter selectors for that component
- **AND** the remaining selector path MUST be the module's component-local path

#### Scenario: Standalone target does not require a component prefix
- **WHEN** a shared module or parameter ref selects accepted participant-owned substructure without family-declared component provenance
- **THEN** its selector MUST preserve its accepted participant-qualified subject meaning
- **AND** the target MUST remain representable without a synthetic component prefix

### Requirement: Internal component keys stay separate from public selectors
Shared target refs with genuine model-component provenance SHALL preserve
internal component keys separately from public component-qualified selector
strings. Refs without that provenance SHALL preserve their accepted participant
and selected substructure without inventing or borrowing an internal component
key. Genuine host-target provenance SHALL remain distinct from the participant
ownership of the selected state; neither a matching selector nor a host or
catalog association alone SHALL establish component ancestry for that state.

#### Scenario: Public selector differs from internal component key
- **WHEN** a model family exposes a public component label such as `unet`,
  `clip_l`, `clip_g`, `mmdit`, or `t5xxl`
- **THEN** target refs MUST preserve that label in the public selector
- **AND** target refs MUST also preserve the internal component key used by
  training code

#### Scenario: Standalone learned state has no internal component key
- **WHEN** a shared parameter ref identifies a learned loss-weight vector owned by a standalone accepted participant without family-declared component provenance
- **THEN** absence of an internal component key MUST NOT prevent valid target construction
- **AND** the ref MUST retain the owning participant and selected substructure rather than borrow a denoiser's component key

#### Scenario: Rejecting a fabricated component association
- **WHEN** a candidate standalone target claims component ancestry based only on a borrowed selector prefix, host association, or catalog association
- **THEN** the resolving consumer MUST reject that unsupported component claim
- **AND** genuine host provenance MUST NOT substitute another participant's ownership for the owner of the selected state
