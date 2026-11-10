## Why

The current run is assembled from a family `TrainingStrategy`, a separate
`TrainingMode`, a separately selected objective, and mutable Trainer fields.
That split leaves contract validation, model state, preparation, optimization,
execution, persistence, and restoration without one coherent acceptance and
authority boundary. The completed model–strategy–Trainer exploration now gives
us enough settled direction and production evidence to move that architecture
into one governing OpenSpec instead of continuing to accumulate parallel design
notes.

## What Changes

- **BREAKING**: Replace the parallel strategy/mode/objective runtime topology
  with a pre-existing Trainer contract, an explicitly authored strategy,
  contract-guided strategy acceptance, one accepted run arrangement, and one
  Trainer engine.
- Derive the contract from the intended Trainer and deliberately supported
  pipeline capabilities without making it code-owned by a Trainer instance.
  The contract guides authoring; fulfillment, the IR system's acceptance step
  that the launcher runs, checks the completed authored strategy and returns
  acceptance or detailed rejection without assembling a strategy or inferring
  missing selections.
- Build the IR system, Kuroiro on xDSL, in `library/workflow`: the language
  and its Python front end, fulfillment, the reference semantics and the
  runtime that executes accepted strategies. Add `library/runstate` for the
  run authority and `library/launch` for contract composition and launch.
- Preserve selected model-, objective-, capability-, and algorithm-specific
  behavior in the accepted arrangement so ordinary execution does not retain
  or call back into the authored strategy object.
- Establish one accepted-run authority for participant and relationship
  identity, current bindings, access views, execution routes, revisions,
  freshness, and atomic transitions. Metadata, loaders, Trainer projections,
  and backend handles do not become competing owners.
- Define coherent materialization, runtime preparation, semantic optimization,
  accepted execution, artifact persistence, and exact-restoration boundaries.
- Keep normal time, infrastructure, synchronization, optimization, triggers,
  and observation under the standard Trainer ownership profile while allowing
  maintained operations, custom structured operations, and explicitly
  authorized imperative regions through contract-governed extensions.
- Dissolve `TrainingMode` by moving its authored choices into the strategy,
  its generic mechanics into Trainer/pipeline systems, and its specialized
  behavior into accepted domain realization, execution operations, or narrow
  capability exchanges according to the behavior's actual meaning. Direct
  parameter training and PEFT composition do not become replacement runtime
  modes.
- Carry the design forward through evidence-backed OpenSpec milestones before
  production implementation. Creating the planning artifacts does not by itself
  satisfy the implementation-readiness gate.
- Use xDSL as the working semantic/prepared IR substrate. Develop the first
  shared language slice through registered region, flow and resource contracts,
  normalization, lifecycle verification and contract-driven preparation for
  model and data-only compositions, specified by the new `workflow-composition`
  capability and checked against an executable reference semantics. This
  bounded research supplies G5 evidence;
  it does not adopt experimental APIs or select a numerical backend. Selected
  computational regions may later lower through a qualified MLIR boundary.
- Extend that bounded shared slice with ordered SSA availability and early
  binding-authority checks, one minimal read/update/unknown-effects contract,
  zero-or-one filtering with explicit claim/provenance/demand settlement, and
  one contract-driven local rewrite. Keep the independent reference, cold
  correspondence and interruption checks; do not infer a broader vocabulary,
  performance claim or production readiness from this extension.

## Capabilities

The bounded language milestones (tasks 6–11) are complete. The last one
qualified scoped resource-backed products: explicit storage/state dependencies
survive producing-call completion until retained consumers physically settle.
They inform G5; they do not adopt production ownership APIs or solve
continuous-run bookkeeping reclamation.

### New Capabilities

- `training-contract`: The Trainer/pipeline-derived contract surfaces, contract-guided
  strategy authoring and acceptance boundary, execution/ownership profiles,
  and explicit standard/custom/extension conformance boundary.
- `accepted-training-execution`: The accepted run arrangement, structured and
  imperative execution meanings, Trainer-retained authority, runtime inputs,
  results, effects, failures, and the eventual step exchange, composed from
  the shared `workflow-composition` contracts.
- `workflow-composition`: The shared language for stateful, heterogeneous data
  and model work, of which training is one domain: region, flow, resource and
  realization contracts; invocation, pending requests and settlement;
  producer-stop coverage; a closed set of access modes including live shared
  reads; version admission; affine item claims and region-scoped leases;
  reproducibility, quiescence and rank-consistency declarations; and an
  executable reference semantics that realizations conform to.
- `run-participant-state`: Authority-scoped participant and relationship
  identities, bindings, routes, views, revisions, transitions, lineage, and
  scoped current-state projections.
- `training-runtime-preparation`: Coherent preparation jobs, backend grouping,
  prepared-route rebinding, freshness, publication, and replacement-only versus
  destructive failure behavior.
- `training-optimization`: Semantic training subjects and optimization units,
  logical and execution grouping, standard non-overlap, Trainer-owned
  realization/advancement, preparation interaction, and extension boundaries.
- `training-artifact-persistence`: Declared semantic products, state
  projections, plans, members, physical resources, consistency, results, and
  the boundary between trained artifacts and runtime restoration, with product
  support declared independently from direct, PEFT, or combined training
  treatment.
- `training-capability-coordination`: Selection, readiness, requests, results,
  lifecycle ownership, and failure semantics for Trainer-recognized caching,
  validation, sampling, persistence, and restoration capabilities.

### Modified Capabilities

- `loaded-model-components`: Keep family-declared top-level component surfaces
  as loading/model evidence, but move canonical run bindings from Trainer and
  the loaded-component tuple into the accepted run authority.
- `adapter-system`: Remove `AdapterMode` as the permanent training-side owner;
  preserve the repository's reusable PEFT integration, explicit currently
  supported method sets, method-specific behavior, accepted
  participant/relationship realization, and narrow capability exchanges where
  the pipeline genuinely requests a named operation.
- `adapter-module-targeting`: Preserve repo-owned target meaning and provenance
  while moving semantic target intent into strategy authoring, concrete target
  resolution into governed PEFT realization, and parameter grouping into the
  later Trainer-owned optimization exchange.
- `optimization-target-refs`: Keep shared component-qualified target meaning
  and provenance policy-neutral, and make live-object access a revision-pinned
  consumer projection rather than participant identity, canonical binding
  state, or evidence that optimization owns adapter targeting.
- `component-qualified-selectors`: Limit required component prefixes and
  internal component keys to targets with genuine component provenance;
  preserve existing model inspection/grouping behavior while admitting
  participant-owned substructure without synthetic component ancestry.
- `model-family-metadata`: Consume durable participant, transition, artifact,
  and lineage projections from accepted authorities without making metadata the
  live runtime owner or deriving identity from Python objects.
- `training-observability`: Consume accepted transition, execution,
  optimization, capability, and artifact results without reconstructing
  topology from `TrainingMode`, an active strategy, or family-shaped Trainer
  fields.
- `repo-owned-lora-method`: Preserve repo-owned LoRA realization and target
  provenance while replacing its `AdapterMode` trigger with the shared PEFT
  integration and accepted adapter-realization path.
- `repo-owned-vera-method`: Preserve repo-owned VeRA realization, shared state,
  and target provenance while replacing its `AdapterMode` trigger with the
  shared PEFT integration and accepted adapter-realization path.

## Impact

The rework affects `train.py`, `library/strategies/`,
`library/training/`, `library/objectives/`, `library/optimization/`, model
loading, adapter integration, checkpointing/restoration, metadata, and the
selected caching, validation, sampling, persistence, and observability paths.
It adds `library/workflow/`, `library/runstate/` and `library/launch/`.
Existing family and domain implementations should be retained where their
responsibilities fit, but their whole-Trainer and active-strategy boundaries
will change. The system itself is built new: today's Trainer, strategy and
mode code is a checklist and a sanity-check reference, never migrated.

The change is tracked by bead `sd-scripts-syv`. It is one governing
architecture change with multiple focused specifications, built new in
reviewed milestones. SDXL is the first detailed case;
SD, SD3, adapter, deferred-loading, compound-participant, and strong imperative
research cases constrain the design before implementation. The active
`expand-model-metadata-identity-and-structure` change overlaps identity,
loading, provenance, and restoration concerns and must be reconciled rather
than silently superseded.

The bounded xDSL language slice in `design.md` D14 and tasks 6–11 is complete
and remains under `docs_design/models-strategy-trainer/` as evidence.
`design.md` D15–D21 record the production design. Tasks 5.1–5.7 govern closing
it and the implementation-readiness decision, and `tasks.md` sections 12–28
are the implementation milestones.
