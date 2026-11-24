## Why

The current strategy, training mode, objective, and mutable Trainer fields share responsibility for one training run without one acceptance boundary or one owner of current participant state. The completed architecture discussion establishes a clearer direction; this change turns it into requirements and a source-backed migration design while preserving model-specific behavior and strong research extensibility.

## What Changes

- **BREAKING**: Replace the parallel strategy/mode/objective runtime topology with contract-guided strategy authoring, validation, an accepted run arrangement, preparation, and one Trainer engine.
- Define the contract from the intended Trainer and supported pipeline capabilities. The contract guides authoring and owns enforcement, including obligations checked later during materialization and preparation.
- Preserve selected model/objective/feature implementations in the accepted arrangement so the authored strategy need not remain a runtime collaborator.
- Establish one run authority for participant identity, bindings, relationships, routes, revisions, and freshness. Backend handles and metadata remain distinct from this authority.
- Define coherent preparation and Trainer-owned optimization, carrying forward settled identity, publication, invalidation, and persistence semantics.
- Support maintained operations, custom structured operations, and explicitly authorized imperative execution through the same contract system.
- Dissolve `TrainingMode` into authored intent, specialized capability behavior, and generic pipeline mechanics.
- Develop the remaining exchanges and migration within this change. Creation of these artifacts does not authorize production implementation; the design gate in `design.md` must first be satisfied.

## Capabilities

### New Capabilities

- `training-contract`: Contract-guided authoring, acceptance, capability selection, and accepted execution independent of the authoring object.
- `run-participant-state`: Run authority, participant and relationship transitions, current routes, history, and persistence identity.
- `training-runtime-preparation`: Coherent preparation attempts, freshness, joint publication, and failure handling.
- `training-optimization`: Semantic optimization units, standard ownership and non-overlap, and realization boundaries.

### Modified Capabilities

- `loaded-model-components`: Preserve family declarations as model-loading input while transferring canonical run bindings to the accepted arrangement.

## Impact

The migration spans `train.py`, `library/strategies/`, `library/training/`, model loading, optimization, adapter integration, objectives, and selected caching/validation/sampling/persistence paths. Metadata and observability consume accepted facts without owning live state. Existing domain implementations should be reused where their responsibilities fit.

Beads issue: `sd-scripts-syv`. One governing change will contain several reviewed vertical implementation milestones. SDXL is the first detailed case; SD, SD3, adapters, compound arrangements, and strong custom/research cases also constrain the design.

The active `expand-model-metadata-identity-and-structure` change overlaps model loading and target references. Its requirements must be reconciled before either overlapping migration proceeds; this proposal does not silently supersede that work. Existing target-selector, grouping, adapter, metadata, and observability requirements remain in force until explicit deltas are justified and added here.
