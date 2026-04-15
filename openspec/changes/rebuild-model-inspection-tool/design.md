## Context

The current `dump_named_parameters.py` effort has drifted because the repo already knows how to load supported model families, but the tool work kept trying to create its own loading abstractions, its own inspection-facing APIs, or its own family-specific knowledge. That has produced two distinct problems:

1. The tool itself has no stable identity and reads like a pile of glue.
2. The library started growing dump-specific seams that do not belong there.

This change needs a design document because the problem is not just "make the script work." We need to settle the boundary between existing model support, the inspection tool, and the minimal metadata that is acceptable to keep in the library.

The current constraints are:

- The tool must inspect real loaded model objects, not a description layer.
- The tool must not use Hydra/training pipeline config composition to decide what exists.
- The tool must not introduce new per-model inspection APIs or component-description structures in library code.
- The already-existing `NAMED_PARAMETER_COMPONENT_NAMES` metadata is acceptable and is the ceiling for extra model-inspection metadata.
- The main use case is parameter inspection for manual training parameter-group work; module types are also required, and state/buffer dumping is secondary.

## Goals / Non-Goals

**Goals:**

- Rebuild the tool so it has one clear identity: inspect supported model runtimes and emit deterministic YAML views.
- Make the default output parameter-centric and grouped under existing top-level component names.
- Support module-tree inspection with module types under the same component ordering.
- Keep any family orchestration or loading adaptation in tool code, not in new library inspection seams.
- Explicitly surface repo-flow issues this work exposes instead of hiding them behind dump-specific abstractions.

**Non-Goals:**

- Creating a new model-description layer, inspection registry, or richer component metadata system.
- Introducing new dump-specific public APIs in model loaders or packages.
- Reusing optimizer-group or training-selection logic to infer what exists in the model.
- Solving every loader brittleness issue uncovered by inspection work if it is unrelated to the tool contract.

## Decisions

### 1. The tool is an inspection tool, not a second loading architecture

The tool SHALL treat existing model support as the source of truth. It may orchestrate that support from tool code, but it must not require the library to expose new dump-specific APIs just to make the script cleaner.

Alternatives considered:
- Add `load_parameter_dump_components(...)` or similar APIs to loaders/packages.
  Rejected because this makes the codebase carry a tool-specific concept it does not otherwise need.
- Build a separate inspection/description layer.
  Rejected because it duplicates information the repo already has and creates ongoing maintenance cost for every model family.

### 2. Parameter inspection is the primary contract

The default command behavior SHALL emit named parameters, ordered under existing top-level component names, because the main use case is manual training parameter-group authoring. Module inspection and state inspection are separate views on the same loaded runtime, not the tool's main identity.

Alternatives considered:
- Make state/buffer dumping the primary output.
  Rejected because it is useful but not the main workflow.
- Make module dumping the default.
  Rejected because module types help orientation, but parameters are the actionable training-facing output.

### 3. Existing component-name metadata is the only accepted extra model metadata

Top-level grouping SHALL use the already-existing `NAMED_PARAMETER_COMPONENT_NAMES`. The tool will not grow any richer component schema, model-specific inspection presets, or per-model dump metadata beyond that.

Alternatives considered:
- Infer top-level component names dynamically from returned objects only.
  Rejected because the tool still needs stable top-level labels for YAML ordering, and the repo already has the accepted minimal metadata for that.
- Add new metadata describing families in more detail.
  Rejected because the user explicitly ruled that out.

### 4. Repo-flow weaknesses exposed by this tool should be documented, not buried

If this work exposes brittle loader behavior, such as meta-tensor materialization edges or awkward non-uniform loading signatures, those findings should be called out in design/tasks as repo-flow issues. The fix for those issues should be evaluated on their own merits, not smuggled in as inspection-tool abstractions.

Alternatives considered:
- Patch around every awkward loader edge inside the tool and move on.
  Rejected because that hides weak seams and makes the tool silently become a second compatibility layer.

## Risks / Trade-offs

- [Existing loader paths are not perfectly uniform] → Keep orchestration in tool code and keep the behavior contract simple; do not solve non-uniformity by adding new dump-specific library APIs.
- [Some loaders may be correct for training but brittle for inspection-only CPU loads] → Prefer correctness-first loading for inspection views and record loader brittleness as explicit repo-flow findings.
- [The already-existing top-level name metadata may not be enough for every future family] → Treat that as a separate architecture decision later; do not preemptively grow a richer metadata system in this change.
- [The tool may still need to touch existing support registries when new models are added] → Minimize those touchpoints and use existing repo support signals wherever possible; if frequent edits are still required, that is evidence of a broader support-flow problem to capture.

## Migration Plan

1. Define the required behavior in a spec centered on runtime inspection outputs.
2. Replace the current hodge-podge script structure with a clean inspection-tool flow.
3. Remove dump-specific loader/package seams introduced by the failed attempts.
4. Validate the tool against representative supported assets, especially the SD3 case that already exposed loader fragility.
5. Record any repo-flow weaknesses exposed during the rewrite as explicit follow-up findings.

## Open Questions

- Which existing support signal should the tool use to discover supported model families so that future model additions do not require constant script surgery?
- For families whose current loading path relies on fragile meta-device flows, is the right fix a loader hardening pass, a correctness-first inspection path, or both?
- Should the shared YAML rendering helpers stay in `library/models/parameter_dump.py`, or should the final tool become fully self-contained once the output shape settles?
