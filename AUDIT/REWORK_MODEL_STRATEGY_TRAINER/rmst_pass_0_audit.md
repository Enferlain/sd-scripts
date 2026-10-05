Pass 0: reconstruct the specification before judging it.

Your task is to produce a source-backed map that subsequent audit passes can
use. This is not a general redesign or the comprehensive semantic audit itself.

Read the current proposal, governing design, task list, and all delta specs
of rework-model-strategy-trainer. Read the companion metadata change's
proposal/design/tasks and relevant deltas, including the shared boundaries.
Compare affected main-spec requirements so unchanged obligations are visible.

Read the research index and training-mechanism sketch to identify supporting
cases, experiments, and their limitations. Follow other supporting records
where necessary to interpret a decision or resolve its history. Do not treat
every historical statement or research suggestion as an accepted requirement.

Produce the following:

1. Source-authority map
   Identify what governs, what supplies evidence, what is chronological history,
   and which explicit corrections supersede earlier positions.

2. Architecture and responsibility map
   Reconstruct contract-guided authoring, fulfillment, accepted run state,
   materialization, preparation, execution, optimization, capabilities,
   persistence/restoration, metadata, and observability.
   Show their relationships and ownership boundaries—not merely a file list.

3. Concept register
   For each important concept, record:

   - its documented meaning;
   - who establishes or owns it;
   - who may change it;
   - who consumes or observes it;
   - applicable identity, revision, lifecycle, and restoration distinctions;
   - exact supporting references.

   Cover run/session, participant/relationship, binding/route/view,
   activity/input/request/attempt, selected behavior and owned state,
   optimization unit, realization/composition, catalog/source,
   product/member/resource, runtime snapshot, and observation.

   Do not invent an owner, identifier, class, or revision field where the
   documents do not establish one. Mark missing, ambiguous, conditional,
   or deliberately deferred information explicitly.

   Keep references selective: give primary authoritative references, plus
   supporting references that establish a materially different constraint,
   exception, aspect, or later correction. Do not catalogue every mention.
   Where useful, label individual properties as established, conditional,
   deferred, ambiguous, or conflicting; separately identify evidence-only
   support. These labels may coexist and need not apply to every entry.
   Distinguish information you could not locate from an established omission
   or an explicit deferral; give the relevant search/coverage limits.

4. Cross-artifact map
   Connect design decisions to requirements/scenarios, completed-task claims,
   representative cases, and experimental evidence.
   Identify shared or duplicated requirements between the two changes,
   unchanged main-spec dependencies, and documented synchronization gates.

5. Disagreement and uncertainty register
   Collect apparent source inconsistencies, unclear authority or supersession,
   potentially missing relationships, and uncertain semantic-versus-G5
   boundaries encountered during reconstruction.
   Give each item a temporary pass-prefixed ID (for example P0-INV-01), exact
   sources, the uncertainty or additional assumption involved, and one or more
   later passes to investigate it. These are investigation items, not confirmed
   findings. Do not resolve them by silently selecting a favorable reading.

6. G5 boundary
   Separate meanings already established from concrete choices left to G5.
   Identify places where you cannot determine which category applies.

7. Routing for the later audits
   Give each planned concern pass its primary sections/specs, related
   cross-cutting requirements, relevant cases, and evidence sources.
   Highlight clauses needing further verification, without declaring them
   correct simply because they fit your reconstructed map.

The planned passes are:

1. Architecture, authoring and ownership.
2. Revisions, lifecycle and preparation.
3. Execution, optimization and failure.
4. Independent progress and capabilities.
5. Products, snapshots and recovery.
6. Metadata, observability and spec compatibility.
7. Whole-run cases and G5 readiness.

Pass 7 must retain whole-run case checks as well as synthesis of earlier
reports. Route cases that exercise several ownership or lifecycle boundaries
to it; reconciling earlier findings alone does not verify those interactions.
Its case checks and finding synthesis should be reported separately, with
important inherited findings independently verified against the sources.

The map must describe the specification actually present in the repository.
Do not normalize away disagreements or impose a universal execution sequence.
Distinguishing concepts does not require separate implementation constructs.

If you encounter an apparent contradiction, record its exact sources as an
item for investigation rather than silently choosing the interpretation that
makes the architecture coherent.

Conclude with coverage and limitations. Do not issue a final “G5 ready” verdict
from this orientation pass.
