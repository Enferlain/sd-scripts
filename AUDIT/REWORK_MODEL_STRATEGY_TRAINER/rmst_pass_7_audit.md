Pass 7: whole-run cases and G5 readiness.

Apply the shared general audit protocol. This is a read-only architectural and
specification audit, not an implementation task or another capability survey.

Audit commit:
6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48

Confirm that the examined sources correspond to that commit. Identify any
additional working-tree content separately; do not silently combine revisions.

Passes 0–6 are navigation and evidence to verify, not authority or a collective
proof of correctness. Read their findings, routed questions and coverage limits.
If a report is unavailable, disclose that limitation rather than reconstructing
its conclusions from its finding IDs.

Local corrections outside this baseline address P1-REV-01, P2-REV-01,
P4-REV-01, P6-REV-01 and P6-NOTE-01. Reference inherited findings without
duplicating them or assuming those corrections are present. The Pass 6 code
fix has passed local checks but its required review is still blocked by provider
quota. This notice is not evidence that any patch is reviewed or sufficient.
Do not issue a corrected-working-tree readiness verdict unless that exact
additional source state has been supplied and separately examined.

Objective

Determine whether the established meanings compose into coherent whole runs,
and whether genuine semantic decisions remain before G5 production-shape work.

Keep two assessments separate:

A. Independently trace concrete cases across the relevant boundaries.
B. Synthesize those traces and the earlier passes into a G5-readiness assessment.

A clean concern report does not prove a combined case. Conversely, the absence
of an integrated production engine does not itself make the semantic design
incomplete: building that engine is later work.

Sources

- Read the governing source-authority rules and reconciliation ledgers.
- Read the complete representative case matrix, G2 worked cases and synthesis,
  D5's exchanges and composed state, and applicable G3/G4 decisions.
- Read G1–G5 gates, deferred production-shape choices, migration boundaries,
  and both active changes' tasks and integration dependencies.
- Follow each case through complete relevant requirements and scenarios in
  training-contract, run-participant-state, loaded-model-components,
  training-runtime-preparation, optimization-target-refs, training-optimization,
  accepted-training-execution, training-capability-coordination,
  training-artifact-persistence, model-family-metadata and
  training-observability. Verify actual capability paths in the change.
- Include the companion expand-model-metadata-identity-and-structure change
  and surviving main-spec obligations wherever the case depends on them.
- Use research/README.md, training-mechanism-sketch.md, the boundary/derivative
  research and architecture proposals to locate the recorded pressure cases
  and their support/evidence dispositions. Proposals do not override adopted
  requirements, and an upstream method's unknown behavior remains unknown.
- Inspect experiment implementations and actual assertions when relying on
  experimental claims. Current production is migration evidence, not the
  target architecture.

7A. Independent whole-run case checks

Choose concrete source-backed variants covering the cases below. Related
variants may share a trace, but preserve their materially different demands.
Explain the coverage and any omitted case; a diffusion-only sample is inadequate.

1. Ordinary training and PEFT composition

Trace ordinary SDXL, PEFT-only and joint direct-plus-PEFT training through
preparation, execution, capabilities and persistence/restoration. Include
scheduled/adaptive behavior and selected-method lifecycle effects.

Check SD/SD3 variation as the recorded limited family comparison, not proof of
generality. Preserve the dissolution of TrainingMode, policy-neutral target
refs, independent product choice and the separation of integration behavior
from method-specific behavior.

2. Distinct participants, uses, gradients and representations

Trace teacher/student or evolving-teacher behavior, one participant in several
roles/views, and a frozen-base gradient path to a learned side-network.
Include a standalone learned participant without family-component ancestry.

Use pixel, video/audio, LLM-containing or multimodal cases from the recorded
research to check absence of universal latent/image/denoiser slots, fixed axes,
one-result assumptions or one model-family anatomy in the engine. Distinguish
which demands are accepted support, extension paths or intentionally deferred.

3. Different optimization arrangements and incomplete effects

Contrast alternating adversarial actions with jointly due Muon/AdamW-style
units and independent contribution/accumulation windows. Include a failure
where one unit advanced and another failed or remains uncertain.

Preserve parameter ownership/alias correspondence, differentiated work,
retained source versions, per-unit outcomes and dependent-use gating. Neither
one action nor one cycle implies atomic updates, rollback, exactly-once replay
or detection of numerical parameter change.

4. Independent production, changing inputs and required feedback

Trace asynchronous text encoding or versioned experience alongside training,
including changing captions/encoder or policy state, out-of-order completion,
multiple-input correlation and structured provenance where several states
contributed to one result.

Combine admission, bounded backpressure, cancellation/failure, input selection
or packing, and required adaptive feedback. Follow independently progressing
owners and their handoffs; do not turn producer internals into compulsory
action-local operations or reduce the run to input → action → optimizer.

5. Granted authority alongside capability readers

Combine a SAM-style two-pass or other strong granted region with validation,
sampling, cache production or capture using affected participant state.

Check the grant transfers real actions rather than permitting duplicate
ownership. Follow temporary state, actual backend completion, retained
derivatives, cleanup and handback. Unsupported backend/authority combinations
must not acquire readiness from a returned callback or clean visible tensors.

The standard profile's trusted-Python declarations are not proof that arbitrary
Python cannot mutate hidden state. Match guarantees to the actual accepted
authority and enforcement/trust boundary; do not silently weaken or strengthen it.

6. Stage change while other work remains outstanding

Trace progressive distillation, participant replacement, trainability change
or model surgery while input production, contribution windows, capability
readers and/or a preparation attempt remain outstanding.

Follow which meanings remain valid, which uses must stop, which state is
preserved/migrated/reinitialized/retired, and which preparation may publish.
Keep numerical evolution, binding/definition/composition revisions, produced
data provenance, prepared generations and execution sessions distinct.

7. Products, snapshots and failure across owners

Combine a trained product with required external dependencies, partial
multi-resource publication, a runtime snapshot and delayed/failed metadata
filing. Product completeness, current usability, snapshot coherence, durable
history and restoration eligibility must not substitute for one another.

Trace a supported coherent continuation with pending inputs/contributions and
selected operation/capability state, plus a required reconstruction failure.
Include late prior-session results and explicit new-run initialization from a
product. Regeneration must follow accepted continuation rules, not inferred
replay permission from missing telemetry.

Required combined recovery check from Passes 5–6

Snapshot K at accepted scope A1 contains participant P. Later, scope A2 retires
P and may declare Q at the same authored address. Some later effects/history
are known; others may be uncertain. Recovery considers K for the same logical
run, with a new execution session.

Determine what the existing requirements actually establish about:
- K's original capture guarantee versus its later restoration eligibility;
- the rule that retired references are not revived;
- preserving same-run identity and prior-session history;
- correspondence between restored current scope and delayed historical facts;
- reconciliation of known/uncertain effects after the cut.

Show whether these rules already admit a coherent supported policy, leave a
necessary decision open, or conflict for an offered recovery claim. Do not
erase A2, automatically revive P, silently start a new run, assume every older
snapshot is unusable, or invent a rollback/branching policy to make the case work.
Do not demand universal recovery from every cut or external effect.

For each trace

Follow the relationships actually needed by that case:
contract-guided authoring and fulfillment; semantic acceptance; identities,
binding/materialization and preparation; executable readiness; activity
scheduling, inputs and selected dynamic behavior; optimization/granted effects;
capability use; publication/metadata; failure, capture and continuation.

This is a trace checklist, not a required execution order. Use hierarchical
whole-run structure and cross-owner relationships rather than one universal
training-step sequence, global clock, result packet or completion boundary.

At significant handoffs identify:
- the owner and accepted obligation;
- the state/dependency/provenance that must remain valid;
- available evidence and the earliest meaningful check;
- reached effects and outcome knowledge, including uncertainty;
- which dependent work may proceed or must remain unavailable;
- continuation/history implications where applicable.

Cite exact requirements/scenarios. Separate source-derived reasoning, recorded
experiments, freshly executed checks and unsupported production guarantees.
No new executable experiment is required to perform this audit. Recommend the
smallest bounded investigation only if an actual unresolved question needs it.

7B. Cross-pass synthesis and G5 readiness

1. Cross-check earlier conclusions against the independent traces. Identify
   incompatible assumptions, duplicate findings, missed relationships and
   routed questions that remain unresolved. Independently verify important
   inherited findings rather than tallying clean reports as votes.
2. Examine checked G1–G4 tasks against their actual acceptance criteria and
   recorded evidence/review limits. Do not convert design completion into
   production support, or require production integration to close a design-only
   task unless its acceptance criteria actually demand it.
3. Distinguish genuine semantic blockers from G5's intended decisions: concrete
   types, representations/lowering, APIs, code placement, supported backend
   mechanisms, conformance tests and exact migration milestones.
4. Map remaining work to 5.1–5.7 and the companion's concrete integration gates.
   Do not mark those gates complete or settle deferred storage/type choices.
   Identify any missing meaning that those tasks cannot safely choose on their
   own without an explicit architectural decision.
5. Assess readiness to START G5 production-shape and migration planning, not
   readiness to implement/cut over the live Trainer. Keep semantic readiness,
   evidence/review prerequisites and corrected-source verification separate.

Return

- 7A: a concise case-to-boundary coverage map and concrete combined traces;
- findings with stable P7-prefixed IDs under the general reporting protocol;
- a cross-pass disposition register, including the older-cut recovery question;
- unsupported completion/support claims, if established, with exact evidence;
- 7B: a qualified G5-entry verdict identifying blockers, bounded follow-up work,
  deliberate G5 choices and inherited findings/corrections outside the baseline;
- evidence and coverage limits, including unavailable reports or source states.

If no new blocker is established, state that with its verified scope; do not
invent one to make the final pass productive. If a blocker remains, give a
concrete consequence and the smallest investigation/correction required.

Do not edit files, apply fixes, change task status, redefine the architecture,
or claim the unexamined corrected checkout is ready.
