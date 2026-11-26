# Design and delta-spec changes from the authoring discussion

Date: 2026-11-26

Status: **proposal for review; nothing applied.** Basis: `design.md` and the
delta specs at `9055aa20` (line numbers refer to it), bead `sd-scripts-syv.77`,
the live sketch `library/strategies/sdxl_test.py`, the research notes
[execution requirements and policies](execution-requirements-and-policies.md),
[resource-aware execution](resource-aware-execution.md) and
[execution delegation review](execution-delegation-review.md), and the later
nine-point review feedback.

Part A lists corrections: points settled in discussion that the text
contradicts. Part B lists open investigations to record as open, with no
normative text. Part C lists decisions still needed. Part D lists follow-ups
outside the design.

## A. Corrections

### A1. The IR takes its input only through the strategy; library code holds nothing of the IR

Settled in syv.77. Everything the IR learns about what to run and how each
piece is used comes through the strategy. Library code (losses, objectives,
models, data, adapters, evaluation) stays plain code, with no IR decorators,
written-form types, effects or relations. "Implementation" means how a
strategy uses a piece of code, not the code. What the system needs to know
about code comes from its knowledge kept by code identity (D16 evidence),
written by people and programs, or from what the strategy states. A strategy
may state or override a piece's role, never what the code does. Knowledge is
descriptive, observed, evaluative or contextual; each fact keeps the scope of
the check that established it; facts relied on for correctness are checked by
the system, while ratings are guidance; a run records the facts it relied on.
Code the system knows nothing about is still usable.

| Where | Text now | Change |
| --- | --- | --- |
| Glossary, Implementation (365) | "Selected code that does the math, declared with its contract and selected by name" | How a strategy uses a piece of code: its role in the strategy |
| D16 seams, Implementations row (3679) | Code "declaring its inputs, outputs, effects, concurrency safety and completion, and which inputs propagate a named relation"; examples include objectives, codecs, serializers | The seam supplies what the system knows about code, from its knowledge or the strategy, not declarations carried by the code |
| D16 owner versions (3684-3692) | "After each call whose declaration says it updates through a lease" | Which calls update is known from the strategy's access and the system's facts about the code |
| Contract (3558-3564) | "capabilities and features declare themselves from their own packages" | Domains such as training register their words; library code registers nothing |
| D15 device capacity (about 3289) | "Implementations declare that they use a device" | Known about the code, or stated by the strategy |
| Code evidence row (4484); spec workflow-composition, "Selected code is identified by evidence the system computes" (189-205) | Evidence includes "a hash of its declaration" | Drop the declaration hash; fixture outputs and source hashes remain (code identity continues in syv.76) |
| Authoring, three kinds of code (4748-4765) | "Implementations ... are called through their declarations" | Activities, the strategy function, and the code they use: existing code as it is, modified code, or the strategy's own code marked as its own (how it is marked is topic 3) |
| Declarations and guidance (5712-5722) | "Components and features declare themselves in three layers" | The three layers (meaning, requirements, provenance) are what the system knows about a component; the rules and guidance table stays |
| D21 dependency rules (5798) | "Libraries may import the run state, the IR system, ..." | Libraries import neither the IR system nor the run state (the run-state half needs confirming) |
| Spec workflow-composition, "One implementation contract bounds resource effects" (51-75) | "Installed implementation contracts SHALL state required resource reads/updates ... and declare possible additional unknown effects" | Same obligations, met from the system's knowledge or the strategy's statement; unknown effects are still honored conservatively by transformations |
| Spec accepted-training-execution, "A call records gradients only on a path to a unit's loss" (163-186) | "inputs that implementations declare as carrying gradients" | Inputs the system knows, or the strategy states, carry gradients. The rule itself stays |
| Spec training-contract, "Authored behavior declares what is needed to judge it" (60-81) | "Authors SHALL declare the inputs, owned state, effect vocabulary ..." | The system knows them, or the author states them |

### A2. Configuration fills the strategy's declared parameters

The text is mostly consistent already (4745, 4972, launch settings at 4463).
Changes:

- "Configuration is a user-facing filter: it only narrows" (4745) becomes:
  configuration fills the strategy function's parameters, which are plain
  values, picks among named options, and roles filled by name with any code
  the system knows for that role. It never adds wiring, and the strategy's
  signature is the complete list of what a config can change.
- Settings types (optimizer, adapter method, dataset description, precision
  recipe) belong to their libraries and are reused by strategies, so a saved
  setup fits any strategy that takes that type. D21 already has config
  compose "the settings types that their owners define" (5789).
- No hand-kept conflict lists. A value is checked by the signature, the
  parameter's type, the settings type's own checks, and fulfillment comparing
  each part's needs with what the run gives it.
- Launch's own part says where and how the run happens: outputs, devices,
  logging, resume. Hydra stays.

### A3. Precision is a recipe per participant

The structure matches D15 (3228-3245) and spec accepted-training-execution,
"Each precision choice belongs to its owner" (187-203). Changes:

- The choice for a participant's calls is a recipe: weight format, activation
  format, compute format, scaling state and the layers it covers (bf16
  autocast, an FP8 training recipe, an NF4 base, W4A8), supplied as library
  code plus the system's facts about it.
- Which recipes fit follows from how the strategy uses the participant
  (trained; frozen with gradients passing through; frozen and only run
  forward), compared with each recipe's facts. Authors don't list allowed
  precisions. They may fix one, name the recipes they maintain, or give one
  slot to several participants.
- The example `realize(@mixed_precision, "bfloat16")` (3237) becomes the
  strategy-level form.

### A4. A read states its version; how the version is provided is delegated

| Where | Text now |
| --- | --- |
| D14 (2798-2800) | "every resource use selects a realization and states its freshness requirement separately" |
| D14, requested work (2836-2845) | "Where the use selects protected live access ... Where a detached realization holds the version, such as a selected snapshot" |
| Which weights work sees (4599-4612) | "A strategy that selects a snapshot instead lets training continue" |
| Front end rules (4819-4823) and example (4766-4790) | `.protected_current` and `.snapshot` select the access mode |
| D18 refusal table (4946) | A silent substitution includes "another access mode than the one selected" |
| Spec workflow-composition, "Resource uses select a defined access mode" (354-390) | "Every resource use SHALL select an access realization"; a target "SHALL reject it rather than substitute another realization"; scenario "A target cannot realize live sharing" forbids substituting a snapshot |

Change: a read states which version it observes: current, at the request,
current at start, or a published version. That meaning stays the author's,
and D14's version rules are unchanged (a request holds one version per
resource; `at_request` is fixed when the request is made). How the version is
provided, by protecting live state while conflicting updates wait or by a
detached copy, is a delegated decision. It is made only with a realization
established to provide that version and the full state the use reads
(buffers, precision scales, adapter state), not only the parameters. A target
that can provide the stated version by no established realization still
rejects it, and an unprotected read is never a substitute. The closed set of
realizations stays as the IR's internal vocabulary that owners realize. Owner
updates and grants are authority, not reads, and stay as selected. The
silent-substitution row becomes "observing a different version, or less state,
than the strategy states". The version names in the front end are
placeholders.

### A5. Overlap and leaving the device are delegated, under established eligibility

| Where | Text now |
| --- | --- |
| D15, device capacity (3277-3290) | "A strategy states only meaning: whether work of different activities may overlap ... Work is presumed not to overlap unless the strategy permits it" |
| D15, leaving the device (3414-3441) | "A strategy may permit a participant to leave the device between uses ... it is a permission", written as `may_leave_device` |
| Language table (4210) | `may_leave_device` |
| Spec workflow-composition, "Version selection, device admission and placement are independent" (392-417) | "A strategy SHALL state only whether work of different activities may overlap"; scenarios that begin "a strategy permits" |

Change: overlapping work and moving a participant's storage off the device are
the system's decisions. It makes them only where it can establish that the
choice preserves what the run requires (A6), for example random state of its
own for overlapped work (3370), no unseen write before restoring a host copy,
and storage that nothing else refers to. The author can still state a
requirement when they genuinely need one (how is B1). What stays: version,
admission and placement are independent; meaning that requires work to be
live together on a device that cannot hold it is rejected; the wait-cycle
rules; only exact copies qualify. Running everything in turn is the current
implementation, not a rule.

### A6. What a delegated choice must preserve

This extends spec accepted-training-execution, "Equivalent execution preserves
required interactions" (102-118), from lowering to every delegated execution
choice: access realization, overlap, residency, recomputation and keeping or
releasing intermediate values. Proposed text:

- The author states the behavior and, where genuinely needed, requirements on
  how it runs. Everything else is delegated.
- A delegated choice keeps the run within the authored set of allowed
  behaviors, not one particular execution. It preserves the consistency the
  run requires (recomputation replays the draws and call context of the work
  it recomputes; uses that share a requested version see it) and each call's
  context: grad mode, module mode and precision setup.
- An alternative is used only where it is established to stay within the
  requirements. Mathematical resemblance is not enough, and an approximate
  algorithm is not an execution choice unless the strategy permits the
  approximation.
- The system records its decisions, what they were based on, and what
  actually ran.

### A7. Unchanged

- Module mode stays as D20 decided (5454-5478): training mode only where a
  module belongs to a unit whose loss the call reaches, so a frozen network on
  the gradient path stays in evaluation mode. The sketch's gradient-path rule
  for module mode was withdrawn.
- Gradient recording stays on a path to a unit's loss (spec, 163-186); only
  where the gradient facts come from changes (A1).
- D18's refusal rule stays; B2 checks how a planner finding no solution fits it.

## B. Open investigations (record as open; no normative text)

- **B1. Expressing execution requirements.** What a requirement can be about,
  stated precisely over observable run events; generic relations over existing
  entities rather than flags for pairs of activities; requirement versus
  preference versus objective; where each is established and enforced;
  unsupported guarantees represented honestly. Nothing shows that the SDXL
  strategy needs one.
- **B2. Choosing among realizations and reporting outcomes.** There is no
  privileged fallback: an unsupported candidate is skipped while others are
  considered. Keep six outcomes distinct, and don't call them all refusals:
  contradictory intent, an unsupported target capability, insufficient
  evidence, resources unavailable now, no solution found within the planner's
  bounds, and a runtime violation. Check how "no solution found" fits D18's
  "refuse what would fail": it is not evidence that anything would fail.
- **B3. Saved plans.** Preparation that uses measurements, free memory or
  earlier runs can derive a different plan, and `load_prepared` refuses a
  saved form that differs from a fresh derivation
  (`library/iro/prepare/prepared.py:188`). The preferred direction is propose
  and validate: save the decision and its evidence, and let a checker
  establish eligibility on load. Not designed; the code keeps re-deriving and
  comparing until it is.
- **B4. Numerical guarantees.** Applicability and guarantees per
  transformation, not one global tolerance. Run-level drift claims need their
  own support. A bit-identical requirement concerns results: another kernel
  qualifies with evidence scoped to what it covers. First-use verification
  checks whichever property the run requires. Reconcile with spec
  workflow-composition, "Numerical choices are fixed for the run" (476-493),
  once delegated choices can change numbers during a run, as kernel selection
  under memory pressure can (D15, about 3383).
- **B5. Random numbers.** Independent streams per drawing call are desirable.
  Open: stream identity (not variable names), keying repeated sample
  occurrences, reconstructing counters on resume, library code that keeps its
  own generator state, and isolating calls. Consistency within a run is
  required (A6); identical draws across different executions are not a
  default invariant.
- **B6. Stream and request bounds** (`outstanding`, `ready`,
  `max_outstanding`). Unresolved. Map what depends on them: items taken but
  unsettled, producer state, resume, and the versions production observes
  under bounded lag. Keep the current implementation.
- **B7. Retry and reporting.** D15 retries an overlapped out-of-memory failure
  only for work without owner effects. If reporting is such an effect,
  reporting at the end helps only before publication, and repeatable reporting
  would need a rule change.
- **B8. Planning, coordination and async.** This is the scheduler/coordinator/
  supervisor topic. Planning may adapt to resources, measurements and runtime
  conditions; ranks are coordinated where they interact (collectives,
  sharding, assignment of work), not identical; planned and actual execution
  are both recorded; reusable performance evidence needs an identity context
  beyond code identity. The runtime surveys (Realm, Legion, StarPU, Parla,
  IREE, Checkmate, DTR) are references for this and for the async topic. An
  external runtime experiment waits until a specific cost or limit is
  identified. The overhead pass (syv.53) and its gate before section 18's
  acceptance still apply.
- **B9. A deliberate gradient cut** (`iro.train.fixed` in the sketch) for a
  value that depends on trained state but should count as a constant, as in
  self-distillation from the weights being trained. Proposed only.
- **B10. Storing knowledge about code.** Where it lives and how people and
  programs update it is not decided (topic 1). Related: `sd-scripts-8bi`.

## C. Decisions needed

- **C1. Unknown correctness facts.** Agreed in topic 1: the system does not
  silently assume a fact; it tells the author what it does not know, and the
  author states the fact or deliberately chooses cautious handling. Later
  feedback: the author may not know either, so the system should use its own
  means first, consider a conservative execution that covers the actual
  uncertainty, and otherwise report the evidence or integration limit.
  Proposed reconciliation:
  - The system's own means (analysis, checks, observation) come first.
  - Where the uncertainty only limits which realizations are valid, the
    system picks one that is valid under every possibility. That is a
    delegated choice and needs no question.
  - A fact needed only to enable an optimization leaves that candidate out,
    without asking.
  - Where the uncertainty concerns the authored behavior itself, such as
    whether a gradient passes or whether a call is stochastic, the topic 1
    rule stands: the author decides, or the system reports the limit.
- **C2. The Trainer's host role** (3506-3525; "What the Trainer consumes",
  4459-4471; spec accepted-training-execution, 285-297; spec
  training-contract, 192-200). Raised, not discussed: setup, execution,
  failure and cleanup are not specific to training and would sit with the IR
  runtime, while training keeps units, their optimizer state and the code
  behind `contribute` and windows.
- **C3. Run settings and fixed requirements.** Run settings can add
  requirements but cannot drop one the strategy fixes, since configuration
  only fills what the strategy offers. Proposed, not discussed.

## D. Follow-ups outside the design (after review)

- CLAUDE.md, "Building the New System", and
  `tests/unit/test_dependency_rules.py`: the library rule from A1, with a check
  that libraries do not import `library.iro`.
- Piece 1 (`sd-scripts-syv.62.4`): `library/losses` and `library/objectives`
  drop `@implementation` and written-form types and become plain functions.
- `tasks.md`: the tasks that build per-use access selection, the overlap
  permission and `may_leave_device` (section 23) follow A4 and A5.
- The sketch: the producer's run-ahead becomes open (B6), and `iro.report`
  inside validation's loop is noted against B7.
