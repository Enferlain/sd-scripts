# Decision and provenance notepad

This is a retrospective reading aid, not another specification. Source claims,
our confidence in their trace, and the architecture's existing disposition are
different things. Entries below follow the joint walkthrough through reading 19,
ending at the pre-audit design frontier. The governing requirements remain in
force; this notepad reconstructs their history and does not replace them.

The [historical source archive](../../archive/models-strategy-trainer/README.md)
now holds the earlier documents. Their bytes and this notepad's decision
dispositions are preserved. Old capture paths resolve through
[archive-relocations.json](archive-relocations.json); current navigation follows
the move.

For each entry, retain: decision or proposal; original source and rationale;
subsequent clarification or replacement and its source; current disposition;
missing detail or G5 consequence; and document placement implications.

Source shorthand:

- **July document**: [exact first committed revision](snapshots/de2dd0eb-strategy_system_direction.md),
  `de2dd0eb`, original path `docs_design/strategy_system_direction.md`.
- **July conversation**: [readable transcript](sessions/00-initial-direction.md),
  backed by stable identities and raw-record locations in its JSON companion.
- **Current design**: [governing OpenSpec](../../../openspec/changes/rework-model-strategy-trainer/design.md).

<a id="h001"></a>

## H001 - The contract defines acceptable interaction with the pipeline

**Decision and original source.** The contract is the pipeline's deliberate
acceptance vocabulary, requirements, choices and compatibility rules. It is
not a nominal universal SD-shaped method list or a prediction of every future
model. July document lines 96-178 and 551-583; July conversation
[user message 030](sessions/00-initial-direction.md#m030), with the clarification
in [032](sessions/00-initial-direction.md#m032).

**Original rationale.** Models can have arbitrary internal definitions. The
pipeline must state how supported behavior can participate, and selected
alternatives create real obligations. Calling everything an optional
capability would avoid specifying those obligations. Supporting new accepted
behavior may deliberately expand the contract and its conformance evidence.

**Current disposition.** Carried forward and made more explicit by
[`training-contract`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-contract/spec.md):
the active contract precedes authoring; authors select and wire explicitly;
fulfillment returns acceptance or detailed rejection and specializes one
authoritative obligation set. The current design's source crosswalk identifies
the inherited direction. H046-H047 now trace the September refinement of that
shared system; the current text is not proof that all of it was settled in July.

**Missing detail or G5 consequence.** Concrete contract/fulfillment APIs,
diagnostic placement and obligation types remain G5 work under the established
requirements. Neither choosing xDSL nor choosing concise IR primitives
authorizes removing selected meanings or returning to a loose callback bag.

**Placement.** Preserve the July document as an origin snapshot and the current
direction as inherited input. New normative changes belong in the governing
design/deltas, not in this notepad or a rewritten historical narrative.

<a id="h002"></a>

## H002 - Reusable features and components serve deliberately assembled behavior

**Decision or direction and original source.** Organize reusable behavior by
what it does, distinguish a component from its use in this repository, and
allow tested selections beyond a permanent pixel/latent or training-subject
class identity. July document lines 232-446; July conversation
[032](sessions/00-initial-direction.md#m032),
[034](sessions/00-initial-direction.md#m034) and
[036](sessions/00-initial-direction.md#m036).

**Original rationale.** Representation, conditioning, adapters, attached
trainables and distillation can vary independently of a family's default
assembly. A discoverable catalog preserves useful existing separation while
making those responsibilities intelligible. The component "grab box" was
explicitly a longer-term direction; arbitrary construction did not establish
compatibility.

**Current disposition.** The distinction survives in the current direction's
"Model, Strategy, And Trainer Boundaries" and in the governing contract's
explicit participant/relationship/feature/capability selections. Selected
behavior must survive in the accepted arrangement. The illustrative
`features/` tree, component-hosting choices and decorator sketches were not
therefore all accepted production layout decisions. Their individual adoption
trails remain to be read.

**Missing detail or G5 consequence.** Determine concrete maintained behavior,
participant and binding types and production module dependencies without
reconstructing a family x mode x objective class matrix. Preserve component
mechanics versus workflow-use semantics even where their eventual package
names differ from the July sketches.

**Placement.** Historical catalog sketches remain design evidence. Accepted
module responsibilities belong with the governing migration design and,
when implemented, the owning module documentation.

<a id="h003"></a>

## H003 - A complete strategy is broader than one model-family integration

**Source and original clarification.** July document lines 210-223 distinguish
family integration from a fulfilled training strategy. July conversation
[049](sessions/00-initial-direction.md#m049) contains imported reviewer opinion
and the user's qualification; [050](sessions/00-initial-direction.md#m050)
contains the assistant's reconciliation. Keep those speakers distinct.

**Original rationale.** A run may involve teacher and student families, an
adapter, an added component and an objective. Calling the complete result a
"fulfilled family strategy" would mistake a maintained default assembly for
the general system boundary. Correcting terminology did not itself choose
separate classes or runtime owners.

**Current disposition.** The current governing proposal defines explicit
strategy authoring and one accepted arrangement with participant and
relationship identities. Ordinary execution depends on accepted runtime
implementations, not the authoring role. The family-integration distinction is
preserved; the precise July name for the final runtime object is not.

**Missing detail or G5 consequence.** Concrete authoring, acceptance and
execution types must represent multi-participant arrangements faithfully.
Neither one family object nor one illustrated training loop defines the
expressiveness criterion.

**Placement.** Keep the external wording in the conversation/history. The
accepted distinction belongs in governing requirements; the imported review
is not an independent normative source.

**Reading 03 direct definition.** User
[018](sessions/03-training-mode-and-objective.md#m018) states explicitly that
strategy means what is to be trained and how, with standard model-family
strategies as maintained examples. User
[022](sessions/03-training-mode-and-objective.md#m022) rejects the assistant's
adapter-specific strategy example; user
[024](sessions/03-training-mode-and-objective.md#m024) clarifies that it would
regress past the current mode-separated architecture to an older design.

<a id="h004"></a>

## H004 - The authored recipe must not be confused with an active runtime strategy

**Original intent, competing proposal and sources.** In July conversation
[017](sessions/00-initial-direction.md#m017), the user explicitly questions
Trainer callbacks after asking that strategy produce one thing. The draft
nevertheless proposes a bound per-run strategy with `load`, `prepare` and
`training_step` in July document lines 509-549. The document labels that
direction insufficiently settled for implementation and lists it among
leanings at lines 692-702. General approval at
[058](sessions/00-initial-direction.md#m058) does not erase those labels.

**Original rationale.** Both proposals respond to split state, component
arguments being passed back to the strategy that loaded them, and the desire
for a coherent handoff. The bound-service solution appeared to hide family
anatomy, but it did not establish that the authoring object should remain the
runtime collaborator. A proposed `PreparedTrainingProgram` with the same
callback methods was separately rejected as a synonym, not as proof that
authoring can produce no executable result.

**Replacement source.** The dated
[26 August topology correction](../../archive/models-strategy-trainer/notes.md#accepted-run-arrangement-topology-correction-2026-08-26)
explicitly replaces active-strategy topology with authoring -> fulfillment ->
accepted arrangement -> governed preparation -> one Trainer engine. It
explicitly preserves the already settled binding, identity, preparation,
publication, persistence and optimization decisions. Reading 10 now traces
the discussion/session chain behind that correction, separately from the
later choice of implementation substrate.

**Current disposition.** Active authoring-strategy callbacks are superseded.
The current direction's "Authoring, Accepted Arrangement, And Runtime State"
and the governing
["Accepted execution survives the authoring object" requirement](../../../openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#requirement-accepted-execution-survives-the-authoring-object)
state the replacement. A selected object can implement an accepted runtime
seam; that does not grant its authoring interface runtime authority.

**Missing detail or G5 consequence.** Design the concrete accepted/executable
representation and production authority/state types while preserving selected
domain behavior. Do not reintroduce a renamed active strategy, duplicate
canonical state or a Trainer that has to rediscover authored choices.

**Placement.** Retain the original leaning and the explicit correction as
separate historical records. A current architectural description must lead
with the corrected topology, not quote the old proposal without its status.

**Reading 01 trace.** The inventory follow-up makes this drift especially
visible. User [014](sessions/01-inventory-and-authoring.md#m014) reiterates a
complete training plan without Trainer knowledge of its parts. Assistant
[015](sessions/01-inventory-and-authoring.md#m015) turns that into a narrow
public façade and proposes live `strategy.bind`, `prepare` and `training_step`
calls. After user [033](sessions/01-inventory-and-authoring.md#m033) says the
Trainer dictates the contract, assistant
[035](sessions/01-inventory-and-authoring.md#m035) again pairs the correct
consumer-defined contract principle with that active-strategy implementation.
The former survives; the latter is the distinct proposal later superseded.

**Reading 02 trace.** The source audit corrects the oversimple Lightning
analogy, but assistant [028](sessions/02-frameworks-and-contract-surfaces.md#m028)
still proposes an active strategy that owns bindings and receives per-batch
calls. After the user's objection, assistant
[030](sessions/02-frameworks-and-contract-surfaces.md#m030) additionally asserts
that fulfillment is only a property and should not have a separate object or
class. Neither assertion is the eventual architecture. The current
[comparison header](../../archive/models-strategy-trainer/framework_pattern_comparison.md#status-and-purpose)
explicitly marks that runtime topology superseded while preserving its source
lessons. User objections to renamed abstractions did not prohibit acceptance
from producing a distinct executable arrangement.

**Reading 03 trace.** Removing mode as a parallel authority does not settle
whether the strategy authoring object remains active. Assistant
[019](sessions/03-training-mode-and-objective.md#m019) correctly broadens
strategy beyond model family but illustrates the result as `Trainer <->
strategy` and again claims no distinct fulfilled/plan abstraction is needed.
Those containment/lifetime conclusions remain proposals, later superseded.
The authored training meaning survives in the accepted arrangement; it does
not disappear with the old active-strategy topology.

**Reading 06 trace.** Q5 moves that enclosure beyond the earlier leaning:
[026](sessions/06-authority-reviews-and-handoff.md#m026),
[029](sessions/06-authority-reviews-and-handoff.md#m029) and the approved
checkpoint [034](sessions/06-authority-reviews-and-handoff.md#m034) explicitly
put a dedicated authority inside a complete active strategy. `f603d074`
commits that answer; `d7c4de7b` retains it after correcting conservative-policy
wording. The August 26 correction therefore supersedes an adopted historical
enclosure, while preserving Q5's authority semantics. H028 records that
distinction; the raw August 21 diagrams are not current runtime architecture.

**Reading 10 trace.** User
[040](sessions/10-accepted-arrangement-and-openspec-transition.md#m040) makes
the authoring-only expectation explicit. The first reconstruction
[048](sessions/10-accepted-arrangement-and-openspec-transition.md#m048) is
qualified after the full old session is reread in
[052](sessions/10-accepted-arrangement-and-openspec-transition.md#m052): the
active-strategy choice had motivations and bundled Q5 approval, but the
enclosure had not been isolated from the valid authority decision. User
[079](sessions/10-accepted-arrangement-and-openspec-transition.md#m079) requests
the record correction; [101](sessions/10-accepted-arrangement-and-openspec-transition.md#m101)
and [105](sessions/10-accepted-arrangement-and-openspec-transition.md#m105)
record it and repair the remaining stale establishment-result section.
`bb763b59` commits that corrected topology.

<a id="h005"></a>

## H005 - Deferring graph design was a bounded sequencing decision

**Decision and original source.** July document lines 625-635 and 677-690;
July conversation [026](sessions/00-initial-direction.md#m026), specifically
the user's request after the embedded response annotation. Graph exploration
was deferred while the model/strategy/Trainer responsibilities were clarified.

**Original rationale.** A genuinely executable graph was different from
renaming callbacks, but the discussion had not yet resolved the existing
system's boundaries. The document explicitly says this pass neither requires
nor forbids a future graph.

**Later change and current disposition.** The 26 August correction makes a
structured accepted representation necessary and stops relegating graph-shaped
execution to a hypothetical future. Current design D14 records the user's
**2 November xDSL substrate selection** for representation, analysis,
transformation and lowering. That selects machinery within the established
architecture. It does not reset the architecture or automatically select a
numerical backend. The intervening research/candidate comparison remains a
later walkthrough phase.

**Missing detail or G5 consequence.** Establish which accepted meanings need
explicit representation and how they lower, including selected finite
numerical regions where qualification supports it. The July deferral cannot
be used as a current prohibition on IR/regions, and xDSL's availability cannot
be used to replace the prior authority and ownership requirements.

**Placement.** Preserve the deferral with its date and scope. Keep the current
substrate decision and its evidence boundary in D14; link later experiments
as evidence rather than revising the July record to appear prescient.

**Reading 10 trace.** User
[068](sessions/10-accepted-arrangement-and-openspec-transition.md#m068) explains
why graphs first arose: the recipe needs accepted executable meaning and a
strong research path. The later refinement
[073](sessions/10-accepted-arrangement-and-openspec-transition.md#m073) requires
structured composition without choosing a literal graph over regions,
schedules or another equivalent representation. This is the historical bridge
from bounded graph deferral to the eventual IR investigation.

<a id="h006"></a>

## H006 - Decorators, names and package sketches were options, not requirements

**Original source.** July document lines 291-327; July conversation
[051](sessions/00-initial-direction.md#m051),
[053](sessions/00-initial-direction.md#m053) and
[055](sessions/00-initial-direction.md#m055). The user explicitly states the
decorator suggestion is not a requirement and asks to record alternatives for
later evaluation.

**Original rationale.** Avoid making long category suffixes carry all meaning;
imports, package context, descriptors or decorators might express it better.
A decorator needs useful contract meaning if adopted. None of this answers
the system's authority or runtime topology by itself.

**Current disposition.** These July API and naming suggestions remain
unselected by the sources checked in this reading. D14 fixes Python authoring
as a premise while leaving the public authoring API and production module
placement to G5. "Python authoring" does not settle every Python spelling.

**Missing detail or G5 consequence.** Evaluate concrete APIs against the
already established contract and executable meaning. Do not treat a historical
syntax example, feature suffix or illustrative folder as an accepted interface.

**Placement.** Keep alternatives in historical/proposal material until a
specific adoption is documented. Current module READMEs should describe
implemented behavior or settled bounded decisions.

**Reading 07 trace.** The user raises readability in
[054](sessions/07-contract-exchanges-and-authority-correction.md#m054), then
explicitly declines making a rigid rule without seeing code in
[056](sessions/07-contract-exchanges-and-authority-correction.md#m056).
[058](sessions/07-contract-exchanges-and-authority-correction.md#m058) favors
organic progress and refinement after a coherent base exists. Analytical names
are therefore scaffolding, not accepted public API names or a newly imposed
readability checklist. Readability remains something to assess on concrete
work, rather than an excuse to reopen settled responsibilities.

**Reading 10 trace.** The decorator proposal remains illustrative. User
[074](sessions/10-accepted-arrangement-and-openspec-transition.md#m074) recalls
it as one idea, and [077](sessions/10-accepted-arrangement-and-openspec-transition.md#m077)
explicitly repeats that qualification after the detailed sketch. The sketch
does not select decorator names, grant APIs, discovery rules or final Python
surface for the language.

<a id="h007"></a>

## H007 - The author makes choices; validation does not assemble the strategy

**Decision and original source.** In reading 01, user
[018](sessions/01-inventory-and-authoring.md#m018) explicitly says the run does
not select or infer the features: repository authors write the strategy, just
as they write model code. Configuration may select a deliberately exposed
alternative; it is not an initial goal to build a Hydra replacement or an
automatic assembler. User [023](sessions/01-inventory-and-authoring.md#m023)
confirms that this preserves the overall goal and asks to record it.

The parent branch supplies an earlier version of this correction at
[033](sessions/01-branch-differences.md#m033), followed by user
[035](sessions/01-branch-differences.md#m035): the builder is literally whoever
writes the code. These are distinct turns and branch versions, not a second
independent architecture discussion.

**Original rationale.** A requirement such as "latent representation needs
compatible encoding" authorizes checking that the author supplied it. It does
not authorize choosing a VAE, cache codec or objective on the author's behalf.
The contract defines the place and rules; the catalog offers implementations;
the author chooses and connects them. Assistant
[030](sessions/01-inventory-and-authoring.md#m030) states that separation after
the user's question about "contract filing."

**Adoption and current disposition.** The clarification enters the direction
document's committed history as "Authored Fulfillment, Not Automatic Assembly"
in `97a81ef4` on 29 July at 22:11 UTC, after its earlier conversation and
editing. It is carried into the governing
[`training-contract`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-contract/spec.md)
requirements for explicit authoring, configuration within exposed bounds and
acceptance without adding dependencies or inferring intent. Runtime bounded
choices remain possible; intentional authoring does not require deciding
every runtime value in advance.

**Interpretation to avoid.** In the parent branch, assistant
[036](sessions/01-branch-differences.md#m036) also infers that there should be
no `TrainingStrategyDefinition` or fulfillment engine. That is the assistant's
additional code-shape conclusion, not the literal user instruction. Current
requirements explicitly require a fulfillment/acceptance mechanism producing
an accepted arrangement. Rejecting automatic intent selection does not reject
verification, normalization, preparation, lowering or an accepted IR artifact.

**Missing detail or G5 consequence.** Define concrete authoring and acceptance
APIs that preserve explicit selected behavior. Specify shared conformance and
diagnostics without making maintainers write per-strategy validators. An
internal builder helper or compiler stage must be judged by what it does;
neither its name nor this historical wording decides its validity.

**Placement.** The authoring invariant belongs in the governing contract.
Keep illustrative factories, class names and the branch's overbroad inference
in the historical transcript with their status, rather than promoting them
into module or API requirements.

**Reading 02 refinement.** The feedback referenced in user
[036](sessions/02-frameworks-and-contract-surfaces.md#m036) distinguishes
requesting a provided capability from selecting its implementation. Assistant
[038](sessions/02-frameworks-and-contract-surfaces.md#m038) carries that into
the direction update. A configuration request for sampling cadence/settings
does not necessarily choose an internal arrangement at all. An intentionally
exposed arrangement choice is a different supported option; neither creates
an implicit assembler.

<a id="h008"></a>

## H008 - Research freedom and conformance are compatible requirements

**Decision and original source.** User
[020](sessions/01-inventory-and-authoring.md#m020) rejects treating the system
as a restricted beginner mode and preserves experimentation beyond maintained
model defaults. Assistant [022](sessions/01-inventory-and-authoring.md#m022)
distinguishes custom implementations of the active boundary from explicit
extensions changing what the Trainer must do. User
[023](sessions/01-inventory-and-authoring.md#m023) says the overall goal has
not changed and asks to update the direction.

**Original rationale.** The reusable standard catalog should make supported
work coherent without making its current decomposition the limit of research.
Equally, a custom implementation cannot claim an unchanged contract while
silently changing observable responsibilities. The cited `sd-mecha` comparison
is an analogy supplied during the discussion, not adoption of its runtime,
model semantics or graph. Its technical claims are retained as historical
discussion, not independently requalified in this walkthrough.

**Adoption and current disposition.** "Conformance Without A Noob Mode" enters
the direction's Git history with `97a81ef4`. The governing contract now states
the maintained shared-conformance path, explicit custom conformance, and
supported execution/ownership profiles. Accepted structured or imperative
runtime behavior has declared authority; custom work does not revive an
active authoring-strategy callback or silently inherit permission from a
library wrapper. The later detailed authority work is traced in H041 and H067.

**Missing detail or G5 consequence.** Provide an explicit extension/conformance
boundary preserving declared effects, ownership, outcomes and settlement.
Separate maintained implementations from the mechanism that verifies their
meaning. Neither a universal image/representation checklist nor an unchecked
callable escape is an adequate realization of this requirement.

**Placement.** The requirement belongs in the governing contract and execution
specs. Keep reference-framework analogies subordinate to that meaning; do not
turn them into a competing architecture or a mandated package vocabulary.

<a id="h009"></a>

## H009 - Intended consumer responsibilities determine the contract

**Decision and original source.** User
[033](sessions/01-inventory-and-authoring.md#m033) asks whether this should be
dictated by the Trainer, which also dictates the contract. Assistant
[035](sessions/01-inventory-and-authoring.md#m035) agrees and qualifies this as
the intended training pipeline, rather than every incidental dependency of
the current Trainer implementation.

**Original rationale.** A contract should express what the intended consumer
legitimately needs. Packaging the existing VAE/text-encoder/denoiser fields in
a bundle does not make their model-specific assumptions generic. Typed
requests, plans and results can be appropriate where their meanings match
actual responsibilities. Whether a boundary is a method, value or bundle is
not the responsibility decision itself.

**Adoption and current disposition.** The intended-Trainer derivation appears
in the direction revision committed in `97a81ef4`. The governing proposal and
design retain it: intended Trainer mechanics and deliberately supported
pipeline capabilities define the contract before authoring. This does not
mean a particular Trainer instance owns the contract definition. The active
strategy-method examples proposed in the same reply are separately superseded
under H004. The later whole-run/shared-language work broadens the machinery;
it does not invalidate this responsibility-based contract derivation.

**Missing detail or G5 consequence.** Derive concrete exchanges from the
established consumers, supported behavior and accepted run authority. Preserve
generic optimization/preparation mechanics and selected domain computation
without copying today's mutable Trainer fields or converting every private
feature into a central-engine operation. Exact module and type placement
remains a concrete G5 question within that design.

**Placement.** Keep the principle in the governing contract/design. The
historical suggestion to put an entire contract under `library/training/`
is a placement proposal in assistant 035, not a settled consequence of the
consumer-defined principle.

**Reading 02 refinement.** User
[034](sessions/02-frameworks-and-contract-surfaces.md#m034) explicitly says
the comparison neither supersedes prior direction nor requires preserving
today's Trainer mechanics. Assistant
[048](sessions/02-frameworks-and-contract-surfaces.md#m048) then proposes
deriving binding, preparation, optimization and execution exchanges jointly,
before naming classes. Consumer-defined contract meaning is not a requirement
to design one side in isolation or copy its existing fields.

<a id="h010"></a>

## H010 - A narrow starting point did not settle a universal scalar-step model

**Original request and additional proposal.** User
[016](sessions/01-inventory-and-authoring.md#m016) asks to derive the contract
from current reality, intended direction and the needs of training, allowing
a technically narrow starting point. Assistant
[017](sessions/01-inventory-and-authoring.md#m017) adds a proposed v1 with one
final differentiable loss, one optimizer plan and one Trainer-controlled step.
Those exact limits are the assistant's proposal, not the user's stated
definition of the eventual system.

**Subsequent qualification.** User
[020](sessions/01-inventory-and-authoring.md#m020) emphasizes research freedom;
assistant [022](sessions/01-inventory-and-authoring.md#m022) itself says a
scalar loss is not universal and discusses multiple losses, optimizers and
phases. The direction revision at `97a81ef4` adopts explicit authoring and
conformance; it does not install the proposed one-loss/one-optimizer list as a
universal rule. No independent acceptance of that list was established in
this passage.

**Current disposition.** The later governing optimization and execution
requirements define independent semantic units, contribution/advancement
policies, whole-run work and profile-governed authority. The
[`training-optimization`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md)
spec separates semantic membership and unit identity from backend optimizer
objects. Its full historical derivation remains a later reading. Scalar
handling in a bounded experiment cannot substitute for those requirements.

**Missing detail or G5 consequence.** Separate an initial supported
realization's limits from language meanings and production coverage. Concise
IR primitives are useful only if they preserve the independently established
data, state, gradient, ownership and lifetime distinctions. A historical wish
to start concretely is not authority to discard them.

**Placement.** Record bounded target limitations in their experiment or target
documentation. Keep accepted general meanings in the governing requirements;
retain this proposed v1 list as history, not a current core language contract.

**Reading 02 trace.** Assistant
[033](sessions/02-frameworks-and-contract-surfaces.md#m033) again proposes a
diffusion-shaped required-concern list and preserving `BatchLossOutput` with
timestep observations. User
[034](sessions/02-frameworks-and-contract-surfaces.md#m034) corrects that
framing in favor of the intended Trainer, its capabilities and authoring
features. Those particular result fields and one-batch protocol were not
thereby established as universal core requirements.

**Reading 03 code correction.** Assistant
[005](sessions/03-training-mode-and-objective.md#m005) reports that the actual
loop backpropagates a modification of `per_sample_loss`, while `loss` is used
for accounting. This corrects the earlier claim that the existing result was
a final backward-ready scalar. Assistant
[012](sessions/03-training-mode-and-objective.md#m012) then recommends a
base-per-sample result. That recommendation is not a universal contract:
the governing execution requirement later separates computation,
optimization, accounting, observations and effects without fixing one loss
producer, tensor or advancement sequence.

<a id="h011"></a>

## H011 - The production inventory is evidence, not a replacement specification

**Source and original purpose.** User
[003](sessions/01-inventory-and-authoring.md#m003) asks for inventory first and
records between milestones. The first committed
[inventory snapshot](snapshots/16801f38-strategy_system_inventory.md) states
its evidential role at lines 3-12. Its state/placement conclusions and
conformance/open-question sections occupy lines 683-1089. Assistant
[013](sessions/01-inventory-and-authoring.md#m013) summarizes the resulting
split-state, mixed-responsibility and nominal-conformance findings.

**Original rationale.** Existing method presence and an aggregate strategy
object do not prove coherent selected behavior or state ownership. Code
observations constrain a redesign, but current call sites and file locations
do not dictate the future contract. The inventory explicitly says placement
changes should follow responsibility and state-boundary design.

**Current disposition.** The governing design classifies the inventory as
code/scenario evidence, distinct from normative direction and adopted deltas.
Its observations, proposed classifications and proposed future contracts must
be distinguished. For example, the first inventory calls `TrainingMode` a
correct axis; the later accepted design deliberately dissolves that runtime
axis. It also treats loaded-component declarations as a strong existing
foundation, while the accepted run authority later owns canonical current
bindings. These are not reasons to discard the inventory; they are reasons to
trace each proposed conclusion's later disposition.

**Missing detail or G5 consequence.** Use the established target and completed
consumer derivation to decide production types, module dependencies and
migration. Refresh historical code observations when an implementation relies
on them. This walkthrough has not rerun the July production-code audit or
revalidated its old graph/source line references against today's checkout.

**Placement.** Preserve the dated baseline and evidence standard, with links
to later decisions. Do not present the first inventory as current production
documentation, an accepted package tree, or independent implementation
qualification. The branch's 729-line draft and the committed 1,089-line
inventory are different records; review assertions do not transfer between
them automatically.

**Reading 03 explicit supersession.** The
[Milestone 6 revision](snapshots/707015ee-strategy_system_inventory.md#milestone-6-training-mode-and-objective-responsibility-classification)
opens by superseding Milestone 4's "correct axis" description. The
conversation, direction consolidation and later classification commit provide
a concrete trail from the old placement judgment to responsibility-based
mode dissolution. This is not merely our retrospective preference.

<a id="h012"></a>

## H012 - References inform realization without taking over design authority

**Decision and source in this reading.** User
[007](sessions/02-frameworks-and-contract-surfaces.md#m007) makes design control
and maintenance responsibility the test for adoption; user
[009](sessions/02-frameworks-and-contract-surfaces.md#m009) favors using the
frameworks as references. User
[034](sessions/02-frameworks-and-contract-surfaces.md#m034) explicitly restores
the prior direction's authority after repeated reopening of settled points.

**Original rationale.** Similar-looking APIs do not show whether another
framework's state ownership, lifecycle, extension assumptions or separable
implementation serve this repository. Vendoring/forking also transfers real
maintenance. A reference can improve a mechanism without selecting the whole
runtime, object hierarchy or problem definition.

**Evidence correction.** Assistant
[020](sessions/02-frameworks-and-contract-surfaces.md#m020) declares a comparison
complete from a largely public-API reading. User
[021](sessions/02-frameworks-and-contract-surfaces.md#m021) challenges the missing
code evidence. Assistant [022](sessions/02-frameworks-and-contract-surfaces.md#m022)
downgrades that conclusion and [025](sessions/02-frameworks-and-contract-surfaces.md#m025)
reports the replacement source-level audit. The
[first committed comparison](snapshots/97a81ef4-framework_pattern_comparison.md)
is the source-level revision, not the rejected documentation-only draft.

**Current disposition.** The governing design classifies the comparison as
subordinate prior art that does not select the project's ownership model.
This is not a permanent ban on framework reuse or a requirement to choose
custom implementation. D14's later xDSL decision selects reusable machinery
within the project-owned architecture. The comparison's original
active-strategy analogy is independently superseded under H004.

**Missing detail or G5 consequence.** Judge concrete reusable machinery by
whether it preserves required representation, authority and interactions,
what custom machinery remains necessary, and its actual integration and
maintenance burden. Do not reopen the goals because a dependency offers a
different default, or preserve existing production mechanics merely because
they already run.

**Placement.** Keep the comparison as dated source-backed research. Preserve
its correction notice and links to adopted decisions; do not move its old
ownership matrix into current production documentation. Historical claims
about external versions have not been requalified by this walkthrough.

<a id="h013"></a>

## H013 - Logical identity and prepared execution are distinct meanings

**Source in this reading.** Assistant
[025](sessions/02-frameworks-and-contract-surfaces.md#m025) summarizes the revised
audit. The first comparison's "Prepared execution state must be modeled
explicitly" and "The Trainer needs a generic preparation exchange" sections
derive a stable logical identity versus current callable/wrapper distinction.
The direction update reports retaining that lesson in
[045](sessions/02-frameworks-and-contract-surfaces.md#m045).

**Original rationale.** A backend can move, wrap, shard, replace or otherwise
prepare executable objects. Calling an old logical/original module can bypass
required behavior; making the wrapper the semantic identity can instead lose
relationships, provenance or persistence meaning. Infrastructure needs concrete
participants, but that need does not justify model-family topology in the
generic coordinator.

**Current disposition.** The governing participant-state and
[`training-runtime-preparation`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md)
requirements retain identities, routes, views, source revisions and matching
prepared guarantees. Joint backend creation does not merge participant
identity or semantic ownership. The originally proposed *bound strategy*
container was later replaced by authority in the accepted arrangement; that
does not discard this binding distinction. It is not a mandate that exactly
two Python handles represent every participant.

**Missing detail or G5 consequence.** Specify concrete authority, participant,
route/view and preparation request/candidate/publication types with freshness
and correspondence. Keep authority-owned routes, optimization state and backend
coordination under their established owners even when one backend call creates
them together. A retained handle alone cannot establish executable readiness.

**Placement.** The reasoning stays in the comparison; the accepted identity
and preparation obligations live in the governing specs. Experimental wrapper
maps and returned tuples are implementation evidence, not production types to
copy unexamined.

**Reading 03 holder qualification.** Assistant
[012](sessions/03-training-mode-and-objective.md#m012) proposes putting the
binding map on a bound strategy; [014](sessions/03-training-mode-and-objective.md#m014)
then describes authority as one trusted answer rather than a giant state bag,
allowing possible domain partitioning. Consistent bindings survive as a
requirement; those holder sketches are not the settled topology. The current
run-authority contract, including coherence across revisions and transitions,
governs any physical partitioning chosen in G5.

**Reading 04 trace.** The Q2 proposal and feedback distinguish the callable
route from original/unwrapped/inspection/artifact access views. H022 records
that distinction and its normal-case versus state-only qualification; H023
records the freshness rule. A retained original module is not automatically
another authorized forward target.

<a id="h014"></a>

## H014 - Core, capability and feature are distinguished by their consumers

**Decision and source in this reading.** User
[034](sessions/02-frameworks-and-contract-surfaces.md#m034) distinguishes the
Trainer's necessary contract, additional pipeline-supported behavior, and
building blocks used internally when authoring. Assistant
[035](sessions/02-frameworks-and-contract-surfaces.md#m035) organizes these as
three surfaces. The [external feedback](snapshots/2026-07-29-contract-surfaces-feedback.md)
then sharpens the consumer rule; it is not the original author of the user's
three-category request. Assistant
[038](sessions/02-frameworks-and-contract-surfaces.md#m038) also clarifies
delegation beyond the central Trainer class.

**Original rationale.** These answer different questions: what the intended
consumer universally requires, which named operations its pipeline can
coordinate, and which reusable contracts authors use to supply that behavior.
Capability does not mean an unstructured optional method. Selected obligations
are real, and the core can require a compatible capability from a category.
A capability can use several features internally without exposing them as
separate central-engine hooks.

**Adoption and current disposition.** User
[039](sessions/02-frameworks-and-contract-surfaces.md#m039) requests the update;
assistant [045](sessions/02-frameworks-and-contract-surfaces.md#m045) reports it;
user [046](sessions/02-frameworks-and-contract-surfaces.md#m046) says they have
read it and proceeds. The section first enters Git in `78df512d` on 30 July,
after the conversation. Current direction and governing contract/capability
requirements preserve the consumer-defined distinction. They do not prescribe
three classes, registries, folders or inheritance levels.

**Missing detail or G5 consequence.** Make the concrete interfaces and their
obligations intelligible. Pipeline-recognized does not mean implemented in the
Trainer class: data, persistence and other coordinators can own their actual
mechanics. Conversely, an internal feature's existence alone must not expand
the engine API. The shared workflow language and one training consumer's core
contract are different layers; a data-only workflow is not required to invent
a training loss to use the shared machinery.

**Placement.** Preserve the distinction in the governing contracts and map
concrete implementation responsibilities when G5 derives the modules. The
illustrative capability attributes, `Clip` class and package imports in the
conversation remain API sketches.

<a id="h015"></a>

## H015 - Contract category and lifecycle/readiness are independent dimensions

**Source and adoption in this reading.** The feedback attached to user
[036](sessions/02-frameworks-and-contract-surfaces.md#m036) explicitly separates
core/capability/feature classification from authored, validated, bound and
prepared states. Assistant [038](sessions/02-frameworks-and-contract-surfaces.md#m038)
accepts that refinement with the qualification that not every item needs
behavior in every state. It is carried into the same requested direction
update as H014 and first committed under its own heading in `78df512d`.

**Original rationale.** A capability can exist in an authored definition before
its concrete state is bound or executable. Calling all of these things
`TrainingStrategy` obscures what facts exist, which uses are legal, which
failures are already knowable, and which prepared binding must receive work.
Adding a feature or capability does not itself prove readiness.

**Current disposition.** The governing design preserves semantic acceptance
versus executable readiness, evaluation at the earliest authoritative evidence
point, and governed preparation/publication. It permits intentionally unbound
state with retained readiness obligations. H004 later separates the authoring
role from the accepted arrangement explicitly; the lifecycle insight was not
a requirement for an active strategy to exist in each state.

**Missing detail or G5 consequence.** Define concrete evidence, candidate,
attempt, readiness and publication types and legal transitions. Do not replace
their distinct meanings with one boolean `ready`, equate accepted declarations
with backend support, or require exactly four public Python classes because
the historical diagram has four columns.

**Placement.** Keep lifecycle and readiness requirements alongside their
governing authority/preparation/execution contracts. Concrete experimental
containers remain evidence for the implementation choice rather than the
definition of the lifecycle.

<a id="h016"></a>

## H016 - Trained-artifact persistence and exact execution restoration differ

**Sources in this reading.** The comparison's "Runtime snapshots and model
artifacts remain separate" finding distinguishes resuming an execution from
publishing a logical trained result. The attached feedback asks for separate
named contracts. Assistant [038](sessions/02-frameworks-and-contract-surfaces.md#m038)
refines the ownership: pipeline checkpointing coordinates the resumable run;
selected runtime behavior contributes state and restoration semantics. That
does not make the strategy the owner of checkpoint orchestration.

**Original rationale.** Optimizer, scheduler, RNG, coordinates and backend state
serve execution continuation. Model/adapter products, formats, provenance and
publication serve trained-artifact consumers. They may share extraction and
consistency work without becoming one overloaded `Checkpointing` concern.
The pipeline coordinating an operation does not take ownership of every
domain-specific serializer or numerical behavior it invokes.

**Adoption and current disposition.** The direction update explicitly reports
runtime-state contribution versus trained-artifact persistence; its separate
section first enters Git in `78df512d`. The governing
[`training-artifact-persistence`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-artifact-persistence/spec.md)
spec preserves distinct artifact identity and exact runtime restoration, with
pipeline coordination and selected domain contributions. The external
suggestion `runtime_state: RuntimeCheckpointing | None` is an API sketch, and
its ownership implication is explicitly qualified in the conversation.

**Missing detail or G5 consequence.** Derive artifact declaration/request/plan/
result types and restoration contributors from the accepted meanings, including
consistency, identity, actual outputs and failure. Cold-loading a semantic or
prepared IR artifact is not by itself exact restoration of a running system's
state. A model export likewise does not establish same-run continuation.

**Placement.** Keep the shared requirements in the governing persistence and
participant-state specs. Production documentation should distinguish the
pipeline coordinator from selected serializers/state contributors; neither
the old family checkpoint file nor a generic backend checkpoint is the whole
architecture.

**Reading 05 trace.** Q4 [024](sessions/05-arrangement-and-persistence.md#m024)
tests the distinction against model, adapter, capability and resume paths.
The supplied [notices](snapshots/2026-08-21-q4-notices.md) are evaluated and
qualified in [028](sessions/05-arrangement-and-persistence.md#m028), then the
user requests the checkpoint in [029](sessions/05-arrangement-and-persistence.md#m029).
`e0613c2f` first commits the settled Q4 section. H026-H027 preserve the
additional product, capture and result distinctions without turning exact
restoration into another artifact kind.

<a id="h017"></a>

## H017 - TrainingMode dissolves by responsibility, without a replacement axis

**Correction and source.** The passage first tries moving mode inside a family
strategy, then restoring it as a permanent peer. User
[013](sessions/03-training-mode-and-objective.md#m013) challenges the first
move; [015](sessions/03-training-mode-and-objective.md#m015) and
[018](sessions/03-training-mode-and-objective.md#m018) clarify the intended
strategy/Trainer relationship. User
[020](sessions/03-training-mode-and-objective.md#m020) favors explicit features/
capabilities but leaves specialized placement dependent on the code. The
adapter-specific class in assistant 021 is rejected in user 022 and its
historical interpretation corrected in user 024.

**Original rationale.** Family behavior, training treatment and objective are
ingredients of an authored training definition. They do not each need a peer
orchestrator with the whole Trainer. Moving every old mode method into one
central engine would instead accumulate technique-specific branches. The mode
bundle contains several owners' responsibilities and must be split, not renamed.

**Classification and current disposition.** User
[028](sessions/03-training-mode-and-objective.md#m028) authorizes classification;
assistant [032](sessions/03-training-mode-and-objective.md#m032) reports it.
Inventory Milestone 6, first committed in `707015ee`, explicitly supersedes
the old mode-as-axis judgment. Governing D12 and
[`adapter-system`](../../../openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md)
carry this forward: authored direct/PEFT/combined intent; governed domain
realization; Trainer optimization/preparation/lifecycle mechanics; selected
execution or narrow capability behavior; persistence/restoration contributions;
and observation over accepted state. No replacement mode authority or mode
value crosses into the accepted arrangement to select Trainer behavior.

**Missing detail or G5 consequence.** Map actual methods and fields to accepted
owners and remove displaced authority during migration. Preserve method-specific
adapter behavior without an adapter x family x objective strategy hierarchy.
Direct parameter training and PEFT need not be mutually exclusive, and artifact
choice is independently declared. A user-facing preset is not a runtime mode
authority.

**Placement.** Keep the classification as dated derivation evidence. Production
placement follows responsibility and dependency direction, not a wholesale
move of `modes/`. Specialized behavior may be delegated without receiving
unrestricted Trainer state.

<a id="h018"></a>

## H018 - Training-subject intent is distinct from optimization realization

**Source and original rationale.** User
[018](sessions/03-training-mode-and-objective.md#m018) distinguishes saying what
should train from optimizer-level parameter filtering. The code-pressure
report [005](sessions/03-training-mode-and-objective.md#m005) and Milestone 6
identify that mode methods mix declaration, `requires_grad` changes, logical/
execution groups, optimizer construction and backend preparation. Coexistence
in one method does not assign them one semantic owner.

**Resulting direction.** Authored strategy meaning selects subjects and
constraints. Trainer/optimization resolves the current parameters, trainability,
groups, synchronization/clipping participants and runtime mechanics under that
accepted meaning. The existing typed `OptimizationPlan` is a starting point,
not adoption of its present fields as the full future contract.

**Current disposition.** Governing D9 and
[`training-optimization`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md)
separate semantic units, membership and revisions from parameter and optimizer
objects. The standard profile retains realization and advancement. Multiple
optimizers or cadences do not automatically require manual authority transfer:
assistant [012](sessions/03-training-mode-and-objective.md#m012) already notes
that distinction, and later accepted policies support independently and jointly
due units. Their detailed derivation remains a later reading.

**Missing detail or G5 consequence.** Define semantic plan, unit, candidate/
runtime, contribution/advancement and publication types. Backend objects, tuple
positions and physical construction order cannot become semantic identity.
Preserve D12's distinction between authored adapter target intent, governed
host-target realization and downstream optimization of the resulting trainable
references.

**Placement.** Semantic subjects belong in accepted strategy/run meaning;
parameter/group realization and normal mechanics belong with optimization and
preparation. Raw optimizer dictionaries or a mode convenience query cannot
remain a competing authoritative projection.

<a id="h019"></a>

## H019 - Objective semantics, optimization inputs and accounting must not collapse

**Source in this reading.** The code-pressure report
[005](sessions/03-training-mode-and-objective.md#m005) distinguishes the current
`loss` accounting value from modified `per_sample_loss` used for backward and
identifies timesteps as objective-specific observations. Assistant
[032](sessions/03-training-mode-and-objective.md#m032) and Milestone 6 classify
DDPM/RF mathematics and adaptive state separately from generic optimization and
observation routing.

**Original rationale.** A runtime bundle can mix concerns without owning them
all. DDPM/RF choice, corruption, targets, prediction conventions and mathematical
weighting describe the learning behavior. Trainable loss-modifier state,
post-loss policy, backward input and accounting can have different consumers
and lifetimes. Trainer can supply timing or route typed observations without
learning diffusion-specific mathematics.

**Current disposition.** Objective choice is authored meaning, not a peer
runtime axis. Its implementation and state survive through accepted runtime
seams rather than an active authoring strategy. Governing execution requirements
separate computation outputs, optimization values, accounting, observations and
effects and reject universal diffusion fields. Calling `loss_modifier` an
optimization capability is a responsibility classification, not a universal
central-engine hook or a settled result API for every numerical modifier.

**Missing detail or G5 consequence.** Make producer/consumer meanings explicit
in execution and optimization exchanges. Preserve adaptive-state inputs,
observation routing and restoration contributions. A field called `loss` does
not prove backward readiness. Do not require placeholder timesteps or freeze
the proposed per-sample packet as the general workflow result.

**Placement.** Numerical/objective behavior belongs with selected domain
implementations; optimization, observation and lifecycle owners consume accepted
results. The old `ObjectiveRuntime` bundle and family step placement are evidence,
not an ownership rule to move unchanged.

**Reading 10 trace.** User
[001](sessions/10-accepted-arrangement-and-openspec-transition.md#m001) rejects
using today's `process_batch()` as the future computation boundary. User
[010](sessions/10-accepted-arrangement-and-openspec-transition.md#m010) also
qualifies EDM2 as an example the system should accommodate, rather than an
inherent optimization owner or mandatory core feature. The correction keeps
domain meaning and selected capability state while leaving the actual
execution representation to be derived.

<a id="h020"></a>

## H020 - Live execution services need explicit access, not ambient Trainer reach

**Source and original rationale.** User
[033](sessions/03-training-mode-and-objective.md#m033) asks what benefits
Lightning's back-reference offers. Assistant
[035](sessions/03-training-mode-and-objective.md#m035) identifies legitimate
needs for live coordinates, observation, prepared optimization services and
research control, alongside hidden dependencies from unrestricted Trainer
access. User [036](sessions/03-training-mode-and-objective.md#m036) favors a
middle ground with an explicit extension route rather than a narrow or unguided
system.

**Distinction retained.** Work may need controlled access to live services.
That does not imply every authored strategy or operation receives the entire
mutable Trainer. `context.require(ManualOptimizationControl)` illustrates scoped
access; its spelling, class and discovery mechanism were not selected APIs.
The external framework claims in that reply are historical discussion, not a
current API qualification performed here.

**Current disposition.** The governing contract offers ownership profiles and
scoped grants before authoring. Accepted operations use only declared
computation/effects and granted authority; changed owners require a supported
extension. Custom behavior can still satisfy the standard profile, and
multiple units need not become manual work. A wrapper's availability,
capability name or callable registration cannot silently grant authority. The
executor is an accepted runtime role, not automatically the authoring strategy.

**Missing detail or G5 consequence.** Define narrow invocation inputs,
projections, service/grant interfaces and handback/outcome obligations. Keep
normal behavior expressive while making changed ownership inspectable.
Replacing `trainer` with an equally unrestricted context preserves the old
hidden authority instead of establishing this boundary.

**Placement.** Offered policies and grants belong in governing contract/
execution interfaces, mechanics in their services, and specialized runtime
behavior behind accepted seams. The historical callback/service-locator sketch
is not a settled production authoring API.

<a id="h021"></a>

## H021 - Participant identity is distinct from its address and implementation

**Original source and rationale.** Assistant
[003](sessions/04-identity-and-execution-routes.md#m003) answers Q1 on July 31:
an explicitly declared logical identity should survive changes in concrete
objects and preparation. Python object, class, source checkpoint, catalog
identity, role and parameter scope do not define it. The user's August recap
[004](sessions/04-identity-and-execution-routes.md#m004) carries this answer
forward rather than asking to restart it. Examples such as
`student.denoiser` and `adapter.style` are readable addresses, not accepted
type names or a mandatory path syntax.

**Reading 07 trace.** The fresh session's
[007](sessions/07-contract-exchanges-and-authority-correction.md#m007) gives the
key/reference split its specific meaning: the key is an authored address;
the authority-issued reference identifies an accepted run incarnation; state
revisions do not replace identity. Retirement closes current use while
preserving history. User [033](sessions/07-contract-exchanges-and-authority-correction.md#m033)
asks whether `denoiser` is a role; the clarification records that shared role
labels cannot establish identity. Later use/role refinements remain separate
from the two-distinct-declarations examples used here.

**Adoption and refinement.** The first committed Q1-Q2 checkpoint is
`144faf03`. Its direction calls the answers provisional architecture strong
enough to constrain the exchange; its inventory calls Q1-Q2 semantically
settled. Those labels refer to bounded semantic progress, not frozen Python
APIs. Later identity/lineage work qualifies the early statement that replacing
an implementation in the same declared slot preserves identity.

**Current disposition.** The governing
[`run-participant-state`](../../../openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md)
spec separates an authored declaration address from an authority-established
run incarnation. Materialization, preparation and compatible replacement
preserve that reference; incompatible replacement, redeclaration and a new run
do not acquire continuity merely by reusing a name. Exact same-run restoration
must restore the accepted identity/revision facts. Readings 07-08 trace the
key/incarnation and durable-identity derivation; H035 separately identifies the
later correction to incomplete-restoration behavior.

The execution requirements also distinguish participant declarations from
individual uses. The early teacher/student examples must not become a rule
that role labels alone create identities. One accepted base participant may
have teacher and student uses with distinct view/gradient requirements; two
separately declared participants remain distinct even with shared origin or
initial state. Shared backing is not an identity oracle or proof that an
arbitrary aliasing arrangement is valid.

**Missing detail or G5 consequence.** Define concrete address/reference,
incarnation, lineage and replacement evidence types. Keep participant identity,
invocation identity, route/view identity and optimization membership distinct.
Do not make every payload, SSA value or repeated call a new persistent
participant. Generic machinery must follow accepted contracts rather than
infer them from strings such as `denoiser`.

**Placement.** Keep identity under the accepted run authority. Loading objects,
metadata identities, source files and authoring keys are associated facts, not
competing canonical identity stores. Public spelling and production module
placement remain concrete design choices within these meanings.

<a id="h022"></a>

## H022 - Routes express real execution differences; views serve other access

**Source and rationale.** Assistant
[008](sessions/04-identity-and-execution-routes.md#m008) proposes one authoritative
binding per logical-component/execution-purpose pair. Feedback supplied in
user [009](sessions/04-identity-and-execution-routes.md#m009) narrows the word
purpose: training, validation or preview labels alone must not create slots.
Materially different prepared/callable requirements may justify a distinct
route. The supplied text is labelled Feedback; its original authorship is not
established by the transcript.

The normal raw-module-to-prepared-wrapper transformation does not create two
equally valid execution targets. Original, unwrapped, inspection, metadata and
artifact handles are typed access views when used for those purposes. A list
of available Python references leaves both dispatch and validity ambiguous.

**Adoption and current disposition.** Assistant
[012](sessions/04-identity-and-execution-routes.md#m012) records Q2 and moves to
Q3. `144faf03` adds the direction section and inventory Milestone 7. The
governing participant-state requirements retain one authoritative binding per
participant/route pair, materially justified extra routes and distinct access
views. Subsequent adapter pressure qualifies the normal one-route example:
state-only participants need no fabricated forward route. That qualification
already appears in the same commit's later provisional Q3 checkpoint, outside
this chapter's Q1-Q2 conversation.

**Interpretation to avoid.** The final reply's standard/extension wording does
not mean every extra route requires manual optimization or a changed ownership
profile. A selected capability can require an additional representation under
the existing accepted contract. Route cardinality, authority grants and
invocation role are different questions.

**Missing detail or G5 consequence.** Design route/view types and preparation
correspondence that preserve these distinctions. Keep ordinary authors from
inventing route names merely for caller labels, without prescribing whether
the internal representation uses a uniform map or another structure. Do not
collapse differently constrained calls into one interchangeable handle or
force state-only adaptation into a fake callable model.

**Placement.** Route/view meaning belongs with run authority and preparation;
domain-specific callable implementations stay behind selected runtime seams.
The sample names `training` and `sampling` are illustrations, not a universal
route enum or public API selected by this discussion.

<a id="h023"></a>

## H023 - Derived representations need explicit freshness, not hidden divergence

**Source and rationale.** The Q2 proposal
[008](sessions/04-identity-and-execution-routes.md#m008) allows different
technical representations of accepted current state. Feedback in
[009](sessions/04-identity-and-execution-routes.md#m009) requires their
relationship to that state to be explicit. A synchronized or refreshed
derivative can be a route; independently evolving state requires a distinct
participant meaning. The EMA example points to separately managed derived
state, not permission to hide it under an arbitrary execution-route name.

**Recorded decision.** The Q2 checkpoint says a route cannot remain silently
authoritative after its freshness guarantee fails. It must become invalid,
be refreshed, or remain allowed only under an explicitly accepted staleness
guarantee. The discussion deliberately leaves the refresh mechanism and
binding-state owner unresolved. It does not promise automatic synchronization
or select eager versus on-use rebuilding.

**Current disposition.** Governing authority/preparation requirements preserve
explicit dependencies, revisions and freshness. Their later conservative
snapshot fallback and finer dependency option are documented decisions, not
rules newly derived by this reading. A failed freshness guarantee prevents
claiming the route is current; the continued existence of a callable, wrapper
or prepared object is insufficient. The early phrase "shared live state means
the same component" is read within a participant's technical representations,
not as a general equality rule collapsing independently declared participants.

**Missing detail or G5 consequence.** Define which changes invalidate a route
or projection, what evidence restores usability, and how preparation candidates
retain exact source correspondence. Tie refresh and publication to the
accepted authority rather than a second cache-owned binding map. Currentness
and declared historical/snapshot use must be distinguishable in the concrete
interfaces; a boolean saying a module was once prepared cannot supply that
meaning.

**Placement.** Store canonical freshness/dependency state with accepted
authority and its governed preparation exchanges. Keep numerical refresh or
compilation work in the appropriate realization implementation. Historical
representation examples do not select cache policy, backend or production
storage layout.

**Reading 07 trace.** The declaration-feedback reconciliation
[027](sessions/07-contract-exchanges-and-authority-correction.md#m027) broadens
coarse freshness from binding changes to every change visible in the depended-on
authority snapshot, including relationships and routes. It also exposes the
in-place-preparation tension: withdrawing guarantees advances authority state,
so an attempt cannot require its pre-withdrawal snapshot to remain unchanged
as its sole completion test. The final coordination decision is traced in H034
and reading 08; this discovery is not itself a selected lease/token API.

<a id="h024"></a>

## H024 - Semantic participants and relationships have independent identity

**Original source and rationale.** The user's Q3
[request](sessions/05-arrangement-and-persistence.md#m001) explicitly says
current generic/PEFT wiring cannot define the system's possibilities. The
expanded answer [005](sessions/05-arrangement-and-persistence.md#m005) tests
injected state, prefix tensors, auxiliary networks and wrappers. Separate
optimization groups, injected submodules, Python containers and checkpoint
files are evidence about representation; none individually defines a
participant.

The supplied [external feedback](snapshots/2026-08-20-q3-external-feedback.md)
sharpens the criterion to an independently addressable semantic participant.
One adapter may have multiple effects; disabling an effect does not retire its
participant. One runtime container may hold several participants. A state-only
participant needs no invented callable. An outermost wrapper does not take
semantic execution ownership from the host it realizes.

**Adoption and correction.** Assistant
[011](sessions/05-arrangement-and-persistence.md#m011) accepts those distinctions.
The supplied corrections in user
[012](sessions/05-arrangement-and-persistence.md#m012) remove trainability from
the participant lifecycle: declared/bound/retired, selected/trainable/frozen,
and relationship activation are separate axes. That turn explicitly requests
settlement; [016](sessions/05-arrangement-and-persistence.md#m016) records it.
`144faf03` is the provisional checkpoint; `e0613c2f` first commits the settled
direction and inventory. The imported critique alone did not settle Q3.

**Current disposition.** The governing
[`run-participant-state`](../../../openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md)
requirements retain independent participant, relationship and optimization
lifecycles, independently addressable relationships, and state-only
participants. Later incarnation and invocation distinctions refine the
original keys. This is not a rule that every feature contribution, tensor,
parameter group or use must become a participant.

**Missing detail or G5 consequence.** Concrete declarations, references,
effect state and access views must preserve independent identity and
validity. Selected domain behavior supplies adaptation meaning; central
machinery must not infer it from the class of a wrapper or an adapter name.
Ordinary authors select maintained behavior through Python; they do not need
to invent a new participant or relationship protocol for every run.

**Placement.** Keep canonical identity and relationship state with accepted
run authority. Adapter realization and parameter structure stay behind their
domain contracts. The inventory's code observations remain evidence for the
distinction, not a universal adapter taxonomy.

<a id="h025"></a>

## H025 - Declaration, realization and governed transitions are different acts

**Original source and rationale.** The [Q3 starting take](snapshots/2026-08-20-q3-starting-take.md)
separates execution rebinding, compatible slot replacement and participant
addition. The pressure test [005](sessions/05-arrangement-and-persistence.md#m005)
adds a crucial correction: an ordinarily authored adapter already exists
semantically before its state is materialized or its effect attached. A loader
temporarily opening an integrated VAE before the selected external VAE does
not establish a semantic replacement.

**Recorded decision.** The final corrections
[012](sessions/05-arrangement-and-persistence.md#m012) retain separate acts:
declaration, materialization, bound-state replacement, route rebinding,
operational relationship transition, arrangement amendment and merge/fold.
Replacement supersedes an established binding and affects dependent facts;
materialization fills a deferred declaration. Merge/fold transforms host state
and records lineage but is not inherently destructive to the adapter.
Participant history, current arrangement and durable artifact provenance are
separate views. A consumed participant need not remain live merely to retain
its history.

Q3 deliberately says authoritative state is **maintained for** a participant.
It does not decide who holds that state; the correction leaves Q5 open.
The first committed settled formulation is `e0613c2f`, after the user explicitly
approves the corrected meaning. The later compaction-recovery reply
[020](sessions/05-arrangement-and-persistence.md#m020) confirms the details,
but the original turns and Git change are the evidence for adoption.

**Current disposition.** Governed authority now evaluates initial realization
and later transitions under the same accepted obligations, with revision and
atomicity requirements. Identity-preserving replacement must remain compatible
with the participant's complete accepted meaning; reusing an address cannot
make an incompatible replacement continuous. Early avoidance of a general
strategy graph was bounded sequencing, not a veto on the later whole-run IR.

**Missing detail or G5 consequence.** Realization proposals, evidence,
publication and invalidation need concrete exchanges that retain these
differences. Python assignment, wrapper construction or a successful factory
call cannot alone publish canonical state. Captured history and artifact
provenance must remain meaningful after live retirement without requiring an
unbounded live-object map.

**Placement.** Keep transition acceptance in the accepted authority and
physical realization in selected domain/runtime services. Keep chronological
evidence and durable product lineage distinct from current bindings.

<a id="h026"></a>

## H026 - Persistence declares semantic products, members and actual resources

**Original source and rationale.** Q4 asks what object persistence saves.
Assistant [024](sessions/05-arrangement-and-persistence.md#m024) reframes that
as producing a purpose-specific artifact from authoritative state. The
evidence spans distributed resume state, full-model exports, adapter state
distributed through targets, SD3 sidecars and capability-owned artifacts.
The object easiest to reach cannot determine product coverage.

**Refinement and adoption.** User
[025](sessions/05-arrangement-and-persistence.md#m025) explicitly asks that the
[notices](snapshots/2026-08-21-q4-notices.md) be checked against the fuller local
context. The evaluated answer
[028](sessions/05-arrangement-and-persistence.md#m028) goes beyond state ownership:
products also select relationships, dependencies, references and declared
derived state. It establishes declaration, request, resolved plan and actual
result as separate stages.

Product, semantic member and physical resource are not one-to-one. One member
can span shards; one file can hold several participants. Packaging or a bundle
does not establish completeness. Coverage, dependencies, semantic
transformations, representation, packaging and consistency remain independent
dimensions; a delta may also embed state and reference external dependencies.
Semantic merge/pruning/derived realization must be declared. Domain serializers
own the resolved mechanical conversions and writing.

The same evaluation corrects its earlier EDM2 description: the active paths
invoke the loss modifier's sidecar save; the mode/filename branch is residual
evidence. This was a historical code finding, not freshly reproduced here.
User [029](sessions/05-arrangement-and-persistence.md#m029) authorizes the
checkpoint; `e0613c2f` first commits the settled Q4 direction and inventory.

**Current disposition.** Governing
[`training-artifact-persistence`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-artifact-persistence/spec.md)
retains these stages and dimensions. Later requirements further distinguish
state variants, construction/use requirements, optional constituents,
publication, dependency availability and retention. They are later
refinements, not claims that August already settled every field.

**Missing detail or G5 consequence.** Concrete product declarations and plans
must preserve selected meaning independently of live wrappers and physical
layout. Adapted, merged and full-model outputs follow accepted product support,
not a training-mode enum or trainability inference. External-use requirements
must come from the selected product, without requiring the original training
Python object as a universal reconstruction dependency.

**Placement.** Keep semantic requirements in the persistence spec; domain
capabilities supply product meaning, serializers supply representation, and
shared infrastructure handles coordinated capture/storage/publication. Retain
the inventory's source-specific pressure cases as migration evidence.

<a id="h027"></a>

## H027 - Coherent capture and observed completion govern persistence claims

**Original source and rationale.** The revised Q4 answer
[028](sessions/05-arrangement-and-persistence.md#m028) requires all contributors
to satisfy one accepted arrangement and Trainer-established persistence
boundary. That does not require identical literal revision counters: a frozen
base and updated adapter can form a coherent product under explicit freshness
and dependency relationships.

Trainer/runtime infrastructure coordinates the boundary and rank mechanics;
product semantics declare any stronger consistency requirements. A stable
capture can support later serialization/publication while training continues.
An artifact projection need not be a copied dictionary or an original module.
The request/plan describes intent; actual paths, resources, sizes, checksums,
partial outputs and failures come from the operation's result. Pre-write
metadata cannot manufacture post-write observations.

**Adoption and current disposition.** The Q4 checkpoint request
[029](sessions/05-arrangement-and-persistence.md#m029) and `e0613c2f` record these
meanings. Current persistence requirements strengthen stable capture with
protected source access through required physical reads, preservation of the
captured meaning through later live transitions, and separate write,
publication and observation outcomes. Product completion is not a guarantee
of self-containment or current availability of every external dependency.
The later derivation of those refinements remains outside this chapter.

**Missing detail or G5 consequence.** Design stable capture, access obligations,
partial results and release around actual physical settlement. A borrowed live
tensor is not proof of stability; writing one file is not proof that a required
multi-member product or requested upload completed. Historical product claims
must remain tied to the captured source even after the live run changes.
Execution restoration remains the distinct contract recorded in H016.

**Placement.** Keep canonical capture meaning with the resolved accepted plan
and durable result. Coordinate physical access through the appropriate owners
and runtime services. Keep logging/metadata as consumers of actual results,
not the source of truth for completion.

<a id="h028"></a>

## H028 - Q5 settled one governed authority; its runtime enclosure was later corrected

**Original source and rationale.** Q5 asks where canonical current bindings
belong without making Trainer understand family anatomy. User
[005](sessions/06-authority-reviews-and-handoff.md#m005) explicitly warns that
unchanged production code is not the target. Six scenarios are modeled
target-first in [008](sessions/06-authority-reviews-and-handoff.md#m008),
including compound research beyond current family shapes. They constrain
authority, updates and access but do not distinguish physical layouts that
all satisfy those semantics.

The historical [Claude](snapshots/2026-08-21-q5-claude-review.md) and
[GLM](snapshots/2026-08-21-q5-glm-review.md) critiques supply discriminating
criteria: controlled single-writer transitions, independent conformance tests,
state/behavior separation, explicit run lifetime, scoped access and migration.
Their opinions, code-verification claims and caveats retain their original
scope. They are not a new review or production qualification. The failed
provider launches and mistaken report/model attribution are not additional
successful critiques.

**Adoption and historical enclosure.** User
[025](sessions/06-authority-reviews-and-handoff.md#m025) asks whether a separate
authority changes what Trainer receives. The answer
[026](sessions/06-authority-reviews-and-handoff.md#m026) separates responsibility
from a second public orchestration axis, but still places the authority inside
an active strategy. After the outside assessment and the user's clarification,
[030](sessions/06-authority-reviews-and-handoff.md#m030) authorizes updating the
documents; [034](sessions/06-authority-reviews-and-handoff.md#m034) records Q5 as
settled. `c3b59068` retains the open scenario layout; `f603d074` first commits
the settled enclosure; `d7c4de7b` retains it after the policy correction.

**Current disposition and supersession.** Q5's lasting result is one dedicated,
contract-governed per-run authority, typed controlled transitions and scoped
coherent projections. The [August 26 topology correction](../../archive/models-strategy-trainer/notes.md#accepted-run-arrangement-topology-correction-2026-08-26)
explicitly relocates that responsibility to the accepted arrangement and
supersedes the active-strategy collaborator. The governing design and
participant-state spec preserve one authority without requiring authoring
objects to survive as runtime counterparts. This does not split the system
into two independently authoritative runtimes or reopen Q1-Q5.

**Missing detail or G5 consequence.** Concrete construction, publication,
authority grants, revisions and consumer projections must implement the
accepted arrangement's obligations. Domain owners may retain their own
optimization or backend state without competing over canonical participant
facts. Python object count does not establish enforcement or ownership.
Optimization ownership remains a separate semantic block; the restored
blocker is explicit in the historical critiques and Q5 checkpoint.

**Placement.** Keep current authority requirements in the governing change.
Retain the scenario inventory and ownership comparisons as derivation.
Mark the old active-strategy enclosure as superseded instead of deleting the
rationale or silently showing its diagram as current.

<a id="h029"></a>

## H029 - Correct conservative policies need not weaken the contract

**Original source and initial overreaction.** The reviews recommend staging
implementation, including broad invalidation before precise dependency
tracking. User [027](sessions/06-authority-reviews-and-handoff.md#m027) disagrees
with the outside assessment's conservative-v1 approach. Assistant
[029](sessions/06-authority-reviews-and-handoff.md#m029) interprets that too
categorically, rejecting invalidate-all as an intentionally weakened design.
The initial Q5 checkpoint and `f603d074` carry that overstatement.

**Correction and adoption.** User
[035](sessions/06-authority-reviews-and-handoff.md#m035) asks for the original
reviews to be reread rather than relying on the compacted summary.
The recovered [GLM report](snapshots/2026-08-21-q5-glm-review.md) distinguishes
semantics from invalidation precision; the
[Claude critique](snapshots/2026-08-21-q5-claude-review.md) also proposes staging.
Assistant [038](sessions/06-authority-reviews-and-handoff.md#m038) corrects the
interpretation: explicit over-invalidation can be correct, while discarding
identity, dependencies, freshness, route capacity or persistence meaning is
not. User [039](sessions/06-authority-reviews-and-handoff.md#m039) accepts making
that correction. `d7c4de7b` first commits the revised wording.

This allows current selections to use only a normal route or current products
without pretending the contract cannot represent anything else. It does not
adopt every outside recommendation for shortcut artifact interfaces or require
the final theoretical system on day one. The original-review reread also
restores identity assignment, compatible replacement and default freshness as
concrete exchange questions.

**Current disposition.** The governing participant-state requirements now
provide a conservative whole-authority snapshot dependency when no narrower
valid dependency set is declared. Precise dependencies can retain unaffected
projections only with valid evidence. This is explicit correct policy under
the contract, not stale-state management left to arbitrary callers. Its later
exchange-level adoption is traced in H033–H034 and reading 08.

**Missing detail or G5 consequence.** Distinguish obligations from precision
and implementation coverage. Choose explicit policies that satisfy accepted
meaning and expose their costs; do not infer architectural incompleteness
merely from unused representational capacity. Equally, a narrow experiment
does not prove all supported forms are implemented or production-qualified.

**Placement.** Keep the correction adjacent to the overstatement in history
and link the adopted policy in the governing spec. A summary that records only
the initial rejection would lose the actual settled conclusion.

<a id="h030"></a>

## H030 - Separate document roles while preserving the reasoning

**Original source and rationale.** The reviews identify duplicated settled
statements as a drift risk. User
[039](sessions/06-authority-reviews-and-handoff.md#m039) asks for documents
separated by concern while preserving the details. The assistant's ensuing
normalization [042-048](sessions/06-authority-reviews-and-handoff.md#m042)
reports removing about 600 lines and declaring direction/evidence/history
cleanly separated.

**Rejected realization.** User
[049](sessions/06-authority-reviews-and-handoff.md#m049) rejects removing so much
text and asks to leave the documents as they were. The assistant reports an
exact restoration from recorded patch history in
[051](sessions/06-authority-reviews-and-handoff.md#m051) and
[052](sessions/06-authority-reviews-and-handoff.md#m052), retaining only the
agreed conservative-policy correction. The user then edits a few lines and
asks for a check. The apparent duplicated excerpt later proves to be overlapping
inspection output, not duplicated file content; only a small sentence fix is
actually needed [063](sessions/06-authority-reviews-and-handoff.md#m063).

The abandoned large rewrite has no committed snapshot established here.
Conversation evidence records the attempt, objection and reported restoration;
the final `f603d074` to `d7c4de7b` diff is small. That is not an independently
reconstructed proof of every intermediate working-tree byte, nor permission
to recreate the rejected reduction.

**Current disposition.** The governing design retains clear source roles:
direction as normative pre-OpenSpec input, inventory as evidence, notes as
chronology, and later exchange design as detailed derivation with adoption
boundaries. Multiple records can retain reasoning without becoming competing
specifications. The current user request continues this history/indexing
approach and explicitly requires preserving decisions, source and rationale,
disposition and G5 consequences.

**Missing detail or G5 consequence.** A later organization pass must preserve
unique rationale, examples, rejected alternatives and the source/status of
overlapping prose. Use status labels, chronology, cross-links and an index
before deleting or moving material. Distinguish adopted requirements from
handoff proposals: the final suggestion of multiple sequential OpenSpecs and
one suggested first name [065](sessions/06-authority-reviews-and-handoff.md#m065)
does not establish their eventual count or partition.

**Placement.** Keep originals in their referenced locations during this audit.
The notepad and readable exports provide navigation and supersession links;
they do not replace the original documents or become another governing spec.

<a id="h031"></a>

## H031 - The contract derives and enforces ordinary composition obligations

**Original source and drift.** The first exchange draft proposes a
participant-specific semantic compatibility filing and later proposal-scoped
assessments. The [first draft](snapshots/3602960a-strategy_contract_exchange_design.md)
and [expanded version](snapshots/acdc2b19-strategy_contract_exchange_design.md)
preserve that working proposal. It was not a consequence of selecting xDSL or
a requirement that every participant carry its own author-written validator.

User [078](sessions/07-contract-exchanges-and-authority-correction.md#m078)
returns to the already documented principle: the contract knows what Trainer
accepts, informs strategy authoring, and can enforce the complete authored
strategy. Ordinary authors must not separately explain how maintained library
behavior is valid. The assistant's earlier explanations had transferred too
much responsibility to the author and treated local validation as a second
authority.

**Correction and adoption.** Assistant
[080](sessions/07-contract-exchanges-and-authority-correction.md#m080) separates
automatic enforcement from automatic assembly: the system may derive
requirements and reject missing/incompatible wiring; it may not choose
providers or infer the intended strategy. After the scoped correction proposed
in [083](sessions/07-contract-exchanges-and-authority-correction.md#m083), user
[084](sessions/07-contract-exchanges-and-authority-correction.md#m084) authorizes
the update. [088](sessions/07-contract-exchanges-and-authority-correction.md#m088)
records it. `7bd8a124` first commits the revised exchange and notes.
Conflicting working EX entries are superseded or corrected, while identity,
transitions, freshness and preparation mechanics survive. This repairs the
downstream draft; it does not reopen the original direction.

**Current disposition.** Governing
[`training-contract`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-contract/spec.md)
requires the active contract before authoring, explicit composition,
fulfillment of the complete definition, and one authoritative run-specific
obligation set. Custom implementations use explicit conformance; new meanings
need an offered extension. The later topology correction makes the accepted
arrangement independent of the authoring object. Neither a central
family-name validator nor arbitrary participant callbacks define the language.

**Missing detail or G5 consequence.** Maintained behaviors must carry reusable
contract meaning and evidence requirements that fulfillment can apply without
per-run validator authorship. Keep the authoring selection, contract
specialization, current authority and reported evidence distinct. The Python
authoring API and xDSL operations must preserve this existing direction;
choosing a substrate did not reset it.

**Placement.** Keep governing meaning in the contract/design and adopted delta
requirements. Retain the rejected filing/assessment proposal in the historical
exchange revisions with its disposition. Conformance mechanics and final type
names remain implementation design, not additional author obligations.

<a id="h032"></a>

## H032 - Semantic acceptance and executable readiness are distinct

**Source and rationale.** The early identity/refinement replies
[007](sessions/07-contract-exchanges-and-authority-correction.md#m007) and
[027](sessions/07-contract-exchanges-and-authority-correction.md#m027) separate
participant meaning from materialization timing and required routes. A
declared participant can remain unbound under an accepted readiness condition;
`deferred` is not a different identity. A state-only adapter needs no callable.

The subsequent contract correction retains that separation while changing its
source: obligations come from the complete supported selection and use, not
handwritten participant validation. Initial fulfillment establishes everything
already knowable. Loading and preparation later report the concrete evidence
needed to authorize the corresponding uses. Later state changes are judged
under the same governing meaning, rather than separate validators inventing
new compatibility interpretations.

**Adoption and current disposition.** The corrected exchange at `7bd8a124`
records contract establishment followed by evidence-governed materialization,
replacement and preparation. Current requirements distinguish accepted meaning
from executable readiness and evaluate each obligation at its earliest
authoritative evidence point. They preserve exact unsatisfied obligations
when evidence is legitimately deferred. Known invalidity cannot be postponed
merely because a later stage also checks readiness.

**Missing detail or G5 consequence.** Concrete acceptance/readiness evidence
must prevent dependent use until its conditions hold, while allowing accepted
work whose future runtime facts are not yet available. Mechanical failure,
invalid candidate evidence, missing capability and a stale prepared result
need distinguishable results. No successful constructor, imported class or
bare Boolean may substitute for required evidence and accepted publication.

**Placement.** Keep semantic acceptance, obligations and readiness under the
same contract-governed authority. Domain producers and preparation backends
report facts and physical outcomes; they do not independently redefine what
the accepted arrangement means.

<a id="h033"></a>

## H033 - Derive machinery from intended consumers, without reopening settled meaning

**Source and rationale.** User
[003](sessions/08-design-frontier-preparation-and-identity.md#m003) reiterates
that production code predates the direction. The dependency analysis
[011](sessions/08-design-frontier-preparation-and-identity.md#m011) therefore
works from intended Trainer responsibilities to required exchanges, accepted
state, enforcement and authoring. Runtime flows forward from authoring;
migration analysis describes today's code. These are different orders, not a
rule that every future workflow must execute one fixed phase sequence.

The user then asks why the open register grows and requests an audit. User
[029](sessions/08-design-frontier-preparation-and-identity.md#m029) distinguishes
architecture from Python details, and
[033](sessions/08-design-frontier-preparation-and-identity.md#m033) asks for a
clear design/current-code/target-code distinction. Different class names,
constructors, generics or registration forms do not make already settled
behavior architecturally unresolved.

**Adoption and disposition.** The fourteen-item audit removes duplicates and
defers code representation, leaving durable identity, preparation reads and
publication as the real binding/preparation frontier. The authorizations
[038](sessions/08-design-frontier-preparation-and-identity.md#m038) and
[048](sessions/08-design-frontier-preparation-and-identity.md#m048) lead to
the `d44b2ccb` checkpoint and first implementation mapping. The user permits
that file provisionally while saying they do not want another living document
[042](sessions/08-design-frontier-preparation-and-identity.md#m042). The mapping
is a bridge, not a competing architecture or replacement task tracker.

**Current disposition and G5 consequence.** Governing requirements retain
the accepted meanings while G5 chooses concrete language/runtime interfaces
and migration. An unchosen type name is not evidence that the architecture
is open. Conversely, a real change in effects, authority, lifecycle, failure
or consumer guarantees needs design attention. Use current code to expose
requirements and migration pressure without promoting its method signatures
or family fields into the future contract.

**Placement.** Keep genuine behavioral questions with design, implementation
derivation with the mapping, chronology with notes and task status in Beads.
Update the actual frontier rather than appending superseded prerequisites
and repeated handoff lists indefinitely.

<a id="h034"></a>

## H034 - Preparation is an attempt-scoped job with one coherent publication

**Source and refinement.** The initial preparation-read answer
[056](sessions/08-design-frontier-preparation-and-identity.md#m056) calls for
a coherent, single-use package. Inline feedback
[057](sessions/08-design-frontier-preparation-and-identity.md#m057) corrects
that overly literal mechanism. The semantic unit is one attempt-scoped job
derived atomically from coherent accepted state; it may involve several
ordered or joint backend calls. Preparation membership exceeds trainability:
frozen state can still require movement, casting or wrapping.

The evaluated rule
[059](sessions/08-design-frontier-preparation-and-identity.md#m059) places
expected-fallible backend work, component preparation, candidate construction,
rank agreement and validity checks before final logical publication. That
publication installs complete authority-owned route/view state together with
the associated Trainer-owned runtime candidate. Backend composites do not
replace participants, and jointly prepared optimizer state does not become
binding-authority state or settle optimization ownership.

**Adoption.** User
[060](sessions/08-design-frontier-preparation-and-identity.md#m060) authorizes
recording the decisions; [064](sessions/08-design-frontier-preparation-and-identity.md#m064)
reports reconciliation. `d44b2ccb` first commits that checkpoint. Replacement-only
failure discards the unpublished candidate while preserving still-valid prior
state. Destructive preparation is different: its recognized attempt first
withdraws affected guarantees, and failure leaves them withdrawn until
explicit re-establishment. Withdrawal does not invalidate the attempt's own
completion basis merely by advancing authority state.

**Current disposition.** The governing
[`training-runtime-preparation`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md)
requirements retain coherent attempts, separate result owners, prepublication
fallibility, destructive withdrawal and stale-result rejection. Later
current-use and inseparable backend-group obligations further constrain
publication. Postpublication observation failure does not roll back accepted
state; process/rank termination belongs to abort/restoration. This is not a
claim of crash durability or guaranteed recovery inside a failed process.

**Missing detail or G5 consequence.** Encode attempt identity, source
correspondence, mutation permissions, complete candidate coverage and owner
handback. Keep final publication free of expected-fallible conversion,
backend work and user callbacks. Do not impose a staged transaction engine
on ordinary replacement solely because destructive mutation needs its own
safety transition. Concrete locks, tokens and state containers remain
implementation choices under these guarantees.

**Placement.** Keep preparation meaning and publication obligations in the
governing exchange/spec. Domain/backend services produce candidates; the
accepted authority and other owners publish their respective state coherently.
Historical placeholder optimizer types do not become selected APIs.

<a id="h035"></a>

## H035 - Durable incarnation, execution session and lineage are distinct

**Source and rationale.** The identity investigation
[073](sessions/08-design-frontier-preparation-and-identity.md#m073) distinguishes
logical run/authority identity from the newly created process's observation
session. Exact same-run restoration can preserve authority, participant
incarnations and relevant revisions across recreated objects only when the
snapshot actually contains and restores those facts. A separately started
run has new participants linked through lineage. Artifacts have their own
identity and describe captured participant state; they are not participants.

The first wording accidentally tied incarnation to current membership. Inline
feedback [088](sessions/08-design-frontier-preparation-and-identity.md#m088)
and its evaluation [090](sessions/08-design-frontier-preparation-and-identity.md#m090)
correct that: an authority-qualified reference permanently identifies one
incarnation, is never reassigned or revived, and remains historical after
retirement. User [091](sessions/08-design-frontier-preparation-and-identity.md#m091)
authorizes the correction; `d44b2ccb` records it. Exact identifier formats and
Python classes remain deferred.

**Later supersession of one fallback.** The August explanation and checkpoint
also say saved state lacking exact-resume identity facts establishes a new
run with lineage. That must not be quoted as the current automatic behavior.
The governing delta's `6db3f1ce` change on October 4 explicitly replaces it:
missing required facts make same-run restoration incomplete; starting a new
run requires a separate explicit request. The [source record](sessions/08-source-records.json)
retains the Git introduction query. This is a targeted disposition check,
not a completed walkthrough of the intervening adoption history.

**Current disposition and G5 consequence.** Preserve authority-qualified
incarnation independently of revisions, process/session, source and authored
address. Retain retirement and known effects in history. Restoration must
state which operation it can fulfill, keep dependent work unavailable when
required evidence is missing, and never manufacture identity continuity or
silently substitute a new run. Participant equality is not inferred from
role labels, matching checkpoint paths or repeated invocation.

**Placement.** Canonical identity belongs to accepted run authority; metadata,
history, persistence and restoration consume its intentional durable meaning.
Execution-session observations remain separately identified, with lineage
relating distinct runs and artifacts where appropriate.

<a id="h036"></a>

## H036 - Metadata work exposed the boundaries that led to this redesign

**Retrospective source and correction.** User
[074](sessions/08-design-frontier-preparation-and-identity.md#m074) explains
the work order after the identity investigation. The first assistant reading
reduces metadata to older evidence. User
[078](sessions/08-design-frontier-preparation-and-identity.md#m078) corrects
that: attempting model metadata directly led into the strategy/Trainer
investigation. User [082](sessions/08-design-frontier-preparation-and-identity.md#m082)
then makes the chronology more precise: substantial general metadata work was
done, but richer model metadata had not been properly completed when the
underlying boundaries proved unsuitable.

**Recorded conclusion.** The corrected account
[084](sessions/08-design-frontier-preparation-and-identity.md#m084) is general
metadata work -> attempt at richer model metadata -> exposed model/strategy/
Trainer problems -> architectural pivot -> later return to model metadata on
a coherent foundation. `d44b2ccb` preserves this in exchange and notes. August
25 dates the retrospective clarification, not the original pivot. This is not
evidence that a mature model-metadata subsystem merely lacked Trainer wiring.

**Current disposition and G5 consequence.** Intentional boundaries, durable
identity, construction, provenance, artifacts and restoration are motivating
requirements. They are not incidental logging concerns introduced after the
architecture. Accepted runtime authority establishes identities and changes;
metadata preserves the appropriate durable projections. Existing metadata
identifier strings and dataclasses do not dictate the eventual language or
runtime storage representation.

**Placement.** Preserve this origin alongside the initial direction and its
historical predecessors. Follow the metadata/model-metadata design links as
requirements evidence when mapping production changes; do not turn this
history pass into an unrelated metadata implementation project.

<a id="h037"></a>

## H037 - Executable evidence earns bounded findings, not production architecture

**Source and authorization.** User
[004](sessions/09-executable-evidence-and-optimization.md#m004) challenges
calling a paper walkthrough testing. After the distinction is corrected,
[007](sessions/09-executable-evidence-and-optimization.md#m007) proposes
isolated code and [010](sessions/09-executable-evidence-and-optimization.md#m010)
authorizes it. The resulting spike uses fake objects without production
integration. Its reported test counts grow as historical reviews expose
counterexamples; [050](sessions/09-executable-evidence-and-optimization.md#m050)
reports the final 28 scenarios. The actual returned final review is preserved
in the [source record](sessions/09-source-records.json); no tests or reviews
were rerun for this history pass.

**Findings carried forward.** Complete amended arrangements must satisfy
their obligations, rather than rechecking only one successor's direct uses.
Prepared usability depends jointly on authority guarantees and Trainer-owned
backend state. An inseparable backend group can be replaced, expanded or
merged, but cannot be partially split while retaining stale coordination.
Relationship changes invalidate dependent prepared state. Destructive
attempts prevent older optimistic work from restoring withdrawn guarantees,
including after destructive failure. `b45f1d4e` first commits the spike and
these findings.

The external [feedback](snapshots/2026-08-25-spike-feedback.md) adds an
important qualification, accepted in
[054](sessions/09-executable-evidence-and-optimization.md#m054) and recorded
in [064](sessions/09-executable-evidence-and-optimization.md#m064): always
invalidating both endpoints is a conservative spike policy, not the universal
meaning of every relationship.

**Current disposition and G5 consequence.** These behavioral requirements
carry into preparation/authority specs. Fake-object success does not establish
backend integration, durable serialization, concurrent safety or final language
shape. The spike's classes, enums and active-strategy vocabulary remain
experimental; the following topology correction explicitly relabels that
container without promoting its API. Reuse established invariants and exact
counterexamples, then qualify the actual production realization separately.

**Placement.** Keep runnable evidence with the spike and its source witnesses.
Keep adopted meaning in governing requirements, chronology in notes and
current-to-target production decisions in their own derivation. A larger test
count does not make the whole isolated program the production template.

<a id="h038"></a>

## H038 - Optimization identity, accepted definition, realization and progress differ

**Source and refinement.** The optimization pass preserves semantic training
subjects and logical groups separately from concrete parameter/execution
groups. The unit-identity proposal
[092](sessions/09-executable-evidence-and-optimization.md#m092), supplied
[feedback](snapshots/2026-08-25-optimization-unit-feedback.md), and evaluation
[095](sessions/09-executable-evidence-and-optimization.md#m095) distinguish
an address, run-scoped incarnation, accepted-definition revision, concrete
runtime realization and mutable progress.

A semantic unit can consume parameters from several participants; it is not
itself a model participant. Recreating parameters, optimizer objects, wrappers
or backend handles need not change its accepted definition. Membership,
grouping or policy changes can advance that definition. Mutable momentum and
scheduler progress change during execution without creating a new definition
revision. A changed binding can invalidate realization while still allowing
the same semantic unit revision to be re-realized.

**Adoption and current disposition.** User
[096](sessions/09-executable-evidence-and-optimization.md#m096) authorizes
recording the refined rule; [100](sessions/09-executable-evidence-and-optimization.md#m100)
records closure of the identity item. The first committed record is
`bb763b59`, also containing later corrections. Governing
[`training-optimization`](../../../openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md)
preserves these distinctions. Same participant or unit identity does not
justify retaining incompatible optimizer state after replacement; state
continuity needs its own accepted evidence.

**Missing detail or G5 consequence.** Resolve authored intent to accepted
semantic membership, then link it to current parameter handles and backend
realization. Preserve logical-to-execution group correspondence, explicit
dependencies and per-unit progress where required. Do not use a list index,
optimizer class or Python object identity as semantic identity.

**Placement.** Unit authority and mutable optimizer state stay with the
optimization owner. Participant authority supplies accepted identities and
binding facts; preparation coordinates the complete realizations without
merging those ownership responsibilities.

<a id="h039"></a>

## H039 - Standard optimization ownership is independent of backend staging

**Source and rationale.** The code-grounded pass and external
[feedback](snapshots/2026-08-25-optimization-feedback.md) distinguish ordinary
Trainer-owned policy from the number of optimizers or an unusual cadence.
The evaluation [079](sessions/09-executable-evidence-and-optimization.md#m079)
corrects a premature optimizer-before-preparation rule: semantic intent and
plan precede realization, and complete runtime precedes publication, but the
selected backend may require parameter-owner preparation or sharding before
concrete optimizer construction.

Trainability, clipping, accumulation, synchronization and module-mode
requirements are related but different projections. One coherent accepted
optimization state does not make them one universal list. Optimizer
train/eval behavior is also distinct from pipeline-coordinated module mode.

**Recorded overlap decision.** Proposal
[107](sessions/09-executable-evidence-and-optimization.md#m107), user
[108](sessions/09-executable-evidence-and-optimization.md#m108), and clarification
[110](sessions/09-executable-evidence-and-optimization.md#m110) establish
one ordinary optimization owner and execution group per selected parameter,
including aliases. Deliberate overlap requires a recognized policy with
ordering, gradient, clipping, zeroing, restoration and backend meaning; a
Boolean permission alone is insufficient. Normal authors select training
intent and supported behavior rather than constructing optimizer mechanics.

**Correction boundary and current disposition.** The final reply in this
chapter still overstates today's batch callback and a possible EDM2 placement.
The next user turn explicitly rejects that promotion. The corrected target
supports standard policies and explicit alternative grants without assuming
that changed ownership belongs to an active strategy. Current requirements
permit independent/joint advancement and observation-dependent policy where
accepted; unusual scheduling or an auxiliary optimizer alone does not imply
imperative transfer. EDM2 remains an example of selected capability behavior,
not inherent core machinery or a mandated unit placement.

**Missing detail or G5 consequence.** Keep semantic authority, realization
staging, accepted advancement policy and scoped runtime grants separate.
Preserve complete publication and distinct projections across backend choices.
Use current code as evidence for pressures, not as a universal call sequence
or reason to require one scalar loss and optimizer step per action.

**Placement.** Keep accepted meanings in optimization and execution contracts;
keep backend staging in realization/preparation. Domain capabilities retain
their semantics while generic orchestration remains with its accepted owner.

<a id="h040"></a>

## H040 - Accepted executable meaning outlives the authoring recipe

**Source and correction.** The early optimization reply describes today's
batch callback and a possible EDM2 placement too strongly. User
[001](sessions/10-accepted-arrangement-and-openspec-transition.md#m001) rejects
that promotion; [010](sessions/10-accepted-arrangement-and-openspec-transition.md#m010)
keeps EDM2 as a capability/extension example, not inherent core behavior.
The ensuing step discussion exposes the deeper error: defining training
meaning does not require the authoring object to remain a per-batch service.

User [040](sessions/10-accepted-arrangement-and-openspec-transition.md#m040)
states that the written strategy should have no further authoring job after
handoff. The historical reconstruction, fuller-context qualification and
accepted-arrangement proposal distinguish that intent from the earlier bound
strategy solution. A renamed `program.step()` or all-purpose lifecycle facade
does not solve the underlying problem. Selected training and capability
behavior, dependencies, authority and reconstruction meaning must survive
acceptance without consulting the authoring role again.

**Adoption.** User
[079](sessions/10-accepted-arrangement-and-openspec-transition.md#m079) asks
for misleading records to be corrected while retaining what remains valid.
The direction, exchange, inventory and comparison are reconciled; the leftover
establishment-result wording is then explicitly fixed. `bb763b59` first commits
the corrected accepted-arrangement topology. Q1-Q5's authority and state
meanings survive; the assumed active-strategy enclosure does not.

**Current disposition.** Governing
[`accepted-training-execution`](../../../openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md)
retains selected executable behavior through accepted runtime seams and
requires whole-run structure, rather than merely an opaque step callback.
A selected object can implement a runtime role; its authoring interface does
not thereby gain execution authority. Current accepted meaning may include
stateful algorithms and bounded runtime decision policies. The early phrase
that all variability is resolved before training must not be read as fixing
every future branch, observation or outcome at authoring time.

**Missing detail or G5 consequence.** Preserve inspectable whole-run meaning,
current authority, reconstruction requirements, dynamic policies and proper
physical realization. Keep private numerical implementation behind supported
seams without reconstructing family logic in Trainer. Performance claims in
this early comparison were unmeasured opinions, not requirements to retain
one dispatch or proof that a proposed lowering is fast.

**Placement.** The accepted arrangement and governed realization belong in
the architecture and current requirements. Retain old callback sketches and
their correction in history. The seven-message copied August discussion is
a companion record, not the original July source or a second specification.

<a id="h041"></a>

## H041 - Structured and imperative behavior compose under explicit authority

**Source and rationale.** User
[061](sessions/10-accepted-arrangement-and-openspec-transition.md#m061) asks
whether authoring can end without moving capabilities into Trainer. The
discussion distinguishes implementation home, author selection and runtime
ownership. Generic request/lifecycle coordination does not transfer sampling,
conditioning, objective or persistence semantics into Trainer core.

User [068](sessions/10-accepted-arrangement-and-openspec-transition.md#m068)
recovers the structured-recipe ambition and demands a strong research path.
The supplied [feedback](snapshots/2026-08-26-arrangement-and-imperative-feedback.md)
and evaluation [073](sessions/10-accepted-arrangement-and-openspec-transition.md#m073)
make two lasting refinements: strategy authors an imperative execution role;
that role executes under accepted grants. Maintained/custom operations and
imperative regions can coexist inside one accepted arrangement instead of
forcing the whole run into a separate manual world.

**Adoption and limits.** The user requests the resulting correction and
`bb763b59` records EX-029 through EX-031 with the corrected topology.
The normal path needs structured accepted meaning, but its graph, operation
tree, regions, schedule, lowered-Python or hybrid realization is not selected
by this August discussion. The user also explicitly qualifies decorators as
one API idea [077](sessions/10-accepted-arrangement-and-openspec-transition.md#m077).
Illustrative decorator names and profile ladders are not adopted APIs.

**Current disposition.** Current execution and workflow requirements preserve
composable regions, declared effects/state, scoped grants, handback and
completion/failure meaning. Research code targets deliberately supported
runtime services rather than arbitrary Trainer fields. Its use of the same
Python object for authoring and a selected runtime implementation is possible,
but the roles and authority remain distinct. Choosing xDSL later supplies
machinery for these project-owned semantics; it does not replace them.

**Missing detail or G5 consequence.** Define the accepted interactions needed
across owners and boundaries, including derivatives, retained values and
physical completion. Do not require every private numerical operation to
become a public graph node. Keep ordinary maintained selection convenient
while explicit extension conformance supports new operations and granted
imperative work without a parallel competing engine.

**Placement.** Keep the common semantic interfaces and authority requirements
in the governing language/workflow design; selected implementations remain
with their domains. Preserve the external discussion's changes of position
and the adopted corrections rather than treating its opening answer as law.

<a id="h042"></a>

## H042 - Intended Trainer goals define the contract; OpenSpec governs further design

**Source and correction.** A real SDXL mapping makes the target concrete, but
user [128](sessions/10-accepted-arrangement-and-openspec-transition.md#m128)
rejects using that one shape as the system definition and reiterates that the
contract exists before strategy authoring. The correction requires materially
different SD/SD3 paths and a strong research case before calling execution
design ready. Current cases supply evidence and reveal pressures; they do not
unilaterally dictate the universal operation vocabulary or pipeline.

User [139](sessions/10-accepted-arrangement-and-openspec-transition.md#m139)
also corrects the apparent reverse authority. Intended Trainer goals and
supported capabilities determine what its contract describes. The contract
then guides and judges authored strategies. Designing API details backward
from intended consumers does not mean selecting a contract after writing an
arbitrary strategy, or letting an external contract overrule the repository's
Trainer design.

**OpenSpec transition and adoption.** User
[142](sessions/10-accepted-arrangement-and-openspec-transition.md#m142) asks
whether the next work belongs in OpenSpec rather than indefinite preliminary
exploration. The answer [144](sessions/10-accepted-arrangement-and-openspec-transition.md#m144)
proposes one governing change, with responsibility -> requirement -> exchange
-> scenario -> milestone -> acceptance-test traceability. User
[145](sessions/10-accepted-arrangement-and-openspec-transition.md#m145) requests
recording that transition; [153](sessions/10-accepted-arrangement-and-openspec-transition.md#m153)
records it after historical review and cleanup. `bb763b59` commits the result,
but no OpenSpec artifact is created in this bounded passage.

**Current disposition and G5 consequence.** The governing change inherits
settled direction and adopted exchange meanings while keeping inventory,
chronology, raw opinions and experimental evidence in their distinct roles.
New governing design belongs in OpenSpec; artifact creation alone does not
establish production readiness. Concrete implementation must retain the broad
supported workflow meanings, required evidence and reviewed migration gates.
One example's successful execution or a framework choice cannot erase those
requirements or authorize conflicting old/new state owners in one run.

**Placement.** Keep proposal/design/spec/tasks under the governing change.
Keep historical derivation, source inventory, experiments and supersessions
linked as inputs. Readings11-12 now recover the actual first draft, archive
and governing rebuild; the August transition alone did not date them.

<a id="h043"></a>

## H043 - A summary is not evidence of complete carry-forward

**Source and correction.** [Reading11](sessions/11-first-openspec-draft-and-reconciliation.md)
separates the five-message orientation root from the actual drafting root.
The latter reports a proposal, design, five deltas and tasks, but user
[015](sessions/11-first-openspec-draft-and-reconciliation.md#m015) asks whether
the strategy documents were read. Assistant
[016](sessions/11-first-openspec-draft-and-reconciliation.md#m016) acknowledges
only selected sections and summary reliance. User
[017](sessions/11-first-openspec-draft-and-reconciliation.md#m017) explicitly
rejects a summary as sufficient; the root ends during a fuller reread.

**Rationale and disposition.** Settled direction must be transferred with its
qualifications, source roles and supersessions. Creating the planning artifacts
and passing structural validation does not demonstrate that transfer. The
later rebuild's source-authority table and reconciliation gates address this
problem; the historical failed review supplies no approval.

**Missing detail or G5 consequence.** Concrete implementation must retain
traceability to adopted requirements and their rationale. Session summaries,
framework selection and test totals cannot replace those obligations. This is
not authority to reopen every settled question or preserve every old API.

**Placement.** Keep the first attempt as historical comparison evidence, with
the exact user correction beside it. Governing artifacts remain in OpenSpec.

<a id="h044"></a>

## H044 - The first draft survives as a ZIP; its production gate was explicit

**Source and evidence.** The [source record](sessions/11-source-records.json)
identifies `docs_design/archive/models-strategy-trainer/packages/mst_pre_astra.zip`. All nine files are preserved;
eight match the original public authoring patches exactly, including the
scenario-title update. The [design](snapshots/2026-09-18-first-openspec-draft/design.md)
and [tasks](snapshots/2026-09-18-first-openspec-draft/tasks.md) explicitly gate
production. Later criticism that this draft was already implementation-ready
must therefore be qualified: its carry-forward and cross-spec coverage were
incomplete, but it did not grant production readiness in those documents.

**Current disposition.** Superseded attempt, not an active requirement source
or a plan to restore. Its four new deltas plus one loaded-component modification
are narrower than the rebuilt package's seven new/eight modified capabilities.
The historical archive operation is ZIP storage, not proven spec sync or
successful OpenSpec archival.

**Missing detail or G5 consequence.** No exact actor/time/method for zipping
or removing the earlier active directory is established. The ZIP has no HEAD
history and is untracked; an all-refs search failed on an unrelated broken
checkpoint ref. These limits do not prevent auditing its actual contents.
Do not use an assistant's later critique to overwrite contrary primary text.

**Placement.** Retain the original ZIP and exact extracted copies as historical
evidence. A future physical relocation should preserve their hash/provenance
and update links; no original was moved in this audit.

<a id="h045"></a>

## H045 - Explicit spec reconciliation preserves metadata meaning without fixed landing order

**Original source and rationale.** [Reading12](sessions/12-governing-rebuild-and-contract-system.md)
rebuilds from the recorded direction, exchanges and evidence. Its first
[review](snapshots/2026-09-19-governing-integration-review.md) finds genuine
remaining conflicts: delta identifiers, loaded-component/provenance duties,
mode-gated scenarios and overlapping metadata requirements. The
[follow-up](snapshots/2026-09-19-governing-integration-follow-up.md) confirms the
fixes, including a metadata-first landing order proposed during that repair.
The final package has seven new/eight modified capability deltas.

**Supersession and adoption.** User
[043](sessions/12-governing-rebuild-and-contract-system.md#m043) says metadata
work must build on or with this rework. User
[045](sessions/12-governing-rebuild-and-contract-system.md#m045) confirms a
practical boundary, keeping the broader metadata system in mind. The
[recorded correction](sessions/12-governing-rebuild-and-contract-system.md#m048)
removes blanket metadata-first ordering. Runtime identity/state/authority
follow this architecture; catalog, structural, source/provenance and durable
history duties survive. Shared-boundary work is coordinated by real milestone
dependencies. A review's endorsed ordering does not override the later user.

**Current disposition and G5 consequence.** The current governing design
retains this runtime/evidence separation. Loading evidence describes what was
produced; accepted authority owns what is current and usable; metadata records
projections. Remove displaced owners and preserve evidence in the same actual
cutover. Do not silently delete requirements by archiving incompatible deltas.
A historical validation failure in the other metadata change is not a fresh
claim about its present state or an authorization to fix it during this audit.

**Placement.** Governing cross-change dependencies belong in the two OpenSpec
planning sets; chronology and original review recommendations remain evidence.
The exact [first checkpoint](snapshots/8ec7249d-rework-model-strategy-trainer/design.md)
and reciprocal metadata files preserve the corrected order.

<a id="h046"></a>

## H046 - One training contract system, with definition and lifecycle responsibilities distinct

**Source and correction.** User
[054](sessions/12-governing-rebuild-and-contract-system.md#m054) rejects
“Trainer-owned” as code ownership, asks what admission actually means, and
keeps checker/rule representation open. Supported Trainer/pipeline behavior
shapes the contract; that does not put the contract inside a Trainer object.
The pass/fail boundary and diagnostics are intended semantics, not a selected
validator implementation. Optional to select remains mandatory to satisfy
once selected; selection does not automatically assemble missing behavior.

The assistant first separates authored acceptance and concrete enforcement.
User [068](sessions/12-governing-rebuild-and-contract-system.md#m068) questions
separate checking mechanisms, then
[070](sessions/12-governing-rebuild-and-contract-system.md#m070) asks for a
model based on the whole system's goals. The corrected proposal is one
Training Contract System with a narrow definition, strategy fulfillment,
governed realization and continued run authority. They share vocabulary,
conformance knowledge and run obligations without requiring one god object,
four classes, or independent validators. User
[076](sessions/12-governing-rebuild-and-contract-system.md#m076) supports it
if it improves the system and [078](sessions/12-governing-rebuild-and-contract-system.md#m078)
authorizes the concrete update; [085](sessions/12-governing-rebuild-and-contract-system.md#m085)
records the reviewed result.

**Current disposition and G5 consequence.** Carried forward in current
`training-contract` and design D1/D2/D5. Fulfillment establishes accepted
meaning and revisioned obligations; later stages consume them instead of
reinterpreting the authored strategy. xDSL is a substrate for those meanings,
not a reason to discard them. Concrete package/API boundaries should keep the
shared authority explicit while respecting different lifetimes and inputs.

**Placement.** Governing meaning belongs in the OpenSpec design/specs; raw
terminology experiments remain visible here. “Admission,” “acceptance check”
and “fulfillment” did not each establish a separate system.

<a id="h047"></a>

## H047 - Check at earliest authoritative evidence; preparation helps create that evidence

**Source and rationale.** User
[064](sessions/12-governing-rebuild-and-contract-system.md#m064) challenges an
“all good” check followed by preparation failure. The distinction is between
invalid authored meaning, unmet concrete realization obligations, and contingent
physical failure. Different evidence times do not establish separate validators.
The [external opinion](snapshots/2026-09-20-contract-system-feedback.md) improves
the progressive-evidence model, but assistant
[075](sessions/12-governing-rebuild-and-contract-system.md#m075) corrects it
before adoption: backend preparation can create wrappers, routes and optimizer
objects that must then be verified before publication. A verified concrete
run cannot be assumed to exist before that work.

**Adopted invariant.** Fulfillment checks authoritatively knowable facts and
records the remaining obligations/readiness checkpoints. Concrete realization
and later transitions use the obligation set from the exact accepted revision.
Permitted amendments may revise obligations; “one source of meaning” does not
mean permanently frozen obligations or predetermined runtime observations.
Preparation collects evidence and produces candidates; it cannot independently
define compatibility or certify its own output. Unsupported requested behavior,
known infeasibility and a supported attempt's OOM/network/backend failure are
distinct outcomes, not proof that semantic acceptance guaranteed success.

**Current disposition and G5 consequence.** Shared contract, preparation,
authority and execution requirements preserve these boundaries. Reject known
invalid combinations before expensive effects; withhold use/readiness until
required evidence exists; retain runtime currency/authority checks. Coordinate
final publication across actual owners. Its coordinator was explicitly open
for G1 at this checkpoint, not silently selected by passive prose or proof
that the later design never resolved it.

**Placement.** Keep the corrected model and readiness rules in governing specs.
Retain the external diagram as opinion with its adopted and rejected pieces
identified, rather than rewriting the supplied text to match the outcome.

<a id="h048"></a>

## H048 - Artifact completeness, successful review and production readiness are different claims

**Source and sequence.** The rebuild's initial expired-token failure is not
approval. User [024](sessions/12-governing-rebuild-and-contract-system.md#m024)
fixes credentials and requests the same review. The initial integration report
finds issues; corrections and follow-up precede the joint walkthrough. The
[contract refinement review](snapshots/2026-09-20-contract-model-review.md)
then finds undefined environment facts, readiness-checkpoint meaning, the open
publication coordinator and small scope/terminology mismatches. The following
historical edits narrow or define these rather than declaring them resolved
by review alone. Some cosmetic terminology outside the approved files remains
for later discussion.

**Current disposition.** `8ec7249d` (September20 03:23:38UTC) is the first
committed governing package, after those walkthrough corrections. Its
nineteen files include all fifteen deltas; four prior records get status updates
and two metadata planning artifacts get reciprocal changes. It is not an exact
snapshot of the thirteen-delta draft before the reviews. Planning files exist
and parse does not mean design gates or production migration are complete.
These historical reviews are not current production qualification.

**Missing detail or G5 consequence.** Later gate completion and implementation
must be read at their actual point in the chronology. The continuing G5 work
needs production conformance, ownership cutover, physical completion and
representative evidence; this provenance capture supplies none of those by
itself. Failed review attempts must remain distinguishable from findings and
successful completed reviews.

**Placement.** Preserve exact first-commit artifacts and original review
reports as dated evidence. Active specs and later disposition govern the
current system; this notepad supplies source/rationale, not a readiness badge.

<a id="h049"></a>

## H049 - Completed authoring preserves dynamic behavior and its initialization

**Original source and rationale.** User
[025](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m025)
challenges the implication that a frozen authored strategy removes runtime
dynamics. The [correction](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m027)
separates ending the authoring role from preserving executable algorithms.
Schedules, observations, feedback, branching, randomness, owned state and
bounded decisions remain active. User
[028](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m028)
authorizes the change; [036](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m036)
records the reviewed correction. A selected object may retain a narrow runtime
role without retaining the broad strategy interface or mutation powers.

**Refinement.** The user supplies the pre-compaction message to check fidelity.
The comparison finds fresh-state initialization implicit; the selected
[039-042](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m039)
exchange records either an accepted initial source or an initialization rule,
including rules that wait for concrete evidence. Fresh initialization is not
restoration, and does not require a live value during authoring. The alternate
redundant confirmation branch is preserved separately.

**Current disposition and G5 consequence.** Carried in accepted-execution and
contract requirements. Static selection/wiring can leave the hot path while
observable schedule semantics survive equivalent lowering. A policy is checked
against accepted authority, not unconditionally accepted. Operation-owned
state needs dependency, freshness, invalidation and continuation meaning;
semantic acceptance cannot reduce a dynamic run to predetermined outcomes.
Python authoring and xDSL do not weaken this requirement.

**Placement.** The original dynamic-behavior
[review](snapshots/2026-09-20-dynamic-behavior-review.md) and exact
[committed requirement](snapshots/58ab91ce-rework-model-strategy-trainer/specs/accepted-training-execution/spec.md)
retain adoption and qualifications. Algorithms remain with selected domain
implementations; this does not mandate a public operation for every tensor step.

<a id="h050"></a>

## H050 - Shared maintained PEFT composes behavior without becoming a replacement mode

**Source and user correction.** Capability provision, request and readiness
are distinguished in [046-048](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m046).
Repository availability alone does not provide a capability to a strategy.
User [051](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m051)
explains that the mode split came from old standalone scripts. User
[053](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m053)
corrects the assistant's over-fragmentation: maintained PEFT methods share
training integration, while each method contains its actual differences. A
strategy supports a declared current set, not every possible future adapter.

**Rationale and adoption.** A reusable authored integration can contribute
participants, relationships, realization behavior, optimization intent,
accepted operations, products and continuation state. That avoids both
per-method Trainer wiring and a renamed monolithic `AdapterMode`. The
FineTuneMode inspection also separates direct parameter training from full-model
output. User [058](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m058)
authorizes reconciliation, requests a raw-message check after compaction, and
[067](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m067)
authorizes the concrete update. The review catches an unconditional method
selection requirement: providing PEFT support must not force every run to use it.

**Current disposition and G5 consequence.** TrainingMode dissolves by real
responsibility; shared PEFT remains reusable authored composition and methods
stay domain-owned. Selection is explicit construction within supported bounds;
a runtime mode/config resolver must not silently assemble behavior. No mode
value or replacement orchestration authority reaches Trainer. Joint base and
adapter training is compatible where its accepted contracts permit it.

**Question raised during this walkthrough, 6 November 2026.** The user notes
that “declared current set” may sound like more maintenance than treating PEFT
as ordinary available feature composition, and may be self-evident from how a
strategy is written. This audit does not infer an extra separately curated
adapter allowlist or settle how support is represented. Whether compatibility
naturally follows from composed features is deferred until the concrete system
can be assessed. This is a present walkthrough concern, not a new September
decision or a change to the governing requirements.

**Placement.** Shared integration and method-specific implementations retain
different jobs; generic preparation/optimization does not stay inside a renamed
feature object. Concrete paths and APIs are not chosen by this historical
classification. H019's original mode decision is confirmed, not discarded.

<a id="h051"></a>

## H051 - Artifact choice is independent from the training treatment

**Original source and refinement.** The artifact/restoration review at
[050](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m050)
retains Q4 identity and completeness distinctions: an artifact can succeed
while a same-run snapshot fails, even when sharing storage/triggers. Trainer
or delegated pipeline coordinates continuation; capabilities, operations,
optimization and backend contribute the state they own.

The [mode/PEFT review](snapshots/2026-09-20-mode-peft-review.md) finds product
independence only implicit in design/tasks. User
[075](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m075)
authorizes making it normative; [078](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m078)
records the requirement. Direct training may emit a full model; PEFT or joint
training may emit adapter, merged full-model or other accepted products. The
product follows its explicit semantic declaration, not a training-mode label.

**Current disposition and G5 consequence.** Preserved in artifact persistence
and its later broader product cases. Saving a product, initializing another
run from it, and restoring the exact same run have different identity,
evidence, transformation and completeness requirements. A serializer or the
selection of training subjects cannot infer semantic product coverage or
promise a resume point from weights alone. Preserve independent actual results
and every required operation-owned continuation contribution.

**Placement.** Product semantics belong in the accepted declaration and domain
contributors; generic persistence/restoration coordinate infrastructure.
No separate fine-tune/adapter mode should be reconstructed for saving. The
[checkpoint requirement](snapshots/58ab91ce-rework-model-strategy-trainer/specs/training-artifact-persistence/spec.md)
is first committed after this explicit review-driven addition.

<a id="h052"></a>

## H052 - Adapter target intent, host resolution, construction and optimization have distinct owners

**Source and correction.** An earlier branch's
[user clarification](sessions/13-branch-differences.md#m001) distinguishes
model-defined structure, strategy treatment and concrete runtime selection.
It is outside the selected page prefix and is not falsely described as an
inherited turn. The selected spec audit independently finds the old
optimization-owned adapter-targeting rule conflicting with the new architecture
at [092-093](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m092).
User [094](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m094)
requests correction and [098](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m098)
authorizes the coordinated update.

**Adopted chain and rationale.** Models/model integration define available
structure. Authored strategy and shared PEFT integration declare semantic target
intent. Governed PEFT realization resolves it against current, scoped host
evidence. The adapter method consumes supplied targets and returns trainable
references with provenance. Trainer optimization realizes parameter
trainability, grouping and advancement. A shared target-reference helper's
physical location does not make optimization own model composition.

The [review](snapshots/2026-09-21-adapter-ownership-review.md) preserves the
finer-than-module future target possibility, absorbed-method guarantees,
config-to-authored-continuation mapping and explicit metadata overlap. Methods
may reject unsupported targets but cannot substitute their own traversal or
policy for the accepted intent. Unsupported overlapping adapter assignments
need declared coordination, rather than resolution inventing precedence.

**Current disposition and G5 consequence.** The governing adapter/target specs
carry the split. Preserve stable provenance and revision/current-host evidence
while avoiding optimizer-owned attachment policy, method-owned model discovery,
mode authority or universal module-only constraints. Broad adapter support is
not permission to accept arbitrary composition without its contracts.

**Placement.** Adapter-domain realization, shared policy-neutral reference
machinery and optimization parameter grouping remain separate responsibilities.
The exact [targeting delta](snapshots/58ab91ce-rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md)
replaces the old ownership requirement rather than leaving contradictory text.

<a id="h053"></a>

## H053 - Representative cases constrain design; deferred API names do not reopen architecture

**Source and adoption.** [082](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m082)
separates case breadth, design gates and production migration. User
[083-085](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m083)
approves clearer joint base-plus-PEFT, teacher/student, video/audio and adaptive
cases. G1 provides the common consumption/evidence frame; G4 supplies complete
capability-specific exchanges. The former must not quietly swallow the latter.

After the fifteen-delta audit, [118](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m118)
classifies four explicit open questions as deferred production-shape choices:
class/module names, reference encoding, publication mechanism and authoring
syntax. The substantial execution, optimization and capability design remains
in G1-G4. User [119](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m119)
authorizes cleanup, including earliest-evidence timing and explicit unsupported
mixed-method overlap without coordination. `58ab91ce` commits the result at
September21 07:02:35UTC. This is not completion of those design gates.

**Current disposition and G5 consequence.** Later gates and language work
must be read at their actual historical point. A deferred name/API decision
cannot be used to reopen identity, authority, source roles or authored dynamic
behavior. Conversely, later implementation evidence is not proof that all
composition semantics were settled by this early audit. Representative cases
are architectural pressures, not a universal SDXL-shaped schema.

The early migration rule allows old/new paths for separate runs while forbidding
hybrid authority within one run. It does not require keeping every intermediate
revision runnable. The later evening discussion challenges that additional
constraint; the current migration plan explicitly distinguishes dependency
order from intermediate runnable states. Reading 14 and H055 trace that correction.

**Placement.** Keep gates, traces and adopted meanings in OpenSpec. Readability,
concrete placement and actual code design require visible examples; lifecycle
terms alone do not dictate a package tree or final API.

<a id="h054"></a>

## H054 - One authority is an ownership invariant, not a large class or a weight-update clock

**Source and reaffirmation.** The authority alternatives at
[006](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m006)
compare Trainer ownership, active strategy, distributed participants, immutable
snapshots with transitions and a broad session facade. User
[007](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md#m007)
chooses one canonical semantic-state authority in the accepted arrangement.
The code shape remains open. Snapshot mappings may hold live handles without
copying models; the authority does not perform loading, backend work or training.

**Refinement and rationale.** Ordinary authorized optimizer updates advance
owned numerical state/progress rather than replacing participant identity or
revising a binding on each step. Structural or semantic changes, replacements,
relationship changes, prepared-route invalidation and retirements remain
explicit transitions. The preparation review also reaffirms that one “job”
means one coordinated attempt, not a thread, queue or new job subsystem.

**Current disposition and G5 consequence.** This preserves the original Q5
single-writer requirement while avoiding a god object or needless public
noun hierarchy. The binding-revision distinction does not waive current-state
provenance, observation epochs, protected reads or retained-use settlement when
consumers depend on changing weights. Those obligations follow their actual
state/effect contracts and later language/runtime work.

**Placement.** Canonical semantic truth belongs to the accepted run; other
owners retain their own optimizer, backend, operation and capability state.
Public class names and a physical registry-versus-snapshot choice are not
selected by this reaffirmation. H028/H040 remain the earlier authority/topology
sources; this passage confirms their corrected placement.

<a id="h055"></a>

## H055 - Concrete implementation design does not require runnable intermediate products

**Source and correction.** The initial implementation-map proposal follows a
renewed goals exercise. User [010](sessions/14-training-first-placement-and-one-run.md#m010)
rejects revalidating the premise and [012](sessions/14-training-first-placement-and-one-run.md#m012)
asks for actual order, changes and missing code. The assistant then turns
internal responsibilities into extra stages and a first runnable-path plan.
User [020](sessions/14-training-first-placement-and-one-run.md#m020) rejects
that constraint: Git preserves the old code; intermediate branch edits may be
unfinished, with focused testing as needed. Visible file/folder/concern sketches
must precede agreement on implementation direction.

**Rationale and disposition.** Work should arrive at one coherent replacement,
not maintain a hybrid product or compatibility facade merely to keep every
intermediate revision operational. The current migration plan explicitly says
construction dependency order does not require every intermediate repository
revision to remain runnable. Its allowance for separate old/new launch paths
is not a mandate to build or preserve such a rollout.

**G5 consequence.** Produce concrete examples and responsibility/file-level
dependency maps before implementation. Reviewed milestones and meaningful
internal tests remain useful; they do not require shipping narrow product
states or preserving displaced authority within the new run. Historical sketches
of a flat step loop or a four-object API are illustrative, not the final
whole-workflow execution model.

**Placement.** Implementation/migration decisions belong in governing design
and tasks. The superseded extra-stage map and alternate four-object sketch
remain visible in their source branches, without becoming a second plan.

<a id="h056"></a>

## H056 - Training is the process; Trainer is an actor within it

**Source and correction.** User [024](sessions/14-training-first-placement-and-one-run.md#m024)
places the contract with strategy concerns and asks for actual jobs before
another invented layout. The A-G annotations at
[027](sessions/14-training-first-placement-and-one-run.md#m027) distinguish
authored definition, handoff, current state, construction/configuration, installed
computation, generic mechanics and capabilities. User
[029](sessions/14-training-first-placement-and-one-run.md#m029) corrects the
equation of `training/` with the Trainer class: the package describes the whole
process, including phases. B/C/D therefore belong to training, rather than an
unowned territory between strategy and Trainer.

**Adopted boundary.** A phase expresses when/how work is coordinated; a
capability describes accepted behavior that can be requested at several points.
Neither distinction alone requires duplicate directories. Domain implementations
stay with their owners. Later user [070](sessions/14-training-first-placement-and-one-run.md#m070)
asks why initial creation is outside state; the correction places creation
beside the state/transition system it establishes. Semantic checking still
uses the contract's one authoritative meaning; no second validator is implied.

**Current disposition and G5 consequence.** Preserve contract-guided authoring,
initial accepted state, governed realization and runtime continuity without
assigning every lifecycle word a public subsystem. Training-owned coordination
does not move model/adapter/objective internals into Trainer. A-G are labels
used to reason about responsibilities, not a rigid runtime order or universal
language grammar. Initial-state creation and subsequent transitions must agree
on the same run identity and obligations.

**Placement.** The provisional tree places contract/checking with strategies,
current state/creation with training, and mechanics/coordination with training
phases/capabilities/Trainer. Exact filenames and internal splits remain subject
to concrete consumers and later G5 design.

<a id="h057"></a>

## H057 - The contract is readable, queryable and the single source of behavior interfaces

**Source and rationale.** User
[042](sessions/14-training-first-placement-and-one-run.md#m042) requires readable
code and queryable guidance, combines features/shared as one concern, and
rejects treating “schema” or “authoring” as predetermined filenames. An
assistant overcorrection collapses substantial code; user
[044](sessions/14-training-first-placement-and-one-run.md#m044) rejects both a
root `strategy.py` and the choice between one giant file and dozens of noun
folders. User [046](sessions/14-training-first-placement-and-one-run.md#m046)
points out that understandable common seams are the contract's job, not a
parallel `base/` interface system.

The detour into current strategy files is then challenged by
[052](sessions/14-training-first-placement-and-one-run.md#m052): already extracted
legacy structure should not reset target design. The actual last messages are
reread after compaction. User
[065](sessions/14-training-first-placement-and-one-run.md#m065) accepts a
provisional contract/features/maintained-strategy grouping with names open;
[077](sessions/14-training-first-placement-and-one-run.md#m077) accepts the
combined strategy/training responsibility sketch on the same basis.

**Current disposition and G5 consequence.** Author guidance, types/protocols,
requirements and checking must derive from the same contract knowledge.
Queryable help cannot become a second registry maintained independently from
verification. Reusable features are implementations; contract feature surfaces
define their requirements. Substantial family behavior stays separated by its
actual job, while maintained or compound strategies compose explicitly.
Neither ordinary users nor every feature addition must hand-build a new
framework interface or a mirror taxonomy.

**Placement.** `contract/`, reusable feature implementations and strategy
packages are the accepted provisional concern groups here. Internal module
names and file counts are illustrative. xDSL does not select the public Python
authoring API or revoke the readability/one-authority requirement.

<a id="h058"></a>

## H058 - Training-first means the intended full system, with current code supplying evidence

**Source and adoption.** User
[079](sessions/14-training-first-placement-and-one-run.md#m079) says training
comes first and [081](sessions/14-training-first-placement-and-one-run.md#m081)
clarifies that this means the broad system the rework intends to enable.
Future extensions are possible, but “extensible later” must not excuse leaving
already intended cases unspecified. The contract follows supported meanings
of the intended process, not a runtime Trainer instance manufacturing a schema.

The user then explicitly authorizes a training-folder sweep at
[084](sessions/14-training-first-placement-and-one-run.md#m084). It identifies
what stays, splits, moves out, moves from strategy into generic coordination,
and genuinely new machinery. That authorization differs from the earlier
unrequested strategy-layout detour. User
[090](sessions/14-training-first-placement-and-one-run.md#m090) accepts the
state placement of missing input/observation concerns and
[094](sessions/14-training-first-placement-and-one-run.md#m094) authorizes
recording the ownership frame.

**Current disposition and G5 consequence.** Reuse actual mechanics where they
fit; do not use their current API, family fields, callbacks or folder names
as the ceiling of accepted behavior. The scope includes compound participants,
multiple optimization units, adaptive state, persistence/restoration, protected
transitions and strong research authority. “Move to Trainer” commonly means
training-owned coordination, while domain computation remains selected behavior.
The historical sweep is not fresh validation of current production code.

**Placement.** The committed `722da952` design first records the training-first
disposition and composed-state frame, with later research included too. It is
not a fresh source audit or an exact snapshot of the first sweep reply. The
copied sweep's coarse adapter-targeting line remains subordinate to H052 and
the specific governing target-resolution requirements.

<a id="h059"></a>

## H059 - One accepted run has structured state and several narrower owners

**Source and correction.** The sweep exposes participant/binding state,
Trainer execution state and behavior-owned state mixed together. The added
input/observation concerns fit under a composed run state, but this does not
make participant authority own progress, inputs, optimization or metrics.
User [100](sessions/14-training-first-placement-and-one-run.md#m100) prefers
`state` as a provisional name; user
[103](sessions/14-training-first-placement-and-one-run.md#m103) challenges an
explanation that sounds like decomposing the completed strategy into unrelated
state objects.

The [correction](sessions/14-training-first-placement-and-one-run.md#m104)
retains one accepted run: accepted meaning/obligations, selected executable
behavior, initialization rules and structured evolving values. There is one
current value for each concern and coherent publication where concerns interact.
It does not require one flat dataclass, one writer for all state, or one object
with methods for every operation. User
[105](sessions/14-training-first-placement-and-one-run.md#m105) then asks for
a real design/language capable of making that composition work.

**Current disposition and G5 consequence.** Preserve coherent whole-run meaning
and scoped ownership while allowing runtime state to evolve. Selected behavior
is more than stored values; an accepted root must retain the executable
algorithms and their obligations. Participant currency, optimizer progress,
input/adaptive state and observation state must not become competing copies
or acquire arbitrary authority from being in the same container. Later IR and
runtime design supply the concrete composition mechanics; this sketch does
not complete them.

**Placement.** Structured training state and transition code belong together;
materializers, backend algorithms, phase order, serializers and logging sinks
do not all move into a `state` package. Naming and aggregation shape remain
provisional until actual code/language choices make the interactions reviewable.

<a id="h060"></a>

## H060 - The provisional tree is a copied sketch and sweep, not a production blueprint

**Source and evidence.** `provisional_tree.md` first appears in `722da952`,
September23 07:01UTC. The [exact 457-line snapshot](snapshots/722da952-provisional_tree.md)
contains entire bodies from this reading's
[076](sessions/14-training-first-placement-and-one-run.md#m076) combined tree
and [089](sessions/14-training-first-placement-and-one-run.md#m089) training
sweep, after comparison-only line-ending normalization. Its top notes explicitly
leave names unresolved. Present-inode birth, modification and first Git capture
are separate dates; none proves who pasted the file.

**Disposition and rationale.** Broad concern grouping receives provisional
user assent; every path, file split, directory category and copied sweep
sentence does not thereby become normative. The state-name preference and
later one-root clarification already go beyond the pasted layout. The copied
line assigning concrete target and parameter resolution to optimization is
coarser than the governing PEFT/optimization separation; it is recorded as a
wording limit, not silently promoted or corrected in the original.

**Missing detail or G5 consequence.** Use this file as historical illustration
of what was under discussion. Derive final modules from adopted semantics and
actual recurring consumers, preserve source/rationale, and check later
supersessions before creating files. A prior diagram is not authority to
manufacture obsolete types after the language and machinery evolve.

**Placement.** Keep it with historical planning/derivation and link its exact
source messages. Its future home should not be a live module README or a
second implementation checklist. No original moved or changed in this audit.

<a id="h061"></a>

## H061 - Research cases pressure a broad system rather than define an SD-shaped Trainer

**Decision, source and rationale.** The user rejects treating SD/SDXL/SD3 variation
as proof of broad expressiveness ([084](sessions/15-research-and-whole-run-language.md#m084)). The research index and
recipe notes supply concrete model, data, precision, distributed, distillation,
control-network and RL demands. Observed, inferred and unknown findings remain
separate. The earlier metadata goals already included independently progressing
data work ([148](sessions/15-research-and-whole-run-language.md#m148)); this is continuity, not a new requirement invented
by a graph experiment.

**Current disposition.** Research is design evidence. Specific requirements
enter the governing design/deltas through explicit adoption and correction.
A framework's taxonomy or a single successful recipe is not the project ontology.
**G5 consequence and placement.** Preserve the already intended breadth and
source-backed limitations. Keep external recipes in research, requirements in
specs, and experiments with their evidence; avoid another broad survey when
an unresolved concrete interaction is the actual question.

<a id="h062"></a>

## H062 - The accepted whole run is composed; the graph is executable meaning

**Decision and source.** The user corrects repeated arguments against a whole-run
graph and requests building the composable IR against current and planned cases
([187](sessions/15-research-and-whole-run-language.md#m187)). The resulting formulation describes the accepted run as
structured regions with dependencies, ownership, effects, lifecycle boundaries
and supported coordination of independently progressing activities
([194](sessions/15-research-and-whole-run-language.md#m194), [196](sessions/15-research-and-whole-run-language.md#m196)). A per-step plan cannot describe every
producer or requested activity's lifetime.

**Rationale/current disposition.** The graph replaces hardwired composition;
it is not only a visualization above a fixed loop. Selected numerical algorithms
can remain internal Python/backend work where their external obligations are
clear. The MLIR analogy originally illustrated extensibility; early claims that
machinery must be custom Python are superseded by later substrate comparison.
**G5 consequence.** Show representation, preparation and actual runtime
coordination as one system. Do not replace accepted whole-run meaning with a
configurable input-action-optimizer pipeline. Keep working sketches as research,
not an additional governing API.

<a id="h063"></a>

## H063 - Completion, progress, physical effects and durable recovery are separate boundaries

**Source and rationale.** Alternating generator/discriminator execution is not
necessarily one atomic optimization transaction ([157](sessions/15-research-and-whole-run-language.md#m157)). External
words such as atomic/commit are qualified rather than adopted wholesale
([254](sessions/15-research-and-whole-run-language.md#m254)). An earlier update may have returned before a later stage
fails; an entered call may mutate state then raise. Run progress, successful
activity completion and a durable checkpoint cannot be inferred from one flag.

**Current disposition.** Later execution/optimization/workflow specs preserve
per-unit and per-activity outcomes, uncertainty, physical completion, settling
and checkpoint-scoped restoration. No rollback is established merely by a toy
control-flow experiment.
**G5 consequence/placement.** Keep independent effects and observable outcomes,
including partial success. Activity relationships determine completion and
recovery policy; a universal training-step sequence must not supply missing
semantics. Preserve failed/limited probes in evidence, not as new language limits.

<a id="h064"></a>

## H064 - Participant uses, optimizer units and input state vary independently

**Source and decision.** One participant may serve several roles without becoming
several identities ([231](sessions/15-research-and-whole-run-language.md#m231)); one action may advance several parameter
sets with different methods ([237](sessions/15-research-and-whole-run-language.md#m237)); input selection, mixture,
packing and resume position are semantic state rather than an opaque dataloader
call ([241](sessions/15-research-and-whole-run-language.md#m241)). Independent production also needs explicit source,
input and version association ([218](sessions/15-research-and-whole-run-language.md#m218)).

**Current disposition.** Identity, use, resource version, unit definition,
physical execution and provenance remain distinct throughout later language
and runtime work. Same-role names or one backend wrapper do not merge owners.
**G5 consequence/placement.** Carry these distinctions through lowered code,
requests, gradients and cold artifacts. Do not require one loss, one optimizer,
one global step or image-shaped inputs. The particular algorithms remain
selected domain behavior; the IR coordinates the relationships they expose.

<a id="h065"></a>

## H065 - Python effect contracts constrain supported use but are not a sandbox

**Source and rationale.** The feedback and response distinguish declared effects
and scoped access from control over arbitrary Python ([265](sessions/15-research-and-whole-run-language.md#m265)).
The first executable action tests wiring/authority and later value-source and
producer handoff cases; its location or passing tests do not make it production.

**Current disposition.** Installed selected implementations remain trusted.
Static checks reject knowable incompatibilities; runtime checks still enforce
issued authority, current versions, actual outcomes and physical settlement.
**G5 consequence/placement.** Do not claim arbitrary mutation or aliases are
prevented by an effect label. Preserve the enforcement/trust boundary and
explicit supported target profiles. Keep the early experiments as evidence,
with their fixed-shape restrictions distinguished from the evolving language.

<a id="h066"></a>

## H066 - Whole-run language development cannot be deferred behind its own prerequisites

**Source and correction.** The September26 discussion rejects presenting the
language as a later topic after repeated case surveys ([008](sessions/16-exchanges-audits-and-g5-correction.md#m008),
[012](sessions/16-exchanges-audits-and-g5-correction.md#m012)). Enough evidence exists to build one whole-run candidate.
The subsequent analysis is explicitly limited to what that fixed probe can
establish ([049](sessions/16-exchanges-audits-and-g5-correction.md#m049)).

**Current disposition.** Optimization, authority and preparation are parts of
the language's semantics, not prerequisites that replace it. One fixed probe
can show an interaction while failing to express general composition.
**G5 consequence/placement.** Develop the shared mechanism and its consumers
as a connected system. Keep experimental observations separate from requirements;
do not turn probe limitations into new semantics or ask the user to redesign
low-level engineering already bounded by accepted intent.

<a id="h067"></a>

## H067 - Optimization semantics include retained gradients, independent units and explicit authority handback

**Source and rationale.** G3 develops unit identity versus realization/progress,
contribution windows and gradient routing, selected advancement policy and
explicit research grants. Standard-profile ownership and exceptional transferred
actions remain distinct; unusual cadence alone does not transfer authority.
The original experiment counts are historical evidence, not proof of all backend
or production arrangements.

**Current disposition.** Governing optimization/execution requirements retain
source association, state/version dependencies, actual per-unit outcomes and
verified completion/handback before retained mechanics resume.
**G5 consequence/placement.** Realize these contracts with shared owners rather
than duplicate mechanisms per example. Retained numerical state and physical
completion survive across local calls/blocks; the later resource-product work
extends the same requirement beyond detached values. Domain algorithms stay
selected; ownership is not inferred from Python object placement.

<a id="h068"></a>

## H068 - Capability coordination remains generic while caching, evaluation and persistence retain domain meaning

**Source and progression.** G4 makes caching, validation/sampling, semantic
products, exact restoration and observations concrete. Current storage or model
writers are migration examples, not the capability envelope. Cache validity,
request eligibility and stored presence are different; evaluation can be adaptive
or require gradients without automatically owning optimizer advancement.

**Current disposition.** The shared design requires explicit readiness,
version/provenance, protection, results and failure/settlement boundaries.
Persistent artifacts and same-run restoration remain separate, including
contributor completeness and independent actual outcomes.
**G5 consequence/placement.** Reuse storage/format/backend mechanics where
appropriate without preserving an old workflow topology. Keep capability-specific
semantics and shared infrastructure distinct; do not cap product types or
input lifetimes at current SD writers/caches.

<a id="h069"></a>

## H069 - Missing restoration facts mean incomplete restoration, not an implicit new run

**Source and correction.** The pinned audit finds a stale introduction that
conflicts with the intended exact-restore rule. The actual correction is recorded
at [484](sessions/16-exchanges-audits-and-g5-correction.md#m484); earlier August fallback wording is superseded explicitly.
Likewise, standalone participant targets must preserve identity without inventing
family membership merely to satisfy a target-reference container.

**Current disposition.** Same-run restoration requires its identity, revisions,
continuation state and readiness evidence. A separate explicit new-run request
creates new identities with lineage. Genuine component provenance remains where
it actually exists.
**G5 consequence/placement.** Preserve source-qualified identities across
artifacts, resume, target resolution and current views. Keep audit reports at
their pinned baseline, corrected requirements at their later revision, and do
not infer supersession merely from a later timestamp.

<a id="h070"></a>

## H070 - The multi-pass audit validates semantics within scope, not a finished implementation

**Source and rationale.** The independent round pins its baseline at
`6db3f1ce` ([452](sessions/16-exchanges-audits-and-g5-correction.md#m452)), separates navigation, architecture, revisions,
execution/failure, capabilities, research breadth, tooling and integration,
and corrects verified findings locally while reports remain unchanged.
The closeout is recorded in `b19717fa`; failed provider attempts are not approval.

**Current disposition.** The source hierarchy, standalone-target correction,
restoration correction, parent/delta reconciliation and coverage limits survive.
Clean audit conclusions concern the reviewed requirements and evidence, not
GPU support, a complete language or production G5.
**G5 consequence/placement.** Use audit reports as scoped verification records,
not a parallel authority or a substitute for concrete mechanism design. The
active governing design/deltas express the accepted result.

<a id="h071"></a>

## H071 - G5 task order is insufficient without the connected execution mechanism

**Source and correction.** The user rejects designing state components in
isolation while the complete machinery is still invisible
([587](sessions/16-exchanges-audits-and-g5-correction.md#m587)). The October5 review of task history finds early G5 wording
still largely inherited from the first OpenSpec. The user explicitly requires
the complete concrete execution mechanism before blindly following it
([600](sessions/16-exchanges-audits-and-g5-correction.md#m600)). Later surface descriptions are challenged for omitting
the actual whole-system graph ([627](sessions/16-exchanges-audits-and-g5-correction.md#m627)).

**Current disposition.** Accepted requirements remain governing input. The
missing work is assembling representation, lowering, state/services,
coordination and completion into an inspectable whole, not rediscovering the
architecture or declaring it absent.
**G5 consequence/placement.** Show the whole system before evaluating isolated
types. Task checkboxes need their design context; completing an experiment or
choosing xDSL does not automatically close production types, APIs and migration.

<a id="h072"></a>

## H072 - The graph must generate execution and coordination rather than decorate another runtime

**Source and correction.** Whole-system mocks are judged by whether the accepted
structure actually determines executable work and oversight. The user challenges
a second run interpretation ([006](sessions/17-connected-candidate-and-ir-machinery.md#m006)) and later rejects a model whose
organization disappears into separately designed control programs
([043](sessions/17-connected-candidate-and-ir-machinery.md#m043)). The graph replaces hardwired composition, not just its
visualization ([063](sessions/17-connected-candidate-and-ir-machinery.md#m063)).

**Current disposition.** One accepted whole-run meaning may have semantic,
prepared and runtime representations with verified correspondence. Different
lowerings are permitted when dependencies, authority, outcomes and lifetimes
are preserved; a second independent source of run meaning is not.
**G5 consequence/placement.** Make preparation and runtime derivation inspectable,
including loops, feedback and independent activity. Candidate names and one
serialized coordinator are implementation choices, not universal semantics.

<a id="h073"></a>

## H073 - Shared ownership mechanisms are developed through interacting pressure cases

**Source and rationale.** The isolated candidate progressively connects dynamic
regions, independently live producers, requested work, correlated values,
retained derivatives, multiple optimizer units and explicit grants. The user
asks that shared mechanisms develop rather than accumulating scenario patches
([500](sessions/17-connected-candidate-and-ir-machinery.md#m500)); subsequent passes consolidate optimization and state/access
ownership.

**Current disposition.** Selection, admission, pending work, protected state,
physical calls and cleanup must agree across owners. A returned worker call
can leave obligations needed by later gradient work. Provider-owned native work
must use the same protection as engine-started calls.
**G5 consequence/placement.** A/B/C/mixed/lifetime probes must exercise this
same owner machinery. Preserve bounded active work and real progress checks;
do not confuse finite receipt/history evidence with bounded total storage for
indefinite execution. The candidate remains research, not production placement.

<a id="h074"></a>

## H074 - Semantic identity and reproducibility require real retained associations and owned randomness

**Source and rationale.** Input identity, origin/version and gradient destination
must survive observation-history erasure. Logical job RNG can preserve selected
results under changed worker order; shared global RNG or changed source state
requires different guarantees. The user correctly distinguishes an untested
claim from an inherent impossibility.

**Current disposition.** Reproducibility is derived from accepted input/state
and RNG policy, not advertised globally because one schedule repeated. Reports
need causal/value association, not merely a matching sequence of numbers.
**G5 consequence/placement.** Carry actual product/input/version links through
admission, retained uses and final consumers. Independent oracles and deliberate
corruption controls are required to distinguish replay consistency from semantic
correctness. Do not manufacture observations to satisfy a checker.

<a id="h075"></a>

## H075 - Backend and free-threading experiments do not choose the system architecture

**Source and correction.** Numerical compiler work proceeds separately from the
whole-run machinery. SDXL/Anima, optimizer and Accelerate examples supply bounded
backend workloads, not a selected future backend or rollout order. A free-threaded
interpreter, imports that keep the GIL disabled, concurrent safety and numerical
performance are separate claims.

**Current disposition.** Lowering suitable regions is an architectural direction;
Triton, CubeCL, cuTile, TileLang, Accelerate and Burn are not mandated foundations.
Python coordination can coexist with selected native/numerical work.
**G5 consequence/placement.** Preserve explicit backend offers, access/gradient/
completion obligations and representative qualification. Keep compiler timing,
CPU execution, GPU/device safety and free-threading results in their actual
scope; a passing compatibility probe does not establish all of them.

<a id="h076"></a>

## H076 - Learn from Togeari without splitting run authority or adopting it by default

**Source and adoption.** The user selects reference learning first and a separate
whole-system candidate next, rejecting a two-runtime hybrid
([839](sessions/17-connected-candidate-and-ir-machinery.md#m839)). The next candidate is clarified as project-owned and informed
by the reference, rather than reusing Togeari wholesale
([846](sessions/17-connected-candidate-and-ir-machinery.md#m846)).

**Rationale/current disposition.** Concrete submission/completion contracts,
replay and reexecution expose useful gaps. Replay can accept wrong recorded
values; independent numerical/causal oracles and corruption controls are stronger
semantic evidence. A dispatch-submission bug is fixed before the next machinery
investigation; it does not invalidate the whole architecture.
**G5 consequence/placement.** Keep reference studies and candidate evidence
separate from governing semantics. Do not create competing runtimes or let a
reference library determine participant/resource/optimization ownership.

<a id="h077"></a>

## H077 - The IR report proposes machinery; custom Python was never mandatory

**Source and progression.** The later IR report frames language, preparation and
runtime as a coherent system. xDSL is recommended for investigation
([885](sessions/17-connected-candidate-and-ir-machinery.md#m885)), not yet adopted. The following fresh root explicitly corrects
any implication that a custom baseline must win.

**Current disposition.** The report is research input. Its framework/backend
amendment permits project-owned semantics on reusable IR machinery. The later
xDSL decision supersedes the early implementation baseline without discarding
what the connected candidate established.
**G5 consequence/placement.** Select infrastructure for typed structure,
verification, transformation and lowering; retain project ownership of resource,
authority, lifecycle, numerical and recovery contracts. Keep the report with
research and link its later adoption limits.

<a id="h078"></a>

## H078 - The system represents heterogeneous stateful workflows; training is a demanding use

**Source and correction.** The user corrects “IR-based training system” to an
IR-based system that can represent training ([011](sessions/18-framework-comparison-and-xdsl-choice.md#m011)). Later selection
criteria explicitly reject judging only one named training fixture
([365](sessions/18-framework-comparison-and-xdsl-choice.md#m365)).

**Current disposition.** Stateful heterogeneous data/model workflows govern the
scope. Training's broad existing/researched requirements are retained rather
than treated as incidental or thrown away.
**G5 consequence/placement.** Generality must preserve state, dependencies,
independent lifetimes and effects through shared contracts. Neither a universal
SD-shaped loop nor a framework's ready-made compiler vocabulary defines the
project's language.

<a id="h079"></a>

## H079 - Framework feasibility and final language semantics are different conclusions

**Source and sequence.** Both xDSL and Pliron earn deeper testing. The user
rejects settling on custom Python in advance ([022](sessions/18-framework-comparison-and-xdsl-choice.md#m022)). xDSL's B/A/C,
local fusion, mixed owners and independent extension establish useful structure;
Pliron0.20.0 is explicitly pinned ([191](sessions/18-framework-comparison-and-xdsl-choice.md#m191)) and tests native typed
regions/interfaces, correspondence, extension and resource analysis.

**Current disposition.** Both are credible machinery substrates. Project code
still supplies effects, resources, authority, workflow completion and execution.
Structural Rust ownership does not establish modeled resource authority.
**G5 consequence/placement.** Keep versioned frontend/target contracts and cold
correspondence. Rejected malformed suffixes, progress starvation, omitted outcome
sets and mutation-in-assertion bugs are concrete findings repaired in evidence;
passing counts alone do not select a complete language.

<a id="h080"></a>

## H080 - Matched comparisons preserve semantics and disclose shared runtime and timing boundaries

**Source and rationale.** The comparison aligns B, extension, correlation/
accumulation, delegated authority, mixed feedback and different-lifetime cases.
A valid unfused opaque-effect barrier is distinguished from an invalid contract.
Independent expectations and actual owner facts, rather than private report
schemas, define agreement.

**Current disposition.** Frontends independently verify/prepare but reuse owner
implementations. Agreement is not evidence of a second ownership runtime.
Pliron's analysis manager is a real infrastructure advantage; xDSL's project
cache is not equivalent framework reuse. Subprocess/serialization setup and
steady-state execution are separate costs.
**G5 consequence/placement.** Use the matched evidence for actual machinery
tradeoffs, not fabricated performance or framework-general safety. Retain failed
probes and target coverage gaps. Compare broken correspondence at the protected
boundary: it must not authorize execution, readiness or dependent effects.

<a id="h081"></a>

## H081 - Python authoring is fixed independently of the machinery language

**Source and correction.** The user states that ordinary users normally use
provided behavior and authoring remains Python regardless of substrate
([392](sessions/18-framework-comparison-and-xdsl-choice.md#m392)). Selection must concern the system machinery rather than
an easier developer story ([394](sessions/18-framework-comparison-and-xdsl-choice.md#m394)).

**Current disposition.** Rust machinery does not force Rust authoring, Python
machinery does not define the author-facing language, and Pliron does not imply
Burn adoption. Existing/default compositions are the ordinary user path.
**G5 consequence/placement.** Judge reusable IR structure, analyses, transforms,
inspection/diagnostics and native integration separately from syntax. A new
feature's extension API must not become mandatory end-user boilerplate.

<a id="h082"></a>

## H082 - xDSL is the accepted working substrate, with MLIR-related opportunities and project semantics

**Source and adoption.** Upstream capability review corrects an exaggerated
regions/graphs contrast ([397](sessions/18-framework-comparison-and-xdsl-choice.md#m397)). Multi-level lowering and MLIR ties
are relevant machinery opportunities. The recommendation at
[401](sessions/18-framework-comparison-and-xdsl-choice.md#m401) is accepted by the user at [402](sessions/18-framework-comparison-and-xdsl-choice.md#m402); this is the
working substrate choice, not merely a continuing experiment recommendation.

**Current disposition.** Governing D14 later records xDSL as the working
substrate. Pliron remains a credible investigated alternative; no fundamental
semantic incapability is established by this selection. Neither native-backend
reuse nor MLIR interoperability is claimed fully qualified by the bounded tests.
**G5 consequence/placement.** Build the project's own language and runtime
contracts on xDSL. Preserve the previous architecture and owner evidence;
choosing infrastructure does not select every IR operation, Python API,
production module or complete lowering path.

<a id="h083"></a>

## H083 - The language proposal is input; explicit amendments determine adopted semantics

**Source and adoption.** The user supplies the0.1 proposal
([007](sessions/19-shared-language-and-current-frontier.md#m007)) and authorizes its translation into governing artifacts
([011](sessions/19-shared-language-and-current-frontier.md#m011)). Subsequent amendments establish shared workflow semantics,
an executable reference and distinct verification/preparation/target evidence.
Illustrative names and every speculative proposal feature are not wholesale
production adoption.

**Current disposition/G5 consequence.** D14 and workflow-composition govern the
accepted bounded slice. Freshness, protection, resource lifetime, reproducibility,
stop/cancellation and scoped effects need explicit shared meanings. Keep research
proposals with their adoption pointers; shared interfaces must serve guidance,
verification and lowering rather than create several competing interpretations.

<a id="h084"></a>

## H084 - Independent reference semantics and owner execution establish different evidence

**Source and progression.** Milestones6-9 develop typed language/regions,
queryable contracts, logical lifecycle/reference, semantic checks, preparation,
owner lowering and cold qualification. Logical success/failure, stop/cancellation,
entered physical calls, cleanup and settlement remain distinct facts.

**Current disposition.** Reference execution derives admissions/requests from
IR; owner execution uses actual runtime services. Their independent expectations,
causal relations and observed values matter more than identical private traces.
Valid but unsupported claims are target/analysis limits, not necessarily malformed
language. Open producers need progressing declared stop relationships.
**G5 consequence/placement.** Preserve normalized/cold correspondence before
factories/effects, source identity in diagnostics and owner facts after archive
erasure. Finite model tests do not prove general deadlock freedom, termination,
GPU safety or bounded historical bookkeeping.

<a id="h085"></a>

## H085 - Shared effects and cardinality contracts support real transformation legality

**Source and correction.** External feedback identifies ordered-SSA availability
and binding-effect/capability incompatibility ([123](sessions/19-shared-language-and-current-frontier.md#m123)). The next
milestone rejects those earlier, uses shared binding requirements for preparation
and legality, adds zero-or-one filtering with demand/provenance/disposition,
and implements a guarded local fusion with rejected boundaries.

**Current disposition.** An admitted input that produces no output still has
a settlement obligation. Dropped outputs must not destroy pending demand or
break claims/backpressure. Unknown effects constrain transformation; purity
alone does not authorize collapsing authority, physical completion or failure.
**G5 consequence/placement.** New behavior must enter through reusable region,
value, resource and binding contracts, not central case-name dispatch. Keep
binding requirements, invocation authority, actual outcomes and physical settling
separate while using the same authoritative meaning.

<a id="h086"></a>

## H086 - Retained products make storage/state dependence explicit beyond call return

**Source and decision.** The user requests the bounded retained-use milestone
([164](sessions/19-shared-language-and-current-frontier.md#m164)) after the copy-only convention fails to cover buffer views
and non-leaf autograd products. It feeds G5 rather than completing it.

**Current disposition.** A retained result names dependencies; borrowers and
derived products keep them protected until actual physical completion and
validation. Explicit release and failure/cancellation cleanup respect dependency
order. A binding remains trusted; borrowing does not grant mutation authority.
The tested profile rejects escapes across yield/capture/publication and does
not establish general alias discovery or arbitrary cross-region ownership transfer.
**G5 consequence/placement.** Preserve actual storage/derivative graphs,
observed versions, parent-child retention and release after dependent settlement.
Keep CPU witnesses, symbolic reference arithmetic and cold corruption controls
in their respective evidence scopes. General tensor/GPU/optimizer routing,
production APIs and bookkeeping retirement remain separate work.

<a id="h087"></a>

## H087 - Choosing xDSL does not reset the architecture or make smallness an end in itself

**Source and correction.** After the retained-product work, the user asks for a
whole-language view, then rejects framing the project as a new authoring-design
exercise detached from months of accepted work ([189](sessions/19-shared-language-and-current-frontier.md#m189)). The assistant
acknowledges the error ([191](sessions/19-shared-language-and-current-frontier.md#m191)); the history audit follows.

**Current disposition.** The goal is coherent composable machinery with the
already established breadth, ordinary Python authoring and provided features.
“Small” may describe a manageable set of shared rules, not a cap on operations,
scenarios or capabilities. Users are not assumed to design unfamiliar low-level
interfaces or re-approve every settled architectural premise.
**G5 consequence/placement.** Resume concrete production design from the preserved
decision chain, adopted D14/workflow semantics and experiment limits. Show the
whole system and its interactions; engineering choices must realize those
requirements rather than silently replace them with a conventional fixed loop.

## Coverage boundaries after reading 19

The chronological walkthrough now reaches the final design correction before
this audit began on 5 November 2026. Readings 00–14 cover the original direction,
its inventory/fork history, Q1–Q5, contract and ownership corrections, the first
OpenSpec draft and rebuild, and training-first placement. Readings 15–19 cover
the broader research, whole-run language direction, G3/G4 exchanges and audits,
connected execution candidate, matched xDSL/Pliron machinery, working xDSL
selection and shared-language milestones through scoped retained products.

The [carry-forward map](current-decisions.md) connects the resulting current
dispositions to these entries and the governing sources. The
[placement manifest](placement-manifest.json) classifies all 68 inventoried
Markdown documents and 28 original root items. The
[supplementary reading](supplementary-sources.md) closes the earlier-notes and
local-archive leads with exact-byte copies or Git correspondences.

Visible selected messages and inspected fork differences are preserved without
duplicate inherited identities. Original bodies, source windows, hashes,
compaction markers and page-prefix relationships remain inspectable. Earlier
receipts/captures are retained; actively appended raw sessions use a fixed
historical-prefix hash, ending before this audit's own messages.

This is reconstruction of the decision chain, not exhaustive recovery of every
registered/unregistered session or incidental tooling/backend thread. Historical
code inventories, implementation claims, tests, benchmarks and reviews were not
independently rerun. Source roles and adoption/correction points were examined;
an old result is still evidence only within its original scope.

The walkthrough itself moved or edited no source document and changed no
governing requirement or task. The subsequent authorized archival moved the
historical originals byte-exactly and updated live document references; captured
provenance remains unchanged. Main G5 design/implementation work was not resumed. Missing
production shapes are not missing architectural intent. The user's H050
maintenance concern is retained for evaluation against a concrete system,
without turning it into a new feature-whitelist requirement.
