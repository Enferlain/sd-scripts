# Notes (user and agent)

This file is chronological history. Every dated section records the state at
that time and may be superseded by a later section. Use the final dated section
for the current handoff, then consult the normative direction and, once created,
the governing OpenSpec design and specs for current authority.

## Current continuation checkpoint (2026-07-30)

The production inventory, framework comparison, and code pressure test are
complete enough. Do not restart by proposing a graph, renaming strategy, or
repeating the Lightning/Fabric comparison.

Current direction:

```text
TRAINER
  defines the training contract and executes the training mechanism
        ↑
STRATEGY
  explicitly defines what is being trained and how
        ↑
MODELS / FEATURES / CAPABILITIES
  are deliberately used to author that strategy
```

Strategy does not mean model family. SD, SDXL, and SD3 strategies are the
maintained default definitions we author for training those standard models.
Model-family behavior is one ingredient, just like objective behavior,
representation features, training-subject treatment, and persistence support.

Do not create adapter-specific or fine-tune-specific strategy classes. That
would regress past the current mode-separated architecture to an older
combinatorial strategy design. Fine-tuning, adapter training, combined
base/adapter training, distillation, or newly attached trainables may instead
be expressed as capabilities/features deliberately selected within a
strategy.

The contract system still has three consumer-defined surfaces:

- core behavior required by the intended Trainer;
- named capabilities coordinated by Trainer or delegated pipeline systems;
- features consumed internally by strategy authors.

Capability is not synonymous with optional. The core may require a compatible
capability from a category, and a selected capability becomes a real
obligation. A capability may provide behavior, typed information for
Trainer-owned mechanics, or both.

Configuration may request use/settings of an already-authored capability. It
does not infer or assemble the strategy. Core/capability/feature classification
is independent from authored/validated/bound/prepared lifecycle state.

`TrainingMode` is not a target top-level runtime authority. Its current
responsibilities must be separated rather than moved as one block:

```text
strategy declaration
  selected training intent and subjects

Trainer / optimization mechanic
  trainable and parameter-group realization, accelerator preparation,
  backward, stepping, and temporal lifecycle

capability/domain implementation
  specialized attachment, extraction, persistence, or other behavior

strategy-internal feature
  reusable behavior used to implement the selected strategy
```

The user currently leans toward mode behavior becoming Trainer concern even
when represented as capabilities, but implementation placement is not yet
decided. It depends on the code shape. Trainer should own the mechanism without
necessarily accumulating technique-specific branches in one class.

Code pressure-test findings:

- `LoadedModelComponent`, `OptimizationPlan`, `OptimizerBuildResult`, and
  `ObjectiveRuntime` are useful typed foundations to evolve.
- the current `LoadedModelComponent.module` conflates logical/original and
  prepared execution bindings;
- `prepare_with_accelerator(trainer)` proves the need for a typed preparation
  plan/result because modes currently mutate modules, optimizer, scheduler,
  gradient-sync handle, and primary trainable through the whole Trainer;
- modes already construct logical/execution optimizer groups, while Trainer
  owns ordinary backward/step lifecycle;
- `BatchLossOutput.loss` is not the tensor used for backward:
  `per_sample_loss` passes through the Trainer-owned loss modifier first;
- diffusion `timesteps` are objective observations, not a universal training
  contract field;
- current `TrainingMode` and `ObjectiveRuntime` interactions must be
  redistributed across the one strategy/Trainer contract boundary.

Next discussion starts with binding and runtime preparation together. Define:

```text
authoritative bound state
stable logical component identity
logical/original binding
prepared execution binding
preparation participants and constraints
prepared-binding result and rebinding guarantees
```

The mode/objective classification is now recorded in
`strategy_system_inventory.md` Milestone 6. It concludes:

- `TrainingMode` dissolves rather than being renamed;
- strategy declares training subjects, objective semantics, and selected
  capabilities;
- Trainer/optimization owns generic trainable realization, precision,
  distributed preparation, optimizer creation, clipping participants, and
  ordinary lifecycle transitions;
- specialized attachment, lifecycle, and persistence behavior remains
  explicit capability/domain behavior coordinated by Trainer;
- objective mathematics remains a strategy-internal feature, while the
  current objective-owned loss modifier is better treated as an optimization
  capability;
- diagnostics should derive from authoritative binding/optimization facts,
  not reconstruct topology through mode.

Two blockers remain before choosing Python APIs:

1. the exact authoritative binding semantics and storage shape;
2. the standard Trainer-owned optimization profile and explicit research
   extension/ownership profile.

After binding/preparation, derive optimization and step exchanges. Only then
choose method names, class/package structure, and create the OpenSpec.

## Binding discussion resumed (2026-08-20)

Questions 1 and 2 now have bounded semantic answers:

- a logical component identity is stable, strategy-scoped, and denotes one
  semantically distinct participant rather than one Python object;
- the standard contract has one normal prepared execution binding per logical
  component;
- named extra routes are capability extensions justified only by materially
  different runtime preparation/callable requirements;
- every logical-component/route pair has exactly one authoritative current
  binding;
- shared or explicitly synchronized derived state may remain one component,
  while independently evolving state requires another component identity;
- original/unwrapped/artifact handles are typed access views, not competing
  execution routes; and
- stale routes cannot remain silently authoritative after their freshness
  guarantee fails.

Still open: concrete identity and route types, refresh mechanics, binding-state
storage ownership, and the full preparation plan/result API. The next question
is how additions and replacements are represented during adapter attachment or
deferred loading.

## Question 3 settled arrangement semantics (2026-08-20)

The initial addition/replacement take was pressure-tested against the current
PEFT path and broader adapter shapes. The semantic question is now settled;
concrete representation, storage ownership, invalidation, and API design remain
downstream.

Current LoRA proves that adding adapter-owned state and attaching its effect are
different operations: the adapter remains separately owned while selected host
`forward` methods are modified in place. VeRA further shows that one adapter
identity may own shared state plus many target-local modules. Prefix-style
state may have no independent forward route, while ControlNet/T2I-Adapter-style
side networks are independently executable participants.

Settled operation taxonomy:

```text
declare a participant
materialize authoritative bound state
replace authoritative bound state
rebind an execution route
transition an operational relationship
amend the arrangement by adding or retiring participants
merge/fold state and record the resulting lineage
```

Materialization and replacement remain distinct. Replacement supersedes a
current binding and carries retirement/supersession and invalidation
consequences. Merge/fold transforms host state using adaptation state, may
retire the adapter from the current arrangement, and records transition and
artifact provenance; it is not inherently destructive to the separately usable
adapter.

Ordinary adapters should normally be declared by the authored strategy before
structural validation. Target resolution and attachment then fulfill that
declaration; they do not make the adapter appear as an arbitrary late list
mutation. Dynamic participant addition remains an explicit extension whose
result must name relationship/capability changes and invalidate affected
prepared routes, optimization plans, and caches.

Q2 is qualified so a logical component represents one independently
addressable semantic participant for which authoritative bound state is
maintained, whether or not it is independently executable. Only
execution-capable logical components require a normal prepared execution
route. State-bearing adapter components may instead expose authoritative state,
optimization, and artifact bindings.

Logical adapter identity follows an independently addressable adaptation
participant, not Python-container boundaries or every injected submodule.
Participant identity is separate from relationship/effect identity, and an
outermost wrapper does not acquire the host's semantic execution ownership.

Keep these axes separate:

```text
participant lifecycle: declared, bound, retired
optimization status: selected/unselected, trainable/frozen
operational relationship lifecycle: declared, resolved, active, inactive, detached
```

Keep these persistence views separate:

```text
current arrangement
run transition history
durable artifact provenance
```

Settled invariants: materialization is not addition; attachment is not
component creation; execution preparation is not arrangement mutation;
componenthood does not imply executability; and historical provenance is not
current bound state. Next discussion question: Q4.

## Question 4 settled persistence semantics (2026-08-21)

The persistence pressure test covered current SDXL and SD3 full-model saves,
LoRA/VeRA adapter exports, EDM2 sidecars, and Accelerator resume snapshots.
They do not serialize one universal runtime object: each product selects
different participant-owned, relationship-owned, derived, or referenced state.

Settled core statement:

> Artifact persistence produces a declared artifact product from a coherent
> semantic projection of authoritative current state. The product identifies
> participant and relationship coverage, dependencies, semantic
> transformations, external representation, packaging, and consistency
> requirements. Trainer/runtime infrastructure establishes the persistence
> boundary; domain serializers render the resolved state into semantic members
> backed by physical resources. Persistence never selects state through
> incidental Python object identity.

Use four stages:

```text
capability declaration
  -> persistence request
  -> resolved artifact plan
  -> artifact result
```

The plan describes intended semantic coverage and expected members. The result
describes what was actually emitted, including resources, formats, sizes,
checksums, references, and partial/failure status. Pre-write metadata may derive
from the plan; post-write facts must derive from the result.

Keep three levels separate:

```text
artifact product: one semantic result
artifact member: one meaningful constituent
physical resource: file, directory, shard, blob, or remote object
```

Bundle is a packaging/cardinality property, not a completeness claim. Keep
semantic coverage, dependency semantics, transformations, representation,
packaging, and consistency as orthogonal dimensions. A coherent product belongs
to one accepted arrangement and one Trainer-established boundary while honoring
the declared freshness relationship of each contributor; this need not require
one universal revision number.

Semantic transformations that affect coverage, dependencies, lineage, or
realization identity are declared by the product/capability. Domain serializers
own mechanical conversion and physical writing. Runtime resume snapshots remain
a separate restoration contract even when they share Trainer timing,
stable-state infrastructure, and storage services with artifact persistence.

Repository correction: the filename-based EDM2 branch in `FineTuneMode` is
residual. Active step, epoch, and final flows call the loss modifier's sidecar
save directly. This supports capability-owned artifact state, while also showing
that the current persistence exchange lacks a typed product plan/result.

Q4 is settled semantically. Concrete APIs, member metadata policy, failure and
partial-result representation, and asynchronous publication remain downstream.
Next discussion question: Q5, ownership of the canonical authoritative binding
map and the narrow views exposed from it.

## Question 5 target-first scenario checkpoint (2026-08-21)

Q5 scenario modeling is recorded in `strategy_system_inventory.md`, Milestone
10. The current code is a pressure oracle, not the target object model: no
production changes have yet implemented the direction under discussion.

The working term **binding authority** means the one contract-governed authority
for accepted current participant, relationship, access-view, and execution-route
bindings. It intentionally does not yet select a class name, package, physical
container, or whether the authority is embedded in or paired with strategy
behavior.

Scenarios modeled:

```text
ordinary multi-component fine-tuning
declared but deferred SD3 participant
authored adapter materialization and attachment
distributed preparation plus artifact persistence
replacement with dependency-aware invalidation
compound teacher/student/adapter research strategy
```

Important pressure findings:

- stable keyed identity in `LoadedModelComponent` is useful evidence, but
  Trainer ownership and family-shaped compatibility setters are not the target;
- materializing a deferred participant advances binding state without amending
  the declared arrangement;
- in-place adapter injection can change authoritative execution semantics while
  Python object identity remains unchanged;
- distributed execution bindings, original/unwrapped access, artifact state
  views, and backend synchronization handles are distinct concerns;
- derived preparation, optimization, cache, route, and artifact projections
  require declared identity/revision dependencies so replacement can invalidate
  them precisely; and
- any core owner assuming one denoiser, one primary trainable, an autoencoder,
  or a text-encoder list fails the compound and pixel-space pressures.

Provisional semantic boundary:

```text
authored strategy/features
  declare semantics and propose typed transitions

contract-governed binding authority
  validates and atomically accepts canonical current state

Trainer/runtime infrastructure
  consumes generic projections and returns preparation/optimization results

metadata/persistence
  consume accepted snapshots and product projections
```

This rules out Trainer family fields, metadata storage, family-mixin attributes,
and unrestricted shared dictionaries as the canonical owner. The leading scope
is the **bound strategy contract**, but the physical layout remains open among:
authority contained by strategy behavior, a small paired Trainer-facing contract
object, or a separate contract-owned authority established during filing.
Comparing those layouts is the remaining Q5 decision; Q5 is not yet marked
settled.

## Question 5 settled after ownership review (2026-08-21)

The six target-first scenarios established the required semantics but could
not choose the physical layout because each candidate could be made to pass
them by construction. Two outside read-only pressure reviews confirmed Q1-Q4,
identified this limitation, and suggested deciding Q5 through ownership
criteria rather than accumulating more normal-path scenarios.

Settled result:

```text
public boundary
  Trainer <-> complete TrainingStrategy

internal contract responsibility
  dedicated per-run binding authority
    canonical participant/relationship/binding/route state
    revisions, dependencies, freshness, and atomic transitions
```

“Separate authority” means a separate responsibility and likely a separately
testable implementation object. It does **not** mean Trainer receives strategy
and binding authority as two peer integrations. The strategy remains the one
complete Trainer-facing object. The authority is governed inside that boundary
so state does not fall back into arbitrary strategy-facet fields.

The deciding criteria were one-writer enforcement, independent testing,
state/behavior separation, one-per-run lifetime, scoped access, and migration
from `LoadedModelComponent`. Concrete construction and exchange APIs remain
open: who creates the authority, how filing seeds it, how typed transition
results are accepted/rejected atomically, and which projections each consumer
may access.

Useful follow-up conformance pressures include failure/rollback, rank
consistency, multi-adapter overlap, compiled routes, EMA, resume, and mid-run
trainability/parameter surgery. These test the settled semantics; they are not
new Q5 ownership blockers.

The initial reaction rejected the recommendation for conservative initial
behavior too categorically. Re-reading the original reviews clarified
the important distinction: migration may be sliced and implementation policy
may be conservative, but the introduced contract must remain compatible with
the settled architecture. Explicitly invalidating every derived projection may be a correct initial implementation of the freshness rules, but it is deliberately imprecise; revisions and freshness remain the authoritative contract semantics.
Current strategies may initially file only the normal route, and current
artifact products may be migrated first, while the types still preserve
support for named routes and declaration/request/plan/result plus
product/member/resource meaning. What remains rejected is a temporary contract
that discards those meanings or leaves stale-state correctness to caller
convention.

The binding exchange must additionally settle when participant identities are
assigned, when replacement preserves one participant versus requiring
retirement and a new declaration, and what default freshness semantics apply
when a route, access view, or derived projection declares no stronger
guarantee.

Binding questions Q1-Q5 are now settled semantically. The next work is concrete
binding/preparation exchanges, followed by the still-separate optimization
ownership block: the standard Trainer-owned optimization profile, explicit
research/extension ownership, optimization exchange, and then step exchange.

## Concrete exchange design record started (2026-08-22)

Concrete downstream work now has an additive, compaction-safe home in
`strategy_contract_exchange_design.md`. The direction document remains
normative for the overall architecture and settled Q1-Q5 semantics; the new
record distinguishes inherited settled inputs, working downstream decisions,
and open exchange questions rather than rewriting the accumulated direction,
inventory, or chronological discussion.

The first working entries preserve the authored-strategy boundary while giving
run-time identity a concrete purpose: an authored semantic participant address
is established as one authority-scoped declaration incarnation; revisions then
describe changes to its binding, relationships, and routes. Retirement closes
that incarnation to current operations without erasing history/provenance
meaning. Replacement preserves identity only while the authored participant
meaning remains fulfilled. A whole-authority snapshot is the conservative
default freshness dependency, while participant/relationship/route revisions
remain the finer authoritative facts.

The runtime-preparation draft also records a semantic failure invariant rather
than prematurely choosing a lease or `preparing` state: an operation permitted
to mutate an authoritative realization in place must withdraw affected
freshness guarantees before mutation, and failure cannot silently restore them.
Prepared execution routes, Trainer-owned optimization runtime, and backend
coordination handles remain separate outputs even when one backend call
produces all three.

## External review refinements to the first exchange draft (2026-08-22)

Two outside read-only reactions agreed with the key/reference split,
replacement rule, readiness separation, conservative freshness policy, and
runtime-preparation ownership split. They also exposed several useful concrete
refinements, which were reconciled against the settled direction rather than
accepted automatically.

Execution-route requirements are now separate declarations associated with a
participant instead of fields inside `ParticipantDeclaration`. This keeps the
participant declaration limited to its authored address and semantic
compatibility filing, lets zero routes express a state-only participant, and
allows selected capabilities to add materially different route requirements
without redefining participant identity. Materialization moves an unbound
participant to bound; `deferred` describes why remaining unbound is currently
allowed under a readiness constraint, not another binding state.

Relationship identity is now an explicit working decision rather than an
implicit `RelationshipRef` vocabulary choice. Independently evolving
operational state, multiple effects from one adapter, parallel effects over the
same endpoints, history, and persistence projections justify authored
relationship addresses being established as run-scoped relationship
references. Their exact Python representation need not copy participant
references.

The coarse freshness fallback now advances for any accepted transition that
changes snapshot-visible authority state, including relationship and route
changes, rather than only participant-binding changes. The external review
also identified that pre-mutation invalidation changes authority state after a
preparation plan was resolved. An in-place-capable attempt therefore needs a
coordination identity distinct from its source snapshot so it can retain the
accepted basis and pre-mutation withdrawal transition through success or
failure. The exact lease/attempt/staged-transition mechanism remains open.

Finally, optimizer/scheduler entries in preparation remain evidence that one
backend may need joint preparation, not an early decision about optimization
construction or request/result types. Those stay subordinate to the separate
optimization-ownership design.

## Candidate compatibility validation boundary drafted (2026-08-23)

The apparently open semantic-compatibility topic was re-read against the
exchange record. Much of its conceptual boundary was already recorded:
authored/versioned requirements plus explicitly wired validation behavior,
strategy-authored composition rather than discovery, typed acceptance, and no
universal role/type taxonomy. The remaining problem was narrowed from
inventing that boundary to making its candidate-validation exchange concrete.

Task-directed code inspection reinforced the distinction. The current
`LoadedModelComponent` associates a family declaration with `module: Any`;
`ModelLoadingStrategy.load_target_model()` returns those live objects in a
tuple and represents deferred state with `None`. Those are useful producer
surfaces but provide no semantic acceptance evidence or atomic authority
transition. `RealizedModelComponentFacts` is durable observation/provenance,
while `OptimizationTargetRef` and `AdapterResolvedTarget` are bounded consumer
projections. None should be promoted into the new compatibility authority.
The current graph generation was `2026-08-22T20:50:39Z`; all six inspected
source paths had matching metadata and no recorded coverage issue, subject to
the normal best-effort caveat.

The working exchange now separates:

```text
explicit authored compatibility filing
  filing identity/version + requirements + composed validation behavior

candidate binding proposal
  participant/transition/proposal identity + expected revisions
  live candidate + filing-specific typed evidence

compatibility assessment
  exact filing/proposal/candidate/dependencies assessed
  named clause outcomes + fulfilled meanings/limitations + accept/reject

authority transition result
  accepted binding/revision/invalidation state, or structured rejection
```

The common proposal envelope coordinates the authority; candidate evidence
remains typed by the explicitly wired filing rather than forced into a
universal participant-kind union or untyped fact dictionary. The authority,
not the proposer, selects the already filed validation behavior. An accepting
assessment is single-use and bound to the exact proposal, participant,
candidate, filing version, transition kind, and observed revisions. It is not
the transition result: only the authority installs state, advances revisions,
and invalidates dependents.

Reusable core/feature/family clauses and custom research clauses are composed
deliberately during strategy authoring. The authority executes that accepted
composition and does not discover validators from roles or object types.
Deferred state supplies no fake `None` candidate; replacement supplies the
prior accepted binding and revision so continuity can be checked where the
filing requires it.

This is recorded as working decisions EX-013 through EX-017. Exact Python
generics/protocols, authority construction, the complete atomic transition
envelope, and in-run filing-version evolution remain open. On resume, first
confirm/refine the expanded decision register, then design authority
construction and initial arrangement establishment.

## User

- versioning for contracts is probably a good idea for the long run. Contracts and trainer might need to evolve over time as more models get added, but the ideal scenario is them not having to, especially the trainer, but this is only possible to accomplish via the repo growing with new capabilites and testing said contract and trainer.

### excerpts

I'd probably make it something like

folder called tools or capabilitiesor features or shared or grab box/
clip.py (a bit dubious since clip is understood as a component but it's not defined and used from a venv, might move to repo hosted definitions, we'll need to weigh the pros and cons)
diffusion/
pixel_diffusion.py
latent_diffusion.py

And such. Thoughts?

Technically, in the ultimate understanding of this system you would also grab box model components so strategies can be built aribtrarily from them, but that's like end game stuff (1 autoencoder vs the other, different text encoders, llms as vision, whatever u desire)

---

regardless of what we do the grab box thing will probaly be the future, at least that's how I feel like right now. When systems become too difficult to reason with, the easuiest method is to just organize them better. the entire strategy system is one attempted organization of the original sd-scripts repo, which was a shithole from an architecture standpoint.

---

what about using decorators instead of names like somethingsomethingfeature or somethingsomethingcomponent or whatever

@feature
class LatentDiffusion

feels like ppl often forget they exist while they sound useful, but maybe not here, I'm not sure

---

my current stance as we discussed before is that I like the contract idea centralizing strategy (your training "plan") and handing it to the active trainer layer so it doesn't need to track or know about individual parts, but that's not how it happens today, and the contracts despite being made "strict" are not really respected in the way we imagined it when we came up with the system. and the part about being able to grab features and eventually components for your strategy is pretty attractive as a goal. Then there's also decisions (vae vs autoencoder lanugage, specific component, names, handling, separation, etc basically same with "clip") about the models folder

I didn't read through the inventory fully yet, this is just my thoughts. we should work with our design docs (direction and now inventory) in general unless we decide on other details and actions. what should we do next?

---

1 let's try to make the contracts like this:

a basic idea built on what we have today, intuition about what we might have in the future extrapolated from the direction we want to move to, what "training a neural network" means, and what is needed to be able to train something in our repo today (this can be very narrow technically so let's start from where we are at)

---

the "run" doesn't select things, these are defined in the strategy build the same as today, just under the new contract. it might constrain or validate it (where this happens is not decided from what I've read so far). the contract defines what must be fulfilled and what is available to use whether it's a slot that needs to be filled or something that you're picking for yourself. we're not going for a super smart mind reader system that automatically picks things based on implication, at least not for now. strategies should be built with intent. we can validate that the choices are correct, but that's different from a feature dragging another with it, that should still be left to whoever is putting the strategy together. (and this is us, we make the default training strategies for models aka what you see currently)

This process is manual, it's not done by a hydra replacement or a config system, at least not initially anyways. It's the same as how model code is still written by hand. strategy is essentially "the model code" in this repo. there might be a config equivalent that can do the task of putting together a strategy in the future to make things more streamlined, but that's not current goal

---

yes. there is essentially no noob mode on strategy besides "trainer needs something to train chief)" and even that can be overridden if someone wants it, or allowed for mutated/adopted/custom strategies besides the base model training strategies as that would go against the spirit of experimentation and research

if you know this repo it takes a similar approach at a base level, although it's not for training [ljleb/sd-mecha](https://github.com/ljleb/sd-mecha)

---

basically we didn't change the overall goal, because we are still the ones that put the strategies together, we just establish the mechanics around the contract (which covers the constraints as well) and the intended way to interact with the strategy system

if any updates should be made to [strategy_system_direction.md](docs_design/models-strategy-trainer/strategy_system_direction.md) now is a good time before context compacts

---

well this should be dictated by the trainer no? which also dictates the contract. idk if this means strategy passes a bundled item that the trainer then needs to unpack, or how this is handled in python/other repos normally

---

we might not necessarily want to be constrained by an existing system unless it allows for modification based on our own designs, that would be the deciding factor about whether or not we adopt something. and we also keep the code in our repo, which means we become the maintainers

---

## Contract authority corrected in the exchange design (2026-08-23)

Direct review of the normative direction, chronological excerpts, inventory,
and current runtime code showed that the candidate participant-level
"semantic compatibility filing" had promoted an external implementation
suggestion into a competing source of contract authority. The direction already
states the governing model: the training contract system defines the Trainer's
accepted core, pipeline capabilities, feature contracts, rules, lifecycle, and
results; it informs and constrains explicit strategy authoring; and only a
validated strategy reaches Trainer.

The user clarified that ordinary strategy authors select intent, components,
features, capabilities, relationships, and deliberately exposed choices, but
must not separately communicate how those known selections are valid. The
contract-enforcement mechanism already knows what the Trainer accepts and how
known library implementations conform. Automatic enforcement is required;
automatic strategy assembly remains rejected. Additional author-supplied
conformance or validity behavior begins only at an explicit custom
implementation, direct-conformance path, or contract/Trainer extension.

`strategy_contract_exchange_design.md` was corrected accordingly:

- participant declarations retain authored identity/addressing meaning while
  semantic obligations are derived from their complete use in the strategy;
- initial binding establishment now begins with contract validation of the
  complete authored definition and produces the accepted `TrainingStrategy`
  plus its internal authority and derived obligations before Trainer receives
  it;
- materialization, replacement, and preparation provide contract-defined
  evidence and are enforced against prospective complete state by that same
  contract-governed authority;
- a separately transferable `CompatibilityAssessment` is no longer assumed;
- EX-007, EX-015, and EX-017 are superseded; EX-008, EX-009, EX-013, EX-014,
  and EX-016 are corrected; and EX-018/EX-019 record the establishment boundary
  and the distinction between automatic enforcement and automatic assembly.

The settled Q1-Q5 binding, revision, freshness, route, relationship,
preparation, and persistence semantics remain unchanged. The normative
direction and evidence inventory did not require conceptual edits. The next
detail question is the concrete authored-strategy input and accepted
establishment result through which the contract system derives obligations and
creates the internal binding authority without adding a second Trainer-facing
strategy wrapper.

## Code-derivation dependency path corrected (2026-08-24)

The user emphasized again that all current production code predates the target
direction and asked for the best dependency path for figuring out the eventual
code. Direct inspection of the launcher, configuration validation, strategy
factory and aggregate ABC, family strategy constructors, Trainer, phases,
mode implementations, and objective boundary confirmed that current code must
remain evidence rather than an upstream design authority.

The active launcher currently validates configuration before a strategy
exists, constructs a family-selected strategy and a separate mode, and lets
Trainer construct a separate objective. Trainer then becomes the mutable owner
of strategy-produced components and passes its projections back into strategy
and mode calls. The aggregate `TrainingStrategy` ABC also mixes operations
that the future design classifies as core exchanges, recognized capabilities,
strategy-internal features, and conditional family behavior. Designing the new
authoring API or validator directly from those classes would preserve the
fragmentation the direction intends to remove.

The exchange record now distinguishes three orders:

```text
runtime direction
  authored definition -> establishment -> accepted strategy -> Trainer

design-dependency direction
  intended Trainer -> exchanges/results -> accepted strategy state
    -> establishment/conformance -> authored input

current-code migration evidence
  launcher/factories -> strategy+mode+objective -> mutable Trainer/phases
    -> typed islands and migration mappings
```

The recommended derivation sequence is therefore: define the intended Trainer
skeleton; build a semantic Trainer-consumption table; derive the accepted
`TrainingStrategy` boundary; complete binding, preparation, optimization, and
step dependencies in that order; derive establishment and known/custom
conformance from the accepted result; design the authored-definition API; and
only then choose final Python placement and migration milestones. This does
not require copying today's Trainer or finalizing all exchanges at once.
Binding and preparation remain first because later exchanges depend on their
accepted current state.

The next bounded discussion artifact is the semantic Trainer-consumption
table: for each core exchange and recognized capability, record request,
result, lifecycle point, required accepted-state guarantees, and side-effect
owner without treating today's method signatures as the answer.

That first frame was added to the exchange record in the same pass. It treats
admission as a pre-Trainer contract boundary; arrangement/binding,
preparation, optimization, and step as the four core surfaces; and caching,
validation, sampling, trained-artifact persistence, and runtime restoration as
recognized pipeline capabilities. Logging, metadata, resource observation,
interruption, and cleanup remain cross-cutting Trainer/pipeline ownership that
consume typed results without owning live strategy state. The frame explicitly
does not promote today's tokenizer, VAE/text-encoder/denoiser,
`trainable_model`, `TrainingMode`, or `ObjectiveRuntime` surfaces into the
future contract. The first row to refine is arrangement materialization and
binding, with preparation kept beside it because both operate over the same
accepted identities.

## Open-question audit separated design from code shape (2026-08-25)

The first eight open-register entries were reviewed against the direction,
exchange decisions, chronological notes, inventory, and relevant metadata
identity precedent. The audit used one rule: an architecture question remains
open only when different answers would materially change behavior, ownership,
lifecycle, or system boundaries. Exact classes, fields, constructors,
registrations, generics, envelopes, and module placement belong to the later
current-to-target code mapping.

Results so far:

- former question 1 has settled establishment/one-strategy semantics; only the
  Python construction API remains;
- former question 2 is deferred implementation design for exposing known
  built-in/custom conformance after the required information is known;
- former question 3 has settled obligation/evidence authority and timing
  principles; concrete types are exchange-local code design;
- former question 4 is resolved and duplicated the construction boundary;
- question 5 remains as a real design question about durable identity versus
  lineage across history, metadata, persistence, and resume;
- former question 6 has settled non-revival/new-incarnation behavior; initial
  key-reuse support is implementation policy;
- former question 7 is concrete bound-state/access-view representation; and
- former question 8 is concrete transition-envelope representation after
  atomic publication behavior was already settled.

Deferred Python and migration decisions now live in
`strategy_system_implementation_mapping.md`, which will later map the current
inventory into target code and OpenSpec milestones. The architecture register
retains question 5 and the not-yet-audited questions 9–14. The former full
remaining-work list was replaced with a three-step current design frontier,
and the compaction handoff was reduced to document roles and the current
frontier.

## Remaining open-question audit (2026-08-25)

Former questions 9–14 were checked against the settled direction, scenario
requirements, and the binding/preparation exchange itself. The broad scoped
read question was narrowed to the preparation projection's exact live access
meanings and coherent multi-participant source state. Consumer boundaries are
otherwise settled: Trainer infrastructure and capabilities receive only their
contract-defined projections, observability receives facts/results rather than
arbitrary live objects, and persistence receives a coherent product-specific
artifact projection.

The in-place preparation semantics are settled; lease/token/API shape is code
design. Distributed ranks participate in one logical accepted transition;
collective mechanics are runtime implementation. Optimization objects remain
Trainer-owned even when jointly prepared, while their exact exchange stays in
the later optimization design. One accepted contract/version now governs one
run authority: permitted arrangement amendments re-evaluate obligations under
that contract, whereas another contract/version or Trainer-contract extension
requires new establishment.

The only remaining preparation-wide architecture question is the commit and
recovery boundary between authority-owned route publication and Trainer-owned
runtime/backend publication. The active register therefore contains only
durable identity versus lineage, preparation read isolation, and prepared-state
commit/recovery. Earlier exact identity, obligation/evidence type, and
transition-envelope sections are marked as deferred code design rather than
open architecture.

## Preparation reads and publication settled (2026-08-25)

Follow-up discussion and external review corrected “single-use preparation
package” into one **attempt-scoped preparation job** derived atomically from
coherent accepted authority state. The job may span several ordered or joint
backend calls and need not be one literal disposable object. Its results remain
bound to the attempt and its source dependencies. Preparation membership is
broader than trainability, and backend composites remain coordination state
rather than replacements for participant identity.

Prepared-state publication was then reduced before introducing a general
transaction protocol. All ordinarily fallible backend work, component-specific
preparation, candidate assembly, rank agreement, and contract/freshness
validation precede final publication. Final publication is an unobservable
logical installation of already-created authority-owned route/view state and
Trainer-owned runtime/backend state; it invokes no expected-fallible external
work, conversion, validation, or callback. Post-publication observation cannot
roll back accepted current state, while process/rank termination aborts the run
and belongs to restoration.

Replacement-only failure therefore discards an unpublished candidate while
preserving still-valid old routes. Destructive in-place preparation remains the
intentional exception: its attempt first publishes a preparing/invalid state and
withdraws affected guarantees, which stay withdrawn after failure until state
is explicitly re-established or replaced. Successful completion uses the same
non-failing final installation rule. Joint optimizer preparation remains opaque
Trainer-owned candidate pressure and does not settle the later optimization
contract. Former questions 9 and 11 are removed from the active register;
durable identity versus lineage is now its only item.

## Durable participant identity and lineage settled (2026-08-25)

The remaining binding/preparation architecture question was checked against the
current model-component, metadata, checkpointing, resume, Trainer, and training
mode code. Today, loaded component keys survive from declarations into live
components, metadata qualifies model realizations and components by a random
Trainer session identifier, and resume restores backend/model state plus
epoch/step. It does not restore a logical run authority, participant
incarnations, or their accepted revisions. A resumed process currently creates
a new Trainer session identifier. This is current-code evidence, not the target
behavior.

The target decision is that, within one logical run authority, each
authority-established participant reference permanently identifies exactly one
participant incarnation. The reference is never reassigned or revived. Its
durable projection combines that authority/run identity, its authored
participant key, and an incarnation discriminator. Materialization,
preparation wrapping/casting, identity-preserving replacement, and route
rebinding preserve the reference; their changing binding, relationship, and
route revisions do not become identity. Retirement closes current use but
preserves the same historical identity for history and provenance. A different
authority/reference is a different incarnation; redeclaring a key after
retirement, if permitted, receives a new reference. Teacher and student remain
distinct even when loaded from one source, and incompatible replacement
establishes a new incarnation. Typed lineage or succession records any
meaningful descent separately.

Artifacts likewise have artifact/model-revision identity and capture accepted
participant/run state; they are not participants. Exact runtime resume may
preserve logical run authority, participant references, accepted arrangement,
and relevant revisions across new Python objects and processes, while using a
separate per-execution session identity for observations. It can do so only if
the snapshot actually persists and restores those facts. Otherwise loading the
snapshot begins new authority-scoped identities connected through explicit
resume/derivation lineage rather than pretending the old incarnation survived.

The work order explains the current integration gap more precisely: substantial
supporting and metadata work came first. When the work then tried to proceed
properly into model metadata, it exposed problems in the repository's existing
model, strategy, Trainer, and checkpointing boundaries. That discovery caused
the pivot into the present rework direction before richer model metadata could
be completed on top of those boundaries. Model metadata was therefore not a
mature architectural layer that the Trainer later failed to adopt; trying to
design and implement it was the pressure test that revealed the architecture
underneath was not yet what was wanted.

Its need for intentional loaded-component boundaries, durable model identity,
component provenance, run realizations, artifacts, and lineage remains a direct
input to this redesign. Current checkpointing's failure to restore a logical
authority and participant incarnations is one of the capabilities the new
architecture must make possible, not evidence that metadata is merely an
external observer with incidental concerns.

Those metadata distinctions therefore directly inform the target requirements.
They still do not require the current
`run/<session>/model/<realization>/component/<key>` string shape to become the
runtime identity mechanism. Participant identity belongs to the binding
authority, which must expose an intentional durable projection to metadata,
persistence, and exact-resume restoration. Metadata records and preserves that
projection rather than reconstructing identity from incidental live objects.
Exact value types, identifier strings, relationship names, and persistence
representations remain code-design decisions rather than reasons to keep the
architecture question open.

## Executable binding/preparation pressure test (2026-08-25)

The completed binding, preparation, identity, lineage, artifact, and resume
semantics were implemented as an isolated executable spike in
`docs_design/models-strategy-trainer/executable_spike/`. It does not import or
modify production code. The purpose was not to select final classes but to see
whether the intended exchanges could form readable Python and whether failure
interleavings exposed missing architecture.

The final spike runs 28 scenarios covering ordinary multi-component training,
deferred materialization, adapter/host relationships, teacher/student/adapter
arrangements, contract-derived conformance, compatible and incompatible
replacement, retirement and succession, replacement-only and destructive
preparation failure, stale results, cross-authority rejection, joint and
disjoint backend groups, artifacts, lineage, exact resume, and new-run loading
from artifacts. Ruff and ty both pass for the two spike files. Independent
review found no remaining correctness issue after several review-driven
iterations.

The implementation exercise confirmed the existing direction and revealed
four details that now belong to the exchange design:

- usable prepared state is a joint view of authority-owned participant/route
  guarantees and Trainer-owned backend-group state; a backend composite is not
  independently current and never replaces participant identity;
- inseparable backend groups constrain later preparation before destructive
  work begins: complete groups may be replaced, expanded, or merged, but an
  overlapping job cannot omit an existing member; retirement evicts the whole
  group and withdraws surviving members until rebuild;
- relationship-dependent prepared views must be withdrawn for every
  relationship revision path, including explicit transitions, endpoint
  replacement, and endpoint retirement, and relationship state cannot change
  while either endpoint is under destructive preparation; and
- destructive attempt ownership must outrank every older replacement-only
  result even when the participant had never previously been prepared. During
  mutation and after destructive failure, withdrawn guarantees cannot be
  restored by stale optimistic publication.

The spike also demonstrated that persistence must select only participants
declared as part of the persistent product, that an invalid participant cannot
start replacement-only work whose publication is already impossible, and that
a successor participating in relationships needs a complete arrangement
amendment rather than participant-only redeclaration. Exact resume can preserve
logical authority, participant references, and accepted revisions while
creating new Python objects and a new execution-session identity; another run
loading the same artifact receives new participant identities plus explicit
artifact lineage.

These conclusions are architectural evidence, not approval of the spike's
names or container shapes. The fake backend does not exercise real
Accelerate/DeepSpeed behavior, actual serialization, concurrency or locking,
process failure during physical publication, or production migration. The
known-implementation catalog and candidate evidence are stand-ins for a later
contract/conformance implementation. Optimization ownership remains purposely
unsettled; jointly preparing an optimizer in a backend call did not decide the
optimization exchange.

The binding/preparation pressure-test item is complete. The next architecture
dependency is the optimization exchange, followed by the step exchange. Only
after those are coherent should the current-to-target implementation mapping
and governing OpenSpec be derived from the full design.

This sequencing conclusion is superseded by the governing OpenSpec transition
decision of 2026-08-27 below: the mapping now exists, and the remaining design
should continue inside the governing change while implementation stays gated.

## Optimization ownership code-evidence pass (2026-08-25)

A Verify-level source pass traced current trainable selection, parameter
grouping, optimizer/scheduler construction, Accelerate/DeepSpeed preparation,
clipping and synchronization, ordinary step execution, schedule-free runtime
transitions, and the EDM2 sidecar. The graph generation was current for the
branch, every cited Python path had no recorded coverage issue and matched
index metadata, and exact source snippets were read for the material claims.
This remains a best-effort coverage signal rather than proof of completeness.

The pass confirmed why optimization ownership is a separate architecture
question rather than a small rename of `TrainingMode`. Today:

- adapter and fine-tune `prepare_trainables()` mix authored-subject resolution,
  domain construction or family post-processing, `requires_grad` realization,
  train/eval transitions, and Trainer-state publication;
- both modes construct useful logical/execution groups and an
  `OptimizationPlan`, but also instantiate the optimizer themselves;
- Trainer separately creates the scheduler, while mode-owned accelerator
  preparation replaces modules, optimizer, and scheduler and publishes the
  synchronization handle and primary trainable through whole-Trainer mutation;
- the loop already owns normal timing, loss modification, backward, clipping,
  optimizer/scheduler advancement, zeroing, and train/eval boundaries, but it
  re-queries the mode for clipping participants and specialized lifecycle
  behavior; and
- EDM2 has a learnable accumulation participant plus its own optimizer,
  scheduler, advancement, zeroing, and artifact contribution. Schedule-free
  and fused policies add different runtime behavior without necessarily
  requiring strategy-owned manual optimization.

The strongest lifecycle clarification is that optimization is semantically
downstream of accepted bindings but physically crosses preparation. A logical
plan and concrete optimizer/scheduler candidates must exist before a backend
can jointly prepare them with modules; the current Trainer-owned optimization
runtime exists only after that preparation result is published. The working
frame is therefore:

```text
accepted bindings + authored intent + Trainer policy
  -> resolved logical plan
  -> realized trainability and optimizer/scheduler candidates
  -> joint runtime preparation where required
  -> published prepared bindings + optimization runtime
  -> Trainer-owned step execution
```

This does not reopen preparation ownership. Backend joint preparation
constrains realization but does not make optimizer objects strategy
participants or give the backend semantic optimization authority.

The target should evolve the existing logical-group/execution-group split while
removing its competing sources of truth. One accepted optimization state must
connect participant-qualified selection and source revisions to realized
parameters, logical and execution groups, clipping/synchronization membership,
optimizer/scheduler policy and runtime, preparation dependencies,
checkpointing, and invalidation. Raw parameter lists, `_train_*` flags,
`_primary_trainable`, `_grad_sync_handle`, and later mode queries are current
evidence, not independent target authorities.

The pass did not settle whether the standard profile contains one or several
optimization units, which auxiliary behaviors fit normal Trainer coordination,
or the exact research/manual takeover contract. Those are the first of the
eight recorded discussion points in the exchange design. A useful working
distinction for that discussion is that custom optimizers, multiple groups,
ordinary auxiliary optimizers, schedule-free transitions, backend-specific
advancement, and Trainer-timed post-step capability behavior may still fit
standard Trainer ownership. Manual/research ownership is needed only when a
strategy deliberately replaces named mechanics such as backward count,
optimizer selection, advancement order, or zeroing; the boundary must prevent
both Trainer and strategy from performing the same action.

## Optimization feedback reconciliation (2026-08-26)

A follow-up review treated the proposed optimization target as an opinion to
check against the direction, preparation design, and current code. Most of the
feedback refined rather than contradicted the target. The exchange design now
records the following working conclusions without promoting them to accepted
OpenSpec requirements:

- the standard Trainer-owned runtime should contain one or more coordinated
  optimization units; a unit is one independently advanced optimizer-owned
  parameter set, not a participant, model, logical group, or optimizer object;
- units need stable run-scoped semantic identity plus revision across backend
  replacement, exact resume, diagnostics, and replanning, while another run
  establishes its own units; exact representation remains downstream;
- one parameter should belong to at most one standard unit and one execution
  group within it. Current fine-tune grouping already rejects overlapping
  explicit groups. Deliberate overlap requires an accepted ordering/state
  policy but may remain Trainer-owned when declarative;
- clipping, accumulation/synchronization, trainability, and runtime-mode needs
  remain separate coherent projections rather than fields collapsed into an
  optimizer unit;
- optimization may declare that modules require training-mode participation,
  but Trainer/pipeline lifecycle owns module train/eval transitions because
  sampling, validation, and execution also depend on module mode. Optimizer
  runtime transitions such as schedule-free `train()`/`eval()` remain
  unit-local behavior invoked at Trainer-owned times;
- unusual deterministic cadence or ordering does not by itself require manual
  optimization. The manual/research boundary begins when a strategy must
  imperatively decide or perform backward, gradient manipulation, advancement,
  or zeroing outside the accepted Trainer policy; and
- the earlier lifecycle overfit current Accelerate/DeepSpeed ordering by
  placing concrete optimizer creation unconditionally before preparation. The
  preparation design had already left the construction point open. The target
  now requires a semantic plan first and a complete published runtime last,
  while one Trainer-owned realization/preparation job may construct and prepare
  concrete objects in the order required by its backend.

The former eight-question sequence was replaced with five genuinely remaining
items: confirm unit identity/revision semantics; confirm standard non-overlap
and the explicit overlap boundary; settle declarative advancement versus
imperative takeover; define backend-flexible realization/preparation exchange
meanings; and then connect the accepted runtime to the step exchange. This does
not reopen Q1-Q5, binding authority, or the settled preparation publication
semantics. A later executable optimization spike may help discover readable
Python shape once these semantics are settled, but it should not substitute
for them or treat today's backend call order as universal architecture.

## Optimization unit identity and revision decision (2026-08-26)

The first remaining optimization question is settled. A focused pass through
current optimizer wrapping, Accelerate/DeepSpeed replacement, EDM2's separate
optimizer, and runtime resume showed that neither an optimizer object nor a
concrete collection of `nn.Parameter` objects can carry semantic unit
identity. The exchange design now distinguishes:

- a non-positional unit address in the accepted plan;
- one run-scoped unit identity for an independently managed optimization
  responsibility;
- a revision of that responsibility's accepted semantic parameter membership,
  optimizer-significant grouping, optimizer/scheduler policy, advancement
  policy, and semantic dependencies;
- the concrete runtime realization of that identity and revision; and
- mutable optimizer/scheduler state and advancement progress.

Wrapping, backend preparation or replacement, process recreation, and
realizing new concrete parameter handles do not by themselves change unit
identity or revision. A runtime dependency change invalidates the current
realization; the same revision may be realized again when its accepted
definition is unchanged. Replanning that changes accepted membership,
grouping, policy, or semantic dependencies advances the revision while
preserving the unit identity. Split, merge, retire-and-recreate, and replacement
of the independently managed responsibility establish new identities.

The initial candidate becomes current only through successful publication of
the complete realization/preparation attempt. Exact resume restores the same
run's unit identities, revisions, mutable state, and training coordinates while
allowing new Python objects and a new process execution. Another run or fork
establishes new unit identities and preserves source-checkpoint provenance
separately; no general optimization-unit lineage mechanism is implied yet.
Unit-local advancement coordinates may be necessary for differently paced
units, but belong to the later advancement-policy and step-exchange decisions.
The remaining optimization register now begins with standard non-overlap and
the explicit boundary for deliberately overlapping optimization.

## Optimization ownership wording correction and overlap decision (2026-08-26)

A direct re-audit of the normative direction, exchange design, chronological
notes, inventory, and current code found that two recent working
interpretations had become more specific than the underlying agreement. This
checkpoint supersedes the earlier phrases "strategy-owned manual
optimization," "research/manual takeover," and "imperative takeover" in the
two preceding optimization notes.

The agreed research path has two levels. A custom strategy may replace the
standard internal decomposition while still satisfying the active Trainer
contract. If an experiment changes what the Trainer must do or who owns a
normally Trainer-owned mechanic, it must target an explicit contract extension
or version. Nothing settled says that the strategy itself must receive or
perform the transferred responsibility. The standard strategy contract
continues to keep generic optimization mechanics Trainer-owned; the exact
owners and exchanges of any future extension must come from a concrete need.

Current `process_batch()` and `BatchLossOutput` are migration evidence only.
The first intended-Trainer table is a derivation frame, and the step exchange
remains a placeholder. The direction requires the accepted strategy to make
its authored objective/step behavior available to that eventual exchange, but
does not yet decide the internal producer, request, result shape, or whether a
method resembling `process_batch()` survives.

EDM2 is likewise a capability/extension example, not an inherent member of the
core optimization model. Its present learned sidecar, loss participation,
optimizer/scheduler, advancement, and artifact state prove that the system must
be able to accommodate such selected behavior. They do not decide that EDM2
must become a standard optimization unit, that its optimizer becomes
Trainer-owned, or what its final exchange and placement will be.

The standard non-overlap question is settled. Each selected semantic parameter
has one optimization unit and one execution group in the standard profile;
concrete aliasing through tied/shared paths also counts. Deliberate overlap
requires a recognized capability or explicit extension with an accepted
coordination policy, not a bare flag or ordinary strategy mutation. A
library-supported capability carries those rules without making its author
restate them. This does not itself transfer optimization ownership to the
strategy.

The remaining optimization register now has three questions: define the
declarative advancement policies supported by the standard Trainer and the
contract-extension boundary; define the backend-flexible optimization
candidate/request/result exchange; and connect the accepted optimization
runtime to the still-open step exchange.

## Accepted run arrangement topology correction (2026-08-26)

A return to the original architecture discussion exposed a topology mistake in
several later records. The binding and preparation work had correctly
established one contract-governed authority for accepted per-run state, but
subsequent wording placed that authority inside a complete `TrainingStrategy`
that would remain active and exchange work with the Trainer at runtime. The
evidence did not require that placement.

This checkpoint supersedes phrases such as `Trainer <-> complete
TrainingStrategy`, "accepted strategy owns the binding authority," and
"strategy computes the step and returns a result" when they describe the
target runtime topology. It specifically supersedes the earlier statement that
the direction requires an "accepted strategy" to make authored step behavior
available at runtime. It also supersedes treating an active strategy as the
ordinary research escape route.

The corrected route is:

```text
authored strategy
  -> contract validation and fulfillment
    -> accepted run arrangement
      -> binding, preparation, and runtime specialization
        -> one Trainer engine executes the arrangement
```

The strategy remains the authored recipe and explicit composition surface. It
does not need to survive contract establishment as an ordinary per-step
collaborator. Contract establishment must preserve the authored training
meaning in an accepted executable form rather than moving model-specific
knowledge into the Trainer.

The accepted arrangement may combine maintained operations, custom structured
operations, and explicitly authorized imperative regions. This is not a binary
choice between a fully declarative graph and a live strategy callback. An
imperative region is a distinct runtime execution role selected through the
authored strategy and accepted by the contract; it is not automatically the
strategy object itself. Custom decorators are one possible authoring API for
such regions, not an architectural decision.

The normal path therefore needs a structured accepted execution
representation, but its exact form remains open. It may eventually be a graph,
tree, region schedule, lowered Python representation, or a hybrid. The earlier
rejection of an all-purpose `PreparedTrainingProgram` facade does not mean that
contract establishment produces no executable result, and graph-shaped
execution is no longer relegated to a hypothetical future feature.

This correction does not reopen the settled participant identity, route,
binding, preparation, atomic publication, invalidation, persistence, or
optimization-unit decisions. In particular, Q5 remains settled at the
responsibility level: one authority accepts and owns canonical current state
for a run. Its topology is now stated accurately—the authority belongs to the
accepted run arrangement, not necessarily to an active strategy.

The remaining design work is to derive the minimum accepted execution
meanings, optimization exchanges, and retained Trainer mechanics from concrete
normal and increasingly imperative cases. Those cases should discover the
representation and escape-hatch API; they should not assume a universal graph,
a `process_batch()` replacement, or a per-step strategy/Trainer conversation in
advance.

## Governing OpenSpec transition decision (2026-08-27)

The work has reached the point where continuing to develop the intended Trainer
only in pre-OpenSpec notes would risk producing a one-off design influence. The
next step is to create one governing model–strategy–Trainer OpenSpec change and
continue the architecture there. One change may contain several focused specs
and many reviewed milestones; this is not a big-bang implementation.

The intended Trainer has design authority. Repository goals and current-code
evidence inform that Trainer; its required mechanics and deliberately supported
pipeline capabilities define the training contract. The contract then guides
and constrains strategy authoring and judges authored filings. It does not force
the Trainer to preserve behavior merely because a current strategy performs it.
Current SD, SDXL, SD3, adapter, and research paths are evidence and pressure
tests. A desired case may cause a deliberate Trainer-design change, after which
the contract follows; no case defines the core automatically.

The OpenSpec should be rooted in the actual code inventory and the complete
discussion record. The direction document supplies settled normative input;
the exchange design and implementation mapping supply evolving source-backed
derivation; the inventory and executable spike supply evidence; framework
comparison remains subordinate prior art; and these notes preserve chronology,
including superseded positions. Once created, the OpenSpec design and specs
become the governing location for new normative decisions rather than another
parallel living document.

The intended Trainer skeleton belongs in the OpenSpec design. Contract
requirements and representative SD/SDXL/SD3/custom scenarios belong in focused
specs. Reviewed vertical migration steps belong in tasks, and acceptance tests
must prove the scenarios. Each material decision should trace from intended
Trainer responsibility through contract requirement, accepted-arrangement
exchange, representative scenario, implementation milestone, and test. This
trace—not an isolated prototype or persuasive note—is what makes the design
control the resulting code.

Implementation must not begin merely because the change exists. Its design,
requirements, and tasks first need the four exchanges, standard and extension
optimization boundaries, multi-family and imperative pressure cases, and a
migration that avoids duplicate current-state authorities or knowingly false
temporary contracts.
