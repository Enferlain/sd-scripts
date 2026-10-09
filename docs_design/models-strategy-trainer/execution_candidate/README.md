# Connected execution candidate

This is isolated executable research, not an adopted architecture, production
package, authoring API, or completed G5 milestone. The governing design and
specs remain unchanged. Names, constructors and the CPU/asyncio target are
candidate choices; their restrictions are not new language requirements.

The purpose of this pass is to show the mechanism **as a connected system**:
an accepted hierarchy and its relationships produce executable work and the
coordination that work requires. It does not start with an input-action-optimizer
loop and make its helpers configurable.

## Start here

Read [examples.py](examples.py), particularly `diffusion()` and `adversarial()`.
They use the same language constructs and engine but different computation,
input relationships, policy and optimization membership. The numerical models
are tiny CPU PyTorch modules, not real SDXL/GAN integrations.
`producer_continuity()` adds a connected, bounded producer with changing source
state and finite demand. Its stop connection is authored; the engine does not
infer it from a training step or queue fullness.
`routed_views()` adds two prepared roles of one frozen participant and two
destination-specific derivatives. Its blocks share an explicit old-state
lifetime, not an implicit global training step; see the walkthrough below.

```python
contract = Contract()              # CPU target's offer/rules already exist
filing = diffusion()               # select and connect maintained behavior
accepted = contract.accept(filing) # known compatibility/wiring checks
attempt = prepare(accepted)        # load, verify, resolve, lower; unpublished
ready = attempt.publish()         # install one coherent current image
await Engine(ready).run()          # invoke machinery derived from that graph
```

`Contract` is deliberately a small probe schema, not the full queryable
training contract or its chosen code placement. The examples directly construct
an intermediate description; a finished ordinary strategy authoring experience
is not being proposed here.

Run from the repository root:

```bash
uv run --no-sync python -m docs_design.models-strategy-trainer.execution_candidate.examples
uv run --no-sync pytest docs_design/models-strategy-trainer/execution_candidate -q
uv run --no-sync ruff check docs_design/models-strategy-trainer/execution_candidate
uv run --no-sync ruff format --check docs_design/models-strategy-trainer/execution_candidate
uv run --no-sync ty check --extra-search-path docs_design/models-strategy-trainer docs_design/models-strategy-trainer/execution_candidate --error-on-warning
```

The extra search path lets `ty` resolve this isolated package beneath the
hyphenated design directory. This candidate uses Python 3.12+ type-alias syntax;
verification used Python 3.13.13 and PyTorch 2.11.0+cu130 on CPU. It changes
neither the project's Python requirement nor dependencies.

## How the whole thing connects

```text
Existing target contract + selected library behavior
                         |
                accepted hierarchical graph
                         |
            preparation derives work and coordination
                         |
       unpublished bindings + optimizer members + callable root
                         |
              checked, coherent installation
                         |
               same generic Engine.run()
                         |
          derived scope machinery and direct work calls
                         |
        canonical current bindings + separate owner state
```

For the diffusion-style case, the retained meaning is:

```text
run: together                                  [shared completion/failure]
├── inputs: together                           [child activity lifetime]
│   ├── pixels: independent provider ──image───────┐
│   └── captions: independent provider ─condition─┤
│       └── private workers use protected encoder │
├── learning: selected repeat policy              │
│   └── learn: correlate both products ◄──────────┘
│       objective → feedback → differentiate → advance
└── validation: requests from learning completion
    └── evaluate using protected current state
```

Hierarchy establishes activity lifetime and failure propagation. Value
references and explicit ordering links establish relationships inside a block.
Provider handoffs and completion/request links connect different owners.

`prepare.py` derives this machinery, once:

| Accepted meaning | Concrete CPU target |
| --- | --- |
| Together scope | An `asyncio.TaskGroup` whose child callables come from the child scopes |
| Independent producer | Selected provider activity and cleanup, bounded addressed ports, optional accepted demand-end connection |
| Repeat with accepted alternatives | Selected live policy plus prebound block callables; only accepted alternatives can be chosen |
| Connected work in a block | Generated Python call chain in dependency order; direct local variables carry values |
| Sequence with cross-block values | One generated chain with block-origin frames, direct locals and work-sized leases |
| Named participant views | Separate prepared callable attachments to the same binding; no extra participant/unit identity |
| Retained old state and explicit last use | A source lease or verified isolated CPU copy, member correspondence, borrowed-value lifetime and release |
| Requested capability | A bounded connection from source completion to selected request policy and protected capability work |
| Participant/unit uses | Protected current projections, final parameter resolution and retained training mechanics |
| Compatible replacement | A checked replacement proposal, candidate preparation, owner continuity checks and coherent publication |

This is not two independently authored runs. `Accepted.root` retains the
meaning. The callable root, block bodies and request connections are derived
attachments. `Ready` does not contain a second authored description, family
choice or separately maintained dependency graph. Generated block source retains
the accepted work addresses for inspection.

For example, `learn` lowers to ordinary calls wired by local variables:

```python
image, condition = await correlate(frame)
loss, error = await objective(frame, image, condition)
await feedback(frame, error)
gradient = await retained_backward(frame, loss)
await retained_advancement(frame, gradient)
```

Those readable names illustrate the actual generated body printed by the demo;
the bound names in generated source are `work_0`, `work_1`, etc. The author did
not separately write that order. Reversing the authored work tuple leaves its
dependency-derived order intact; changing the relationships changes execution.

The engine does not branch on diffusion, D/G, SAM, batch shape or capability
names. Neither does it walk the static graph each iteration. Selected runtime
policy and algorithms remain live: feedback changes later computation, policies
choose accepted alternatives, capability requests depend on live completion,
and operations can suspend. Finishing authoring is not freezing execution.

## State, preparation and authority

`state.py` holds one installed image. Its bindings are the participant
authority; optimizer runtime and selected policy/provider/capability state are
distinct mutable owner sections. Ordinary implementations receive only their
declared models and owner state, not the authoring object or whole Trainer.

Initial preparation includes frozen participants, verifies concrete objects,
resolves final parameter paths, consolidates aliases within one unit, rejects
overlap between different units, creates CPU optimizers, and lowers the graph.
Only then may publication install the complete image. A stale attempt cannot
publish. This is logical single-process publication, not distributed commit.

Compatible replacement keeps participant and semantic unit identity, revises
the binding, rebuilds affected optimizer runtime, and preserves unit progress
and known outcomes. This probe explicitly resets affected optimizer state;
selected owner continuity is preserve/reset/reject. Unaffected plain CPU units
can remain prepared when their exact members still correspond. This is not
evidence for reusing a distributed composite or supporting destructive in-place
preparation.

Provider owner state is bound at activity startup, not during lowering. An
activity-lifetime registration pins that exact object through selected work,
private worker joining and explicit cleanup. A replacement requiring reset of
that state rejects while the activity is active, including during cleanup.
After quiescence it can reset; accepted preservation retains the same object.
This does **not** implement live continuation migration or certify that a
selected preservation policy is semantically appropriate for every provider.

Ordinary numerical progress does not replace participant identity, binding or
prepared generation. A conservative numerical epoch tracks returned optimizer
calls and declared writes; it is **not proof that tensor values changed**.
Admission checks current provenance, including changes within the same block.
Each producer retains the actual protected source stamp it used, even if
publication occurs after replacement or a trainable-source update. Payload
details such as caption revision belong to the selected input policy.

Declared producer source writes now have a returned-effect boundary: successful
exit from `use_source()` increments the conservative source epoch **before**
releasing protection. Failed/interrupted writes withhold that source. A provider
may select a read-only subset of its accepted access but cannot upgrade it.
Entry stamps are not certificates for intermediate writes. The positive fixture
completes its write interval, then reads under fresh protection to stamp the
new product. Pre-write products keep their earlier provenance and cannot pass
an exact-current admission policy after the write. Explicit writes in granted
work also report returned effects after verified handback.

## Producer continuity and demand end

This pass adds one concrete lifecycle policy, not a universal run sequence:

```text
encoder-work: independent producer [private owner + bounded workers]
    ├── old-state product 2 ─────┐
    ├── returned source write   │ ready port capacity = 2
    ├── new-state product 1 ────┤
    └── private pending/produced│ work remains separately accounted for
                               ▼
learning: finite selected policy
    wait for bounded production → admit product 1 → objective → backward → advance
    whole lifetime ends ───────────────► encoder-work stop request
                                         ├── close/discard unused ready work
                                         ├── cancel and join selected activity
                                         └── selected private-work disposition/cleanup
                                              owner becomes quiescent
```

`Producer(stop_when="learning", remaining="discard", cleanup=...)` derives
a normal-completion connection to that scope, rather than a per-iteration
callback. The target can instead be an independent `Together` containing all
consumers; it cannot be an ancestor containing the producer, a repeat block
invocation, or a lifetime leaving another active consumer of those ports behind.
Those knowable target mismatches reject during acceptance, before loading.
Full-lifetime completion dependencies also reject mutually waiting groups;
that analysis is separate from input-readiness and lease waits, since making a
product ready does not require its producer's whole lifetime to finish.
If the producer finishes ahead of demand, its private owner is released after
successful cleanup but port coordination remains until demand ends. Failure
instead aborts the accepted shared lifetime with its failed ports/products
retained for diagnosis; it is not reported as normal demand-end discard.
If demand already ended, new private work is not started. Without an explicit stop connection,
finite completion remains provider-owned and overproduction can still block.
Disabled requested capabilities are not eligible demand lifetimes, and their
uninvoked bodies do not acquire the target's runtime-only protection limits.

The example bounds ready + private pending + produced-unpublished work at four,
with at most two ready values. This is selected-provider evidence, **not** a
generic guarantee inferred from channel capacity. Channel coordination records
claims and ready-work discards; the selected owner records its private-work
disposition. Neither publication nor discard is reported as consumption.
Their diagnostic records are bounded and not durable exactly-once receipts.

Normal accepted stop is distinct from failure: cancelling cooperative read work
and successfully joining/disposing its remainder can leave state usable. An
interrupted write still leaves its source uncertain. Unplanned failure gates
source/owner use and closes failed ports before awaited cleanup finishes.
Further parent cancellation cannot detach running cleanup or make owner reset
safe prematurely. Cancellation before owner registration/selected work is
recorded as not started, with no private cleanup or effects invented; a planned
pre-start stop does not turn its port into a failed producer.
If cleanup fails, primary interruption/failure, cleanup
failure and previously returned optimization outcomes remain distinguishable;
no rollback or replay is inferred. The implementation/cleanup bodies are trusted
to join their private workers before returning **or raising**; coroutine
completion is quiescence evidence only under that selected conformance.
Detached tasks, uncooperative code, external
side effects and a general drain/transfer/resume policy are not implemented.

Evaluation calls likewise retain selected return/primary failure separately
from mode-restoration failures, attempt all captured restorations, and withhold
uncertain state. A returned declared write remains recorded even if cleanup or
output checking fails afterward. Invalid-return evidence is checked before
cleanup, so a cleanup failure cannot hide it either. Ordinary calls no longer inspect module trees
for evaluation-mode snapshots they do not need; this is not a measured speedup.

The two-pass example is an explicitly offered authority profile. It grants
backward, temporary edits and intermediate reset. It must restore the view and
hand back a checked final contribution. Clipping, final optimizer call and
final reset remain retained mechanics. Duplicate backward ownership and final
advancement are rejected. Restoring weights and then raising is not successful
handback. The underlying Python implementations remain trusted: access seams
and declarations do not sandbox arbitrary code receiving live PyTorch objects.
`implementation_returned` means exactly that, not accepted output, verified
handback or recorded effects. A granted implementation can return and then
fail handback; its unit remains unverified and its source is withheld rather
than receiving returned-effect evidence.

## Frozen views, routed gradients and cross-block last use

This follow-up implements the next pressure case from the external feedback,
using fixture B in `trainer-architecture-concrete.md` as an oracle and the
governing design/specs as authority. It adds **candidate** mechanics where the
initial representation could not honestly express the case: values and
contributions were block-local; every block had to advance its own units;
named execution views and old-state last use were absent.

The connected example is `routed_views(isolated=..., recompute=...)`:

```text
routed-work: sequence                         [one cross-block lifetime]
├── first-loss
│   retain old F/a/b ──protected source OR isolated copy─────────┐
│   selected Python: target F without gradients; h = conduit F(a)
│   L1 = h*b ──differentiate ONLY A(a)──ga ready                │
│   isolated version: advance A now                            │
├── second-loss                                                │
│   L2 = h*h*b, shared h OR recomputed through old F/a ◄────────┤
│   differentiate ONLY B(b)──gb ready                           │
│   release old source after final derivative use ◄────────────┘
│   protected-live version: advance A now
└── finish-b
    advance B from its cross-block contribution
```

For `F(a)=k*a`, frozen `k=1`, `a=2`, `b=3`, the authorized gradients are
`A=3`, `B=4`. Adding the losses and differentiating both subjects gives
`A=15`, `B=6`, which is not this accepted meaning. Tests compute both results
with direct PyTorch and compare the candidate's actual gradient buffers, not
just two returned tokens. At learning rate 0.1 the intended updates produce
`a=1.7`, `b=2.6`; F has neither a gradient nor an optimization unit.

### What preparation derives

`Run.views` describes `target` and `conduit` roles of **the same** `frozen`
participant. Its prepared image contains separate `PreparedView` callables,
both attached to the same module/binding. A selected call receives only the
requested views through `Context.views`; a named view does not expose another
raw model through `Context.models`. Replacement reattaches both current views
to the new binding while preserving participant identity. Retained isolated
views attach to the old copied representation without creating an incarnation.

In this CPU target, target invocation disables gradient recording and briefly
uses evaluation mode; the conduit records input gradients even though F's
own parameters are frozen. Mode restoration preserves mixed child modes on
return or failure. Forward and restoration failures are both retained. These
are synchronous view calls: they never suspend with a changed module mode.
This proves neither thread safety nor asynchronous backend completion.

`Sequence` gives these blocks an explicit order and a shared value scope.
Preparation resolves the cross-block links into direct local variables, binds
each work implementation and emits separate block-origin frames. The root
engine remains unchanged. The hot path does not rediscover the work graph.
`Differentiate` still delegates formulas and tapes to PyTorch. Preparation
conservatively retains a tape when another derivative request names that same
retained source; the shared-tape and recomputation variants both run. This
conservative rule is not a minimal tape-memory planner.

Gradient routing resolves each unit's accepted members, including known aliases
within that unit. An isolated source has explicit member correspondence to
those current destinations; its cloned Parameter objects do not become the
semantic targets. First-order returned gradients are copied into independent
pending buffers: even a custom backward returning source storage cannot make
a later source write silently change a pending contribution.

### Readiness, advancement and publication are different

| Point in the sequence | Protected current old state | Explicit isolated old state |
| --- | --- | --- |
| A's gradient ready, B's derivative still due | A remains pending; old F/a/b is protected | A may advance; B still uses the real old copy |
| B recomputes/differentiates | Uses the unchanged live source | Uses copied old storage and its original stamps, not A's new current epoch |
| Last old-state use finishes | Source protection can be released | Copied-source lifetime can be closed |
| A/B contributions are ready but unadvanced | Pending unit state still forbids replacement in this target | Same restriction; copying does not authorize gradient/state migration |

The source lifetime is visible to coordination, not hidden in an arbitrary
Python closure or an optimizer object. It protects registered participant
state across blocks until actual synchronous last use. For the live variant,
an external writer waits; an attempted same-sequence early update is rejected
rather than waiting on its own future release. A ready gradient cannot bypass
the check through the standard advancement service.

The isolated variant performs real CPU module/state copying under source
protection. It checks copied registered state, member alias correspondence and
disjoint storage. Origin stamps keep the same participant/binding identity
and old numerical epoch. A revision label, wrapper or copied reference is not
an isolated version. This is a narrow copying target, **not** a proof that
arbitrary custom-module private state, hooks, external resources or RNG-driven
algorithms admit faithful copying.

Borrowed values from retained work have a conservative lifetime in this
candidate. They cannot be passed to later work after release, nor can their
derivatives silently switch to current/some other retained destinations.
Finalized first-order contributions outlive that source through their own
pending-gradient/preparation correspondence. Calls do not receive retention
or contribution coordination handles. Arbitrary trusted Python remains able
to violate declarations through hidden references or mutations; this is no
sandbox or static proof of all gradient/effect behavior.

### Failure and continuation evidence

Unit records preserve derivative-origin invocation, advancement invocation,
shared execution-region association and old source stamps. A's and B's clocks
advance independently. If A returns and B mutates then raises, A remains
`returned`, B is `uncertain`, reset after B is `not_attempted`, and later work
has no entered trace. No all-or-nothing result, numerical-change detection,
rollback or automatic retry is inferred.

Failure/cancellation drains this region's remaining source lifetimes while
withholding its possibly affected participants/owners. Repeated cancellation
does not detach lifetime cleanup or release protection before cleanup finishes.
Pending gradients and uncertainty are **not** erased by release. The candidate
rejects capture and replacement while relevant unfinished work remains; it
only captures the finished quiescent cut. It does not persist pending tapes,
restore cross-block execution, migrate gradients or recover a partial update.
Recovery would require a compatible prior coherent snapshot or another
explicit supported policy, neither of which this fixture implements.

`test_routed_views.py` provides 29 added cases: direct numerical routing,
separate views, mixed-mode restoration, live versus genuinely separate old
storage, shared versus recomputed work, update eligibility, blocked writers,
pending replacement, partial outcomes, cancellation, borrowed-value release,
source aliases and absence of hot-path graph discovery.

The present target supports one first-order contribution per unit in a
Sequence, with one final advancement and an explicit protected/isolated
derivative source. Per-work leases alone do not make an unprotected
forward-to-backward gap safe, so those descriptions reject before loading.
The existing granted two-pass handback remains block-local; it is not falsely
accepted as a Sequence window. Retained calls use named synchronous view
modes, not the broad evaluation flag that may span asynchronous Python work.
The target does not yet support multiple
contributions to that window, repeated/dynamic Sequence alternatives,
multiple retained-source provenance in one derivative, higher-order work,
views spanning asynchronous/device use, or input waits within a retained
sequence. Those are candidate limits, not restrictions on the intended
whole-run language. Ordinary blocks retain their existing conservative
whole-block lease path; sequences need smaller leases because acquiring a
future update's write permission before an earlier release would deadlock.

The next separate investigation is comparative lowering/specialization of the
**same accepted graph**, preserving its boundaries and selected behavior. It
should not introduce a second independently authored run description.

Verification of this follow-up: 128 tests pass together (116 candidate cases
and the 12 unchanged execution tests); scoped Ruff lint/format, strict `ty`
with the extra search path above, example execution and diff checks pass.
The required `review-mcp` attempt did **not** produce a verdict: it failed
with provider 429/1308 (five-hour quota exhausted). Subsequent self-checks
closed the falsely accepted unprotected sequence, block-local grant and broad
retained evaluation shapes noted above and reran the gates. Bead
`sd-scripts-syv.7` remains open awaiting required review; no review approval,
production readiness, architectural adoption or G5 completion is claimed.

## Evidence and its limits

Verification checkpoint, 2026-10-09: 87 candidate cases plus the 12 unchanged
`tests/unit/training/test_execution.py` cases pass (99 total); scoped Ruff,
format checking, `ty --error-on-warning` and diff whitespace checks are clean.
All six examples execute. A repeated-run check completed 100 connected producer
runs without remaining asyncio tasks. These are correctness/cleanup checks,
not performance or arbitrary-worker-conformance evidence.

The required review of the main follow-up confirmed all four defect fixes and
found no functional blocker or spec violation. Its minor findings were addressed
with explicit never-started cancellation semantics, successful-versus-failed
early-finish documentation, rejection of dormant demand targets, inactive-body
target-limit handling and a returned-versus-verified-handback test. Small local
follow-up guards also reject declared completion cycles and preserve invalid
output evidence alongside cleanup failure; affected quality gates were rerun.

The candidate test cases exercise:

- The same engine with differing whole-run relationships, including a state-only
  activity needing no input/loss/optimizer placeholders.
- Dynamic feedback, accepted runtime choices, requested versus dormant validation,
  and no static wiring traversal or retained authoring object on the hot path.
- Independently progressing production, private workers, out-of-order completion,
  bounded backpressure, correlated multi-input claim, actual source provenance,
  trainable-source changes and provider cancellation/cleanup. A readiness wait
  does not take a lease needed by the producer; a claim race cannot silently
  wait again under a conflicting lease. Insufficient reordering capacity gives
  a concrete failure with products retained, rather than an indefinite wait.
- Frozen preparation membership, failed/stale publication, final parameter
  correspondence, aliases, compatible replacement and stable unit progress.
  Physical sharing cannot hide behind separate participant addresses or
  distinct Parameter objects; borrowed-current replacement is rejected.
- Joint gradient work with separate unit outcomes; zero learning rate does not
  become a false numerical-change claim.
- Granted two-pass work, protection through suspension and handback, duplicate
  ownership rejection, partial optimizer failure and invalid output after effects.
- A finished, quiescent **in-memory** capture with distinct owner positions,
  and rejection of unusable state or unfinished handoff/contribution work.
- Malformed descriptions and unowned channels rejected before loading/lowering;
  invoked runtime-bound violations distinguished from authored rejection,
  including malformed request choices and unhashable work identity.
- Direct and indirect declared lease/source wait cycles rejected before
  loading, with an acyclic later-wait case still executable. Shared target
  protection semantics drive acceptance checks and lowering; concrete storage,
  member and backend evidence remains in preparation. Finished-cut
  capture cannot mistake an initialized, never-started run for completion.
- Connected finite-demand producer shutdown, a composite demand lifetime,
  active/reset versus preserved owner identity, startup after replacement,
  partial source effects, private and ready-work disposition, repeat cancellation,
  early uncertainty gating and separate primary/cleanup outcomes. Tests include
  cleanup failure after an already returned optimizer advancement.

When A's optimizer returns and B's optimizer mutates then raises, the candidate
retains A's returned/reset result and B's uncertain/unattempted result. It
withdraws current state from further use after releasing protection. There is
no transaction, automatic retry, rollback or proof that an optimizer return
changed parameters. Diagnostic histories are bounded, not durable receipts.

This target is intentionally narrow:

- CPU synchronous tensor work and cooperative asyncio activities only. It does
  not establish GPU completion, multiple execution views, distributed collective
  agreement, threads/processes, FSDP/DeepSpeed or compiled numerical lowering.
- Optimizer construction offers SGD only. The joint-unit example establishes
  coordination and separate outcomes, not real Muon/AdamW backend integration.
- One local contribution window per unit per block, with one final advancement.
  No accumulation across blocks, derivative continuations, retained autograd
  versions, optimizer-specific closures or atomic multi-unit advancement.
- Together/Repeat/Requested/Block exercise hierarchy, but do not implement every
  possible region or cross-owner scheduling relationship in the governing design.
- Prefix input readiness waits occur before acquiring participant/owner leases;
  admission and all-member claim occur under protection against current state.
  A readiness race rejects without waiting again under that protection.
  Beyond that, protection is conservative across the whole block. Cold target
  checking derives wait edges between later-joining blocks, their producers and blocks
  holding conflicting source access, and rejects potential cycles (including
  indirect ones). The analysis can reject feasible conditional schedules and
  is not a global liveness proof for arbitrary Python or private provider work.
  Such patterns need finer acquisition/lifetime evidence or smaller scopes;
  this is not a restriction on the intended language.
- Participant footprints must be disjoint for this target. Sharing modules,
  tensors or storage between participants requires relationship-aware protection
  and invalidation that is not implemented here. Distinct Parameter views of
  the same storage also need an explicit alias mapping; aliases referring to
  the same Parameter object within one unit remain supported. Replacement must
  provide a fresh physical realization, not borrow current objects/storage.
  Detection covers registered modules, parameters and buffers with shared
  objects or matching storage base addresses. It is not proof about private
  Python attributes, arbitrary external-memory overlap or the whole heap.
- The handoff target fails closed when a full feed cannot serve the current
  correlated demand. That is a concrete bounded-reordering policy, **not proof
  of global deadlock**. It may reject a schedule that a different multi-consumer,
  demand-aware or reserved-slot target could execute. It does not evict queued
  work, expand capacity or automatically choose another input. Capacity and
  production order must accommodate the accepted consumption order here.
- Producer completion is provider-owned unless an explicit accepted demand-end
  relationship selects this target's ready-work discard policy. It does not
  infer demand end or automatically stop arbitrary overproduction. Other
  retain/drain/transfer policies and live continuation migration are unimplemented.
  A producer without that relationship can still block on backpressure after
  its finite consumer ends. Cooperative cleanup must join private work; there
  is no hard shutdown deadline or proof of private worker conformance.
  Duplicate-key detection covers currently ready products, not a durable
  exactly-once publication history.
- Replacement only; no general topology/contract amendment, destructive mutation
  recovery or backend-specific dependency reconstruction.
- Validation is a tiny selected capability, not the production caching, sampling,
  artifact, restoration, metadata or resource-observability integrations.
- Capture is not durable publication or same-run restoration. It accepts only a
  finished quiescent run with no unfinished provider/handoff/contribution work.
  No unfinished-work recovery, replay guarantee or restoration implementation.
- Output checking, frames, projections, diagnostic records and coroutine dispatch
  still have cost. Direct-call lowering is established; G3.7's performance gates
  are not demonstrated by these correctness tests.

## What this pass establishes

There is now a connected, executable candidate for deriving both executable
work and coordination from the retained graph. Different arrangements use the
same mechanism without family-specific engine branches or a universal training
step. Preparation and execution use the same current-state authority.

The follow-up establishes one governed producer lifetime and repairs four
reproduced defects in the initial candidate: stale owner objects after reset,
unreported producer writes, overwritten cleanup/primary outcome classification,
and knowable target checks deferred until after loading. It does not adopt these
constructors as the production API or close G5. The subsequent frozen-view
pass above adds routed derivatives and cross-block windows; comparative
specialization remains a separate investigation. No broader guarantees are
inferred from either fixture.

That is evidence for this realization approach, not proof that it satisfies
every requirement or is the best production implementation. The next review
can examine actual authoring readability, concern boundaries and the cost of
the derived machinery, rather than infer them from an architectural diagram.
The code-level pass also exposed and corrected a coordination defect: taking a
whole-block write lease before input readiness could block the producer whose
read access was needed to supply that input. The separate readiness/admission
lowering above is executable evidence, rather than just a declared safe-access
rule. Contribution tokens also carry opaque preparation correspondence keys,
not public optimizer handles, and rejected policy output withdraws any possibly
mutated policy state.

The first required review supported the graph-derived core and identified
bounded out-of-order queue waiting, malformed descriptions, unowned channels,
an unused replacement setting and conflated runtime/static rejection. Those
findings have code corrections and regression tests; the replacement setting
was removed because this target offers only explicit reset, not a configurable
transfer policy. Shared handoff conditions and restoration phase order also
have executable guards. The broader target limitations above are unchanged.

The follow-up verified those corrections and identified indirect lease/source
cycles, malformed request choices and a refilled-feed race diagnostic. The
cycle analysis and common choice validation above address them, with negative
and positive regression cases. A separate self-check demonstrated that updating
one of two participants sharing a Parameter changed the other without advancing
its stamp; this target now rejects physical sharing until it can lower the
necessary relationships. These are backend-mechanism findings, not additional
architectural requirements. All 51 candidate cases plus the 12 unchanged tests
in `tests/unit/training/test_execution.py` pass together.

The final, narrowly scoped review found no actionable findings in those cycle,
physical-footprint, runtime-choice, admission-race and completed-cut guards.
It noted minor diagnostic limits for exotic tensor errors and very deep
dependency recursion, and the registered-state scope of footprint evidence;
the latter is now explicit above and in the implementation docstring. This is
review of the candidate's corrections, not G5 or all-backend certification.

Independent work without explicit dependency links uses a stable authored
tiebreak in this candidate. Authors must express required effect ordering;
swapping unrelated work can change which joins are lease-free prefixes. The
dependency-order test establishes preservation of explicit relationships,
not equivalence of every permutation of unconnected effectful work.

The production API, specialized lowerings, backend support, broader conformance
and G5 implementation order remain unselected.
