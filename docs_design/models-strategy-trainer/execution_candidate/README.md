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
uv run --no-sync pytest docs_design/models-strategy-trainer/execution_candidate/test_candidate.py -q
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
| Independent producer | Selected provider activity, bounded addressed handoff ports, close/failure handling |
| Repeat with accepted alternatives | Selected live policy plus prebound block callables; only accepted alternatives can be chosen |
| Connected work in a block | Generated Python call chain in dependency order; direct local variables carry values |
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

Ordinary numerical progress does not replace participant identity, binding or
prepared generation. A conservative numerical epoch tracks returned optimizer
calls and declared writes; it is **not proof that tensor values changed**.
Admission checks current provenance, including changes within the same block.
Each producer retains the actual protected source stamp it used, even if
publication occurs after replacement or a trainable-source update. Payload
details such as caption revision belong to the selected input policy.

The two-pass example is an explicitly offered authority profile. It grants
backward, temporary edits and intermediate reset. It must restore the view and
hand back a checked final contribution. Clipping, final optimizer call and
final reset remain retained mechanics. Duplicate backward ownership and final
advancement are rejected. Restoring weights and then raising is not successful
handback. The underlying Python implementations remain trusted: access seams
and declarations do not sandbox arbitrary code receiving live PyTorch objects.

## Evidence and its limits

The 51 candidate test cases exercise:

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
  publication, with an acyclic later-wait case still executable. Finished-cut
  capture cannot mistake an initialized, never-started run for completion.

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
  Beyond that, protection is conservative across the whole block. Cold lowering
  derives wait edges between later-joining blocks, their producers and blocks
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
- Producer completion is provider-owned under this all-children-finish target.
  It does not infer end-of-demand or automatically stop overproduction. A
  producer exceeding its consumer's finite demand can block on backpressure;
  lifecycle/stop relationships for such arrangements remain unimplemented.
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
