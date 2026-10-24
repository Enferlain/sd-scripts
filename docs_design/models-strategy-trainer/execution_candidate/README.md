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
The [same-graph lowering comparison](#same-graph-block-lowering-comparison)
below now contrasts two CPU emission paths without changing those meanings.

```python
contract = Contract()  # CPU target's offer/rules already exist
filing = diffusion()  # select and connect maintained behavior
accepted = contract.accept(filing)  # known compatibility/wiring checks
attempt = prepare(accepted)  # load, verify, resolve, lower; unpublished
ready = attempt.publish()  # install one coherent current image
await Engine(ready).run()  # invoke machinery derived from that graph
```

`Contract` is deliberately a small probe schema, not the full queryable
training contract or its chosen code placement. The examples directly construct
an intermediate description; a finished ordinary strategy authoring experience
is not being proposed here.

Run from the repository root:

```bash
uv sync --locked
uv run --locked python -m docs_design.models-strategy-trainer.execution_candidate.examples
uv run --locked pytest --noconftest docs_design/models-strategy-trainer/execution_candidate tests/unit/training/test_execution.py -q
uv run --locked ruff check docs_design/models-strategy-trainer/execution_candidate
uv run --locked ruff format --check docs_design/models-strategy-trainer/execution_candidate
uv run --locked ty check --extra-search-path docs_design/models-strategy-trainer docs_design/models-strategy-trainer/execution_candidate --error-on-warning
```

The extra search path lets `ty` resolve this isolated package beneath the
hyphenated design directory. `--noconftest` excludes the old training stack's
Hydra/schema fixtures; these tests do not use them. It does not disable the
candidate's own tests or pytest plugins.

### Repository Python 3.14t environment

The repo's `.python-version` selects `3.14t`; project, Ruff and ty metadata
target Python 3.14. uv 0.12+ installs the project dependencies and default `dev`
group from the root `uv.lock`. The temporary `execution-research` group and old
Torch/CUDA extra matrix have been replaced by one Torch 2.14.1 / torchvision
0.29.1 stack from the CUDA 13.0 index. The examples still use tiny CPU tensors;
installing a CUDA build does not make this candidate a CUDA/backend test.
NumPy and the updated image, Hugging Face and optimizer dependencies are now
part of the project environment, not excluded from candidate installation.
uv honors `UV_PROJECT_ENVIRONMENT` here (currently `.venv-wsl`); without that
override it uses the project's `.venv`. This is not a separate research project
or a pip-managed environment.

The current `pyproject.toml` records the tested baseline and remaining dependency
considerations: OpenCV needs a source build with uv's configured Limited API
workaround, so initial synchronization must allow builds. OpenCV, grpc and
Triton can re-enable the GIL; native Windows support and several optimizer
upgrade checks remain outstanding. Installation is not full Trainer/API
compatibility, thread-safety or concurrent-performance evidence. A free-threaded
build and a workload that actually keeps the GIL disabled are separate checks;
Forcing `PYTHON_GIL=0` overrides extension declarations and does not demonstrate
safe free-threaded execution; the checkpoint below does not force it.

Initial final-stack environment check, 2026-10-11: uv **0.12.22** reports the lock and
installed environment synchronized with no changes needed. Python **3.14.5t**,
Torch **2.14.1+cu130**, torchvision **0.29.1+cu130**, NumPy **2.5.3** and pytest
**9.1.1** pass all **145 tests**. The GIL stays disabled through pytest/Torch
imports but becomes enabled during the tests when `triton._C.libtriton` loads.
Consequently the final GIL assertion in the command below fails despite the
tests passing. This checkpoint does not repeat the earlier examples, lint/type
or compilation gates against the new dependency/tool versions.

Historical Python migration checkpoint, 2026-10-10 (superseded dependency selection):
Python **3.14.5t**, Torch **2.11.0+cpu**,
pytest **9.0.2**, Ruff **0.15.17** and ty **0.0.49**. All **145 tests**, all eight
examples, scoped lint/format and strict typing pass. A CPU
`torch.compile(..., backend="aot_eager")` forward/backward smoke test passes.
The test runner checked that the GIL was disabled before imports, after imports
and after `pytest.main`; the compilation runner checked it before and after
compilation/backward. This is compatibility evidence, not concurrent execution
or performance evidence. Earlier checkpoints below used Python 3.13.13 and
Torch 2.11.0+cu130 on CPU and remain historical records.

To run the tests with separate GIL checks in the same process (this intentionally
fails if a dependency enables the GIL):

```bash
uv run --locked python - <<'PY'
import sys
import sysconfig
assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
assert not sys._is_gil_enabled()
import pytest
import torch
assert not sys._is_gil_enabled()
result = pytest.main([
    "--noconftest", "docs_design/models-strategy-trainer/execution_candidate",
    "tests/unit/training/test_execution.py", "-q",
])
assert not sys._is_gil_enabled()
raise SystemExit(result)
PY
```

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

In this CPU target, `gradients=False` disables gradient recording **and cuts
returned tensor paths**, including identity/pass-through outputs. It does not
mutate the input's gradient flags or claim isolated output storage. Supported
results are tensors, plain tuples/lists/dicts with plain scalar keys, and plain
scalars/`None`; opaque/custom result structures reject rather than certify a
cut. That rejection is after model invocation and may follow effects, not a
harmless pre-execution rejection. Differentiable views keep their result seam.

The target view briefly uses evaluation mode; the conduit records input
gradients even though F's own parameters are frozen. Call-level evaluation
covers each declared underlying participant once, including named-view-only
access, without exposing its raw model through `Context.models`. A view with
`evaluation=False` inherits the enclosing mode rather than forcing training.
Mode restoration preserves mixed child modes on return or failure. Forward
and restoration failures are both retained. These are synchronous view calls:
they never suspend with a changed module mode. This proves neither thread
safety nor asynchronous backend completion.

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
protection. It checks checkpoint-visible state and all registered parameters
and buffers, including nonpersistent buffers, their values and tensor metadata,
registered module types/modes, path-to-object alias relationships, and disjoint
storage. Equal deduplicated parameter counts alone cannot establish member
correspondence: which paths refer to the same member must survive copying.
Invalid copies reject before retained computation and release their source
lease. Origin stamps keep the same participant/binding identity
and old numerical epoch. A revision label, wrapper or copied reference is not
an isolated version. This is a narrow copying target, **not** a proof that
arbitrary custom-module private state, hooks, external resources or RNG-driven
algorithms admit faithful copying.
The equality check fails closed for NaN-containing registered tensor state,
even for a faithful copy. This is a conservative target limitation inherited
from its checkpoint-visible state check, not a universal numerical-policy rule.

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

`test_routed_views.py` now provides 46 cases: direct numerical routing,
separate views, mixed-mode restoration, live versus genuinely separate old
storage, shared versus recomputed work, update eligibility, blocked writers,
pending replacement, partial outcomes, cancellation, borrowed-value release,
source aliases and absence of hot-path graph discovery. Seventeen follow-up
cases cover the four independently reproduced review findings: changed alias
partitions despite equal member counts, changed nonpersistent buffers,
identity outputs escaping a no-gradient cut, and ignored call-level evaluation
through named views. Positive unchanged-copy cases, registered module modes,
nested supported/unsupported view outputs, post-invocation effects, raw-plus-
named deduplication, and combined primary/cleanup failures are included.

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

The follow-up investigation is comparative lowering/specialization of the
**same accepted graph**, preserving its boundaries and selected behavior. It
is now recorded in the comparison below; it does not introduce a second
independently authored run description.

Original follow-up verification: 128 tests passed together (116 candidate cases
and the 12 unchanged execution tests); scoped Ruff lint/format, strict `ty`
with the extra search path above, example execution and diff checks pass.
The required `review-mcp` attempt did **not** produce a verdict: it failed
with provider 429/1308 (five-hour quota exhausted). Subsequent self-checks
closed the falsely accepted unprotected sequence, block-local grant and broad
retained evaluation shapes noted above and reran the gates. That original
checkpoint remained open awaiting required review; it made no review approval,
production readiness, architectural adoption or G5 completion claim.

Correction checkpoint: 145 tests pass together (133 candidate cases plus the
12 unchanged execution tests). Scoped Ruff lint/format, strict `ty` with the
extra search path, example execution and diff whitespace checks pass. The
required `review-mcp` correction review against `HEAD` (`8103760`) returned:
all four defects corrected, failure/lease cleanup verified, no concrete
functional regression. Its two useful non-blocking notes are addressed by the
NaN-copy limitation above and a two-failing-module cleanup test. The affected
checks pass again. Neither these fixes nor that review select production APIs,
establish thread/backend safety, or advance G5.

## Evidence and its limits

This subsection records the original block/producer checkpoint. The later
frozen-view/Sequence investigation above supersedes its limits concerning
multiple views and retained cross-block work; the comparison below supplies
the subsequent local cost evidence. Neither follow-up removes the remaining
backend, authority, concurrency or recovery limits.

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
specialization is the separate comparison below. No broader guarantees are
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

## Same-graph block lowering comparison

This 2026-10-11 experiment compares a small specialization of the existing
target, not a second execution architecture. `prepare(accepted)` retains the
normal `"calls"` path. `prepare(accepted, lowering="inline")` selects the
alternative during preparation. The author does not supply another graph,
callback sequence or set of compatibility rules.

Both preparations derive their work from the **same Accepted description**:

```text
accepted hierarchy, selected implementations and relationships
                              |
                         preparation
                     /                  \
         calls: guarded wrappers    inline: guards in block body
                     \                  /
              shared invocation and failure rules
              unchanged whole-run coordination/state
```

The paired tests give each realization freshly initialized models, owner state
and optimizer state. Those are separate experimental executions for comparison,
not two simultaneously current physical images of one live run. Preparation
publication still uses the existing authority and generation checks.

### What actually changes

The baseline block calls a guarded async wrapper for each piece of work. That
wrapper records entry, invokes the selected work, checks its output and records
return; it also classifies failure and withholds possibly affected state.

The specialized block emits that wrapper shell directly into its generated
body. It removes one Python async-wrapper invocation per work item. The raw
invocation, output rule and failure rule are shared with the baseline through
`Lowering.bind_parts()`, rather than reimplemented as a weaker fast path.
Selected Python/tensor computation is **not** expanded or fused into this body.

Both paths retain:

- Dependency ordering and accepted source addresses, with no hot-path graph walk.
- Leading input-readiness waits outside participant leases, followed by protected
  admission and all-member claim. Work identity and actual provenance survive.
- Live context, owner state, bounded runtime choices, observations and feedback.
- Gradient routing, preparation correspondence and independent unit outcomes.
- Declared effects, protected use, mode restoration, output checks and failure
  withholding. Implementation return is not confused with clean handback.

Together, Repeat, Requested, producer lifecycle, replacement coordination and
Engine are unchanged. The cross-block Sequence emission deliberately uses its
existing path in both modes; identical Sequence source in the tests verifies
that fallback. Retained derivative lifetimes are not silently optimized away.
The new choice is a candidate CPU-target option, not a production authoring API.
Updated Ruff/ty also required explicit strict zips and async-generator annotations
in the touched implementation; these do not add new state or ownership rules.

### Conformance evidence

`test_specialized_lowering.py` adds 25 cases. The comparisons check values,
gradients, modes, owner state, progress, identities/revisions, handoff association,
separately known outcomes and unusable state. They do not demand identical private
callback traces or physical preparation-object identities.

The same-description positive cases cover adaptive diffusion-like work with
independent inputs and requested validation, alternating adversarial actions,
joint units, granted two-pass work, compatible replacement, and retained/live
versus isolated/recomputed views. The latter check the unchanged fallback, not
new Sequence specialization. Authoring-object release and absence of static
wiring discovery are checked in both modes.

Negative/composition cases establish the boundaries that the optimization must
not erase:

- A advances, B mutates then fails, and C remains unattempted. Known A results,
  B uncertainty and C non-entry remain distinct; further unsafe use stops.
- An operation mutates declared state and returns invalid output. Effects remain
  visible, dependent work does not run, and primary/output failure survives a
  separate mode-cleanup failure.
- Incorrect preparation correspondence cannot advance a unit. A returned granted
  region without restored state cannot hand authority back for advancement.
- One inadmissible member prevents claiming either joined input. Cancellation
  releases protection only into withheld, not falsely usable, participant state.
- Independent production publishes key 2 under encoder state 0, changes the
  encoder, then publishes key 1 under state 1. The values happen to match, but
  their identity/provenance does not: only the admissible key 1 contributes;
  demand-end cleanup explicitly discards unused key 2. Bounded capacity, owner
  cleanup and completion relationships remain intact in both lowerings.

The checks are CPU/cooperative asyncio evidence. They do not prove arbitrary
Python conformance, actual thread races, GPU completion or durable recovery.

### Matched local cost

`lowering_cost_probe.py` measures prepared root/block execution, excluding
acceptance, loading, code generation and Engine entry/exit. It compares the
same four cheap operations, guards, live contexts, protected state effects and
bounded diagnostics. A separate tiny joint Torch/SGD case includes numerical
work; it is not a framework-dispatch-only measurement.

The recorded pass uses three fresh sequential processes, 200 warm-up invocations
per path and seven paired samples with alternating order. Cheap samples contain
10,000 invocations; joint samples contain 500. GC is collected before each sample
and disabled during timing. Allocation diagnostics run separately. An initial
trial overlapping the test runner was excluded, not pooled into these results.

| Cheap four-operation block | Calls median | Inline median | Median paired saving | Paired saving MAD |
| --- | ---: | ---: | ---: | ---: |
| Process 0 | 31.42 µs | 29.61 µs | 1.60 µs | 0.45 µs |
| Process 1 | 31.44 µs | 29.72 µs | 1.84 µs | 0.36 µs |
| Process 2 | 30.77 µs | 29.53 µs | 1.24 µs | 0.52 µs |

Paired savings are computed from paired samples, not the difference between
independent medians. This is about **4–6%** of the baseline cheap-block cost,
with the required checks still present. Peak traced memory above the starting
level fell from 5,736 to 5,416 bytes in all three processes; retained memory after
GC was 272 bytes in both paths. These quantities are not cumulative allocation
rate and do not establish an allocation-free hot path.

The joint case showed median paired savings of 10.40, 9.18 and 6.36 µs, but
paired MAD was 1.97, 6.84 and 23.71 µs respectively. That spread is too noisy
to support a general numerical/optimizer speedup claim.

Raw samples, environment, source hashes and measurement scope are retained in
[lowering_cost_results.json](lowering_cost_results.json). The environment was
WSL2 Python 3.14.5t with Torch 2.14.1+cu130, but **the GIL was enabled** after
native imports. No unsafe GIL override was used; no GPU kernels or threaded
execution were measured. Reproduce with:

```bash
uv run --locked python -m docs_design.models-strategy-trainer.execution_candidate.lowering_cost_probe --runs 3 --samples 7 --iterations 10000 --warmup 200
```

### Finding and remaining limits

The same graph can supply a more specialized physical emission without becoming
another independently authored run or losing the tested owner-visible meanings.
The engine does not need a model/algorithm-specific branch for this choice.
Removing a wrapper gives a modest local benefit; it is not evidence that all
coordination cost disappears, or that more aggressive lowering is automatically
valid. Raw async invocations, context/projection allocation and guard work remain.

This comparison does not choose a numerical compiler, establish FSDP/DeepSpeed
integration, specialize cross-block lifetimes, demonstrate threaded safety,
implement restoration, select production APIs or satisfy the full G3.7 migration
gate. More aggressive fusion/capture would need its own evidence for retained
interaction boundaries, differentiation, authority and failure semantics.
No governing design, spec, task checkbox or live Trainer code changed.

Verification: all 170 tests pass together (158 candidate cases plus the 12
unchanged execution tests), all eight examples run, and scoped Ruff lint/format,
strict `ty` and diff whitespace checks pass on the current uv environment.
The expected Triton GIL-fallback warning is not treated as GIL-free evidence.
Required `review-mcp` review against `HEAD` (`33f73a4`) returned with **no
functional regressions found** and no unsupported claims. It verified the
shared guard/failure rules, zero/single/multiple output handling, unchanged
Sequence fallback, publication checks, the new tests and agreement between
the README numbers and raw results. This is review of the scoped experiment,
not architectural adoption or G5 readiness.

Its minor/informational observations delimit the evidence further:

- The probe uses `sys._is_gil_enabled()` in the pinned Python 3.14t environment.
  Availability outside that interpreter profile is not tested or promised.
- `gc_during_timing: false` records the implemented timing policy, not an
  independently observed continuous GC-state trace. The selected fixtures have
  no GC-changing behavior. A future gate must verify its instrumentation too.
- Tracemalloc reports Python-tracked allocations, not native Torch storage or
  allocator growth.
- There is no equivalent direct-Python reference arm here: the comparison is
  calls versus inline, not the full production G3.7 gate. That gate still needs
  its separately required reference and broader conformance coverage.

These do not expose a lowering defect or require expanding this experiment's
target. Documentation format and diff checks pass again after recording them;
runtime code and the measured source hashes are unchanged after review.

## Bounded worker placement and live interaction (2026-10-11)

This experiment addresses a limitation of cooperative execution: a synchronous
selected call runs on the coordinator thread until it returns. The same accepted
graph can now place selected **synchronous CPU calls** in host workers:

```python
ready = prepare(
    accepted,
    lowering="inline",  # "calls" remains supported too
    worker_calls=("compute",),  # addresses of accepted synchronous Calls
    worker_limit=2,
).publish()
await Engine(ready).run()
```

The placement choice is bound during preparation, not rediscovered by traversing
the graph per invocation. It adds no semantic graph construct or second authored
schedule. `Engine` is unchanged. The generated root owns a bounded worker-pool
lifetime; an unpublished attempt owns no running threads. A semaphore limits
submitted/in-flight work rather than relying on an unbounded executor queue.
This experiment's `worker_calls` matches bare work names across active blocks:
the same name in two alternatives/siblings places both. Work names are only
block-locally unique; this is not a final scope-qualified placement API.
`worker_limit` must be a positive integer even when placement is empty.

### Ownership and interaction

```text
Accepted graph -> existing whole-run lowering
                         |
                  coordinator / leases
                         |
                  scoped prepared Context
                         v
                 selected call in worker
                         |
               return + failure/cleanup evidence
                         v
             coordinator / existing run-state rules
```

The coordinator retains admission, participant/owner-state reservations,
invocation association, numerical-epoch accounting, gradient contribution,
optimization, outcome publication and run completion. A worker receives only
the selected call's models/views, owned state, inputs and coordinates—not
`RunState`, the whole strategy or optimizer services. Ordinary Python is still
trusted to conform; this is not a mutation sandbox.

**Work need not finish before other activities interact with it.** The live
interaction test has an independently progressing producer publish a bounded,
thread-safe endpoint through an ordinary accepted input handoff. After the worker
enters, the producer sends a value and receives an acknowledgement while the
call remains active. The selected implementation owns that protocol and decides
where it can service messages. No universal image-specific fields, private
producer scheduling or new message-language construct enter the IR.

That interaction does not permit conflicting current-state access. The block
reservation remains in force through worker execution/cleanup and the block's
remaining work. Another access or compatible replacement waits. Named eval views
also acquire physical exclusion: changing shared module modes must not overlap
another user's access. This reservation does not grant numerical writes or
advance numerical epochs. Preparation rechecks the target's declared lease/source
cycles after adding exclusion; one previously valid cooperative arrangement is
rejected for this worker realization rather than deadlocking silently.

### Cancellation and physical completion

| Observed boundary | Meaning in this target |
| --- | --- |
| Waiting for capacity, or physical future successfully cancelled before start | Selected work did not enter; no effects or cleanup are invented for it |
| Entered work interrupted by coordinator cancellation | Signal a cooperative stop; retain reservation and drain physical execution, including repeated cancellation |
| Implementation returns after cancellation | Preserve the returned fact/effects, withhold uncertain current use and do not certify run completion or advance downstream optimization |
| Worker subsequently fails | Preserve its actual cause and the cancellation evidence; no rollback or automatic retry |
| Invalid output plus failed eval cleanup | Preserve invalid-return evidence and cleanup failures separately; the call was reached and may have effects |

The optional `Context.stop_requested` query exposes a stop request to selected
worker-placed work that can cooperate; it is `None` on the cooperative path.
It cannot terminate an arbitrary native call or force
an implementation to inspect its queue. A non-cooperating call must finish before
its protection can be released; a call that never finishes can prevent shutdown.
Host-thread completion is sufficient only for this synchronous CPU target, not
for asynchronously enqueued GPU/backend work. Python's
[Future cancellation rules](https://docs.python.org/3.14/library/concurrent.futures.html#concurrent.futures.Future.cancel)
are the physical distinction, not an atomic training-update guarantee.
The existing `entered` trace names dispatch entry, which can precede capacity
waiting; it does not prove physical entry. `worker_not_started` disambiguates a
dispatch that never invoked selected work. Surrounding physical-context failure
retains interruption and any selected-return evidence too. The root's explicit
lifetime guard is not stripped by optimization and preserves a primary failure
alongside an invalid-lifetime error; it is a guard on the generated target, not
a new general-purpose recovery mechanism.

Python context variables and the supported CPU numerical context—grad mode,
inference mode and CPU autocast enablement/dtype/cache—are captured for each
submission and restored around execution. They are checked against direct calls
and across reused workers. PyTorch documents
[grad mode](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html) and
[autocast](https://docs.pytorch.org/docs/2.14/amp.html) as thread-local;
copying Python context variables alone would not preserve them.

### Evidence and limits

The new focused tests cover both emissions, real CPU values/gradients and unit
outcomes for adaptive diffusion-like, alternating adversarial and jointly due
optimization arrangements; producer progress/live messages; conflicting access
and replacement; bounded capacity; cancellation before/after entry and repeated
cancellation; late failure; output/cleanup failure; supported context propagation;
and rejection of unestablished realizations. Race checks use explicit barriers
and bounded test cleanup, not elapsed-time speedup assertions.

At this checkpoint, only ordinary selected synchronous `Call` placement was established. Known async
implementations are rejected during preparation. A disguised async return fails
after invocation with possible effects; it does not silently fall back to the
cooperative path. Ordinary async implementations still have that separate path.
At this checkpoint, any active `Sequence` in a worker-enabled arrangement—including
a disjoint one—was deliberately rejected pending cross-block/retained-lifetime
evidence. The composition experiment below supersedes that restriction. Disabled
capability blocks are not active.
Granted two-pass work, backward and optimizer advancement are not moved to
threads. A long coordinator-side backward/optimizer call can still block it.

This is not evidence of arbitrary CPU-module thread safety, RNG isolation,
complete PyTorch TLS capture, task-local numerical context across suspended
coroutines, distributed/GPU completion, performance speedup, recovery or a
production worker API. Pure-Python execution with the GIL can interleave; it is
not parallel Python, and a native call holding the GIL can still prevent other
Python threads from progressing.

Fresh-process observation in the current uv environment: Python 3.14.5t and
Torch 2.14.1+cu130 start this import path with the GIL disabled, but preparing
the numerical joint fixture enables it through `triton._C.libtriton`. The test
run reports the same fallback warning. No unsafe GIL override was used. Passing
on 3.14t, retaining a disabled GIL and safe concurrent execution remain separate
claims. No live Trainer, governing OpenSpec artifacts or G5 checkboxes changed.

Verification before review: 204 tests pass together (192 candidate cases,
including 34 new worker cases, plus the 12 unchanged execution tests). All eight
existing examples run; scoped Ruff lint/format, strict `ty` and diff whitespace
checks pass. This is scoped experiment evidence, not a production migration gate.

The initial required `review-mcp` review against `HEAD` (`dad4b2a5`) read both
new untracked files and the supporting code. It found no mainline correctness
bug and confirmed the lease/drain, attribution, context and physical-exclusion
rules. Follow-up corrections preserve failures outside the selected-call return,
make the lifetime guard explicit, validate empty-placement options, document
global-name matching and rejection timing, and remove a synthetic checking-name
collision without changing the accepted graph. Six additional tests exercise
those edges.

Final verification: **210 tests pass** (198 candidate cases, including 40 worker
cases, plus 12 unchanged execution tests). All eight examples run again after the
corrections; scoped Ruff lint/format, strict `ty` and diff checks pass again.
The six added cases cover outer-context failure after cancellation/selected return
under both emissions, primary-failure preservation by the lifetime guard,
empty-placement option validation, matching names across siblings and checking
without synthetic-name collisions. The original 34 worker cases also passed
three repeated runs before the corrections.

The focused follow-up `review-mcp` review against the same `HEAD` read the new
files from disk, verified every correction, and found **no correctness regression**.
Its sole actionable item was this final verification record, now added. It noted
two safe/conservative edges without requesting changes: an executor-construction
failure can leave an unusable pre-start semaphore until the next lifetime, and
an exceptional numerical-context capture would conservatively withhold state
although selected work did not enter. Neither is a new supported recovery claim.
Only documentation changed after that review; formatting/diff gates were rerun.
The experiment is complete, not a production API selection or G5 milestone.

## Workers and cross-block retained state (2026-10-22)

This bounded composition experiment connects the existing synchronous CPU worker
placement to the existing `Sequence`/`Retain`/`Release` machinery. The accepted
graph, its vocabulary, the root engine, worker-pool lifetime, derivative routing
and optimizer authority do not change. Preparation still chooses placement;
the graph does not itself select host threads.

```python
accepted = Contract().accept(routed_views(recompute=True))
direct = prepare(accepted).publish()
worker = prepare(
    accepted,
    worker_calls=("first-objective", "second-objective"),
    worker_limit=2,
).publish()
```

For this arrangement, worker return and source last use are different boundaries:

```text
retain old state ---- worker forward returns ---- later derivative ---- release
     |                         |                        |                |
     +------------- protected live source or isolated copy -------------+
                independent production can progress alongside
```

### What needed connecting

A source retained from current state already owns a cross-block reservation.
Named evaluation views also temporarily mutate module modes. Acquiring an
exclusive mode reservation only when the worker enters would require upgrading
our own retained read reservation, potentially waiting on ourselves. Omitting
that exclusion would let another reader observe a temporary mode.

Preparation now derives the physical exclusion from the retained calls' declared
views and acquires it with the live source. This target conservatively holds
that exclusion for the **whole retained lifetime**, including gaps between
calls. Later reads and derivative requests in the same region reuse their
existing source protection rather than reacquire it. An isolated copy owns its
own views and storage, so its temporary modes do not reserve the current source.
The physical exclusion grants no numerical-write authority and advances no
numerical epoch. Non-retained calls in a sequence also receive the existing
named-mode exclusion when the worker realization is enabled.

Calls receive the same scoped runtime context and no retention/coordination
handle. They return tensors and ordinary declared results. PyTorch retains the
tape; backward/contribution, advancement, state checks and outcome association
remain on the coordinator. The existing sequence emission still resolves
cross-block values to direct locals. Both `calls` and `inline` configurations
exercise it; this is not new inline specialization of `Sequence`.

The existing worker cancellation path drains physical execution, including
selected cleanup, before returning or raising. Only then can sequence failure
cleanup release remaining retained sources. Release does not erase a ready
gradient, an already returned optimizer outcome, declared owner-state effects
or uncertainty. A cancelled queued call retains its `worker_not_started`
evidence and creates no gradient or numerical-write event; sequence-wide failure
withholding remains deliberately conservative even in that case.

### Evidence and limits

`test_worker_lifetimes.py` supplies 37 composition/conformance cases across both
emission configurations, plus two explicit limitation probes described below:

- The same accepted routed-view arrangement runs directly and in workers, with
  protected current state versus real isolated copies and shared-tape versus
  recomputation variants. Model state, source stamps, coordinates and
  per-unit outcomes agree; gradients match the independently established A=3,
  B=4 oracle. Contribution and advancement entry remain coordinator-side.
- A bounded producer publishes while a worker is still active. A conflicting
  reader and replacement remain blocked after worker return, while A's gradient
  is ready and B still needs earlier state. Last-use release, not thread return,
  permits conflicting access; unused production is disposed at explicit demand
  end.
- Cancellation at the first or later worker, repeated cancellation, late failure,
  invalid output and ordinary computation failure preserve effects and partial
  outcomes and drain leases/owned-state reservations. No dependent B derivative
  or optimizer action runs after the interrupted work.
- Capacity-wait cancellation never enters selected work or invents gradient
  outcomes. Ordinary sequences without retention can also use workers.
- A retained source that needs temporary mode exclusion can still supply its own
  ordinary reads and later derivative destinations. Known nested conflicting
  source acquisitions (including a new isolated copy) and non-retained mode
  upgrades of one's own live source reject during preparation rather than
  deadlock. The loading-counter fixtures confirm rejection precedes loading.

The result supports composition of these existing mechanisms; it does not
require another execution-language construct. It does **not** establish minimal
reservation duration, a general scheduler, arbitrary private wait-cycle analysis,
concurrent use of arbitrary modules, RNG isolation, all PyTorch thread-local
state, parallel Python speedup, or GPU/distributed completion. Backward and
optimizer calls can still block the coordinator. Ordinary Python remains trusted
to obey declared access/effects and to finish its own synchronous work.
Cross-block snapshot/restoration, arbitrary nested/dynamic sequences and the
other previously documented target limits remain unimplemented. The experiment
does not change production APIs, governing specs or G5 checkboxes.

The required `review-mcp` review returned with no blocking defect in the core
composition and verified source correspondence, same-region protection reuse,
physical cancellation drain, mode exclusion and partial outcome preservation.
It identified two further limits, now exercised explicitly rather than hidden
behind a general private-wait disclaimer:

- **Undetected cross-scope reservation cycle at this checkpoint.** Two independent sequences can
  retain X and Y exclusively, then each request a read of the other's source
  before releasing its own. The target then accepted this graph; both
  acquisitions wait and neither release can run. A deterministic barrier probe
  reaches that condition and then cancels, confirming both lifetimes drain.
  Correct cancellation is not evidence of normal progress or safe scheduling.
  This is a visible graph-level access relationship, not merely opaque Python
  behavior. Broader cross-scope reservation readiness remains unestablished.
- **Conservative false rejection.** A sequence can finish a mode-changing call
  before waiting on an input from a producer reading that participant. Its
  per-item reservations do not overlap the input wait, but the inherited
  whole-block lease/source analysis rejects the worker realization as a cycle.
  A companion probe demonstrates the direct execution and the conservative
  preparation rejection. This is a target limitation, not a language rule.

Both were carried into **`sd-scripts-syv.13`**, addressed by the preparation pass
below. This composition checkpoint did not
select a general scheduler or claim those interactions are solved. The two
limitation probes are evidence of gaps, not successful conformance coverage.
The review's reservation-duplication note is documented in the helper: physical
uses coalesce on acquisition. Its optional extra freshness assertion was not
required for correctness under the existing held-source invariants. The reviewer
read the new test file explicitly from disk; it remains untracked until the user
chooses to stage it, like other new experiment files. No staging/commit/push was
performed merely to satisfy the review's process suggestion.

Verification at that checkpoint: **249 tests pass** (237 candidate cases, including 37 new
composition cases and two limitation probes, plus 12 unchanged execution tests).
All eight examples, scoped Ruff lint/format, strict `ty` and diff whitespace checks
pass. The new file passed three repeated processes before review (37 cases) and
again after adding the limitation probes (39 cases). After review, only tests,
documentation and an explanatory code comment changed; runtime behavior did not.
The observed Triton GIL-enablement warning remains: this is CPU
correctness/coordination evidence, not a GIL-disabled or performance qualification.

## Preparation follows access lifetimes (2026-10-22)

The next bounded pass addresses both recorded cases through one CPU-target
preparation analysis. `access.py` derives where each activity can wait, what it
already holds there, what it next requests, and which input producer it awaits.
These are private, preparation-only facts from the accepted graph, not another
authored run, a new execution-language vocabulary or a runtime scheduling loop.
The use helpers are shared with lowering and source acquisition so the check
follows the target's actual reservation boundaries:

| Execution shape | Access held while waiting for input |
| --- | --- |
| Ordinary block's prefix joins | None; prefix readiness precedes its aggregate reservation |
| Ordinary block's later joins | The block's participant and owner-state reservation |
| Sequence join (where admitted by the existing contract) | Its own per-item uses plus any live retained sources; prior completed calls have released their per-item access |

Live retentions last until explicit release. Isolated copies request their source
only while copying, then retain no current-source lease. Physical evaluation-mode
exclusion remains distinct from numerical-write authority. Joint acquisition is
all-or-none: its requested resources are not falsely counted as already held.
Participant resources and owned-state resources use separate keys.

Preparation compares only independently progressing activities. Ordered sequence
blocks and alternative blocks of one repeated activity are not simultaneous
waiters. Read/read overlap is compatible; an edge needs an actual read/write or
write/write conflict, or a declared input-production dependency. Candidate wait
cycles use at most one phase of each activity. Acyclic tails are removed before
searching phase-compatible cycles; this is cold-path analysis, not a production
scalability or hot-path-cost qualification.

The result is a bounded readiness decision, **not automatic rescheduling**:

- The original mutual wait now raises `NotReady` during preparation, before
  participant loading or owner-state initialization. The diagnostic names the
  waiting work around the cycle. No hidden serialization, copy, early release,
  retry or changed algorithm is introduced to make it run.
- The recorded evaluation-view-then-input sequence prepares and executes under
  both CPU emission configurations, with the same accepted root and observations
  as direct execution. Its finished view call is not held across the later join.
- Two-/three-scope cycles and a cycle combining a retained model, an owned-state
  lock and a producer input are rejected. Compatible readers, one-way conflicts,
  release-before-request, real isolated copies and ordered non-simultaneous
  phases remain executable. A prefix join under a sequence is distinguished
  from an ordinary block prefix rather than assumed lease-free everywhere.

`test_access.py` adds 19 cases; the two former limitation probes in
`test_worker_lifetimes.py` are now rejection/execution regression tests. No
`Engine`, accepted IR, worker-drain protocol, numerical formulas, derivative or
optimizer ownership, live Trainer, governing spec or G5 checkbox changes.

This does not prove general liveness. Producer-internal acquisition/queue
protocols, selected Python's hidden waits, backpressure while privately holding
access, worker-capacity dependencies inside private protocols, dynamic backend
completion and arbitrary resource scheduling are not inferred. Producer source
uses are a conservative declared envelope, not an inspection of its algorithm.
The pre-existing contract's other probe-target restrictions are unchanged; this
pass replaces the worker preparation recheck, not the eventual full contract.
At this checkpoint, the contract still rejected a sequence join after any
retention, even after release. The repeated-region extension below admits
joins after release; its access analysis retains the actual lease boundaries.
There is no new universal scheduling policy, GIL-disabled execution, GPU safety,
recovery or performance claim.

Verification refreshed on 2026-10-23: **269 tests pass** (257 candidate cases and
12 unchanged execution tests); all eight examples, scoped Ruff lint/format,
strict `ty` with the documented extra search path, and diff whitespace checks
pass. The 59 focused access/lifetime cases pass in three separate processes.

## Repeated multi-block regions and accumulation (2026-10-23)

`accumulation()` in `examples.py` connects an independent input producer with
a live repeat policy whose alternatives are complete multi-block sequences:

```text
Together
|-- input producer: bounded, addressed microbatch channels
`-- learning: accepted live policy
    |-- two-contribution sequence
    |   |-- input -> retain -> objective -> contribute -> release
    |   |-- input -> retain -> objective -> extend contribution -> release
    |   `-- clip/advance/reset the completed window
    `-- three-contribution sequence: one additional contribution block
```

The alternatives are prepared once, not concurrent siblings. Feedback from
selected computation changes the next policy choice: the example selects a
two-contribution window then a three-contribution window; zero-loss feedback
instead selects the two-contribution alternative again. The normal engine is
unchanged. No algorithm-specific loop is hidden inside an ordinary operation.

### What the composition required

- `Repeat` can select a prepared `Sequence` as well as a `Block`. Each sequence
  invocation has fresh locals and a distinct execution-region correspondence.
  The selected work key and payload reach all of its blocks. Per-block progress
  remains separate from successful whole-region completion, which can trigger
  capability work. Capability invocations may interleave; their association is
  not inferred from a guessed global invocation order.
- `Differentiate.accumulate` explicitly links the preceding contribution into
  the next one. Acceptance checks one unbranched chain per unit, matching
  destinations, and final advancement from the latest contribution. This target
  still permits one contribution per unit per block and one final advancement
  per unit per sequence. It does not implicitly sum unrelated losses.
- The optimization owner accumulates independently stored gradients and checks
  the current window, originating invocations, prepared members, execution
  region and destination-state stamps. An old or copied token, foreign region,
  missing gradient or intervening destination write cannot authorize a later
  contribution/update. Normalization remains selected mathematics; the engine
  does not guess averaging from a microbatch count. Clipping is applied once
  at the declared completed window, before its optimizer call.
- A later microbatch may retain numerical state while its region owns a verified,
  unchanged pending window. Its preceding source lease must be released before
  another input wait; a join with a live retention still rejects. Releasing a
  source does not erase pending gradients or permit replacement/publication.
- The preparation access analysis attributes the sequence's phases to its repeat
  activity, not to independently progressing microblocks. Existing cross-activity
  cycle rejection therefore still applies without treating ordered alternatives
  as simultaneous owners.

Pending-window outcome records are retained independently of the bounded
completed-history deque. A 36-contribution failure test verifies that diagnostic
history eviction cannot discard live origins. When a joint advancement returns
for A but fails inside B, all contributing origins retain A's returned/reset
outcome and B's uncertain/unattempted outcome separately. There is no atomic
update, rollback, parameter-change detection or safe automatic retry.

### Evidence and limits

`test_repeated_regions.py` adds 37 cases. Direct PyTorch gradients/weights agree
with the connected example under both CPU emissions, with and without selected
worker calls. Other cases cover genuinely different feedback choices, mixed
state-only sequence/block alternatives without input or optimizer placeholders,
independent production between contributions, complete-window clipping,
capability request identity, malformed/incomplete chains, missing later input,
intervening writes, repeated worker cancellation/drain, partial advancement,
long-window outcome retention and preparation-time wait-cycle rejection.
Missing gradient storage and failed pre-window evidence also prevent extension,
advancement, replacement/publication or capture. Sequence bodies currently share
one emission path under both lowering flags; these runs qualify compatibility
with both configurations, not independent sequence-specialization performance.
Static discovery helpers are disabled after preparation in the numerical test;
changing policy choices do not reconstruct or rediscover either alternative.

This extends the existing hierarchy; it is not arbitrary recursive composition.
The repeated alternatives are finite, predeclared blocks or sequences of blocks.
It does not add nested repeats inside a sequence, dynamically grow an open
window, carry a window between repeat iterations, or generalize unit routing
mid-window. Destination freshness uses conservative participant-level numerical
epochs rather than fine-grained parameter versions. Backward/advancement remain
coordinator-owned CPU work; GPU/distributed synchronization, native free-thread
safety, higher-order gradients and durable continuation are not established.
The earlier private-provider/backpressure and backend limits remain.

Verification: **306 tests pass** (294 candidate cases plus 12 unchanged execution
tests), and all nine examples execute. Scoped lint, formatting, strict types
and whitespace checks pass. This remains isolated candidate evidence; no live
Trainer, governing requirements or G5 task status changes.

## Integrated pending-work ownership (2026-10-24)

The accumulation investigation exposed parallel bookkeeping: a pending
invocation, a current contribution and live outcome records were independently
updated by ordinary backward and by the two-pass grant. These now belong to
one private pending-work lifetime per optimization unit:

```text
idle -> entered/unverified -> verified handback -> advancement claimed
                                                   |
                                  reached mechanics recorded per unit
                                                   |
                                  all retained mechanics returned -> idle
```

An explicit further contribution extends an unchanged verified window before
advancement. Any incomplete/failed entry or advancement stays unfinished;
there is no automatic reset, reopening or retry. In a partial joint failure,
even a unit whose optimizer and reset returned retains its live evidence until
the whole selected advancement completes or an unimplemented recovery policy
resolves it. Claiming work is not evidence that a backend call was attempted.

Both ordinary backward and the granted executor register entry through the
same optimization-owner operation before doing work. The grant still owns
only its accepted two-pass phases and proves restoration before the owner
accepts its handback; it no longer writes runtime readiness or outcome fields.
Both paths publish their verified contribution through the same owner. Final
clipping, optimizer advancement and reset remain retained mechanics.

The old `pending`, `window` and `active_outcomes` attributes are derived
diagnostic projections, not three writable sources of truth. Live records
remain separate from bounded historical diagnostics; reports copy owner facts
rather than changing them. Retention asks whether an unfinished window has a
valid same-region continuation. Publication, replacement and capture use the
same owner-held unfinished-work condition, including failed work that never
produced a ready gradient. Final advancement claims every selected unit before
clipping/backend callbacks, so re-entry or an early failure cannot reuse the
old readiness. The reached optimizer/reset outcomes remain independently
reportable, without numerical-change detection or atomicity claims.

`test_optimization_ownership.py` adds 26 cases covering both entry paths and CPU emissions,
unverified entry, duplicate handback, failed extension, read-only diagnostics,
clipping/optimizer/reset failure, backend re-entry and partial joint evidence
after history eviction. A grant can progress while an independent unit has
pending gradients; wrong-invocation handback and repeated failed-grant entry
reject without erasing that work. Existing connected cases still supply the numerical,
worker cancellation/drain, independent-production and dynamic-policy evidence.
This is an internal lifecycle integration, not a new graph construct, public
authoring API, numerical algorithm, scheduler or production backend. All earlier
composition, recovery, GPU/distributed and performance limits remain.

Verification: **332 tests pass** (320 candidate cases and 12 unchanged execution
tests), all nine examples execute, and scoped lint, formatting, strict types
and whitespace checks pass.

## Integrated current-use ownership (2026-10-24)

Lowering now binds access requirements and calls the state owner's `use()`
boundary. It no longer scans active retentions, decides which leases cover a
read, merges borrowed projections, or enumerates remaining source lifetimes
during cleanup. The owner validates and supplies those projections, reserves
only uncovered participants/state, and drains the region's remaining sources.

Borrowing a protected current read does not acquire another lease or release
the original one. New retentions have independent lifetimes; nested read/read
reservations remain possible, while self-upgrades or nesting under an own
writer reject before waiting. Real isolated copies cover no current-source
access after copying. Borrowing checks source freshness/usability before and
after any wait for uncovered members, so it cannot supply a source released or
withheld during that wait. A different region cannot borrow another region's
exclusive source. Physical mutable-mode exclusion remains distinct from the
selected operation's numerical-write authority and numerical epochs.

Provider source access is owned by the same state protocol: runtime requests
can select an allowed read-only subset, but cannot expand participant, write
or view authority. Successful returned-write evidence is filed before its
protection ends; failed writes remain unavailable without a false returned
epoch. The provider still owns its queue, private workers and continuation.

Preparation and execution share ordinary-block reservation/prefix derivation,
participant access merging, per-work sequence uses, and own-region reuse/
conflict rules. The coarse lease/source check formerly in `Contract` is gone.
One preparation check runs before participant loading and owner initialization
for both cooperative and worker targets, using their actual mode-exclusion
policy. Target conflicts now raise `NotReady` there; explicit scope-completion
relationships are still checked during acceptance. No chosen order is changed
and no runtime scheduler is added.

Two concrete consequences are executable: a sequence can finish a declared
write before waiting on a producer that needs the model, and it can finish a
model use before publishing replacement later in the same block. The ordinary
aggregate-block versions still reject where their own lease would prevent
completion. The same accepted mutually retaining graph can execute with
cooperative views yet reject a conflicting worker realization before loading.
These are target/lifetime differences, not changed training algorithms.

Region cleanup attempts all of its own sources even if one release fails,
preserves primary and cleanup failures separately, and leaves other owners'
lifetimes alone. Failed releases remain tracked rather than pretending their
protection ended. Closing claims the token before the first release await, so
neither an in-flight close nor a failed close can lend protection to later work.
A failed or cancelled physical release leaves the token closing and withholds
its current-source participants, whether physical release happened or not.
Waiters reject that unavailable state even if its physical lease remains held;
replacement also rejects a closing retained lifetime rather than waiting forever
on its uncertain physical protection. Unrelated participants stay usable. No retry or recovery of an uncertain close
is implemented. Existing worker tests retain the physical-future drain before
release, including repeated cancellation. Worker capacity ordering is unchanged.

The cold check is not a promise of runtime publication readiness. Replacement
still requires a globally finished retained/contribution boundary. In
particular, a foreign isolated retention holds no current-source lease, but
replacement rejects while that derivative lifetime remains live; this target
does not wait for or migrate its continuation. A sequence-Replace regression
checks that rejection in both lowerings without changing the installed image.
Owner-state continuity and unfinished optimization remain runtime guards too.

`test_current_use.py` exercises these positive and negative boundaries alongside
the existing numerical, producer, accumulation and worker cases. This pass does
not prove global liveness: hidden Python waits, private producer queue/worker
protocols and worker-capacity dependencies remain outside the declared analysis.
Borrowed calls rely on the accepted region's awaited, sequential lifetime;
concurrent direct retain/use/release calls for the same region are unsupported.
This is not an arbitrary-Python sandbox or support for unjoined background
users. GPU/device completion, distributed coordination, native free-thread safety, durable
restoration and hot-path performance remain unqualified. No live Trainer,
governing requirement or G5 task changes are made.

In particular, this pass does not qualify mutable-model-mode protection when a
provider owns native workers while selected calls use cooperative placement.
That needs a separate test of the physical access policy across both owners;
the selected-call worker tests do not establish it.

Verification: **374 tests pass** (362 candidate cases and 12 unchanged execution
tests), including 42 current-use cases. All nine examples execute; scoped lint,
formatting, strict types and whitespace checks pass.
