# xDSL workflow experiments

This isolated research package implements **B, A and C** of the plan
recorded in `sd-scripts-syv.31.8`, as `sd-scripts-syv.31.9`,
`sd-scripts-syv.31.10` and `sd-scripts-syv.31.11`. It tests xDSL as the
actual semantic and prepared IR infrastructure for a system that can represent
data and model workflows. Production code, governing OpenSpec requirements and
framework selection remain unchanged. B's data-only experiment is described
below; [A's owner integration](#a-correlated-data-model-work-and-evaluation)
adds bounded training/evaluation evidence, and [C's granted authority](#c-offered-two-pass-authority-and-retained-advancement)
checks a selected two-pass region. Fusion remains a later handoff.

The Python author in [fixture.py](fixture.py) constructs nested xDSL operations:

```text
   cursor + current policy
            |
         selector -> composed data [double -> local [offset]]
            ^                            |
            |                       bounded mailbox
            |                            |
     policy application <--- receipt <- scorer -> publisher
            |                (validated publication acknowledgement)
      feedback acknowledgement

   stop: close selection -> settle selected work -> drain calls -> release
```

The selected `scalar-feedback-v1` extension fixes a tiny algorithm: select key
`k` at the actual policy revision with value `k + bias`; map `2*x`, then `x+1`;
score by squaring; publish and validate the returned receipt; update policy
bias to `score % 5` and increment its revision. Source revision is zero for
this synthetic source. These are fixture semantics, not universal workflow
rules. Source IDs, entity references, endpoint types and the feedback/stop
contracts appear in structural IR. Block arguments/yields expose recursive
local composition; an SSA endpoint names a stream carrying many tickets.

The accepted horizon is **two selected tickets awaiting feedback
acknowledgement**, including selected calls, pending handoffs, buffered work and
pending control. The scalar target chooses one mailbox slot, retained through
scoring, publication and feedback acknowledgement. Its dedicated control task
can apply feedback while the producer is blocked publishing into that slot.
No lock or resource borrow is held across a blocked publication.

`stop()` is an explicit external session action. It closes selection and lets
already selected work settle; it is independent of a learning clock. On a
selected failure, selection closes, already acknowledged work stays
acknowledged, the failed ticket remains failed, and work the failed consumer
can no longer publish is abandoned after physical uses settle. Cancellation
also closes selection and drains; it remains cancellation even if selected
work subsequently fails. No rollback is inferred.

The implementation uses three boundaries:

| Boundary | Implementation and evidence |
|---|---|
| Structural and semantic acceptance | [dialects.py](dialects.py) supplies IRDL schemas and types; [prepare.py](prepare.py) adds the selected extension's wiring, identity, entity-use, local-chain and lifecycle checks. The result is detached canonical xDSL text and deferred obligations. |
| CPU preparation | Resolves versioned symbolic implementations and checks their roles/effects; emits `exec` recipes with owner references, source ancestry, call chains, handoff and feedback/completion requirements. Runtime dispatch tables are extracted from that prepared xDSL. Horizon three is semantically valid but unsupported by this target. |
| Live execution | [runtime.py](runtime.py) owns cursor/policy, aggregate admission, keyed feedback, mailbox and physical-use lifetimes. Payloads, futures and live callables are outside the semantic IR. Runtime facts are observations, not another executable program. |

The structural document and versioned binding manifest reconstruct and execute
in a fresh process with the authoring builder disabled. Preparation resolves
wiring once; current policy, ticket admission, receipts and physical completion
remain live checks. No authored Python closure is serialized.

The binding ABI is `int -> Awaitable[object]`, followed by exact integer output
validation; publication additionally validates an equal returned receipt.
A binding must return or raise only after its own physical uses and cleanup
finish. The coordinator shields selected service lifetimes from cancellation.
Tests use real host threads to check that stop/repeated interruption cannot
release them early. Arbitrary detached work behind a callback is outside this
ABI. The scalar runtime and its control API are confined to one event loop.

The previous candidate's `state.Feed` and `Workers` import Torch and depend on
its larger runtime types. B therefore implements the new small data protocol
without restructuring them. It carries forward their addressed association,
post-invocation validation and physical-drain principles, with independent
checks. A now exercises participant/optimization ownership through an optional
domain adapter, described below. There is no Torch import, dummy optimizer, training
result, numerical compiler or general scheduler in this data-only path.

## Run and verify

From the repository root, use the existing uv-managed Python 3.14t interpreter
with an isolated environment. The compiled requirements pin xDSL **0.74.0** and
all experiment/test dependencies; no project dependency or lockfile refresh is
needed.

```bash
uv venv /tmp/sd-scripts-xdsl-b-env --python .venv-wsl/bin/python
uv pip sync --python /tmp/sd-scripts-xdsl-b-env/bin/python \
  docs_design/models-strategy-trainer/xdsl_candidate/requirements.txt

/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.xdsl_candidate \
  --dump-dir /tmp/xdsl-b
/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.xdsl_candidate \
  --load /tmp/xdsl-b/accepted.mlir --manifest /tmp/xdsl-b/bindings.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/sd-scripts-xdsl-b-env/bin/python -m pytest \
  docs_design/models-strategy-trainer/xdsl_candidate -q -p pytest_asyncio.plugin
/tmp/sd-scripts-xdsl-b-env/bin/ruff check docs_design/models-strategy-trainer/xdsl_candidate
/tmp/sd-scripts-xdsl-b-env/bin/ruff format --check docs_design/models-strategy-trainer/xdsl_candidate
/tmp/sd-scripts-xdsl-b-env/bin/ty check --python /tmp/sd-scripts-xdsl-b-env/bin/python \
  --extra-search-path docs_design/models-strategy-trainer \
  docs_design/models-strategy-trainer/xdsl_candidate/{dialects,prepare,runtime,fixture,conformance,__main__,test_ir,test_runtime}.py
```

The driver asks for stop after four feedback acknowledgements. Already selected
work settles, so the final ticket count may be larger than four; incidental
scheduling is not a contract. CLI output records the actual GIL state and
separates acceptance, preparation and complete session elapsed time. The last
includes startup/drain and is not a steady-state throughput measurement.

## Evidence and present limits

The first B implementation passes **83 tests**, Ruff checks/formatting and ty.
Tests cover canonical reconstruction, author detachment, rejected wiring and
symbol/identity/type contracts, unsupported targets and incompatible bindings,
actual policy provenance, bounded progress under blocked publication, slow
scoring, partial failure, invalid returned output, duplicate/premature feedback,
stop and repeated cancellation during physical work, and retained facts with
zero diagnostic history. [conformance.py](conformance.py) has no framework or
runtime imports: it independently checks this fixture's arithmetic, associations,
policy transitions, aggregate bound and required causal lifetimes. Twelve
deliberately corrupted reports and a shared incorrect implementation demonstrate
checker sensitivity beyond agreement between reconstructions.

The [required review](evidence/review.md) found no blocking correctness or
ownership issues. Small follow-up corrections enforce the loaded horizon in the
CLI oracle, check prepared chain roles and binding/completion descriptors, report
the installed xDSL version and reject invalid stop counts before execution.
The affected checks pass after those corrections. The review's defensive notes
about arbitrary internal service faults and private test probes do not extend
this fixed target's qualification to a general runtime.

[evidence/](evidence/) retains the accepted/prepared IR, binding manifest, one
execution report and qualification/source metadata. The live associations are
bounded by the horizon; the experiment retains completed facts for its externally
bounded test recorder. It does not claim bounded archival memory for an
indefinitely running production service, storage durability or restoration.

xDSL concretely supplies typed operations/attributes, region/block structure,
SSA use tracking, isolation/terminator checks, parsing/printing and cloning.
Our extension still supplies semantic acceptance, symbolic binding policy,
target matching, domain/progress protocols and independent conformance. Generic
memory-effect traits do not establish workflow purity or removability; even
`pure` scalar calls may fail. No generic DCE or fusion pass is applied here.

This makes xDSL promising for representation and diagnostics in B. C's granted
authority integration is described below; the planned semantics-preserving
fusion experiment remains open. Python 3.14t execution and observed GIL state do not establish
arbitrary extension concurrency safety, GPU/distributed interoperability,
production performance or a final framework choice.

## A: correlated data, model work and evaluation

A is an optional extension, with the same `wf.session`, entity declarations,
typed endpoints, recursively composed local data stages and accepted snapshot
boundary as B. [model_dialect.py](model_dialect.py) and
[model_contract.py](model_contract.py) add model use, atomic correlation,
contribution windows and requested evaluation. Authoring, acceptance and text
reconstruction need no Torch import; numerical implementation is selected later.
The common acceptance code has an explicit installed-contract hook and retains
no author or contract callback in the accepted document.

```text
finite source 1..6 -> exclude 3,6 -> fanout
                                    |-- composed [double -> local [offset]] --|
                                    |-- protected frozen encoder -------------|-> atomic correlation
                                                                                  |
                                                    two contributions -> advance -> completion request
                                                                                    |
                                                                       current-at-start evaluation
```

The frozen encoder starts at 1.5 and produces `1.5*k/4`. The other path produces
`2*k+1`. The learner starts at .8; each loss is
`(w*(left+right)-k/2)^2/2`. Samples `(1,2)` and `(4,5)` form two explicitly
normalized accumulation windows, each followed by one SGD update at .05.
The first completed window requests evaluation; its actual protected start may
observe learner epoch 1 or 2. Reported request input membership remains the
first window even when the evaluator uses epoch 2. Triggering, starting and
physical completion are distinct facts.

[model_adapter.py](model_adapter.py) lowers the prepared xDSL into the existing
candidate's `Run`/`Producer`/`Sequence`/`Requested` records and invokes **one
existing Engine**. Those records are generated backend artifacts. A reuses its
participant/current-use owner, source leases, atomic `Feed`/`join`, retained
input links, ordinary contribution/SGD ownership, evaluation mode protection and
physical `Workers` drain. No existing candidate or production code is changed.
The composed producer's two paths are visible in accepted IR; its generated
callback is a target artifact, with private TaskGroup/worker placement.

The prepared IR retains a verified semantic subtree, resolved versioned call
bindings, completion rules and explicit sample/window/slot address recipes.
The latter make this backend's window-key limitation inspectable. Physical
port identity, logical sample, slot, branch, scalar contract and actual current
source stamps are checked for every member before `join` claims either one.
Runtime dispatch and the address mapping are extracted and checked from the
prepared xDSL. The opaque backend participant IDs are normalized to declared
entity names only in portable observations; admission uses the real owner stamps.

This target supports at most four statically selected samples and two
contributions per window. Known finite source keys and authoritative exclusions
are resolved before execution; their count bounds outstanding logical keys by
the declared source horizon. This is not a dynamic filtering/prefetch algorithm.
Other positive full-window counts are semantically valid and fail target
preparation. Stop/failure drains entered physical uses and closes unadmitted
ports against further claims. Unconsumed records remain inspectable in this
bounded failed run; an incomplete contribution stays owned and cannot become an
advancement. The report's `released` flag covers physical protection/producer
lifetimes, while `pending` separately reports unfinished numerical ownership.
The optional adapter distinguishes external cancellation from cancellation of
peer work caused by another failure, including worker work that never entered.

With the repository's existing Torch environment:

```bash
uv run --no-sync --with-requirements docs_design/models-strategy-trainer/xdsl_candidate/requirements.txt \
  python -m docs_design.models-strategy-trainer.xdsl_candidate.model_demo --dump-dir /tmp/xdsl-a
uv run --no-sync --with-requirements docs_design/models-strategy-trainer/xdsl_candidate/requirements.txt \
  python -m docs_design.models-strategy-trainer.xdsl_candidate.model_demo \
  --load /tmp/xdsl-a/accepted.mlir --manifest /tmp/xdsl-a/bindings.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync \
  --with-requirements docs_design/models-strategy-trainer/xdsl_candidate/requirements.txt \
  python -m pytest docs_design/models-strategy-trainer/xdsl_candidate -q -p pytest_asyncio.plugin
/tmp/sd-scripts-xdsl-b-env/bin/ty check --python .venv-wsl/bin/python \
  --extra-search-path docs_design/models-strategy-trainer \
  --extra-search-path /tmp/sd-scripts-xdsl-b-env/lib/python3.14t/site-packages \
  docs_design/models-strategy-trainer/xdsl_candidate
```

The A handoff passed **158 tests** (83 original B, 17 model semantic tests,
58 optional owner-integration tests). Its isolated environment had **100 pass
and one optional module skip**. Actual GIL state is recorded, never forced.
The A suite covers both existing CPU emissions, evaluation on/off a worker,
changed production order, zero diagnostic history, missing/bad/swapped/stale
second-member rejection without partial consumption, retained active input facts
after completed-history eviction, invalid post-entry output, real physical
cancellation/failure drain, trigger/completion separation and fresh-process
reconstruction. Twelve corrupted reports and a shared incorrect loss prove
sensitivity of the independent [A oracle](model_conformance.py).

[A evidence](evidence_a/) retains its IR, plan, bindings, observations and
qualification metadata. B's original evidence is the historical B snapshot
committed as `2131f31d`; A records current shared-source hashes separately.
The [interactive OMP code review](evidence_a/review.md) passed and independently
reran all 158 tests after `review-mcp` failed with a provider usage-limit error.
A small follow-up reads normalization from decoded IR instead of a target
literal; the affected gates pass after that correction. Broader architecture
viewpoints remain deferred until C and fusion have results.

## C: offered two-pass authority and retained advancement

[grant_dialect.py](grant_dialect.py) and [grant_contract.py](grant_contract.py)
add a model-only extension to the same session/snapshot infrastructure. Its
typed `!grant.contribution<@weights>` SSA edge connects one granted region to
one retained advancement. The phase body declares the exact offered protocol:
`backward -> temporary_edit -> intermediate_reset -> backward -> restore`.
The selected region receives those services; the standard owner retains
clipping, the final optimizer call and final reset. Unsupported authority splits
or phase protocols reject semantic acceptance. An accepted request can still
fail preparation when the concrete grant is not offered or its bindings and
parameter correspondence cannot be resolved.

```text
model .8 -> offered write grant [two-pass binding]
                        |
              return + actual handback verification
                        |
           verified contribution for @weights
                        |
             retained [clip? -> SGD -> reset]

raise / incomplete / interruption -> drain physical uses -> retain unverified work
```

C carries no dummy flow horizon, mailbox capacity or feedback field. Those
properties are optional in the shared structural container and remain required
by the A/B semantic contracts that use them. C's source IDs, subject/unit
references, delegated/retained authority, five phases and typed handback survive
printing, parsing and preparation without importing Torch. The prepared recipe
retains the checked semantic subtree, concrete target, owner protocol, versioned
bindings, source ancestry and physical-return/verification requirement. Its
dispatch tables and final step settings are extracted from the reparsed plan;
tests deliberately corrupt those descriptors and require rejection.

[grant_adapter.py](grant_adapter.py) selects the maintained numerical callable
from `execution_candidate.examples.two_pass` unchanged. It discards that example's
authored `Run`, generates a `TwoPass -> Advance` backend artifact from prepared
xDSL, and invokes one existing Engine. Participant/current-use protection,
`GrantedAccess`, pending contribution ownership, actual handback verification,
clipping, functional SGD and reset remain existing services. No production or
legacy candidate source changes are needed. The test-only `selected` facts
observe numerical return and phase history; execution never consults them to
authorize an update. A static SSA edge likewise conveys no live readiness.
This single-member target derives concrete unit membership from the granted
subject and advancement parameter; it has no general unit-member schema.

For `(2*w-1)^2` at initial `.8`, the first gradient is `2.4`; perturbing by `.1`
gives temporary weight `1.04` and second gradient `4.32`. Restoring `.8`, then
applying retained SGD at `.05` gives **`.584`** with a cleared final gradient.
A retained clipping bound of `1` instead gives approximately **`.75`**. The
stdlib-only [C oracle](grant_conformance.py) derives these results without the
numerical binding or owner implementation and checks phase/source attribution,
owner outcomes and final release. A shared no-op optimizer passes agreement
between both CPU emissions but fails this independent oracle.

| Observed case | Required owner outcome |
|---|---|
| Complete, restored region returns with final gradient | `granted_ready`, one retained step/reset, no pending work |
| Region restores and raises, returns incomplete, loses its gradient, or edits after restore | `unverified_grant`, no attempted optimizer/reset, pending work and unusable subject remain |
| Restored region is suspended or physical work is still running | Conflicting current use remains blocked; even a correctly addressed fabricated contribution cannot advance |
| Repeated cancellation, with successful or failed physical return | Protection remains until actual drain; interruption and any physical failure survive, with no verified handback |
| Retained optimizer or reset raises after effects | Verified handback remains distinguishable from uncertain advancement/reset; pending evidence survives and cannot authorize retry |

The complete suite passes **245 tests**, including **83 C tests** (23 semantic,
60 optional numerical/owner tests) and four post-review A plan-bound regressions. The environment without Torch passes
**123 tests and skips two optional modules**. The combined numerical suite's
actual GIL state is false before and after execution. Separately, **39 existing
owner/optimizer/grant checks** pass; the standard-optimizer comparison there
loads Triton and enables the GIL, so that older oracle is not evidence of
GIL-disabled execution. Ruff, formatting and ty pass. [C evidence](evidence_c/)
records the exact snapshots, success/failure observations, qualification and
required correctness-review status. The [C review](evidence_c/review.md) found no
blocking correctness defects. Small follow-ups derive C's report parameter from
the decoded declaration and recheck A's horizon/capacity after plan decoding;
all 245 tests and static gates pass afterward. A's evidence remains the historical A
snapshot committed as `6d103b14`; C records the changed shared-source hashes.

The standalone physical-failure-after-cancellation probe also printed an
asyncio shield diagnostic. A [stdlib-only reproduction](evidence_c/shield_diagnostic.json)
on this installed Python 3.14.5 shows the same exception-handler message even
when another drain observes the exception. The required review found no C
ownership defect: the failed physical use and interruption remain in the report,
and protection ends only after actual drain. No older worker-service change is
included here.

```bash
uv run --no-sync --with-requirements docs_design/models-strategy-trainer/xdsl_candidate/requirements.txt \
  python -m docs_design.models-strategy-trainer.xdsl_candidate.grant_demo --dump-dir /tmp/xdsl-c
uv run --no-sync --with-requirements docs_design/models-strategy-trainer/xdsl_candidate/requirements.txt \
  python -m docs_design.models-strategy-trainer.xdsl_candidate.grant_demo \
  --load /tmp/xdsl-c/accepted.mlir --manifest /tmp/xdsl-c/bindings.json
```

C strengthens the representation case: a nonstandard numerical region and its
retained owner obligations are inspectable, typed and reconstructable without
making common infrastructure dispatch on a training algorithm. It also exposes
the division of work: xDSL does not offer or verify live grants, prove binding
arithmetic, own parameters, or drain private threads. The selected binding must
join private physical work before returning or raising; tests use the existing
`Workers` service for that boundary. Grant services restrict available operations
within this cooperative ABI; they do not form a Python security sandbox. Phase
ops describe a fixed protocol, with numerical execution behind a versioned
binding, rather than a general differentiable numerical IR.

B uses its small scalar realization; A and C each reuse the existing owner
engine. These results do not yet demonstrate a single dynamic feedback/model
run. Fusion, broader combined-runtime behavior, GPU/distributed execution,
durable restoration and production performance remain unqualified. Framework
selection, including whether custom machinery is preferable, depends on the
remaining evidence. Outside architecture viewpoints remain deferred until the
planned fusion comparison is complete.
