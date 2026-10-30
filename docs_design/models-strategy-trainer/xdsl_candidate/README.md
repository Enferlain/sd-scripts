# xDSL workflow experiments

This isolated research package implements **B, A and C** of the plan recorded
in `sd-scripts-syv.31.8`, as `sd-scripts-syv.31.9`, `sd-scripts-syv.31.10`
and `sd-scripts-syv.31.11`. Later sections record local fusion, a mixed ownership
workflow and an independent local operation/schema extension. It tests xDSL as the
actual semantic and prepared IR infrastructure for a system that can represent
data and model workflows. Production code, governing OpenSpec requirements and
framework selection remain unchanged. B's data-only experiment is described
below; [A's owner integration](#a-correlated-data-model-work-and-evaluation)
adds bounded training/evaluation evidence, and [C's granted authority](#c-offered-two-pass-authority-and-retained-advancement)
checks a selected two-pass region. The [local fusion comparison](#local-data-fusion)
now tests one conservative transformation of prepared dispatch.

Latest combined qualification: **471 research tests pass** with the GIL
disabled before and after the full run; the Torch-free environment passes
**261 tests** with three optional backend modules skipped. Four milestone
reviews and the returning-filter/contract correction review approved the work;
final validation is recorded in [the current qualification](evidence_progress/qualification.json).

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

The original scalar fixture uses `int -> Awaitable[object]`, followed by exact integer output
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
execution report and qualification/source metadata. Base B's checked-in
`evidence/bindings.json` is historical schema 1; use a newly generated schema-2
manifest with current code. The live associations are
bounded by the horizon; the experiment retains completed facts for its externally
bounded test recorder. It does not claim bounded archival memory for an
indefinitely running production service, storage durability or restoration.

xDSL concretely supplies typed operations/attributes, region/block structure,
SSA use tracking, isolation/terminator checks, parsing/printing and cloning.
Our extension still supplies semantic acceptance, symbolic binding policy,
target matching, domain/progress protocols and independent conformance. Generic
memory-effect traits do not establish workflow purity or removability; even
`pure` scalar calls may fail. No generic DCE is applied; the optional local
fusion below preserves stage validation and failure rather than removing calls.

This makes xDSL promising for representation and diagnostics in B. C's granted
authority integration and local fusion are described below. Python 3.14t execution and observed GIL state do not establish
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
viewpoints were deferred at that handoff. The subsequent external review led
to the prepared-plan repair described below.

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

The C handoff passed **245 tests**, including **83 C tests** (23 semantic,
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
engine. The mixed experiment below now demonstrates a bounded dynamic
feedback/model run through one owner engine. Broader combined-runtime behavior,
GPU/distributed execution, durable restoration and production performance remain
unqualified. Framework selection, including whether custom machinery is
preferable, depends on the accumulated evidence and remaining tradeoffs.

## Prepared-plan correspondence repair

The external review of `6bea859b` exposed a cold verification gap: A could
resolve a different compatible implementation under an unchanged stage origin;
B could accept changed bounds, target or feedback descriptors. Independent
oracles detected wrong execution, but preparation did not consistently reject
the broken correspondence. These were faulty-emission/pass probes, rather than
failures of the ordinary generator. `sd-scripts-syv.31.14` addresses this before
the planned fusion comparison.

Decoded A/B plans now check their identity, concrete target, contract, bounds
and disposition against accepted meaning. Every semantic implementation
selection has exactly one prepared call with its selected symbol, installed
version, role/effect/completion requirements and actual source ancestry. B
additionally checks chain order and the complete owner/control recipes; A checks
the retained semantic recipe, owner protocol and address correspondence before
loading models. Unknown executable properties and unsupported operations reject
readiness. The repair initially required one-to-one call correspondence. Local
fusion below retains that leaf-binding correspondence and adds checked
many-to-one dispatch grouping.

The current prepared vocabulary has these authorities:

| Fields | Authority and consumption |
|---|---|
| `exec.plan.accepted_digest`, `contract`, `target` | Checked identity of the accepted document and supported backend. They cannot select an unimplemented target by changing a label. |
| `horizon`, `capacity` | Concrete choices constrained by accepted bounds and the selected target. B consumes them for live admission/mailbox control; A checks them against its finite address/owner realization. |
| `stop_rule`, `feedback` | Required fixed-protocol descriptors, checked against the accepted disposition and supported coordinator. They are not arbitrary instruction strings. |
| `exec.call.implementation`, `binding_version`, `role`, `effect`, `completion` | Checked correspondence to the semantic selection and installed ABI. The resolved implementation is invoked; the backend supplies the verified completion mechanism. |
| `exec.call.origin`, `ancestors` | Checked source correspondence retained for outcomes and diagnostics. A compatible binding under another origin does not satisfy it. |
| `exec.chain.role`, `transport`, ordered calls | The role and call order construct dispatch. Transport identifies the supported fixed handoff/lifetime protocol and is checked. |
| B `exec.select`, `exec.handoff`, `exec.feedback` properties and owner declarations | Required scalar-profile contracts matched exactly to accepted entities/endpoints. The versioned backend implements selection, update, retention and acknowledgement; it does not interpret the descriptive arithmetic strings. |
| A `model_data.recipe` properties and semantic body | Checked owner-protocol contract and accepted subtree. This is input to owner-program construction, not a fully lowered machine program. |
| A `model_data.address` | Sample/window/slot correspondence is checked. Concrete port names are consumed by publication/admission and must satisfy the target's distinct-member constraints. |
| xDSL locations and ordinary diagnostic attributes | Non-authoritative metadata for this backend. Execution obligations cannot be established solely by such annotations. |

The repaired suite passes **309 tests**, including 64 new regression cases.
The environment without Torch passes **167 tests and skips two optional
modules**. Actual GIL state stays false before/after the full suite. Ruff,
formatting and ty pass. [Repair evidence](evidence_verification/) records the
original and rejected mutation cases, source hashes and the [approved review](evidence_verification/review.md).
Small reviewed follow-ups normalize readiness failures, pin missing/novel chains,
and align C's decoded structural-failure classification. xDSL IRDL already rejects
undeclared properties; the target checks also enforce its known protocol profile.
Earlier B/A/C evidence directories remain their historical snapshots.

This repair preserves the small fixed-protocol runtime. It does not introduce
a general interpreter, a plugin system or a new effect algebra. The subsequent
fusion, mixed feedback/model workflow and independent operation/schema tests
are recorded in the following sections; no framework or production choice follows
from this defensive verification alone.

## Local data fusion

`sd-scripts-syv.31.12` adds an actual xDSL rewrite of the prepared local binding
selections. The accepted semantic document is unchanged. [fusion.py](fusion.py)
uses `RewritePattern`, `PatternRewriteWalker`, cloning and replacement to group
adjacent eligible calls into an `exec.fused` region. The group retains every
leaf implementation selection, ordered source ID and ancestor chain:

```text
unfused: exec.call(data.double) ; exec.call(data.offset)
fused:   exec.fused(origins=[data.double,data.offset]) {
           exec.call(data.double) ; exec.call(data.offset)
         }
```

The rewrite queries two small semantic interfaces in
[local_contract.py](local_contract.py). A local transform describes its input
and output types, one-to-one cardinality, declared resource footprint and
possible outcomes. A transparent local region declares ordered composition
without a separate live-resource lifetime. Eligibility requires the supported
scalar contract, no declared resource effects, known preserved outcomes and
transparent ancestry. Unknown semantics and effectful stages are barriers.
Model uses, publication, feedback and granted phase protocols are outside the
eligible group. These interfaces serve this pass; they do not establish a
general effect or plugin system.

After parsing, preparation rechecks group profile, closed body, complete leaf
bindings, ordered origins and semantic stage order. This matters especially in
A, whose original ungrouped bindings were consumed by origin lookup. Grouping
cannot convert metadata order into a different algorithm. The optional policy
is explicit as `prepare(..., fuse=True)` / `prepare_model(..., fuse=True)` and
`--fuse` in both CLIs. Fresh-process tests reconstruct both realizations with
the authoring builder disabled.
The binding manifest reports actual formed groups, including zero when the
requested rewrite finds only barriers. Maximal grouping is not a semantic
requirement; complete leaf correspondence and order are.

[local_runtime.py](local_runtime.py) makes the execution difference concrete:
unfused leaves enter separate guarded coroutines; a fused group uses one
generated coroutine with the guard shell inlined for its stages. Numeric
bindings still run, in order, and validate before the next stage can enter.
Code generation interpolates trusted templates and integer indexes; source
labels remain data. The shared observer ABI records entry, physical return,
validation and release. B retains its original per-stage clocks/outcomes. A
now uses that common guarded-stage ABI and retains sample/source/outcome facts;
its model admission, state and optimization owners are unchanged.

For four A records, local dispatch observations change from eight to four while
all eight data-stage outcomes survive. B likewise uses one rather than two
dispatches per completed local chain. These counters are observations, not
readiness or admission authority. C has no eligible local stages; the rewrite
is a checked no-op and its grant conformance still passes.

The suite passes **356 tests**, including rejected corrupt groups, effect
barriers, invalid pure output, raised failures, prevention of later-stage entry,
physical stop/repeated cancellation with successful or failed return, changed
production order, independent A/B numerical/association checks and C's no-op.
An intentionally shared incorrect binding still passes agreement between the
two realizations and fails the independent oracle. The no-Torch environment
passes **196 tests and skips two optional modules**. Actual GIL state stays
false before/after the full run; Ruff, formatting and ty pass.

[Fusion evidence](evidence_fusion/) retains prepared text, full B observations,
one fused A snapshot, source hashes, raw costs and review status.
The [required review](evidence_fusion/review.md) approved the rewrite and runtime
distinction. Small follow-ups corrected the manifest's actual-group diagnostic
and removed the obsolete flat dispatch field; all 356 tests and static gates
pass afterward. The cost files retain their original measurement snapshot,
before these descriptor/unused-field cleanups and the typing-only Protocol
correction; hot guards and selected bindings did not change or get resampled.
[fusion_probe.py](fusion_probe.py) separates acceptance/preparation from an
isolated local-dispatch probe with real yielding scalar bindings and a small
validating observer. Seven alternating pairs of 1,000 records gave local
medians of **6.575 ms unfused / 6.392 ms fused**, ratio **1.029**, and preparation
medians **10.55 / 11.99 ms**. Raw samples are retained; this small single-process
probe establishes no statistical or production speedup. It excludes complete
session coordination, model owners and GPU work, and does not compare against
the previous execution candidate. The first timing run is retained without
resampling for a more favorable result.

The concrete infrastructure reused here is xDSL's traits, recursive IR,
cloning, pattern walker/replacement, structural verification and text round-trip.
The legality rules, source correspondence, binding/observer ABI and executable
construction remain project work. The prepared calls have no SSA results, so
this rewrite does not exercise SSA result remapping or prove that generic
framework analyses understand our ownership semantics. The following sections
record continuous mixed feedback/model behavior and independent operation/schema
composition as separate qualifications.

```bash
/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.xdsl_candidate \
  --fuse --dump-dir /tmp/xdsl-b-fused
/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.xdsl_candidate.fusion_probe \
  --dump-dir /tmp/xdsl-fusion-costs
```

## Mixed continuous selection and shared model ownership

The bounded [mixed contract](mixed_contract.py) combines B's open selection and
learning-driven feedback with A's real owner services. Its accepted document
contains dynamic filtering, shared-participant model preprocessing, atomic
correlation, ordinary contribution/advancement, post-step feedback and an
external stop checkpoint. [mixed_workflow.svg](mixed_workflow.svg) shows the
feedback loop, containment and shared state.

One existing `Engine` runs an open `Producer` beside a learning `Repeat`.
The producer owns cursor, policy and three outstanding tickets; learning sends
post-advance weight/input observations through a bounded message transport.
The producer's receiver validates membership, updates bias to
`int(abs(weight)*10)%5`, increments revision and retires exactly that ticket.
Selection uses the actual applied revision and `key + bias`. Key multiples of
four are excluded only after feedback exists, so key zero is admitted and key
four is dynamically excluded. This policy is an installed fixture, not a
universal feedback or training rule.

The **same trainable participant** supplies detached preprocessing reads and
receives updates. Admission requires the same binding and actual protected
production provenance; it permits an older numerical epoch at later claim.
The numerical checker independently reconstructs the weight at that observed
epoch. Each loss is `(w*(left+right)-selected/2)^2/2`; one contribution advances
at learning rate `.001`. Existing retained-state, input, unit, physical-worker
and progress services own these lifetimes. There is no second scalar session
or run coordinator. Common local dispatch/fusion handles the pure branch.

This CPU profile reaches a declared checkpoint after three windows and waits
for **external** stop. Stop closes selection, discards unclaimed ready products
and cancels/joins private production. The pressure driver deliberately keeps
key six in a real worker while key five waits on bounded publication. Stop and
repeated interruption cannot release that model read before physical return;
a subsequent physical failure remains visible alongside completed updates.
An interruption that wins before normal producer completion can retain ready
products in **closed** feeds for diagnosis. The oracle accounts for retained,
discarded and never-published products separately. Retired outstanding tickets
are labelled `closed_without_feedback`, not fabricated acknowledgements.

Run the pressure driver (Torch optional outside this extension):

```bash
uv run --no-sync --with-requirements docs_design/models-strategy-trainer/xdsl_candidate/requirements.txt \
  python -m docs_design.models-strategy-trainer.xdsl_candidate.mixed_demo \
  --fuse --dump-dir /tmp/xdsl-mixed
```

Accepted/prepared text and the version manifest reconstruct with the author
builder disabled in a fresh process. Corrupt target, bounds, source ancestry,
implementation, stage order, retained meaning or owner protocol reject before
model loading. Independent arithmetic and report-corruption controls also
reject a shared wrong implementation across fused/unfused paths. Diagnostics
can be evicted entirely without losing live association or completion evidence.

This qualifies one explicit stop checkpoint, CPU payload/profile and feedback
policy. It does not qualify arbitrary mid-gradient stop, indefinite archival
memory, general plugin discovery, GPU/distributed execution or a framework
selection. As in C, CPython's shield machinery can log a physical exception
during repeated cancellation even though the joined owner report retains it;
that diagnostic is not evidence of detached work. First failures, qualification
and required review are retained in [evidence_mixed/](evidence_mixed/).

The shared `ack_before_next_selection` descriptor means selection reflects all
feedback already applied before it; the horizon permits selecting other tickets
before a predecessor's acknowledgement. The pressure driver requires its explicit
key-six gate and fails promptly if that control is absent. Planned shutdown also
relies on the existing producer cancelling private tasks before closing their
ports; reversing that target ordering would need renewed qualification. Failed
runs return a report whose `status` callers must inspect; interruption propagates
after the report and drain are retained.

## Independent operation, payload schema and declared resource

[pair_extension.py](pair_extension.py) installs an independent `pairing.pack`
operation and `pairing.vector<width>` payload type. It uses B's existing session
contract, selector, mailbox, feedback and runtime. No new whole-workflow verifier,
owner engine, adapter or core operation-name case is added for it.

Inside the existing recursive local region, the original scalar result enters
`pairing.pack`, which reads the versioned frozen coefficients `(1, 2)` and
produces the typed pair `(value, 2*value)`. An ordinary `data.map` binding selects
the first lane. This deliberately preserves B's surrounding numerical behavior
while making the internal payload independently observable. A wrong second lane
passes the old final-scalar oracle and fails the separate payload oracle.

The concrete common interfaces are now:

| Interface | Checked meaning |
|---|---|
| `wf.flow` payload | An open schema name or structurally valid payload type, with codec installation checked during preparation, rather than three core payload strings. The independent type verifies positive width itself; unsupported positive widths can be accepted and then rejected for missing target codec support. |
| `LocalTransform` | Typed unary input/output, one-to-one cardinality, declared resources and possible outcomes. `local_maps` retains its old name but analyzes interfaces and SSA links rather than a closed `MapOp | LocalOp` list. |
| `TransparentLocal` | An ordered region with no separate resource lifetime; its final yield must match its declared output schema. |
| Local payload/resource ABI | Versioned input/output codecs, exact resource symbol/access/version/content digest, immutable snapshot scope and an installed lowering. Only declared snapshots reach the lowering factory. |

The bounded resource access here is **`read_frozen` configuration**. Frozen
integer tuples and read-only maps prevent mutation through this ABI. This does
not implement live model leases, mutable resources, security isolation or a
universal effect system. Installed code remains trusted. Domain `negative_input`
raises a declared failure; validation failures and undeclared outcomes stop
before later local work or publication can enter.

Prepared B plans now contain `exec.local_signature` for every local stage.
Its schemas, codec versions, resource identity and outcomes are compared with
the accepted operation interfaces and installed registry **after decoding and
before any lowering factory runs**. The manifest uses schema 2 and records the
same codec/resource/outcome information. Missing/duplicate/corrupted signatures,
unsupported bindings and illegal fusion reject during preparation. Historical
manifests remain historical snapshots and are rejected if they differ from the
current realization; this is not a stable artifact-compatibility promise.

The unchanged fusion eligibility logic fuses the original pure scalar pair and
keeps both resource-bearing pack and unfamiliar-schema unpack as barriers.
Grouped/un-grouped execution preserves all six call facts per completed ticket.
The two existing scalar owner adapters explicitly reject accepted unsupported
local schemas during preparation; they do not silently acquire this new ABI.

```bash
/tmp/sd-scripts-xdsl-b-env/bin/python \
  -m docs_design.models-strategy-trainer.xdsl_candidate.extension_demo \
  --fuse --dump-dir /tmp/xdsl-extension
```

Fresh-process tests disable the authoring builder and reconstruct from accepted
text plus the installed dialect/codecs/version manifest. Tests also cover malformed
schema/cardinality/wiring, missing support, resource identity/access/outcome
mismatch, decoded contract corruption before lowering, invalid payload after
entry, declared/undeclared failures, a shared hidden-lane mistake and real physical
work draining through repeated cancellation. The combined gates and required
review are retained in [evidence_extension/](evidence_extension/).

This demonstrates local composability and exposes useful minimal interfaces.
Whole-session service contracts are still fixture profiles; generic cardinality,
mutable resources, arbitrary region semantics and multi-unit overlap remain
unqualified. xDSL provides the structural/type/parser/rewrite substrate, while
project code still owns these semantics, binding policy, runtime authority and
independent oracles. These results strengthen its candidacy without settling
xDSL versus a custom system.

The archive-erasure audit also found and corrected an incidental mixed-workflow
dependency: post-step feedback had read the last contribution from its recorder.
It now receives the actual admitted record through the existing Sequence value
link. Both owner lowering modes complete with every source/learning observation
archive disabled, while retained owner input facts independently reconstruct the
final parameter. The before-fix failure is retained in mixed evidence.

Domain failure names cannot impersonate runtime outcomes such as `returned` or
`invalid_output`; they are rejected or classified as an undeclared outcome.
Missing local source identity also has an explicit preparation error, independent
of Python assertion optimization.


## Returning-filter progress and runtime contract repair

The follow-up review of `504dcf8` reproduced three local defects. A source that
kept excluding records never filled the admitted-ticket horizon and could run
without suspending. Matching operation/binding declarations could omit baseline
outcomes that execution still emitted. B's ticket removal was inside an assertion
and disappeared under `python -O`.

The mixed target now yields after at most **64 selection attempts** without a
capacity wait, including excluded attempts, then rechecks shutdown. This target
quantum lets loop callbacks, model work, learning feedback and cancellation run.
It does not preempt a blocking/non-returning binding or bound archival memory for
an indefinitely running service. Horizon accounting still counts admitted tickets;
excluded keys do not gain fabricated admissions.

`LocalDescription.outcomes` must explicitly include `returned`, `invalid_output`
and `raised`; additional names describe domain failures. `raised` names an
exceptional exit category, recorded in detailed guard facts as `entered_failed`.
Semantic acceptance and
cold resolution both reject omissions, even when the installed binding agrees
with the incomplete declaration. Scalar retirement performs removal and checks
identity explicitly, so optimized execution has the same ownership effect.

The progress probes use individually fast predicates with a finite excluded
stretch. They check a callback before the sentinel, actual model/learning feedback
while exclusion continues, and cancellation before the finite fallback. Optimized
subprocesses execute full B and extension sessions; the independent oracles run
in the parent with assertions enabled. Archive-erasure qualification now covers
both owner lowerings and both local fusion modes, including repeated cancellation
coupled with physical failure while the source read remains protected.

[Progress/contract evidence](evidence_progress/qualification.json) retains the
before/after probes, environment, source hashes, final gates and required review.
Older evidence folders remain snapshots of their recorded source revisions.
