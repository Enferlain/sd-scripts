# Matched xDSL and Pliron comparison

Research plan · 1 November 2026 · preparation `sd-scripts-syv.31.24` · milestone `sd-scripts-syv.31.25`

The next milestone compares **IR machinery for stateful, heterogeneous data and
model workflows** on six existing case meanings. It ends with a substrate recommendation,
material weaknesses and conditions that would reverse the recommendation. The
production language need not be finished first: both implementations can inform
its design. xDSL remains the provisional working baseline during the comparison.

This is a preparation contract, not evidence that the missing ports exist. The
[evidence inventory](candidate_comparison.md), governing
[OpenSpec change](../../openspec/changes/rework-model-strategy-trainer/), and
[IR report](../../openspec/changes/rework-model-strategy-trainer/research/ir_architecture_report.md)
retain their respective roles. No production API, complete language, runtime
implementation or numerical backend is selected here.

## Fixed scope and starting point

The starting source is commit `2f4c9353940d5f467e8fb7ceb2233fd88243baa4`.
The experiment pins are xDSL **0.74.0** and Pliron **=0.20.0**. Preserve those
versions for this milestone; a necessary version change needs a recorded reason
and rerun of the affected comparison, rather than an unnoticed environment drift.

| Case | xDSL at the starting point | Pliron at the starting point | Work needed |
| --- | --- | --- | --- |
| B: data-only production and feedback | Implemented | Implemented | Retain as a matched control. |
| A: correlation and accumulation | Implemented | Not implemented | Independent Pliron representation and preparation. |
| C: temporary optimization authority | Implemented | Not implemented | Independent Pliron representation and preparation. |
| Mixed feedback/model ownership | Implemented | Not implemented | Independent Pliron representation and preparation. |
| Independent payload/resource extension | Implemented; scalar/resource barriers | Implemented; resource-aware grouping | Align intended transformation legality and analysis exercise. |
| Different-lifetime composition | Not implemented | Implemented | Independent xDSL representation and preparation. |

The [frozen descriptive fixtures](comparison_fixtures.json) and
[first implementation slice](xdsl_candidate/lifetime_case.md) now record control
setup and the xDSL lifetime port (`.31.25.1`, `.31.25.2`). The table above is the
starting-point inventory. The subsequent [Pliron C record](pliron_candidate/grant_case.md)
adds independent authority representation/preparation and persisted-plan checks
for both C frontends (`.31.25.3`). The [grouping comparison](grouping_comparison.md)
now aligns typed unary/frozen-read legality and analysis preservation/invalidation
(`.31.25.4`). The [Pliron A record](pliron_candidate/correlation_case.md) now adds
independent correlation/accumulation representation and cold checks (`.31.25.5`).
The [Pliron mixed record](pliron_candidate/mixed_case.md) adds independent
feedback/model representation and cold checks (`.31.25.6`). These implementation
records retain current evidence. The [machinery evidence](machinery_comparison.md)
now records the matched inventories, diagnostic sample and phase/process
observations (`.31.25.7`), including the retained endpoint qualification correction.
The decision remains outstanding.

Both candidates may reuse the selected numerical implementations, existing owner
engine, state/grant/optimization services, physical workers, fixtures and
independent checkers. This reuse does not validate a second ownership runtime.

Each candidate must independently express and verify its dependencies, accesses,
pending obligations, authority and accepted/prepared correspondence using its
own IR machinery. Shared code may install bindings, execute declared services or
project actual observations for a checker. It must not resolve the interesting
semantics in a shared executable graph before either framework receives them.
One candidate's acceptance, verifier, parser or lowering must not implement the
other's IR path. Incidental imports from a reused Python target are recorded
separately from actual use of xDSL machinery.

## Six case meanings

The arithmetic and causal requirements below are independent of operation names,
Python classes, printed syntax, prepared recipes and report dictionary shapes.
The linked checkers are starting expectations, not a production comparison ABI.

**B — feedback control.** Select increasing keys with `x=key+bias`; compute
`2*x`, then `+1`, square and acknowledged publication. Apply `score % 5` as the
next policy bias only after its acknowledged feedback. Use aggregate outstanding
bound two and request stop after four acknowledgements. Already admitted work
may settle afterward; equal final ticket counts or interleavings are unnecessary.
Use [check_b](xdsl_candidate/conformance.py). Retain feedback under blocked
publication, selected-stage failure, cancellation with held physical work and
subsequent failure, and wrong computation/feedback-provenance controls.

**A — correlation and accumulation.** Use samples `(1,2,4,5)`, excluding `3,6`,
with explicit window/slot membership. Prepare left `2*k+1` and protected frozen
encoder right `1.5*k/4`; the learner feature is their sum and target is `k/2`.
Starting weight is `0.8`; loss is `(w*feature-target)^2/2`. Sum two contributions
per window and apply ordinary SGD at `0.05` once per window. This case uses
**summing**, unlike the lifetime case's mean. One evaluation requested by window
zero retains its trigger membership but may observe protected epoch one or two
at actual start. Use [check_a and check_admission_failure](xdsl_candidate/model_conformance.py).
An invalid/missing correlated member must claim neither member. Retain an earlier
contribution's pending ownership, physical-drain evidence, and wrong source,
window, association and evaluated-epoch controls.

**C — temporary authority.** Starting at `w=0.8`, use loss `(2*w-1)^2`, first
backward, temporary edit `w += 0.1*gradient`, intermediate reset, second backward,
restoration and verified handback. Retained SGD at `0.05` uses the second gradient
against the restored original weight, then resets. Without clipping the gradients
are `2.4` and `4.32`, temporary weight is `1.04`, and final weight is `0.584`.
Also run the existing optional clipping policy with norm `1.0`. Use
[reference, check_c and check_unverified](xdsl_candidate/grant_conformance.py).
Restoration without verified handback cannot authorize advancement. Retain failed
handback, interruption during physical use, mode/protection restoration, and
entered/returned/uncertain/unattempted optimizer/reset outcomes. A plausible
weight alone cannot establish a successful authority transfer.

**Mixed — feedback with changing model state.** Start the learner at `0.8` and
perform three one-contribution learning windows, SGD at `0.001`, stopping at the
explicit checkpoint. Selection is `v=key+bias`; left is `2*v+1`, right is a
protected read of the same learner times `v/4`, and target is `v/2`, using halved
squared error. Feedback after each update sets `bias=int(abs(w)*10) % 5`;
after revision zero, keys divisible by four are excluded. Use aggregate horizon
three and capacity one per correlated input port. Reconstruct arithmetic from
the **actual protected production epoch**, which may precede admission. Use
[check_mixed](xdsl_candidate/mixed_conformance.py). Require blocked-publication
stop, live physical model read through interruption/failure, truthful discarded
inputs, feedback without diagnostic archives, and progress during long stretches
of individually fast, returning exclusions. Do not require a fixed selection
trace or silently replace production state with the latest state.

**Extension — independent typed operation and frozen resource.** After B's two
maps, read frozen coefficients `(1,2)`, pack `(v,2*v)` into a distinct payload,
then return its first lane. B's scalar result is deliberately unchanged. Reuse
[check_extension](xdsl_candidate/extension_conformance.py) and the existing
declared `negative_input`, invalid-output and undeclared-outcome controls. The
hidden second lane must be independently checked. Align the transformation below
without adding extension operation identities to central validation. Record which
shared interfaces, registry code and target support actually needed modification.

**Lifetime composition — independent producer and requested consumer.** Use the
complete [case contract](pliron_candidate/composition_case.md): frozen coefficient
two, records `x=2*(key+1)`, target `key+1`, training windows `(0,1)` and `(2,3)`,
mean gradients, initial `0.8`, and SGD at `0.01`. Weights are `0.77`, `0.635`.
Window two requests evaluation once; cause members `(2,3)` differ from its input
record `4`. Protected current-at-start epoch is two, prediction `6.35`, loss
`0.91125`. Aggregate selected/unsettled bound is three; ready budget is two
across ports, with one physical producer. Use the independent
[composition checker](pliron_candidate/composition_conformance.py).
Required witnesses are late and prefetched record four, pending requests keeping
the group alive, partial member acquisition, orderly stop before/after window
selection, root cancellation of partial/pending work, and training/evaluation
failure. Hold evaluation four, ready record five and physical record six; group
completion requests producer stop, while root completion waits for physical
drain. Preserve completed evaluation alongside later producer failure. Erase
diagnostic histories during execution. Consumer-group completion must not wait
for producer termination and create a cycle.

## Common transformation policy

The extension is the matched transformation exercise. Group contiguous ordered
unary stages with compatible adjacent payload types, one-to-one cardinality,
transparent enclosing local regions, and known pure or pinned frozen-read
effects. A schema change itself is permitted. Preserve every leaf's source,
validation, declared/domain/runtime outcomes, entry and physical-completion
boundaries. Frozen snapshots retain symbol, access, version and content identity;
grouping does not create mutable-resource authority or extend a physical lease.

| Variant | Required transformation result |
| --- | --- |
| Double, offset, frozen pack, first lane | Four eligible local map calls become one group, including typed transitions. Score/publication remain outside. |
| Double, opaque offset, frozen pack, first lane | Double and opaque offset remain single dispatches; frozen pack and first lane group: three dispatch groups. The offset declares unknown effects through the existing ordinary-map contract; pack retains its required frozen-read contract. |
| Saved group forged across that barrier | Cold verification rejects the claim before it can authorize dependent execution/effects. |
| Guard/domain/physical failure within a group | Preserve the failing leaf and partial outcomes; later leaves and publication do not enter. |

The opaque-offset variant must first pass semantic acceptance and **successful
unfused execution** in both candidates. Reuse the existing Pliron opaque offset
and its supported binding description. Merely changing pack's required
`read_frozen` effect would invalidate its contract and would not test grouping
legality. After that valid execution, a forged saved group crossing the offset
barrier must be rejected for transformation legality, not ordinary pack rejection.

Have a resource-use/stability summary drive legality in both implementations.
Grouping preserves that semantic summary; dispatch-count analysis changes.
Changing the semantic resource/effect contract must invalidate/recompute the
summary and affect legality. Record the actual cache mechanism, preservation and
invalidation in each candidate; framework-managed caching and project caching
must not be reported as the same contribution. Cold loading recomputes legality
from verified source/contracts rather than trusting serialized analysis results.
Counts establish the chosen transformation, not a framework speed advantage.
SSA result replacement is outside this milestone unless a required
transformation exposes a concrete need for it.

## Evidence required for every matched case

1. Retain candidate-specific accepted/prepared documents, binding/target
   descriptors, exact source/version hashes and the selected policy parameters.
   Successful structural verification alone cannot establish domain legality.
2. Execute through the shared owner/worker services and project actual owner
   facts into independent arithmetic, association, outcome and causal checks.
   Compare permitted relationships, including alternative A evaluation epochs,
   rather than identical clocks or private scheduler traces. Never synthesize a
   successful observation that execution did not provide.
3. Cold-load in a fresh process with authoring disabled and the other candidate's
   IR construction/parsing/preparation disabled. Corrupt relevant bindings,
   source/role correspondence, resource/schema/outcome contracts and the case's
   authority or lifetime claims. Broken correspondence must not publish readiness,
   authorize execution or permit effects dependent on the invalid claim. Existing
   no-factory corruption cases retain their stronger boundary; pure construction
   before verification is not a universal disqualification.
4. Run an optimized child with an assertion-enabled parent oracle. Retain both
   real failure injection (a wrong shared implementation can execute successfully
   and must fail the oracle) and corrupted-observation controls. Replay alone is
   not execution validation. Preserve archive-erasure and actual physical-drain
   controls where those observations form part of the case.
5. Identify supported semantic contracts versus target limitations. Include one
   relevant valid-but-unsupported combination and honest rejection. Record
   fixture-specific validators/lowerings as project code, not generic framework
   composition. A limitation can be a decision result; do not expand the language
   simply to make every unsupported program pass.

No one-to-one copying of test files or a common report schema is required. A
per-case receipt records these observations, matched parameters, unsupported
behavior and check commands. Review any new oracle projection for independence
and retain both raw facts and its mapping. The original custom candidate supplies
runtime/reference evidence; rebuilding it to match all six IR cases is outside
this two-framework milestone.

## Development, diagnostics and measurement

For each case retain a source-linked inventory separating framework facilities
from project semantic verification, target policy, bridge and runtime services.
For the extension and one lifetime/authority contract, record the exact modules
and checks affected by the same semantic change. Reuse the changes already needed
for the ports/alignment; do not manufacture a second extension. Count duplicated
semantic reasoning and explain why it exists. Lines of code and initial build
effort may be reported as context, not stand-alone quality scores.

Use a fixed diagnostic sample: a missing required binding, incompatible payload,
wrong frozen-resource identity/access, unsupported target contract, forged
prepared correspondence, and invalid authority/lifetime claim in the relevant
case. Retain actual accept/prepare/cold-load errors side by side. Judge whether
they identify the source Python activity/selection, violated obligation,
expected/provided contracts and proposed realization without reading a central
validator. Record location-mapping work and missing context explicitly.

Keep **initial implementation cost, ongoing maintenance, machinery ergonomics,
and Python integration** as separate judgments. Document registration, inspection,
error propagation and artifact compatibility policy, including the current
Pliron lifetime target's whole-manifest equality for unused bindings. Rust
ownership does not establish modeled resource ownership. Native-backend reuse
needs a named shared artifact/interface/analysis and version-compatible evidence;
it is not a benefit awarded for using the same framework name.

Measurements use these six small workloads only. Before recording results,
freeze the harness commands, timing boundaries and repetition counts in the
measurement task. Default protocol: three warm-ups, twenty measured paired runs
per warm phase, alternating candidate order; ten fresh-process starts per
candidate and case. Retain every sample, failures, median and interquartile range.
Do not retry or select a favorable subset of failed/noisy runs.

Report separately: author construction, semantic acceptance/verification,
preparation without/with the matched rewrite, serialize/parse, cold validation,
Python binding materialization, and end-to-end invocation. A combined CLI phase
must keep its combined label until instrumented: do not subtract noisy startup
measurements to claim a native body time. Repeated one-request CLI processes do
not establish a warm in-process native measurement. Pliron release-build time
is a separate development observation, excluded from run time. Record Rust optimization level,
Python imports/GIL state, dependency pins, CPU/thread settings and cache policy.
Compare the same workload and transformation policy. The current process/JSON
bridge is a measured integration choice, not an intrinsic Pliron cost; xDSL's
in-process boundary is likewise part of its current implementation.

Record fresh-process peak resident memory for the Python parent and native child
separately; do not add independent peaks and call them simultaneous total memory.
Distinguish this from any measured retained IR allocation. Startup/import and
Torch memory are not inferred to be framework IR memory. A phase that cannot be
isolated honestly stays combined or unmeasured. No throughput, GPU or Rust
performance advantage is inferred from these small CPU preparation workloads.

Correctness processes stay separate: Torch-free B/extension checks and optional
Torch owner checks, with GIL state observed after ordinary imports and execution.
Do not force the GIL off. The retained broader-suite mixed progress failure
under `sd-scripts-syv.31.23` has an unresolved combined-state/timing cause; the
isolated GIL-on witness passed. Resolve it or document a reproducible isolation
before using affected progress/timing results. It does not block representation
ports, and an ignored failure is not a successful control.

## Work order and endpoint

Beads is the work/status authority. The scheduled order is control setup,
xDSL lifetime, Pliron C, transformation alignment, Pliron A, Pliron mixed,
matched evidence, then the decision record. Independent steps need not share
implementation dependencies; each significant slice has quality gates and
required review before handoff. This preparation does not start those ports.

| Work | Bead | Prerequisites |
| --- | --- | --- |
| Freeze fixtures and B control | `sd-scripts-syv.31.25.1` | Preparation and recorded review outcome. |
| xDSL lifetime | `sd-scripts-syv.31.25.2` | Control setup. |
| Pliron C | `sd-scripts-syv.31.25.3` | Control setup; scheduled after lifetime. |
| Aligned transformation | `sd-scripts-syv.31.25.4` | Both reciprocal representation slices. |
| Pliron A | `sd-scripts-syv.31.25.5` | Pliron C; scheduled after transformation. |
| Pliron mixed | `sd-scripts-syv.31.25.6` | A and aligned resource contracts. |
| Matched evidence | `sd-scripts-syv.31.25.7` | Control, ports and transformation. |
| Decision record | `sd-scripts-syv.31.25.8` | Matched evidence, or explicit early-stop limitation. |

Correctness, correspondence, independent numerical/causal checks and physical
settlement are gates. Among implementations that meet them, assess semantic
expression, framework contribution, extensibility, diagnostic quality, Python
integration, maintenance and measured costs, each with concrete evidence and
limitations. There is no weighted total that can trade a correctness failure for
speed, nor a test-count or implementation-language ranking.

Close with a decision record naming the preferred working substrate, decisive
advantages, material weaknesses, unresolved evidence, and conditions that would
reverse the choice. If evidence is tied or inconclusive, retain xDSL and state
what switching advantage remains unearned. A demonstrated material limitation
may justify an earlier decision with coverage gaps recorded; silently dropping
a hard case cannot establish matched coverage. An early stop must explicitly
revise the remaining Beads dependencies and leave unimplemented slices open or
deferred, rather than mark them complete to unblock the decision. Initial implementation cost may
be excluded from the recommendation, but maintenance and integration consequences
must remain visible.

No new pressure cases are added unless an implementation reveals a specific
unanswered question material to the decision. Record that question and bound the
extra work in Beads first. General language composition, mutable-resource
replacement, SSA-changing rewrites, in-process native bindings, backend migration,
GPU/distributed completion and durable recovery remain outside this milestone.

## Preparation verification

The [preparation record](matched_comparison_preparation.json) retains source
hashes, local-link checks and assertion-enabled independent-checker replay.
The existing inventory accepts nine saved reports and rejects six corruptions;
the lifetime replay accepts eleven reports and rejects nine corruptions. This
checks the reusable expectations against retained observations. It does not
re-execute the archived runtimes, establish matched coverage or measure costs.
No runtime, oracle, production code, governing OpenSpec or user-report changes
are part of this preparation. Implementation and its reviews belong to the open
milestone work above.

The preparation review attempt failed during analysis with a provider usage
limit (HTTP 429); it supplied no approval. That outcome is retained separately
from the successful local checks. That attempt produced no independent
assessment and reviewed no implementation slice.

The separately supplied external assessment on 1 November approved proceeding
after clarifying the broader workflow scope and a valid opaque-offset barrier.
Those clarifications are incorporated above. The assessor verified the attached
Markdown hashes but explicitly did not rerun the recorded replay checks. Its
assessment does not turn the failed provider attempt into a successful review.
