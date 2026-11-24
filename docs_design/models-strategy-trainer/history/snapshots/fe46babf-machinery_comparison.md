# Matched machinery evidence

2 November 2026 · `sd-scripts-syv.31.25.7` · base `139ab2dc`

This records the six-case comparison's machinery, diagnostics and bounded CPU
observations. It supplies evidence for the separate decision task; it does not
select a final language, runtime or substrate. xDSL remains the provisional
working baseline. The [plan](matched_comparison.md), [frozen protocol](evidence_comparison/protocol.json)
and [qualification](evidence_comparison/qualification.json) define the boundaries.

## Six-case receipts and ownership boundary

[The case directory](evidence_comparison/cases/) retains accepted/prepared text,
installed manifests and actual reports for both candidates: unfused/fused B,
extension, A and mixed; ordinary C; and late-input lifetime composition.
Every completed workflow sample runs its independent numerical/causal checker outside
the timer. The receipt tests independently recheck all twenty retained reports.
Earlier pressure, cancellation, authority and corruption qualifications remain
in their case records. This collection does not replace them.

| Case | xDSL representation and project checks | Pliron representation and project checks | Existing execution authority |
| --- | --- | --- | --- |
| B | [dialects](xdsl_candidate/dialects.py), [accept/prepare](xdsl_candidate/prepare.py): roles, endpoints, calls, bounds, receipts | [dialect](pliron_candidate/native/src/dialect.rs), [accept/prepare](pliron_candidate/native/src/prepare.rs): graph/SSA shape, roles, endpoints, contracts | [scalar runtime](xdsl_candidate/runtime.py) and selected fixture bindings |
| Extension | [independent dialect](xdsl_candidate/pair_extension.py), [local contracts](xdsl_candidate/local_contract.py), [payload resolution](xdsl_candidate/payloads.py), [grouping](xdsl_candidate/fusion.py) | [independent dialect](pliron_candidate/native/src/pair_extension.rs), [contracts](pliron_candidate/native/src/contract.rs), [resource analysis/rewrite](pliron_candidate/native/src/prepare.rs) | Same scalar runtime, codecs, frozen snapshots and per-stage guards |
| A | [semantic checks](xdsl_candidate/model_contract.py), [prepared correspondence](xdsl_candidate/model_adapter.py) | [typed correlation/accumulation](pliron_candidate/native/src/correlation/mod.rs) and [shape interfaces](pliron_candidate/native/src/correlation/dialect.rs) | [model target](model_target.py): atomic joins, retained gradients, requested current reads |
| C | [authority contract](xdsl_candidate/grant_contract.py), [prepared checks](xdsl_candidate/grant_adapter.py) | [typed delegated phases/handback](pliron_candidate/native/src/grant/mod.rs) | [grant target](grant_target.py) and original retained/grant/optimization services |
| Mixed | [feedback contract](xdsl_candidate/mixed_contract.py), [saved graph checks](xdsl_candidate/mixed_adapter.py) | [producer/repeat/feedback/stop contracts](pliron_candidate/native/src/mixed/mod.rs) | [mixed target](mixed_target.py), actual protected production epochs and producer-owned policy |
| Lifetime | [activity types](xdsl_candidate/lifetime_ir.py), [verification/recipes](xdsl_candidate/lifetime.py) | [activity contracts/recipes](pliron_candidate/native/src/composition/mod.rs) | [composition runtime](pliron_candidate/composition_runtime.py) and original owner engine |

Each frontend independently handles accepted meaning and prepared correspondence.
A/C/mixed/lifetime reuse one owner implementation. Agreement between their reports
is not independent evidence for a second ownership runtime. B/extension also reuse
one scalar runtime. Native IR object ownership establishes neither model access
rights nor authority over gradients, parameters, physical work or input claims.

## What the frameworks replace

Both supply real typed operations, regions, operands/results, def/use structure,
verification support, reconstruction and rewriting. Project code still defines
participant identity, membership, resource stability, authority, feedback,
completion, target limits and every executable runtime service above.

Pliron explicitly distinguishes graph and SSA-CFG region kinds. This is used in
B, A, mixed and lifetime, rather than storing a separate custom graph in an opaque
attribute. Its structural equivalence maps actual definitions/uses; domain
validators still enforce which relationships are legal. Native matchers and
rewriters move real prepared calls. B/extension's `Passes` and `AnalysisManager`
cache the resource-use summary used for grouping, preserve it through grouping,
and invalidate dispatch counts. Correct preservation declarations and the
resource legality policy remain project responsibilities.

xDSL supplies IRDL definitions, traits/interfaces, SSA operands/results and
verification, parsing/printing, structural comparison and rewriting facilities.
The matched extension uses [project-owned resource/dependency caching](xdsl_candidate/fusion.py),
with semantic-change invalidation and cold recomputation. This is not the same
framework contribution as Pliron's analysis manager. Mixed needs a small
[forward-result mapping adaptation](xdsl_candidate/mixed_adapter.py) before xDSL
0.74 structural comparison; lifetime uses its [explicit equivalence helper](xdsl_candidate/lifetime_ir.py).
Both helpers add project maintenance, while the surrounding domain verifiers
remain necessary regardless of framework.

Prepared calls form ordered recipes without SSA results. The experiments qualify
real grouping and preservation of leaf guards/completion, not result replacement,
use remapping, arbitrary inlining or a reusable general region-composition model.
No additional demonstration was manufactured for this collection.

## Existing change touchpoints and duplicated reasoning

The same independent extension adds a pair payload and pinned coefficient read,
then returns the scalar required downstream. Its aligned rule permits typed
one-to-one pure/frozen stages and leaves a valid opaque offset as a barrier.

| Change layer | xDSL modules/checks touched | Pliron modules/checks touched |
| --- | --- | --- |
| Independent registration and numerical implementation | [pair extension](xdsl_candidate/pair_extension.py): `PairSchema`, `PackContract`, codecs, frozen binding/lowering | [pair dialect](pliron_candidate/native/src/pair_extension.rs), [Python registration](pliron_candidate/pair_extension.py): payload/operation interfaces, codecs and frozen binding |
| Shared contract vocabulary | [local contract](xdsl_candidate/local_contract.py), [payloads](xdsl_candidate/payloads.py): signatures, resource uses, outcomes and codecs | [native contracts](pliron_candidate/native/src/contract.rs), [bridge](pliron_candidate/bridge.py): signatures, resource uses, outcomes and installed manifest |
| Typed chain, preparation and grouping legality | [prepare](xdsl_candidate/prepare.py), [fusion](xdsl_candidate/fusion.py): decoded signatures, snapshots, resource summary, adjacency and cold groups | [prepare](pliron_candidate/native/src/prepare.rs): typed chain, manifest checks, resource analysis, matcher and cold groups |
| Execution | Existing guarded [local runtime](xdsl_candidate/local_runtime.py), receiving only declared snapshots | Same guarded local runtime through the bridge |

Extension identities stay in registration modules; central chain verification and
grouping follow shared interfaces. Nevertheless both implementations must reason
about the same obligation in **three distinct places**: semantic resource uses,
installed realization/cold correspondence, and runtime selected-call completion.
Those checks protect different boundaries. This is a count of conceptual boundary
categories, not an exhaustive count of duplicate lines or call sites. The language-level rule is duplicated
across the two experimental frontends; eliminating that experimental duplication
is different from eliminating the required boundary checks. The paired local
summary prevents a separate rewrite-only legality interpretation within each
candidate.

For an authority example already implemented in C, changing what the selected
strategy may do or what constitutes verified handback reaches the following
existing seams in both candidates:

- xDSL [grant dialect](xdsl_candidate/grant_dialect.py), [contract](xdsl_candidate/grant_contract.py) and [adapter](xdsl_candidate/grant_adapter.py): typed handback, exact delegated/retained split, closed phase protocol, installed profile and prepared claims.
- Pliron [grant dialect](pliron_candidate/native/src/grant/dialect.rs) and [verification/preparation](pliron_candidate/native/src/grant/mod.rs): typed phase chain/offer, the same split and phase order, installed offer, and complete expected-plan correspondence.
- Both [grant target](grant_target.py) and original owner/grant services: actual phase history, state restoration, final gradient and verified transfer before retained advancement. A legal static phase list does not prove a selected callable performed those actions.

This is a retrospective map of the ports and existing corruption/handback work,
not a new authority feature. Static meaning, installed correspondence and actual
runtime handback are three separate checks of related claims. Frameworks enforce
IR structure; they cannot infer that a Python callable restored the model.

## Fixed diagnostic sample

[Sixteen actual errors](evidence_comparison/diagnostics.json) cover eight paired
samples, including identity/access and authority/lifetime separately. The initial
lifetime selector's `AttributeError` was a collector defect, not a verifier result;
[that initial probe is retained](evidence_comparison/diagnostics_probe_initial.json).
The corrected sample contains only actual `Rejected`/`NotReady` exceptions.
The collector now refuses to overwrite either the diagnostic data or its
provenance sidecar and accepts a separate output directory.

| Violation and boundary | xDSL message quality | Pliron message quality |
| --- | --- | --- |
| Missing required map binding; prepare | Names `data.double`, symbol and required `map/pure` role | Names `data.double` and missing symbol; omits required role |
| Wrong pack output payload; accept | Names `pair.pack` and required record-to-vector shape | Names `pair.pack` and explicit expected/provided input/output types |
| Wrong frozen identity/version; prepare | Names `pair.pack` and resource; terse incompatibility | Names source, target, resource and expected/provided version/digest |
| Wrong resource access; prepare | Names source; says resource declaration differs from lowering | Gives expected/provided binding/schema/resources/outcomes, with a verbose combined message |
| Unsupported A count; prepare | Names `window` and supported bounds | Says accepted but unsupported and lists target limits; no source identity |
| Forged completion; cold | Names `model.encoder` and binding/completion mismatch | Broad semantic/source/binding/address/protocol correspondence failure; no differing field/source |
| Missing restoration phase; accept | Names selected activity and incomplete/reordered protocol | Names `phase.4.restore` and expected/provided phase |
| Consumer ignores pending evaluation; accept | Names `consumers` and expected/provided pending policy | Names `consumers` and expected/provided pending policy |

Neither currently maps these diagnostics automatically to an originating Python
file/line or a user activity selection. Fixture source IDs provide attribution;
parser locations refer to structural text. Most semantic errors are project
strings, and native errors cross as JSON embedded in an exception message.
Pliron's more detailed frozen-resource diagnostic is useful; its generic cold
mismatch loses context. xDSL is also inconsistent about expected/provided detail.
Improved source mapping and structured error propagation are concrete integration
work for either substrate, not inferred framework features.

## Costs and integration are separate observations

**Initial implementation cost:** retained commits show registration, validators,
bridges and owner-target extraction, but no reliable person-hour measurement
exists. The cached release build is recorded separately. Lines of code and Rust
versus Python are not quality scores, and no fresh install/build comparison was
performed.

**Ongoing maintenance:** both retain fixture-specific domain validators and
prepared protocols; A/C/mixed repeat generic collection/recipe patterns that are
candidates for future consolidation. xDSL also owns its dependency cache and the
noted equivalence adaptations. Pliron requires Rust dialect/build maintenance,
version-compatible native artifacts and the Python contract/manifest boundary.
Its reusable region, rewrite and analysis infrastructure is concrete; it does
not erase semantic or bridge maintenance.

**Machinery ergonomics:** Python fixtures, inspection and project checks are
local in xDSL. Pliron supplies explicit native region semantics and managed
analyses, but registration and semantic diagnostics cross language/process
boundaries. The current contracts are experimental and case-specific; neither
implements a finished authoring language or broad artifact compatibility.

**Python/native integration:** both register selected Python implementations and
versioned contracts. Pliron currently serializes requests/results and starts one
process per request; that is an implementation choice, not an intrinsic framework
cost. Its Python parents still import base xDSL definitions through reused target
fixtures. xDSL lifetime imports shared Pliron-named binding/runtime modules while
keeping its IR path independent. An in-process binding is unmeasured and not
implemented. Backend reuse still lacks a demonstrated common descriptor,
analysis, compilation interface or artifact beneficiary; its separate 0.19.0 pin
is not changed by this 0.20.0 experiment.

## Measurement record

The [raw observations](evidence_comparison/measurements.json) retain 2,800 measured
phase samples, 420 warm-up samples and 120 fresh-process starts. All process and
sample failure counts are zero. Every completed workflow sample passed its
independent oracle outside the timer. [The compact summary](evidence_comparison/summary.json)
retains medians/IQRs and memory definitions. The next tables report milliseconds.

The three warm-ups and twenty measured paired runs alternate candidate order.
Fresh starts use ten runs per candidate/case. Both use the existing environments,
ordinary imports with the GIL disabled, one numerical thread, normal GC and
unflushed OS caches. Timing is sequential within one driver; the host is not
CPU-isolated. One diagnostic capture ran concurrently during early collection;
all observations, including wide IQRs, remain retained. No resampling or favorable
subset was used. Fresh-start spreads are especially unsuitable for small ranking
claims.

The exact measured [case adapter](evidence_comparison/measured_sources/cases.py.source)
and [driver](evidence_comparison/measured_sources/measure.py.source) remain hashed
with the raw results. Subsequent changes bind immediate loop callbacks, make constructor type flow
explicit, add a separate-output option and add the separately declared natural
endpoint below. They do not
replace the recorded samples. The current collector refuses to overwrite an
existing result.


### Endpoint qualification correction

The first collector's `end_to_end` label was broader than its implementation:
for B/extension/A/C/mixed it converted preparation to a saved artifact and then
cold-reconstructed it before execution. That repeats preparation for coupled
xDSL APIs. Those rows and fresh-process starts remain valid **cold-roundtrip**
observations, retained with their original labels and source hashes. They are
not evidence for the natural author-to-execution hot path or an intrinsic
framework speed advantage.

The separately frozen [natural-endpoint protocol](evidence_comparison/endpoint_protocol.json)
collects the five affected cases using each public API's returned prepared
object directly when available. Native owner frontends still need their actual
materialization/verification boundary. Lifetime already used its natural
prepare/pressure path in the first collection, so its valid samples were not
resampled. The [additional raw endpoint observations](evidence_comparison/natural_measurements.json)
remain a distinct collection; no failed/noisy sample was removed, pooled away
or replaced, and differences between the two endpoints are not optimization
gains. This correction is part of the retained collector history.

The additional endpoint retains **360 individual measured observations** across paired repetitions, 54 warm-up observations and
100 fresh starts. It has no process failures. Values below use the **primary unfused policy**, median (IQR), ms;
lifetime uses its original valid natural endpoint, marked separately. Fused
natural rows remain separate in `natural_measurements.json` and are not pooled.

| Case | xDSL natural | Pliron natural | xDSL fresh natural | Pliron fresh natural |
| --- | ---: | ---: | ---: | ---: |
| B | 16.733 (1.504) | 10.950 (0.689) | 331.904 (56.940) | 339.735 (58.063) |
| extension | 26.823 (4.165) | 14.450 (8.394) | 384.276 (35.458) | 346.848 (50.410) |
| A | 52.638 (5.992) | 32.611 (3.618) | 2928.024 (98.498) | 2855.748 (215.334) |
| C | 15.034 (0.983) | 16.121 (1.219) | 2974.616 (242.878) | 2926.775 (201.387) |
| mixed | 48.932 (2.072) | 34.136 (1.793) | 2995.440 (165.399) | 2924.981 (153.040) |
| lifetime (original) | 112.612 (14.670) | 33.430 (2.209) | 2932.701 (163.275) | 2824.191 (136.572) |

In this corrected endpoint C is close, with the xDSL median slightly lower;
the other five cases have lower Pliron medians in their recorded collections.
Fresh-start spreads, differing public API choreography and tiny CPU workloads
still prevent an intrinsic framework ranking. The cold-roundtrip and natural
tables measure different operations and must not be pooled or subtracted.

### Recorded API and cold-roundtrip costs, primary unfused policy

Median (IQR), ms. These are overlapping API boundaries, not additive stages.

| Case | Frontend | Accept | Prepare | Cold | Materialize | Cold roundtrip execution |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B | xdsl | 1.575 (0.257) | 12.081 (0.568) | 11.610 (0.853) | 11.211 (1.225) | 26.263 (1.224) |
| B | pliron | 3.713 (0.385) | 5.340 (0.475) | 4.689 (0.627) | 4.599 (0.233) | 14.284 (1.080) |
| extension | xdsl | 1.826 (0.240) | 15.773 (0.627) | 14.261 (0.674) | 14.074 (0.658) | 33.381 (1.683) |
| extension | pliron | 4.322 (0.444) | 5.999 (0.276) | 4.921 (0.505) | 4.874 (0.391) | 16.244 (0.434) |
| A | xdsl | 2.952 (0.248) | 33.130 (0.756) | 31.969 (6.181) | 31.333 (3.748) | 78.102 (5.736) |
| A | pliron | 4.723 (0.375) | 5.885 (0.350) | 7.671 (0.505) | 9.329 (0.508) | 27.444 (2.265) |
| C | xdsl | 1.519 (0.191) | 18.802 (3.452) | 17.207 (2.447) | 16.894 (6.368) | 38.034 (23.552) |
| C | pliron | 4.891 (0.332) | 5.644 (0.494) | 7.138 (0.800) | 7.823 (0.795) | 19.308 (2.450) |
| mixed | xdsl | 3.143 (0.502) | 31.280 (1.068) | 29.295 (2.066) | 28.938 (1.233) | 87.331 (21.217) |
| mixed | pliron | 5.451 (0.332) | 6.394 (0.430) | 8.670 (0.541) | 9.829 (0.358) | 35.523 (9.712) |
| lifetime | xdsl | 16.604 (0.613) | 25.593 (0.920) | 60.882 (24.111) | 63.776 (25.666) | 112.612 (14.670) |
| lifetime | pliron | 4.825 (0.420) | 5.803 (0.279) | 8.857 (6.892) | 11.012 (5.298) | 33.430 (2.209) |

### Author construction and artifact JSON

Median (IQR), ms; whole-artifact JSON is not IR parsing.

| Case | xDSL author | Pliron author | xDSL JSON | Pliron JSON |
| --- | ---: | ---: | ---: | ---: |
| B | 0.491 (0.045) | 0.029 (0.009) | 0.042 (0.004) | 0.093 (0.013) |
| extension | 0.598 (0.049) | 0.037 (0.009) | 0.058 (0.004) | 0.127 (0.005) |
| A | 0.944 (0.086) | 0.050 (0.013) | 0.067 (0.006) | 0.131 (0.020) |
| C | 0.602 (0.078) | 0.051 (0.018) | 0.040 (0.006) | 0.092 (0.009) |
| mixed | 1.584 (0.336) | 0.073 (0.040) | 0.073 (0.006) | 0.149 (0.010) |
| lifetime | 0.408 (0.054) | 0.056 (0.011) | 0.101 (0.025) | 0.212 (0.034) |

### Matched local grouping

Median (IQR), ms. C and lifetime have no selected rewrite in this protocol.

| Case | Frontend | Prepare unfused | Prepare fused | Cold execution unfused | Cold execution fused |
| --- | --- | ---: | ---: | ---: | ---: |
| B | xdsl | 12.081 (0.568) | 14.042 (3.647) | 26.263 (1.224) | 29.882 (4.953) |
| B | pliron | 5.340 (0.475) | 5.510 (0.527) | 14.284 (1.080) | 14.640 (1.027) |
| extension | xdsl | 15.773 (0.627) | 18.709 (2.982) | 33.381 (1.683) | 44.385 (17.634) |
| extension | pliron | 5.999 (0.276) | 6.469 (0.445) | 16.244 (0.434) | 17.683 (5.394) |
| A | xdsl | 33.130 (0.756) | 54.919 (18.524) | 78.102 (5.736) | 86.484 (11.928) |
| A | pliron | 5.885 (0.350) | 7.211 (1.575) | 27.444 (2.265) | 29.048 (2.604) |
| mixed | xdsl | 31.280 (1.068) | 42.534 (12.759) | 87.331 (21.217) | 92.050 (9.816) |
| mixed | pliron | 6.394 (0.430) | 7.313 (2.177) | 35.523 (9.712) | 34.391 (4.335) |

### Native spans and fresh processes

Native spans below are accept-request medians in ms, inside fresh native processes;
fresh Python columns are median (IQR), ms, for the recorded cold-roundtrip invocation.

| Case | Native request body | Native outcome serialization | xDSL fresh | Pliron fresh |
| --- | ---: | ---: | ---: | ---: |
| B | 1.747 | 0.006 | 345.788 (155.415) | 321.684 (28.448) |
| extension | 2.109 | 0.008 | 324.975 (29.878) | 341.169 (68.455) |
| A | 2.542 | 0.008 | 2800.240 (90.499) | 2670.183 (259.236) |
| C | 2.231 | 0.006 | 2777.157 (291.585) | 2680.731 (225.809) |
| mixed | 3.167 | 0.009 | 2911.089 (71.627) | 2772.615 (167.648) |
| lifetime | 2.513 | 0.008 | 2932.701 (163.275) | 2824.191 (136.572) |

Native prepare/cold bodies and parent JSON encode/decode/subprocess spans remain
separate in the compact/raw records. The cached release build took **30.16 s**
in this environment, including its current cache/dependency state; it is a
development observation excluded from every run-time sample.

### Fresh-process peak RSS

Median of per-process peaks, MiB; native children are separate from their parents.

| Case | xDSL parent | Pliron parent | Native child |
| --- | ---: | ---: | ---: |
| B | 39.7 | 39.5 | 4.6 |
| extension | 39.7 | 39.7 | 4.6 |
| A | 570.3 | 569.2 | 4.8 |
| C | 568.3 | 567.3 | 4.6 |
| mixed | 571.4 | 570.4 | 5.0 |
| lifetime | 569.8 | 568.8 | 4.8 |

Author construction compares xDSL IR-object construction with Pliron text
construction. Acceptance already includes each API's printing/reparsing and
verification. The artifact phase measures whole-artifact Python JSON encoding
and decoding, not isolated IR parse/print. Preparation/cold/materialization
remain labelled according to the actual APIs: xDSL A/C/mixed includes owner
realization; B/extension includes local lowering on both; Pliron materialization
includes another native load request. These phases overlap and must not be added
to obtain an end-to-end result.

Opt-in native spans measure the body of the existing request handler (stdin read,
request deserialization, IR work and response construction) and Rust outcome
serialization separately. Python encode/decode and subprocess wall spans are
also retained. Neither subtraction nor repeated one-request processes establishes
native process-startup time or warm in-process Rust performance. The native body
still contains request I/O and is not a pure parser/verifier microbenchmark.

Fresh-process parent peak RSS and each native child's `VmHWM` are separate
per-process peaks. They are never added as a simultaneous total. Parent peaks
include imports/Torch/harness and cannot be called framework IR allocation.
Retained IR allocation, isolated startup, factory-only time and every isolated
print/parse phase remain unmeasured. No GPU, distributed or native numerical
backend performance is established.

### Provenance and review follow-ups

The natural and diagnostic collections did not originally record independent
binary/compiler/CPU fingerprints. Their inherited phase-collection metadata is
therefore a **historical provenance gap**. A [post-collection audit](evidence_comparison/post_collection_provenance.json)
found the then-current binaries byte-identical to the first collection, before
review-fix rebuilds; it cannot prove their identity at every earlier sample.
Those cohorts remain unchanged and are not retroactively backfilled or resampled.
Future natural and diagnostic collections capture the same binary, toolchain,
CPU and build-environment metadata as the phase collector.

The [initial review](evidence_comparison/review_initial.json) validated the
reported numbers and endpoint separation, then identified retention, labeling
and provenance gaps. Follow-ups add diagnostic overwrite protection/output
routing, the natural table's explicit unfused label, and future provenance.
Both drivers now retain subprocess timeout/launch failures, including partial
output, and natural workers retain individual failed/GIL-regression rows.
Focused regressions exercise these failure paths without recollecting timings.
The [CLI validation receipt](evidence_comparison/review_fix_validation.json) is
hash-pinned with the other artifacts. The [follow-up review](evidence_comparison/review_followup.json)
confirmed the five resolutions and table values; its small receipt-pin gap is
closed here. The phase collector now also rejects warm-up-only failures, matching
the natural collector. These guard changes do not alter the retained samples.
The profiling bridge intentionally normalizes error types; diagnostics use the
original bridge methods. Native serialization times serde's tagged `Ok`/`Err`
outcome, so a failed request's serialization is an error envelope rather than a
successful payload. The collections contain no such failed request medians.

The current isolated [progress gate](evidence_comparison/isolation.json) passes
both returning-filter/actual-feedback witnesses with the GIL disabled. Every
measurement worker records actual import/GIL/CUDA/Triton state; the legacy
standard-SGD/Triton import path is absent. This is reproducible isolation for
these observations. It does **not** fix or explain the older combined-suite
failure tracked in `sd-scripts-syv.31.23`, which remains open.

The original cold-roundtrip endpoint shows lower Pliron medians in this protocol;
standalone acceptance is often slower through the CLI. These rows do not measure
the natural hot path of coupled xDSL APIs. Different
API choreography, repeated validation, authoring boundaries, host noise and
shared runtime costs limit attribution. These are current implementation costs,
not an intrinsic framework speed ranking. This collection leaves the substrate
recommendation and reversal conditions to `sd-scripts-syv.31.25.8`.

## Reproduce

```bash
export CARGO_TARGET_DIR=/tmp/sd-scripts-pliron-b-target
cargo build --release --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml --bins
/tmp/sd-scripts-xdsl-b-env/bin/python \
  -m docs_design.models-strategy-trainer.evidence_comparison.measure \
  --binaries "$CARGO_TARGET_DIR/release" --out /tmp/new-framework-comparison
/tmp/sd-scripts-xdsl-b-env/bin/python \
  -m docs_design.models-strategy-trainer.evidence_comparison.natural \
  --binaries "$CARGO_TARGET_DIR/release" --out /tmp/new-natural-endpoint
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.evidence_comparison.diagnostics \
  --binaries "$CARGO_TARGET_DIR/release" --out /tmp/new-diagnostic-comparison
```

Use a separate output location for a new collection. Compare its source hashes
and protocol with this record; keep failures and avoid selecting a favorable run.
The timing drivers accept `--torch-free-python` and `--owner-python` to select
existing interpreters on another machine; the shown paths are this environment.
The current collectors preserve the documented case meanings and separate endpoint boundaries;
this record's exact measured source snapshots remain available above.
