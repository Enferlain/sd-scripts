# Candidate evidence and reusable expectations

Research assessment · 30 October 2026 · `sd-scripts-syv.31.18`

This prepares comparison with another candidate. **The language design and final
implementations remain open.** The original execution candidate and the xDSL
experiment implement selected cases; neither establishes the production language,
authoring interface, complete semantic grammar, extension ABI or runtime design.
The comparison expectations below are test-case meanings, not a new language or
shared production protocol.

The governing [OpenSpec change](../../openspec/changes/rework-model-strategy-trainer/)
retains authority over existing requirements. The [IR research report](../../openspec/changes/rework-model-strategy-trainer/research/ir_architecture_report.md)
supplies proposals and experiment criteria. This assessment neither promotes the
report's grammar sketches nor changes governing requirements. It also does not
select custom infrastructure as a required destination.

## Decisions that must remain separate

| Question | Current position | Still open |
| --- | --- | --- |
| Required behavior | Accepted meaning survives authoring; readiness, state authority, input association and failure meanings remain explicit. | How a complete language expresses and composes these obligations. |
| Authoring | Python ecosystem integration is the report's premise. | The author-facing constructs, syntax, types, validation experience and extension mechanism. Python hosting does not answer these questions. |
| Semantic representation | Both experiments retain an accepted hierarchy. xDSL additionally exercises typed data composition. | The common operation families, region semantics, flow/cardinality rules, effects and valid combinations. |
| Prepared representation | xDSL makes a selected realization inspectable and reconstructable; the original derives executable coordination. | The production plan vocabulary, its correspondence rules, versioning and compatibility promises. Distinct acceptance/readiness does not prescribe a particular storage format. |
| Structural infrastructure | Custom objects and xDSL have concrete experimental implementations. | Framework selection and machinery implementation language. A framework can inform the language design without defining its semantics for us. |
| Execution and ownership | The original owner services supply substantial evidence; xDSL A/C/mixed reuse them. B has a separate, small data-only realization. | The final runtime/service boundaries and implementation. Reuse in an experiment does not adopt those classes as production APIs. |
| Numerical execution | Selected numerical work can remain behind declared bindings. | Backend integrations, device ordering/completion and target-specific qualification. Framework selection does not select a numerical compiler. |

The useful result so far is feasibility evidence for hierarchical activities,
typed flows, persistent resources and separate preparation. A complete language
design still has to give supported compositions precise meanings. Existing owner
requirements constrain that design; fixture constructors and dialect names do
not settle it.

## What the existing experiments establish

The [original assessment](execution_candidate/system_assessment.md) and
[xDSL record](xdsl_candidate/README.md) contain the detailed evidence and dated
limits. The following conclusions concern these repository implementations,
not every possible custom or framework-backed system.

| Concern | Original candidate | xDSL experiment | Bounded conclusion |
| --- | --- | --- | --- |
| Whole-run execution | Accepted relationships derive calls, coordination, scoped access, requests and stop relationships. | A/C/mixed lower into that owner engine; B runs without training or Torch. | Whole-run meaning can cover data and model work. There is no independently implemented second model-owner runtime here. |
| Structure and reconstruction | Project records, explicit links and custom traversal; prepared closures/services and inspection output. | IRDL operations/types, regions, SSA connections, canonical text and symbolic binding manifests. Fresh-process checks disable authoring builders. | xDSL supplies useful structural infrastructure. Durable restoration and stable artifact compatibility remain unqualified. |
| Semantic/prepared correspondence | Preparation checks and derives a realization from the accepted graph. | Decoded bounds, source ancestry, stage completeness, bindings and fixed protocols are checked before owner construction/lowering factories. | Prepared text needs domain correspondence checks in addition to structural verification. The external-review counterexamples and repairs are part of this evidence. |
| Data composition | Providers expose their relationships while much internal work remains a selected implementation. | Recursive unary local regions expose data stages beside model work. B and mixed exercise feedback, bounds and explicit stop. | Selected data composition works. General joins, changing cardinality and arbitrary lifecycle nesting have not been established. |
| Rewriting | Calls/inline and access specializations retain the same accepted graph. | A real prepared rewrite groups eligible local calls; guarded execution preserves leaf validation and failure. | Framework rewriting is useful on this case. The prepared calls have no SSA results, so SSA result remapping has not been exercised by this pass. |
| Extension | Closed probe vocabulary plus selected implementation seams. | An independent operation, typed pair, frozen resource and lowering use common local interfaces. | Local extension works without core operation-name dispatch. Whole-session profiles and owner adapters remain fixture-specific; mutable resources and general region/cardinality interfaces are unresolved. |
| Runtime obligations | Broader evidence covers pending work, retained access, multi-unit outcomes, workers, RNG and input lifetimes. | Selected cases preserve those obligations; archive-erasure tests also cover fusion and cancellation with physical failure. | The original runtime evidence carries forward substantially through reuse. Broader xDSL parity is a separate investigation. |
| Progress and outcomes | Ownership distinctions and physical draining have regression evidence. | The latest fixes qualify excluded-selection progress, mandatory local runtime outcomes and retirement under optimized Python. | These are concrete target/ABI fixes. A quantum of 64 and the current outcome labels are not proposed universal language constructs. |
| Costs | Recorded local dispatch, access and worker probes have specific workload/environment limits. | A tiny fusion probe separates preparation from local dispatch. | There is no matched cross-candidate performance or total complexity result. Test totals are not comparative scores. |

The report's Section J experiment has substantial bounded evidence: A/B/C,
author-independent reconstruction, an actual transformation, independent checks
and failure/physical-lifetime pressure. Its common-composition question remains
partly answered because each whole-session profile still supplies dedicated
verification and lowering. The mixed experiment and independent local extension
strengthen that answer without demonstrating a complete recursive language.

## Expectations that can be shared now

Share the meaning of a selected case and the evidence needed to check it. A
future candidate can use its own representation and test adapter. It does not
need the existing `Ready`, `RunState`, dialect classes, emitted source or exact
private scheduler trace.

| Expectation | Evidence the adapter must retain | Useful negative control |
| --- | --- | --- |
| Actual input/source association | Logical consumer/member/key plus the binding and numerical state actually used. | Change the association while keeping the numerical result unchanged. |
| Independent computation | Case-defined inputs, losses/gradients, updates and intermediate products where externally relevant. | Make both lowerings share a wrong implementation; independently calculated expectations must reject it. |
| Feedback causality | Which observation authorized an update and which applied policy a later selection actually used. | Delay, duplicate, omit or relabel feedback without changing a convenient final scalar. |
| Addressed partial outcomes | Contribution, entered/returned/uncertain/unattempted advancement and reset, affected availability and pending work. | Relabel an uncertain or unattempted unit as returned while retaining the same weights. |
| Physical lifetime | Selected entry, actual completion/failure, protection release and final disposition. | Release protection before a held physical use finishes; cancellation must not erase its later failure. |
| Progress and bounded work | Outstanding admission across handoffs/control, exclusion disposition, stop and settlement. | Fill transport or keep returning exclusion results while heartbeat, feedback or cancellation must progress. |
| Accepted/prepared correspondence | Source/role, binding, resource/schema/outcome requirements and target support. | Corrupt a decoded plan while leaving it structurally valid; reject before constructing owners or invoking lowering factories. |

These are separate checks. A plausible final weight does not prove correct
association, handback or advancement. Agreement between two lowerings sharing
an engine or binding does not independently establish correctness. Clock values
in current reports encode required causal relationships; their exact values and
incidental interleavings are not comparison requirements. Observations must come
from real execution; an adapter must not manufacture missing completion or
provenance evidence to satisfy a checker.

## Existing case meanings and checker entry points

These workloads differ. Comparing the original adaptive result directly with
A's final weight would compare different algorithms. Select the same case and
parameters on both candidates before making a numerical or cost comparison.

| Case | Selected meaning | Existing independent checker |
| --- | --- | --- |
| Original adaptive learning | Keys 1/2/3; encoder factor 1.5; squared loss, SGD at .05, and next factor `1/(1+loss)`. Checks input/source and optimization associations, gradients, weights and feedback order. | [conformance.py](execution_candidate/conformance.py): `adaptive_reference`, `check_learning`. [Test adapters and corruption controls](execution_candidate/test_conformance.py) observe actual owner facts. |
| Original partial advancement | A's update/reset returns; B's entered update is uncertain; C is unattempted. Those outcomes and unavailable state are checked independently of A's returned weight. | [conformance.py](execution_candidate/conformance.py): `check_partial`. Unit order is a case choice, not a universal optimizer schedule. |
| B: data-only feedback | Select `x=k+bias`, then `2*x`, `+1`, square and acknowledged publication; apply `score%5` policy feedback. Default outstanding bound two; explicit external stop and physical drain. | [conformance.py](xdsl_candidate/conformance.py): `check_b`. It accepts mapping-shaped facts without framework/runtime imports. |
| A: correlation and accumulation | Samples 1/2/4/5, explicit window/slot mapping, two contributions per window, model-assisted preparation and requested evaluation. An evaluation may observe either permitted current-at-start epoch. | [model_conformance.py](xdsl_candidate/model_conformance.py): `check_a`, `check_admission_failure`. A rejected join must claim neither member. |
| C: temporary authority | First backward, temporary edit, intermediate reset, second backward, restoration, verified handback and retained advancement. Restoration alone cannot authorize advancement. | [grant_conformance.py](xdsl_candidate/grant_conformance.py): `reference`, `check_c`, `check_unverified`. Phase order and optional clipping are case parameters. |
| Mixed feedback/model ownership | Three learning windows; the same participant supplies protected preprocessing reads and receives updates. Feedback changes selection; actual production state may be older than admission state under the selected rule. | [mixed_conformance.py](xdsl_candidate/mixed_conformance.py): `check_mixed`. Numerical reconstruction uses the observed protected production epoch, not an assumed latest state. |
| Local pair extension | Read frozen coefficients `(1,2)`, construct `(v,2*v)`, then select the first lane. B's final scalar stays unchanged. | [extension_conformance.py](xdsl_candidate/extension_conformance.py): `check_extension`. The hidden second lane must be checked separately from B's scalar result. |

The six checker modules already avoid runtime/framework imports. There is no
need to extract another common library in this preparation step. Their fact
classes/report dictionaries are experimental test interfaces, and some fields
still need a candidate-specific projection. This assessment does not freeze
those shapes or assert that all checks share one portable report schema.

Reuse requires three things: the selected case definition, a truthful adapter
from the new implementation, and the corresponding positive/negative checks.
Case labels such as `data.double` can be mapped to equivalent source roles;
that mapping must preserve stage attribution rather than hide substitutions.
Where an existing checker is too tied to a private report shape, preserve its
independent arithmetic and causal expectations while adapting the test seam.
Record unsupported behavior instead of inventing a successful observation.

Run the assertion-based checkers with assertions enabled. Optimized-runtime
qualification uses a separate normal-Python parent checker; `python -O` is not
a valid way to run these oracles.

## Language and implementation questions still to resolve

- Which composition operations exist, and what startup, completion, failure,
  cancellation and draining behavior each enclosing region defines.
- Which flow/cardinality, correlation, exclusion and missing-member rules form
  common contracts, and which belong to domain extensions.
- How types, entity uses, effects, aliases and changing source state expose
  enough meaning for legality checks without turning private mathematics into
  mandatory common nodes.
- How extension interfaces declare semantics and target support, including
  unknown behavior and unsupported combinations. The current local traits are
  evidence for a small interface, not a complete plugin specification.
- What authoring produces, what normalization preserves, and how accepted
  meaning corresponds to a selected prepared realization across version changes.
- Which owner/runtime implementation boundaries realize those contracts, and
  how alternative numerical/device targets establish ordering and completion.

These questions can be informed by another framework experiment. They do not
require a finished public language before comparison, and successful execution
of a few profiles must not silently settle them.

## Basis for testing another candidate

The preparation gate is the evidence/expectation inventory above, supported by
the [current verification record](candidate_comparison_evidence.json). Another
candidate can now be screened using the same case meanings. A small existing
case can establish structural fit before committing to A/C/mixed owner
integration or a full deep dive. It need not reproduce xDSL's dialect syntax.

Use the proposed new whole-workflow composition case only as a later common
pressure test for promising candidates. Broader original-candidate parity and
matched cost measurements also remain later work. A framework decision should
consider semantic fit, generic infrastructure actually reused, missing custom
analyses and machinery-development ergonomics together. This assessment leaves
the winner, language design and final implementations unselected.

## Verification boundaries

The verification record retains the exact independent-oracle replay driver,
its result, existing-test command/results, source hashes and review outcome.
The replay loads six checker modules with Torch, xDSL and named runtime imports
blocked, validates nine saved reports and rejects six deliberate corruptions.
The original adaptive reference also passes its checker. Saved-report replay
checks checker reuse against historical observations; it does not re-execute
those archived runtimes or requalify their manifests.

The existing original conformance tests and xDSL research suite are also run
against the current source. That supplies real positive executions and broader
negative controls in addition to saved-report replay. No new runtime, general
comparison ABI, language design, production code or governing-spec change is
introduced by this preparation.

The required independent review attempt failed during analysis with a provider
quota error (HTTP 429). It supplied no review approval; the retry remains tracked
in `sd-scripts-syv.31.18`.

Graph tools returned `Transport closed`, including the coverage check. This
assessment therefore uses direct source/test inspection and executed checks;
it makes no exhaustive graph-coverage claim.
