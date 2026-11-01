# Matched controls and xDSL lifetime composition

Research implementation · 1 November 2026 · `sd-scripts-syv.31.25.1`, `.31.25.2`

xDSL independently represents and prepares the existing
[different-lifetime case](../pliron_candidate/composition_case.md). Paired B
controls establish a common starting point for the
[matched comparison](../matched_comparison.md). Pliron C/A/mixed, resource-policy
alignment, matched costs and the framework decision remain open. The scope is
**stateful, heterogeneous data and model workflows**; production language,
runtime APIs, numerical backends and framework adoption remain open.

## Frozen meanings and B controls

[comparison_fixtures.json](../comparison_fixtures.json) records six existing
case meanings and parameters. This is descriptive fixture data, not an executable
graph or common language/plan schema. Each candidate independently constructs and
verifies its operations, relationships and realization.

The [control setup](../evidence_matched/control_setup.json) executes normal and
cold B in each candidate: aggregate horizon two, mailbox capacity one, stop after
four acknowledged feedbacks and both local fusion selections. Independent B
arithmetic, feedback provenance, selected failure, blocked publication and held
physical work through stop/repeated cancellation and later failure are checked
in [test_matched_controls.py](test_matched_controls.py). Extra admitted tickets
and private interleavings may differ.

The opaque control is a valid ordinary **offset** with `may_effect`, followed by
the unchanged `read_frozen` pack and first-lane operation. Both candidates first
execute it successfully unfused, including the actual offset effect and hidden
pair lane. Each then rejects a structurally valid forged group containing double
and offset. Rejection therefore does not depend on pack's resource/schema
restrictions: Pliron identifies the ineligible offset, and xDSL the unsupported
grouping boundary. The broader resource grouping alignment is not implemented.

xDSL B's `prepare` now accepts optional `saved_text` and `saved_manifest`. Supplied
text follows the existing decoded source/role/binding/recipe/signature/group
checks. Correspondence and manifest disagreement reject before resource lowering
factories. This exposes the existing validation path for cold controls; the
scalar runtime is unchanged.
Fresh normal and optimized children exercise this saved-text path with both B
authors and Pliron RPC/acceptance/preparation/load disabled; the normal parent
runs the independent oracle.

## Lifetime representation and preparation

Arithmetic is unchanged: source `x=2*(key+1)`, target `key+1`, initial learner
`0.8`, two two-member windows with **mean** gradients and SGD `0.01`. Updates
produce `0.77`, `0.635`; evaluation of record four observes protected epoch two,
predicts `6.35` and has loss `0.91125`. Cause members two/three are retained
separately from evaluation's own input four.

Actual typed SSA model/stream/completion/request/sample/gradient/measurement
values connect `life.*` operations and nested regions. The producer, consumer
group, repeat, request, evaluator, accesses and child/root completion are
represented. Pending requests extend the group lifetime. Group completion
requests producer stop; root completion waits for both. Observation histories
do not supply these relationships.

| Concern | Framework contribution | Project responsibility |
| --- | --- | --- |
| Definitions | IRDL operations, endpoint types, typed operands/results, block arguments, traits and SSA use tracking. | `ActivityContract` policy fields and activity meaning. |
| Regions | Nested xDSL regions, lexical isolation, terminators and forward SSA references. | This version has no built-in Pliron-style graph/SSACFG distinction. Domain traits mark graph versus ordered bodies; verification checks ordered value/gradient chains. |
| Composition | Tree walking and actual value/region identities. | Pending request/recipient agreement, input/model association, window admission, normalization, protected use, all-and-only child settlement and producer-stop coverage. |
| Preparation | Cloning, generic print/parse and structural equivalence. | Inspectable `life_exec` scope/call/access/request/stop/port declarations and selected-target support. |
| Cold correspondence | Framework comparison against a regenerated expected plan. | Pre-map corresponding result definitions and check their types for graph forward references. The framework still checks actual operands. Typed stop/model/gradient substitutions have rejection controls. |
| Execution | Existing owner engine, inputs, gradients, optimization, requests, protection and physical workers. | Existing scalar provider bookkeeping and fixture gates, shared with the Pliron case. |

[lifetime.py](lifetime.py) verifies xDSL operations and values. It does not call
Pliron's collector, verifier, parser, preparation or RPC. Its target projection
is derived after those checks and passed to the existing adapter located in
[composition_runtime.py](../pliron_candidate/composition_runtime.py). Binding
descriptors, selected implementations, owner witnesses and independent numerical
expectations are shared. That adapter follows independent IR acceptance and
preparation; it is not a common graph deciding semantics beforehand. This is
independent IR machinery evidence, not a second ownership implementation.

The target supports one producer/group, two two-member windows, one evaluator,
horizon three, ready budget two across ports and one physical producer. Wider
windows, cardinalities and horizons can pass semantic acceptance and then be
rejected by the target. Source identities can all be renamed.

Saved `profile` and hash fields are inspection observations. Cold loading parses
both full documents, verifies installed bindings, regenerates the expected plan
and checks correspondence before model factories. Whole-manifest equality
includes unused bindings, so additive registry growth requires regeneration.
The verified profile and callable mapping are snapshotted for execution.
The envelope allows only its defined execution fields and explicit observation
fields (`profile`, `accepted_sha256`, optional `nodes`). Other fields reject
before factories, so an unknown claim cannot be mistaken for verified data.

## Qualification and limits

There are **85 lifetime checks** (56 document/framework, 29 owner) and **30 paired
control checks**. The wider Torch-free gate passes **589 tests with five optional
owner skips**; the owner gate passes **623 tests**. Both observe the GIL disabled
before/after ordinary imports and execution, without forcing it off. These gates
overlap; totals are not unique test counts or framework scores. Ruff, Python-source
formatting and ty cover changed code. Historical Markdown backend snapshots keep
their original generated text and are excluded from formatting.

Normal/optimized fresh children disable both authoring builders and Pliron
acceptance/preparation/load/RPC. The normal parent checks independent arithmetic
and causal observations. An intentionally wrong shared objective executes but
fails the oracle. Malformed envelopes and forged claims reject before model
factories; saved inspection data never supplies missing successful observations.

Late and prefetched input both execute. Evaluation can remain pending after
training without holding learner protection while awaiting input. Holding
evaluation four, ready five and physical six exhausts horizon three. Group
completion stops the producer while six remains active; release waits for actual
drain. Completed evaluation and later failure survive together, including repeated
cancellation. Other witnesses cover partial member acquisition, local stop versus
root cancellation, failed training/evaluation, pre-entry cancellation and live
archive erasure. Shared owner procedures execute through the new xDSL frontend;
no Pliron semantic/prepared program supplies those executions.

[Evidence](../evidence_matched/) retains six live/cold control reports and eleven
xDSL lifetime reports. Independent replay blocks framework/runtime imports,
accepts all seventeen reports and rejects fifteen deliberate corruptions. Replay
checks saved observations; it is not runtime execution qualification.

The old combined-suite progress witness under `sd-scripts-syv.31.23` remains open.
These gates isolate the affected legacy optimizer import path and observe the
GIL disabled. They do not establish that combined suite's cause or repair it.
Capture can log the existing CPython shielded-future diagnostic during injected
failure; owner facts still record physical settlement, failure and release.

[Qualification](../evidence_matched/qualification.json) records exact scope,
commands, sources, preservation, results and implementation review outcome.
The failed provider review and external assessment of the plan remain separate.
The implementation review completed with no blocking correctness, authority or
liveness finding. Its small follow-ups added direct property/operand comparator
checks, explicit envelope fields and fresh B saved-plan children. The capture
helper documents its default-name constraint; semantic multiplicity remains a
selected-target limitation rather than a language-wide prohibition.
General composition, mutable resource replacement, SSA-changing rewrites, native
sharing, matched costs, GPU/distributed completion and durable recovery remain
unqualified. xDSL remains the provisional working baseline.

## Reproduce

Using the existing optional-Torch qualification environment:

```bash
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.xdsl_candidate.lifetime_demo \
  --mode late --dump-dir /tmp/xdsl-lifetime
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.xdsl_candidate.lifetime_demo \
  --mode prefetched --load /tmp/xdsl-lifetime/artifact.json
```

`--fail`, `--cancel` and `--erase-histories` exercise shared pressure controls.
The xDSL lifetime path needs no native executable. The paired B capture also
needs the existing locked Pliron 0.20.0 executable:

```bash
/tmp/sd-scripts-xdsl-b-env/bin/python \
  -m docs_design.models-strategy-trainer.evidence_matched.capture_controls \
  --binary /tmp/sd-scripts-pliron-b-target/debug/rmst-pliron-b
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.evidence_matched.capture_lifetime
/tmp/sd-scripts-xdsl-b-env/bin/python \
  -m docs_design.models-strategy-trainer.evidence_matched.replay
```

Existing Python environments and Torch installation are reused. No Torch/CUDA
download or production migration is part of this slice.
