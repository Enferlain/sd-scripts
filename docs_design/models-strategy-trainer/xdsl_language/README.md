# Shared xDSL language slice: milestone 6

This isolated package implements structural construction, normalization and an
executable lifecycle reference for the bounded slice in
[D14](../../../openspec/changes/rework-model-strategy-trainer/design.md) and
[workflow-composition](../../../openspec/changes/rework-model-strategy-trainer/specs/workflow-composition/spec.md).
Training and data-only work use the same operation definitions and reference
protocols. The extra consumer scope is another composition of those definitions.

**Normalization is not whole-lifetime acceptance or executable readiness.**
Milestone 7 supplies the shared lifecycle/ownership verifier. Milestone 8
derives prepared execution through the existing owner engine; milestone 9
qualifies integrated interruption/interleavings. This package currently does
not prepare or execute that owner target, train a real model, or select public
APIs, a numerical backend or a production migration.

## Representation and contract queries

[dialects.py](dialects.py) defines actual xDSL operations, types and regions.
Persistent resource and endpoint references are typed SSA values; records and
live model state are outside the IR. Independently invoked bodies use explicit
capture operands and block arguments. xDSL verifies types, region structure,
isolation and use tracking; local project traits check capture signatures.

The installed traits in [contracts.py](contracts.py) return typed descriptions:

| Query | Derived information |
| --- | --- |
| `RegionSemantics` | Once/sequence, parallel join, serial repeat, requested-handler, producer, scoped-access and finite-choice protocols; actual entry/yield/capture types; invocation sources and bounds; closure/interruption/settlement rules |
| `ResourceSemantics` | Actual resource operand, explicit access realization, separate freshness/version-selection policy and required use extent |
| `FlowSemantics` | Actual payload type, aggregate outstanding/ready bounds and owner-specific admission policy with lineage metric and comparison boundary |
| `ReferenceSemantics` | Typed request-source, request-destination, version-owner and stop references exposed for normalization and later dependency analysis |
| `RealizationSemantics` | Versioned selected binding, actual input/output types and required physical/validation boundary |

A shared `Decision` description carries reproducibility/rank declarations and
assumptions through region, flow and resource queries. Region contracts also carry explicit quiescence declarations;
an empty declaration means none is offered. These are preserved contract
parameters, **not verified reproducibility, distributed-execution or restoration
claims**. Their composition checks are milestone 7 work.

Semantic symbols (`SymbolOpInterface` / `sym_name`) are distinct from source
origins. This first slice uses one program-wide flat symbol namespace;
namespaced modules and cross-program linking are not implemented. Changing a
diagnostic origin does not retarget a semantic reference. Invocation identities
combine parent invocation, static origin and a local ordinal; template/service,
repeat-iteration, production-selection and accepted-request entry are recorded
separately.

[normalize.py](normalize.py) clones the accepted document, expands operations
offering the maintained sequence-expansion interface and uses xDSL's rewriter
to remap result uses. It resolves typed references, checks request payload/entry
correspondence and derives ancestry from the actual region tree. Source coverage,
contract versions, bounds, policy references and links survive normalization.
The normalizer checks before/after region, flow, resource and realization
contracts, semantic parameters and actual SSA-provider wiring. Cloned resource
uses compare by their real SSA provider, not Python object identity. Each
completion-request source must belong to its target handler's declared set.
The authoring object is detached. Parsing consumes the whole input and requires
one program; neither trailing garbage nor a second root becomes an accepted
document. Unknown semantic fields and missing registered definitions reject.

The normalizer checks local structure only. It does not certify producer-stop
coverage, all-path affine ownership, capability escape, hold/wait legality,
whole-lifetime cycles, source completeness or target feasibility.

## Reference semantics and observable comparison

[reference.py](reference.py) is a single-caller, Torch-free interpreter over
normalized IR. Each invocation follows the control skeleton
`created -> pending -> admitted -> running -> closing -> settling -> terminal`.
Logical success/failure, stop/cancellation requests, physical call outcomes,
cleanup and product dispositions remain separate facts.

`enabled()` exposes legal next logical commands; `step(choice)` supplies an
explicit scheduling choice. `held` and `failures` supply external physical
completion/failure choices. `request_stop(symbol)` closes new repeat/producer
admission at its selected boundary. `cancelled = True` is an external run
cancellation input. `run()` uses a deterministic round-robin choice and returns
when settled or blocked; callers must check `settled`. No actual threads,
futures, device calls or elapsed-time delays are modeled.

The interpreter derives admissions and requests from IR and owned protocol
state. A handler template creates no admitted request. A completion-triggered
request explicitly names an enclosing source and its `settled_success`
boundary. The source retains the emission obligation; request registration
precedes publication of that source's completion. A request waiting for input
or registration capacity keeps its enclosing lifetime open. Orderly stop
preserves already admitted iterations and their requests; cancellation records
unfulfilled work and drains modeled entered calls. Failure-triggered sibling
closure is distinct from a user cancellation request.
Request-source closure and internal stop relationships resolve within their
actual enclosing invocation. A source that closed in an earlier consumer
iteration cannot close the next iteration's handler.

[observations.py](observations.py) fixes the first comparison contract:

- A fact key contains owner/source origin, invocation path, kind and the
  operation-local occurrence or owned item/request identity.
- Facts retain input associations, request cause, observed resource version,
  logical/physical outcome, update, cleanup and disposition. Trigger provenance
  and the handler's input identity are distinct.
- Required causal edges cover local sequential interactions, child/parent
  settlement, product selection/publication/admission/disposition, completion
  requests and their physical use, and consumer completion before producer stop.
- Absolute timestamps, private queue order, global call counters and transient
  internal state observations are excluded. Values use exact fixture comparison;
  a future numerical target needs its separately qualified numerical policy.

The reference is an oracle, not a runtime trace replay. It never consumes an
observed admitted-work trace to determine expected admissions or requests.
Diagnostic history can be disabled without changing its progress or numerical
results. Its fact projection is test evidence and never execution authority.
Parent settlement also waits for accepted pending requests when no handler body
was invoked. An incomplete source/handler composition can remain blocked;
direct self-referential settled-source waits receive a typed diagnostic.
Structural normalization is not the milestone-7 completeness/cycle check.
Cancellation gives pending requests an explicit disposition.

## Fixture computations and supported reference target

[fixtures.py](fixtures.py) constructs the IR. [standins.py](standins.py) supplies
versioned, typed fixture implementations and domain payload validators through
the same binding interface; the reference has no case-name dispatch or model
arithmetic. The computations operate on copied fixture values and explicitly
declared updates. They are not Python/PyTorch model or optimizer substitutes in
an owner-integrated run.

The model fixture retains the earlier independent arithmetic: two two-member
windows update `0.8 -> 0.77 -> 0.635`, then requested evaluation consumes record
4 at model epoch 2, predicts `6.35` and reports loss `0.91125`. The producer has
an aggregate outstanding bound of three and a separate ready bound of two.
The data-only fixture selects under changing policy, scores `(2*x+1)^2`,
acknowledges feedback and updates the policy through its owner protocol.

The reference currently supports serial count-bounded repeats, serial bounded
handlers, unbounded-total/bounded-live producers, no-result parallel joins,
two-alternative finite choice and declared enclosing-source completion requests.
Its persistent resource/endpoint declarations belong to the program root;
nested persistent declarations are rejected as a reference-target limitation.
It models protected current reads, current-at-start isolated snapshots, initial
published snapshots and ordinary owner updates. It records actual owner,
incarnation, lineage and update ordinal. It realizes any-with-provenance item
admission. Other version-admission evaluation and classification are milestone
7/8 work. Grant profiles, arbitrary published-version selection, parallel result
merging, distributed targets and coordinated cuts produce an explicit reference
limitation or remain declaration-only; they do not become weaker semantics.
Invocation outcomes are retained for inspection in this finite exploration
tool; its live-obligation bounds do not claim bounded total oracle memory.

Protected access excludes incompatible updates through all modeled entered
uses. A snapshot owns an isolated copy after coherent selection and does not
block subsequent updates. Modeled completion, host return, a version stamp or a
successful reference result do not qualify a real resource as usable or prove
device ordering, thread safety, backend interoperability or performance.

## Run the bounded checks

Reuse the existing qualified environment; no root dependency or lockfile update
is needed. A fresh isolated environment can use the existing pinned
[requirements](../xdsl_candidate/requirements.txt), including xDSL **0.74.0**:

```bash
uv venv /tmp/sd-scripts-xdsl-b-env --python .venv-wsl/bin/python
uv pip sync --python /tmp/sd-scripts-xdsl-b-env/bin/python \
  docs_design/models-strategy-trainer/xdsl_candidate/requirements.txt

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/sd-scripts-xdsl-b-env/bin/python -m pytest \
  --noconftest docs_design/models-strategy-trainer/xdsl_language \
  -q -p pytest_asyncio.plugin
/tmp/sd-scripts-xdsl-b-env/bin/ruff check docs_design/models-strategy-trainer/xdsl_language
/tmp/sd-scripts-xdsl-b-env/bin/ruff format --check docs_design/models-strategy-trainer/xdsl_language
/tmp/sd-scripts-xdsl-b-env/bin/ty check --python /tmp/sd-scripts-xdsl-b-env/bin/python \
  --extra-search-path docs_design/models-strategy-trainer \
  docs_design/models-strategy-trainer/xdsl_language
```

Checks include three genuine IR compositions, registered operation expansion,
SSA remapping, explicit-capture corruption, source/symbol separation, complete
document parsing, independent numerical expectations, late input/pending work,
iteration admission versus orderly stop, held physical uses under cancellation,
late failure, retained partial updates, missing/corrupted observable facts and
fresh normal/optimized children with authoring absent. Seeded schedules are
bounded smoke checks; the integrated exploration/qualification gate remains
milestone 9. Observe actual GIL state before/after imports and tests without
forcing it. Interpreter and IR objects remain confined to their calling thread.
