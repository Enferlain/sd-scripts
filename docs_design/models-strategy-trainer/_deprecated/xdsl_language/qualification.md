# Qualification of the first shared language slice

Milestone 9 tests the shared representation and derived CPU owner target for
stateful, heterogeneous data and model workflows. It qualifies a bounded
language slice. Production API/type placement, migration and the wider language
0.1 remain separate work under section 5 and D14.

The retained [record](evidence_qualification/qualification.json) contains detached
accepted/normalized/prepared documents, exact source and report hashes, actual
child outputs, external choices, process exits and independent parent verdicts.
Normal and `-O` children use the same installed contracts and saved documents.
Their fixture authoring imports are blocked; the parent keeps assertions enabled.
The `schedules/*.json.gz` files retain complete target/reference facts and owner
observations for all 24 schedule pairs, compressed losslessly with gzip.
The wrong numerical implementation has a valid contract and completes, but fails
the independent oracle. A forged target claim is rejected before resource factories.

| Boundary | Witness |
| --- | --- |
| Model, data-only and nested consumer scope | Eight seeds per composition; target yields at invocation admission, acquisition, reservation, publication, take, selected-call entry/return and completion-triggered stop. Reference independently chooses enabled logical commands; it receives no runtime trace. Full consumer observations and required causal edges are compared. |
| Late versus prefetched evaluation input | Delay record 4 until the request exists, or require it ready during window two. A pending/active request keeps the consumer group alive. |
| Local stop before versus after window admission | A pending second-window attempt can decline body entry; a committed second window finishes and requests evaluation. A three-window fixture proves the third is not admitted. |
| Root cancellation and pending request | Cancellation closes broader admission; the owned registration settles independently of diagnostics. |
| Cancellation during actual physical use | A host worker holds measurement; protected current access remains live after cancellation until that worker physically returns or fails. |
| Completed evaluation and later producer failure | Hold production 6, complete and publish evaluation, request producer stop, then fail that held physical call. The product, completed request and producer failure all survive. |
| Consecutive returning exclusions | Initial-published snapshot products stay stale after two independent owner writes. At least 128 consecutive admission rejections coexist with heartbeat progress and root cancellation; their versions and decisions match the independent reference. This exercises admission exclusion, not a new filter language operation. |

The two model arrangements run all eight pressure choices in fresh normal and
optimized children; data-only runs normally in both. The returning-exclusion
case and both negative controls also run in both modes. The fixed external
phase choices are supplied independently to reference and runtime. Finite
seed exploration is not exhaustive model checking, deterministic host scheduling
or a proof of fairness for arbitrary Python implementations.

The qualification fixtures explicitly declare a `products` state resource and
registered publication calls. A measurement or score becomes a real owned
product through the existing `owner_update` protocol. This adds no lifecycle
operation or scheduler. The reference uses independent numerical stand-ins and
the parent checks rational arithmetic/feedback recurrence without taking actual
admissions or recorded values as expected input.

Full consumer-fact equality and causal comparisons cover the normal seeded
runs and orderly stops before/after admission. Other fixed pressure choices
compare independently derived resource values/epochs and request dispositions;
their claim settlement, protected use and failure/drain checks are separate
owner witnesses, rather than equality of the full exception-wrapping traces.

At every checkpoint in erased runs, the harness drops language facts, outcome
archives, owner traces, generated-source diagnostics and feed handoff/discard
lists. Live invocation state is separate from its outcome projection. Final
checks reconstruct values/epochs from owners, updates from unit step/reset
receipts, request identity/disposition from the registration ledger, ticket
settlement from the flow ledger, completion from the source-closure receipt,
and physical drain from the completed worker lifetime and absent live leases.
Request registration is checked before a pending packet can authorize execution.
The remaining owner receipts are bounded-case state/evidence, not a proposed
production retention policy. No report is rewritten to imitate an oracle.

Physical failure and cancellation are inspected through the original engine's
failure envelopes, including their nested causes. The inherited worker helper
can also log a shielded-future exception during cancellation plus physical
failure. Those stderr diagnostics are retained; they are not used to manufacture
a successful outcome. The owner failure, request/ticket disposition and withheld
resource usability remain authoritative.

Both environments are the existing Python 3.14.5t / xDSL 0.74.0 environments.
The CPU target uses the existing Torch 2.14.1+cu130 installation. Every retained
child records actual pre/post import GIL state and its complete module inventory;
no `PYTHON_GIL` setting forces the result. The language-only cohort remains
Torch-free. No installation or root dependency changes are required.

To retain a new full qualification attempt, use an empty directory; the collector
refuses to overwrite an earlier attempt:

```bash
PYTHONPATH=docs_design/models-strategy-trainer:/tmp/sd-scripts-xdsl-b-env/lib/python3.14t/site-packages \
  .venv-wsl/bin/python -m xdsl_language.qualification /tmp/language-qualification-new
```

The source freeze guard covers this package's Python files and the two named
research reports. Detached input artifacts, including the deliberate forgery,
are hashed before child execution and checked again afterward. Documentation
and generated output files are outside that source freeze guard.

Give parallel pytest cohorts distinct `--basetemp` directories. One retained
integrated attempt lost its shared default temporary directory during concurrent
runs; fixture setup and child-output writes failed. Its logs remain in
[development failures](evidence_qualification/development-failures.json), and
the final owner gate uses a dedicated task directory without changing code,
seeds or assertions.

This establishes no new performance, accelerator, distributed or native backend
qualification. The same original owner engine still implements the execution
mechanics. Open items include general modules/linking, parallel result/claim
merging, richer request bounds, generalized resource/access realizations,
quiescence/restoration/distributed decisions, general autodiff and derivative
routing, and production contracts/readiness/migration under G5. The target's
explicit limitations in [README](README.md) still apply.
