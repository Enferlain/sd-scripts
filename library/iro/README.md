# `library/iro`: the IR system

Kuroiro, the IR a strategy is read into, and the system that accepts, checks
and will run it ([design](../../openspec/changes/rework-model-strategy-trainer/design.md)
D14, D16–D18, D21). Operations print under the `iro` prefix. It is built on
xDSL and imports only xDSL, observability fact types and base; Torch loads
only in its device adapters, which arrive with the runtime.

This is the language half (section 13). The front end that reads Python
strategies, fulfillment, preparation, the runtime and device adapters come in
later sections.

## Stages

```text
  written form or (later) Python front end
     |  text.parse            -> accepted strategy
     |  normalize.normalize   expand derived words, place releases,
     |                        resolve what requests, stops and streams name
     |  verify.verify         claims, leases, accesses, calls, lifecycles,
     |                        streams, decisions -> obligations, or a rejection
     |  accepted.accept       -> AcceptedRun: written forms, pins, obligations,
     |                        authority seed
     |  accepted.cold_load    re-derive everything under the pins
  reference.Reference         run a normalized strategy's lifecycles with stand-ins
```

Every stage reports problems as `diagnostics.Rejection`: every independent
problem, each with a rule code, the authored names, where it was written and
a fix. Codes by area:

| Area | Rules |
| --- | --- |
| `iro-text-` | Reading the written form: syntax, names, captures, shapes |
| `iro-norm-` | Names used twice or unresolved, wrong kind, request type or source, expansion that does not end, normalization that would change meaning, a normalized form missing a release |
| `iro-claim-` | A claim settled zero or several times, used after settling, captured into a body that does not run once, used by more than one parallel branch, or used without a claim contract |
| `iro-lease-` | A lease leaving its access; a retained result leaving its activity, released wrongly, or keeping state that an update needs |
| `iro-access-` | `at_request` outside requested work, a nested access that waits for itself, an access to something that is not a resource |
| `iro-call-` | An implementation or materializer not pinned in `uses`, a resource passed without a lease, a call returning claims or leases |
| `iro-life-` | A request outside its source, a stop of something that is not a producer, a producer nothing stops, a stop that does not cover a consumer, points that wait for each other |
| `iro-flow-` | A stream without exactly one producer, admission by an owner production never reads |
| `iro-decide-` | On several ranks, a decision that does not agree |
| `iro-run-` | A pinned definition missing, at another definition version or with another declaration; parts of a saved run that do not follow from its strategy; a pinned implementation not installed at its version and declaration |

## Modules

| Module | Holds |
| --- | --- |
| `words/` | Kuroiro's core operations and types by area: structure, lifetimes, data, access, calls and model use |
| `operation_contracts.py` | Region, flow, resource, realization and decision contracts, read by every stage through traits |
| `composition.py` | The contract: definitions with area, definition version and declaration digest, pins for a run, `extend` for domain vocabularies, the readable printout |
| `text.py` | The written form: printer and parser with exact round trip; a domain word brings its own form through its definition |
| `normalize.py` | Normalization and the relationships it resolves |
| `verify.py` | Verification, `Obligations` and readiness checkpoints (core execution, or a handler's first request), and the reproducibility report |
| `diagnostics.py` | The shared diagnostic |
| `accepted.py` | The accepted run, its authority seed and cold loading |
| `seams.py` | The owner, implementation and device interfaces the runtime will call, and an implementation's `Declaration` |
| `frontend/` | The `iro` namespace authors import; so far the `@implementation` declaration |
| `reference/` | The reference semantics and declared stand-ins |

## Pins

An accepted run pins the definitions its strategy uses and the
implementations it selects, each with its version and a digest of its
declaration (D16). Cold loading refuses a run whose pinned definitions are
missing or changed, and ignores definitions it does not use, so new
vocabulary never stops a saved run from loading. `check_implementations` is
the Trainer's check of installed implementations against the pins; a run
accepted without installed implementations never passes it.
`tests/unit/iro/test_versions.py` keeps the digest of every definition
version and implementation version in `recorded_versions.json`: a
declaration that changes under a recorded version fails it, and
`IRO_RECORD_VERSIONS=1` records only new versions.

## Written forms

A strategy prints in one canonical layout, and printing a parsed strategy
gives the same text. The accepted form is what authors' strategies become; a
normalized strategy says `normalized` after its name, shows its releases, and
marks each statement an expansion created with `from(...)`. The conventions
are D16's.

## Limits of the reference semantics

It realizes every core lifecycle protocol, with these limits, each reported
as `ReferenceLimit`: a repeat runs a number of times, not by policy; grants
are not realized; requests are emitted at settlement, not at completion.
Physical completion of calls is a separate step that tests may hold, fail or
reorder.

## Tests

`tests/unit/iro/` holds the unit tests and `tests/unit/iro/conformance/` the
conformance suite rebuilt from the research fixtures: a training-shaped
lifetime composition, its rearrangement with an extra consumer scope, the
design's data-only control and a filtering composition, each run against
independently computed expectations. `tests/unit/iro/accepted_runs.py` is the
test-only accepted-run builder (D1). `tests/unit/test_dependency_rules.py`
checks D21's dependency rules.
