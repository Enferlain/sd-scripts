# Mixed feedback and changing model state on Pliron 0.20.0

Matched mixed case · 2 November 2026 · `sd-scripts-syv.31.25.6`

This adds independent native representation, verification, preparation and cold
correspondence for the [matched mixed case](../matched_comparison.md). It reuses
the original owner engine, numerical bindings and independent causal/arithmetic
checker. xDSL remains the provisional baseline. Final language, framework,
runtime implementation and numerical backend choices remain open.

## Case meaning

The producer owns increasing keys and policy state. Selection computes
`v = key + applied_bias`; after revision zero, keys divisible by four are
excluded before fanout. The left path computes `2*v+1`. The right path reads
the **same trainable learner** under protection and produces `production_w*v/4`.
Its detached result retains the actual protected binding and numerical epoch.
Admission permits that epoch to precede the epoch used for learning; it cannot
silently replace the produced feature with the latest model value.

The learner starts at `0.8`. Each of three windows claims one correlated pair,
builds `(admission_w*(left+right)-v/2)^2/2`, differentiates, and advances ordinary
SGD at `0.001` before reset. Post-step feedback carries the actual admitted
input association and returned weight. The producer's receiver validates it,
sets `bias = int(abs(weight)*10) % 5`, increments revision and retires the ticket
once. Diagnostic lists never authorize this update.

| Boundary | Required meaning |
| --- | --- |
| Selection | Observe every feedback update already applied; other admitted tickets may precede their predecessors' acknowledgements. |
| Admission bound | At most three selected, unsettled tickets across work, ready products and pending feedback; each input port has capacity one. |
| Correlation | Validate both ticket/ordinal members and the actual production binding before claiming either. |
| Model read | Hold protection through physical return and validation, then release it before bounded publication. |
| Checkpoint | Three returned learning updates reach the declared external-stop checkpoint. |
| Producer shutdown | Stop selection, discard unclaimed ready products, join private tasks and physical work, and label tickets closed without inventing feedback. |
| Run completion | Both consumer settlement and independent producer draining finish. |

The existing [whole-system graph](../xdsl_candidate/mixed_workflow.svg) shows the
shared state, feedback and containment. Its meaning is reused; one scheduling
trace or equal final counts across implementations are not required.

## Representation and machinery

`rmst-pliron-mixed` is a new executable in the existing package pinned to
**`pliron = "=0.20.0"`**. Cargo dependencies, LLVM requirements and the separate
backend pin are unchanged. Native A's model/unit/stream/gradient operations and
shape interface are reused directly; scalar `wf.map`, `wf.local` and `wf.yield`
retain their existing definitions. New mixed operations describe cursor, policy,
selection, acknowledgement, post-step observation and stop obligations.

The session and producer are actual graph regions. Typed feedback and stop
results defined later in the session feed the producer's isolated block
arguments. Inside the producer, selection's acknowledgement depends on its
policy receiver. The repeat and local regions are ordered SSA regions; the
post-step observation consumes advancement, the admitted input and its bound
model. Native dominance rejects future uses in local computation while graph
feedback reconstructs successfully.

Project verification checks exact wiring, containment, source identities,
policy/stop protocols, retained authority and matching subject/unit/input links.
Pliron supplies typed IR objects, parsing/printing, region verification, structural
equivalence and rewriting. Those services do not supply the modeled resource
ownership or a scheduler. This remains a fixture-specific validator and target,
not a general workflow language.

Preparation reuses the `ac_exec` declaration/local-call vocabulary. It retains
the accepted subtree and inspectable source/parent, access, policy, admission,
checkpoint and draining claims. The native matcher groups consecutive pure
scalar calls; the existing resource-aware extension remains the primary matched
analysis exercise. Every stage retains its validation and completion boundary.
Cold loading parses complete documents and compares the decoded plan with a
freshly prepared expected plan through Pliron structural equivalence. Installed
bindings and the whole saved manifest, including unused bindings, must agree.
Saved hashes and JSON profiles are inspection output and cannot authorize work.

[mixed_target.py](../mixed_target.py) extracts the existing xDSL owner-facing
mixed implementation into frozen target declarations. Each frontend independently
verifies its IR before calling it. The existing single `Engine`, atomic joins,
retained gradients, source protection and physical workers still own execution.
This is shared runtime evidence, not a second ownership implementation.

xDSL mixed now accepts saved prepared text/manifest and checks them before
factories. In xDSL 0.74, default structural comparison encounters operands before
mapping later definitions. Its mixed graph check seeds corresponding result
definitions, checks their types, then uses framework comparison for the full
structure and use relationships. This small project-owned adaptation supports
the existing forward feedback references; it is recorded as machinery work.
The xDSL CLI adds `--prepared` with `--load` and preparation-time manifest checking.

The shared target incidentally imports base xDSL definitions through reused
scalar fixtures/local dispatch. Pliron mixed never invokes xDSL mixed/model IR
parsing, acceptance, preparation or rewriting. Fresh native children block those
modules and base IR entry points; fresh xDSL children block Pliron. Python
bindings remain a cooperative selected-call ABI.

## Qualification

The [qualification](evidence_mixed/qualification.json) records sources, artifacts,
commands, preservation and review outcomes. There are 71 new document/native
cases and 78 owner cases, including 16 fresh normal/`python -O` children across
both frontends, owner lowerings and local grouping. An assertion-enabled parent
checks those children's actual reports. Valid alternative horizons, capacities
and window counts pass acceptance and fail target preparation honestly.

The complete gates pass **792 Torch-free tests with eight optional owner module
skips** and **861 owner tests**; the scopes overlap. Actual GIL state remains
disabled before and after, and CUDA remains uninitialized. Locked native build,
tests, Clippy, Rust formatting, Ruff and ty pass in the existing environments.

The required `review-mcp` attempt [failed on provider quota](evidence_mixed/review_mcp_failure.json)
and supplied no assessment. The user-authorized [GPT-6.1 Sol medium fallback](evidence_mixed/review_sol.md)
found no actionable issues and independently passed all 149 new cases plus 58
existing xDSL mixed cases. It did not repeat the full gates or retained captures.

Four raw saved artifacts and 22 actual execution records are retained. The
capture reuses the tested pressure helpers and original target driver; it does
not implement an independent scheduler. Pure [replay](evidence_mixed/replay.py)
without either framework, Torch or the owner runtime accepts 20 truthful records,
rejects two actual shared wrong-map executions and rejects 20 observation
corruptions. Archive-erasure records reconstruct numerical results from retained
owner input provenance and parameters. The first three initial tickets supply
that bounded oracle; later selection interleavings remain unconstrained.

The physical pressure case holds key six while key five waits to publish.
Stop and repeated cancellation retain the real read until physical completion;
later failure remains alongside three completed updates. The inherited CPython
shielded-future diagnostic can still be emitted under injected late failure.
The original worker is unchanged, and reports retain interruption, failure and
actual draining. Separate tests force initial production before learning, then
check epoch-zero features through later admissions. Ordinary runs may observe
other permitted epochs; they do not have to reproduce that schedule.

This supports one scalar learner/unit, the installed policy ABI, one-contribution
windows, horizon three, port capacity one and the explicit three-update checkpoint.
It establishes neither arbitrary stop points, indefinite archival memory,
mutable-resource authority from Rust ownership, GPU/distributed behavior, native
backend sharing nor a framework performance advantage. All six case meanings
now have reciprocal representation paths. Matched diagnostics/costs and the
substrate decision remain separate tasks.

## Reproduce

```bash
export CARGO_TARGET_DIR=/tmp/sd-scripts-pliron-b-target
cargo build --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml --bin rmst-pliron-mixed
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.mixed_demo \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-mixed" --fuse --dump-dir /tmp/pliron-mixed
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.mixed_demo \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-mixed" --fuse --load /tmp/pliron-mixed/artifact.json
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.evidence_mixed.capture \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-mixed"
/tmp/sd-scripts-xdsl-b-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.evidence_mixed.replay
```
