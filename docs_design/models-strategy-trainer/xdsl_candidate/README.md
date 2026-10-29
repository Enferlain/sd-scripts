# xDSL data-only workflow experiment

This isolated research package implements **B, the first handoff** of the plan
recorded in `sd-scripts-syv.31.8`, as `sd-scripts-syv.31.9`. It tests xDSL as the
actual semantic and prepared IR infrastructure for a system that can represent
data and model workflows. Production code, governing OpenSpec requirements and
framework selection remain unchanged. A, C and the fusion comparison are later
handoffs; this result does not establish training authority or transformation
conformance.

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
checks. Reuse of participant/optimization ownership through domain adapters
remains untested until A/C. There is no Torch import, dummy optimizer, training
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
  docs_design/models-strategy-trainer/xdsl_candidate
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

This makes xDSL promising for representation and diagnostics in B. It leaves
the difficult A/C owner integrations and the planned semantics-preserving fusion
experiment open. Python 3.14t execution and observed GIL state do not establish
arbitrary extension concurrency safety, GPU/distributed interoperability,
production performance or a final framework choice.
