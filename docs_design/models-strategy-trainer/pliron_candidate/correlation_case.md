# Correlation and accumulation on Pliron 0.20.0

Matched case A · 1 November 2026 · `sd-scripts-syv.31.25.5`

This adds independent Pliron representation, verification, preparation and cold
correspondence for A in the [stateful workflow comparison](../matched_comparison.md).
It reuses the existing numerical implementations, admission/state/optimization
owners and scalar oracle. xDSL remains the provisional baseline; final language,
framework and runtime implementations remain open.

## Case meaning

Samples `(1,2,4,5)` remain after authoritative exclusion of `3,6`. For sample
`k`, the left branch computes `2*k+1`, while a protected frozen encoder produces
`1.5*k/4` on the right. Correlation checks sample, window, slot, branch and current
source provenance for both members before claiming either. No intermediate
observation list authorizes admission.

The learner starts at `0.8`; loss is `(w*(left+right)-k/2)^2/2`. Each window
**sums two contributions**, then ordinary SGD advances once at `0.05` and resets.
This differs from the lifetime case's mean. Independent arithmetic gives weights
`0.8 -> -0.60625 -> 10.419848...`.

Window zero requests evaluation once. Its retained input association remains
the first window's four member uses, while the learner is read at protected
current state when evaluation actually starts. Epoch one or two is permitted;
the captured normal runs observed epoch two. Trigger membership does not select
an old learner snapshot or certify that physical evaluation has finished.

| Boundary | Required meaning |
| --- | --- |
| Source fanout | Exclusion applies before both paths; logical sample/window/slot mapping remains explicit. |
| Admission | Check both members and actual frozen-source stamps under the existing atomic join; rejection consumes neither. |
| Accumulation | Earlier gradients and input associations remain owned if a later member cannot be admitted. |
| Advancement | One retained-owner step/reset follows both contributions; completed-history eviction cannot authorize or erase pending work. |
| Requested evaluation | Preserve the successful first-window cause separately from protected current-at-start state. |
| Physical completion | Encoder/evaluator work drains through interruption and later failure before protection is released; evaluation mode is restored. |
| Run completion | Source, learning and requested work settle through the existing engine. |

## Representation and implementation boundaries

`rmst-pliron-correlation` uses the same locked **`pliron = "=0.20.0"`** package;
Cargo dependencies, LLVM requirements and the numerical backend pin are unchanged.
The new `ac` types distinguish stream, model, unit, correlated input, gradient,
window events and completion. The session is a graph region; local preprocessing
and window bodies are ordered SSA regions. Actual operands/results and block
arguments carry the relationships, rather than a project graph hidden in JSON.

The existing native `wf.map`, `wf.local`, `wf.yield` and record type are reused
for scalar preprocessing. A's `CorrelationContract` describes operation shapes;
domain checks enforce wiring, containment, identities, normalization, authority,
request consistency and completion coverage. Pliron supplies region/SSA
verification and structural reconstruction. Rust ownership of those objects does
not establish ownership of modeled parameters or input claims.

The window's region arguments bind its correlated input, learner and unit.
Its gradient feeds retained advancement and a window-end value. Separate outer
results carry window events and training completion; evaluation consumes events,
and run completion covers both training and evaluation. SSA values describe
workflow dependencies, not one new static value per runtime sample or a schedule
in which every earlier printed activity must finish first.

Preparation constructs an inspectable `ac_exec.plan` with the semantic subtree,
binding/access/authority/request/settlement declarations and logical-to-physical
addresses. A native matcher/rewriter groups consecutive pure scalar calls when
requested; every call retains its source, binding version and completion contract.
The matched frozen-resource analysis exercise remains separately qualified in
[the grouping record](../grouping_comparison.md); A adds no new general analysis.

Cold loading parses complete raw documents, verifies accepted meaning, checks
installed bindings/target limits, and compares the decoded plan structurally
with a freshly generated expected plan. Saved projections and document hashes
are inspection output. The whole saved binding manifest must agree with installed
contracts, including unused bindings. A has its own experimental artifact schema
`1`; broad artifact compatibility is not promised.

[model_target.py](../model_target.py) extracts A's existing owner-facing code
from xDSL into frozen target declarations. Each frontend independently verifies
its IR before constructing that owner program. Admission, retained gradients,
requested work and physical workers still have one existing authority. Shared
numerical bindings and runtime checks do not constitute an independent second
ownership implementation.

xDSL A additionally accepts saved prepared text/manifest and checks them before
model factories; its CLI exposes `--prepared` with the corresponding `--load`.
Complete expected-plan equivalence supplements its existing source/address/
binding/protocol checks. The shared target imports the existing scalar fixture
functions and pure local runtime; this incidentally imports xDSL definitions.
Pliron A never invokes xDSL model parsing, acceptance, preparation or rewriting.
Fresh native children block the model IR modules and base IR entry points;
fresh xDSL children block Pliron. Numerical bindings must cooperatively join any
private physical work. Encoder/evaluator completion is supplied by the existing
owner/worker services.

## Qualification

The [qualification](evidence_correlation/qualification.json) records exact source
and artifact hashes, commands and review outcomes. New coverage comprises 61
document/native tests and 77 owner tests, including 16 fresh normal/`python -O`
children across both frontends, local grouping and the two owner lowerings.
It tests neither-member claim on all four fault categories at every sample,
partial retained work after history eviction, binding substitution, malformed
documents, forged addresses/protocols and early rejection before factories.

The final declared gates pass **721 Torch-free tests with seven optional owner
module skips** and **783 owner tests**. Actual GIL state remains disabled before
and after. These scopes overlap and their totals are not additive. Locked native
build/test/Clippy, Rust formatting, Ruff and ty pass without installing packages,
forcing the GIL off or initializing CUDA.

The required `review-mcp` attempt [failed with provider quota](evidence_correlation/review_mcp_failure.json)
and supplied no assessment. The user-authorized [GPT-6.1 Sol medium fallback](evidence_correlation/review_sol.md)
returned no actionable findings and independently passed 122 A tests. It did
not rerun the 16 fresh-process cases, broader gates or retained captures.

Eight normal artifacts and 34 actual reports are retained. Independent
[replay](evidence_correlation/replay.py), without either framework, Torch or the
runtime, accepts 32 truthful success/failure reports, rejects two real shared
wrong-loss computations, and rejects 17 observation corruptions. The wrong loss
can agree across both owner lowerings while failing the independent expectation.
Held physical reads preserve live protection through repeated cancellation and
later failure; successful/failed evaluation checks retain or restore model mode
at the appropriate boundary. The original worker service is unchanged and may
emit the previously observed CPython shielded-future diagnostic during injected
late failures; the actual reports retain failure/interruption and draining facts.

This target supports one frozen encoder, one scalar learner/unit, at most four
selected samples in two-member windows, scalar local payloads and one requested
evaluation. Wider semantic counts/horizons or parameter names can be accepted
then rejected as unsupported by the target. This does not establish arbitrary
composition, a numerical tensor language, GPU/distributed behavior, native backend
sharing or a performance advantage. The mixed port and matched diagnostic/cost/
decision records remain open.

## Reproduce

From the repository root, reuse the existing free-threaded environments:

```bash
export CARGO_TARGET_DIR=/tmp/sd-scripts-pliron-b-target
cargo build --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml --bin rmst-pliron-correlation
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.correlation_demo \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-correlation" --fuse --dump-dir /tmp/pliron-a
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.correlation_demo \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-correlation" --fuse --load /tmp/pliron-a/artifact.json
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.evidence_correlation.capture \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-correlation"
/tmp/sd-scripts-xdsl-b-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.evidence_correlation.replay
```
