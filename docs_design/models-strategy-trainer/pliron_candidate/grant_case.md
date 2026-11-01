# Temporary optimization authority on Pliron 0.20.0

Matched case C · 1 November 2026 · `sd-scripts-syv.31.25.3`

This adds independent Pliron representation, acceptance, preparation and cold
correspondence for the existing two-pass authority case. It follows the frozen
[comparison meaning](../matched_comparison.md), using the same numerical
callable, grant/optimization owners, physical workers and independent scalar
oracle as xDSL C. xDSL remains the provisional baseline. Language design and
the final framework/runtime choices remain open.

## Authority contract

A trainable scalar starts at `w=0.8`. The selected implementation computes
`(2*w-1)^2`, performs backward, temporarily edits `w += 0.1*gradient`, resets
intermediate gradients, computes backward again, then restores the original
weight. The selected scope delegates those operations; final clipping,
advancement and reset remain with the ordinary owner.

Independent arithmetic gives first gradient `2.4`, temporary weight `1.04`,
second gradient `4.32` and final retained-SGD weight `0.584` at learning rate
`0.05`. A second initial weight `0.6` and optional norm clipping at `1.0` exercise
the same declared computation. Both existing owner lowerings, `calls` and
`inline`, execute each parameter combination.

| Boundary | Required meaning |
| --- | --- |
| Profile selection | The target must actually offer `restored-two-pass-v1`; an accepted requirement cannot create that capability. |
| Model/unit association | The delegated model and optimization unit must identify the same parameter owner. |
| Delegated region | Exactly five ordered phases consume and produce an actual typed access chain, ending in a restoration yield. |
| Handback claim | Static IR describes the required claim. It does not manufacture an owner-verified runtime token. |
| Runtime verification | Actual phase history, restoration, current state and final gradient must satisfy the existing grant owner before advancement becomes ready. |
| Retained advancement | One addressed handback feeds its matching unit; only the retained owner performs clipping, final SGD and reset. |
| Physical completion | Exclusive protection remains held through physical return, validation and handback. Interruption joins entered work before release. |
| Failure | Restored state can remain unverified. Entered optimizer/reset work may have effects and remain uncertain; failed state is unavailable and cannot be retried as ready. |

The fixture has no data streams or per-record admission. Its static handback
has exactly one retained consumer, and session completion covers all retained
outcomes. These checks do not establish arbitrary authority composition.

## Independent machinery and shared execution

The new `rmst-pliron-grant` executable lives in the existing locked native
package, still pinned to **`pliron = "=0.20.0"`**. No Cargo dependency, LLVM
requirement or backend pin changes. The downloaded release's actual APIs are
used, rather than those of a later checkout with the same version string.

| Concern | Framework machinery | Project code |
| --- | --- | --- |
| Representation | Registered `ga` operations/types, isolated single-block SSA regions, actual operands/results and block arguments. | `AuthorityContract` defines each operation's attribute and typed endpoint shape; the verifier checks the fixed authority protocol and associations. |
| Ordered use | Pliron verifies SSA dominance and terminators. | Phase order, the uninterrupted access chain, one handback consumer and final completion coverage remain semantic checks. |
| Reconstruction | Native parse/print, fresh contexts and structural equivalence with definition/use mapping. | Complete-document consumption, accepted contracts and version-specific compatibility policy. |
| Preparation | Native construction of a `ga_exec.plan` with semantic and declaration regions. | Explicit source, binding, access, offered-profile, handback and retained-owner recipes. |
| Cold correspondence | Framework structural equivalence checks a decoded plan against the freshly derived expected plan. | Both raw documents, installed contracts and the entire saved manifest are checked before model factories. Saved projections and hashes are inspection output. |
| Execution | Both frontends call the same existing owner target after their independent IR checks. | Numerical bindings, live grant readiness, actual model protection, optimizer outcomes and physical draining. |

The region's SSA chain is `ga.access -> phase ... -> ga.yield`; the enclosing
`ga.execute` produces `ga.handback`, and `ga.advance` consumes it with the unit
to produce `ga.done`. The prepared handback declaration explicitly says
`not_an_owner_verified_token`; advancement requires
`owner_verified_handback_not_restoration`. These are verified target contracts,
not general instruction strings interpreted as arbitrary programs.

[grant_target.py](../grant_target.py) extracts the existing C owner-facing code
from xDSL's adapter. It consumes immutable target declarations and selected
callables, then constructs the existing owner program. Neither frontend shares
an executable graph before representing/verifying its own IR. Pliron imports no
xDSL machinery on its execution/cold path. xDSL never invokes the Pliron
verifier. This is independent IR evidence using one runtime implementation.

xDSL C additionally exposes `saved_text`/`saved_manifest` through its existing
decoded-plan validation path. Complete expected-plan structural equivalence
rejects extra roots and unexpected structure; manifest validation precedes owner
construction. Its CLI accepts `--prepared` with the corresponding accepted text.

Manifests deliberately require whole saved/installed agreement, including unused
entries. Native C artifacts use their own schema `1`, independently of B's
schema `2` and the lifetime artifact. Prepared operation declarations have no
SSA results. This slice introduces no rewrite or analysis; resource-aware
grouping alignment is the next comparison task.

## Evidence and limits

The [qualification](evidence_grant/qualification.json) records exact sources,
artifacts, commands and review outcome. Fifty new document/native tests and
55 new owner tests cover arithmetic, typed phase/source/authority correspondence,
unoffered capabilities, binding substitution, malformed documents, unsupported
targets and forged prepared claims. Seven additional xDSL owner tests check
persisted prepared loading and early rejection before factories.

Actual held workers demonstrate protection and missing handback while numerical
work has restored the weight but not physically returned. Repeated cancellation,
physical failure and their combination drain before release; a fabricated
contribution cannot authorize retained advancement. Other controls preserve
uncertain optimizer/reset outcomes after effects, unavailable state and failed
readiness after diagnostic/outcome history eviction.

Fresh normal and `python -O` children cold-load each frontend's saved prepared
document for both owner lowerings with authoring disabled. The normal parent
checks independent arithmetic and outcomes. The Pliron children block xDSL IR
imports; the xDSL children block Pliron. No GIL override is used.

The final scoped gates pass **639 Torch-free tests with six optional owner-module
skips** and **685 owner tests**, with the GIL disabled before/after both. The
scopes overlap in reused checks and their totals are not additive. Native locked
build/test/Clippy, Rust formatting, Ruff and ty pass. These elapsed gate times
are validation receipts, not framework performance measurements.

Eight normal artifact variants and 23 actual execution reports are retained.
The independent [replay](evidence_grant/replay.py) imports neither framework,
Torch nor the runtime: it accepts 21 truthful success/failure reports, rejects
two real executions with an intentionally shared no-op optimizer, and rejects
12 observation corruptions. Lowering agreement alone accepts that shared
numerical error, demonstrating why the independent oracle is required.

The required [review-mcp attempt](evidence_grant/review.md) failed with a provider
HTTP 429 quota error. Its saved status reports `result_ok=false`; it supplied no
assessment or approval. The executed local gates and replay remain separate
evidence. This C implementation is handed off with that review gap explicit.

The inherited CPython 3.14.5 shielded-future diagnostic can appear when an
injected late physical failure races cancellation. The owner report retains
both failure and interruption and the worker is joined; the original worker
service is unchanged. This observation is not a new drain guarantee.

This is a cooperative Python ABI: selected implementations must join private
physical work before returning or raising. Rust ownership of IR objects does
not establish exclusive ownership of modeled parameters. Five phase contracts
behind one numerical binding do not constitute a numerical tensor language,
proof of binding arithmetic, security isolation, general authority composition,
GPU qualification or a performance advantage. A/mixed ports, aligned grouping
and the matched diagnostic/cost/decision records remain open.

## Reproduce

From the repository root, using the existing free-threaded environments:

```bash
export CARGO_TARGET_DIR=/tmp/sd-scripts-pliron-b-target
cargo build --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml --bin rmst-pliron-grant

/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.grant_demo \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-grant" --clip-norm 1 --dump-dir /tmp/pliron-c
/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.grant_demo \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-grant" --load /tmp/pliron-c/artifact.json

/tmp/sd-scripts-pliron-composition-owner-env/bin/python \
  -m docs_design.models-strategy-trainer.pliron_candidate.evidence_grant.capture \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-grant"
/tmp/sd-scripts-xdsl-b-env/bin/python \
  docs_design/models-strategy-trainer/pliron_candidate/evidence_grant/replay.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/sd-scripts-xdsl-b-env/bin/python -m pytest \
  --noconftest docs_design/models-strategy-trainer/pliron_candidate/test_grant.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/sd-scripts-pliron-composition-owner-env/bin/python -m pytest \
  --noconftest docs_design/models-strategy-trainer/pliron_candidate/test_grant_runtime.py \
  docs_design/models-strategy-trainer/xdsl_candidate/test_grant_runtime.py \
  -q -p pytest_asyncio.plugin
```
