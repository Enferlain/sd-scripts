# Matched resource-aware grouping

Research milestone · 1 November 2026 · `sd-scripts-syv.31.25.4`

Both xDSL **0.74.0** and Pliron **=0.20.0** now implement the selected
[transformation policy](matched_comparison.md#common-transformation-policy).
The independent pair extension uses frozen coefficients `(1,2)` and a different
intermediate payload type. Every stage retains its own validation, source,
outcome and physical completion. This closes the earlier policy mismatch; it
does not establish framework performance or complete the comparison.

| Accepted producer | Unfused | Grouping selected |
| --- | --- | --- |
| Double, offset, frozen pack, first lane | Four individual calls. | One group with all four leaves, including both schema transitions. |
| Double, opaque offset, frozen pack, first lane | Four individual calls; the offset's actual effect is observed. | Double and offset remain individual calls; pack/first form a group: three dispatches. |
| Valid opaque source with a forged group crossing offset | The valid unfused program executes first. | Both cold verifiers reject the forged group before invoking the resource factory, even with forged analysis observations claiming eligibility. |

Ordered one-to-one schemas must connect, and enclosing local regions must be
transparent to resource lifetime. Known pure or pinned frozen reads are eligible.
Unknown effects remain barriers. A schema transition or additional declared
domain outcome does not itself prevent grouping. Snapshot identity includes
symbol, access, version and content digest; grouping neither creates mutable
authority nor extends a physical lease.

## Actual machinery

xDSL's [fusion.py](xdsl_candidate/fusion.py) replaces the old exact scalar-only
predicate with a resource-use/stability summary derived through `LocalTransform`
and `TransparentLocal`. The framework pattern walker/rewriter clones the plan
and groups compatible runs. The same summary policy checks decoded groups;
installed codecs, signatures, outcomes and snapshot correspondence are still
verified before factories. Neither analysis nor rewriting dispatches on the
pair operation's identity.

`LocalAnalyses` is a **project-owned, per-preparation cache**. It compares full
immutable dependency snapshots for source contracts, typed endpoints, effect,
outcomes, source identity and enclosing lifetime. Grouping preserves the flattened
semantic dependencies. A changed effect/resource/schema/outcome/ancestry causes
recomputation on the next request. Dispatch-group observations have a separate
dependency key. xDSL supplies the IR and rewrite infrastructure here; this cache
is custom code, not a framework analysis manager. It is confined to preparation,
with no shared mutable global state or execution-time authority.

Pliron's existing `ResourceAnalysis` already supplies the same intended policy.
The framework's `AnalysisManager` and `Passes` manage caching and invalidation.
`FusionPass` declares source/resource analyses preserved and dispatch analysis
invalidated. The domain summary, eligibility rules and correctness of preservation
claims remain project responsibilities. No new native grouping algorithm was
needed for this milestone.

Two native regression passes actually change the semantic source and matching
prepared/header/binding claims. Making offset opaque changes legality; advancing
the frozen resource version changes its identity. Neither pass declares the old
semantic analyses preserved. Their rewritten artifacts cold-verify against the
updated accepted documents and manifest; they are not incomplete-correspondence
examples being mistaken for legality tests.

| Evidence | xDSL project cache | Pliron framework-managed cache |
| --- | --- | --- |
| Ordinary four-to-one grouping | Resource computation stays at one; dispatch computation changes from one to two. | Resource/source computations stay at one; dispatch computation changes from one to two. |
| Opaque barrier | Fresh summary identifies offset as unknown; three groups. | Fresh summary identifies offset as unknown; three groups. |
| Semantic contract edit, then grouping | Effect, version and content-digest tests each recompute the resource summary; unchanged dependencies are then reused. | Effect/version probes produce `(groups, dispatch computations, source computations)` of `(4,1,1)`, `(4,2,2)`, then `(3,3,2)` or `(1,3,2)`; resource computations remain two through grouping. |
| Invalidation mechanism | Rebuild and compare dependency values on access; dispatch dependencies can remain valid across unrelated edits. | Changed pass results invalidate analyses unless explicitly preserved. |

These mechanisms establish different kinds of infrastructure reuse. Matching
counter values do not make custom caching equivalent to framework-managed
analysis infrastructure. Cold loading derives legality from verified documents
and installed contracts; saved analysis output is never an execution input.

## Validation and boundaries

The [qualification](evidence_grouping/qualification.json) records exact sources,
artifact hashes, commands and review outcome. The 21 new Python cases cover
both candidates, normal/opaque variants, horizons one/two, successful unfused
execution before grouping, forged cold claims, dependency changes and fresh
normal/`python -O` reconstruction. Two native tests add actual semantic mutation
and preservation/invalidation witnesses. Existing extension and fusion cases
continue checking invalid payloads, domain/undeclared outcomes, hidden-lane
corruption and held physical work through interruption/failure.

Final scoped local gates pass **660 Torch-free tests with six optional owner
module skips**, and **706 owner tests**, with the GIL disabled before and after.
The overlapping gate totals are not additive. Native all-target tests, locked
builds, Clippy, Rust formatting, Ruff and ty pass. No GIL override, package
installation or GPU initialization is used.

The required [review-mcp attempt](evidence_grouping/review.md) failed with a
provider HTTP 429 quota error; the saved status reports `result_ok=false`.
It supplied no assessment or approval of this grouping slice. The user's report
of no actionable external review input concerned the preceding committed work
through `4558da99`; it is recorded separately from this failed attempt.

The user subsequently authorized a [GPT-6.1 Sol medium fallback review](evidence_grouping/review_sol.md).
It found no actionable findings, independently reran 102 focused Python tests
and all three native tests, reproduced the retained replay without changing it,
and verified current source/artifact hashes. It did not repeat the full 660/706
gates or capture new runtime evidence. No implementation changes were required;
the failed provider attempt remains recorded as failed.

Eight live/cold positive variants and ten negative-control executions retain
actual reports. The [independent replay](evidence_grouping/replay.py) imports no
framework or runtime: it accepts 16 truthful success/failure reports, rejects
two real hidden-lane mistakes, and rejects 18 observation corruptions. The scalar
projection still accepts those hidden-lane mistakes, showing why the independent
payload expectation matters. Held fused reads preserve a live invocation through
repeated cancellation until the worker returns with its physical failure; first
lane, scoring and publication do not enter afterward. Diagnostic histories are
empty throughout these captures.

Fresh Pliron children disable xDSL IR entry points and native author/preparation
wrappers; the existing bridge still incidentally imports xDSL-defined codec/data
classes through its reused Python target. That import is separate from using
xDSL to parse, verify or rewrite Pliron IR. Fresh xDSL children block the Pliron
package. Both normal and optimized children execute saved fused text and return
raw observations to a normal parent checker.

The frozen descriptive fixtures, owner runtime, stage guard/compiler, native
dependency pins, original user report and historical evidence remain unchanged.
Historical qualification source hashes describe their own milestones. This
record contains current grouping sources; it does not rewrite older records to
make them look current.

This remains a fixed unary local transformation with cooperative Python bindings
and immutable snapshots. It does not establish mutable-resource authority, SSA
result replacement, broader workflow coverage, native numerical sharing or
performance superiority. Pliron A/mixed ports and matched machinery/diagnostic/
phase-cost evidence remain open. xDSL stays the provisional working baseline;
language and final implementation choices remain open.

## Reproduce

From the repository root, reuse the existing free-threaded environments:

```bash
export CARGO_TARGET_DIR=/tmp/sd-scripts-pliron-b-target
cargo build --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml
cargo test --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml --all-targets -- --nocapture

/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.evidence_grouping.capture \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-b"
/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.evidence_grouping.replay

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/sd-scripts-xdsl-b-env/bin/python -m pytest --noconftest \
  docs_design/models-strategy-trainer/xdsl_candidate/test_grouping_comparison.py \
  docs_design/models-strategy-trainer/xdsl_candidate/test_fusion.py \
  docs_design/models-strategy-trainer/xdsl_candidate/test_extension.py \
  docs_design/models-strategy-trainer/pliron_candidate/test_extension.py \
  -q -p pytest_asyncio.plugin
```
