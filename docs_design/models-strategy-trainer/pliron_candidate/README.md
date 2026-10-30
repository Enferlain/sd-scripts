# Pliron 0.20.0: B and an independent local extension

Research experiment · 30 October 2026 · `sd-scripts-syv.31.19`, `.31.20`

This implements the **data-only feedback B case** with actual Pliron semantic
and prepared operations. It screens a concrete alternative to the provisional
xDSL working baseline. Language design, public authoring, final framework and
runtime implementation remain open. Production code, governing requirements,
the user's IR report and the separate numerical backend are unchanged.
The independent pair/frozen-resource extension below additionally tests shared
semantic interfaces and an analysis that authorizes or declines local grouping.

The baseline [comparison](../candidate_comparison.md) defines the evidence
boundaries. B selects `x = key + current_bias`, maps `2*x`, then `+1` inside a
nested local region, squares, validates the returned publication receipt and
applies `bias = score % 5`. Outstanding tickets remain owned until feedback
acknowledgement. The accepted bound defaults to two. External stop closes
selection, settles selected work and drains physical calls before release.

## Version and isolation

[native/Cargo.toml](native/Cargo.toml) pins **`pliron = "=0.20.0"`** from
crates.io; [Cargo.lock](native/Cargo.lock) fixes the full dependency resolution.
The published crate has checksum
`7395b93d3fc99ee1923027fedead143eb63e4c76b6f6f6d79ebfac6698956317`
and source revision `435536c8611af34c48c8f73d78e77894a05d103d`, recorded in its
`.cargo_vcs_info.json`. This is the release, not the later checkout that also
declares 0.20.0. The release's operation constructor differs from that checkout.
All relied APIs were checked against the downloaded release source.

Core Pliron supplies this slice without `pliron-llvm`, LLVM or the upstream
workspace's LLVM/JIT development dependencies. Builds use an external target
directory. No upstream manifest, source, backend pin or project Python lockfile
is modified. Python requirements reuse the existing pinned Torch-free B target.

## Boundaries actually implemented

| Boundary | Implementation | What remains project-owned |
| --- | --- | --- |
| Python authoring | [author.py](author.py) emits registered Pliron structural text with entities, typed SSA endpoints and nested composition. | This fixture is a tiny authoring adapter, not a proposed public syntax. |
| Semantic acceptance | [dialect.rs](native/src/dialect.rs) defines operations/types/interfaces. The session uses `RegionKind::Graph`; ordered composition/local regions use `RegionKind::SSACFG`. | [prepare.rs](native/src/prepare.rs) checks entity roles, exact wiring, unique source IDs, the unary chain, feedback/receipt and stop contracts. |
| Reconstruction | Pliron parses/prints; framework structural equivalence checks a reparse, and a fresh context verifies without retaining authoring objects. | Locations/names are ignored for structural equivalence; artifact compatibility is unqualified beyond this version. Text is not a public, version-independent canonical format. |
| Preparation | An `exec.plan` retains the accepted subtree plus selection/handoff/control recipes and producer/consumer call chains. | Source ancestry, stage completeness/order, exact binding/version, codec/outcomes, bounds, target and completion protocols. Horizon three is accepted but unsupported by this target. |
| Transformation | A real `MatchRewrite` moves eligible producer runs into `exec.fused` through the framework rewriter and listener. A resource-use analysis supplies the grouping decision. | Ordered unary stages may use pinned immutable snapshots. Unknown effects form conservative barriers. Every stage retains source, validation, failure and physical completion. |
| Analyses | Actual `Passes`/`AnalysisManager` manage dispatch count, accepted structural hash and local resource-use/stability summaries. | The resource analysis and its legality policy are project code; the framework supplies caching, invalidation and preservation. |
| Execution | [bridge.py](bridge.py) consumes only the native-verified projection, then invokes B's existing Python bindings, guarded local compiler and `ScalarSession`. | All live admission, policy, receipt, stop and physical-lifetime semantics. This is reused runtime evidence, not an independent ownership implementation. |
| Independent expectation | The existing import-independent [`check_b`](../xdsl_candidate/conformance.py) checks live facts and deliberate corruptions. | Framework verification cannot establish that an installed binding implements the intended arithmetic. |

The graph-region probe accepts a forward endpoint definition. The same sort of
forward use in an ordered local SSA region fails Pliron's dominance verifier.
Neither stream endpoint implies a new static SSA value per runtime ticket.
The feedback edge is explicit; a graph region does not schedule it for us.

The accepted subtree in a decoded plan must be structurally equivalent to the
supplied accepted program through Pliron's definition/use mapping. Domain
checks then require every prepared claim to match. A hash alone cannot authorize
execution. The Python adapter asks native code to verify raw documents against
installed contracts, checks agreement with the saved binding manifest, and then constructs
executable dispatch. Saved JSON projection fields are inspection output; cold
loading does not trust them as an executable program.

Every parse requires the entire document: one root operation, its outlined
attributes, trailing whitespace and end-of-input. Pliron 0.20.0 has no comment
syntax in this grammar; comment-like suffixes are rejected. A second root or
garbage cannot disappear during acceptance or survive unverified in a cold
artifact's retained text.

Fusion changes actual Python dispatch from two guarded coroutines to one
generated guarded coroutine, retaining both stage guards. The prepared calls
have no SSA results, matching the baseline transformation's scope: this does
**not** qualify SSA result remapping. The matcher runs only after semantic and
binding checks, and transformed text passes structural and correspondence checks
again before materialization. No DCE or generic memory-effect inference is used.

The analysis observations are `(dispatch groups, dispatch computations, source
computations)`: unfused `[(2,1,1), (2,1,1)]`; fused
`[(2,1,1), (1,2,1)]`. Repeated requests hit the cache. The changed pass invalidates
dispatch analysis and explicitly preserves the unchanged source analysis.
The framework supplies that invalidation mechanism; correct preservation
declarations remain the pass author's responsibility.

## Independent payload/resource extension and legality analysis

[pair_extension.rs](native/src/pair_extension.rs) registers `pairing.pair`,
`pairing.pack` and `pairing.first` in a separate native module. The Python
[extension module](pair_extension.py) provides authoring, codec and binding
installation. Pack reads named frozen coefficients `(1,2)`, returns `(v,2*v)`,
and declares the domain outcome `negative_input` in addition to mandatory
runtime outcomes. First consumes the pair and returns its first lane. B's final
scalar computation and policy remain unchanged.

The shared [contracts](native/src/contract.rs) and `LocalStageInterface` now
describe payload schemas/codecs, cardinality, resource requirements and outcomes.
Generic local verification follows actual SSA input/result types through nested
regions. The core verifier, preparation, bridge and rewrite contain no pair
operation/type-name dispatch. The separate registration module and target
registry supply the extension-specific contracts and implementations. These
are experimental interfaces, not a selected production extension ABI.

Preparation checks native descriptions against installed signatures, codec
versions, resource access/version/digest and outcomes. Cold loading recomputes
the same correspondence. Only after all raw IR and manifest checks pass can a
lowering factory receive a read-only map of its declared immutable snapshots.
An unrelated installed resource is not included. The reused runtime validates
the actual input/output schema at every stage, including within a fused group.
The scalar legacy registry adapter accepts only its default record codec and
no custom dialect registration; extensions use `TargetRegistry` explicitly.
The manifest also advertises the Python runtime's reserved outcomes. Native
preparation checks exact agreement with its vocabulary before lowering, so a
future cross-language change fails readiness rather than silently diverging.

`ResourceAnalysis` summarizes the composed producer's local stages through those
interfaces, including nested local regions. It records pinned reads, unknown
sources and per-source grouping decisions. The matcher consumes the cached
summary; decoded fused groups are checked against a freshly obtained summary.
The policy is shared rather than reimplemented by operation name in each phase.
Known reads remain visible in a partial summary even when additional effects
are unknown; that source still appears among grouping barriers.

| Case | Requested grouping | Actual local groups | Resource decision |
| --- | --- | --- | --- |
| Pair extension, unfused | Off | `double`, `offset`, `pack`, `first` | Known frozen read retained. |
| Pair extension, fused | On | `(double, offset, pack, first)` | Typed schemas compose; only pure/frozen uses; all stage guards retained. |
| Opaque offset, unfused | Off | Four individual stages | Valid executable case with the same numerical result and an observed side effect. |
| Opaque offset, fused | On | `double`, `offset`, `(pack, first)` | Unknown resource effects prevent grouping through offset; its invocation and effect remain. |

The opaque case is a valid unfused program. A forged, structurally valid group
crossing that stage is rejected by resource analysis even though its individual
source/binding signatures are correct. This demonstrates a transformation
decision, separately from rejection of malformed resource metadata. The target
conservatively declines effects it cannot establish; it does not assert that
every declined grouping would be semantically impossible under another policy.

The resource summary is computed once per preparation pipeline and explicitly
preserved because grouping leaves semantic resource uses unchanged. A fresh
summary comparison checks that preservation claim. Dispatch analysis is
invalidated and recomputed after grouping; source analysis remains preserved.
The frozen extension changes the dispatch analysis from `(4,1,1)` to `(1,2,1)`;
the opaque case changes it to `(3,2,1)`.

This does not introduce resource replacement or a stable-use scope. Immutable
snapshot contracts and the existing selected-call physical completion ABI
supply this target's guarantees. Rust ownership of IR objects does not provide
exclusive authority over modeled resources, Python models or device buffers.

Diagnostic controls cover invalid extension schemas, missing/stale resources,
codec/signature/outcome mismatches, altered decoded claims, unknown effects and
unsupported targets. Errors identify the source activity, relevant resource or
schema, expected/provided evidence and selected target where applicable. The
process boundary remains the development interface; no in-process binding is
required for these checks.

Native manifests/plans now use **schema 2**. Checked-in B artifacts were
regenerated and cold-loaded alongside the four extension variants. Schema-1
artifacts from the initial B commit remain historical; regenerate them with the
current driver. Artifact compatibility is still experimental.

## Reproduce

From the repository root, reusing the earlier isolated environment:

```bash
export CARGO_TARGET_DIR=/tmp/sd-scripts-pliron-b-target
cargo build --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml

/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.pliron_candidate \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-b" --dump-dir /tmp/pliron-b-unfused
/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.pliron_candidate \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-b" --fuse --dump-dir /tmp/pliron-b-fused
/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.pliron_candidate \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-b" --load /tmp/pliron-b-fused/artifact.json

/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.pliron_candidate \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-b" --extension --fuse --dump-dir /tmp/pliron-pair
/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.pliron_candidate \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-b" --extension --load /tmp/pliron-pair/artifact.json
/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.pliron_candidate \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-b" --extension --opaque --fuse --dump-dir /tmp/pliron-opaque
/tmp/sd-scripts-xdsl-b-env/bin/python -m docs_design.models-strategy-trainer.pliron_candidate \
  --binary "$CARGO_TARGET_DIR/debug/rmst-pliron-b" --extension --opaque --load /tmp/pliron-opaque/artifact.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/sd-scripts-xdsl-b-env/bin/python -m pytest --noconftest \
  docs_design/models-strategy-trainer/pliron_candidate docs_design/models-strategy-trainer/xdsl_candidate \
  -q -p pytest_asyncio.plugin --tb=short

cargo clippy --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml
cargo fmt --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml -- --check
/tmp/sd-scripts-xdsl-b-env/bin/ruff check docs_design/models-strategy-trainer/pliron_candidate
/tmp/sd-scripts-xdsl-b-env/bin/ruff format --check docs_design/models-strategy-trainer/pliron_candidate
/tmp/sd-scripts-xdsl-b-env/bin/ty check --python /tmp/sd-scripts-xdsl-b-env/bin/python \
  --extra-search-path docs_design/models-strategy-trainer docs_design/models-strategy-trainer/pliron_candidate
```

To recreate the small Python environment, use the existing free-threaded
interpreter with `uv venv /tmp/sd-scripts-pliron-b-env --python .venv-wsl/bin/python`,
then `uv pip sync --python /tmp/sd-scripts-pliron-b-env/bin/python
docs_design/models-strategy-trainer/pliron_candidate/requirements.txt`.
Substitute that environment's path in the commands above. The Python test
fixture builds the exact locked native package once per test session.

The native CLI also takes one JSON request on stdin: `accept`, `prepare` or
`load`. Its diagnostics are returned on stderr with nonzero exit status.
The Python driver retains accepted/prepared text, binding manifest, native
projection, live report and environment/analysis observations. It stops after
four acknowledgements by default; already selected work may increase the final
ticket count. Report clocks express causal relationships, not an identical
scheduler trace required across candidates.

## Evidence and comparison

Current qualification: **195 Pliron tests** (118 B/boundary plus 77 extension)
and **261 existing Torch-free tests pass**, with three optional Torch owner modules skipped.
One native Rust regression also passes. Ruff, formatting,
ty and Rust Clippy pass. Normal and optimized fresh-process execution disable
the author builders and xDSL parse/preparation functions; the independent oracle
runs in the normal parent. GIL state is checked after ordinary imports and live
execution without forcing it off. [evidence/qualification.json](evidence/qualification.json)
retains commands, environment, exact sources, artifacts and review status.
The extension adds positive and negative resource-aware grouping, cold schema/
resource/outcome corruption before factories, hidden-lane corruption detection,
declared/undeclared outcomes, typed output rejection, real physical resource
reads through stop/repeated cancellation and failure, and normal/optimized
fresh-process reconstruction with the oracle in the normal parent. The extension
review completed; the earlier milestone reviews below remain historical.
The [extension review](evidence_extension/review.md) found no exercised execution
or authorization bug and identified small robustness/reporting/documentation
follow-ups. Those were corrected: legacy registry guarding, partial-footprint
reporting, checked runtime outcome vocabulary and the opaque cold-load command.
The quality gates above ran again after these corrections; the qualification
records that post-review scope separately.

The initial `review-mcp` review completed with **no actionable findings**. It
read the new files directly as well as the reused target code and saved
artifacts; its assessment is retained in [evidence/review.md](evidence/review.md).
It verified the implementation and counted the cases; the executed test results
above are the local qualification, not an independent reviewer rerun.

External review of `5a737c27` subsequently found two concrete gaps: the native
parser accepted a valid prefix without checking EOF, and the root `*.txt` ignore
rule omitted the locally present requirements file from the commit. The shared
parser now composes whitespace and EOF checks; a local `.gitignore` exception
makes `requirements.txt` visible to normal Git addition. The new regressions
exercise semantic acceptance, preparation and both cold-load documents, prove
rejection before executable projection, and preserve ASCII/Unicode whitespace.
Twelve rejection cases failed before the fix; all fifteen new cases pass after
it. A fresh environment installed all twelve dependencies from the documented
file and executed the saved fused plan. The correction's required `review-mcp`
review completed with **no actionable findings**, retained in
[evidence/review_document_boundary.md](evidence/review_document_boundary.md).
The qualification file retains the exact manual source-visibility check for
repeating before a commit; it is not a repository hook or an automated test.

Coverage includes invalid semantic wiring/types/identity, acceptance versus
target readiness, binding/schema/outcome failures, cold plan corruption before
dispatch compilation, real fusion/analysis changes, acknowledged feedback under
blocked publication, stage failures/invalid output, independently detected wrong
arithmetic, zero diagnostic history, and stop/repeated cancellation during a real
thread's physical work, including failure after interruption. One initial test
injected `value + 1` into the offset binding, accidentally preserving its correct
behavior; both expected-failure checks correctly failed. The injection was
corrected to a wrong computation before the passing qualification.

| Dimension | What this slice establishes | Comparison limit |
| --- | --- | --- |
| Semantic fit | Explicit graph versus ordered SSA regions naturally represent B's workflow and local chains. Persistent entity references and typed feedback endpoints remain inspectable. | This does not establish general joins, cardinality changes, lifecycle/authority composition or mutable resource semantics. xDSL already represents B successfully. |
| Machinery ergonomics | Derive macros, interfaces, generic syntax, structural equivalence and mutation listeners replace generic bookkeeping. | Domain validation and target protocol checks remain substantial Rust code. No interactive Python object binding or diagnostic UX comparison was built. |
| Infrastructure reuse | Parsing/printing, dominance/use tracking, structural comparison, rewriter worklist and cached-analysis invalidation are used directly; an interface-based resource analysis now drives grouping. | The domain analysis remains project-owned. No matched maintenance/complexity or performance result. |
| Python boundary | Detached documents and versioned binding descriptors support cold verification and execution. | Process/JSON handoff is deliberately narrow and adds conversion and invocation work. The adapter still imports xDSL because the reused target/types/fixtures live there; no xDSL IR machinery implements this experiment. |
| Native integration | The machinery can reside in Rust and use real Pliron values/contexts/interfaces. | The local backend pins `=0.19.0` and uses CubeCL numerical IR helpers. No shared module/context/dialect/analysis or zero-copy numerical handoff is exercised here; a framework name alone proves no reuse. |

**Assessment:** Pliron now supports an independently defined local schema,
operations and frozen resource through common contracts. Its cached analysis
participates in a real transformation decision. Widening the shared interfaces
and target registry required core changes once; the extension-specific names and
rules remain in their own module. This improves the evidence for extensibility
and analysis composition, while whole-session acceptance remains the fixed B
profile and generic local-region structure remains project-defined.

The broader grouping policy is also project code. xDSL's earlier extension
target deliberately retained scalar/schema/resource barriers; these different
dispatch counts do not establish a framework performance advantage. Both cases
use the same independent pair arithmetic and hidden-lane expectations. Keep
xDSL as the working baseline. A bounded different-lifetime composition case is
the next useful representation question if continuing this candidate; backend
sharing still needs a named beneficiary. Matched costs, broad parity, in-process Python bindings,
backend version convergence, GPU/distributed completion and durable restoration
remain outside this slice.
