# Pliron 0.20.0: bounded B comparison

Research experiment · 30 October 2026 · `sd-scripts-syv.31.19`

This implements the **data-only feedback B case** with actual Pliron semantic
and prepared operations. It screens a concrete alternative to the provisional
xDSL working baseline. Language design, public authoring, final framework and
runtime implementation remain open. Production code, governing requirements,
the user's IR report and the separate numerical backend are unchanged.

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
| Transformation | A real `MatchRewrite` moves producer calls into `exec.fused` through the framework rewriter and listener. | Only ordered, pure, resource-free maps with the selected exact baseline contract are eligible. Each stage's source, validation, failure and physical completion remain visible. |
| Analyses | Actual `Passes`/`AnalysisManager` cache and invalidate two project-defined analyses. | The analyses calculate producer dispatch count and accepted structural hash; this is not reuse of a workflow legality algorithm. |
| Execution | [bridge.py](bridge.py) consumes only the native-verified projection, then invokes B's existing Python bindings, guarded local compiler and `ScalarSession`. | All live admission, policy, receipt, stop and physical-lifetime semantics. This is reused runtime evidence, not an independent ownership implementation. |
| Independent expectation | The existing import-independent [`check_b`](../xdsl_candidate/conformance.py) checks live facts and deliberate corruptions. | Framework verification cannot establish that an installed binding implements the intended arithmetic. |

The graph-region probe accepts a forward endpoint definition. The same sort of
forward use in an ordered local SSA region fails Pliron's dominance verifier.
Neither stream endpoint implies a new static SSA value per runtime ticket.
The feedback edge is explicit; a graph region does not schedule it for us.

The accepted subtree in a decoded plan must be structurally equivalent to the
supplied accepted program through Pliron's definition/use mapping. Domain
checks then require every prepared claim to match. A hash alone cannot authorize
execution. The Python adapter compares the installed and saved binding manifests,
asks native code to verify the raw documents again, and only then constructs
executable dispatch. Saved JSON projection fields are inspection output; cold
loading does not trust them as an executable program.

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

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/sd-scripts-xdsl-b-env/bin/python -m pytest --noconftest \
  docs_design/models-strategy-trainer/pliron_candidate docs_design/models-strategy-trainer/xdsl_candidate \
  -q -p pytest_asyncio.plugin --tb=short

cargo clippy --locked --manifest-path docs_design/models-strategy-trainer/pliron_candidate/native/Cargo.toml --all-targets -- -D warnings
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

Current qualification: **103 experiment tests** and **261 existing Torch-free
tests pass**, with three optional Torch owner modules skipped. Ruff, formatting,
ty and Rust Clippy pass. Normal and optimized fresh-process execution disable
the author builders and xDSL parse/preparation functions; the independent oracle
runs in the normal parent. GIL state is checked after ordinary imports and live
execution without forcing it off. [evidence/qualification.json](evidence/qualification.json)
retains commands, environment, exact sources, artifacts and review status.

The required `review-mcp` review completed with **no actionable findings**. It
read the new files directly as well as the reused target code and saved
artifacts; its assessment is retained in [evidence/review.md](evidence/review.md).
It verified the implementation and counted the cases; the executed test results
above are the local qualification, not an independent reviewer rerun.

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
| Infrastructure reuse | Parsing/printing, dominance/use tracking, structural comparison, rewriter worklist and cached-analysis invalidation are used directly. | Generic infrastructure is useful; no preexisting domain legality analysis is demonstrated. No matched maintenance/complexity or performance result. |
| Python boundary | Detached documents and versioned binding descriptors support cold verification and execution. | Process/JSON handoff is deliberately narrow and adds conversion and invocation work. The adapter still imports xDSL because the reused target/types/fixtures live there; no xDSL IR machinery implements this experiment. |
| Native integration | The machinery can reside in Rust and use real Pliron values/contexts/interfaces. | The local backend pins `=0.19.0` and uses CubeCL numerical IR helpers. No shared module/context/dialect/analysis or zero-copy numerical handoff is exercised here; a framework name alone proves no reuse. |

**Recommendation:** Pliron earns a bounded follow-up on the independent
payload/frozen-resource extension, chiefly because the region interface and
analysis/pass machinery work on an actual equivalent case. Keep xDSL as the
working baseline. The result supports continued comparison, not replacing that
baseline or selecting a final language/runtime. An owner-integrated case should
follow only if the extension or a precisely identified native boundary supplies
additional value. Matched costs, broad parity, in-process Python bindings,
backend version convergence, GPU/distributed completion and durable restoration
remain outside this slice.
