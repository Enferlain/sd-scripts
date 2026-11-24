# Review: `rework-model-strategy-trainer` refinement (design.md, 4 delta specs, tasks.md)

## Summary

The refinement is internally coherent and faithful to the settled direction. I found **no contradictions** with `strategy_system_direction.md`, the Q1–Q5 decisions, or the EX-001–EX-031 register: fulfillment-under-a-pre-existing-contract matches the "contract establishment" correction; the revisioned run-specific obligation set matches EX-007/EX-014/EX-016/EX-029; one-contract-per-authority matches EX-020; preparation publication/failure semantics match EX-021/EX-022/EX-025–EX-028; and semantic-acceptance vs executable-readiness is cleanly split at both design (D5 prose) and spec level. The change also does not reopen the accepted-arrangement topology or reintroduce an active-strategy callback. The issues below are gaps and precision problems, not architectural errors.

## What it gets right

- **No duplicated validation authority**: `training-runtime-preparation` explicitly forbids preparation from "defin[ing] new compatibility meaning or certif[ying] its own result," and requires pre-backend rejection when authoritative facts already prove incompatibility — a direct, correct implementation of the earliest-evidence rule.
- **Obligation state has one home**: `run-participant-state` lists `obligation` in the authority's canonical state, requires obligation revisions in every proposal, and forbids competing canonical copies (including by metadata/observability). `D7`'s "same obligations for initial realization and later transitions" is well scenario-covered.
- **Dual-ownership prevention is symmetric**: execution spec ("Trainer MUST stop owning exactly the transferred action") and optimization spec ("Trainer MUST not also advance") agree.

## Findings

### 1. Medium-high — "available environment facts" is ungrounded and unmanaged
`training-contract/spec.md:71` is the **only** occurrence of "environment" in the entire change. Design D2 (lines 236–244) enumerates the knowledge split as authored-definition facts vs inherently-deferred facts and never mentions environment; the exchange record's establishment section likewise only permits "known library conformance information." Introducing environment-derived fulfillment-time rejection creates three unaddressed obligations:
- **Provenance/determinism**: the accepted obligation set becomes environment-dependent, yet nothing requires recording *which* environment facts were used — inconsistent with a change that demands revisioned, projectable authority state.
- **Distributed runs**: whose environment is authoritative when ranks disagree?
- **Restore**: if environment changes between snapshot and restore (different device, different backend), are environment-derived obligations re-evaluated? The restoration requirements preserve identity/revision state but are silent here.

Suggest either grounding environment facts in D2 with these rules, or narrowing the scenario to authored selections + contract + known conformance until G1 defines environment-evidence semantics.

### 2. Medium — final cross-owner publication has no named executor
The preparation spec commits to atomic behavior ("Final publication SHALL install only complete, already-validated authority and Trainer state") entirely in passive voice. D5 says "the authority and Trainer publish … together," D8 says publication "is an unobservable installation" — but neither the design nor any spec assigns who *coordinates* the joint commit across two ownership surfaces (authority routes/views + Trainer backend/optimization state). The exchange record names "one publication coordinator" and explicitly leaves its ownership open; that openness did not get carried into the OpenSpec as an open question or gate item. Similarly, the preparation job "SHALL be derived atomically" — by the authority, the accepted arrangement, or Trainer infrastructure? Given the review criteria (gaps in authority/ownership) and that task 1.4 will freeze these rows, the deriving/publishing owner should be stated or explicitly listed under G1/`design.md` Open Questions.

### 3. Medium-minor — "readiness checkpoint" is a load-bearing but undefined concept
Used normatively in `training-contract` (line 77), `accepted-training-execution` (line 12), and `training-capability-coordination` (line 43), yet no delta requirement establishes that readiness checkpoints are contract-derived during fulfillment and recorded in the accepted arrangement — that exists only in the supporting exchange record (EX-008) and design prose. Relatedly, the fulfillment-time vs. later-request-time split for configuration (contract spec says config "MAY parameterize" an exposed capability; capability spec permits "or later request validation") is left implicit. One defining requirement in `training-contract` would close both.

### 4. Minor — terminology drift around the fulfillment boundary
Four names for one mechanism: **"strategy fulfillment"** (design, contract/execution specs), **"the acceptance mechanism"** (`training-contract` ×2), **"strategy acceptance check"** (`training-capability-coordination` ×3, `training-optimization` ×2), and **"strategy-side acceptance mechanism"** (proposal.md:20 — which could even be misread as fulfillment living inside the strategy, contradicting D2). Since task 1.4 explicitly verifies "no … independent validator authority," using a different noun phrase per spec invites exactly the perception of a second validator. Unify on "strategy fulfillment" (with "the training contract system" as the owner).

### 5. Minor — scenario asymmetry weakens the earliest-evidence rule
The "Runtime evidence is intentionally deferred" scenario requires fulfillment to establish every decision "knowable **from the authored definition**," while the sibling scenario's knowledge base is "authored selections, active contract, known implementation conformance, and available environment facts." As written, the deferral scenario could be read as excusing contract/conformance/environment checks. Align the two knowledge bases.

### 6. Minor — title/body mismatch on the bypass requirement
"Raw authored definitions **cannot** bypass strategy fulfillment" (absolute) vs. body "The **normal** Trainer construction path SHALL accept only…" (qualified). The scope of non-normal paths (tests, tools, migration shims) is undefined; either drop "normal" or state the exception boundary.

### 7. Minor — tasks 1.4 row list doesn't match D5
Task 1.4 enumerates "strategy fulfillment, **governed realization**, binding, preparation, …" (11 items), but "governed realization" is the umbrella phase (D1 diagram, D8), not a D5 row, and D5's "Materialize and bind" appears as "binding." Listing the umbrella as a peer row will muddy the 1.4 deliverable. Also note the fulfillment-vs-acceptance-check drift should be fixed *before* 1.4 builds the traceability matrix.

### 8. Nits
- "unless an explicit migration contract is later defined" (`training-contract`, one-contract requirement) conditions a SHALL on a nonexistent artifact; consider scoping it out and leaving migration to a future delta.
- The candidate-failure and stale-route invariants each appear in two specs (execution vs. preparation; execution vs. capability). This is acceptable reinforcement — both consistently name run-specific obligations as the criterion — but keep one canonical phrasing to avoid drift.

## Suggested actions

1. Ground or narrow the "environment facts" scenario (finding 1) — highest value before task 1.4 freezes the evidence model.
2. Name the preparation-job deriver and final-publication coordinator, or add them to G1/Open Questions explicitly (finding 2).
3. Add one defining requirement for readiness checkpoints (and the config timing split) in `training-contract` (finding 3).
4. Unify the fulfillment terminology across all 15 delta files (finding 4) and align tasks 1.4 with D5's rows (finding 7).

None of these block the change's stated purpose; the architecture and obligation model are sound, and the production gate remains properly closed behind G1–G5.

---
## Review Trace
- Workspace: /home/imi/Projects/sd-scripts
- Diff target: unstaged
- Context entries requested: 3
- Context files loaded: 20
- Files selected for initial diff: 6
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 62547 prompt tokens, 475 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: get_uncommitted_changes
- Tool call: list_repository_tree
- Model call: API usage 66745 prompt tokens, 316 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_changed_files
- Tool call: read_files
- Model call: API usage 69256 prompt tokens, 9793 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 79574 prompt tokens, 135 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 81106 prompt tokens, 302 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 82386 prompt tokens, 140 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 84964 prompt tokens, 137 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 86216 prompt tokens, 4344 completion tokens
- Review complete after 8 model call(s)