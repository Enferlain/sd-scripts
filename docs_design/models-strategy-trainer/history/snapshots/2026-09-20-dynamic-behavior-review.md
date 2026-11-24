Review complete. The staged tree only contains an unrelated `.gitignore` tweak, so the unstaged diff covers the full review scope. All four focus files plus the surrounding specs were evaluated against the provided normative context.

## Overall Assessment

The refinement is coherent and does what it claims: it separates *completed authoring* from *preserved dynamic runtime behavior* without regressing the settled architecture. Specifically verified as consistent:

- **Topology**: "authoring role is complete… no longer an ordinary runtime dependency" plus the two-seam permission matches the 2026-08-26 topology correction and EX-029/EX-030 ("even if one Python object for both"). Nothing reintroduces Trainer ↔ active-strategy as the standard topology.
- **Dynamic behavior**: the new requirement, D10's four-kinds-of-change taxonomy, and the "no frozen linear sequence" clause correctly counterbalance "static choices leave the hot path" without contradiction.
- **Run authority**: the new "Behavior requests a structural transition" scenario routes structural changes through the authority's transition/republication path, consistent with `run-participant-state`.
- **Trainer mechanics / research extension**: the standard-profile and extension-boundary requirements are untouched; bounded runtime decisions are explicitly capped at fulfillment-accepted authority, and kind-4 change matches "One accepted contract governs one run authority".
- **Restoration**: adaptive owned state "MUST participate in applicable persistence or exact-restoration exchanges" is consistent with the capability-coordination and artifact-persistence specs.

Below are issues worth fixing before G2/G3 derive vocabulary from these texts.

## Findings

**1. (Moderate) Unconditioned acceptance in the contract spec — `training-contract/spec.md`, "A decision is deliberately made during execution".**
"fulfillment MUST accept the decision policy…" has no compatibility qualifier. Read literally, *any* decision policy must be accepted, which conflicts with "Custom behavior uses an explicit conformance or extension path" and "Imperative regions receive only accepted authority" when the decision's required authority exceeds the standard profile. The sibling scenario "Author selects adaptive behavior" hedges correctly with "MUST NOT be rejected *merely because*…". Mirror that: e.g., "fulfillment MUST accept the decision policy, its alternatives' bounds, authority, and effects **where they are permitted by the active contract and profile**".

**2. (Moderate) Mechanism-level overcommitment on schedule evaluation — `accepted-training-execution/spec.md`, "Schedule changes behavior during a run", and `tasks.md` 3.6.**
"execution MUST evaluate the selected schedule at runtime" prescribes an evaluation mechanism rather than observable semantics. It can be read to forbid ahead-of-time lowering of a deterministic schedule even when observably equivalent — in tension with D10's "only by reducing… to a frozen linear sequence" (which permits partial lowering) and with the hot-path performance goal. Task 3.6's "without precomputing scheduled or adaptive behavior" reads as a blanket precomputation ban with the same problem. Prefer semantics-level wording: "the current behavior MUST be determined by the accepted schedule semantics for the current coordinate" / "without resolving scheduled or adaptive behavior to fixed authoring-time outcomes".

**3. (Minor) Broken normative grammar — `accepted-training-execution/spec.md` line 85.**
"This requirement SHALL NOT precompute or prohibit accepted schedules…" — a requirement cannot "precompute"; the SHALL clause has no valid subject/action. Use "SHALL NOT be construed to require precomputing, or to prohibit, …".

**4. (Minor) Terminology drift for one concept.**
The same idea appears as "permitted run-state transitions" (D2), "allowed transitions" (D4 arrangement list), "permitted structural transitions" (contract spec), "permitted transitions" (execution spec), and "structural transition" (execution scenario) — where the scenario's scope (bindings, preparation, optimization, capability state) is actually *broader* than "structural". G2/G3 will derive vocabulary from these; unify on one defined term (D10's "authority-governed transition" is the best candidate).

**5. (Minor) The requirement-level widening from "authored strategy object" to "authored strategy's broad construction interface".**
This is intentional (it enables the two-seam case), but it widens more than needed: a per-step call into a narrow method of the *whole* authored strategy object would not be a dependency on the "broad construction interface", effectively permitting a renamed `process_batch()`-style whole-strategy callback. The design guards against this only in prose. Consider stating in the spec that runtime may depend only on *accepted executable implementations and their accepted seams*, never on the strategy object in its authoring role.

**6. (Minor) Testability of the two-seam scenario.**
"retaining that implementation MUST NOT preserve broad authoring or arrangement-mutation powers" is not directly observable. Express the observable contract: the runtime seam receives only accepted inputs/scoped projections, and any arrangement change it causes is visible only as an authority proposal/transition. Also, its interplay with "Authoring object is released" is only conditionally consistent (the WHEN clauses differ, so no logical contradiction), but one clarifying line — that "no longer retained" means not retained *in any capacity, including as an implementation* — would close a pedantic loophole.

**7. (Minor) Operation-owned state invalidation is introduced but unconstrained.**
The refinement adds "owned runtime-state definitions"/"operation-owned state" and requires restoration participation, but nothing yet ties owned-state validity to authority transitions (e.g., adaptive state derived from observations of a participant that is later replaced or rebound). `run-participant-state`'s authority-owned state list doesn't cover it. Tasks 2.1/2.6/3.5 gesture at this ("owned-state dependency", "operation-owned state updates"); make sure G2 explicitly derives freshness/invalidation semantics for owned state rather than leaving it implicit.

**8. (Nit) "its possible effects" in "Author selects adaptive behavior".**
For observation-driven algorithms this must mean a declared effect vocabulary/bounds, not exhaustive enumeration of concrete effects. Consistent with the bounds philosophy, but worth one clarifying phrase so G2 doesn't overburden the authoring API.

## Suggested Actions

Fix findings 1–3 in this refinement (they are normative-wording defects that G2/G3 would inherit). Findings 4–6 are cheap terminology/boundary tightenings best done now while the vocabulary is fresh. Findings 7–8 can be recorded as explicit G2 inputs in `tasks.md` 2.6 if not addressed directly. No architectural contradiction was found; the refinement does not retain the strategy as runtime peer, freeze dynamic behavior, overgrant operation authority, or commit to a representation beyond the one mechanism-level phrase flagged in finding 2.

---
## Review Trace
- Workspace: /home/imi/Projects/sd-scripts
- Diff target: unstaged
- Context entries requested: 4
- Context files loaded: 20
- Files selected for initial diff: 4
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 64289 prompt tokens, 9772 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: get_uncommitted_changes
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 74639 prompt tokens, 2889 completion tokens
- Review complete after 2 model call(s)