# Review: `rework-model-strategy-trainer` OpenSpec planning package

## Summary

The package is architecturally faithful. It carries forward the settled direction (contract-first authoring → passive strategy → accepted arrangement → one Trainer engine), the Q1–Q5 identity/binding/freshness semantics, the D8 preparation publication rules, optimization unit identity and non-overlap, artifact product/plan/result staging, the TrainingMode dissolution mapping (D12), and the topology correction that supersedes the active-strategy runtime. It does not invent unresolved code shapes (names, IR, envelopes are explicitly deferred to gates), correctly demotes `mst_pre_astra.zip` to comparison evidence, and preserves the production-readiness gate (tasks sections 1–5 are design-only; section 5.7 defers apply work to a separate workflow).

However, there are **three broken delta/main-spec requirement matches**, a **direct normative contradiction with the active metadata change**, and **two main specs with `AdapterMode`-gated scenarios that have no deltas** — i.e., precisely the "explicit reconciliation" failures the change says it exists to fix.

## High-severity findings

### 1. Three MODIFIED requirements are renamed, breaking delta↔main matching

OpenSpec matches MODIFIED/REMOVED requirements to the main spec by header. All three of these titles have no counterpart in the main spec:

| Delta (rework) | Main spec actual |
|---|---|
| `adapter-system`: "Adapter persistence belongs to **the accepted adapter product path**" (line 18) | "Adapter persistence belongs to **the adapter-training path**" (`openspec/specs/adapter-system/spec.md:63`) |
| `optimization-target-refs`: "Module and parameter refs expand from **accepted component projections**" (line 66) | "Module and parameter refs expand from **declared components**" (`openspec/specs/optimization-target-refs/spec.md:85`) |
| `training-observability`: "Backbone observability derives from **accepted participant projections**" (line 21) | "Backbone observability derives from **declared loaded components**" (`openspec/specs/training-observability/spec.md:151`) |

Consequences, depending on tool behavior:
- If `openspec validate --strict` checks modified-requirement existence (it does in current openspec-cli), **validation fails**, contradicting the notes' claim that strict validation passes and making task 1.2's re-verification more than a formality.
- If it passes structurally, the old requirements survive unmodified **alongside** the new ones — including `adapter-system`'s old requirement whose scenario mandates "the save behavior MUST be owned on the training side by the adapter path through `AdapterMode`", directly contradicting the new ADDED/REMOVED deltas and the whole D12 dissolution.

Fix: either keep the original headers (renaming is cosmetic here), or use explicit REMOVED(old)+ADDED(new) pairs so the replacement is auditable.

### 2. Direct contradiction with the active `expand-model-metadata-identity-and-structure` change on `loaded-model-components`

Both active changes MODIFY "Model loading returns the loaded-component surface" with incompatible normative text:

- Metadata change delta: the loaded-component surface "MUST remain the **authoritative** top-level component representation for downstream runtime code"; the trainer "MUST treat the loaded-component surface as its **primary** top-level model representation" and "MUST retain the typed loading/provenance state"; deferred loading returns typed updates via `load_denoiser_lazily()`.
- Rework delta: the surface "SHALL NOT become a competing canonical runtime binding collection"; Trainer "MUST NOT retain a competing authoritative loaded-component collection or fixed family slots"; candidates enter use "through accepted participant materialization".

These cannot both hold. Worse, the rework's replacement text **omits the typed loading-result/provenance-evidence obligations and the deferred-loading typed-update scenario entirely**, so whichever change archives second silently deletes the other's requirements — the exact "silent supersession" the proposal (Impact, last paragraph) promises to avoid. The rework delta pre-commits one side of the reconciliation that task 1.2 defers. The provenance-evidence retention obligation currently has no home in any rework spec (`model-family-metadata`'s single ADDED requirement covers identity projection, not loading evidence).

Fix: merge the requirement text so provenance/source evidence obligations survive under the new authority model (loading evidence ≠ binding authority), and record the archive-order sequencing constraint in `design.md` now rather than only as a future task.

## Medium-severity findings

### 3. Missing deltas: `repo-owned-lora-method` and `repo-owned-vera-method` still gate scenarios on `AdapterMode`

- `openspec/specs/repo-owned-lora-method/spec.md:12`: "WHEN `AdapterMode` prepares a PEFT run whose selected method is `lora` …"
- `openspec/specs/repo-owned-vera-method/spec.md:9`: "WHEN `AdapterMode` prepares a PEFT run whose selected method is `vera` …"

After `TrainingMode`/`AdapterMode` dissolution these WHEN triggers can never fire, leaving normative scenarios unsatisfiable. Design D12 and the proposal enumerate exactly six affected capabilities and miss these two — notable because the design's own critique of the archived draft was that it "did not resolve main-spec requirements that still make `AdapterMode` the permanent owner." Minor related residual: `training-observability` main spec's "Training observability uses repo-owned structured signals" scenario still reads "WHEN trainer, **mode**, or phase code emits observability facts" and is not covered by the delta.

### 4. Cross-change sequencing is asserted but not encoded

The metadata change's in-flight sections 4–6 (typed loading result, trainer setup retention, deferred-materialization events at trainer boundaries) modify exactly the boundaries the rework plans to replace, and its section 8 gate depends on beads `sd-scripts-edl`/`sd-scripts-d79`. Task 1.2 says to *record* sequencing constraints, but nothing in the package states which change lands first, whether the rework's G4 reconciliation (task 4.6) must wait for metadata sections 1–7 or only 1–6, or how the SDXL vertical cutover (migration step 4) interacts with a partially landed metadata change. If the metadata change archives mid-way through the rework's gates, the rework deltas (written against pre-metadata main specs, per finding 2) will delete its obligations. This needs an explicit ordering statement, not just an audit task.

## Low-severity findings

5. **`optimization-target-refs` interaction with the metadata change's ADDED requirement.** The metadata change adds "Target refs can reference accepted structural identities" (live refs preserved "alongside their live execution references"); the rework modifies "Shared optimization target refs" to require live handles "only through revision-pinned consumer views." Reconcilable, but the rework delta never references it, and the rework's `top-level component target refs derive from declared loaded components` text doesn't account for revision/realization-qualified structural identity. Should be captured in the task-1.2 audit list explicitly.

6. **Stale temporal reference carried into a new delta.** The rework's modified "Existing grouping behavior remains stable" keeps "Adapter grouping remains component-LR based **in this change**" — "this change" originally referred to the shared-target-refs change; reusing it in the rework delta makes the scope ambiguous. Reword to "until an explicit adapter-grouping change."

7. **Pre-existing ecosystem defect worth noting while touching these specs:** `openspec/specs/repo-owned-vera-method/spec.md` is delta-shaped (starts with `## ADDED Requirements`) rather than a proper main spec — evidence that main-spec hygiene checks are weak, which is the root enabler of findings 1 and 3.

8. **Cosmetic:** `mst_pre_astra.zip` sits inside `openspec/changes/` rather than `archive/` or `docs_design/`; the design correctly treats it as evidence, but its placement invites tooling/tool confusion. Also, tasks section 1 is labeled as "G1" but 1.1–1.3, 1.5 are carry-forward/audit work beyond G1's "complete Trainer consumption meanings" scope — a labeling nit only.

## Verified as correct (no action)

- **Faithful carry-forward:** D1–D13 map cleanly onto the direction document and exchange-design decisions; superseded positions (EX-007/015/017/018, the active-strategy topology, the `PreparedTrainingProgram` façade, Lightning adoption) are rejected, not promoted. The 2026-08-26 topology correction (passive authored strategy, authority inside the accepted arrangement) is reflected in D4/D6 and the `accepted-training-execution` "survives the authoring object" requirement.
- **No invented code shapes:** specs stay at SHALL/scenario level; class names, IR choice, envelopes, and authoring API are explicitly gated (G5, Open Questions). Spike vocabulary is not renamed into requirements.
- **Gates preserved:** tasks.md header and section 5.7 keep production apply behind completed gates, user review, and strict validation; D13 and migration rule 4 forbid dual canonical stores and renamed mode authorities.
- **Baseline accuracy:** the design's "current production baseline" matches the actual code (`train.py:18–29` builds strategy+mode then Trainer; `trainer.py` builds the objective in `__init__` and stores `self.mode`; mode owns checkpoint routing).
- **Internal consistency:** 13 delta dirs = 7 new + 6 modified as the proposal claims; G1–G5 ↔ sections 1–5; the representative case matrix matches tasks 2.1–2.5; spec cross-references (execution↔optimization authority transfer, capability readiness↔route freshness, persistence↔restoration separation) are mutually consistent.

## Recommended priority

1. Fix the three renamed MODIFIED headers (or convert to REMOVED+ADDED) and re-run strict validation.
2. Rewrite the `loaded-model-components` modified requirement to carry the metadata change's provenance/evidence obligations under the new authority model, and record archive-order sequencing in `design.md`.
3. Add `repo-owned-lora-method` / `repo-owned-vera-method` deltas (or scenario rewording) so no main-spec scenario is gated on a dissolved `AdapterMode`.
4. Add the optimization-target-refs structural-identity item to the task-1.2 audit list and fix the "in this change" wording.

---
## Review Trace
- Workspace: /home/imi/Projects/sd-scripts
- Diff target: unstaged
- Context entries requested: 6
- Context files loaded: 21
- Files selected for initial diff: 4
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 97694 prompt tokens, 277 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: list_repository_tree
- Model call: API usage 98243 prompt tokens, 288 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result truncated: read_files from 24598 to 20000 chars
- Tool call: read_files
- Tool result truncated: read_files from 34866 to 20000 chars
- Tool call: read_files
- Model call: API usage 107328 prompt tokens, 1436 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 111500 prompt tokens, 147 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 112996 prompt tokens, 85 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: search_repository
- Model call: API usage 113570 prompt tokens, 156 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result truncated: read_files from 23107 to 20000 chars
- Tool call: read_files
- Tool call: list_repository_tree
- Model call: API usage 117948 prompt tokens, 219 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_files
- Model call: API usage 119108 prompt tokens, 6238 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 128243 prompt tokens, 193 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 130702 prompt tokens, 641 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_files
- Model call: API usage 133555 prompt tokens, 333 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 134809 prompt tokens, 855 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 136571 prompt tokens, 197 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 137313 prompt tokens, 4858 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 142258 prompt tokens, 2298 completion tokens
- Review complete after 15 model call(s)