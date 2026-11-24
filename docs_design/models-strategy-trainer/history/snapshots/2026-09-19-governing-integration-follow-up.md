# Review: `rework-model-strategy-trainer` follow-up verification

## Verdict

The follow-up is sound. All five verification points check out against the repository state; I found **no blocking issues**, three minor follow-ups worth recording before milestone 1 closes.

## Verification against the five criteria

### 1. MODIFIED requirements match main specs and change semantics correctly ✓

I verified every MODIFIED/REMOVED header against its main spec, plus scenario-set preservation:

| Delta | Headers match main spec | Scenario handling |
|---|---|---|
| `loaded-model-components` | ✓ (2/2) | Metadata-change's four scenario names preserved 1:1 (incl. the two not yet in main spec — valid under wholesale MODIFIED); 3rd scenario added intentionally |
| `optimization-target-refs` | ✓ (5/5) | All 9 existing scenario names preserved; bodies correctly de-mode-ified |
| `adapter-system` | ✓ (2 MODIFIED, 2 REMOVED) | Both scenario pairs preserved; removals carry proper Reason + Migration |
| `training-observability` | ✓ (3/3) | All scenario pairs preserved; "phases, modes, or strategies" language replaced with accepted projections |
| `repo-owned-lora/vera-method` | ✓ (1/1 each) | Both scenarios preserved; `AdapterMode` trigger replaced with accepted participant/capability path |

Semantic changes are correct: loading evidence vs. binding authority split, live-object access demoted to revision-pinned views, adapter persistence moved to capability + infrastructure, observability decoupled from mode/strategy introspection. The `Existing grouping behavior remains stable` rewrite also fixes the old "in this change" temporal trap ("before a later explicit adapter-grouping change").

### 2. Typed loading/provenance evidence survives; authority moves ✓

The `loaded-model-components` delta keeps every metadata-change obligation: typed result contents (observations, selections, bindings, decisions, transformations, limitations), evidence retention "required for catalog resolution and realization-composition filing," and deferred-update evidence preservation — while stating "Loading evidence SHALL NOT become a competing canonical live-binding collection" and routing currency exclusively through accepted participant transitions. Consistent with `run-participant-state` and the `model-family-metadata` ADDED delta. The rework's ADDED `model-family-metadata` requirement does not collide with the metadata change's two MODIFIED headers there.

### 3. Cross-change sequencing coherent and non-destructive ✓

`loaded-model-components` and `optimization-target-refs` are the real overlap (both changes carry pending deltas on the same requirements). Design.md's rule — amend the metadata change before either syncs/archives, metadata lands first, rework rebases, production gated on the typed loading contract — is coherent, and the rework delta is already written against the post-metadata-archive shape. Nothing deletes the loading-evidence obligations.

### 4. Residual AdapterMode-gated scenarios all replaced ✓

Repository-wide search confirms the only remaining `AdapterMode` references in main specs are: `adapter-system` (5 clauses — covered by 2 modifications + 2 removals), `adapter-module-targeting` lines 14/68/106 (covered: realization-request pass-through; clause dropped with ownership covered by the ADDED "Pipeline coordination replaces mode ownership"; "Trainer/pipeline orchestration MUST own lifecycle timing rather than `AdapterMode`"), `repo-owned-lora-method` line 12 and `repo-owned-vera-method` line 9 (both replaced). **Zero** `TrainingMode`/`TrainingStrategy`/`process_batch` references remain in any main spec, and no capability outside the eight modified ones encodes the old topology. The malformed delta-shaped `repo-owned-vera-method` main spec is explicitly tracked in task 1.2.

### 5. No new contradictions or premature code shapes ✓

All `AdapterMode`/`TrainingMode` mentions inside the change are negative/removal statements; existing-seam references are hedged ("`LoadedAdapterRuntime` or its accepted successor"); design Open Questions still defer names/encodings/authoring API. The optimization-target-refs delta stays consistent with the unmodified main-spec selector-compatibility requirement and with `training-optimization`'s tied-alias dedup rule; no conflict with the metadata change's ADDED structural-identity requirement (overlap is recorded in design.md).

## Findings (minor)

1. **Stale scenario titles in `optimization-target-refs` delta** — "Fine-tune mode consumes parameter targets" and "Adapter mode consumes module targets" keep mode vocabulary in titles although the bodies are correctly rewritten. MODIFIED requirements replace wholesale, so renaming is safe and would remove the last "mode" residue from the change's own specs. (Presumably kept for main-spec traceability; either rename or note the intent.)

2. **Dropped prohibition in `loaded-model-components` delta** — the metadata change's requirement text prohibits "family-specific provenance fields"; the rework version keeps only the tuple/evidence-free prohibitions. The prohibition survives via the "Families have different source layouts" scenario, but restore the clause (or confirm scenario coverage) when rebasing against the amended metadata main spec, so the rebase can't be read as weakening it.

3. **One-sided archive guard** — the "SHALL be amended before either change syncs or archives" obligation lives only in this change's design.md; the metadata change has no marker, and its section 14 archive handoff doesn't reference the boundary. The dependence is bidirectional: if the metadata change archived unamended, its "trainer MUST treat the loaded-component surface as its primary top-level model representation" wording would become main spec (the pending rework delta resolves it — fine), but if the rework archived first, the metadata change's pending delta would silently reintroduce trainer-owned authoritative-surface wording at its own archive. Consider making the metadata-change amendment an explicit tracked deliverable (e.g., named output of tasks 1.2/4.6) rather than prose-only.

4. *(Info)* The `adapter-module-targeting` delta drops the main-spec "`AdapterMode` MUST remain the training-side orchestrator…" clause without an inline replacement; ownership is covered by the `adapter-system` ADDED requirement. Cross-delta coverage is coherent, just split — acceptable.

## Notes

- The strict-validation claim is structurally consistent: every MODIFIED/REMOVED header matches its main spec exactly, REMOVED entries include Reason/Migration, and all delta sections are well-formed.
- This round's uncommitted diff touches only `docs_design/` records (status handoffs pointing at the governing change) — appropriate, no normative drift introduced there.

---
## Review Trace
- Workspace: /home/imi/Projects/sd-scripts
- Diff target: unstaged
- Context entries requested: 11
- Context files loaded: 28
- Files selected for initial diff: 9
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 43693 prompt tokens, 9376 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: get_uncommitted_changes
- Tool call: search_repository
- Model call: API usage 55774 prompt tokens, 430 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 57995 prompt tokens, 738 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 59060 prompt tokens, 2733 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: list_repository_tree
- Model call: API usage 62025 prompt tokens, 155 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 63055 prompt tokens, 4773 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 67898 prompt tokens, 2255 completion tokens
- Review complete after 7 model call(s)