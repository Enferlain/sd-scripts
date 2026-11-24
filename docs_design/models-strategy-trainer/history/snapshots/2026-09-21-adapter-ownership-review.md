# Review: `rework-model-strategy-trainer` planning correction

## Summary

The correction is coherent and well-propagated. The three-way targeting split (semantic intent → strategy authoring, concrete resolution → governed PEFT realization, trainability/grouping → Trainer optimization) and the publication / new-run-initialization-with-lineage / exact-restoration separation are stated consistently across `proposal.md`, `design.md` (D11/D12), all adapter-related deltas, `training-artifact-persistence`, and `tasks.md`. I verified MODIFIED/REMOVED heading integrity against every inherited main spec and searched all main specs for residual mode ownership. No contradiction inside this change's own artifacts. The findings below are completeness gaps and cross-change conflicts, mostly items milestone 1.2 must explicitly record.

## Verified consistent

- **Targeting split**: `adapter-module-targeting` ADDED requirement, `adapter-system` MODIFIED "Optimization-to-adapter boundaries stay contract-based" ("SHALL NOT make optimization the owner of semantic adapter target intent or host-target resolution"), `optimization-target-refs` MODIFIED "Target refs are policy-neutral", lora/vera MODIFIED bodies, design D12 table + "two distinct meanings" paragraph, and tasks 2.2/5.5 all agree. No artifact still assigns semantic targeting or host resolution to optimization; "Existing grouping behavior remains stable" was correctly retained *without* carrying over the old targeting ownership.
- **Persistence/restoration separation**: design D11's new adapter paragraph, `adapter-system` ADDED "Adapter products, artifact loading, and exact restoration are distinct", the MODIFIED continuation requirement (strict continuation ≠ same-run restoration; new incarnation + lineage), `training-artifact-persistence` ("Artifact products are independent from training treatment", "Runtime restoration is not trained-artifact persistence", "Artifacts have identity distinct from participants"), `run-participant-state`, `model-family-metadata` ADDED, and tasks 4.3/4.4 all state the same identity/lineage/result semantics in both directions.
- **Delta mechanics**: every MODIFIED requirement name and retained scenario heading matches its main spec (adapter-module-targeting ×5, adapter-system ×3, optimization-target-refs ×5, lora ×1, vera ×1; I also verified loaded-model-components ×2 and training-observability ×3 against their main specs — added scenarios only, which is valid). All REMOVED names match main-spec requirement names. The two mode-named `optimization-target-refs` scenario headings are retained exactly as design.md claims, with bodies assigning no mode ownership.
- **Mode-gated coverage**: searching all 12 main specs, every `AdapterMode`/mode reference falls inside a requirement this change MODIFIES or REMOVES. `optimizer-group-lr-behavior` and `component-qualified-selectors` encode no old targeting ownership and correctly need no delta (grouping stays optimization-owned; selector surface unchanged).
- **No accidental loss in `optimization-target-refs`**: moving "live module/parameter object" fields to revision-pinned consumer views is the intended correction per the proposal, and selectors/provenance are preserved.

## Findings

1. **Cross-change MUST conflict on `loaded-model-components`** (medium). Both active changes MODIFY "Model loading returns the loaded-component surface" / scenario "Trainer stores the primary loaded-component state" with conflicting retainers: the metadata change's delta says "the trainer MUST retain the typed loading/provenance state required for catalog resolution and realization composition filing", while this change's delta says "Trainer MUST NOT retain a competing authoritative loaded-component collection … the accepted arrangement or its scoped loading/provenance projection MUST retain the typed evidence…". Design.md acknowledges the overlap category and forbids independent archiving, but this concrete conflicting MUST is not yet recorded anywhere. Task 1.2 should capture it verbatim; the metadata delta will need revision when the shared boundary lands. Relatedly, the metadata change's `optimization-target-refs` ADDED requirement uses "WHEN **optimization** builds a target" and "MUST preserve its **existing live parameter** and selector behavior" — vocabulary/semantics that predate this change's revision-pinned views and "accepted consumer" actor. 1.2's current wording ("revision/realization-qualified structural identities in shared target refs") only partially covers this.

2. **`adapter-module-targeting` main "Method-local adapter config owns runtime settings translation" left unchanged** (medium-low). "The forward adapter config surface SHALL **select** an adapter method explicitly" now competes with `adapter-system` ADDED ("strategy construction … SHALL select that participant's method from the declared set") and design D2/D12 (configuration may not choose implementations; selection is explicit strategy construction before fulfillment). It can be read as config parameterizing a declared choice, but as written the unchanged requirement makes the config surface the selection authority. Either MODIFY it (config selects within the strategy's declared supported set) or record it in the 1.2 audit as compatible-by-interpretation.

3. **Two scenario meanings dropped with the removed targeting requirement** (low). The REMOVED "Optimization-owned adapter module targeting" carried "Absorbed methods move toward fully repo-owned algorithm implementations" (general absorbed-method end state; only the loha-specific copy survives in the MODIFIED loha requirement) and "First slice does not define all absorbed methods as module-only" (future methods may need parameter-granular binding). The replacement ADDED requirement and the provenance requirement are module-worded; nothing now explicitly admits parameter-granular adapter realization for future methods (only `optimization-target-refs` supports parameter targets, with no adapter-side statement). If deliberate, fine; otherwise re-home one clause.

4. **Continuation-config ownership unstated** (low). The MODIFIED "Continuation intent defaults to strict continuation" keeps `peft.continue_from` / `peft.continue_mode` as the normative decision surface, while `adapter-system` ADDED makes continuation intent a strategy-declared, pre-materialization concern. Who maps config keys to authored continuation intent (strategy construction vs. later capability request) is unspecified — this is exactly the seam where a config-driven path could re-form around the accepted arrangement. One clarifying clause would close it.

5. **Stale migration-scoped requirements inherited silently** (low). `adapter-module-targeting` main "Adapter target refs remain compatible during migration" (premised on "the shared target-ref migration", references `module` field) and "Mixed-method overlap semantics remain out of scope for this slice" ("this slice" = the old loha spike) are compatible but reference superseded frames; `optimization-target-refs` main "Target refs preserve selector compatibility" and `component-qualified-selectors` retain "optimization target refs" naming that this change elsewhere disclaims as ownership. Leaving them unchanged is defensible, but 1.2's audit should list them as recorded unchanged dependencies rather than inheriting them implicitly.

6. **`repo-owned-vera-method` main spec is malformed** (info, already tracked). It begins with "## ADDED Requirements" and has no `## Requirements` section, so this change's MODIFIED requirement cannot match it and `openspec validate --strict` will fail until the main spec is normalized. Task 1.2 covers this — flagging it as a hard prerequisite for the validation checkbox, not a new finding.

7. **Terminology drift** (nit): "governed PEFT realization" (ADDED requirement, adapter-system migration text, design) vs "governed adapter realization" (adapter-module-targeting REMOVED migration text) vs "adapter-domain realization" (lora/vera bodies). Same concept; unify before G5 naming work, since task 5.5 explicitly derives names here.

## Minor notes

- `proposal.md`'s `training-artifact-persistence` capability blurb doesn't mention product-independence-from-training-treatment, which is now a normative ADDED requirement in that delta; harmless, but the blurb is the skim-level index of what changed.
- `adapter-module-targeting` ADDED scenario "optimization MUST receive **only** the trainable parameter references returned after adapter realization" is scoped to adapter materialization, but "only" could be misread as excluding base-parameter subjects in joint training; the design and task 2.2 make the joint case clear, so this is wording only.

Overall: the correction achieves its stated intent with no internal contradictions; the actionable work is recording the two concrete cross-change conflicts (finding 1) and closing the config-selection/continuation ownership leftovers (findings 2 and 4) in milestone 1.2.

---
## Review Trace
- Workspace: /home/imi/Projects/sd-scripts
- Diff target: HEAD
- Context entries requested: 6
- Context files loaded: 23
- Files selected for initial diff: 9
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 53636 prompt tokens, 14996 completion tokens
- Model call: model requested 3 tool call(s)
- Tool result truncated: read_files from 33627 to 20000 chars
- Tool call: read_files
- Tool call: list_repository_tree
- Tool call: search_repository
- Model call: API usage 73637 prompt tokens, 1231 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 78764 prompt tokens, 4519 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Tool call: list_repository_tree
- Model call: API usage 84809 prompt tokens, 716 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 87199 prompt tokens, 3495 completion tokens
- Review complete after 5 model call(s)