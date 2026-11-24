# Document roles and current homes

This plan separates authority, derivation, discussion and evidence. It is an
organizational result of the walkthrough and subsequent authorized archival.
Sixteen historical source/package files now live under
`docs_design/archive/models-strategy-trainer/`. Their bytes are unchanged;
no production layout is chosen.

The [source-authority rule](../../../openspec/changes/rework-model-strategy-trainer/design.md)
already decides which content governs. A newer timestamp, a location under
`docs_design`, or a favorable review does not create authority by itself.
[placement-manifest.json](placement-manifest.json) classifies every one of the
68 inventoried Markdown documents and every original root item individually.

## Where each class lives

| Class and current records | Role and current disposition | Current home / handling |
| --- | --- | --- |
| Active OpenSpec proposal, design, tasks and 17 delta specs | Governing active change, including accepted carry-forward and later amendments. Task status is not evidence of production cutover. | Keep under `openspec/changes/rework-model-strategy-trainer/`. Sync/archive only through a separately authorized change workflow when its gates are satisfied. |
| `strategy_system_direction.md` | Settled pre-OpenSpec architecture inherited by the change. Current text includes later revisions. | [Archived direction](../../archive/models-strategy-trainer/strategy_system_direction.md). Governing links follow the move; its inherited authority is preserved. |
| `strategy_contract_exchange_design.md` | Detailed derivation with settled, working and superseded EX entries. Adopted requirements live in OpenSpec. | [Archived exchange derivation](../../archive/models-strategy-trainer/strategy_contract_exchange_design.md). EX identifiers, statuses and original bytes are preserved. |
| `strategy_system_implementation_mapping.md` | Source-to-target responsibility derivation and ordering, preceding final production placement. | [Archived responsibility map](../../archive/models-strategy-trainer/strategy_system_implementation_mapping.md). It remains derivation, not a final module checklist. |
| `notes.md`, `strategy_discussion.md`, `strategy_discussion_external.md`, `provisional_tree.md` | Chronology, copied primary/external discussions and a copied provisional sketch. External proposals and historical assistant suggestions do not become requirements automatically. | Archived together in `docs_design/archive/models-strategy-trainer/`, linked from the archive index. Source messages and supersession links remain in history. |
| `strategy_system_inventory.md` and `inventory_01_02.7z` | Historical production-baseline evidence and two earlier inventory drafts. Old code observations do not govern the target. | [Archived inventory](../../archive/models-strategy-trainer/strategy_system_inventory.md); its original 7z is in the archive packages folder. Readable draft copies remain in history. |
| `framework_pattern_comparison.md` | Prior art; its old active-strategy topology is explicitly superseded. Boundary lessons remain useful. | [Archived prior art](../../archive/models-strategy-trainer/framework_pattern_comparison.md), with later adopted/rejected lessons traced in history. |
| Broad training/data research, `answer-edited.md`, mechanism sketch, architecture reports, Mock A and language proposal | Research input and evolving sketches. Individual decisions are adopted only through the governing design/deltas. Custom-baseline recommendations and illustrative operation/API spellings are not current mandates. | Keep under the active change's existing `research/` folder. Its index should continue pointing to the governing adoption; do not scatter these inputs among production module READMEs. |
| `candidate_comparison.md`, `matched_comparison.md`, `grouping_comparison.md`, `machinery_comparison.md` and their JSON/evidence | Sequential comparison records with distinct scope, aligned semantics and shared-runtime qualifications. Later selection does not erase the earlier measurements. | Keep current paths with their fixtures, source hashes and evidence. These are retained experimental bundles; a later bundle relocation would require its own import/reference and replay checks. |
| `executable_spike/`, `execution_candidate/`, `xdsl_candidate/`, `pliron_candidate/`, `xdsl_language/` | Runnable research and qualification packages. The first candidate's owners are reused by later targets; Pliron's native and Python pieces remain one experiment. | Keep each package and retained evidence together at present. A later `experiments/` grouping requires an actual code/import/command migration, not a Markdown-only move. Their READMEs continue describing their implemented bounded behavior. |
| `model_target.py`, `grant_target.py`, `mixed_target.py`, comparison fixtures and root `evidence_*` directories | Shared experimental targets, controls and retained observations referenced by multiple candidates. | Keep with their owning experiment/comparison graph. Determine callers and replay paths before relocation; a directory name alone does not establish ownership. No structural code audit or move is claimed here. |
| May strategy notes/critique and contract-pressure audit outside the starting folder | Antecedents about the older family interface, optional seams, context, objective/caching split and possible plugin shapes. They are not an adopted alternative to the July–November system. | Archived in `docs_design/archive/models-strategy-trainer/pre-direction/`; the supplementary history and relocation map retain their original paths and dates. |
| Local ZIP package snapshots | Exact dated-content witnesses, not extra active changes. The first abandoned draft has a separately reconstructed authoring trail. The two later ZIPs match recorded Git blobs. | Archived in `docs_design/archive/models-strategy-trainer/packages/`, with the original inventory 7z. Exact bytes, package/member provenance and historical paths are preserved. |
| This `history/` workspace | Derived, source-qualified navigation, readable sessions, immutable captures and current-disposition pointers. It is not new architectural authority. | Keep as the audit record. [Current decisions](current-decisions.md) is the entry point; decision details remain in the notepad and original governing sources. |

## Preservation and navigation

[archive-relocations.json](archive-relocations.json) records every source and
archive path, byte hash and move. Archived originals, all readable message
bodies, exact Git snapshots, source records and earlier validation receipts
remain byte-identical. Current OpenSpec and history references use the archive
locations. Relative links inside archived originals retain their original
context; use the maintained history and archive indexes for current navigation.

The original placement manifest retains the inventory's source paths and adds
`current_path`/`relocation_performed` for moved records. Historical receipts
remain records of the earlier captures, rather than being rewritten to pretend
that archived documents always lived at their present locations.

Runnable packages, comparison documents/fixtures and retained evidence keep their
existing paths. The active OpenSpec change is still active. This archival does
not choose a production package tree or begin the main implementation.
