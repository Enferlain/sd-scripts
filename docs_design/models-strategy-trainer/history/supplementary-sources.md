# Earlier notes and local archive closure

These records close the bounded provenance leads left by the initial folder
inventory. They do not extend the main design walkthrough into an exhaustive
audit of every older repository session. Exact paths, bytes, hashes, Git matches
and inode/entry dates are in
[supplementary-source-records.json](supplementary-source-records.json).

Original files now live in the
[historical archive](../../archive/models-strategy-trainer/README.md).
The source records retain the paths at capture time;
[archive-relocations.json](archive-relocations.json) resolves their current
locations without changing their bytes or the earlier provenance.

## May antecedents

The first Git record for
[`strategy_architecture_ideas.md`](snapshots/pre-direction/64203e41-strategy_architecture_ideas.md)
is `64203e41`, 5 May 2026 23:21:37 UTC. It collects experience with the denoiser
execution seam, definition-oriented contracts, explicit signatures and scoped
runtime context. It explicitly permits a larger future redesign; it does not
establish the later whole-run architecture.

[`strategy_critique.md`](snapshots/pre-direction/f36fcc7c-strategy_critique.md)
and the
[`strategy_contract_pressure_audit.md`](snapshots/pre-direction/f36fcc7c-strategy_contract_pressure_audit.md)
first enter Git at `f36fcc7c`, 7 May 2026 00:40:21 UTC. The critique says it is
code-only and proposes a typed family-plugin/facet shape. The pressure scan is
narrower, classifying current required/optional and family-internal seams and
marking some observations resolved. They contain different recommendations.
Neither proposal supersedes the later accepted arrangement, contract-derived
conformance or whole-workflow direction simply because it raises a related
problem. Their useful rationale is separation of responsibilities and explicit
contracts. Their old fixed family interfaces are historical implementation
suggestions. This audit classifies their text; it does not rerun their source
claims against current production.

## Two inventory drafts

`inventory_01_02.7z` contains exactly two files, now readable as exact copies:

- [v01](snapshots/inventory-archive/strategy_system_inventoryv01.md),
  31,011 bytes, reader-reported entry time 26 July 2026 01:25:17 UTC.
- [v02](snapshots/inventory-archive/strategy_system_inventoryv02.md),
  34,671 bytes, reader-reported entry time 29 July 2026 00:08:14 UTC.

These are earlier inventories, not alternate target specifications. Both label
their current-code scope and defer final APIs; v02 already distinguishes broad
lifecycle and internal feature seams, state ownership, conditional requirements
and misleading generic component slots. Questions still open in these drafts
must be interpreted through the later discussion and governing dispositions.
Their code claims were not rerun. Archive entry metadata does not establish
authorship, archive creation or user adoption, and these drafts are not exact
copies of an already captured Git inventory snapshot.

The reader ran in an isolated `uv --no-project` process. One optional extraction
dependency enabled that process's GIL. This is archive-tool behavior, not a
new execution-candidate or free-threading qualification. No repository
environment, lockfile or archive content was changed.

## Local OpenSpec packages

| Archive | Observed contents and correspondence | Disposition |
| --- | --- | --- |
| `mst_pre_astra.zip` | The abandoned first draft: nine file entries preserved in [reading 11's sources](sessions/11-source-records.json), with eight exact original authoring-patch correspondences. | Earlier draft evidence; rebuilt governing package follows in reading 12. |
| `rework-model-strategy-trainer922.zip` | Nineteen files, each matching the same path and blob at `58ab91ce`, the September walkthrough checkpoint. | Historical package snapshot; its older design/API questions are not the current frontier. |
| `rework-model-strategy-trainer.zip` | Forty-one files plus nineteen directory entries. Every file matches Git at both `5218fdb1` and `504dcf88`; these are commits where the archived path/blob set is present, not inferred archive-creation dates. | Later package snapshot including research and Mock A; no unmatched local prose needs promotion into current authority. |

ZIP entry times have no specified timezone and are not archive-creation times.
The exact comparison checks file contents, not filesystem timestamps or an
assumption that all entries were written in one session. The original ZIPs now live in the historical archive packages folder; there is no need to duplicate all their bodies because the
governing revisions and source sessions are already accessible.

## Coverage boundary

The central July–November lineage, its inspected fork/page differences, adopted
decisions and resulting governing/experimental checkpoints are reconstructed.
Earlier notes and local archives are classified above. Unregistered sessions,
unavailable/non-text attachments and every incidental backend or tooling
conversation have not been exhaustively recovered. This does not identify an
unresolved architectural decision or make the audit an independent rerun of
historical tests, reviews, benchmarks or source-code inventories.
