# Historical walkthrough workspace

This workspace was requested on 5 November 2026 to recover the reasoning from
the original model/strategy/Trainer discussions through the evolving OpenSpec
and the current language work. It records provenance and supports a joint,
step-by-step reading. It does not establish new architecture or replace the
governing change.

Tracked work: `sd-scripts-syv.32`. Task status belongs in Beads. The notepad here
records decisions and evidence, not an implementation checklist.

The earlier source documents and local packages are now in the
[historical archive](../../archive/models-strategy-trainer/README.md).
[archive-relocations.json](archive-relocations.json) maps their original paths
to the byte-identical archive files. Captured transcripts, snapshots, inventories
and validation receipts keep their original bytes and historical paths.
The [archival checks](archive-validation.json) record byte preservation, current
links and the separately tracked pre-existing strict-validation warnings.

## Start here

The requested historical reconstruction is caught up through the pre-audit
design frontier. This remains a side quest: production G5 and the original
system design have not been replaced or marked complete.

1. [Current decisions and remaining production choices](current-decisions.md):
   the concise carry-forward map, including explicit supersessions.
2. [Decision notepad](decision-notepad.md): H001–H087 preserve original source,
   rationale, current disposition and missing detail/G5 consequence.
3. [Placement plan](placement-plan.md) and [per-file manifest](placement-manifest.json):
   what the documents are and where they now live.
4. [Document chronology](document-map.md), [branch map](sessions/branch-map.md)
   and [source index](sessions/index.json): dates, lineage and exact captures.
5. [Earlier notes and local archives](supplementary-sources.md): bounded closure
   of the initial provenance leads.

## Readable chronological conversations

These preserve complete visible message bodies, including historical proposals
and rejected interpretations. Their presence does not make them current rules.
Distinct fork suffixes are linked from the corresponding source/branch records.

| Reading | Phase |
| --- | --- |
| 00 | [Initial model-strategy-Trainer direction, 21-22 July 2026](sessions/00-initial-direction.md) |
| 01 | [Inventory and explicit strategy authoring, 22-29 July 2026](sessions/01-inventory-and-authoring.md) |
| 02 | [Framework comparison and contract surfaces, 29 July 2026](sessions/02-frameworks-and-contract-surfaces.md) |
| 03 | [Code pressure, strategy meaning and mode/objective classification, 29-31 July 2026](sessions/03-training-mode-and-objective.md) |
| 04 | [Logical identity and execution routes: Q1-Q2, 31 July-20 August 2026](sessions/04-identity-and-execution-routes.md) |
| 05 | [Arrangement changes and artifact persistence: Q3-Q4, 20-21 August 2026](sessions/05-arrangement-and-persistence.md) |
| 06 | [Canonical authority, reviews, corrections and handoff: Q5, 21-22 August 2026](sessions/06-authority-reviews-and-handoff.md) |
| 07 | [First exchange design and contract-authority correction, 22-23 August 2026](sessions/07-contract-exchanges-and-authority-correction.md) |
| 08 | [Design frontier, preparation publication and durable identity, 24-25 August 2026](sessions/08-design-frontier-preparation-and-identity.md) |
| 09 | [Executable binding evidence and optimization groundwork, 25-26 August 2026](sessions/09-executable-evidence-and-optimization.md) |
| 10 | [Accepted-arrangement correction and governing OpenSpec transition, 26 August 2026](sessions/10-accepted-arrangement-and-openspec-transition.md) |
| 11 | [First OpenSpec draft and the direct-source correction, 18 September 2026](sessions/11-first-openspec-draft-and-reconciliation.md) |
| 12 | [Governing rebuild and one training contract system, 19-20 September 2026](sessions/12-governing-rebuild-and-contract-system.md) |
| 13 | [Dynamic behavior, PEFT and the initial spec reconciliation, 20-21 September 2026](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md) |
| 14 | [Training-first placement and one coherent run, 21-22 September 2026](sessions/14-training-first-placement-and-one-run.md) |
| 15 | [Research pressures and the whole-run language direction, September 2026](sessions/15-research-and-whole-run-language.md) |
| 16 | [Exchange design, independent audits and the G5 correction, September-October 2026](sessions/16-exchanges-audits-and-g5-correction.md) |
| 17 | [Connected execution candidate and IR machinery investigation, October 2026](sessions/17-connected-candidate-and-ir-machinery.md) |
| 18 | [xDSL and Pliron comparison and working-substrate choice, October-November 2026](sessions/18-framework-comparison-and-xdsl-choice.md) |
| 19 | [Shared language, retained products and continuity into G5, November 2026](sessions/19-shared-language-and-current-frontier.md) |

The later capture descriptions retain each reading's context. References to
what was “next” describe the sequence at that checkpoint; the full frontier
and remaining limits are recorded above and in the closure documents.

## How to read the evidence

- **User instruction, assistant proposal, imported opinion, and accepted
  requirement are distinct.** A pasted review is not automatically the user's
  decision. Quoted assistant text inside a user annotation is not the user's
  own architectural position. General approval of a document does not turn its
  explicitly unresolved leanings into settled decisions.
- **Read historical revisions in order.** A current file can incorporate a
  much later correction; its first Git commit does not date every paragraph.
  Supersession should identify the affected claim and the replacing source.
- **Keep original rationale.** A later replacement can preserve a problem,
  invariant, or constraint while changing the proposed realization. Replacing
  the latter does not discard the former.
- **Separate document role from physical placement.** Classification is the
  first step. The subsequent authorized archival follows those roles, preserves
  the original bytes and records each move. Physical archival does not alter
  the governing carry-forward rule.
- **The existing authority rule still applies.** See the governing design's
  [source-authority table](../../../openspec/changes/rework-model-strategy-trainer/design.md#source-authority-and-carry-forward-rule).
  This audit cannot silently reopen a settled requirement or promote a
  research convention into one.

## Dates and session provenance

[inventory.json](inventory.json) captures hashes, Git commit history including
renames, author/committer dates, and filesystem birth/mtime/ctime for the 68
Markdown files selected from the original folder and current change, excluding
`evidence*` directories. The placement manifest now classifies every selected document and original
root item; this does not claim a new line-by-line source-code or evidence audit.

Git dates establish when content entered the recorded history. Filesystem
birth dates establish when the present inode was created; copying or checkout
can make them later than the writing. `mtime` dates content modification;
`ctime` dates inode metadata change and is not creation time. A date written in
a document is another claim to correlate, not something to substitute for
these observations. Offsets are preserved in the JSON; the chronology table
uses UTC.

Codex forks can replay inherited messages with new record timestamps. The
initial export follows one selected fork's message order and resolves stable
message or turn identities through its four-session ancestry. It records both
the selected record and earliest matching record found in that ancestry.
Repeated text in distinct turns is retained. The export does not claim that
its inspected ancestry exhausts every copy or alternate branch.

The second export extends that chain to eight sessions. Differences from its
selected path are retained in an appendix, deduplicated by stable identity and
content together. They are alternate branches, not later events to append to
the selected conversation. This preserves edited prompts, abandoned inventory
attempts, the context-recovery discussion, and the literal clarification that
the builder is whoever writes the strategy. Similar text in distinct turns is
not automatically a duplicate.

The readable transcript contains visible user and assistant messages, including
commentary and final answers. It omits reasoning, tool traffic, generated
context, and compaction summaries. Compaction boundaries are marked. Original
message bodies, historical links, typos, annotations, and pasted reviews are
preserved; old absolute links can consequently point to moved files. Non-text
attachments need their original source and are not reconstructed from prose.

[sessions/index.json](sessions/index.json) lists 27 registered primary-thread
candidates from 16 July onward and the inspected ancestry. Those candidates
have not all been read or established as relevant. The index is not an
exhaustive search of unregistered sessions, alternate forks, or attachments.

The [machine-readable first export](sessions/00-initial-direction.json) retains
identifiers, source lines, body/content hashes, raw-file hashes, excluded-record
counts, and compaction locations. The raw session files remain outside the repository and are not edited by the
capture. An active session continues to grow; readings 18–19 hash only their
fixed historical prefixes, before this audit began.

The [second export](sessions/01-inventory-and-authoring.json),
[branch-difference export](sessions/01-branch-differences.json), and
[snapshot source record](sessions/01-snapshot-sources.json) retain the same
provenance for that reading. The
[third export](sessions/02-frameworks-and-contract-surfaces.json) and its
[sources](sessions/02-source-records.json) also record attachment provenance
and the separate dates at which the new direction sections entered Git.
The first committed framework comparison is a source-level revision; the
earlier documentation-only draft is represented by its conversation trail,
not falsely labelled as that committed revision.

[Reading 03](sessions/03-training-mode-and-objective.json) and its
[sources](sessions/03-source-records.json) preserve the direction/notes
consolidation at `78df512d` and the inventory/notes classification at
`707015ee`. The latter explicitly supersedes the first inventory's view of
mode as a permanent axis. Its August 1 commit date is separate from the
July 30 discussion and editing.

[Reading 04](sessions/04-identity-and-execution-routes.json) and its
[sources](sessions/04-source-records.json) preserve the first committed
Q1-Q2 direction, inventory and notes checkpoint at `144faf03`. That same
commit also contains the subsequent provisional Q3 adapter discussion. The
copy is labelled accordingly; its later material is not silently attributed
to the earlier Q1-Q2 exchange.

[Reading 05](sessions/05-arrangement-and-persistence.json) and its
[sources](sessions/05-source-records.json) follow that provisional Q3 into
explicit settlement, then Q4's persistence checkpoint. The exact `e0613c2f`
snapshots first commit both settled questions. The supplied starting take and
external notices remain separate from the corrections and checkpoint decisions.

[Reading 06](sessions/06-authority-reviews-and-handoff.json) and its
[sources](sessions/06-source-records.json) distinguish the open scenario
checkpoint `c3b59068`, Q5 settlement at `f603d074`, and the corrected wording
at `d7c4de7b`. The underlying Claude critique and visible GLM report are retained,
so their actual findings can be compared with the assistant's initial summary
and later correction. The normalization attempt was rejected and reversed;
its proposed deletions are not adopted as this audit's organization policy.

[Reading 07](sessions/07-contract-exchanges-and-authority-correction.json) and
its [sources](sessions/07-source-records.json) preserve the first exchange
draft at `3602960a`, the expanded but still provisional validator proposal at
`acdc2b19`, and its correction at `7bd8a124`. The last commit records the
previous day's correction; its date does not date every idea it contains.

[Reading 08](sessions/08-design-frontier-preparation-and-identity.json) and its
[sources](sessions/08-source-records.json) preserve the `d44b2ccb` checkpoint
and first implementation mapping. A later-disposition pointer identifies
`6db3f1ce`: the October 4 correction forbidding incomplete same-run restoration
from silently becoming a new run. That stronger rule is kept separate from
the earlier August formulation.

[Reading 09](sessions/09-executable-evidence-and-optimization.json) and its
[sources](sessions/09-source-records.json) retain the original final spike
review, three supplied feedback documents, exact code-blob witnesses and the
first committed findings. The recorded 28-scenario result is historical
fake-object evidence, not a new test run or production-language qualification.

[Reading 10](sessions/10-accepted-arrangement-and-openspec-transition.json) and
its [sources](sessions/10-source-records.json) preserve the corrected
`bb763b59` checkpoint, supplied region/authority feedback, final transition
review and two discussion companions. Seven complete visible-message bodies
match the copied `strategy_discussion.md` after line-ending normalization for
comparison; its snapshot remains byte-exact. This establishes its August
context without calling it the original July discussion.

Capture receipts describe the files as observed at their respective capture
times; an earlier receipt is not a promise that subsequently extended audit
notes still have their earlier hashes. New readings preserve earlier transcript
and snapshot bytes. Review claims inside a historical transcript retain their
original scope and do not count as a current review or runtime qualification.

## Reading 11: first governing draft and source reconciliation

[First OpenSpec draft and direct-source correction](sessions/11-first-openspec-draft-and-reconciliation.md)
preserves 19 messages: five from the separate orientation misstart and fourteen
from the original root's September resumption. The draft is written in the
latter, then challenged for its partial reread and summary reliance.
The [source record](sessions/11-source-records.json) captures all nine entries
of the earlier ZIP, with eight exact matches to original authoring patches.
The manifest is generator output, not a recovered authored patch.

The archive's own design and tasks gate production. Its missing carry-forward
and cross-spec reconciliation cannot be equated with a claim that production
was ready. The rebuild is traced next, rather than treating the first draft as
the current governing baseline.

## Reading 12: governing rebuild and the shared contract model

[Governing rebuild and one training contract system](sessions/12-governing-rebuild-and-contract-system.md)
adds 87 messages from the original September 19 fork through the first
committed governing checkpoint. It preserves three original review reports,
the supplied opinion and its corrections, the cross-spec fixes, and the user
walkthrough that establishes revisioned run obligations as the shared source
for fulfillment, realization and later authority.

The [source record](sessions/12-source-records.json) records all nineteen files
of the first governing package at `8ec7249d`, four updated historical input
records, and the two reciprocal metadata planning artifacts. That commit is
September 20 03:23:38 UTC, after the first two walkthrough items; it is not an
exact copy of the earlier draft before the reviews and refinements.

Next is the remainder of that initial walkthrough, beginning with participant
state and authority, binding/preparation, and the model-structure versus
recipe-treatment correction. Broad research and later machinery/language
experiments still follow in the historical sequence.

## Reading 13: the completed initial walkthrough

[Reading13](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md)
covers the authority/preparation review, authored dynamic behavior and
initialization, capability facts, shared maintained PEFT, product independence,
adapter target ownership and the remaining design gates. It ends at the
September21 07:00UTC cleanup, before that evening's implementation-map and
placement discussion. H049-H054 record the main refinements.

The [source record](sessions/13-source-records.json) preserves exact page
continuation cutoffs, three original reviews, twelve changed governing files
at `58ab91ce`, and the nineteen-file package hashes. The seven unchanged files
already match the preceding checkpoint snapshots. Six suffix messages outside
the chosen page boundaries remain in the appendix; inherited conversation is
not replayed. The later migration argument is previewed only to prevent an
early coexistence allowance from being read as a mandate to keep every edit
runnable.

## Reading 14: placement, training first and one coherent run

[Reading14](sessions/14-training-first-placement-and-one-run.md) follows the
rejection of runnable-intermediate constraints and speculative lifecycle
layers through the provisional strategy/training package map. It records
training process versus Trainer, one readable/queryable contract, reusable
feature implementations, initial run creation beside state, and one accepted
run whose internal sections retain narrower owners. H055-H060 carry the
adoption, limits and document classification.

The [provisional tree](snapshots/722da952-provisional_tree.md) is now traced:
it copies the combined package sketch and the later training sweep, with two
whole-message correspondences. Its naming and a compressed targeting sentence
remain historical, not a final production tree. The original is untouched.
The [source record](sessions/14-source-records.json) preserves eleven exact
page-prefix relationships, including the next page and the short side fork.
The next reading begins the programming-model/language questions and broader
research that motivate the later whole-workflow machinery.

## Reading 15: Research pressures and the whole-run language direction, September 2026

[Research pressures and the whole-run language direction, September 2026](sessions/15-research-and-whole-run-language.md) preserves 419
selected visible messages across 11 source captures, with 23 distinct
branch messages. The [source record](sessions/15-source-records.json)
retains exact windows, hashes, source roles and Git checkpoints.
Broader research, ordinary and adversarial training, staged topology, independent producers, stateful inputs and the first action/whole-run language experiments.

## Reading 16: Exchange design, independent audits and the G5 correction, September-October 2026

[Exchange design, independent audits and the G5 correction, September-October 2026](sessions/16-exchanges-audits-and-g5-correction.md) preserves 631
selected visible messages across 11 source captures, with 18 distinct
branch messages. The [source record](sessions/16-source-records.json)
retains exact windows, hashes, source roles and Git checkpoints.
G3/g4 exchange completion, early executable research, restoration correction, pinned multi-pass audit and the rejection of following g5 component checkboxes without a connected execution mechanism.

## Reading 17: Connected execution candidate and IR machinery investigation, October 2026

[Connected execution candidate and IR machinery investigation, October 2026](sessions/17-connected-candidate-and-ir-machinery.md) preserves 897
selected visible messages across 11 source captures, with 20 distinct
branch messages. The [source record](sessions/17-source-records.json)
retains exact windows, hashes, source roles and Git checkpoints.
Connected whole-system mocks, graph-derived execution, shared owners, state/gradient/physical lifetimes, independent oracles, free-threading and backend qualifications, togeari learning and the next ir machinery candidate.

## Reading 18: xDSL and Pliron comparison and working-substrate choice, October-November 2026

[xDSL and Pliron comparison and working-substrate choice, October-November 2026](sessions/18-framework-comparison-and-xdsl-choice.md) preserves 403
selected visible messages across 1 source captures, with 0 distinct
branch messages. The [source record](sessions/18-source-records.json)
retains exact windows, hashes, source roles and Git checkpoints.
Framework screening, xdsl and pinned pliron experiments, independent extensions, matched six-case comparison, machinery tradeoffs and explicit xdsl working selection.

## Reading 19: Shared language, retained products and continuity into G5, November 2026

[Shared language, retained products and continuity into G5, November 2026](sessions/19-shared-language-and-current-frontier.md) preserves 191
selected visible messages across 1 source captures, with 0 distinct
branch messages. The [source record](sessions/19-source-records.json)
retains exact windows, hashes, source roles and Git checkpoints.
Language proposal and explicit amendments, shared semantic slice/reference, verification, lowering, cold/independent qualification, binding effects/filtering/fusion and retained storage/autograd products; ends immediately before this historical audit.
