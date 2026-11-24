# Document map and observed chronology

This walkthrough follows the central decision lineage from the July direction
through the November shared-language and retained-product experiments, ending
before this audit began. It distinguishes the first governing draft, rebuild,
later corrections, connected candidate, matched frameworks and accepted xDSL
working choice. The [current carry-forward map](current-decisions.md) is the
short entry point; the notepad retains rationale and current disposition.

Git revisions, source messages and inode metadata serve different purposes.
Historical code/review/test/benchmark claims are not newly rerun. The earlier
documents are now [archived](../../archive/models-strategy-trainer/README.md)
without changing their bytes. Prior immutable captures retain original paths;
[archive-relocations.json](archive-relocations.json) resolves moved files.
Live source references follow the move. The
[per-file classification](placement-manifest.json) and
[placement plan](placement-plan.md) organize records without moving originals.

## What the records are and where they belong

The authority descriptions below follow the governing design's existing
[source table](../../../openspec/changes/rework-model-strategy-trainer/design.md#source-authority-and-carry-forward-rule).
The table preserves the initial classification. The completed archival and
current homes are recorded in the [placement plan](placement-plan.md) and
relocation map; earlier proposed homes below are not outstanding tasks.

| Existing record or group | Role and current disposition | Placement assessment for the walkthrough |
| --- | --- | --- |
| `strategy_system_direction.md` | Consolidated pre-OpenSpec direction, including later corrections. Normative input inherited by the governing change. Its present text is not the July original. | Keep its referenced location during the audit. Any later historical-source home must preserve its authority role and incoming links. Read its Git revisions, not only the current text. |
| `strategy_contract_exchange_design.md` | Detailed binding, preparation, identity, optimization and execution derivation. `SETTLED INPUT` is inherited; working EX entries are adopted only where the governing change says so. | Preserve as a derivation record. Eventual relocation must retain EX identifiers, status distinctions and links; it must not become a competing living spec. |
| `notes.md` | Chronological checkpoints, including positions later superseded. | A chronology/history home is appropriate in principle. Preserve the dated original and make its correction chain navigable. |
| `strategy_discussion.md` | Retrospective copied August 26 topology discussion, first committed in `bb763b59`. Seven complete message bodies correspond to the primary session. It is not the original July conversation or an independent requirement source. | A historical discussions home with an episode/source wrapper is appropriate in principle. Preserve the correspondence and original text rather than relabelling it as the original July session. |
| `strategy_discussion_external.md` | Separate external opinions, first committed in `bb763b59`. The discussion moves from a prepared-runtime proposal through a fixed-component narrowing to one extensible engine and normative contract. Its opening recommendation is not its final position. | Keep as an explicitly external historical discussion, alongside its later evaluation and adopted refinements. Do not merge it into the primary transcript or treat all proposals as accepted. |
| `strategy_system_inventory.md` | Production-code and pressure-scenario evidence from the investigation. Its first committed revision is now preserved separately; alternate branches report earlier inventory drafts. Its recorded baseline does not automatically describe current production. | Historical baseline/evidence home is plausible. Retain the baseline revision and distinguish code observations, proposed classifications and target requirements. An old placement verdict is not automatically a current migration decision. |
| `strategy_system_implementation_mapping.md` | Source-backed current-to-target responsibilities and dependency order. | Keep as derivation/migration evidence; reconcile its claims against the governing change before treating a package shape as selected. |
| `framework_pattern_comparison.md` | Subordinate prior art. Reading 02 distinguishes its documentation-only first pass from its first committed source-level revision. Its current header explicitly supersedes the old active-strategy topology while preserving the boundary evidence. | Historical research/comparison home is plausible. Preserve revision dates and which lessons were adopted, rejected, or left unused; do not treat its original ownership matrix as the current target. |
| `provisional_tree.md` | Historical copy of two September22 replies: the combined provisional package tree and the training-folder sweep. Broad responsibilities receive provisional user assent; filenames, internal splits and a compressed targeting sentence are not final authority. | Keep as a historical working sketch linked to its exact source messages and governing dispositions. It is not the intended future module README or a production file-creation checklist. |
| `candidate_comparison.md`, `matched_comparison.md`, `grouping_comparison.md`, `machinery_comparison.md` and companion JSON | Sequential October-November comparison evidence, covered in reading 18. Matched frontends share owners; working xDSL selection occurs after the comparison and does not turn timings into a general ranking. | Keep linked to their fixtures and evidence. A comparison index can organize them without changing artifact paths. |
| `executable_spike/`, `execution_candidate/`, `xdsl_candidate/`, `pliron_candidate/`, `xdsl_language/` | Isolated implementation and qualification evidence. Their roles evolved and must be read by revision. They are not automatically production packages or public APIs. | Preserve experiment/evidence structure while auditing it. Moving runnable files can change imports, commands and recorded paths, so these need more than a document move. |
| Root `*_target.py`, comparison fixtures, `evidence_*` directories | Implementation, fixtures and retained observations used by experiments. Not yet a content audit. | Follow the owning experiment and evidence links before considering consolidation. |
| `inventory_01_02.7z` | Exactly two earlier inventory drafts, inspected and copied byte-for-byte in supplementary sources. Entry dates are metadata, not authorship or archive-creation dates. | Keep original as evidence; future history/archives placement is suitable. Readable copies and hashes are already available in this history workspace. |
| Active `openspec/changes/rework-model-strategy-trainer/{proposal,design,tasks}.md` and `specs/` | Current governing change. Planning-artifact presence is distinct from completed G5 or production migration. | Keep in OpenSpec. Record historical evolution and carry-forward links here; do not create a second specification in the audit. |
| `docs_design/archive/models-strategy-trainer/packages/mst_pre_astra.zip` | Superseded first September draft, preserved exactly and matched to its authoring patches. It gates production but lacks full carry-forward and spec-conflict coverage. | Evidence archive, not an active change or synced main spec. Keep the original during the audit; extracted copies already live under history snapshots. A later move should preserve hashes, source records and incoming links. |
| Active change `research/` | Research inputs with adoption controlled by design/deltas. Supplied documents and illustrative APIs are not all accepted interfaces. | Keep the authority distinction visible; readings 15–19 trace the cases, mechanism sketches, machinery report and adopted language amendments. |

The original audit classified documents before moving them. Subsequent archival
preserves the historical source bytes; this `history/` directory keeps the
audit, readable captures and current navigation.

## Recorded starting points

Dates below are Git **author dates in UTC**, not claimed dates of idea origin.
Author and committer dates, full hashes, all observed file history and filesystem
timestamps are in [inventory.json](inventory.json).

| Date in Git | Commit | Event established by the file history |
| --- | --- | --- |
| 22 Jul 2026, 02:32 | `de2dd0eb` | First recorded `docs_design/strategy_system_direction.md`, 785 lines. The initial conversation export covers 21 Jul, 18:03 through 22 Jul, 01:40 UTC. |
| 26 Jul, session records | No matching committed draft established | An alternate branch reports using the new folder and producing a 729-line inventory. These are session statements, not a recovered Git snapshot of that draft. See the branch-difference export. |
| 29 Jul, 02:27 | `16801f38` | Direction moved into `docs_design/models-strategy-trainer/` with changes (`R098`); inventory and notes added. The folder's history begins later than its direction document's history. |
| 29 Jul, 22:11 | `97a81ef4` | Framework comparison added; direction revised. This is also the first committed direction revision containing the explicit-authoring and research-conformance clarifications already discussed earlier that morning. |
| 29 Jul, 21:57-22:17 in conversation | Source messages, then `78df512d` | User states the three contract categories; attached feedback sharpens their consumer/lifecycle distinctions; the direction is updated. Those new sections first enter Git on 30 July, not in the comparison commit just before the update. |
| 30 Jul, 21:15 | `78df512d` | Direction/notes revised after the framework discussion. Read the diff before extracting what was settled. |
| 30 Jul, 21:31 in conversation; 1 Aug, 03:17 in Git | `707015ee` | Mode/objective classification adds inventory Milestone 6 and explicitly supersedes the earlier "correct axis" treatment of `TrainingMode`. Notes replace the three-blocker handoff with two remaining semantic questions. |
| 20-21 Aug | `144faf03`, `e0613c2f`, `c3b59068`, `f603d074` | Further direction revisions, with commit subjects referring to adapters, artifacts and Q5. Their dated checkpoints and session turns must establish the actual decisions. |
| 20 Aug, 15:00 in conversation; 17:17 in Git | `144faf03` | Q1-Q2 checkpoint introduces settled binding semantics and inventory Milestone 7. The same commit also includes the later provisional Q3 adapter checkpoint; those are separate parts of the discussion. |
| 20 Aug, 17:12 provisional and 20:47 settled in conversation; 21 Aug, 02:47 in Git | `144faf03`, then `e0613c2f` | Q3 pressure-tests adapters beyond PEFT. External feedback and user-supplied corrections establish participant/relationship identity and distinct state transitions. The first commit is provisional; the second records settlement together with Q4. |
| 21 Aug, 02:28 evaluation and 02:45 checkpoint in conversation; 02:47 in Git | `e0613c2f` | Q4 establishes semantic product/member/resource separation, coherent state capture, declaration/request/plan/result stages and distinct restoration semantics. The evaluation qualifies imported notices, including EDM2's active path. |
| 21 Aug, 19:59 scenario checkpoint; 21:21 in Git | `c3b59068` | Six target-first Q5 scenarios constrain authority semantics but leave its physical layout open. Current production anatomy remains evidence, not the target. |
| 21 Aug, 21:27 settlement; 23:47 in Git | `f603d074` | Q5 selects one dedicated, governed per-run authority, then housed inside an active strategy. The commit still contains the overly categorical rejection of conservative implementation policies. That enclosure is later superseded; the one-authority requirement survives. |
| 21 Aug, 23:46 correction; 22 Aug, 00:18 in Git | `d7c4de7b` | Rereading original reviews corrects conservative-policy wording. An intervening large normalization is rejected and reported restored; user edits and a small final correction remain. The final diff is small, not the proposed 600-line reduction. |
| 22 Aug, 20:19 | `3602960a` | Exchange-design record first added. |
| 23 Aug, 04:35 discussion and 05:04 in Git | `acdc2b19` | Candidate-validation exchange still assumes participant-specific semantic filings and proposal assessments. Those are working decisions, later corrected, not adopted production interfaces. |
| 23 Aug, 19:21 correction; 24 Aug, 01:51 in Git | `7bd8a124` | Contract system judges the complete authored arrangement and derives ordinary obligations; participant-authored validators cease to be the required route. Q1-Q5 and valid binding/preparation mechanics survive. |
| 25 Aug, 04:06 | `d44b2ccb` | Implementation mapping first recorded. |
| 24-25 Aug in conversation; 25 Aug, 04:06 in Git | `d44b2ccb` | Fourteen open entries are audited; code-shape choices leave the architecture register. Preparation becomes an attempt-scoped coherent job with fallibility before final publication. Durable identity, retirement and metadata-origin wording are settled and corrected. |
| 25 Aug, 17:02 spike handoff; 19:27 in Git | `b45f1d4e` | First isolated binding/preparation spike: 28 recorded scenarios after review-driven corrections. Its two Python files are new evidence, with no production integration. |
| 25-26 Aug discussion; 26 Aug, 22:20 in Git | `bb763b59` | First committed optimization groundwork includes semantic unit identity/revision, separate realization/progress, standard non-overlap and backend-flexible staging. The same commit includes the following topology and ownership corrections; it is not a frozen copy of every intermediate proposal. |
| 26 Aug, 22:20 | `bb763b59` | Pre-OpenSpec checkpoint; both `strategy_discussion*.md` files added. `notes.md` dates the accepted-arrangement correction to 26 Aug. |
| 26 Aug, 18:47 topology handoff and 22:20 transition handoff | `bb763b59` | Corrected arrangement replaces active-strategy runtime enclosure while retaining Q1-Q5. The day ends with one governing OpenSpec as the next design location, distinct from production readiness. Its note uses the local August 27 date. |
| 18 Sep, 00:03-00:04 | Separate orientation root | The registered misstart reads the later checkpoint and makes no edits. It does not create the draft. |
| 18 Sep, 00:10-00:34 | Original August root, raw6093-6323 | First five-delta OpenSpec draft; user challenges partial direct reading and summary reliance. The exact draft survives in `mst_pre_astra.zip`; ZIP creation/removal actor and time remain unestablished. |
| 19 Sep, 19:03-19:47 | Original September19 fork | Rebuild from recorded sources, initial credential failure, integration review and follow-up; package expands to seven new/eight modified capabilities. |
| 20 Sep, 01:09-03:06 | Same fork, walkthrough items1-2 | Runtime semantics govern metadata overlap without blanket landing order. User challenges Trainer-owned contract and separate validators; corrected shared fulfillment/realization/authority model is approved and reviewed. |
| 20 Sep, 03:23 | `8ec7249d` | First committed governing package, after those corrections; proposal/design/tasks, fifteen deltas and reciprocal metadata gate. Not the rejected five-delta draft. |
| 20 Sep, 19:42-19:59 | Explicit paginated continuations | Authored behavior remains dynamic after the authoring role ends; initialization sources/rules are made explicit. Older model-structure and redundant-confirmation suffixes remain separate branch evidence. |
| 20 Sep, 20:27 through 21 Sep, 07:00 | Same selected continuation | Capability facts, common PEFT integration, artifact independence and adapter-target ownership are reconciled across specs. Remaining implementation-shape questions are separated from design gates. |
| 21 Sep, 07:02 | `58ab91ce` | First committed initial walkthrough/double-check: twelve changed governing files; seven package files unchanged. Later evening migration/placement proposals are not in this checkpoint. |
| 21 Sep, 22:02 through 22 Sep, 05:49 | Paginated implementation/placement discussion | User rejects renewed premise checking, runnable-intermediate constraints and lifecycle-word packages; provisional contract/features/strategy and state/phases/capabilities/Trainer map is accepted with names open. |
| 22 Sep, 06:08-23:10 | Training-first sweep and state clarification | User authorizes a targeted training sweep, accepts input/observation state with separate owners, and asks how one coherent run survives structured state. The last turn calls for a real programming model/language. |
| 23 Sep, 07:01 | `722da952` | First committed `provisional_tree.md` copies the combined sketch and sweep; governing design records training-first disposition and narrower participant authority, together with later September23 research. |
| 27 Sep, 16:30 | `2eed40b6` | Research corpus and evolving mechanism sketch recorded; ordinary and adversarial cases drive whole-run semantics, not a universal SD training sequence. |
| 1 Oct, 04:00 | `e0a6f9bb` | G3 exchange/conformance and initial cost evidence recorded. Thresholds are later clarified as migration gates, not permanent semantic constants. |
| 3 Oct, 00:08 | `00d9bb63` | G3 findings reconciled into the execution/optimization/contract specs. |
| 4 Oct, 04:39 | `6db3f1ce` | G4 caching, evaluation, persistence, restoration and observation exchanges recorded; completed design gates do not mean production implementation. |
| 5 Oct, 03:41 | `b19717fa` | Pinned audit closeout records corrections and dispositions. The subsequent G5 discussion requires an assembled concrete mechanism. |
| 9 Oct, 04:29 | `5218fdb1` | First executable connected mock recorded. Later pressure cases refine shared owners and specialized lowerings. |
| 28 Oct, 14:29 | `2977aad4` | Original execution-candidate checkpoint with independently checked associations/RNG and repaired dispatch submission; not production adoption. |
| 29 Oct, 13:09 | `2131f31d` | First xDSL B candidate recorded; A, C, mixed and extension follow on the same owner boundary. |
| 30 Oct, 16:08 | `5a737c27` | Explicitly pinned Pliron 0.20.0 B experiment recorded; independent review later finds parser suffix acceptance and omitted requirements file. |
| 30 Oct, 19:43 | `98fbbb23` | Pliron extension/resource analysis recorded, with parser/dependency repairs; its accepted scope remains bounded. |
| 31 Oct, 13:16 | `2f4c9353` | Different-lifetime composition recorded: repeated training, requested evaluation, consumer settlement and separate producer shutdown. |
| 30 Oct, 11:18 | `99bb13e1` | `candidate_comparison.md` first recorded. |
| 1 Nov, 16:19 | `7386bba3` | `matched_comparison.md` first recorded. |
| 1 Nov, 20:15 | `0055062d` | `grouping_comparison.md` added. |
| 2 Nov, 12:04 | `fe46babf` | `machinery_comparison.md` added. Reading 18 traces the subsequent explicit working xDSL selection, recorded in governing D14. |
| 3-5 Nov | `aa05e859` through `f984c41a` | Language input and shared-language milestones appear in governing-change history. Reading 19 reaches this current pre-audit frontier; experimental completion feeds G5 without selecting production APIs. |

## Timestamp observations that affect interpretation

- The direction document's present inode was created on **19 September**, but
  its followed Git history starts on **22 July**. The creation date does not
  establish the writing date.
- `strategy_discussion.md` has an August filesystem date and was committed on
  **26 August UTC / 27 August local time**. Its retrospective account must not
  be placed at the beginning merely because it is called a discussion.
- The notes distinguish a **27 August transition decision**, a **19 September
  OpenSpec rebuild**, and a **20 September local-time commit**. These describe
  different events, not a single interchangeable date.
- The active OpenSpec files have November inode dates while Git records them
  in September. The inode dates alone cannot establish when the change began.
- A model-system question recorded natively at **21 July, 18:03 UTC** was
  replayed into forks at 22:17 and 22:38. The transcript links the parent
  occurrence and retains the selected-fork timestamp separately.
- The selected inventory passage starts with the **22 July, 20:12** follow-up,
  although it appears in a session file created on 29 July. The eight-session
  parent chain establishes that this is inherited conversation, not a new
  discussion beginning with that filename's date.
- The first committed inventory on **29 July, 02:27** has **1,089 lines**. A
  26 July branch reports a **729-line** draft, and another inventory attempt
  runs late on 28 July. Those are distinct states; one branch's review claim
  cannot certify a later inventory revision. The selected 29 July handoff
  explicitly says the review tool was unavailable.
- User corrections about intentional authoring appear at **29 July, 02:01** in
  a parent branch and **02:10** in the selected child. The literal "builder is
  whoever writes the code" clarification is in the parent branch at **02:05**.
  Stable identities retain both versions without pretending they are the same
  message or discarding the branch-only clarification.
- `16801f38` adds only the inventory cross-reference to the earlier direction
  document. The explicit-authoring and "Noob Mode" sections enter its Git
  history with `97a81ef4` that evening. The session records the decisions and
  editing earlier; the first commit containing them is a different timestamp.
- `97a81ef4` is timestamped **29 July, 22:11:15 UTC**. The request to record the
  latest three-surface/lifecycle discussion follows at **22:11:28**; its
  completion message is at **22:17:54**. Git pickaxe checks place the first
  introduction of all three section headings in `78df512d`, **30 July,
  21:15:39**. That later revision also contains subsequent discussion, so it is
  not an exact frozen copy of the document at 22:17 on the 29th.
- The external feedback attachment's present inode dates to **29 July,
  22:03:21 UTC** and the user references it at **22:03:38**. The attachment date
  is captured separately; its filename in this audit identifies the episode,
  not proof of when its contents were originally composed.
- The consolidation reply at **30 July, 01:58 UTC** records the strategy/mode
  corrections. `78df512d` commits direction and notes at **21:15**, before
  the user requests responsibility classification at **21:23**. Its notes
  still identify classification as work to do. It is not the completed
  classification merely because both events occur on July 30.
- The classification handoff is at **30 July, 21:31 UTC**. Its first recorded
  inventory milestone is committed as `707015ee` on **1 August, 03:17 UTC**.
  That commit changes the notes from three blockers to two. These establish
  the adoption/handoff trail, not acceptance of every early API sketch.

- The Q1 answer is dated **31 July, 03:17 UTC**. The user resumes on
  **20 August, 14:48**, explicitly carrying that answer forward and asking for
  Q2 next. The transcript contains no question-numbering correction here.
- The Q2 checkpoint completes at **20 August, 15:00 UTC**. `144faf03` is
  timestamped **19:17:47 +02:00 / 17:17:47 UTC**, after the next Q3 pressure
  test and its provisional checkpoint. Its direction, inventory and notes
  therefore establish the first committed Q1-Q2 record without being an exact
  snapshot of the files at the end of the Q2 turn.
- The first `Settled Binding Semantics` section calls its answers provisional
  architecture strong enough to constrain exchange design. Inventory Milestone
  7 calls Q1-Q2 semantically settled. Preserve those original status words;
  neither freezes a Python representation nor reverses the current governing
  adoption of the subsequently refined meanings.

## Reading 01: inventory and explicit authoring

The [selected continuation](sessions/01-inventory-and-authoring.md) contains
35 visible messages from 22 July, 20:12 through 29 July, 02:54 UTC. The
[branch map](sessions/branch-map.md) identifies eight linked sessions; the
[difference appendix](sessions/01-branch-differences.md) retains 36 messages
absent from the selected path. Those include context recovery, revised
inventory instructions and an explicit author-versus-builder correction.

The [1,089-line inventory](snapshots/16801f38-strategy_system_inventory.md) is
an exact Git snapshot. Its evidence standard, milestone conclusions, state and
placement classifications, conformance scenarios and open questions are
historical inputs. For example, it calls `TrainingMode` a correct axis; the
later direction and governing change deliberately dissolve that runtime axis.
That is a proposed architectural classification requiring a later disposition,
not an immutable code fact.

H007-H011 in the [notepad](decision-notepad.md) distinguish intentional
authorship, research conformance, consumer-defined contract meaning, proposed
initial limits, and the inventory's evidential role. The active callback
façade repeatedly proposed alongside them remains separately tracked by H004.
That reading ends immediately before "wait are we just redoing
lightning/fabric." Reading 02 follows that comparison without substituting it
for the earlier source.

## Reading 02: framework evidence and the contract's distinct surfaces

The [48-message continuation](sessions/02-frameworks-and-contract-surfaces.md)
runs from 29 July, 03:03 through 22:57 UTC in the same selected session. None
of those messages appear in the seven inspected earlier ancestors. The
[external feedback](snapshots/2026-07-29-contract-surfaces-feedback.md) referenced
in the conversation is preserved byte-for-byte with its own source record.

The user's explicit three-category clarification at 21:57 precedes the
feedback. The feedback adds precision about configuration requests, lifecycle
as an independent axis, delegated capability coordination and runtime-state
contribution versus artifact persistence. The user requests a direction update
and later says they have read it. The first Git revision containing those new
sections is later than that conversation, as recorded above.

The [first committed comparison](snapshots/97a81ef4-framework_pattern_comparison.md)
is 897 lines and records the historical Lightning 2.6.5 / Accelerate 1.11.0
source audit. The [direction at that commit](snapshots/97a81ef4-strategy_system_direction.md)
is 990 lines. The transcript also retains the earlier claim that a comparison
was complete, the user's demand for implementation evidence, and the
assistant's downgrade/replacement of its documentation-only conclusions.
No independent current qualification of those framework versions was run here.

The current comparison's August-26 correction, committed in `bb763b59`,
preserves logical/prepared identity, preparation, authority and persistence
lessons while superseding active-strategy topology. Reading 10 reconstructs the
actual August correction; its original earlier comparison stays unchanged.

H012-H016 in the [notepad](decision-notepad.md) record those lessons and their
current/G5 consequences. H004, H007, H009 and H010 gain further source traces.
That reading ends before selected raw line 13059, "okay go check the code and
tell me what you think." Reading 03 follows that pressure test. Earlier draft
recovery and old source-code references remain separate evidence questions,
not reasons to reset established meanings.

## Reading 03: removing mode as an authority, without losing its behavior

The [38-message continuation](sessions/03-training-mode-and-objective.md)
covers 29 July, 23:07 through 31 July, 00:01 UTC, in the same selected session.
It includes the code-pressure findings, opposing mode-placement proposals,
the user's explicit definition of strategy, rejection of adapter-specific
strategies, responsibility classification, and the controlled-service
alternative to a whole-Trainer back-reference.

Its [sources](sessions/03-source-records.json) preserve four exact Git states:
the [1,524-line direction](snapshots/78df512d-strategy_system_direction.md) and
[182-line notes](snapshots/78df512d-notes.md) before classification, then the
[1,293-line inventory](snapshots/707015ee-strategy_system_inventory.md) and
[196-line notes](snapshots/707015ee-notes.md) after it. Inventory Milestone 6
expressly supersedes Milestone 4's description of mode as the correct axis.
That is an explicit disposition trail rather than an inference from today's
architecture.

H017-H020 distinguish dissolving mode authority, declaring subjects versus
realizing optimization, objective mathematics versus observations/accounting,
and scoped live services versus ambient Trainer access. H003, H004, H010,
H011 and H013 gain further source traces. Typed foundations and source
observations are historical evidence; current code lines and framework APIs
were not independently requalified here.

The chapter still contains active-strategy examples and a proposed per-sample
loss result. Those remain distinct from the surviving responsibility decisions
and their later accepted-arrangement form. Its "two blockers remain" checkpoint
describes that investigation's state, not present-day production readiness.

That reading ends before raw line 13895, "Need to walk through those
questions and elaborate a bit more. starting with the first one." Reading 04
follows the July 31 answer and August resumption; the recap carries Q1 forward
rather than correcting its numbering.

## Reading 04: identity, routes, views and freshness

The [12-message continuation](sessions/04-identity-and-execution-routes.md)
contains the July 31 Q1 answer, the user's August recap, the Q2 proposal,
supplied feedback and recorded resolution. It stays in the same selected
session. No additional parent-branch material matches this bounded passage.

The [source record](sessions/04-source-records.json) preserves the first
committed checkpoint: [1,584-line direction](snapshots/144faf03-strategy_system_direction.md),
[1,433-line inventory](snapshots/144faf03-strategy_system_inventory.md), and
[260-line notes](snapshots/144faf03-notes.md). The snapshot's later Q3 material
includes a first qualification for state-only participants, but that adapter
discussion is not represented as part of the earlier Q2 exchange.

H021-H023 record identity apart from implementation/source/role, additional
routes versus typed access views, and explicit freshness rather than arbitrary
module-reference selection. Current-disposition pointers distinguish authored
addresses from run incarnations, compatible from incompatible replacement,
participant declarations from individual uses, and state-only participants
from executable ones. Readings 05–10 trace their later derivation.
The original Q1-Q2 transcript remains unchanged by these qualifications.

Reading 05 continues at raw line 14034 with Q3, then Q4. These two questions
reach the same committed checkpoint and are read together below.

## Reading 05: arrangement transitions and semantic persistence

The [35-message continuation](sessions/05-arrangement-and-persistence.md)
covers raw lines 14034-14587, including the Q3 compaction-recovery check and
Q4's checkpoint. All matching messages occur in the selected continuation,
without additional inherited copies among the inspected ancestors.

The [source record](sessions/05-source-records.json) retains three supplied
documents: the Q3 starting take, external feedback on the expanded Q3 answer,
and Q4 notices. The Q3 starting take is a hypothesis; the second attachment is
explicitly external feedback. For Q4 the user explicitly warns that the notice
author has less context. Neither upload alone settles the architecture.

Q3 progresses from the [provisional inventory](snapshots/144faf03-strategy_system_inventory.md)
to the [settled direction](snapshots/e0613c2f-strategy_system_direction.md),
[inventory](snapshots/e0613c2f-strategy_system_inventory.md) and
[notes](snapshots/e0613c2f-notes.md). The latter three also include Q4; they
are not claimed as an exact file state at Q3's earlier conclusion. Q3 is
explicitly accepted after corrections to storage ownership, materialization
versus replacement, non-destructive merge and trainability versus lifecycle.

H024-H027 record those distinctions and Q4's semantic products, coherent
capture and truthful post-write results. The current-disposition pointers
preserve the later accepted-run authority rather than reviving an active
strategy object as its necessary holder. Historical code observations and
paper comparisons remain historical evidence; they were not freshly rerun.

Reading 06 continues at raw line 14588 and completes that selected session.

## Reading 06: canonical authority, corrected review interpretation and handoff

The [65-message continuation](sessions/06-authority-reviews-and-handoff.md)
covers raw lines 14588-15633. Q5 progresses from target-first scenarios to
ownership criteria: enforceable transitions, independent tests, state/behavior
separation, run lifetime, scoped access and migration. The user explicitly
warns against preserving unchanged production layout. The historical reviews
support the semantic direction but point out that more normal-path scenarios
cannot discriminate layouts that all satisfy them by construction.

The [source record](sessions/06-source-records.json) retains exact snapshots
of `c3b59068`, `f603d074` and `d7c4de7b`, plus the supplied outside assessment
and both historical review reports. The Claude saved file matches the
historical raw read; the GLM copy is an exact visible report extraction from
the original hcom transcript read. Failed launches and mistaken provider
attribution remain historical integration failures, not additional reviews.
No review was newly commissioned during this capture.

H028 distinguishes Q5's lasting authority requirement from its active-strategy
enclosure, explicitly superseded on August 26. H029 records the initial
overreaction to conservative policies and the correction accepted after
rereading the original reports. H030 records document roles alongside the
user's rejection of aggressive deduplication. The intermediate normalization
has no committed snapshot established here; its attempted reduction and
reported restoration are retained as conversation evidence. The small final
Git diff does not show the intermediate rewrite as an adopted organization.

The optimization-ownership blocker is restored explicitly; Q1-Q5 settlement
does not complete the architecture. The final suggestion of multiple
sequential OpenSpecs and the name `establish-bound-training-strategy-contract`
is a handoff proposal, not a settled partition or an implemented change.

The next primary session is `01a026d5-a88b-7843-a1c0-1b51047c8182`.
Its first substantive user message explicitly quotes this handoff and points
back to the old session. The new header has no fork parent, so continuity
is established by that message rather than fabricated ancestry. Its header
and first message are **22 August, 00:18:51 and 00:18:55 UTC**, despite the
rollout filename containing `02-18-51`. Use record/header timestamps rather
than interpreting the filename as UTC. The next phase follows actual binding
and preparation exchange design, optimization decisions and the correction to
accepted-arrangement topology.

## Reading 07: exchange drafting and correction of contract authority

The [88-message continuation](sessions/07-contract-exchanges-and-authority-correction.md)
covers the new root's raw lines 1-1163. Its first user message explicitly
continues the old thread. The initial proposal for multiple sequential
OpenSpecs is replaced by a recommendation for one coherent governing change
with multiple specs and reviewable milestones. This is still planning: no
OpenSpec is created during this passage. The durable exchange file begins in
response to the user's request to bind progress to an artifact before compaction.

The [source record](sessions/07-source-records.json) preserves three supplied
outside replies and exact exchange/notes snapshots at `3602960a`, `acdc2b19`
and `7bd8a124`. The first predates the declaration-feedback reconciliation.
The second includes the participant-validator proposal and subsequent naming
discussion. The third first commits the correction reported on August 23;
it is not claimed as an exact file snapshot at every earlier discussion turn.

The passage separates authored addresses, run incarnations, roles, revisions,
readiness and independent relationship identity. It then exposes a design
drift: requiring each participant to carry author-wired compatibility rules
makes ordinary authors responsible for a second validity system. The user
returns to the original contract direction. The correction supersedes that
proposal while retaining typed evidence, governed state transitions, freshness
and separate preparation-output owners. H031-H032 record its lasting meaning.

The remaining active-strategy enclosure is explicitly labelled historical;
the August 26 accepted-arrangement correction still follows later. Working
names are not selected APIs, and the user's readability concern is not
promoted into the rigid gate they explicitly declined without seeing code.
The next reading starts at raw line 1164 with design-dependency order and the
audit of questions that had already been resolved in meaning.

## Reading 08: real design frontier, preparation publication and identity

The [94-message continuation](sessions/08-design-frontier-preparation-and-identity.md)
covers raw lines 1164-2189 in the same root. It distinguishes the design
dependency from execution order and migration evidence: derive required
exchanges from intended Trainer responsibilities, not today's aggregate ABC
or family fields. The user then challenges why open questions keep growing.
The audit separates already settled behavior, later Python choices and genuine
remaining architectural decisions.

The [source record](sessions/08-source-records.json) preserves the first
committed [implementation mapping](snapshots/d44b2ccb-strategy_system_implementation_mapping.md),
[exchange](snapshots/d44b2ccb-strategy_contract_exchange_design.md) and
[notes](snapshots/d44b2ccb-notes.md) at `d44b2ccb`. The initial mapping is a
small deferred code-design bridge, not the eventual full production map or
another source of architecture. The user explicitly permits leaving it for
now while declining another growing living document.

Preparation-read and publication questions are resolved through an
attempt-scoped coherent job and a final logical publication containing no
expected-fallible backend work. The destructive path first withdraws affected
guarantees; failure cannot silently recover them. Backend composites and
optimizer state retain their distinct owners. These are semantic decisions,
not tested production behavior or crash-durability guarantees in this reading.

The identity discussion distinguishes permanent authority-qualified
incarnation, changing revisions, logical run, execution session, artifact and
lineage. Retirement closes current use without changing historical identity.
It also contains a later-superseded fallback: missing exact-resume identity
facts were described as starting a new run. The governing delta explicitly
tightens that in `6db3f1ce` on October 4: report incomplete same-run restoration
and require an explicit separate new-run request. This is a targeted later
disposition check; the intervening October history is not claimed complete.

H033-H036 record these decisions and the user's retrospective origin
correction: substantial general metadata work led to an attempt at richer
model metadata, which exposed unsuitable strategy/model/Trainer boundaries.
It was not a mature model-metadata layer merely awaiting integration. August
25 dates the clarification, not the original architectural pivot.

The binding/preparation architecture register reaches zero open entries for
that bounded scope. Optimization, execution and migration remain ahead. The
next reading starts at raw line 2190: how to test the design, the distinction
between paper consistency and executed evidence, and the first isolated spike.

## Reading 09: executable evidence and semantic optimization

The [114-message continuation](sessions/09-executable-evidence-and-optimization.md)
covers raw lines 2190-4092. The user rejects calling a paper walkthrough a
test and authorizes new, isolated code. The spike uses fake objects and a
preparation backend surrogate, then gains scenarios from several historical
reviews. The final report records 28 passing scenarios. It does not establish
real backend integration, serialization durability or a production API.

The [source record](sessions/09-source-records.json) retains the actual returned
final review and three external feedback documents. `b45f1d4e` supplies exact
exchange/notes snapshots and hashes for the two code blobs. `bb763b59` first
commits the subsequent optimization work but also includes the next chapter's
corrections. The record distinguishes those Git dates from the earlier
discussion and testing claims. No scenario or review was rerun for this audit.

H037 records the useful backend-group, cross-owner publication, stale-result
and destructive-attempt findings while retaining the invalidation-policy
qualification. H038-H039 distinguish optimization identity/revision from
concrete parameters, wrappers and mutable progress; reject ordinary parameter
overlap; and keep semantic ownership separate from backend construction order.

The passage ends with a reply that overstates current `process_batch()` and
possible EDM2 placement as target conclusions. The next user message, raw
4093, rejects those assumptions. The correction and the eventual accepted-run
arrangement must be read next; this endpoint is not the current architecture.

## Reading 10: accepted arrangement and the governing-design transition

The [153-message continuation](sessions/10-accepted-arrangement-and-openspec-transition.md)
covers raw lines 4093-6092. The user first rejects premature claims about a
future batch callback, EDM2 placement and strategy takeover. The deeper
discussion then exposes the authoring/runtime conflation. Its first
retrospective explanation is qualified after rereading the whole old session:
the active-strategy interpretation had reasons and bundled approval during
Q5, but had not been isolated from binding ownership as a separate choice.

The [source record](sessions/10-source-records.json) preserves the final
direction, exchange, inventory, comparison, mapping and notes at `bb763b59`,
plus the two discussion companions, supplied feedback and original transition
review. The copied primary discussion has seven whole-body correspondences
with this episode after line-ending normalization for comparison. The external
discussion is a separate attributed conversation; no matching stable session
identities are invented for it.

H040 traces the accepted arrangement as retained executable meaning rather
than an active authoring service or renamed lifecycle facade. H041 records
maintained/custom operations and granted imperative regions in one arrangement,
with capability semantics remaining outside generic Trainer machinery.
Graphs, regions, schedules and decorators are implementation/API options at
this stage. Early claims that all variability must be frozen or that one call
has insignificant overhead are not current semantic restrictions or measured
performance evidence.

H042 traces the final authority/order corrections: intended Trainer goals
and supported capabilities define the contract before authoring; inspecting
SDXL first cannot define the universal structure. SD, SD3 and a strong research
case remain required comparative pressures. The final decision starts one
governing OpenSpec now as a design location, while production migration remains
gated. Settled direction is inherited input; the inventory, chronological
notes, external opinions and spike retain their distinct evidence roles.

The September continuation is now recovered in
[Reading11](sessions/11-first-openspec-draft-and-reconciliation.md) and the
[rebuild](sessions/12-governing-rebuild-and-contract-system.md). The source
window after this August checkpoint resumes on September18; the separate
orientation root and pre-draft rebuild fork are explicitly distinguished.
The ZIP location/content gap is resolved, with creation/removal limits below.

## Earlier leads outside the starting folder, now classified

The first direction-writing session says earlier partial notes existed. The
tree at `de2dd0eb` contains `docs_design/archive/models-strategy-trainer/pre-direction/strategy_architecture_ideas.md`,
`docs_design/archive/models-strategy-trainer/pre-direction/strategy_critique.md`, and
`docs_design/archive/models-strategy-trainer/pre-direction/strategy_contract_pressure_audit.md`, alongside model-metadata
research. Their May Git history and distinct roles are now documented in
[supplementary sources](supplementary-sources.md), with exact copies and date
metadata. Their proposals are antecedents, not competing current requirements.

The September earlier-draft lead is resolved by the actual
`docs_design/archive/models-strategy-trainer/packages/mst_pre_astra.zip`, documented below. It was outside the
normal archive directory. Its storage is not evidence that delta specs were
synced to main specs; no archive/sync operation is established for that draft.

## September draft provenance recovered in Reading 11

The separate September 18 orientation thread has five messages and no edits;
it is not the source of the draft. The original August root resumes at raw
6108 and writes the first `rework-model-strategy-trainer` draft. User questions
at raw 6270/6280 establish that a summary and partial reread were insufficient;
the assistant acknowledges this and begins reconciliation before that root ends.
See [Reading11](sessions/11-first-openspec-draft-and-reconciliation.md).

The earlier archive is [mst_pre_astra.zip](../../../docs_design/archive/models-strategy-trainer/packages/mst_pre_astra.zip),
not a directory under OpenSpec's usual archive home. All nine non-directory
entries have exact preserved copies under `snapshots/2026-09-18-first-openspec-draft/`.
Eight match the source patches exactly, including the recorded loaded-component
scenario-title correction. Their paths, sizes, hashes and ZIP timestamps are
in [11-source-records.json](sessions/11-source-records.json). The generator's
manifest is the ninth entry. This closes the location/content gap, not the
unknown creator or exact time/method of ZIP creation and directory removal.
ZIP timestamps have no timezone; filesystem dates do not supply those facts.

The archive is untracked and has no history in HEAD ancestry. An all-refs
query encountered an unrelated broken Codex checkpoint ref, so absence from
all Git refs is not established. No ref repair was attempted.

Its design/tasks explicitly say production is gated. Later criticism of
incomplete transfer and missing spec deltas is distinct from the overly strong
claim that the draft itself declared implementation readiness. Storing a ZIP
also does not establish that OpenSpec synchronized or archived its requirements.

## Governing rebuild and first checkpoint, Reading 12

The original September19 file branches from the August root before the draft,
at header ordinal6088. Its first user turn explicitly asks to recover the
months of detail. It audits the ZIP and current spec conflicts, then rebuilds
the change. The registered candidate's later October path is not the original
September file or the date of this creation. [Source records](sessions/12-source-records.json)
preserve the exact header, parent/hash, all new message locations and the next
reading boundary. No inherited body is duplicated.

The first successful historical review finds that the rebuild itself still
misses three delta/main heading matches, typed loading/provenance preservation,
residual `AdapterMode`-gated LoRA/VeRA and observability scenarios, and explicit
cross-change sequencing. Fixes grow thirteen capability deltas to fifteen.
The follow-up approves a fixed metadata-first order, but user discussion then
supersedes it: this rework governs overlapping runtime semantics; broader
metadata obligations survive and may be developed alongside those boundaries.
Neither entire change is required to land first. Original reports remain
verbatim; the final checkpoint records the later user correction.

H045-H048 in the [notepad](decision-notepad.md) distinguish that sequence from
the central contract refinement. Contract describes supported meaning;
fulfillment judges explicit authored work and establishes revisioned obligations;
realization gathers evidence, including backend preparation; publication exposes
only verified ready state; continued authority uses the same obligation source.
One system does not require one object or independent validators. Preparation
is not postponed until after verifying a concrete result it has yet to create.

The [first committed package](snapshots/8ec7249d-rework-model-strategy-trainer/design.md)
records these meanings. Its remaining design gates and historical validation
claims do not complete production G5. The next chronological passage starts
raw1577 on September20 16:48UTC, continuing the initial walkthrough.

## Initial walkthrough completion, Reading 13

[Reading13](sessions/13-dynamic-behavior-peft-and-spec-reconciliation.md)
covers 123 selected messages from three physical capture files. Their headers
report the same logical session and original August fork; later pages also
record `history_base`. The exact prefix relationships are:

| Predecessor capture | Selected prefix end | Next page | Verified boundary |
| --- | --- | --- | --- |
| Original September19 file | raw1720, ordinal7807 | `...01a0c057...jsonl` | end ordinal7808 exclusive; 13,947,078 prefix bytes |
| `...01a0c057...jsonl` | raw286, ordinal8093 | `...01a0c066...jsonl` | end ordinal8094 exclusive; 2,612,206 prefix bytes |

These are verified byte lengths and hashes, not guessed ancestry from names.
The [six-message appendix](sessions/13-branch-differences.md) retains the
excluded suffixes: the model-structure/recipe-treatment clarification and an
initialization assent followed by an unnecessary confirmation request. The
selected continuation revises those prompts. Distinct branches are not
concatenated or silently treated as inherited messages.

The first following [committed design](snapshots/58ab91ce-rework-model-strategy-trainer/design.md)
is September21 07:02:35UTC, after the initial spec audit. Its twelve changed
OpenSpec files are exact copies. The package's other seven files match already
preserved `8ec7249d` bytes. `.gitignore` is part of that historical commit but
outside this architecture capture; no audit edit touches it.

The core refinements are H049-H054: dynamic executable meaning survives
finished authoring; shared PEFT does not become a new mode; artifact products
are independent from training treatment; semantic adapter targeting and host
resolution are not optimizer policy; design gates are distinct from deferred
API choices; one authority is an ownership invariant, not a prescribed giant
class or a revision bump on every optimizer update.

The early per-run coexistence rule is recorded accurately as an allowance.
It is not a requirement to keep every intermediate repository revision
runnable. The current migration plan explicitly makes that distinction; the
later September21-22 conversation challenges runnable slices, extra stages
and placement-by-lifecycle-word. That is the next chronological reading,
before the broader pressure-case research and language investigations.

## Placement and coherent run-root provenance, Reading 14

[Reading14](sessions/14-training-first-placement-and-one-run.md) preserves
106 selected messages and a [26-message appendix](sessions/14-branch-differences.md).
Eight prefix relationships select nine pages, including two pages whose
prompts were replaced before contributing visible messages. Exact byte offsets
and ordinal limits are checked. The next-page boundary and the separate short
`01a0c753` fork's two pages are also verified: eleven prefix relationships in
all. The appendix preserves alternate layouts and corrections without
presenting them as the chosen continuation.

The final provisional map is a responsibility map. Contracts/checking and
reusable strategy implementations are distinct from accepted training state,
phase order, capability coordination and Trainer mechanics. Initial creation
seeds that same state root; it does not create a permanent neutral handoff
subsystem. The user prefers `state`, explicitly with names still provisional.
An internal state section is not a competing state authority or an unrelated
run object.

`provisional_tree.md` is **457 lines** at its first commit, `722da952`.
After comparison-only CRLF normalization, it contains the entire bodies of
selected message076 (combined tree) and089 (training sweep). The exact copy
is retained. Present-inode birth is September22 07:51+02:00; modification is
September23 02:07+02:00; the first Git capture is September23 09:01+02:00.
These are different observations. None establishes who pasted it or converts
its example paths into final interfaces.

The sweep copy compresses concrete target and parameter resolution into
training optimization. The more precise governed adapter-targeting delta
from the preceding checkpoint remains authoritative: authored semantic intent,
governed PEFT host resolution, method construction, then optimizer parameter
grouping/advancement. This wording difference is recorded rather than silently
fixed in the source. The committed governing design's specific delta meanings
and source roles are retained; the copied sweep is evidence, not an override.

The `722da952` design snapshot is a later first capture of the September22
updates, not an exact end-of-day draft: it already contains ensuing research
and reconciliation. That later material is the next phase to trace. The next
page is `...01a0cb63...`, header September22 23:11UTC, followed by its revised
`...01a0cb67...` continuation. Their own prefix cutoffs must guide the next export.

## Later phase 15: Research pressures and the whole-run language direction, September 2026

[Reading15](sessions/15-research-and-whole-run-language.md) traces broader research, ordinary and adversarial training, staged topology, independent producers, stateful inputs and the first action/whole-run language experiments.
The [sources](sessions/15-source-records.json) distinguish experiment and
review evidence from adopted requirements. Original documents and prior
immutable captures are unchanged. Dates come from exact message records and
commit metadata; present inode dates are not treated as authorship.

## Later phase 16: Exchange design, independent audits and the G5 correction, September-October 2026

[Reading16](sessions/16-exchanges-audits-and-g5-correction.md) traces G3/G4 exchange completion, early executable research, restoration correction, pinned multi-pass audit and the rejection of following G5 component checkboxes without a connected execution mechanism.
The [sources](sessions/16-source-records.json) distinguish experiment and
review evidence from adopted requirements. Original documents and prior
immutable captures are unchanged. Dates come from exact message records and
commit metadata; present inode dates are not treated as authorship.

## Later phase 17: Connected execution candidate and IR machinery investigation, October 2026

[Reading17](sessions/17-connected-candidate-and-ir-machinery.md) traces connected whole-system mocks, graph-derived execution, shared owners, state/gradient/physical lifetimes, independent oracles, free-threading and backend qualifications, Togeari learning and the next IR machinery candidate.
The [sources](sessions/17-source-records.json) distinguish experiment and
review evidence from adopted requirements. Original documents and prior
immutable captures are unchanged. Dates come from exact message records and
commit metadata; present inode dates are not treated as authorship.

## Later phase 18: xDSL and Pliron comparison and working-substrate choice, October-November 2026

[Reading18](sessions/18-framework-comparison-and-xdsl-choice.md) traces framework screening, xDSL and pinned Pliron experiments, independent extensions, matched six-case comparison, machinery tradeoffs and explicit xDSL working selection.
The [sources](sessions/18-source-records.json) distinguish experiment and
review evidence from adopted requirements. Original documents and prior
immutable captures are unchanged. Dates come from exact message records and
commit metadata; present inode dates are not treated as authorship.

## Later phase 19: Shared language, retained products and continuity into G5, November 2026

[Reading19](sessions/19-shared-language-and-current-frontier.md) traces language proposal and explicit amendments, shared semantic slice/reference, verification, lowering, cold/independent qualification, binding effects/filtering/fusion and retained storage/autograd products; ends immediately before this historical audit.
The [sources](sessions/19-source-records.json) distinguish experiment and
review evidence from adopted requirements. Original documents and prior
immutable captures are unchanged. Dates come from exact message records and
commit metadata; present inode dates are not treated as authorship.
