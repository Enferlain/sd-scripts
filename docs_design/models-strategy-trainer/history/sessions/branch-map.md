# One conversation through successive branches

The inspected passage follows the parent chain below. A different session ID can preserve the same discussion while replaying its records with new timestamps. The thread title does not establish a new topic. Shared conversation is exported once; distinct branch material has its own appendix.

[Chapter00](00-initial-direction.md) | [Chapter01](01-inventory-and-authoring.md) | [Branch differences](01-branch-differences.md)

| Session | Parent | Header date UTC | Inventory-window messages shared / different from selected path |
| --- | --- | --- | --- |
| `019f585b-9603-72c3-8ce7-d89324108d2e` | `none` | 2026-07-12T22:03:45.038Z | Before this episode |
| `019f6cec-7e2a-7d41-9362-bf247eb3ffb3` | `019f585b-9603-72c3-8ce7-d89324108d2e` | 2026-07-16T21:54:25.987Z | Before this episode |
| `019f86c1-533d-70d0-8e4d-90cd9e9b62de` | `019f6cec-7e2a-7d41-9362-bf247eb3ffb3` | 2026-07-21T22:17:24.549Z | Before this episode |
| `019f86d4-43a2-73a2-a37a-ba7c5a26a4f5` | `019f86c1-533d-70d0-8e4d-90cd9e9b62de` | 2026-07-21T22:38:05.739Z | 2 / 1 |
| `019f8c75-6709-7f02-9e99-a3d5d9e5fffd` | `019f86d4-43a2-73a2-a37a-ba7c5a26a4f5` | 2026-07-23T00:52:12.181Z | 2 / 16 |
| `019fa69a-b709-7b73-bb57-abe2021bfdb9` | `019f8c75-6709-7f02-9e99-a3d5d9e5fffd` | 2026-07-28T02:43:05.109Z | 2 / 15 |
| `019fab38-3678-72c3-b798-adb2e9e225b9` | `019fa69a-b709-7b73-bb57-abe2021bfdb9` | 2026-07-29T00:13:35.746Z | 17 / 4 |
| `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f` | `019fab38-3678-72c3-b798-adb2e9e225b9` | 2026-07-29T02:06:11.949Z | 35 / 0 |

The selected continuation is `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`. Its chapter01 window has 35 messages. Across inspected ancestors, 36 distinct messages in the same bounded window do not occur in that path. These are identity/content differences, not a count of architectural disagreements.

Notable differences to inspect:

- The 21-July branch ends with an unrelated preview-button problem that is absent from the selected continuation.
- The 23-July branch includes graph-health checks and a 26-July inventory start; the 28-July branch takes another inventory route.
- The 28-July attempt includes a context-loss/recovery discussion. The 29-July revised inventory instruction explicitly asks to preserve findings between milestones.
- The short 29-July parent branch contains the literal clarification that the builder is whoever writes the strategy code. The selected child revises the earlier authoring message and continues with the same explicit-authoring distinction.

This is one inspected lineage, not proof that all siblings and unregistered copies have been discovered. A branch-only assertion is labelled as such and is not automatically a requirement of the selected path. The raw hashes and exact source lines are retained in both JSON exports.

## Chapter02 continuation

[Framework comparison and contract surfaces](02-frameworks-and-contract-surfaces.md)
continues in the same selected session, from raw lines 12193-13058 on 29 July.
All 48 messages in this passage occur in that session and none match records
in its seven inspected earlier ancestors. They are new continuation rather
than another replay of chapter01. No additional branch-difference appendix is
needed for this interval in the inspected chain.

User message 036 refers to an external feedback attachment. Its
[readable copy](../snapshots/2026-07-29-contract-surfaces-feedback.md) preserves
that separate source; it is not silently folded into the user's own words.

## Chapter03 continuation

[Code pressure and mode/objective classification](03-training-mode-and-objective.md)
continues from raw lines 13059-13894 in the same selected session. Its 38
messages span 29 July, 23:07 through 31 July, 00:01 UTC. They match records only
in that selected continuation within the inspected eight-session chain, so
there is no new ancestor-difference appendix for this interval.

The chapter stops before the user's request to walk through the first of the
five binding/preparation questions. The historical sequence of corrections is
kept intact; a corrected proposal is not silently substituted into an earlier
reply.

## Chapter04 continuation

[Logical identity and execution routes](04-identity-and-execution-routes.md)
continues from raw lines 13895-14033 in the same selected session. Its 12
messages include the July 31 Q1 answer and August 20 resumption through the
Q2 checkpoint. They occur only in the selected continuation within the
inspected eight-session ancestry. The calendar gap does not establish a new
conversation or require replaying the inherited chapters.

The August recap confirms Q1 had been addressed and continues at Q2. Its
quoted recap and the later turn labelled Feedback retain their original
attribution. This reading does not invent a numbering error or infer who
originally authored the supplied feedback.

## Chapter05 continuation

[Arrangement changes and artifact persistence](05-arrangement-and-persistence.md)
covers raw lines 14034-14587 in the same selected session. Its 35 messages
contain Q3's provisional checkpoint, supplied feedback, corrected settlement
and compaction-recovery check, followed by Q4's evaluation and settlement.
Matching messages occur only in that continuation among the eight inspected
ancestors. The two compaction markers are retained without generated summaries.
The three attachments are exact copies with separate provenance; their claims
are not silently recast as user decisions.

## Chapter06 and the fresh-session handoff

[Canonical authority, reviews and handoff](06-authority-reviews-and-handoff.md)
covers raw lines 14588-15633, the end of the selected long session. Its 65
messages occur only in that continuation among the inspected ancestors.
Two compaction markers remain visible without generated summaries.

The next thread is a new root, not another recorded fork:
`01a026d5-a88b-7843-a1c0-1b51047c8182`, header
`2026-08-22T00:18:51.669Z`, no `forked_from_id`.
Its [first user message](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:9)
quotes the previous handoff and explicitly identifies the old session as its
raw background. That establishes semantic continuation without inventing a
fork relationship. Reading 07 begins its export; this bridge is not used to
pretend it inherited the old session's records.

## Chapter07 in the new primary root

[First exchange design and contract-authority correction](07-contract-exchanges-and-authority-correction.md)
covers raw lines 1-1163 of `01a026d5-a88b-7843-a1c0-1b51047c8182`.
It has 88 visible messages and three compaction markers. There is no recorded
fork parent; the semantic predecessor is the earlier thread's explicit handoff.
The registered candidate index identifies this root, but this bounded reading
does not claim exhaustive discovery of unregistered siblings or copies.

## Chapter08 continuation

[Design frontier, preparation publication and durable identity](08-design-frontier-preparation-and-identity.md)
covers the next raw lines 1164-2189 in the same root, with 94 visible messages
and two compaction markers. Inline external feedback remains in the original
user turns. The retrospective origin corrections and context-recovery checks
are retained rather than collapsed into the first inaccurate summaries.

## Chapter09 continuation

[Executable evidence and optimization groundwork](09-executable-evidence-and-optimization.md)
covers raw lines 2190-4092 in the same root: 114 visible messages and three
compaction markers. Historical review waits and lost-output recovery remain
visible. The recovered final spike report is separate from the author's
summary and the supplied external feedback. No new worker or reviewer was
launched during this historical capture.

## Chapter10 continuation and the September gap

[Accepted-arrangement correction and OpenSpec transition](10-accepted-arrangement-and-openspec-transition.md)
covers raw lines 4093-6092 in the same root, with 153 visible messages and four
compaction markers. The first user correction and the eventual accepted-run
topology remain in sequence; intermediate proposals are not silently replaced
by the final answer. The next substantive message is raw 6108 on September 18.
That calendar gap alone is not a new session or fork. The separately registered
`01a0b1d2-0138-7c11-bc81-4ee074b48204` misstart still needs correlation during
the governing-draft reading.

## Chapter11: orientation root and September draft continuation

[First draft](11-first-openspec-draft-and-reconciliation.md) has 19 messages.
Messages 001-005 are the separate `01a0b1d2-0138-7c11-bc81-4ee074b48204`
root, September 18 00:03-00:04 UTC, and perform orientation only. Messages
006-019 cover raw 6093-6323 of the original August root, including its
00:08 compaction and 00:10-00:34 drafting/correction. Chronological adjacency
is not fork ancestry; neither root is deduplicated into the other.

The September 19 original file for `01a0bb09-899f-7d32-957d-99b317e588ce`
was found outside the registered candidate path. Its header records the August
root as parent at ordinal 6088, before the September draft. The current
registered candidate points to a later October file with that same logical
session ID; its title or latest header must not date the September creation.
The next chapter records the actual original path, hash and fork boundary.

## Chapter12: original rebuild file, not the later registered copy

[Governing rebuild](12-governing-rebuild-and-contract-system.md) uses
`archived_sessions/rollout-2026-09-19T20-59-19-01a0bb09-899f-7d32-957d-99b317e588ce.jsonl`.
Its header time is September19 18:59:19.603UTC and its parent is
`01a026d5-a88b-7843-a1c0-1b51047c8182`, ordinal6088 exclusive. Ordinal6088
is not raw line6088. This branches before the first draft rather than inheriting
its later correction turns. The first actual user record is raw12 at19:03UTC.

The registered candidate with that logical ID currently points to an October2
file. The original file and that later file are separate captures of an evolving
logical thread; latest header/registration dates cannot date earlier decisions.
Raw1-1576 yield87 new visible messages and three compaction markers. The
August parent is hashed and retained as provenance; none of these new bodies
is a repeated inherited message from earlier exports. Next is raw1577 in the
same original file, before following the later September forks.

## Chapter13: paginated history bases and six distinct suffix messages

[Selected walkthrough](13-dynamic-behavior-peft-and-spec-reconciliation.md)
uses original September19 raw1577-1720, `...01a0c057...` raw1-286, then
`...01a0c066...` raw1-1404. The three captures report one logical session ID.
Their `history_base` records give page-specific predecessors, end ordinals and
byte offsets; those exact prefixes are verified in
[13-source-records.json](13-source-records.json). They are not three new roots
or a fabricated new `forked_from_id` chain. Three compaction markers are
retained in their correct chronological positions despite raw line resets.

The original file's raw1721-1739 and the next page's raw287-299 are outside
the selected continuation cutoffs. Their six visible messages are exported
once in [branch differences](13-branch-differences.md). The first supplies the
literal model-structure/strategy-treatment clarification; the next selected
prompt instead stresses dynamic behavior. The second revises an assent and
avoids the intervening redundant confirmation branch. Both original bodies
and distinct turn identities survive.

The selected reading stops after the first spec double-check. Next begins
`...01a0c066...` raw1405, September21 22:02:43.758UTC. Subsequent paginated
captures must use their own history-base cutoffs before appending migration
and placement replies. Dates and similar titles alone are insufficient.

## Chapter14: eight selected prefix links and the short layout fork

[Selected placement discussion](14-training-first-placement-and-one-run.md)
continues `...01a0c066...` from raw1405 through raw1833, then follows
`c63a`, `c673`, `c69d`, `c6a6`, `c6e6`, `c758`, `c77d` and `c78f` at their
explicit history-base cuts. The two `c69d`/`c6a6` selected prefixes contain
header/context records only; their actual proposed replies are distinct suffix
material, preserved in the appendix. This is not evidence those pages were
empty or that their proposals were adopted.

The [branch appendix](14-branch-differences.md) contains22 excluded suffix
messages and4 from the separately registered `01a0c753` layout fork and its
revised page. All26 retain distinct identity/content keys. The suffixes include
the earlier runnable-slice rejection and four-object sketch, validation-location
question, alternate layouts, an ordinary-Python-function authoring hypothesis
and naming variants. Their source order is grouped by branch, not fabricated
as one chronological continuation.

[14-source-records.json](14-source-records.json) verifies eleven exact byte
prefixes: eight selected links, the next `c78f`→`cb63` link, and the two links
through the short layout fork. Header logical session IDs, original fork
metadata and history-base page IDs remain separate fields. Four compaction
markers are retained without generated summaries. The next selected export
must follow `cb63` and then `cb67` using their actual cutoffs.

## Chapter15: Research pressures and the whole-run language direction, September 2026

[Selected reading](15-research-and-whole-run-language.md) contains 419 messages from
11 physical captures. 11 recorded page-prefix
relationships are verified against exact ordinals and byte offsets.
23 distinct suffix messages remain in the separate appendix when present;
0 earlier exported suffix records are referenced
instead of copied. Active source sessions use a fixed historical-prefix hash.

## Chapter16: Exchange design, independent audits and the G5 correction, September-October 2026

[Selected reading](16-exchanges-audits-and-g5-correction.md) contains 631 messages from
11 physical captures. 10 recorded page-prefix
relationships are verified against exact ordinals and byte offsets.
18 distinct suffix messages remain in the separate appendix when present;
0 earlier exported suffix records are referenced
instead of copied. Active source sessions use a fixed historical-prefix hash.

## Chapter17: Connected execution candidate and IR machinery investigation, October 2026

[Selected reading](17-connected-candidate-and-ir-machinery.md) contains 897 messages from
11 physical captures. 11 recorded page-prefix
relationships are verified against exact ordinals and byte offsets.
20 distinct suffix messages remain in the separate appendix when present;
0 earlier exported suffix records are referenced
instead of copied. Active source sessions use a fixed historical-prefix hash.

## Chapter18: xDSL and Pliron comparison and working-substrate choice, October-November 2026

[Selected reading](18-framework-comparison-and-xdsl-choice.md) contains 403 messages from
1 physical captures. 0 recorded page-prefix
relationships are verified against exact ordinals and byte offsets.
0 distinct suffix messages remain in the separate appendix when present;
0 earlier exported suffix records are referenced
instead of copied. Active source sessions use a fixed historical-prefix hash.

## Chapter19: Shared language, retained products and continuity into G5, November 2026

[Selected reading](19-shared-language-and-current-frontier.md) contains 191 messages from
1 physical captures. 0 recorded page-prefix
relationships are verified against exact ordinals and byte offsets.
0 distinct suffix messages remain in the separate appendix when present;
0 earlier exported suffix records are referenced
instead of copied. Active source sessions use a fixed historical-prefix hash.
