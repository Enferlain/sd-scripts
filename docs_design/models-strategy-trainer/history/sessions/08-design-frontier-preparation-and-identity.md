# Design frontier, preparation publication and durable identity, 24-25 August 2026

Continuation of [chapter07](07-contract-exchanges-and-authority-correction.md) in the same primary session, raw lines 1164-2189. This reaches the end of the binding/preparation architecture audit before any executable spike is started.

The [exchange](../snapshots/d44b2ccb-strategy_contract_exchange_design.md), [notes](../snapshots/d44b2ccb-notes.md), and [first implementation mapping](../snapshots/d44b2ccb-strategy_system_implementation_mapping.md) preserve the committed result. The mapping separates deferred code choices from unresolved architecture; it is not a second governing design. The user explicitly warns against another growing living document.

The preparation decision is one coherent attempt-scoped job, possibly several backend calls; ordinary fallibility precedes final logical publication. Destructive in-place work first withdraws affected guarantees and cannot restore them merely because preparation failed. Publication/rank termination is not claimed crash durability.

The identity discussion separates permanent authority-qualified incarnation, revisions, per-process observation session, artifacts and lineage. The metadata chronology is corrected twice: general metadata work exposed unsuitable boundaries during the attempt to develop richer model metadata; it was not a mature model-metadata subsystem simply waiting for Trainer integration.

**Later disposition:** this passage says missing exact-resume identity facts lead to a new run. The current [restoration requirement](../../../../openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md#requirement-runtime-identity-and-lineage-remain-separate) instead reports incomplete same-run restoration and requires a separately requested new run. The source record dates that later correction. The active-strategy enclosure also remains historical and is subsequently replaced by the accepted arrangement.

Captured 2026-11-05T22:16:27.132751+00:00. 94 visible user/assistant messages. Exact bodies, annotations, historical links and line endings are preserved. Reasoning, tools, generated context and compaction summaries are omitted.

[Machine-readable provenance](08-design-frontier-preparation-and-identity.json) | [Sources](08-source-records.json) | [Notepad](../decision-notepad.md)

## Message index

- [001 user: what we'll need is what the best dependency path for figuring out what the code will be. like maybe ](#m001)
- [002 assistant: Yes—that should be our first task tomorrow: determine the investigation order before designing more ](#m002)
- [003 user: Ready to look into it. the caveat is still the same as before. all code currently is before any modi](#m003)
- [004 assistant: I’ll treat today’s code as evidence of Trainer’s existing dependency pressures, not as the target ar](#m004)
- [005 assistant: The exploration constraints fit: we’ll map and reason, not implement. I’m now establishing the curre](#m005)
- [006 assistant: The first useful distinction is emerging: we need two dependency paths, not one. One is the target s](#m006)
- [007 assistant: The current path confirms the caveat matters in a very specific way: today’s launcher validates conf](#m007)
- [008 assistant: The OpenSpec exploration workflow is keeping this at the design boundary: I’m changing only the evol](#m008)
- [009 assistant: The first semantic slice is clear enough to record without inventing classes: an intended-Trainer co](#m009)
- [010 assistant: The Beads workflow is now useful as durable handoff: I’m recording this dependency-path correction o](#m010)
- [011 assistant: Yes: design backward from the intended Trainer—but not mechanically backward from today’s `Trainer` ](#m011)
- [012 user: the open question/discussion/handoff section keeps growing, how come?](#m012)
- [013 assistant: Because I’ve been using those sections as append-only safety nets. After the earlier normalization r](#m013)
- [014 user: can you speak in words that I can understand, what normalization problem?](#m014)
- [015 assistant: Sorry—I meant the cleanup attempt from August 22.](#m015)
- [016 user: shortening is not the problem, why does it keep getting longer?](#m016)
- [017 assistant: It keeps getting longer because I keep adding newly discovered sub-questions without removing or rep](#m017)
- [018 user: Okay, let's go through the items step by step and see if they're actually resolved or not. I want to](#m018)
- [019 assistant: Let’s do exactly that, one item at a time. I’ll use the exploration workflow to compare each questio](#m019)
- [020 assistant: There are fourteen entries. We’ll start with question 1 only and separate what is already decided fr](#m020)
- [021 assistant: ### Question 1](#m021)
- [022 user: Okay, next one](#m022)
- [023 assistant: Question 2 is about how maintained implementations are recognized as fulfilling contract vocabulary.](#m023)
- [024 assistant: ### Question 2](#m024)
- [025 user: check 3 4 and 5 next](#m025)
- [026 assistant: I’ll check 3–5 together, but keep their conclusions separate. The main thing to distinguish is wheth](#m026)
- [027 assistant: Question 5 may already have a reusable answer in the metadata identity system, so I’m checking that ](#m027)
- [028 assistant: ### Question 3](#m028)
- [029 user: OK so it's as I thought, most of them look like we already talked about and it's just the code level](#m029)
- [030 assistant: Yes, your reading is right. The register is mixing two different things:](#m030)
- [031 user: Okay what I want is for the architecture and design to be relatively well fleshed out. We have multi](#m031)
- [032 assistant: That’s the right level to aim for.](#m032)
- [033 user: You can just move those to the code related notes or something, that's should be the clear distincti](#m033)
- [034 assistant: Exactly. The process should be:](#m034)
- [035 user: Okay, go check the next batch please](#m035)
- [036 assistant: I’ll check questions 6–8 next using the same test: does the answer change architecture or behavior, ](#m036)
- [037 assistant: ### Question 6](#m037)
- [038 user: okay, can you take care of the ones we looked at so far where needed (like when not positive to have](#m038)
- [039 assistant: I’m applying the distinction now. Questions 1–4 and 6–8 will leave the architecture register; their ](#m039)
- [040 assistant: The separation is now in place. I’m checking the edited documents, then I’ll record the audit result](#m040)
- [041 assistant: Done.](#m041)
- [042 user: don't really want another living document. but it's fine to leave it for now. please check the remai](#m042)
- [043 assistant: Understood. I won’t grow the implementation-mapping file further while we audit this. I’ll check for](#m043)
- [044 assistant: I’m checking the actual exchange, scenario, and direction text now. The key distinction will be whet](#m044)
- [045 assistant: After checking the remaining items against the direction, scenarios, and exchange design, only two r](#m045)
- [046 user: so what should happen to them? are they fine as they are](#m046)
- [047 assistant: No, they are not fine as they are. Leaving them there would falsely imply that all six are equally u](#m047)
- [048 user: okay, can you do that please](#m048)
- [049 assistant: I’ll clean the exchange document itself—both the register and the corresponding `OPEN` wording—so th](#m049)
- [050 assistant: The cleanup has one important consequence beyond shortening the register: the body currently labels ](#m050)
- [051 assistant: The document is now internally aligned: only preparation reads and prepared-state commit/recovery re](#m051)
- [052 assistant: The register now contains only three genuine architecture questions:](#m052)
- [053 user: Alright, now let's progress on the remaining things that genuinely need to settle before getting int](#m053)
- [054 assistant: We should take these in dependency order, starting with the preparation read boundary. It is the inp](#m054)
- [055 assistant: The recorded scenarios constrain most of the preparation input already, but I’m checking the current](#m055)
- [056 assistant: I think the preparation-read question can now be settled architecturally.](#m056)
- [057 user: 3rd party feedback:](#m057)
- [058 assistant: That feedback is useful and catches one overstatement in my wording: the preparation input is one co](#m058)
- [059 assistant: The feedback improves the design. I agree with it, with one clarification: “non-failing publication”](#m059)
- [060 user: How about taking care of that? unless there's something else we need to consider before](#m060)
- [061 assistant: There is one caveat worth preserving, but it does not block the decision: destructive in-place prepa](#m061)
- [062 assistant: The final pressure check did not expose another preparation-level choice. I am also keeping the larg](#m062)
- [063 assistant: The design record now reflects the corrected semantic unit and the simplified publication rule. I’m ](#m063)
- [064 assistant: Recorded and reconciled throughout the design.](#m064)
- [065 user: after what?](#m065)
- [066 assistant: After settling **durable identity versus lineage**.](#m066)
- [067 user: what does durable identity versus lineage mean](#m067)
- [068 assistant: It asks: **when do we say this is still the same participant, and when do we say it is a new partici](#m068)
- [069 user: well, how about you check the code for what exists today, and then based on our existing design docu](#m069)
- [070 assistant: Yes—that is the right way to settle it. I’ll trace today’s component identity through loading, metad](#m070)
- [071 assistant: The current metadata system already has a strong precedent: model and component IDs are qualified by](#m071)
- [072 assistant: The important gap is now visible: today’s metadata component ID is already run-qualified, but today’](#m072)
- [073 assistant: The code and existing metadata design point to a fairly clear answer.](#m073)
- [074 user: Well, the work order was stuff -> metadata -> going into models/strategy/etc planning, so metadata m](#m074)
- [075 assistant: I’ll record it now, with your ordering caveat made explicit: the existing metadata system is evidenc](#m075)
- [076 assistant: The decision is now in the exchange design and chronological notes. I’m checking that the active reg](#m076)
- [077 assistant: Recorded in:](#m077)
- [078 user: context was summarized, please check the past 2-3 turns in session 01a026d5-a88b-7843-a1c0-1b51047c8](#m078)
- [079 assistant: I’ll verify the actual last few exchanges from that session, then compare them with what I recorded.](#m079)
- [080 assistant: I checked the actual exchanges. The identity/resume decision itself was recorded correctly, but I sl](#m080)
- [081 assistant: I checked the actual final exchanges. The identity/resume conclusion was right; the interpretation o](#m081)
- [082 user: well yeah we never really got to model metadata properly, when getting into it we got into strategy ](#m082)
- [083 assistant: Right—that is more precise again. We did substantial general metadata work; then, when trying to pro](#m083)
- [084 assistant: Yes—that is the accurate history:](#m084)
- [085 user: And so, where does that leave us? I think you said this but idk if anything is different](#m085)
- [086 assistant: I’m reconciling that earlier explanation against the decision we just recorded. I’m using the explor](#m086)
- [087 assistant: The quoted explanation was the question before we investigated the code. It is now settled, and the ](#m087)
- [088 user: Is this right?](#m088)
- [089 assistant: I’m checking this against the settled retirement and cross-authority rules. The feedback is pointing](#m089)
- [090 assistant: Yes. The feedback catches a genuine wording problem in our recorded sentence.](#m090)
- [091 user: take care of that please and then tell me what we'Ll go through tomorrow](#m091)
- [092 assistant: I’ll correct the identity invariant everywhere it currently implies retirement might end the incarna](#m092)
- [093 assistant: The correction is in place: retirement now ends current usability, not identity. I’m doing the consi](#m093)
- [094 assistant: Taken care of.](#m094)

<a id="m001"></a>
## 001 User - 2026-08-24T01:51:01.948Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1164](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1164).

what we'll need is what the best dependency path for figuring out what the code will be. like maybe backwards from trainer -> validation/contract -> strategy or whatever. we'll look into it tomorrow


<a id="m002"></a>
## 002 Assistant (final_answer) - 2026-08-24T01:51:11.041Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1168](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1168).

Yes—that should be our first task tomorrow: determine the investigation order before designing more types.

The likely path is:

`Trainer’s actual needs → contract vocabulary/capabilities → pre-Trainer enforcement boundary → accepted runtime strategy → authored strategy input`

Then trace forward from authoring to confirm both directions meet cleanly at contract establishment. That should let the code’s real dependencies determine the abstraction, instead of inventing the strategy representation first.

We’ll verify this order against the direction, inventory, and current Trainer/strategy code before committing to it.

<a id="m003"></a>
## 003 User - 2026-08-24T18:24:24.592Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1176](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1176).

Ready to look into it. the caveat is still the same as before. all code currently is before any modification towards the direction we've been discussing

<a id="m004"></a>
## 004 Assistant (commentary) - 2026-08-24T18:24:32.994Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1180](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1180).

I’ll treat today’s code as evidence of Trainer’s existing dependency pressures, not as the target architecture. I’m using the OpenSpec exploration workflow to keep this pre-implementation, plus the codebase-memory workflow to trace real dependencies before reading candidate source directly. I’ll also keep `sd-scripts-syv` as the durable checkpoint.

<a id="m005"></a>
## 005 Assistant (commentary) - 2026-08-24T18:24:43.634Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1186](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1186).

The exploration constraints fit: we’ll map and reason, not implement. I’m now establishing the current graph generation and the durable design state, then I’ll trace from Trainer outward in both directions so legacy coupling doesn’t masquerade as a future dependency.

<a id="m006"></a>
## 006 Assistant (commentary) - 2026-08-24T18:25:31.686Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1217](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1217).

The first useful distinction is emerging: we need two dependency paths, not one. One is the target semantic path—what must be true before Trainer can execute. The other is the migration path through today’s code. I’m tracing the launcher, strategy construction, validation order, and Trainer’s first consumption points to keep those separate.

<a id="m007"></a>
## 007 Assistant (commentary) - 2026-08-24T18:27:10.395Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1269](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1269).

The current path confirms the caveat matters in a very specific way: today’s launcher validates configuration before a strategy exists, then gives Trainer three separate behavioral authorities—strategy, mode, and objective. So those APIs cannot define the target sequence. They only show which meanings the future accepted strategy must absorb or replace. I’m checking coverage now, then I’ll record the corrected derivation order in the exchange document.

<a id="m008"></a>
## 008 Assistant (commentary) - 2026-08-24T18:27:38.134Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1281](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1281).

The OpenSpec exploration workflow is keeping this at the design boundary: I’m changing only the evolving exchange record and chronological notes, not production code. The main correction is to reverse the design sequence while preserving the runtime sequence—derive the accepted strategy from the intended Trainer’s needs, then derive validation and authoring from that result.

<a id="m009"></a>
## 009 Assistant (commentary) - 2026-08-24T18:29:02.183Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1293](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1293).

The first semantic slice is clear enough to record without inventing classes: an intended-Trainer consumption frame. This is where the external caveat changes the method most—current tokenizers, VAE/text-encoder/denoiser fields, mode hooks, and objective runtime are not presumed Trainer requirements unless they survive the responsibility filter.

<a id="m010"></a>
## 010 Assistant (commentary) - 2026-08-24T18:29:41.638Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1300](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1300).

The Beads workflow is now useful as durable handoff: I’m recording this dependency-path correction on `sd-scripts-syv`, including that the next refinement is binding/materialization from the intended Trainer’s consumption side—not the authored Python shape.

<a id="m011"></a>
## 011 Assistant (final_answer) - 2026-08-24T18:30:11.345Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1310](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1310).

Yes: design backward from the intended Trainer—but not mechanically backward from today’s `Trainer` class.

```text
Design dependency:

intended Trainer responsibilities
  → required exchanges and results
  → accepted TrainingStrategy state
  → contract establishment/conformance
  → authored strategy API
```

Runtime still flows in the opposite direction:

```text
authored definition
  → contract establishment
  → accepted TrainingStrategy
  → Trainer
```

The code inspection confirmed why this distinction matters:

- Current config validation happens before the strategy exists.
- Trainer receives separate strategy and mode objects, then independently builds the objective.
- Trainer stores strategy-produced components and passes its VAE/text-encoder/denoiser projections back into strategy and mode calls.
- The current `TrainingStrategy` ABC mixes core operations, pipeline capabilities, internal features, and conditional methods that merely raise.

Those are migration pressures, not the future contract.

I recorded the dependency method in [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:90) and added the first intended-Trainer consumption table at [line 232](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:232). It currently distinguishes:

- Pre-Trainer contract admission
- Core binding, preparation, optimization, and step exchanges
- Caching, validation, sampling, persistence, and restoration capabilities
- Cross-cutting Trainer-owned observation and cleanup

It explicitly avoids assuming that tokenizers, VAE/text-encoder/denoiser fields, `TrainingMode`, or `ObjectiveRuntime` belong in the future Trainer-facing surface.

The revised investigation order is at [line 1219](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:1219), with the chronological checkpoint in [notes.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:653) and bead `sd-scripts-syv`.

Our next concrete task is now properly bounded: refine the arrangement/materialization row into exact request, result, state transition, side-effect, and failure semantics, with runtime preparation beside it because both operate on the same accepted identities.

<a id="m012"></a>
## 012 User - 2026-08-24T19:19:05.307Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1316](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1316).

the open question/discussion/handoff section keeps growing, how come?


<a id="m013"></a>
## 013 Assistant (final_answer) - 2026-08-24T19:19:29.382Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1321](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1321).

Because I’ve been using those sections as append-only safety nets. After the earlier normalization removed too much reasoning and was reverted, I biased toward preserving every unresolved detail. As decisions became more concrete, I added their downstream questions without consolidating the navigation sections.

That has made one document do three overlapping jobs:

- Open questions: every unresolved design detail
- Discussion order: the entire remaining roadmap
- Compaction handoff: another summary of both

The underlying design is not necessarily expanding that quickly; much of the growth is decomposition of existing questions. But the presentation is becoming misleading and harder to use.

A better shape would be:

- **Open questions:** grouped reference list of genuine unresolved decisions
- **Current frontier:** only the next 2–3 dependency-ordered questions
- **Compaction handoff:** a very short pointer to the frontier, decision register, direction, and inventory
- **Notes:** chronological history and superseded reasoning
- **Bead:** durable session-level status

So yes, this is partly an overcorrection on my side. We should compact those navigation sections carefully—without deleting the detailed reasoning that caused the previous normalization problem.

<a id="m014"></a>
## 014 User - 2026-08-24T19:20:23.987Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1327](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1327).

can you speak in words that I can understand, what normalization problem?


<a id="m015"></a>
## 015 Assistant (final_answer) - 2026-08-24T19:20:30.309Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1331](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1331).

Sorry—I meant the cleanup attempt from August 22.

We tried to reorganize the design documents so each one had a clearer purpose. That cleanup removed too much of the reasoning you wanted to preserve, so you asked me to undo it and restore the documents.

Since then, I’ve been overly cautious about deleting or replacing anything. I kept adding new questions and handoff instructions instead, which is why those sections are getting longer.

So the simple answer is: I reacted too strongly to the earlier bad cleanup. We can shorten these sections without removing the actual design discussion.

<a id="m016"></a>
## 016 User - 2026-08-24T19:20:55.641Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1337](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1337).

shortening is not the problem, why does it keep getting longer?


<a id="m017"></a>
## 017 Assistant (final_answer) - 2026-08-24T19:21:13.238Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1341](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1341).

It keeps getting longer because I keep adding newly discovered sub-questions without removing or replacing the older entries.

For example, yesterday the next task was “design the authored-strategy input.” Today we determined that we first need to work backward from what Trainer needs. I inserted that new prerequisite but left the authored-input task and every later task in the list. That made the list longer instead of simply changing what comes next.

I’ve also been copying each new correction into the handoff section, even though it could just point to the updated design.

So the growth is caused by how I’m maintaining the document:

- New prerequisite found → appended
- Existing future work → retained
- New status → repeated in the handoff
- Detailed implementation questions → kept open even when they are not relevant yet

It shouldn’t continue growing like that. Those sections should describe the current unresolved shape, not accumulate everything we have ever considered.

<a id="m018"></a>
## 018 User - 2026-08-24T20:21:13.670Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1348](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1348).

Okay, let's go through the items step by step and see if they're actually resolved or not. I want to see if the open questions are actually still not cleared or if they have already been solved

<a id="m019"></a>
## 019 Assistant (commentary) - 2026-08-24T20:21:26.540Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1352](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1352).

Let’s do exactly that, one item at a time. I’ll use the exploration workflow to compare each question against the normative direction, later decisions in the exchange document, chronological notes, and code evidence. I won’t change or remove an item until we’ve agreed whether it is resolved, partly resolved, still open, or simply premature.

<a id="m020"></a>
## 020 Assistant (commentary) - 2026-08-24T20:21:32.308Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1358](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1358).

There are fourteen entries. We’ll start with question 1 only and separate what is already decided from what the wording still leaves open.

<a id="m021"></a>
## 021 Assistant (final_answer) - 2026-08-24T20:21:54.131Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1364](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1364).

### Question 1

> What exact Python authoring and establishment boundary lets the contract system judge the complete authored definition and return the same public `TrainingStrategy` abstraction in an accepted state rather than exposing a second Trainer-facing wrapper?

This is partly resolved.

Already decided:

- The complete authored strategy is checked before Trainer receives it.
- Trainer receives one complete, accepted `TrainingStrategy`.
- The binding authority lives inside that strategy boundary.
- Trainer does not receive a second strategy-like object.
- A raw, unchecked authored definition cannot reach Trainer.

Still unresolved:

- The exact Python construction API that turns the authored definition into the accepted strategy.
- Whether that involves a builder, factory, class method, separate authored-definition type, or another mechanism.

We deliberately postponed that Python choice until we know the exact interface the intended Trainer needs.

So the original question contains a lot of material that is already settled. The remaining open question is simply:

> Once the Trainer-facing strategy surface is known, what Python API constructs and validates an accepted `TrainingStrategy`?

My verdict: **partly resolved, with one smaller implementation question legitimately remaining—but it is not a current question yet.**

Does that classification look right to you?

<a id="m022"></a>
## 022 User - 2026-08-24T20:35:48.208Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1370](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1370).

Okay, next one


<a id="m023"></a>
## 023 Assistant (commentary) - 2026-08-24T20:35:54.391Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1374](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1374).

Question 2 is about how maintained implementations are recognized as fulfilling contract vocabulary. I’m checking whether we already chose a mechanism or only settled the rules that mechanism must obey.

<a id="m024"></a>
## 024 Assistant (final_answer) - 2026-08-24T20:36:19.919Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1384](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1384).

### Question 2

> How does the contract system identify known library implementations and their conformance without automatic assembly, `hasattr()` discovery, family-name branches, or a stringly typed global registry?

This one is genuinely unresolved.

Already decided:

- Strategy authors explicitly select and wire implementations.
- The contract system knows how maintained library implementations conform.
- It validates those selections but does not automatically choose or assemble them.
- Custom implementations use an explicit conformance path.
- We do not want support inferred from method presence, model-family names, or scattered config branches.

Not decided:

- How a built-in implementation carries its conformance information in Python.
- Whether that uses an explicit catalog, attached descriptor, decorator, class metadata, typed registration object, or something else.
- How the contract system connects the selected implementation to that information.

The direction explicitly left those mechanisms as alternatives to test later. The current adapter registry is useful evidence for a typed registration pattern, but it is not an answer for the strategy contract system.

A clearer version of the remaining question is:

> How does a selected built-in implementation expose its contract conformance information?

My verdict: **still open, but premature until the contract exchanges tell us what conformance information an implementation actually needs to expose.**

<a id="m025"></a>
## 025 User - 2026-08-24T20:45:00.462Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1390](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1390).

check 3 4 and 5 next


<a id="m026"></a>
## 026 Assistant (commentary) - 2026-08-24T20:45:13.843Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1394](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1394).

I’ll check 3–5 together, but keep their conclusions separate. The main thing to distinguish is whether we settled the meaning and merely lack a Python representation, or whether the underlying behavior is still undecided.

<a id="m027"></a>
## 027 Assistant (commentary) - 2026-08-24T20:45:45.942Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1405](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1405).

Question 5 may already have a reusable answer in the metadata identity system, so I’m checking that code before calling it genuinely open. This is codebase exploration only; the current metadata types are evidence, not automatically the strategy-contract answer.

<a id="m028"></a>
## 028 Assistant (final_answer) - 2026-08-24T20:46:43.058Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1446](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1446).

### Question 3

> What typed representation lets the contract derive obligations, and which evidence is available during establishment versus materialization or preparation?

This is mostly resolved in meaning.

Already decided:

- The contract system derives obligations from the complete authored strategy.
- Obligations may cover participants, relationships, routes, readiness, capabilities, or the whole arrangement.
- Strategy authors do not write ordinary validators.
- Establishment uses facts already knowable from the authored selections.
- Loading and preparation later provide facts that could not be known earlier.
- Runtime evidence is typed and defined by the applicable contract.
- A live object, Boolean claim, or unstructured dictionary is insufficient.

Still unresolved:

- The actual Python types.
- Exactly which facts each individual exchange needs.

That remaining work cannot be answered once for the entire system; it must be filled in while defining each exchange.

Verdict: **the broad architectural question is resolved. Replace it with smaller exchange-local work rather than retain it in this form.**

---

### Question 4

> Who constructs the authority, and how is it prevented from accepting a raw or differently versioned strategy?

This is already resolved.

The documents say:

- Successful contract establishment creates the per-run authority.
- It seeds the authority with the accepted contract version, authored arrangement, derived obligations, identities, and initial revisions.
- Trainer only accepts the resulting established `TrainingStrategy`.
- A raw authored definition cannot be passed to Trainer.
- An already established authority cannot be seeded inconsistently.
- The authority is internal to the complete strategy boundary.

The remaining Python enforcement mechanism belongs to question 1: how the construction API prevents callers from bypassing establishment.

Verdict: **resolved and duplicated by question 1. Remove it as a separate open question.**

---

### Question 5

> What is the durable projection of `ParticipantRef`, distinct from its live typed runtime use?

This remains genuinely open, although several requirements are settled.

Already decided:

- `ParticipantRef` identifies one accepted participant incarnation within one run authority.
- It remains stable across loading, replacement, preparation, and route changes.
- Retirement does not erase its historical meaning.
- It cannot depend on Python object identity.
- Metadata and artifact provenance need a durable representation of it.

The metadata system provides useful precedent. It currently produces identifiers such as:

```text
run/<run-id>/model/<realization-key>/component/<component-key>
```

But that is not automatically the answer. A strategy participant is broader than a loaded model component, and the identifier may need to distinguish multiple incarnations of the same authored key.

The remaining question can be stated more clearly:

> How do metadata and history identify one authority-scoped participant incarnation after its live Python reference is gone?

Verdict: **still open and worth keeping, with clearer wording.**

<a id="m029"></a>
## 029 User - 2026-08-24T20:53:52.523Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1452](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1452).

OK so it's as I thought, most of them look like we already talked about and it's just the code level decisions which I don't want to talk about that much unless architecture or design related&#x20;


<a id="m030"></a>
## 030 Assistant (final_answer) - 2026-08-24T20:54:08.028Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1456](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1456).

Yes, your reading is right. The register is mixing two different things:

- Actual unresolved architecture or behavior
- Later Python/API implementation choices

That makes the architecture look less settled than it really is.

Using the stricter standard:

- Question 1: architecture settled; Python construction API later.
- Question 2: mostly a later implementation-design choice once the contracts are known.
- Question 3: architecture settled; individual types belong to each exchange’s implementation design.
- Question 4: fully resolved and duplicated.
- Question 5: genuinely design-related because durable identity affects metadata, history, persistence, and possibly resume.

We should continue reviewing the remaining items with this rule:

> Keep something in the open-question register only if different answers would materially change system behavior, ownership, or architecture.

Exact dataclass shapes, factory methods, generics, and similar choices can wait until we translate the design into an OpenSpec or implementation. They do not need to remain presented as unresolved architecture.

<a id="m031"></a>
## 031 User - 2026-08-24T20:58:43.407Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1462](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1462).

Okay what I want is for the architecture and design to be relatively well fleshed out. We have multiple notes and an ongoing design Document. A lot of these things are stuff we will see how to do when we're working on the actual code and what it needs to become, it's not something I'm qualified enough to go into detail preliminarily


<a id="m032"></a>
## 032 Assistant (final_answer) - 2026-08-24T20:59:05.423Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1466](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1466).

That’s the right level to aim for.

Before implementation, the design should clearly establish:

- What responsibilities exist
- Who owns each responsibility
- What state is authoritative
- Which lifecycle transitions are allowed
- What Trainer may rely on
- What strategies explicitly define
- What the contract validates
- What each exchange means, produces, and may change
- How failures, replacement, custom strategies, and extensions behave
- Which dependencies must point in which direction

We do not need to decide prematurely:

- Exact Python classes or dataclasses
- Factory versus constructor versus class method
- Decorators versus catalogs versus attached descriptors
- Generic type parameters
- Exact field names
- Exact identifier string formats
- Final file placement before the recurring code stages are visible

Those choices should emerge while translating the settled design into code. I can inspect the real dependency pressures, propose the most coherent implementation, and bring you only decisions that materially affect readability or architecture. You shouldn’t have to pre-design low-level machinery you have no useful basis for judging yet.

For the register, we should therefore:

- Remove questions whose architecture is already settled.
- Remove code-shape questions from the architecture register.
- Keep only questions where different answers would change ownership, behavior, lifecycle, or system boundaries.
- Add one simple statement that Python representations and placement will be derived during OpenSpec/implementation.

We can apply that standard while reviewing questions 6 onward, then clean the document once we know the complete result.

<a id="m033"></a>
## 033 User - 2026-08-24T21:05:38.558Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1472](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1472).

You can just move those to the code related notes or something, that's should be the clear distinction. Because the openspec will be made through -> design and architecture -> what the code is -> what it needs to become&#x20;


<a id="m034"></a>
## 034 Assistant (final_answer) - 2026-08-24T21:06:00.433Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1477](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1477).

Exactly. The process should be:

```text
architecture and design
        ↓
compare with what the code currently is
        ↓
work out what the code needs to become
        ↓
write the OpenSpec and implementation milestones
```

The documents already cover the first two parts:

- `strategy_system_direction.md`: architectural direction and settled decisions
- `strategy_contract_exchange_design.md`: detailed design still being worked out
- `strategy_system_inventory.md`: what the current code actually does
- `notes.md`: chronological discussion history

Code-level questions should be moved out of the architecture question register into a clearly separate implementation-mapping area. That later area can cover concrete types, construction APIs, registration mechanisms, file placement, and migration from current classes.

Let’s finish checking questions 6–14 first. Then we can reorganize them once:

- Keep genuinely unresolved architecture questions in the design document.
- Move deferred Python and code-organization questions into code-related notes.
- Remove resolved duplicates.
- Keep the handoff focused only on the actual current design frontier.

That will leave us with a clean path into the eventual OpenSpec instead of mixing all stages together.

<a id="m035"></a>
## 035 User - 2026-08-24T22:16:22.713Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1485](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1485).

Okay, go check the next batch please


<a id="m036"></a>
## 036 Assistant (commentary) - 2026-08-24T22:16:30.899Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1489](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1489).

I’ll check questions 6–8 next using the same test: does the answer change architecture or behavior, or is it only a later code representation choice? I’m continuing in exploration mode and won’t reorganize the files until we finish the full audit.

<a id="m037"></a>
## 037 Assistant (final_answer) - 2026-08-24T22:17:06.363Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1500](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1500).

### Question 6

> Can a retired authored key be reused later?

The important behavior is already settled:

- A retired `ParticipantRef` never becomes active again.
- Reusing the same authored key, if supported, creates a completely new participant incarnation with a new reference.
- History must continue distinguishing the retired participant from the new one.

Whether the first implementation permits key reuse at all is an implementation-policy choice. It may initially reject reuse without weakening the architecture.

Verdict: **resolved at the design level. Move initial key-reuse support to code-related notes.**

---

### Question 7

> What exact bound-state and access-view types replace `LoadedModelComponent.module`?

The architecture is already settled:

- Logical participant identity is separate from its current live object.
- The authoritative bound state is separate from prepared execution routes.
- Original, unwrapped, artifact, inspection, and similar handles are purpose-specific access views.
- Access views carry freshness rules.
- A participant may have state without being independently executable.
- The binding authority owns all of this state.

The remaining issue is the exact Python classes and fields.

Verdict: **code-level question. Move it out of the architecture register.**

---

### Question 8

> What proposal/result envelope supports atomic multi-participant and participant-plus-relationship transitions?

The required behavior is already settled:

- Producers submit typed proposals.
- Proposals name expected revisions.
- The authority evaluates the complete proposed next state.
- Related participant and relationship changes can be accepted atomically.
- Acceptance publishes the whole state change.
- Rejection publishes none of it.
- Results report changed revisions, invalidations, failures, and historical transition facts.

What remains is whether Python represents this using one generic transaction envelope, several operation-specific request types, a batch container, or another arrangement.

Verdict: **architecture settled; exact envelope is code design. Move it to code-related notes.**

So questions 6–8 do not need to remain open architecture questions.

<a id="m038"></a>
## 038 User - 2026-08-24T23:00:06.372Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1506](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1506).

okay, can you take care of the ones we looked at so far where needed (like when not positive to have them there in their current form) and then we'Ll check the remiaingv questions


<a id="m039"></a>
## 039 Assistant (commentary) - 2026-08-24T23:00:56.148Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1514](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1514).

I’m applying the distinction now. Questions 1–4 and 6–8 will leave the architecture register; their remaining Python and code-organization choices will go into a separate implementation-mapping document. Question 5 will remain, rewritten as the actual cross-run/history design issue. Questions 9–14 will stay untouched until we audit them.

<a id="m040"></a>
## 040 Assistant (commentary) - 2026-08-24T23:02:15.795Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1530](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1530).

The separation is now in place. I’m checking the edited documents, then I’ll record the audit result on the existing Bead so the next session does not reintroduce these code-shape items as architectural blockers.

<a id="m041"></a>
## 041 Assistant (final_answer) - 2026-08-24T23:02:38.853Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1544](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1544).

Done.

- Removed former questions 1–4 and 6–8 from the architecture register.
- Reworded former question 5 around the actual unresolved design issue: durable identity versus lineage.
- Left questions 9–14 untouched for our next audit.
- Moved deferred Python/API choices into [strategy_system_implementation_mapping.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_implementation_mapping.md:1).
- Reduced the discussion roadmap to a three-step [Current Design Frontier](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:1213).
- Shortened the compaction handoff.
- Recorded the audit in [notes.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:715) and bead `sd-scripts-syv`.

The remaining architecture register is at [line 1181](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:1181). `git diff --check` passes. No production code changed.

<a id="m042"></a>
## 042 User - 2026-08-24T23:09:24.684Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1550](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1550).

don't really want another living document. but it's fine to leave it for now. please check the remaining questions


<a id="m043"></a>
## 043 Assistant (commentary) - 2026-08-24T23:10:10.014Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1556](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1556).

Understood. I won’t grow the implementation-mapping file further while we audit this. I’ll check former items 9–14 against the settled exchange behavior and scenarios, then separate: already answered, still-needed design work, and later code mechanics.


> Compaction boundary at raw line 1560, 2026-08-24T23:10:54.451Z. The generated summary is omitted.

<a id="m044"></a>
## 044 Assistant (commentary) - 2026-08-24T23:10:58.103Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1567](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1567).

I’m checking the actual exchange, scenario, and direction text now. The key distinction will be whether each item still represents an architectural choice, or whether the architecture already constrains it and only the concrete implementation remains.

<a id="m045"></a>
## 045 Assistant (final_answer) - 2026-08-24T23:12:45.877Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1608](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1608).

After checking the remaining items against the direction, scenarios, and exchange design, only two represent current binding/preparation design work. One is future optimization design; the others are already settled or implementation choices.

- **9 — Scoped reads: partially unresolved, but far too broad as written.**  
  We already know the boundaries: Trainer receives exchange-specific projections; capabilities receive only contract-authorized views; observability receives facts/results rather than arbitrary live objects; persistence receives a coherent product-specific snapshot. The remaining current question is narrower: **what must a preparation snapshot expose, and how does it preserve a coherent multi-participant read?** That belongs beside the existing snapshot/read-isolation section, not as a question covering every consumer. See [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:230) and [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:718).

- **10 — In-place preparation: architecturally settled.**  
  The important behavior is already recorded: affected guarantees are withdrawn before mutation; failure cannot silently restore them; and the attempt needs an authority-recognized identity distinct from its source snapshot. Whether that becomes a lease, token, staged transition, or something else is code design. Remove this from the architectural question register. See [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:1010).

- **11 — Publication and recovery: genuinely unresolved and important.**  
  This is the real gap. Preparation can produce both authority-owned route changes and Trainer-owned optimizer/backend runtime. We still need one clear commit boundary so a failure cannot leave one side published and the other side absent. This is architectural behavior, even though the eventual mechanism will be code. Keep it, phrased plainly:  
  **When does a prepared result become committed, and how is the run kept coherent if publishing either the strategy routes or Trainer runtime fails?**

- **12 — Rank agreement: architecturally settled.**  
  The direction already says there is one logical run authority and that ranks/backend replicas are not separate authorities. Consequently, a distributed preparation transition succeeds as one coordinated transition or does not become accepted. The barrier, broadcast, collective, and rank-zero mechanics belong to distributed-runtime implementation. Remove this from the architectural register.

- **13 — Optimization preparation: unresolved later design, not a current binding question.**  
  The settled part is enough for binding/preparation: optimizer and scheduler objects may participate in a joint backend operation, but remain Trainer-owned and outside the binding authority. Their exact request/result cannot be decided coherently until the optimization exchange is designed. Move this concern to the existing Optimization Exchange section rather than either the current question register or code notes. See [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:1104).

- **14 — Obligation evolution: effectively settled by the existing architecture.**  
  A run authority is established under one accepted contract/version. An explicit arrangement amendment may add or retire participants, relationships, routes, or capabilities and therefore re-evaluate obligations **under that same contract**. Changing the Trainer contract or activating a different contract extension is not an ordinary runtime amendment; it requires a newly established accepted strategy/authority. Supporting live contract-version upgrades could be introduced later as a distinct feature, but should not be assumed now. Remove this question and record that rule.

So the cleaned current architectural frontier would contain:

1. Former item 5: durable identity and lineage.
2. A narrowed item 9: preparation snapshot contents and coherent read isolation.
3. Item 11: preparation commit, publication, and recovery.

Item 13 waits under optimization design. Items 10 and 12 become implementation concerns, and item 14 becomes a settled rule. I haven’t edited the documents yet.

<a id="m046"></a>
## 046 User - 2026-08-24T23:13:29.722Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1614](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1614).

so what should happen to them? are they fine as they are


<a id="m047"></a>
## 047 Assistant (final_answer) - 2026-08-24T23:13:48.050Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1618](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1618).

No, they are not fine as they are. Leaving them there would falsely imply that all six are equally unresolved architectural questions.

They should be cleaned up as follows:

- **9:** Replace with the narrower preparation-snapshot/read-isolation question.
- **10:** Remove from the open register. Preserve the settled in-place-mutation rules; defer the exact token/lease/API.
- **11:** Keep, but rewrite plainly around the preparation commit boundary and failure recovery.
- **12:** Remove. The one-logical-authority rule is settled; rank coordination is runtime implementation.
- **13:** Remove from the current register. The Optimization Exchange placeholder already holds this later design work.
- **14:** Remove and record the settled rule: obligations may be recalculated through amendments under the run’s accepted contract; changing contract/version requires new establishment.

The related `OPEN` sections in the body should also be adjusted so they do not reintroduce the removed questions under different wording.

Afterward, the active register should contain only:

1. Durable identity versus lineage.
2. Preparation snapshot contents and coherent read isolation.
3. Preparation publication/commit and recovery.

That would accurately show what architectural work is actually left.

<a id="m048"></a>
## 048 User - 2026-08-24T23:17:06.633Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1624](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1624).

okay, can you do that please


<a id="m049"></a>
## 049 Assistant (commentary) - 2026-08-24T23:17:13.359Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1628](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1628).

I’ll clean the exchange document itself—both the register and the corresponding `OPEN` wording—so the removed items do not remain ambiguously open elsewhere. I’m using the OpenSpec exploration guidance because this is still design clarification, and the Beads workflow to keep the existing architecture issue’s handoff accurate.

<a id="m050"></a>
## 050 Assistant (commentary) - 2026-08-24T23:18:09.426Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1641](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1641).

The cleanup has one important consequence beyond shortening the register: the body currently labels the in-place protocol, rank agreement, and optimization joining as one unresolved preparation problem. I’m separating the settled invariants from the single remaining commit/recovery question so the document tells the same story everywhere.

<a id="m051"></a>
## 051 Assistant (commentary) - 2026-08-24T23:19:45.268Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1670](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1670).

The document is now internally aligned: only preparation reads and prepared-state commit/recovery remain marked `OPEN`; the former representation questions are explicitly labeled deferred code design. I’m recording the audit outcome on the existing bead, without creating another tracking document or issue.

<a id="m052"></a>
## 052 Assistant (final_answer) - 2026-08-24T23:20:08.466Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1680](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1680).

The register now contains only three genuine architecture questions:

- Durable identity versus lineage.
- Preparation snapshot/read isolation.
- Prepared-state commit and recovery.

I also:

- Converted in-place preparation and rank coordination into settled requirements.
- Moved optimizer preparation to the later optimization exchange.
- Recorded that one contract/version governs one run authority.
- Relabeled earlier type/envelope questions as deferred code design.
- Updated the chronological [notes](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:750) and bead `sd-scripts-syv`.

The main changes are in [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:747). Markdown fences are balanced and no trailing whitespace was found. No production code or OpenSpec artifacts changed.

<a id="m053"></a>
## 053 User - 2026-08-24T23:24:49.749Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1686](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1686).

Alright, now let's progress on the remaining things that genuinely need to settle before getting into the code


<a id="m054"></a>
## 054 Assistant (commentary) - 2026-08-24T23:25:11.321Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1691](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1691).

We should take these in dependency order, starting with the preparation read boundary. It is the input to preparation; the commit/recovery rule cannot be designed coherently until we know exactly what preparation is allowed to receive and rely on. I’m using the OpenSpec exploration guidance to keep this at architecture level, grounded in the recorded scenarios rather than inventing Python APIs.

<a id="m055"></a>
## 055 Assistant (commentary) - 2026-08-24T23:25:21.799Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1697](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1697).

The recorded scenarios constrain most of the preparation input already, but I’m checking the current preparation paths once more for anything the inventory may have compressed—especially joint model/optimizer/dataloader preparation and original/unwrapped access. I’m using the codebase knowledge-graph skill for that source-grounding; current code remains evidence, not the target API.

<a id="m056"></a>
## 056 Assistant (final_answer) - 2026-08-24T23:27:03.263Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1750](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1750).

I think the preparation-read question can now be settled architecturally.

Preparation should not “look things up” from the strategy. The strategy’s authority should produce one complete, single-use preparation package from one coherent accepted state:

```text
binding authority
  -> coherent preparation package
       exact participants and source objects
       meaning of each source object
       target execution routes
       preparation constraints
       joint-preparation groups
       mutation permission
       source revisions/dependencies
  -> Trainer-owned backend preparation
```

For each prepared participant, the package says:

- Which accepted participant it is.
- Which current representation the backend may consume—such as its unprepared binding, current execution route, or specifically permitted original/unwrapped view.
- Which execution route the result will replace or establish.
- Device, dtype, wrapping, compilation, and joint-preparation constraints.
- Whether the backend promises replacement-only behavior or may mutate the supplied object.
- The exact revisions on which this instruction depends.

It does not expose:

- A general participant dictionary.
- Arbitrary strategy anatomy or role-based lookup.
- Artifact/persistence views unless preparation explicitly requires that access meaning.
- Trainer-owned optimizer, scheduler, or backend state as though those belonged to the strategy authority.
- An indefinitely reusable snapshot.

Coherence does not require locking the whole strategy during slow backend work:

- The authority creates the complete multi-participant package atomically.
- Replacement-only preparation may run optimistically. Before acceptance, the authority checks that its dependencies are still current; otherwise the result is discarded.
- In-place-capable preparation first marks the affected state as being prepared and withdraws its freshness guarantees, using the already-settled attempt identity.
- Relationship state only appears through the constraints and dependencies it imposes. The backend does not receive unrestricted relationship access.

The current code supports this boundary. Fine-tuning and adapter preparation currently inspect the whole Trainer to choose denoisers, text encoders, adapters, optimizers, and schedulers; DeepSpeed then creates a separate composite synchronization handle. It also shows that “participants to prepare” cannot mean merely “trainables”—a frozen denoiser may still need movement or casting for execution. See [adapter_mode.py](/home/imi/Projects/sd-scripts/library/training/modes/adapter_mode.py:243) and [finetune_mode.py](/home/imi/Projects/sd-scripts/library/training/modes/finetune_mode.py:234). The graph index had no recorded coverage gaps for the inspected preparation paths.

This closes former item 9 at the architecture level. What remains later is only how that package is represented in Python.

The next dependency is item 11: exactly how that prepared package and its result cross one commit boundary without route state and Trainer runtime diverging. I can record this decision and proceed directly into that problem.

<a id="m057"></a>
## 057 User - 2026-08-25T00:32:02.592Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1756](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1756).

3rd party feedback:

Treat the package as one complete attempt-scoped preparation job derived atomically from coherent accepted state, not necessarily as one backend call or literally single-use object. Preserve the important code-derived facts that preparation membership is broader than trainability and that backend composites such as DeepSpeed handles do not replace participant identity. For prepared-state commit/recovery, first determine how much fallible work can be forced before publication and whether final publication can be reduced to non-failing state replacement; only design staged commit/recovery for failure modes that genuinely remain. Keep the simple replacement-only path distinct from the genuinely destructive in-place-mutation path, and do not let joint optimizer preparation prematurely settle the later optimization contract. &#x20;


<a id="m058"></a>
## 058 Assistant (commentary) - 2026-08-25T00:32:46.596Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1762](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1762).

That feedback is useful and catches one overstatement in my wording: the preparation input is one coherent attempt-scoped job, but it need not be one literal object, one backend call, or disposable after a single method invocation. I’m applying the OpenSpec exploration guidance again while testing the proposed “make publication non-failing” rule against the settled preparation failure model.

<a id="m059"></a>
## 059 Assistant (final_answer) - 2026-08-25T00:33:33.927Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1769](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1769).

The feedback improves the design. I agree with it, with one clarification: “non-failing publication” means publication performs no expected fallible work—not that Python can survive process termination, memory corruption, or similar catastrophic failure.

My earlier “single-use package” wording was too concrete. The correct semantic unit is one preparation job:

- It is scoped to one preparation attempt.
- Its strategy-owned inputs are derived atomically from one coherent authority state.
- It may contain several ordered or joint operations.
- It may require several backend calls.
- Its eventual Python representation is undecided.

The current preparation paths reinforce this: preparation includes more than trainable objects, and DeepSpeed’s composite handle is backend coordination state—not a replacement participant.

For commit and recovery, the simplest coherent rule is:

```text
fallible work
  backend calls
  wrapping / compilation / movement
  component-specific preparation
  candidate construction
  cross-rank agreement
  freshness and contract validation
          |
          v
fully ready unpublished candidate
          |
          v
non-failing logical publication
  authority route state
  + Trainer-owned runtime state
          |
          v
observation / history / callbacks
```

Publication must therefore contain no backend calls, conversions, hooks, user callbacks, validation, or other work expected to fail. It only installs already-created, already-validated state and makes that new state observable. Whether code realizes this with guarded state swaps, one generation switch, or another representation comes later.

The two preparation paths remain importantly different:

- **Replacement-only:** Existing routes remain valid throughout preparation. Any failure before publication discards the candidate. No staged recovery protocol is needed.
- **In-place mutation:** Affected guarantees are withdrawn before mutation. If preparation fails, those routes remain invalid until explicitly repaired, replaced, or rematerialized. This path needs its existing preparing/attempt state, but that does not justify imposing staged transactions on ordinary replacement preparation.

If the process or a distributed rank dies during the final internal publication, there is no continuing run whose two sides must be reconciled; the run aborts and the separate resume/restoration contract applies. We should not complicate normal preparation in an attempt to recover an already-failed process.

Jointly prepared optimizer or scheduler values can be treated only as opaque Trainer-owned candidate state here. Their contents, construction, and lifecycle wait for the optimization exchange.

This appears sufficient to settle both remaining preparation questions:

- Former item 9 becomes the attempt-scoped coherent preparation-job rule.
- Former item 11 becomes “all expected fallibility before publication; publication is an unobservable, non-failing logical state replacement,” with the destructive in-place failure state handled separately.

After recording that, durable identity versus lineage would be the only active pre-code question left in the register.

<a id="m060"></a>
## 060 User - 2026-08-25T00:53:57.848Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1775](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1775).

How about taking care of that? unless there's something else we need to consider before


<a id="m061"></a>
## 061 Assistant (commentary) - 2026-08-25T00:54:16.181Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1779](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1779).

There is one caveat worth preserving, but it does not block the decision: destructive in-place preparation necessarily publishes a preliminary “not currently valid/preparing” state before fallible work. The non-failing-publication rule applies to the final prepared result; it does not erase that safety transition. With that distinction explicit, the scenarios are covered, so I’ll record both preparation decisions and remove them from the active register. I’m using the exploration and Beads guidance to update only the design record and existing issue history.

<a id="m062"></a>
## 062 Assistant (commentary) - 2026-08-25T00:54:48.682Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1786](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1786).

The final pressure check did not expose another preparation-level choice. I am also keeping the larger sequence honest: settling these two questions will leave only durable identity in the current register, but it will not mean all pre-code architecture is finished—we still need the binding/preparation scenario pass and then the optimization and step exchanges.

<a id="m063"></a>
## 063 Assistant (commentary) - 2026-08-25T00:56:04.593Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1797](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1797).

The design record now reflects the corrected semantic unit and the simplified publication rule. I’m adding the concise outcome to the existing bead; no new issue or document is being created.

<a id="m064"></a>
## 064 Assistant (final_answer) - 2026-08-25T00:56:30.797Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1807](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1807).

Recorded and reconciled throughout the design.

Key updates:

- Preparation is now an attempt-scoped coherent job, potentially spanning multiple backend calls.
- All expected-fallible work precedes final publication.
- Replacement-only and destructive in-place preparation remain separate paths.
- Joint optimizer preparation remains deferred to the optimization exchange.
- Former questions 9 and 11 are removed from the register.

See [exchange design](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:768), [publication decision](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:1134), and [chronological notes](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:778). The bead was also updated.

Durable identity versus lineage is now the register’s only active item. After that, we still need to pressure-test the completed binding/preparation design and design the optimization and step exchanges before moving into code. Markdown checks passed; no production code or OpenSpec artifacts changed.

<a id="m065"></a>
## 065 User - 2026-08-25T00:59:19.562Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1813](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1813).

after what?


<a id="m066"></a>
## 066 Assistant (final_answer) - 2026-08-25T00:59:26.962Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1817](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1817).

After settling **durable identity versus lineage**.

Then the sequence is:

1. Pressure-test the completed binding/preparation design.
2. Settle optimization ownership and its exchange.
3. Settle the training-step exchange.
4. Map current code to the target architecture.
5. Create the governing OpenSpec.
6. Begin implementation.

<a id="m067"></a>
## 067 User - 2026-08-25T01:25:17.535Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1823](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1823).

what does durable identity versus lineage mean


<a id="m068"></a>
## 068 Assistant (final_answer) - 2026-08-25T01:25:52.192Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1827](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1827).

It asks: **when do we say this is still the same participant, and when do we say it is a new participant that came from an earlier one?**

For example:

- A denoiser is wrapped by Accelerate: same participant, new execution representation.
- Its weights are replaced with another compatible checkpoint during the same run: potentially the same participant, with a new binding revision.
- A student is copied to create a teacher: two different participants with shared lineage.
- A checkpoint from run A starts run B: run B’s denoiser is a new participant derived from run A’s artifact—not literally the same runtime participant.
- An incompatible replacement retires the old participant and declares a new one; the new one may record that it succeeded or derived from the old one.

So:

```text
identity
  “this is the same independently tracked participant”

lineage
  “this different participant/state/artifact came from that earlier one”
```

The unresolved part concerns durable records and resume:

- What identifier should history and metadata store for one run participant?
- Does resuming the exact same run restore the same participant incarnation, or establish a new incarnation linked to the saved one?
- How do artifacts refer to the participant state they captured?
- When another run loads that artifact, how do we preserve provenance without falsely claiming it is the same participant?

Within one live authority, the basic rule is already clear: `ParticipantRef` stays stable through materialization, preparation, and compatible replacement; binding and route revisions record its changing state. Across authorities or runs, reusing an authored address such as `model.denoiser` is not enough to mean “the same participant.” That boundary is what we still need to settle.

<a id="m069"></a>
## 069 User - 2026-08-25T01:35:23.895Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1833](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1833).

well, how about you check the code for what exists today, and then based on our existing design documents we figure it out?


<a id="m070"></a>
## 070 Assistant (commentary) - 2026-08-25T01:35:35.071Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1837](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1837).

Yes—that is the right way to settle it. I’ll trace today’s component identity through loading, metadata/artifact provenance, checkpoint publication, and runtime resume, then compare those meanings with the already-settled `ParticipantKey`/`ParticipantRef` rules. I’m using the codebase graph for structural discovery and OpenSpec exploration guidance so current code remains evidence rather than silently becoming the target.

<a id="m071"></a>
## 071 Assistant (commentary) - 2026-08-25T01:36:26.120Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1891](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1891).

The current metadata system already has a strong precedent: model and component IDs are qualified by the run, while artifacts point back to the accepted realization. I’m now checking whether the active metadata design already distinguishes “same source across runs” from “same run-scoped realization,” because we should reuse that distinction rather than invent a parallel identity concept.

<a id="m072"></a>
## 072 Assistant (commentary) - 2026-08-25T01:38:16.958Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1965](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1965).

The important gap is now visible: today’s metadata component ID is already run-qualified, but today’s resume checkpoint does not save or restore that identity—it restores model/optimizer state and epoch/step, while a newly constructed Trainer generates a new random session ID. I’m checking the direction’s exact “resume this execution” language before deciding whether a future full resume should preserve the incarnation or explicitly create a successor.

<a id="m073"></a>
## 073 Assistant (final_answer) - 2026-08-25T01:40:12.423Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1983](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1983).

The code and existing metadata design point to a fairly clear answer.

What exists today:

- A live component is identified only by its family-local `key`; loading copies that key from `LoadedModelComponentSpec` into `LoadedModelComponent`. There is no runtime incarnation identity or binding revision yet. See [components.py](/home/imi/Projects/sd-scripts/library/models/components.py:25).
- Metadata creates a run-qualified realization ID:
  `run/<session>/model/training-target`.
- Component metadata then adds the component key:
  `.../component/<component-key>`. See [model.py](/home/imi/Projects/sd-scripts/library/metadata/dataclasses/model.py:12).
- Trainer creates that metadata once from a newly generated random `session_id`. See [trainer.py](/home/imi/Projects/sd-scripts/library/training/runners/trainer.py:199) and [trainer.py](/home/imi/Projects/sd-scripts/library/training/runners/trainer.py:737).
- Artifacts point back to that run realization with a `derived_from` relationship.
- Runtime resume restores Accelerator state, adapter/full-model state, optimizer state, and epoch/step. It does not restore the run, realization, or component identities. See [checkpointing.py](/home/imi/Projects/sd-scripts/library/training/checkpointing.py:45) and [state_io.py](/home/imi/Projects/sd-scripts/library/adapters/shared/state_io.py:72).

So today, restarting Trainer for a resume generates a new run identity even though execution state continues. That is a missing identity capability, not a target rule we should preserve.

The active metadata design already establishes several useful distinctions:

- A run realization remains stable while its composition receives new revisions.
- Loading the same cataloged source in another run creates a new run realization.
- Device movement, wrappers, and other transient execution changes do not create new source/model identity.
- Saved training output receives artifact/model-revision identity and lineage to its source and producing run.
- An artifact is not the run realization that produced it.

That leads to this target rule:

> A participant remains the same incarnation exactly while it belongs to the same logical run authority and retains the same authority-established participant reference.

Its durable identity therefore needs the meaning of:

```text
logical run / authority identity
  + authored participant address
  + participant incarnation identity
```

The authored address remains useful and stable, but it cannot identify the incarnation alone. Binding, relationship, and route revisions describe changes to that incarnation; they are not new participant identities.

Examples:

- Materialization, wrapping, casting, preparation, and compatible replacement: same participant incarnation, newer state revisions.
- Teacher and student loaded from the same checkpoint: different participant incarnations with shared source provenance.
- Retire and later redeclare the same authored key: new incarnation; the old one never revives.
- Incompatible replacement: new incarnation, optionally related to the retired participant by typed succession or derivation history.
- Artifact produced from a participant: new artifact/model revision linked to the captured participant state.
- New training run initialized from that artifact: new participant incarnation linked through artifact/model lineage.

Resume needs one explicit distinction:

- **Exact runtime resume** restores the logical run/authority identity, participant incarnations, accepted arrangement, and relevant revisions. New Python objects and a new process do not make them new logical participants.
- **Starting a new run from saved state or weights** creates a new run authority and new participant incarnations, with `resumed_from`/`derived_from` lineage.

A resumed process should still receive a new execution-session identity for operational observation, while retaining the restored logical training-run identity. Today’s `session_id` conflates those two meanings.

If a snapshot does not contain enough authority and participant identity state to perform the first operation, it cannot honestly claim identity-preserving exact resume. Today’s checkpoints fall into that category.

This appears to settle durable identity versus lineage without choosing UUID classes, string formats, or metadata record names. It also exposes one future code requirement: split logical training-run identity from per-process execution-session identity and include authority identity in the runtime restoration contract.

<a id="m074"></a>
## 074 User - 2026-08-25T02:50:15.401Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1989](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1989).

Well, the work order was stuff -> metadata -> going into models/strategy/etc planning, so metadata might not be as entwined with everything yet. Anyways can you record what you said please? Context is gonna summarize any time now


<a id="m075"></a>
## 075 Assistant (commentary) - 2026-08-25T02:50:39.007Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1994](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1994).

I’ll record it now, with your ordering caveat made explicit: the existing metadata system is evidence for durable qualification and lineage distinctions, not the owner or normative shape of participant identity. The binding authority remains authoritative; metadata only observes its durable projection. I’m using the exploration and Beads guidance to update the existing design, notes, and issue handoff only.


> Compaction boundary at raw line 1998, 2026-08-25T02:51:33.178Z. The generated summary is omitted.

<a id="m076"></a>
## 076 Assistant (commentary) - 2026-08-25T02:52:28.920Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2014](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2014).

The decision is now in the exchange design and chronological notes. I’m checking that the active register and handoff actually reflect it, then I’ll mirror the conclusion onto the existing bead so it survives independently of this context.

<a id="m077"></a>
## 077 Assistant (final_answer) - 2026-08-25T02:52:58.075Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2024](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2024).

Recorded in:

- [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:704) — normative working decision, EX-023/EX-024, and updated frontier.
- [notes.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:808) — code evidence and chronological rationale.
- Bead `sd-scripts-syv` — durable handoff comment added.

The metadata caveat is explicit: it provides useful precedent for qualification and lineage, but predates this architecture. The binding authority owns participant identity; metadata only records its durable projection.

The binding/preparation architecture register now has no unresolved items. Next is pressure-testing these semantics against the scenarios and current code before moving to optimization exchange design. Documentation passed `git diff --check`; no code or tests were changed.

<a id="m078"></a>
## 078 User - 2026-08-25T02:54:24.417Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2030](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2030).

context was summarized, please check the past 2-3 turns in session 01a026d5-a88b-7843-a1c0-1b51047c8182 to make sure you got the stuff right. and yeah what i meant was to answer your finding about metadata not being reflected in the trainer/Tcheckpointing properly. metadata predates the current direction, but the current direction directly came up as a result of it. metadata->model metadata->strategy/trainer rework direction


<a id="m079"></a>
## 079 Assistant (commentary) - 2026-08-25T02:54:38.053Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2034](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2034).

I’ll verify the actual last few exchanges from that session, then compare them with what I recorded. Your clarification changes the emphasis: metadata is not merely older, incidental evidence—it is the preceding architectural layer that exposed the need for the model → strategy/Trainer redesign, even though today’s Trainer/checkpointing has not yet integrated its identity model properly.

I’m using the OpenSpec exploration guidance for the design correction and Beads so the durable handoff stays consistent.

<a id="m080"></a>
## 080 Assistant (commentary) - 2026-08-25T02:55:23.056Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2049](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2049).

I checked the actual exchanges. The identity/resume decision itself was recorded correctly, but I slightly misread your caveat.

You were not saying “metadata is merely older evidence, so keep it at arm’s length.” You were explaining the integration gap I found: the metadata and model-metadata work came first, exposed what the existing model/strategy/Trainer boundaries could not express, and directly led to this redesign. Therefore today’s Trainer/checkpointing not restoring those identities is unfinished architectural integration—not evidence that metadata’s semantics are incidental. I’m correcting that emphasis now.

<a id="m081"></a>
## 081 Assistant (final_answer) - 2026-08-25T02:56:33.012Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2076](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2076).

I checked the actual final exchanges. The identity/resume conclusion was right; the interpretation of your caveat was slightly wrong.

The correct sequence is:

```text
supporting groundwork
→ metadata
→ model metadata
→ model/strategy/Trainer redesign direction
```

Model metadata exposed the need for intentional component boundaries, durable identity, provenance, and restoration. Therefore, Trainer/checkpointing not restoring those identities today is an unfinished integration that motivated this redesign—not evidence that metadata is merely incidental.

The direction already says this explicitly in [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:55). I corrected the emphasis in:

- [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:739)
- [notes.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:841)
- Bead `sd-scripts-syv`

The remaining distinction still holds: metadata’s requirements directly shape the architecture, while the future binding authority establishes runtime participant identity and supplies its durable projection to metadata, persistence, and exact-resume restoration. The current metadata string format does not have to dictate the eventual runtime representation.

<a id="m082"></a>
## 082 User - 2026-08-25T03:06:40.563Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2082](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2082).

well yeah we never really got to model metadata properly, when getting into it we got into strategy and then that led us to realize it wasn't as we wanted it to be. we spent the last ton of work on metadata and then that just exposed a lot of stuff about the repo


<a id="m083"></a>
## 083 Assistant (commentary) - 2026-08-25T03:06:53.379Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2086](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2086).

Right—that is more precise again. We did substantial general metadata work; then, when trying to proceed into model metadata properly, the existing strategy/Trainer/model boundaries proved too unclear to support it. That investigation caused the pivot into this redesign before model metadata itself was properly completed.

So “model metadata directly caused the redesign” is directionally right but chronologically too compressed. I’ll correct the durable wording and bead handoff so they do not imply a mature model-metadata layer already existed. I’m using the exploration and Beads guidance for that record correction only.

<a id="m084"></a>
## 084 Assistant (final_answer) - 2026-08-25T03:07:50.000Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2109](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2109).

Yes—that is the accurate history:

```text
substantial general metadata work
→ attempt to proceed into richer model metadata
→ strategy/model/Trainer boundaries proved unsuitable
→ pivot into the current architecture redesign
→ return to model metadata on a coherent foundation later
```

So model metadata was the pressure test, not a mature subsystem that Trainer simply failed to integrate. The active model-metadata OpenSpec also reflects this: its planning exists, but implementation barely began before the architectural pivot.

I corrected that chronology in the [exchange design](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:739), [notes](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:841), and bead `sd-scripts-syv`. The identity requirements remain relevant because they were among the things that exposed the underlying problem.

<a id="m085"></a>
## 085 User - 2026-08-25T03:13:34.727Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2115](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2115).

And so, where does that leave us? I think you said this but idk if anything is different



"It asks: **when do we say this is still the same participant, and when do we say it is a new participant that came from an earlier one?**

For example:

- A denoiser is wrapped by Accelerate: same participant, new execution representation.
- Its weights are replaced with another compatible checkpoint during the same run: potentially the same participant, with a new binding revision.
- A student is copied to create a teacher: two different participants with shared lineage.
- A checkpoint from run A starts run B: run B’s denoiser is a new participant derived from run A’s artifact—not literally the same runtime participant.
- An incompatible replacement retires the old participant and declares a new one; the new one may record that it succeeded or derived from the old one.

So:
```kotlin
identity
  “this is the same independently tracked participant”

lineage
  “this different participant/state/artifact came from that earlier one”
```

The unresolved part concerns durable records and resume:

- What identifier should history and metadata store for one run participant?
- Does resuming the exact same run restore the same participant incarnation, or establish a new incarnation linked to the saved one?
- How do artifacts refer to the participant state they captured?
- When another run loads that artifact, how do we preserve provenance without falsely claiming it is the same participant?

Within one live authority, the basic rule is already clear: `ParticipantRef` stays stable through materialization, preparation, and compatible replacement; binding and route revisions record its changing state. Across authorities or runs, reusing an authored address such as `model.denoiser` is not enough to mean “the same participant.” That boundary is what we still need to settle."


<a id="m086"></a>
## 086 Assistant (commentary) - 2026-08-25T03:13:41.516Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2119](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2119).

I’m reconciling that earlier explanation against the decision we just recorded. I’m using the exploration guidance because this is still architecture clarification, but I won’t change anything unless the comparison reveals a real mismatch.

<a id="m087"></a>
## 087 Assistant (final_answer) - 2026-08-25T03:14:07.745Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2129](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2129).

The quoted explanation was the question before we investigated the code. It is now settled, and the metadata chronology does not change the answer—it explains why current Trainer/checkpointing cannot implement it yet.

The settled rule is:

> A participant remains the same incarnation while it belongs to the same logical run authority and retains the same authority-established `ParticipantRef`.

Therefore:

- Accelerate wrapping, casting, materialization, preparation, route rebinding, and compatible replacement preserve the participant.
- “Compatible” means the binding authority accepts that the replacement still fulfills the same participant meaning. Otherwise it is a new incarnation.
- Teacher and student are different participants even if they share a checkpoint.
- Retiring and redeclaring the same authored key creates a new incarnation.
- Starting another run from an artifact creates new participants under a new authority.
- Lineage connects those different participants or artifacts without claiming they are identical.

The formerly unresolved durable questions now have answers:

- History and metadata need a durable projection containing the logical run/authority identity, authored participant address, and incarnation discriminator. Exact dataclass and string formats remain code design.
- Exact resume preserves the same authority and participant incarnations—but only if the checkpoint persists and restores their identity, arrangement, and relevant revisions.
- A restarted process gets a new execution-session identity without necessarily becoming a new logical training run.
- If saved state lacks those identity facts, loading it establishes a new run/incarnations with lineage. Current checkpoints lack those facts.
- An artifact gets its own artifact/model-revision identity and refers to the participant state and accepted authority snapshot it captured. It is not itself the participant.

So the architecture question is closed. What remains is implementing a future checkpoint/restoration contract that can actually uphold it. The metadata work exposed that requirement; it did not already solve it inside today’s Trainer.

This is recorded in [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:704). Next is the scenario pressure-test of the completed binding/preparation design, followed by optimization and step exchanges.

<a id="m088"></a>
## 088 User - 2026-08-25T03:42:31.736Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2135](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2135).

Is this right?



Feedback said: Almost. I’d change one part: **“while it belongs to the same logical run authority”** is a little misleading, because after retirement it no longer belongs to the *current arrangement*, but it is still the same historical participant incarnation. The docs explicitly preserve the meaning of a retired reference for history/provenance.

The cleaner invariant is:

> **A participant remains the same incarnation for as long as it is identified by the same authority-established&#x20;****`ParticipantRef`****.**

Then separately:

> **Materialization, compatible replacement, preparation, and route rebinding do not change that incarnation; retirement closes it to further current-state use but does not create a different historical identity.**

If you want the authority scope explicit:

> **Within one run authority, a&#x20;****`ParticipantRef`****&#x20;identifies one participant incarnation. The incarnation remains the same across changes to its bound state and execution representation until it is retired.**

I think that’s less circular than “same incarnation while it retains the same ref,” while still making the core rule obvious:
```text
same ParticipantRef
    = same participant incarnation

new ParticipantRef
    = new incarnation
```

The `ParticipantKey` is then just the authored address; it’s not what determines whether two runtime incarnations are the same.


<a id="m089"></a>
## 089 Assistant (commentary) - 2026-08-25T03:42:51.094Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2139](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2139).

I’m checking this against the settled retirement and cross-authority rules. The feedback is pointing at a real wording flaw, not changing the underlying design; I’m using the exploration guidance to separate the identity invariant from the participant’s live lifecycle state.

<a id="m090"></a>
## 090 Assistant (final_answer) - 2026-08-25T03:43:16.874Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2145](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2145).

Yes. The feedback catches a genuine wording problem in our recorded sentence.

Retirement changes whether a participant can still participate in current state; it does not change or erase its identity. I would use this invariant:

> Within one logical run authority, each `ParticipantRef` permanently identifies exactly one participant incarnation. The reference is never reassigned or revived. State changes preserve it; retirement closes it to further current use while preserving its historical and provenance meaning.

Therefore:

```text
same authority-qualified ParticipantRef
    = same incarnation, even after retirement

new ParticipantRef
    = new incarnation
```

One small correction to the feedback: “until it is retired” could still imply that identity ends at retirement. The participant’s active lifecycle ends, but its historical identity remains.

The authority qualification also matters. If `ParticipantRef` does not intrinsically include its authority, then the durable identity is effectively:

```text
(run authority, ParticipantRef)
```

Two authorities cannot establish the same logical incarnation merely by producing equivalent local reference values.

And yes, `ParticipantKey` is only the authored address. It helps establish and discuss the participant, but does not distinguish separate runtime incarnations created from the same declaration.

So this is a wording refinement, not a change to the architecture. The current design sentence should be corrected accordingly.

<a id="m091"></a>
## 091 User - 2026-08-25T03:57:18.167Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2151](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2151).

take care of that please and then tell me what we'Ll go through tomorrow


<a id="m092"></a>
## 092 Assistant (commentary) - 2026-08-25T04:02:30.491Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2155](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2155).

I’ll correct the identity invariant everywhere it currently implies retirement might end the incarnation, update the existing bead, and then give you the exact starting point for tomorrow. I’m using the exploration guidance for the design record and Beads for the persistent handoff; no production code or OpenSpec implementation will change.

<a id="m093"></a>
## 093 Assistant (commentary) - 2026-08-25T04:03:14.947Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2174](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2174).

The correction is in place: retirement now ends current usability, not identity. I’m doing the consistency check and recording tomorrow’s precise continuation point on the bead.

<a id="m094"></a>
## 094 Assistant (final_answer) - 2026-08-25T04:03:44.913Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 2184](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:2184).

Taken care of.

The corrected invariant now says that each authority-established `ParticipantRef` permanently identifies one incarnation. Retirement closes current use but preserves historical/provenance identity.

Updated in:

- [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:706)
- Decision register [EX-023](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:1322)
- [notes.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:820)
- Bead `sd-scripts-syv`

Tomorrow we start with the final binding/preparation pressure test. We’ll take the six recorded target scenarios—ordinary fine-tuning, deferred loading, adapter attachment, distributed preparation, replacement/invalidation, and a compound research strategy—and walk each through:

```text
authored
→ contract-validated
→ authority established
→ materialized
→ prepared
→ published
→ executed/persisted/resumed
```

We’ll check that participant identity, relationships, replacement, failure behavior, preparation, consumer access, artifacts, and resume all remain coherent. This is a consistency pass, not reopening the decisions or choosing Python class names.

If that exposes no architectural gap, we move to:

1. Optimization ownership and exchange.
2. Training-step exchange.
3. Current-code-to-target mapping.
4. The governing OpenSpec.

No production code or OpenSpec implementation changed tonight.

