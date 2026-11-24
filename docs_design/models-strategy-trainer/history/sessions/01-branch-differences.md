# Alternate-branch differences for the inventory passage

Read alongside [chapter01](01-inventory-and-authoring.md). These are branch-specific records omitted from that selected path. They include an unrelated preview-button message, graph-health checks, an earlier inventory attempt and recovery discussion, and an explicit author-versus-builder clarification. They are retained without repeating the inherited main conversation.

Captured 2026-11-05T18:00:27.008450+00:00. 36 visible user/assistant messages. Bodies are preserved, including historical annotations, pasted material and line endings. Tool traffic, reasoning and generated context are excluded.

[Machine-readable provenance](01-branch-differences.json). [Conversation branch map](branch-map.md).

For each message, the first source is the earliest matching stable identity and content found in the inspected ancestry. The selected record shows the location from which this export was taken. An inherited record timestamp is not automatically its original conversation time.

## Message index

- [001 user: the preview button still goes out of the node sometimes and becomes unclickable until refreshing](#m001)
- [002 user: before inventory, is codebase-memory working properly? just do a small check that you'd normally do](#m002)
- [003 assistant: I’ll do a small normal health check: confirm the indexed project/generation, look up one known strategy s](#m003)
- [004 assistant: The index itself is healthy and current. One useful limitation showed up in the check: it finds the concr](#m004)
- [005 assistant: Mostly yes, with one important limitation.](#m005)
- [006 user: treat low-confidence suffix_match edges with multiple candidates as provisional and expand OVERRIDE relat](#m006)
- [007 assistant: Yes—that is the right next step. I’ll inventory the existing contract surface, callers, implementations, ](#m007)
- [008 assistant: The direction document has already been moved into the new `docs_design/models-strategy-trainer/` area; G](#m008)
- [009 assistant: I’m treating this as an Auditor-tier inventory over the active model/strategy/trainer path: bounded to th](#m009)
- [010 assistant: The first strong inventory result is already clear: the graph’s low-confidence polymorphic edges really d](#m010)
- [011 assistant: The inventory is exposing concrete state-boundary defects, not just aesthetic concerns: compatibility pro](#m011)
- [012 assistant: The inventory pass is substantially complete; I’m checking that the interrupted write landed cleanly, the](#m012)
- [013 assistant: The inventory file is intact at 729 lines and the working tree reflects the intended document relocation:](#m013)
- [014 assistant: Two important outcomes are emerging from the check. First, the inventory confirms this is not mainly an i](#m014)
- [015 assistant: The document now cleanly distinguishes what this inventory completes from what remains design work. We ha](#m015)
- [016 assistant: The review confirmed the contract count, family matrix, state hand-back, hidden validation dependency, sa](#m016)
- [017 assistant: The inventory step is complete and recorded in \[strategy_system_inventory.md\](/home/imi/Projects/sd-scrip](#m017)
- [018 user: then let's do the inventory before the other points. this is the working docs folder for this will be the](#m018)
- [019 assistant: I’ll inventory the current model–strategy–trainer system before returning to the unresolved design questi](#m019)
- [020 assistant: The repository already contains the discussion record at `docs_design/models-strategy-trainer/strategy_sy](#m020)
- [021 assistant: I’m in the production inventory now: mapping the active launcher through trainer, modes, strategy facets,](#m021)
- [022 assistant: The inventory confirms the important shape: the repository already builds one aggregate family strategy, ](#m022)
- [023 assistant: Inventory completed at \[strategy_system_inventory.md\](/home/imi/Projects/sd-scripts/docs_design/models-st](#m023)
- [024 user: quick meta question. you spent a lot of time going through the relevant code, and then you hit context li](#m024)
- [025 assistant: No, I didn’t lose all of it—but I did lose the verbatim working context.](#m025)
- [026 user: that's not great](#m026)
- [027 assistant: No, it isn’t ideal. Compaction is lossy: a good summary preserves conclusions, but it cannot guarantee ev](#m027)
- [028 user: that's not the same thing](#m028)
- [029 assistant: You’re right. Writing notes or rereading the code is not the same as retaining the original exploration.](#m029)
- [030 user: yes it does, go check the file for the session](#m030)
- [031 assistant: You’re right—I treated the session ID as merely descriptive when it can identify the persisted transcript](#m031)
- [032 assistant: You were right. I checked the session file:](#m032)
- [033 user: the "run" doesn't select things, these are defined in the strategy build the same as today, just under th](#m033)
- [034 assistant: Correct. My “run selects concerns” framing was wrong. The strategy builder selects and assembles them int](#m034)
- [035 user: I think you maybe misread what I said. there is no "builder", builder is literally whoever writes the cod](#m035)
- [036 assistant: Yes, I misread you. There is no proposed builder abstraction or generic runtime assembly step.](#m036)

### Distinct material first found in branch `019f86d4-43a2-73a2-a37a-ba7c5a26a4f5`

This is alternate-branch material, not a continuation after the preceding selected-path chapter. Shared messages are omitted from this appendix. Gaps between source lines can include shared messages, tool traffic or generated context.

<a id="m001"></a>
## 001 User - 2026-07-23T00:51:16.639Z

Source: [019f86d4-43a2-73a2-a37a-ba7c5a26a4f5, raw line 11435](/home/imi/.codex/archived_sessions/rollout-2026-07-21T22-38-05-019f86d4-43a2-73a2-a37a-ba7c5a26a4f5.jsonl:11435).
Export record: `019f86d4-43a2-73a2-a37a-ba7c5a26a4f5`, raw line 11435, 2026-07-23T00:51:16.639Z.

the preview button still goes out of the node sometimes and becomes unclickable until refreshing


Non-text content is not reconstructed; see the raw record.

### Distinct material first found in branch `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`

This is alternate-branch material, not a continuation after the preceding selected-path chapter. Shared messages are omitted from this appendix. Gaps between source lines can include shared messages, tool traffic or generated context.

<a id="m002"></a>
## 002 User - 2026-07-26T00:56:09.668Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11444](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11444).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11444, 2026-07-26T00:56:09.668Z.

before inventory, is codebase-memory working properly? just do a small check that you'd normally do


<a id="m003"></a>
## 003 Assistant (commentary) - 2026-07-26T00:56:14.429Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11448](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11448).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11448, 2026-07-26T00:56:14.429Z.

I’ll do a small normal health check: confirm the indexed project/generation, look up one known strategy symbol, trace one call edge, and check coverage for the files used. I’m using the codebase-memory skill because this is specifically validating that exploration path.

<a id="m004"></a>
## 004 Assistant (commentary) - 2026-07-26T00:56:53.389Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11494](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11494).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11494, 2026-07-26T00:56:53.389Z.

The index itself is healthy and current. One useful limitation showed up in the check: it finds the concrete SDXL method and source correctly, but an inbound trace to that override reports no caller because the trainer calls through the abstract `TrainingStrategy` surface. I’m checking that abstract edge now so we know whether this is normal dynamic-dispatch behavior or a broader trace problem.

<a id="m005"></a>
## 005 Assistant (final_answer) - 2026-07-26T00:57:17.755Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11518](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11518).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11518, 2026-07-26T00:57:17.755Z.

Mostly yes, with one important limitation.

- Index is `ready`, full mode, current generation `2026-07-26T00:54:44Z`.
- Symbol search correctly found the base, SD, SDXL, and SD3 `process_batch` implementations.
- Source retrieval and coverage checks worked correctly with no recorded gaps.
- Polymorphic call tracing is unreliable here: `_run_epoch_steps()` calls `strategies.process_batch(...)`, but the graph resolves that edge only to SD3 and misses the base/SD/SDXL implementations.

So codebase-memory is healthy for discovery, definitions, source retrieval, and most structural work. During the inventory, I’ll use it as intended but verify dynamically dispatched strategy calls against source rather than trusting that trace edge as exhaustive. No files were changed.

<a id="m006"></a>
## 006 User - 2026-07-26T01:16:06.732Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11525](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11525).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11525, 2026-07-26T01:16:06.732Z.

treat low-confidence suffix_match edges with multiple candidates as provisional and expand OVERRIDE relationships for polymorphic calls

let's go through the inventory step, unless I'm forgetting something. new work docs location is docs_design/models-strategy-trainer/strategy_system_direction.md

<a id="m007"></a>
## 007 Assistant (commentary) - 2026-07-26T01:16:12.755Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11529](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11529).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11529, 2026-07-26T01:16:12.755Z.

Yes—that is the right next step. I’ll inventory the existing contract surface, callers, implementations, state flow, and responsibility placement, treating ambiguous `suffix_match` call edges as provisional and expanding `OVERRIDE` relationships before drawing polymorphism conclusions. I’m using the OpenSpec exploration and codebase-memory exploration/tracing skills because this is pre-design architectural research, not implementation.

<a id="m008"></a>
## 008 Assistant (commentary) - 2026-07-26T01:16:28.226Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11542](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11542).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11542, 2026-07-26T01:16:28.226Z.

The direction document has already been moved into the new `docs_design/models-strategy-trainer/` area; Git sees that as the old path deleted plus the new folder untracked, which is expected until committed. I’ll keep the inventory beside it and won’t alter production code or the active metadata OpenSpec during this exploration.

<a id="m009"></a>
## 009 Assistant (commentary) - 2026-07-26T01:16:43.288Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11552](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11552).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11552, 2026-07-26T01:16:43.288Z.

I’m treating this as an Auditor-tier inventory over the active model/strategy/trainer path: bounded to the base contracts, SD/SDXL/SD3 strategies, `library/models`, training modes/phases/runner, objectives, and directly involved cache/sampling/checkpoint seams. That means complete pagination inside the scope, exact source for material boundaries, and explicit disclosure where dynamic dispatch or index coverage limits certainty.

<a id="m010"></a>
## 010 Assistant (commentary) - 2026-07-26T01:20:11.010Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11673](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11673).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11673, 2026-07-26T01:20:11.010Z.

The first strong inventory result is already clear: the graph’s low-confidence polymorphic edges really do collapse several calls onto SD3. Expanding `OVERRIDE` relations and checking source shows three implementations for most facets, while some sibling-mixin methods such as tokenization are not represented as overrides at all. I’m therefore using the graph for the contract/override set and literal source verification for every trainer-facing call site.

<a id="m011"></a>
## 011 Assistant (commentary) - 2026-07-26T01:22:46.417Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11718](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11718).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11718, 2026-07-26T01:22:46.417Z.

The inventory is exposing concrete state-boundary defects, not just aesthetic concerns: compatibility projections are sometimes passed where the contract requires lists, a fine-tune encoder wrapper assignment mutates a temporary projection rather than `loaded_components`, and some supposedly supported mode/family combinations fail only during checkpoint metadata or saving. I’m recording these as evidence and pressure points, not fixing them during exploration.

<a id="m012"></a>
## 012 Assistant (commentary) - 2026-07-26T01:26:38.830Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11730](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11730).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11730, 2026-07-26T01:26:38.830Z.

The inventory pass is substantially complete; I’m checking that the interrupted write landed cleanly, then I’ll validate it against the direction document and finish the evidence-backed handoff. I’m keeping ambiguous `suffix_match` call edges provisional and relying on expanded `OVERRIDE` relationships plus source verification for polymorphic behavior.

<a id="m013"></a>
## 013 Assistant (commentary) - 2026-07-26T01:26:51.652Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11740](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11740).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11740, 2026-07-26T01:26:51.652Z.

The inventory file is intact at 729 lines and the working tree reflects the intended document relocation: the old standalone path is deleted and the new folder contains the direction document plus the inventory. I’m now checking the inventory’s matrices and concrete defect claims, since those are the parts that should guide the next design rather than merely describe it.

<a id="m014"></a>
## 014 Assistant (commentary) - 2026-07-26T01:27:11.356Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11750](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11750).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11750, 2026-07-26T01:27:11.356Z.

Two important outcomes are emerging from the check. First, the inventory confirms this is not mainly an inheritance problem: the current system lacks a single notion of “this selected run is fulfilled,” so unsupported combinations survive construction and fail later. Second, the strongest immediate design pressure is the live component surface—trainer compatibility projections already cause concrete replacement and sampling-shape defects. I found one Markdown fence error and am adding an explicit link between the direction and inventory documents before review.

<a id="m015"></a>
## 015 Assistant (commentary) - 2026-07-26T01:27:56.016Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11774](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11774).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11774, 2026-07-26T01:27:56.016Z.

The document now cleanly distinguishes what this inventory completes from what remains design work. We have finished the current-code audit; we have not yet written the cross-scenario conformance cases or chosen the revised contract/state shape. I’m requesting a review focused only on factual accuracy against the active strategy/model/trainer code—not asking the reviewer to replace the architectural conclusions from our discussion.

<a id="m016"></a>
## 016 Assistant (commentary) - 2026-07-26T01:30:57.589Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11785](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11785).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11785, 2026-07-26T01:30:57.589Z.

The review confirmed the contract count, family matrix, state hand-back, hidden validation dependency, sampling mismatches, projection defect, and fulfillment analysis. It flagged one detail worth resolving rather than accepting casually: the exact SD3 weighted-prompt failure type. I’m checking the actual conditioning path now and will correct the inventory if its exception label is too specific.

<a id="m017"></a>
## 017 Assistant (final_answer) - 2026-07-26T01:31:39.881Z

Source: [019f8c75-6709-7f02-9e99-a3d5d9e5fffd, raw line 11799](/home/imi/.codex/sessions/2026/07/23/rollout-2026-07-23T00-52-12-019f8c75-6709-7f02-9e99-a3d5d9e5fffd.jsonl:11799).
Export record: `019f8c75-6709-7f02-9e99-a3d5d9e5fffd`, raw line 11799, 2026-07-26T01:31:39.881Z.

The inventory step is complete and recorded in [strategy_system_inventory.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_inventory.md). The relocated [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md) now links to it.

The main findings are:

- The current contract contains 25 operations across 11 facets, mixing trainer-facing lifecycle operations with family-internal collaboration.
- Strategy construction proves Python interface completeness, not that the chosen family, mode, objective, representation, and persistence paths form a valid training run.
- The strategy/trainer relationship repeatedly passes loaded objects back and forth without one clear state owner.
- Compatibility projections already cause concrete defects:
  - element assignment through `Trainer.text_encoders` mutates a temporary list;
  - sampling passes a single encoder where its declared contract expects a list.
- Validation bypasses `process_batch()` and depends on undeclared sibling-MRO helpers.
- Caching, sampling, checkpointing, and `process_batch()` each combine responsibilities belonging to several architectural areas.
- The existing loaded-component declarations are the strongest basis for a better model boundary.
- The models/strategies split remains valid, but orchestration, runtime policy, IO, and generic objective/loss behavior have leaked across it.

Codebase-memory was used conservatively: ambiguous multi-candidate `suffix_match` calls were treated as provisional, `OVERRIDE` paths were expanded, and polymorphic claims were verified against source.

Review confirmed the inventory’s factual accuracy. It raised one possible discrepancy about SD3 weighted-caption failure, but direct source verification confirmed the inventory is correct: SD3 explicitly raises `NotImplementedError`.

What remains before proposing implementation is no longer “more general inventory.” It is focused design work:

1. Write conformance scenarios for current latent diffusion, pixel diffusion, adapters, added components, distillation, and less image-shaped models.
2. Decide the minimum trainer-facing contract.
3. Decide how loaded component state is owned and safely replaced.
4. Define what a fulfilled training strategy records and validates.
5. Select a first bounded migration that fixes a real defect and proves the architecture.

Only documentation changed, so no runtime tests were needed. Markdown structure, local links, and whitespace checks passed.

### Distinct material first found in branch `019fa69a-b709-7b73-bb57-abe2021bfdb9`

This is alternate-branch material, not a continuation after the preceding selected-path chapter. Shared messages are omitted from this appendix. Gaps between source lines can include shared messages, tool traffic or generated context.

<a id="m018"></a>
## 018 User - 2026-07-28T23:56:34.643Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11445](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11445).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11445, 2026-07-28T23:56:34.643Z.

then let's do the inventory before the other points. this is the working docs folder for this will be the area docs_design/models-strategy-trainer


<a id="m019"></a>
## 019 Assistant (commentary) - 2026-07-28T23:56:41.440Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11449](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11449).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11449, 2026-07-28T23:56:41.440Z.

I’ll inventory the current model–strategy–trainer system before returning to the unresolved design questions. I’m using the OpenSpec exploration workflow to keep this as research rather than premature implementation, and codebase-memory for verified structural/call-path evidence. I’ll keep the working material under `docs_design/models-strategy-trainer/` and won’t use subagents.

<a id="m020"></a>
## 020 Assistant (commentary) - 2026-07-28T23:57:37.345Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11484](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11484).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11484, 2026-07-28T23:57:37.345Z.

The repository already contains the discussion record at `docs_design/models-strategy-trainer/strategy_system_direction.md`; it was moved there from the old top-level location and is currently untracked, so I’ll preserve that move and add a separate evidence inventory rather than mixing observed facts into the direction document. The graph index is current at `de2dd0e` with no recorded gaps in the model/strategy/training Python scopes; I’m treating this as a Verify-tier inventory and will still check every cited path explicitly.

<a id="m021"></a>
## 021 Assistant (commentary) - 2026-07-29T00:04:03.991Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11671](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11671).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11671, 2026-07-29T00:04:03.991Z.

I’m in the production inventory now: mapping the active launcher through trainer, modes, strategy facets, and model components. I’m keeping “observed current behavior,” “architectural pressure,” and “proposed direction” separate so the inventory doesn’t quietly become a design decision.

<a id="m022"></a>
## 022 Assistant (commentary) - 2026-07-29T00:05:31.665Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11724](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11724).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11724, 2026-07-29T00:05:31.665Z.

The inventory confirms the important shape: the repository already builds one aggregate family strategy, but it does not yet receive or own one coherent run binding. The trainer and modes repeatedly unwrap, mutate, and pass model state back into that aggregate, while several “contract” methods are only family-internal mixin seams. I’m recording that distinction explicitly rather than treating the current ABC list as the future contract.

<a id="m023"></a>
## 023 Assistant (final_answer) - 2026-07-29T00:09:29.229Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11760](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11760).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11760, 2026-07-29T00:09:29.229Z.

Inventory completed at [strategy_system_inventory.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_inventory.md).

It covers:

- All 25 current strategy contract operations and their actual callers.
- Trainer-facing operations versus family-internal mixin collaboration.
- SD, SDXL, and SD3 fulfillment differences.
- Loaded-component ownership and trainer/strategy/mode state flow.
- Training, validation, caching, preparation, sampling, and saving paths.
- Existing compatibility constraints.
- Misplaced or mixed-responsibility code.
- The concrete questions to resolve before designing the replacement contract.

The main conclusion is that we already construct one aggregate strategy, but it is simultaneously a trainer façade, internal mixin bus, and partial state holder. The loaded-component surface is useful; ownership and fulfillment semantics around it are what need redesign.

No production code changed and no tests were needed. `git diff --check` passed. The inventory Beads issue `sd-scripts-0pd` is closed. I skipped review because this is an exploratory research document, not an implementation/OpenSpec milestone.

<a id="m024"></a>
## 024 User - 2026-07-29T00:09:35.075Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11768](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11768).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11768, 2026-07-29T00:09:35.075Z.

quick meta question. you spent a lot of time going through the relevant code, and then you hit context limit and it compacted the context. did you lose all the exploration details you got from before that? 019fa69a-b709-7b73-bb57-abe2021bfdb9 is this session


<a id="m025"></a>
## 025 Assistant (final_answer) - 2026-07-29T00:09:53.492Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11772](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11772).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11772, 2026-07-29T00:09:53.492Z.

No, I didn’t lose all of it—but I did lose the verbatim working context.

When compaction happened, the earlier messages and raw tool outputs were replaced with a detailed structured summary. That summary retained:

- The files and architecture paths inspected.
- The graph-index generation and coverage state.
- The complete contract-method classification.
- Trainer/strategy/mode ownership findings.
- Family-specific behavior and placement concerns.
- Dirty-worktree and Beads state.
- What still needed verification before writing the inventory.

What was lost was the exact raw output, intermediate reasoning, and any minor observation the compaction summary didn’t preserve.

After compaction, I used that summary as a handoff and reopened the important source files—contracts, family assembly, loading, checkpointing, modes, training/validation flows, component declarations—and reran the final coverage check. So the inventory wasn’t written solely from compressed recollection.

The durable record is now the inventory document. If some subtle detail had existed only in my pre-compaction reasoning and wasn’t summarized or written down, then yes, that detail could have been lost. Session `019fa69a-b709-7b73-bb57-abe2021bfdb9` is the session you’re referring to.

<a id="m026"></a>
## 026 User - 2026-07-29T00:10:00.767Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11778](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11778).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11778, 2026-07-29T00:10:00.767Z.

that's not great


<a id="m027"></a>
## 027 Assistant (final_answer) - 2026-07-29T00:10:15.807Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11782](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11782).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11782, 2026-07-29T00:10:15.807Z.

No, it isn’t ideal. Compaction is lossy: a good summary preserves conclusions, but it cannot guarantee every observation, caveat, or connection survives.

The better working method for investigations this large is to make the exploration durable as it happens:

- Write findings progressively into the working research document.
- Record exact paths, symbols, and unresolved questions before continuing.
- Treat the conversation context as temporary working memory, not the only evidence store.
- Break the investigation into bounded slices and checkpoint each slice.
- After compaction, verify against those notes and source rather than relying on the generated summary.

In this case, the compaction summary happened to be unusually detailed, and I reread the important code afterward. So I’m confident the inventory’s major conclusions survived. But the process still exposed a real risk: smaller incidental findings could have disappeared unnoticed.

For future architecture research, I should begin the inventory document early and build it alongside exploration, rather than doing most of the investigation in context and writing afterward.

<a id="m028"></a>
## 028 User - 2026-07-29T00:10:26.252Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11788](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11788).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11788, 2026-07-29T00:10:26.252Z.

that's not the same thing


<a id="m029"></a>
## 029 Assistant (final_answer) - 2026-07-29T00:10:42.448Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11792](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11792).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11792, 2026-07-29T00:10:42.448Z.

You’re right. Writing notes or rereading the code is not the same as retaining the original exploration.

After compaction, I no longer had the full pre-compaction reasoning and raw evidence in active context. I had a generated summary. It retained a lot, but anything omitted from that summary was unavailable to me. Reopening files can reconstruct facts, but it cannot guarantee reconstruction of every connection, doubt, or discarded interpretation from the original exploration.

The session ID does not automatically restore that context to me. The faithful remedy would be to retrieve and replay the earlier transcript for that session, if it is locally accessible—not substitute documentation or a second investigation and call them equivalent.

<a id="m030"></a>
## 030 User - 2026-07-29T00:11:04.731Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11798](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11798).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11798, 2026-07-29T00:11:04.731Z.

yes it does, go check the file for the session


<a id="m031"></a>
## 031 Assistant (commentary) - 2026-07-29T00:11:09.615Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11802](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11802).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11802, 2026-07-29T00:11:09.615Z.

You’re right—I treated the session ID as merely descriptive when it can identify the persisted transcript. I’ll locate that session file and check what remains available beyond the compacted active context.

<a id="m032"></a>
## 032 Assistant (final_answer) - 2026-07-29T00:12:02.735Z

Source: [019fa69a-b709-7b73-bb57-abe2021bfdb9, raw line 11820](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl:11820).
Export record: `019fa69a-b709-7b73-bb57-abe2021bfdb9`, raw line 11820, 2026-07-29T00:12:02.735Z.

You were right. I checked the session file:

[rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl](/home/imi/.codex/sessions/2026/07/28/rollout-2026-07-28T02-43-05-019fa69a-b709-7b73-bb57-abe2021bfdb9.jsonl)

It is an append-only transcript containing the original exploration—not merely the compacted summary. At the time I checked, it contained:

- 13,145 JSONL records / about 55 MB.
- 2,492 reasoning records.
- 1,733 custom tool calls and their outputs.
- 445 function calls and 444 outputs.
- The raw model/strategy/trainer exploration immediately preceding compaction.
- Separate `compacted` records describing the replacement context.

So the accurate distinction is:

- Compaction removed the details from my active context.
- It did **not** remove them from the session.
- I can recover the original reasoning and tool evidence from that file using the session ID.

I was wrong to speak as though omitted details might be irretrievable without first checking the persisted session. The session ID is operationally useful, and I should use the transcript after compaction when earlier exploration materially affects the work.

### Distinct material first found in branch `019fab38-3678-72c3-b798-adb2e9e225b9`

This is alternate-branch material, not a continuation after the preceding selected-path chapter. Shared messages are omitted from this appendix. Gaps between source lines can include shared messages, tool traffic or generated context.

<a id="m033"></a>
## 033 User - 2026-07-29T02:01:25.877Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11953](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11953).
Export record: `019fab38-3678-72c3-b798-adb2e9e225b9`, raw line 11953, 2026-07-29T02:01:25.877Z.

the "run" doesn't select things, these are defined in the strategy build the same as today, just under the new contract. it might constrain or validate it (where this happens is not decided from what I've read so far). the contract defines what must be fulfilled and what is available to use whether it's a slot that needs to be filled or something that you're picking for yourself. we're not going for a super smart mind reader system that automatically picks things based on implication, at least not for now. strategies should be built with intent. we can validate that the choices are correct, but that's different from a feature dragging another with it, that should still be left to whoever is putting the strategy together. (and this is us, we make the default training strategies for models aka what you see currently)

<a id="m034"></a>
## 034 Assistant (final_answer) - 2026-07-29T02:02:57.021Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11961](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11961).
Export record: `019fab38-3678-72c3-b798-adb2e9e225b9`, raw line 11961, 2026-07-29T02:02:57.021Z.

Correct. My “run selects concerns” framing was wrong. The strategy builder selects and assembles them intentionally; the run configuration only supplies inputs and constraints to that builder.

The intended flow is:

```text
CONTRACT
defines obligations, slots, extension seams, and semantics
        │
        ▼
STRATEGY BUILDER
explicitly chooses implementations and wires them together
        │
        ▼
VALIDATION
checks completeness and compatibility without adding anything
        │
        ▼
FULFILLED TRAINING STRATEGY
the complete executable plan handed to the trainer
        │
        ▼
TRAINER
executes it without knowing its internal parts
```

There should be no dependency resolver that sees `LatentRepresentation` and automatically decides to attach a VAE, cache codec, objective, or anything else. Whoever writes the builder must do that deliberately.

A future SDXL builder might conceptually resemble:

```python
def build_sdxl_strategy(cfg) -> FulfilledTrainingStrategy:
    representation = SdxlLatentRepresentation(
        autoencoder=SdxlAutoencoderAdapter(...),
    )

    conditioning = SdxlClipConditioning(
        clip_l=...,
        clip_g=...,
        weighted_prompts=SdxlWeightedPrompts(...),
    )

    if cfg.objective.path == "ddpm":
        objective = DdpmObjectiveIntegration(...)
        sampling = SdxlDdpmSampling(...)
    elif cfg.objective.path == "rectified_flow":
        objective = RectifiedFlowObjectiveIntegration(...)
        sampling = SdxlFlowSampling(...)
    else:
        raise UnsupportedStrategyConfiguration(...)

    definition = TrainingStrategyDefinition(
        representation=representation,
        conditioning=conditioning,
        predictor=SdxlUnetPredictor(...),
        objective=objective,
        training_subject=...,
        sampling=sampling,
        persistence=SdxlPersistence(...),
    )

    return fulfill_training_strategy(definition)
```

The builder explicitly made every choice. `fulfill_training_strategy()` would inspect and validate the result, not alter it.

If `SdxlFlowSampling` needs rectified-flow semantics, that relationship can be declared for validation:

```text
SdxlFlowSampling requires objective semantics: rectified_flow
```

But that means:

```text
check that the builder supplied them
```

not:

```text
automatically locate and attach RectifiedFlowObjectiveIntegration
```

### Contract versus feature catalog

I would narrow one part of the wording slightly: the contract should define what is available to fulfill conceptually, but it should not necessarily enumerate every concrete implementation.

Otherwise, adding a new autoencoder adapter or conditioning implementation would mutate the core contract every time.

A cleaner separation is:

```text
Contract
  defines:
    required obligations
    builder choice-points
    accepted interfaces
    lifecycle semantics
    compatibility rules
    trainer-visible results

Feature/component catalog
  provides:
    LatentRepresentation
    PixelRepresentation
    SdxlClipConditioning
    Sd3TextConditioning
    DdpmObjectiveIntegration
    RectifiedFlowObjectiveIntegration
    ...
```

The contract says, for example:

```text
the strategy must provide model-input preparation
the strategy must provide a differentiable training step
the strategy may fulfill input preparation using any conforming representation
```

The catalog says which representation implementations currently exist. The builder chooses one explicitly.

That distinction makes the contract stable while letting the grab box grow.

### Three kinds of contract entries

The next design table should distinguish:

1. Required obligations

   Every build must fulfill them.

   Examples:

   - authoritative bound state;
   - trainable parameter/optimization description;
   - differentiable training-step execution;
   - lifecycle preparation;
   - trainer-facing result semantics.

2. Builder choice-points

   The builder must make an explicit choice, but the contract does not prescribe one implementation.

   Examples:

   - representation handling;
   - objective integration;
   - predictor invocation;
   - training subject;
   - perhaps conditioning, depending on how we represent unconditional models.

3. Available extension seams

   The contract defines how these participate if the builder deliberately includes them.

   Examples:

   - caching;
   - sampling;
   - validation;
   - persistence formats;
   - weighted prompts;
   - additional trainable participants;
   - metadata/query exposure.

Once included by the builder, these are not optional in any meaningful sense. They are part of that strategy and must work.

### Validation timing remains open

You are also correct that we have not decided where validation happens. There are at least two kinds:

```text
construction validation
  all required filings exist
  declared feature semantics are compatible
  configuration constraints are supported

bound-state validation
  expected components actually loaded
  roles and types match
  replacements remain coherent
  required runtime facilities exist
```

Those could happen in one place or at two lifecycle boundaries. We should decide that later. The important present rule is that validation observes the builder’s choices; it does not make them.

So the next contract exercise should use a table like:

| Concern | Contract obligation | Builder decision | Validation evidence | Trainer visibility |
| --- | --- | --- | --- | --- |
| bound state | required | component integration | loaded roles and lifecycle state | opaque |
| training subject | required | base model, adapter, added components | parameter groups are valid | generic optimization description |
| representation | explicit choice | latent, pixel, other | input/output tensor semantics agree | opaque |
| conditioning | explicit filing where used | CLIP, T5, LLM, unconditional | predictor accepts payload | opaque |
| objective | required | DDPM, RF, future | predictor/target/loss semantics agree | final loss only |
| sampling | explicitly included by builder | family/objective implementation | compatible bound state | generic result |
| persistence | explicitly included by builder | adapter/full-model formats | serializer supports bound state | generic artifact result |

That is the next useful addition to the direction document: intentional construction, validation without auto-wiring, and the first contract-obligation table.

<a id="m035"></a>
## 035 User - 2026-07-29T02:05:01.147Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11967](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11967).
Export record: `019fab38-3678-72c3-b798-adb2e9e225b9`, raw line 11967, 2026-07-29T02:05:01.147Z.

I think you maybe misread what I said. there is no "builder", builder is literally whoever writes the code for the strategy


<a id="m036"></a>
## 036 Assistant (final_answer) - 2026-07-29T02:05:35.465Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11972](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11972).
Export record: `019fab38-3678-72c3-b798-adb2e9e225b9`, raw line 11972, 2026-07-29T02:05:35.465Z.

Yes, I misread you. There is no proposed builder abstraction or generic runtime assembly step.

“Building the strategy” means us writing the strategy implementation:

```text
CONTRACT
    ↓ implemented deliberately by a developer
SdxlTrainingStrategy / Sd3TrainingStrategy / FutureModelStrategy
    ↓ instantiated by the existing factory
TRAINER
```

The human-authored strategy class is the fulfillment. `build_training_strategy()` remains essentially a factory that selects and instantiates that authored implementation; it does not compose features dynamically.

The feature grab box is for the person writing the strategy:

```python
class SdxlTrainingStrategy(TrainingStrategy):
    def __init__(self, cfg):
        self.representation = SdxlLatentDiffusion(...)
        self.conditioning = SdxlClipConditioning(...)
        self.objective = SdxlObjectiveHandling(...)
```

Or it might use inheritance, decorators, direct function calls, or another organization we choose later. Regardless, the strategy author explicitly chose those pieces in source code.

Nothing sees `SdxlLatentDiffusion` and automatically attaches:

- an autoencoder;
- a cache codec;
- CLIP;
- DDPM;
- sampling;
- persistence.

If SDXL supports both DDPM and rectified flow, that support exists because its strategy author deliberately implemented the branch:

```python
if cfg.objective.path == "ddpm":
    ...
elif cfg.objective.path == "rectified_flow":
    ...
else:
    raise ...
```

The configuration selects a path the strategy intentionally exposes; it does not assemble the strategy.

So the corrected distinction is:

```text
contract
  defines what a valid strategy implementation must provide
  defines the vocabulary and feature seams available to strategy authors

strategy source code
  explicitly chooses components and reusable features
  implements every required contract behavior
  deliberately exposes any supported configuration choices

validation
  checks that this authored strategy can handle its configuration and live state
  does not add, select, or wire features

trainer
  receives the completed strategy instance
  does not know its internal parts
```

That also changes how we should approach the next contract draft. We should not design a `TrainingStrategyDefinition`, fulfillment engine, dependency resolver, or strategy builder.

We should classify the contract surface into:

- What every authored training strategy must implement for the trainer.
- What reusable feature contracts are available to strategy authors.
- What remains ordinary internal implementation with no base contract.
- What configuration choices an individual strategy deliberately advertises.
- What conformance tests prove that an implementation genuinely honors the contract.

The strongest concise statement is:

```text
A strategy is hand-authored code that fulfills the training contract.
Features are tools for writing that strategy, not pieces selected by the run.
The trainer executes the resulting strategy as one opaque training plan.
```

That is what should replace the builder-oriented language from my previous answer.
