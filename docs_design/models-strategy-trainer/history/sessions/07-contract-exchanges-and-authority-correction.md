# First exchange design and contract-authority correction, 22-23 August 2026

Continuation of [chapter06](06-authority-reviews-and-handoff.md) in the explicitly linked new primary session. The first user turn carries Q1-Q5 forward; there is no recorded fork parent.

The [first exchange draft](../snapshots/3602960a-strategy_contract_exchange_design.md) and [expanded candidate-validation draft](../snapshots/acdc2b19-strategy_contract_exchange_design.md) are read against the [authority correction](../snapshots/7bd8a124-strategy_contract_exchange_design.md). This preserves the rejected participant-owned validator approach instead of presenting all early EX entries as current.

The lasting correction is that ordinary authors choose and wire maintained behavior; the training contract system derives and enforces its obligations. Custom conformance begins only on an explicit extension path. The later August 26 accepted-arrangement correction also supersedes the active-strategy enclosure used in these diagrams; it does not discard the contract-authority or binding meanings.

The last session suggested several sequential OpenSpecs; this session recommends one coherent governing change with multiple specs and reviewable milestones. Those are historical planning recommendations, not competing currently active authorities. The three supplied replies are preserved separately, including suggestions not adopted unchanged.

Captured 2026-11-05T22:12:32.467805+00:00. 88 visible user/assistant messages. Exact bodies, annotations, historical links and line endings are preserved. Reasoning, tools, generated context and compaction summaries are omitted.

[Machine-readable provenance](07-contract-exchanges-and-authority-correction.json) | [Sources](07-source-records.json) | [Notepad](../decision-notepad.md)

## Message index

- [001 user: Hi! We're opening a new chapter for working on our repo. Get familiar with the preliminary documenta](#m001)
- [002 assistant: I’ll use the Beads workflow to recover the durable task context, then the codebase-memory exploratio](#m002)
- [003 assistant: The durable context is intact: `sd-scripts-syv` is already in progress, Q1–Q5 are recorded as settle](#m003)
- [004 assistant: I’m oriented. I read the direction, inventory, notes, Beads context, development guide, and current ](#m004)
- [005 user: External reply and explicit authority order](#m005)
- [006 assistant: I’ll treat the pasted response as an external critique of our current proposal, not as authority. I’](#m006)
- [007 assistant: Understood. I’ll treat these responses as informed peer review: useful for exposing distinctions, bu](#m007)
- [008 user: Okay I understand the topic now. How should we start working by the way? is this multiple openspec i](#m008)
- [009 assistant: I’m using the OpenSpec exploration workflow here because this is exactly the boundary between prelim](#m009)
- [010 assistant: One existing OpenSpec is directly adjacent: `expand-model-metadata-identity-and-structure`. I’m chec](#m010)
- [011 assistant: I’d make this one coherent OpenSpec change containing multiple capability specs and implementation m](#m011)
- [012 user: We need to bound any progress in something concrete, like a file or evolving openspec, because 258k ](#m012)
- [013 assistant: I want to create a dedicated evolving pre-OpenSpec design now: `docs_design/models-strategy-trainer/](#m013)
- [014 assistant: The exchange design is now concretely bounded in the repository. I’ve kept every new point visibly c](#m014)
- [015 assistant: I’ve started the concrete record in strategy_contract_exchange_design.md.](#m015)
- [016 user: Okay, what to look into first?](#m016)
- [017 assistant: First, I want to inspect the repository’s existing declaration and identity surfaces—not preparation](#m017)
- [018 assistant: The first inspection already gives us a useful boundary: the repo has several “identity-like” types,](#m018)
- [019 assistant: First: define the minimum `ParticipantDeclaration` through scenarios, before choosing Python types.](#m019)
- [020 user: I don't really know how to approach the it so you have to guide us](#m020)
- [021 assistant: I’ll drive the design from here and bring you concrete choices with consequences, rather than asking](#m021)
- [022 assistant: I’ve pressure-tested the declaration against the four cases. The important result is that “deferred”](#m022)
- [023 assistant: You don’t need to supply the approach—I’ll derive it from the settled direction, code evidence, and ](#m023)
- [024 user: Two outside views on declaration/readiness/preparation](#m024)
- [025 assistant: I’ll read both as external peer-review snapshots against our current exchange record. I’ll keep anyt](#m025)
- [026 assistant: Both reviews are useful, and they expose four real refinements. I’m adopting three directly: route r](#m026)
- [027 assistant: Both external views were useful and mostly aligned with our direct findings. I reconciled them into ](#m027)
- [028 user: So were the viewpoints useful? how come they can be useful despite us having access to the code and ](#m028)
- [029 assistant: Yes—largely because the new direction does not exist in production code yet.](#m029)
- [030 user: Yeah that explanation makes sense. we are at about 205k/258k](#m030)
- [031 assistant: “Filing” does not mean a file on disk, registry entry, or user configuration. It means the strategy’](#m031)
- [032 assistant: “Semantic compatibility filing” is a working term for the strategy’s authored rules answering:](#m032)
- [033 user: denoiser is a role label no?](#m033)
- [034 assistant: Yes—exactly. In the current code, `denoiser` is a generic role label, and my example reused it insid](#m034)
- [035 assistant: Yes—exactly.](#m035)
- [036 user: is that a useful clarification?](#m036)
- [037 assistant: Yes. It was already implicit, but making it explicit prevents a likely design mistake.](#m037)
- [038 user: Okay, that's nice. But wait, what were we doing?](#m038)
- [039 assistant: We were designing the concrete contracts for the model–strategy–Trainer architecture before turning ](#m039)
- [040 user: "The next substantive design problem remains the semantic compatibility filing: what typed facts and](#m040)
- [041 assistant: You’re right—that is where we actually were. I skipped ahead in my recap.](#m041)
- [042 user: We didn't discuss and record anything about these?](#m042)
- [043 assistant: I’m checking the design record and chronological notes directly. I’m using the OpenSpec exploration ](#m043)
- [044 assistant: Yes—we discussed and recorded quite a lot. I understated the existing work in my previous answer.](#m044)
- [045 user: Can you look into that? after that we have to wrap up for now because it's late](#m045)
- [046 assistant: Yes. I’ll treat this as the last bounded design pass tonight: determine how much of the compatibilit](#m046)
- [047 assistant: The Beads skill is relevant because we’re ending the session after this pass; it requires the durabl](#m047)
- [048 assistant: The inspection is narrowing the problem cleanly. Current code gives us three useful constraints: `Lo](#m048)
- [049 assistant: The key design conclusion is now strong enough to record: compatibility validation should be authori](#m049)
- [050 assistant: The design record now contains the bounded result. I’m doing the wrap-up checks next: validate the M](#m050)
- [051 assistant: I looked into it and recorded the result.](#m051)
- [052 user: why are the names of things so odd? like "ParticipantDeclaration" and "RealizedModelComponentFacts" ](#m052)
- [053 assistant: Because they come from two different naming pressures, and both favor precision over naturalness.](#m053)
- [054 user: I really hope we can work in a way where we'll end up with readable code that's intuitive and acutal](#m054)
- [055 assistant: That concern should be a design constraint now, not something we discover after implementation.](#m055)
- [056 user: I don't want to make rules like that without seeing it](#m056)
- [057 assistant: That’s fair. We shouldn’t turn an understandable concern into a rigid rule before seeing the concret](#m057)
- [058 user: it's better to progress in an organic way. you can always finetune the end result after the base is ](#m058)
- [059 assistant: Agreed. We’ll let the architecture emerge from concrete exchanges and real model paths, then refine ](#m059)
- [060 user: Let's get back to what we were doing. Where are we now?](#m060)
- [061 assistant: We are still in concrete pre-OpenSpec design. No production API has been chosen or implemented.](#m061)
- [062 user: why are compatibility rules int he strategy definition?](#m062)
- [063 assistant: Because the strategy defines what each authored participant is supposed to mean. But my phrase “stra](#m063)
- [064 user: this is still too ambiguous](#m064)
- [065 assistant: Yes. The ambiguity exists because we have been using “compatibility” for three different questions.](#m065)
- [066 user: I still don't get it. Can you please properly look into this instead of coming up with it on the spo](#m066)
- [067 assistant: You’re right. My last two answers were extrapolations from the new draft, not a reconstruction of wh](#m067)
- [068 assistant: The direction document already shows that my “component owns the runtime validator” correction was a](#m068)
- [069 assistant: The inventory makes the missing distinction clearer: today’s real compatibility checks are about sel](#m069)
- [070 assistant: I’ve now separated the repository’s established design from the hypothesis I introduced. I’m doing o](#m070)
- [071 assistant: You were right to push back. After reading the direction, inventory, notes, and current code, my ear](#m071)
- [072 user: lets go through that please. lets make sure to properly inspect the relevant context&#x20;](#m072)
- [073 assistant: Yes. I’ll treat the participant-level compatibility draft as untrusted and rebuild the question from](#m073)
- [074 assistant: The first important result is already clear: the repository has two compatibility moments today, eve](#m074)
- [075 assistant: I inspected the relevant direction, inventory, notes, and runtime paths. The first thing we need to ](#m075)
- [076 user: what makes the most sense from a coherence and intuitiveness standpoint? I do not want validation in](#m076)
- [077 assistant: The most coherent target is one validation system behind the complete `TrainingStrategy`, with every](#m077)
- [078 user: this still does not really align with what I want. whatever happens to the authored strategy between](#m078)
- [079 assistant: You’re right. I was still treating validation as something assembled out of rules contributed by the](#m079)
- [080 assistant: Yes. Your model is more coherent, and it changes the center of the design:](#m080)
- [081 user: a lot of this is stuff we already talked about conceptually, these are just detail questions, and I'](#m081)
- [082 assistant: Yes, I think the exchange design needs modification—not because the overall architecture changed, bu](#m082)
- [083 assistant: Yes. We should modify the exchange design before continuing. The normative direction already says wh](#m083)
- [084 user: go ahead please](#m084)
- [085 assistant: I’ll update the exchange design in three coordinated pieces: replace the participant-authored compat](#m085)
- [086 assistant: The core rewrite is in place. The binding/preparation mechanics were preserved, while the authority ](#m086)
- [087 assistant: The contract-authority correction is captured. I’m doing one final clean-diff check, then I’ll hand ](#m087)
- [088 assistant: Done. The exchange design now treats the training contract system as the authoritative judge of the ](#m088)

<a id="m001"></a>
## 001 User - 2026-08-22T00:18:55.693Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 9](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:9).

Hi! We're opening a new chapter for working on our repo. Get familiar with the preliminary documentation and tools.&#x20;

We're continuing with instructions from the previous session: "Resume the model–strategy–trainer architecture work. Read the direction, inventory, and notes under `docs_design/models-strategy-trainer/`, treating the direction document as normative, the inventory as evidence, and notes as chronological history. Check bead `sd-scripts-syv`. Q1–Q5 are settled; begin concrete binding and runtime-preparation exchange design without reopening them."

If needed, search in session 019fab9f-4de5-7951-9cf5-d8acfe3d4e2f as it contains the full raw discussion and research leading up to this point, but prefer talking with me or looking at our notes and codebase.


<a id="m002"></a>
## 002 Assistant (commentary) - 2026-08-22T00:19:01.839Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 13](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:13).

I’ll use the Beads workflow to recover the durable task context, then the codebase-memory exploration workflow to map the relevant architecture against the design documents. I’ll treat the direction as normative, preserve Q1–Q5 as settled, and focus the result on the concrete binding/runtime-preparation exchange.

<a id="m003"></a>
## 003 Assistant (commentary) - 2026-08-22T00:19:35.836Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 42](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:42).

The durable context is intact: `sd-scripts-syv` is already in progress, Q1–Q5 are recorded as settled, and the next unresolved work is exactly the binding/preparation exchange. The code graph is healthy on the current `model-strategy-trainer` branch; its only reported parse gaps are unrelated dashboard/PowerShell files. I’m now reading the three design records in their intended authority order before tracing the current runtime paths.

<a id="m004"></a>
## 004 Assistant (final_answer) - 2026-08-22T00:22:39.933Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 163](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:163).

I’m oriented. I read the direction, inventory, notes, Beads context, development guide, and current code paths. I did not need the raw prior session. No files were changed.

The current checkpoint is unambiguous:

- Q1–Q5 stay settled.
- One complete strategy is the Trainer-facing integration.
- A dedicated per-run binding authority is internal to that strategy boundary.
- The immediate work is the binding and runtime-preparation exchanges.
- Optimization ownership and the optimization/step exchanges follow afterward.

The current code confirms the documented pressure:

- [`LoadedModelComponent`](/home/imi/Projects/sd-scripts/library/models/components.py:43) has one `module` slot serving logical and prepared meanings.
- [`Trainer.setup()`](/home/imi/Projects/sd-scripts/library/training/runners/trainer.py:307) takes ownership of the loaded tuple.
- Trainer’s compatibility setters replace components by generic role and can even synthesize missing entries, without revisions or invalidation.
- Adapter creation, materialization, relationship attachment, and state publication are collapsed inside [`AdapterMode.prepare_trainables()`](/home/imi/Projects/sd-scripts/library/training/modes/adapter_mode.py:115).
- Accelerator preparation mutates modules, optimizer, scheduler, synchronization handle, and primary trainable through the whole Trainer in both [`AdapterMode`](/home/imi/Projects/sd-scripts/library/training/modes/adapter_mode.py:243) and [`FineTuneMode`](/home/imi/Projects/sd-scripts/library/training/modes/finetune_mode.py:234).

The graph evidence is current as of generation `2026-08-22T00:12:14Z`; all relevant paths reported matching metadata and no recorded coverage gaps.

My concrete starting shape is:

```text
authored strategy filing
  -> authority accepts participant/relationship declarations
  -> authority issues typed runtime participant references
  -> materializers propose keyed binding transitions
  -> authority atomically accepts/rejects them
  -> authority exposes a revision-pinned preparation projection
  -> Trainer infrastructure prepares concrete objects
  -> preparation result returns keyed route bindings and backend handles
  -> authority atomically accepts route rebindings
  -> Trainer publishes prepared optimization/backend runtime
```

The binding exchange should be several typed operations, not one `bind()`:

| Operation | Principal input | Accepted result |
|---|---|---|
| Declaration | Authored participant keys, semantic obligations, relationships, executability | Participant references plus arrangement revision |
| Materialization | Participant reference, expected revision, bound state/access views, source facts | New binding revision and invalidation outcome |
| Replacement | Same reference, expected binding revision, replacement realization and lineage | Identity-preserving supersession or rejection |
| Relationship transition | Relationship reference and expected revision | New relationship state/revision and invalidations |
| Arrangement amendment | Explicit additions/retirements and relationship changes | New arrangement revision |
| Route rebinding | Preparation result tied to its source snapshot | New authoritative route revisions and freshness |

Three working defaults look strongest:

1. **Identity assignment:** strategy authors assign stable semantic `ParticipantKey`s; the authority validates them and issues typed run-scoped `ParticipantRef`s. Runtime code stops passing bare strings, but semantic identity remains authored rather than inferred.

2. **Replacement:** identity is preserved when the participant’s declared semantic obligations remain the same. Source, implementation, wrapper, device, dtype, and prepared representation may change. If the new state no longer fulfills the declared participant meaning or relationships, replacement is rejected and an explicit retire-and-declare amendment is required.

3. **Freshness:** when a derived route or access view declares no narrower dependency policy, it defaults conservatively to the exact authority revision from which it was produced. Any later binding-affecting transition invalidates it. More precise participant/relationship dependency sets can be added without weakening the contract.

For runtime preparation, the plan/result should separate three kinds of output:

- authoritative execution-route bindings, committed to the binding authority;
- prepared optimizer/scheduler state, owned by Trainer’s later optimization runtime;
- backend coordination handles such as a DeepSpeed composite or accumulation handle, which are never automatically logical participants.

One important preparation constraint follows from real backends: preparation may mutate objects in place. A plan therefore needs to declare whether an item is replacement-only or may mutate its bound object. Replacement-only failure leaves authority unchanged. A potentially in-place preparation attempt must suspend the affected route first; failure leaves it explicitly unprepared rather than pretending the old route remains valid.

The first downstream choice I’d confirm is the identity mechanism: should we carry forward the authored `ParticipantKey` plus authority-issued typed `ParticipantRef` split?

<a id="m005"></a>
## 005 User - 2026-08-22T00:29:31.057Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 171](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:171).


# Files pasted by the user:

## "Yeah — this new-session agent is very well oriented. It picked up the important…": /home/imi/.codex/attachments/3c2b4f9d-7932-47cc-a3ab-a72cb1905bcf/pasted-text.txt

## My request:
I sometimes use a separate agent (with less direct context about the codebase) for details that I can't articulate fully, so these responses usually are best read as replies. Our own findings with direct access to the codebase and up to date notes probably take precedence if anything sounds contradictory


Readable attachment copy: [2026-08-22-establishment-feedback.md](../snapshots/2026-08-22-establishment-feedback.md).

<a id="m006"></a>
## 006 Assistant (commentary) - 2026-08-22T00:29:37.062Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 175](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:175).

I’ll treat the pasted response as an external critique of our current proposal, not as authority. I’ll separate useful refinements from assumptions that conflict with the normative direction, current inventory, or live code.

<a id="m007"></a>
## 007 Assistant (final_answer) - 2026-08-22T00:30:31.168Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 188](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:188).

Understood. I’ll treat these responses as informed peer review: useful for exposing distinctions, but subordinate to our direction document, evidence inventory, notes, and live code.

This reply is largely consistent with our findings. I would adopt its core recommendation, with two important refinements.

First, `ParticipantKey` should be called an authored semantic address, not the complete logical identity. The authority-established `ParticipantRef` is what identifies one accepted participant incarnation:

```text
ParticipantKey
  authored declaration address
  reusable across equivalent strategy filings/runs

ParticipantRef
  identity of one accepted declaration in one run authority
  stable across materialization, replacement, and preparation

revisions
  changing binding, relationship, and route state of that identity
```

Second, retirement should not make a reference meaningless everywhere. It should close the reference to further live binding, route, or relationship operations while remaining resolvable in transition history and artifact provenance. That preserves the settled separation between current arrangement, run history, and durable provenance.

I would therefore record the working invariant as:

> Strategy authors assign semantic participant keys. Establishing the authored arrangement for a run causes the binding authority to validate each declaration and issue a run-scoped participant reference identifying that accepted declaration incarnation. The reference survives materialization, identity-preserving replacement, preparation, and route rebinding. Retirement prevents further current-state operations through the reference but does not erase its historical or provenance meaning. Binding, relationship, and route revisions describe changing state associated with the reference rather than changing its identity.

I would not yet promise that a retired key may be redeclared in the same authority. The contract can initially reject key reuse. If later allowed, it must always issue a new reference; the old reference can never revive.

The other refinements from the reply are also sound:

- The strategy authors the arrangement; the authority validates and establishes it. The authority never invents participants.
- An authority-wide snapshot revision is a conservative freshness token, not a replacement for participant, relationship, and route revisions.
- For in-place preparation, we should settle the invariant before choosing `preparing`, leases, or another state representation:

> An operation permitted to mutate an authoritative realization in place must withdraw the affected freshness guarantees before mutation begins. Failure cannot silently restore those guarantees without establishing that the realization remains valid.

- Prepared execution routes belong to the binding authority; optimizer/scheduler runtime and backend coordination handles remain separate Trainer-owned runtime concerns.

So my answer to the original question is now: yes, the key/reference split has earned its existence, with `ParticipantRef` representing the run-scoped logical identity and `ParticipantKey` representing its authored address.

<a id="m008"></a>
## 008 User - 2026-08-22T00:34:11.630Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 194](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:194).

Okay I understand the topic now. How should we start working by the way? is this multiple openspec items or one big one?


<a id="m009"></a>
## 009 Assistant (commentary) - 2026-08-22T00:34:14.759Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 198](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:198).

I’m using the OpenSpec exploration workflow here because this is exactly the boundary between preliminary design and deciding the change structure. I’ll compare one change versus several against the settled architecture and the repo’s milestone/review rules, without creating artifacts yet.

<a id="m010"></a>
## 010 Assistant (commentary) - 2026-08-22T00:34:49.604Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 210](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:210).

One existing OpenSpec is directly adjacent: `expand-model-metadata-identity-and-structure`. I’m checking its declared scope before recommending whether this architecture work belongs inside it or should govern it as a separate dependency.

<a id="m011"></a>
## 011 Assistant (final_answer) - 2026-08-22T00:35:35.136Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 221](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:221).

I’d make this one coherent OpenSpec change containing multiple capability specs and implementation milestones—not several independent OpenSpec changes, and not a big-bang implementation.

```text
one OpenSpec change
├── proposal.md
├── design.md
├── specs/
│   ├── training-strategy-contract/
│   ├── binding-authority/
│   ├── runtime-preparation/
│   ├── optimization-runtime/
│   ├── training-step-exchange/
│   └── strategy-conformance/
└── tasks.md
    ├── 1. Contract foundations
    ├── 2. Binding authority
    ├── 3. Loading/materialization integration
    ├── 4. Runtime preparation
    ├── 5. Optimization exchange
    ├── 6. Step/objective exchange
    ├── 7. Family and adapter migration
    └── 8. Remove legacy authorities and verify
```

Why one change: the exchanges constrain each other. Binding revisions determine what preparation consumes; preparation determines optimization runtime; optimization ownership determines what the step may return or do. Splitting those into independently designed OpenSpecs would risk freezing partial APIs or relying on temporary contracts that contradict the final architecture.

Why it is not “one huge thing”: each numbered task section is a reviewable milestone. We implement, test, run `review-mcp`, hand off, and only then proceed. The contract can be final-shaped from the beginning while production migration happens incrementally.

Before creating that change, we should finish the remaining pre-OpenSpec design:

1. Complete the binding exchange.
2. Complete the runtime-preparation exchange.
3. Settle the standard Trainer-owned optimization profile and explicit research extension.
4. Define the optimization exchange.
5. Define the step exchange.
6. Turn the existing pressure scenarios into concrete conformance requirements.
7. Decide the minimum migration sequence that keeps the active path working between milestones.

That follows the direction document’s own gate: the four exchanges and optimization ownership should be defined before the OpenSpec locks down Python/API and migration choices.

Some things should remain separate follow-up changes:

- feature-catalog/package reorganization;
- arbitrary model-component composition or the “component grab box”;
- implementing hypothetical EMA, teacher/student, or extra compiled routes;
- precise dependency-targeted invalidation beyond the conservative correct baseline;
- external strategy plugins and contract version negotiation;
- more ambitious persistence/publication behavior beyond migrating current products correctly.

The existing `expand-model-metadata-identity-and-structure` OpenSpec should not absorb this architecture work. It is already a large metadata change with a different authority. Its loading/provenance sections currently assume an “authoritative loaded-component surface”; those sections should eventually be amended to consume the new binding/materialization exchange rather than independently defining runtime authority. Fortunately, only its baseline tasks are complete, so this dependency can be corrected before its loading-contract milestone begins.

For Beads, I’d use:

- `sd-scripts-syv` until this preliminary design is ready to become an OpenSpec;
- then a new implementation epic/feature bead linked to the OpenSpec;
- child beads corresponding to the numbered implementation milestones;
- a dependency or explicit gate from the model-metadata loading milestone to the new binding-contract milestone.

So the immediate start is not code or an OpenSpec proposal yet. We should finish the binding and preparation exchange design in the direction/notes, then move through optimization and step. Once those are coherent, we create one governing OpenSpec change with multiple specs and milestone sections.

<a id="m012"></a>
## 012 User - 2026-08-22T00:42:29.222Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 227](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:227).

We need to bound any progress in something concrete, like a file or evolving openspec, because 258k tokens are available before the context compacts. we are at 57% 148k/2568k and it usually does it at 218k or so. Fortunately this is with already having read a lot of the relevant things

so what do u want to do?


<a id="m013"></a>
## 013 Assistant (commentary) - 2026-08-22T00:43:05.421Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 234](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:234).

I want to create a dedicated evolving pre-OpenSpec design now: `docs_design/models-strategy-trainer/strategy_contract_exchange_design.md`.

That gives us a compaction-safe place for the concrete binding, preparation, optimization, and step exchanges while preserving the existing document roles. I’ll seed it with the settled Q1–Q5 constraints, clearly label working versus settled downstream decisions, record the first exchange tables, and link it additively from the direction and notes. Once those four exchanges are coherent, this file becomes the source for the single governing OpenSpec rather than forcing a premature proposal today.

<a id="m014"></a>
## 014 Assistant (commentary) - 2026-08-22T00:45:13.194Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 245](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:245).

The exchange design is now concretely bounded in the repository. I’ve kept every new point visibly classified as inherited, working, or open, so future compaction won’t accidentally turn provisional mechanics into settled architecture. I’m checking the resulting document links and diff now, then I’ll identify the exact first decision to work through.

<a id="m015"></a>
## 015 Assistant (final_answer) - 2026-08-22T00:45:48.444Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 253](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:253).

I’ve started the concrete record in [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md).

It now contains:

- document authority and completion rules;
- settled Q1–Q5 inputs;
- authored key versus run-scoped participant identity;
- revision and conservative freshness semantics;
- binding operation tables;
- materialization and replacement rules;
- runtime-preparation plan/result boundaries;
- in-place mutation failure invariants;
- current-code evolution mapping;
- working decision and open-question registers;
- optimization/step placeholders;
- an explicit compaction handoff.

I also added only small links/checkpoints to the existing direction and notes, preserving their accumulated content. `git diff --check` passes; no code or OpenSpec artifacts were changed.

From here, I want us to use this rhythm:

1. Discuss one concrete exchange question.
2. Immediately record the accepted/refined result in this file.
3. Keep provisional mechanics visibly marked as working.
4. Continue until binding, preparation, optimization, and step are coherent.
5. Generate the governing OpenSpec from this record.

The first thing I want to settle is initial arrangement establishment:

```text
authored strategy definition
  -> participant and relationship declarations
  -> contract validation
  -> per-run authority establishment
  -> ParticipantRefs + initial arrangement snapshot
```

Specifically, we should define the minimum `ParticipantDeclaration`: what semantic obligations it must carry so replacement and fulfillment can be validated, while excluding live modules, source checkpoints, device/dtype, trainability, optimizer state, and other runtime facts. That gives every later exchange a stable foundation.

<a id="m016"></a>
## 016 User - 2026-08-22T01:32:22.992Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 259](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:259).

Okay, what to look into first?


<a id="m017"></a>
## 017 Assistant (commentary) - 2026-08-22T01:32:35.442Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 263](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:263).

First, I want to inspect the repository’s existing declaration and identity surfaces—not preparation yet. I’m using the codebase graph to compare family component declarations, adapter target identities, metadata component identities, and optimization references. That will tell us what `ParticipantDeclaration` can evolve from and which current concepts must stay separate.

<a id="m018"></a>
## 018 Assistant (commentary) - 2026-08-22T01:33:23.214Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 317](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:317).

The first inspection already gives us a useful boundary: the repo has several “identity-like” types, but each serves a different concern. I’m recording that evidence in the exchange design before proposing declaration fields, so we do not accidentally make metadata IDs, optimizer targets, or family roles into the canonical participant contract.

<a id="m019"></a>
## 019 Assistant (final_answer) - 2026-08-22T01:33:56.679Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 328](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:328).

First: define the minimum `ParticipantDeclaration` through scenarios, before choosing Python types.

I inspected the current identity-like surfaces and recorded the findings in [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md). They serve different concerns:

- `LoadedModelComponentSpec`: family component declaration.
- Metadata component IDs: durable observations/provenance.
- `OptimizationTargetRef`: live substructure selected for optimization.
- Adapter targets: relationship endpoints.
- Train flags: optimization intent.

None should become the participant contract wholesale.

Next, we should write four example declarations side by side:

1. Ordinary bound SDXL denoiser.
2. Declared-but-deferred SD3 denoiser.
3. State-bearing adapter with a host relationship.
4. Teacher and student initialized from the same source.

For each, ask:

```text
What must be known when the strategy is authored?
What must be validated when the run authority is established?
What may remain unknown until materialization?
What can change without changing participant identity?
What requires retirement and a new declaration?
```

That should derive the declaration’s minimum fields. My initial boundary is:

```text
probably declaration-owned
  authored key
  semantic obligations
  lifecycle/materialization expectation
  whether independently executable
  declared execution-route requirements

separate relationship declaration
  typed relationship identity
  participant endpoints
  intended relationship semantics

not declaration-owned
  public/display name
  Python module
  source checkpoint or realization
  device/dtype/wrapper
  trainability/frozen state
  optimizer groups
  active/detached relationship state
  current revisions
```

The key unresolved piece is “semantic obligations”: enough structured meaning to validate identity-preserving replacement, without inventing a universal `denoiser | vae | text_encoder` taxonomy. That is what I’d pressure-test first using those four declarations.

<a id="m020"></a>
## 020 User - 2026-08-22T20:11:29.711Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 336](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:336).

I don't really know how to approach the it so you have to guide us


<a id="m021"></a>
## 021 Assistant (commentary) - 2026-08-22T20:11:36.813Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 340](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:340).

I’ll drive the design from here and bring you concrete choices with consequences, rather than asking you to invent the contract vocabulary. I’m using the OpenSpec exploration stance while we develop the pre-OpenSpec record: I’ll pressure-test a candidate `ParticipantDeclaration` against current and future scenarios, record the reasoning immediately, and only stop for decisions where your intent materially changes the architecture.

<a id="m022"></a>
## 022 Assistant (commentary) - 2026-08-22T20:12:40.013Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 354](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:354).

I’ve pressure-tested the declaration against the four cases. The important result is that “deferred” and “required before training” should not be baked into participant identity. They are lifecycle/fulfillment constraints over the participant. I’m recording that separation now, along with a concrete minimum declaration shape and why it avoids a universal SD role enum.

<a id="m023"></a>
## 023 Assistant (final_answer) - 2026-08-22T20:13:41.437Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 365](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:365).

You don’t need to supply the approach—I’ll derive it from the settled direction, code evidence, and pressure scenarios, then bring you understandable architectural choices.

I’ve completed the first guided step and recorded it in [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md).

The four scenarios produced this working declaration:

```text
ParticipantDeclaration
  authored key/address
  explicitly wired semantic compatibility filing
  zero or more execution-route declarations
```

In plain language:

- The key says which authored participant this is: `model.denoiser`, `adaptation.main`, etc.
- The semantic filing contains the strategy-wired rules for deciding whether loaded or replacement state still fulfills that participant’s intended meaning.
- Route declarations say whether it needs no callable route, the normal route, or a materially different named route.

Readiness is separate:

```text
declaration
  model.denoiser exists

binding lifecycle
  declared/unbound -> bound

fulfillment requirement
  normal route must be prepared before training-step execution
```

That matters because an eager denoiser and a deferred SD3 denoiser can have the same declaration while materializing at different times. Likewise, a LoRA adapter can be a real participant without needing an independent forward route.

The next thing I’ll investigate and design is the semantic compatibility filing itself. The likely answer is a combination of:

- typed semantic facts;
- explicitly wired validation behavior;
- no universal `denoiser | vae | text_encoder` enum;
- no automatic feature resolver;
- no arbitrary callback hidden inside a passive state record.

I’ll compare that against current strategy validation, family feature composition, adapter construction, and research-strategy needs, then record a recommended boundary. You only need to intervene when the resulting behavior does not match how you want strategy authoring or experimentation to feel.

<a id="m024"></a>
## 024 User - 2026-08-22T20:24:57.085Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 371](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:371).


# Files pasted by the user:

## "1. Ordinary SDXL denoiser When the strategy is authored, we need to know: key: …": /home/imi/.codex/attachments/45af2535-7ae5-4419-81ff-6edc8801ac7a/pasted-text.txt

## "The four scenarios did exactly what they were supposed to do. They ruled out th…": /home/imi/.codex/attachments/4932f456-db28-47cd-b601-b5142d9e0e3b/pasted-text.txt

## My request:
2 snippets from external the view for recent state


Readable attachment copy: [2026-08-22-declaration-view-1.md](../snapshots/2026-08-22-declaration-view-1.md).

Readable attachment copy: [2026-08-22-declaration-view-2.md](../snapshots/2026-08-22-declaration-view-2.md).

<a id="m025"></a>
## 025 Assistant (commentary) - 2026-08-22T20:25:02.060Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 375](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:375).

I’ll read both as external peer-review snapshots against our current exchange record. I’ll keep anything that sharpens the design, flag anything that conflicts with the normative direction or live evidence, and immediately capture accepted refinements in the evolving file. I’m using the OpenSpec exploration stance for that review; no production implementation is involved.

<a id="m026"></a>
## 026 Assistant (commentary) - 2026-08-22T20:25:40.718Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 387](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:387).

Both reviews are useful, and they expose four real refinements. I’m adopting three directly: route requirements become separate authored declarations rather than fields inside `ParticipantDeclaration`; relationship identity becomes an explicit working decision; and coarse freshness covers every snapshot-visible authority transition. I’m also recording the self-invalidating in-place-preparation tension as a required coordination problem rather than hiding it behind the existing snapshot wording.

<a id="m027"></a>
## 027 Assistant (final_answer) - 2026-08-22T20:27:06.432Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 403](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:403).

Both external views were useful and mostly aligned with our direct findings. I reconciled them into the evolving design and chronological notes.

Accepted refinements:

- `ParticipantDeclaration` is now smaller:

```text
ParticipantDeclaration
  authored key
  semantic compatibility filing
```

- Execution routes are separate associated declarations:

```text
ExecutionRouteDeclaration
  participant key
  route meaning and preparation/freshness requirements
```

This lets zero routes naturally represent a state-only adapter and lets capabilities add routes without redefining participant identity.

- `deferred` is not a binding state. Materialization is simply:

```text
declared/unbound -> bound
```

Readiness constraints explain why remaining unbound is temporarily acceptable.

- Relationship identity is now explicit rather than implied:

```text
RelationshipKey -> run-scoped RelationshipRef -> relationship revisions
```

This is justified by independently evolving effects, parallel relationships, history, and persistence—not merely symmetry with participants.

- Conservative snapshot freshness now advances after any transition changing snapshot-visible authority state, including relationship and route changes.

- Optimizer/scheduler entries in preparation are explicitly only backend-pressure placeholders. Their exact types and construction remain subordinate to optimization ownership.

- In-place preparation now records the newly exposed coordination requirement:

```text
plan resolved from R17
  -> guarantees withdrawn before mutation
  -> authority becomes R18
  -> completion cannot merely require R17 to remain current
```

Such preparation therefore needs an authority-recognized attempt/transition identity distinct from its source snapshot. Whether that becomes a lease, attempt reference, or staged transition remains open.

One recommendation from the first response was not adopted: putting materialization requirements inside `ParticipantDeclaration`. Our stronger conclusion is that eager versus deferred materialization changes fulfillment timing, not participant meaning. The declaration’s semantic filing still defines what valid bound state means; separate readiness constraints define when it must exist.

These refinements are recorded in [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md), with the review effects also captured in [notes.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md). `git diff --check` passes.

The next substantive design problem remains the semantic compatibility filing: what typed facts and explicitly wired behavior validate materialization and identity-preserving replacement without creating a universal taxonomy or automatic resolver.

<a id="m028"></a>
## 028 User - 2026-08-22T20:30:19.783Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 409](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:409).

So were the viewpoints useful? how come they can be useful despite us having access to the code and not the external check. is it because we don't actually have code based on the new direction discussion


<a id="m029"></a>
## 029 Assistant (final_answer) - 2026-08-22T20:30:42.811Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 413](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:413).

Yes—largely because the new direction does not exist in production code yet.

The current code is a pressure oracle:

```text
current code
  tells us what information and lifecycle behavior must survive

direction document
  tells us what the target architecture means

external review
  tests whether our proposed reasoning is internally coherent
```

Direct code access gives us an enormous advantage for questions like:

- Does Accelerate replace or mutate this object?
- Which component handles are currently retained?
- Can adapters exist without their own forward route?
- What does checkpointing actually extract?
- Which mode methods mutate Trainer?
- Would a proposed migration preserve SD/SDXL/SD3 behavior?

But code cannot directly answer normative questions such as:

- Should readiness be part of participant identity?
- Should a relationship have independent runtime identity?
- What does identity-preserving replacement mean?
- Who ought to own prepared routes?
- Which concepts should become separate declarations?

The active code embodies the architecture we are trying to replace. If we simply inferred the target from it, we would reproduce `TrainingMode`, Trainer family fields, and `LoadedModelComponent.module` under new names.

The external viewpoints were useful as fresh logical reviews of our written model. Less context can occasionally help here: they are more likely to notice a contradiction in the document instead of unconsciously filling the gap with repository knowledge.

Concrete examples:

- They noticed that we had deliberately separated deferred readiness from binding state, but one table still said `unbound/deferred -> bound`.
- They noticed that `RelationshipRef` had appeared throughout the exchange without being recorded as an explicit working decision.
- They noticed that “binding-affecting transition” was too narrow for whole-snapshot freshness because relationship and route changes also matter.
- They exposed the self-invalidating preparation sequence where withdrawing freshness changes the authority state after the plan’s source snapshot was captured.

Those are logical consistency findings; they do not require repository access.

Their lack of context also produced suggestions we did not accept unchanged:

- One response put materialization requirements back inside `ParticipantDeclaration`; our direction and scenario analysis supported keeping readiness separate.
- One response questioned whether relationships need stable identity, but Q3/Q4 already require relationship identity, independent transitions, history, and persistence coverage. We used the critique to make the decision explicit, not to reopen it.

So the authority order is:

```text
settled direction
  > current code/inventory evidence
  > our concrete exchange reasoning
  > external critique as a hypothesis/check
```

External reviews are valuable because architecture is partly a consistency and meaning problem. Code access makes our final judgment better, but it does not make us immune to missing contradictions in a new design that has not yet been implemented.

<a id="m030"></a>
## 030 User - 2026-08-22T20:32:07.652Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 419](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:419).

Yeah that explanation makes sense. we are at about 205k/258k

what did you mean about semantic compatibility filing?


<a id="m031"></a>
## 031 Assistant (commentary) - 2026-08-22T20:32:33.457Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 424](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:424).

“Filing” does not mean a file on disk, registry entry, or user configuration. It means the strategy’s explicitly authored answer to: “What must a concrete realization satisfy to count as this participant?” I’m adding that plain-language definition to the exchange record now so the term survives compaction without ambiguity.

<a id="m032"></a>
## 032 Assistant (final_answer) - 2026-08-22T20:33:10.301Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 435](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:435).

“Semantic compatibility filing” is a working term for the strategy’s authored rules answering:

> What must concrete state satisfy to count as this participant, and when can replacement state preserve the participant’s identity?

“Filing” does not mean a file, metadata record, registry, configuration, or plugin. It follows our contract vocabulary: the strategy explicitly supplies one valid fulfillment of a contract concern.

Conceptually:

```text
ParticipantDeclaration
  key = model.denoiser
  semantics = strategy-wired compatibility rules
```

Later:

```text
loader produces candidate module/state
             ↓
semantic rules validate candidate
             ↓
typed compatibility result
             ↓
binding authority accepts or rejects materialization
```

For an SDXL denoiser, those rules might establish that the candidate:

- can be used through the predictor behavior authored by this strategy;
- accepts/produces the tensor semantics expected by the selected representation, conditioning, and objective features;
- supports the access meanings required for training and persistence;
- satisfies applicable structural and precision constraints.

That does not mean:

```text
kind = DENOISER
class must be UNet2DConditionModel
source must be this checkpoint
```

A different implementation can replace the original while retaining identity if the strategy’s explicitly wired behavior can validate and use it as the same participant.

For a LoRA-like adapter, the semantic filing would be different:

- it owns independently managed adaptation state;
- it can resolve and establish its declared host effects;
- it provides the state access needed for optimization and artifact persistence;
- it may require no independent execution route.

Teacher and student might use the same predictor compatibility filing while remaining separate participants because their keys and run-scoped references differ. The filing describes compatible meaning; it does not create identity.

The likely conceptual contents are:

```text
typed requirements and claims
  inspectable compatibility constraints

explicitly wired validation behavior
  evaluates concrete materialization/replacement
  returns a typed acceptance or rejection
```

We need both because:

- labels such as `denoiser` are too weak;
- Python class or protocol checks only prove shape/method presence;
- the central binding authority must not learn every model-family concept;
- arbitrary semantic compatibility cannot always be inferred through reflection.

Some checks will be generic and machine-verifiable. Others will be feature-specific or authored assertions backed by strategy conformance tests. The authority’s generic responsibility is only to ensure that the declaration’s explicitly wired compatibility boundary authorized the transition; it does not decide what “denoiser,” “adapter,” or “teacher” means itself.

The exact Python shape remains open. It might eventually be a typed immutable requirements value paired with strategy-wired validator behavior. I would avoid one callback embedded in a passive dataclass, because the semantic facts should remain inspectable and testable separately from the behavior that validates a live candidate.

I added this clarification directly to [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md).

<a id="m033"></a>
## 033 User - 2026-08-22T20:49:13.848Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 441](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:441).

denoiser is a role label no?


<a id="m034"></a>
## 034 Assistant (commentary) - 2026-08-22T20:49:31.635Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 446](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:446).

Yes—exactly. In the current code, `denoiser` is a generic role label, and my example reused it inside a participant address without making the distinction explicit. The full authored key may contain a descriptive role-like word, but identity comes from the complete declared address/incarnation, never from the `denoiser` label. I’m correcting that ambiguity in the design record.


> Compaction boundary at raw line 450, 2026-08-22T20:50:12.666Z. The generated summary is omitted.

<a id="m035"></a>
## 035 Assistant (final_answer) - 2026-08-22T20:50:36.731Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 473](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:473).

Yes—exactly.

In the current code:

- `roles=("denoiser",)` is a generic role label.
- `key="denoiser"` happens to reuse the same word, but that is a convention—not an identity rule.
- `student.denoiser` and `teacher.denoiser` may share the `denoiser` role while remaining distinct participants.
- Identity comes from the complete authored `ParticipantKey`, then its authority-established run incarnation (`ParticipantRef`).

I clarified this explicitly as EX-012 in the [exchange design](</home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md>). This overlap between today’s key and role strings is precisely the sort of migration detail we must not accidentally elevate into the new architecture.

<a id="m036"></a>
## 036 User - 2026-08-23T01:04:04.639Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 481](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:481).

is that a useful clarification?


<a id="m037"></a>
## 037 Assistant (final_answer) - 2026-08-23T01:04:14.137Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 485](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:485).

Yes. It was already implicit, but making it explicit prevents a likely design mistake.

A role is many-to-one: both `teacher.denoiser` and `student.denoiser` can have the `denoiser` role. A participant key must uniquely address one authored participant. Because current code uses `"denoiser"` for both a key and a role, that accidental overlap could otherwise leak into the new binding API, compatibility dispatch, or replacement rules.

So EX-012 doesn’t add a large new concept; it protects an important boundary:

`role/classification ≠ authored address ≠ run-scoped identity`

<a id="m038"></a>
## 038 User - 2026-08-23T01:05:04.673Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 491](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:491).

Okay, that's nice. But wait, what were we doing?


<a id="m039"></a>
## 039 Assistant (final_answer) - 2026-08-23T01:05:21.139Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 495](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:495).

We were designing the concrete contracts for the model–strategy–Trainer architecture before turning them into OpenSpec.

The durable working document is the [strategy contract exchange design](</home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md>). The plan is still one governing OpenSpec change with several implementation milestones, not several disconnected changes.

We were working through the first two exchanges:

1. **Binding exchange** — how a strategy declares participants and relationships, how one authority assigns run-scoped identities, and how materialization/replacement occurs without Trainer inventing identity.
2. **Runtime-preparation exchange** — how Accelerate/DeepSpeed preparation produces fresh execution routes and Trainer-owned runtime state without mutating everything through `Trainer`.

We had zoomed into the minimum `ParticipantDeclaration`:

```text
authored ParticipantKey
explicit semantic compatibility filing
```

The `denoiser` question arose while ensuring that:

```text
role label ≠ participant address ≠ run-scoped ParticipantRef
```

So it was a useful local correction, but not the main task.

The next concrete step is to finish the binding exchange by specifying **authority construction and initial arrangement establishment**:

```text
authored participant/relationship declarations
        ↓
binding authority validates and accepts them
        ↓
ParticipantRef / RelationshipRef identities
        ↓
initial declared-but-unbound snapshot
```

That should define the exact request, result, validation failures, ownership, and what projections the strategy and Trainer receive. Then we can design materialization and replacement against that established arrangement.

<a id="m040"></a>
## 040 User - 2026-08-23T01:07:30.034Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 501](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:501).

"The next substantive design problem remains the semantic compatibility filing: what typed facts and explicitly wired behavior validate materialization and identity-preserving replacement without creating a universal taxonomy or automatic resolver."

that was the last "next" type of line from the topic


<a id="m041"></a>
## 041 Assistant (final_answer) - 2026-08-23T01:07:44.993Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 505](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:505).

You’re right—that is where we actually were. I skipped ahead in my recap.

The `denoiser` clarification was a small part of defining the **semantic compatibility filing**. It established one negative constraint: the filing cannot be selected from, or satisfied merely by, a generic role label.

What remains is to define its positive contract:

```text
candidate materialization or replacement
        ↓
typed facts about that candidate
        ↓
strategy-wired compatibility behavior
        ↓
typed acceptance evidence or rejection
```

We still need to decide:

- Which typed facts describe a candidate realization?
- What additional evidence is required for replacement?
- How does the filing express strategy-specific requirements without becoming a universal component taxonomy?
- What typed result proves that the binding authority may accept the transition?
- How are multiple feature-specific checks composed without an automatic resolver?

The best next move is to draft that exact request/result shape and pressure-test it against ordinary SDXL, deferred SD3, an injected adapter, and teacher/student participants. Only after that should we move to authority construction, because the authority’s acceptance behavior depends on this contract.

<a id="m042"></a>
## 042 User - 2026-08-23T03:40:33.896Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 511](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:511).

We didn't discuss and record anything about these?


<a id="m043"></a>
## 043 Assistant (commentary) - 2026-08-23T03:40:40.207Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 515](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:515).

I’m checking the design record and chronological notes directly. I’m using the OpenSpec exploration workflow because we’re recovering the precise boundary between what was already decided and what remains genuinely open; I won’t treat my recap as the source of truth.

<a id="m044"></a>
## 044 Assistant (final_answer) - 2026-08-23T03:41:31.192Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 541](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:541).

Yes—we discussed and recorded quite a lot. I understated the existing work in my previous answer.

The [semantic compatibility filing section](</home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:195>) already establishes that:

- It contains **authored, versioned semantic rules** for materialization and identity-preserving replacement.
- It combines **typed requirements/claims** with **explicitly wired validation behavior**.
- The strategy or its deliberately selected features wires that behavior.
- It is not selected by a role enum, Python class, registry, reflection, or automatic resolver.
- Reusable standard filings and custom research filings may both reach the same authority-facing boundary.
- Several participants may share a filing without sharing identity.
- Validation may combine:
  - centrally testable structural facts;
  - feature-specific checks;
  - authored assertions backed by conformance tests.
- The authority requires a typed fulfillment/compatibility result but does not understand model-specific concepts itself.
- The filing must answer five recorded questions covering realization, replacement, permitted access meanings, required revalidation, and unfulfilled obligations.

We also recorded concrete examples:

- SDXL predictor/input/output, access, structural, and precision requirements;
- adapter state, target/effect, optimization, and persistence requirements;
- teacher and student sharing compatibility semantics but retaining distinct identities.

What remains open is narrower:

- the exact candidate-facts types;
- the concrete request/result objects;
- how reusable and feature-specific checks compose in Python;
- what evidence a successful result carries;
- how replacement proposals include the previous accepted realization;
- whether the filing is represented as a contract value plus validator, a protocol, or another explicit mechanism.

So the real next step is not “invent the semantic compatibility boundary.” It is:

> Confirm the already drafted boundary, then turn it into a concrete candidate-validation request/result design.

The document’s “Immediate Discussion Order” still describes step 2 too broadly and should be refined once we agree this reading is correct.

<a id="m045"></a>
## 045 User - 2026-08-23T04:30:23.557Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 547](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:547).

Can you look into that? after that we have to wrap up for now because it's late


<a id="m046"></a>
## 046 Assistant (commentary) - 2026-08-23T04:30:31.638Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 551](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:551).

Yes. I’ll treat this as the last bounded design pass tonight: determine how much of the compatibility-validation exchange is already settled, inspect the closest current typed boundaries, and record a concrete request/result proposal plus the remaining decisions. Then I’ll leave the bead and working documents in a clean handoff state rather than advancing into authority construction. The OpenSpec exploration skill keeps this to design artifacts—no production implementation.

<a id="m047"></a>
## 047 Assistant (commentary) - 2026-08-23T04:30:37.079Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 557](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:557).

The Beads skill is relevant because we’re ending the session after this pass; it requires the durable issue to carry the exact next boundary. The codebase-memory skill is also relevant because candidate facts should be derived from existing typed seams, not invented only from prose. I’m using a task-directed verification scope around model components, strategy contracts, binding-like results, and compatibility validation.

<a id="m048"></a>
## 048 Assistant (commentary) - 2026-08-23T04:32:19.738Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 620](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:620).

The inspection is narrowing the problem cleanly. Current code gives us three useful constraints: `LoadedModelComponent` proves the candidate must carry a live object but is too weak because that object is `Any`; metadata facts prove durable observation is not live acceptance evidence; optimization target refs prove consumer-local paths/tags are not participant compatibility. The new exchange therefore needs a proposal-scoped typed evidence boundary rather than expanding any of those existing types.

<a id="m049"></a>
## 049 Assistant (commentary) - 2026-08-23T04:33:15.041Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 641](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:641).

The key design conclusion is now strong enough to record: compatibility validation should be authority-invoked, proposal-scoped, and non-transferable. The universal part is only the transition envelope; model-specific candidate evidence remains typed by the explicitly authored filing. This avoids both extremes—the current `Any` tuple and a universal model taxonomy. I’m capturing it as working decisions, not pretending we have selected final Python classes.

<a id="m050"></a>
## 050 Assistant (commentary) - 2026-08-23T04:34:47.962Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 662](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:662).

The design record now contains the bounded result. I’m doing the wrap-up checks next: validate the Markdown diff, verify that only the intended design/notes files changed, and append a concise resume checkpoint to `sd-scripts-syv` without closing it—the architecture work is still in progress.

<a id="m051"></a>
## 051 Assistant (final_answer) - 2026-08-23T04:35:25.144Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 676](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:676).

I looked into it and recorded the result.

The [exchange design](</home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:295>) now defines the compatibility boundary as working decisions EX-013–EX-017:

- a small common proposal envelope plus filing-specific typed evidence;
- authority-invoked validation;
- single-use, proposal-scoped assessments;
- assessment separated from the authoritative transition result;
- explicit strategy-authored validation composition;
- no universal model taxonomy, role-based lookup, or automatic resolver;
- deferred `None` is not treated as a candidate;
- replacement validation receives the previous binding and revision.

The chronological evidence and reasoning are preserved in [notes.md](</home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:477>). I also updated bead `sd-scripts-syv`, leaving it in progress.

Still open for next time:

- exact Python generic/protocol representation;
- authority construction and initial arrangement establishment;
- complete atomic materialization/replacement types;
- rules for changing a filing version during a run.

This remains pre-OpenSpec exploration, so no production code or OpenSpec requirements were created. Only the two design documents are modified, and `git diff --check` passes. We can resume by reviewing EX-013–EX-017, then move directly into authority construction.

<a id="m052"></a>
## 052 User - 2026-08-23T04:56:56.587Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 683](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:683).

why are the names of things so odd? like "ParticipantDeclaration" and "RealizedModelComponentFacts" and similar

<a id="m053"></a>
## 053 Assistant (final_answer) - 2026-08-23T04:57:20.631Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 687](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:687).

Because they come from two different naming pressures, and both favor precision over naturalness.

`RealizedModelComponentFacts` is an existing metadata-system name:

- **Realized** distinguishes a model actually loaded for one run from a catalog/source model.
- **Component** means one observed top-level model component.
- **Facts** means immutable, persistable observations—not live authoritative state or behavior.

It is cumbersome because each word prevents a particular ownership misunderstanding.

`ParticipantDeclaration` is newer working vocabulary from this design:

- **Participant** avoids assuming everything is a conventional model component or executable module. Adapter-owned state can participate without an independent forward route.
- **Declaration** distinguishes the authored statement that something exists from its later materialization and binding.

That said, your reaction is valid. Names such as `SemanticCompatibilityFiling`, `CandidateBindingProposal`, and `CompatibilityAssessment` are currently analytical labels, not recommended final Python API names. They are verbose because we are separating concepts that the current code conflates.

Before OpenSpec, we should perform a vocabulary/API pass. Likely shorter candidates include:

| Working design term | Possible code-facing name |
| --- | --- |
| `ParticipantDeclaration` | `ParticipantSpec` or `ComponentSpec` |
| `SemanticCompatibilityFiling` | `BindingContract` |
| `CandidateBindingProposal` | `BindRequest` |
| `CompatibilityAssessment` | `CompatibilityResult` or `BindingCheck` |
| authority transition result | `BindResult` |
| `ParticipantRef` | probably keep, unless we settle on `ComponentRef` |

So the oddness is partly intentional scaffolding and partly inherited metadata terminology. It would be a mistake to copy all these noun piles directly into production code without a naming pass.

<a id="m054"></a>
## 054 User - 2026-08-23T05:00:01.398Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 693](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:693).

I really hope we can work in a way where we'll end up with readable code that's intuitive and acutally makes sense when put together. When worknig on the metadata system I didn't have much input because I'm not familiar with how it should work, but going from the current strategy/trainer that I understand to something more abstract that is hard to read wouldn't be a success. we'll see when we get to working on the actual code I guess


<a id="m055"></a>
## 055 Assistant (final_answer) - 2026-08-23T05:00:27.355Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 697](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:697).

That concern should be a design constraint now, not something we discover after implementation.

A successful result should make the ordinary path easier to follow:

```text
strategy declares what exists and what training means
strategy loads or constructs its model state
Trainer prepares the declared training subjects
Trainer runs the strategy’s step
Trainer coordinates saving
```

The binding authority may need strict internal machinery for replacement, freshness, and distributed preparation, but that machinery should not dominate an SDXL strategy or the Trainer loop. Most strategy authors should not manually manage references, revisions, assessments, clauses, and transition tokens during ordinary training.

A useful readability gate would be:

- Can you open the SDXL strategy and understand what gets trained?
- Can you follow the Trainer’s main lifecycle without learning a miniature framework?
- Are common operations named plainly: load, bind, prepare, step, save?
- Is complicated authority machinery confined to lifecycle boundaries?
- Does adding a normal model family require model knowledge, rather than expertise in the binding implementation?
- Does the abstraction remove duplicated bookkeeping instead of merely renaming it?

The strict types should primarily protect seams. They should not force every caller to speak internal protocol vocabulary.

And the current odd names are already useful feedback. If `ParticipantDeclaration` and `SemanticCompatibilityFiling` make the architecture harder to explain, we should not become attached to them. The metadata system’s `Facts` terminology also does not need to spread into runtime strategy code.

I would treat “the ordinary strategy and Trainer code must remain intuitive to someone who understands the current system” as a real acceptance criterion for the eventual OpenSpec. If the new code is formally correct but appreciably harder for you to read, that is an architectural failure, not merely unfamiliarity.

<a id="m056"></a>
## 056 User - 2026-08-23T05:01:50.209Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 703](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:703).

I don't want to make rules like that without seeing it


<a id="m057"></a>
## 057 Assistant (final_answer) - 2026-08-23T05:02:07.059Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 707](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:707).

That’s fair. We shouldn’t turn an understandable concern into a rigid rule before seeing the concrete design.

For now, it’s simply a pressure to remember: abstraction and readability need to be evaluated together. Once we have candidate Python shapes, we can compare a real SDXL strategy and Trainer flow side by side with today’s code. Then we can judge what is clearer, what is necessary ceremony, and what should be renamed or removed.

The current terminology remains design scaffolding—not a commitment to the eventual API. I haven’t recorded your comment as a normative requirement.

<a id="m058"></a>
## 058 User - 2026-08-23T05:03:18.632Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 713](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:713).

it's better to progress in an organic way. you can always finetune the end result after the base is there


<a id="m059"></a>
## 059 Assistant (final_answer) - 2026-08-23T05:03:23.734Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 716](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:716).

Agreed. We’ll let the architecture emerge from concrete exchanges and real model paths, then refine naming, ergonomics, and layout once there is something coherent to evaluate.

The important distinction is that the current design terms preserve meaning while we reason; they don’t dictate the final code vocabulary. We can simplify the surface once the underlying responsibilities are stable, instead of trying to perfect an API before its shape exists.

<a id="m060"></a>
## 060 User - 2026-08-23T17:34:03.456Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 722](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:722).

Let's get back to what we were doing. Where are we now?


<a id="m061"></a>
## 061 Assistant (final_answer) - 2026-08-23T17:34:32.235Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 726](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:726).

We are still in concrete pre-OpenSpec design. No production API has been chosen or implemented.

The settled foundation is Q1–Q5:

- stable logical participant identity;
- separate materialization, replacement, execution preparation, relationships, and arrangement changes;
- revision/freshness semantics;
- persistence projections;
- one contract-governed per-run binding authority behind the complete strategy.

The latest working design covers semantic compatibility validation:

```text
strategy-authored compatibility rules
        ↓
candidate materialization/replacement
        ↓
proposal-scoped validation
        ↓
authority accepts or rejects the transition
```

EX-013–EX-017 currently say that validation uses filing-specific typed evidence, is invoked by the authority, produces single-use evidence for one exact proposal, remains separate from the actual state transition, and is explicitly composed by the strategy rather than discovered automatically. These are working decisions, not final API names or accepted OpenSpec requirements.

Our exact frontier is now:

```text
strategy definition
    participant + relationship declarations
    compatibility rules
              ↓
        [CURRENT GAP]
              ↓
per-run authority
    accepted declarations
    ParticipantRefs / RelationshipRefs
    initial declared-but-unbound state
```

So the next topic is **authority construction and initial arrangement establishment**:

- Who constructs the per-run authority?
- What exactly does the complete strategy provide?
- When are authored keys converted into run-scoped references?
- How are duplicates, missing endpoints, conflicting declarations, and invalid compositions rejected?
- What initial snapshot is returned to the strategy and Trainer?

After that, we complete materialization/replacement requests and results around the compatibility work we just drafted. The durable state is in the [exchange design](</home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md>) and bead `sd-scripts-syv`.

<a id="m062"></a>
## 062 User - 2026-08-23T17:35:46.119Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 732](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:732).

why are compatibility rules int he strategy definition?


<a id="m063"></a>
## 063 Assistant (final_answer) - 2026-08-23T17:36:02.016Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 736](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:736).

Because the strategy defines what each authored participant is supposed to mean. But my phrase “strategy-authored compatibility rules” was too broad.

Ownership should be split:

| Concern | Owner |
| --- | --- |
| Generic lifecycle invariants—valid reference, current revision, legal transition | Core binding contract/authority |
| Reusable model or feature semantics—what qualifies as an SDXL predictor, adapter state, etc. | Repo-owned component/feature implementation |
| Which participant uses which semantic implementation | Strategy composition |
| Invoking validation and accepting/rejecting state | Binding authority |

A strategy should therefore not contain a pile of handwritten validation logic. It deliberately wires reusable semantics while being assembled:

```text
SDXL strategy
  model.denoiser
    uses SDXL predictor binding contract
```

The authority cannot choose that contract from the word `denoiser`, because a custom predictor or teacher may require different semantics. Trainer also should not learn SDXL anatomy. Conversely, the candidate producer cannot simply declare itself compatible.

So “in the strategy definition” means **explicitly selected during strategy composition**, not necessarily implemented inline by the strategy class. The core contract determines what must be validated; reusable model/features provide much of the behavior; the strategy says which behavior applies to its participants. This also preserves the earlier decision that the run does not automatically assemble or infer the strategy.

<a id="m064"></a>
## 064 User - 2026-08-23T17:40:08.163Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 742](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:742).

this is still too ambiguous


<a id="m065"></a>
## 065 Assistant (final_answer) - 2026-08-23T17:41:36.929Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 748](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:748).

Yes. The ambiguity exists because we have been using “compatibility” for three different questions.

| Question | When answered | Proper owner |
| --- | --- | --- |
| Is the assembled strategy coherent? | Strategy construction | Components/features contribute constraints; strategy construction validates their combination |
| Can this loaded object realize this declared participant? | Materialization/replacement | The participant/component definition |
| Is this state transition legal and current? | Runtime binding | Binding authority |

Those should not be one “semantic compatibility filing.”

A concrete SDXL shape might eventually resemble:

```python
strategy = TrainingStrategy(
    denoiser=SdxlUNetComponent(...),
    objective=EpsilonPrediction(...),
    conditioning=SdxlConditioning(...),
)
```

Here:

- `SdxlUNetComponent` knows how to load and validate an SDXL UNet realization.
- `EpsilonPrediction` and `SdxlConditioning` declare their compatibility constraints.
- Strategy construction verifies that those selected pieces agree.
- The binding authority handles only lifecycle correctness: the participant exists, the revision is current, replacement is legal, and the accepted state changes atomically.
- Trainer consumes the completed strategy contract without knowing SDXL anatomy.

For replacement:

```text
new UNet candidate
    ↓
SdxlUNetComponent checks that it can realize the same declared component
    ↓
affected feature/relationship constraints are revalidated
    ↓
binding authority accepts the replacement atomically
```

So there should not be an abstract bag of “compatibility rules in the strategy definition.” The strategy explicitly contains component and feature definitions, and those definitions bring their own requirements and validation behavior. The strategy’s job is composition, not manually wiring arbitrary validators.

This means the current design likely conflates:

- component realization contracts;
- strategy-composition constraints;
- authority transition rules.

That is the ambiguity you are detecting. We should split those before moving into authority construction; EX-013–EX-017 should remain provisional until that correction is pressure-tested against an actual SDXL strategy.

<a id="m066"></a>
## 066 User - 2026-08-23T17:43:17.263Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 754](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:754).

I still don't get it. Can you please properly look into this instead of coming up with it on the spot. Read our notes and check the code


<a id="m067"></a>
## 067 Assistant (commentary) - 2026-08-23T17:43:42.676Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 758](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:758).

You’re right. My last two answers were extrapolations from the new draft, not a reconstruction of what our architecture actually says and does. I’m going back to the normative direction, chronological notes, inventory evidence, and the real SDXL strategy/Trainer path. I’ll distinguish documented conclusions from my inference and won’t modify the design until the evidence supports a correction. I’m using the OpenSpec exploration and codebase-memory workflows for that read-only investigation.

<a id="m068"></a>
## 068 Assistant (commentary) - 2026-08-23T17:44:17.156Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 774](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:774).

The direction document already shows that my “component owns the runtime validator” correction was also too quick. Its established use of compatibility is primarily about whether the complete authored strategy fulfills the Trainer-defined contract and whether selected concerns/features agree—not yet a settled per-object validator API. I’m reading the surrounding sections now so we can recover the intended boundary exactly.

<a id="m069"></a>
## 069 Assistant (commentary) - 2026-08-23T17:44:59.762Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 792](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:792).

The inventory makes the missing distinction clearer: today’s real compatibility checks are about selected run concerns—objective/prediction, caching versus text-encoder training/offload, weighted captions, persistence, precision—not about proving that an arbitrary Python module is “semantically the same participant.” The latter appears only as an unresolved replacement condition in the binding work. I’m tracing the SDXL construction and validation path now to see what is actually authored, what is inferred from config/family, and what Trainer consumes.


> Compaction boundary at raw line 890, 2026-08-23T17:50:08.434Z. The generated summary is omitted.

<a id="m070"></a>
## 070 Assistant (commentary) - 2026-08-23T17:50:12.946Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 897](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:897).

I’ve now separated the repository’s established design from the hypothesis I introduced. I’m doing one final pass over the project-level architecture guidance and the exact current execution path, then I’ll give you a correction with file-backed evidence—not another proposed abstraction.

<a id="m071"></a>
## 071 Assistant (final_answer) - 2026-08-23T17:51:25.508Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 915](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:915).

You were right to push back. After reading the direction, inventory, notes, and current code, my earlier explanation was not properly grounded.

The actual answer is: compatibility rules are not “in the strategy definition” in the sense I described.

The intended relationship is:

```text
Trainer contract
  defines required concerns and compatibility rules

Strategy
  explicitly chooses and wires an arrangement that claims to satisfy them

Models / features / capabilities
  provide implementations, facts, and local constraints

Contract validation
  checks whether the complete authored arrangement is valid

Binding authority
  maintains the accepted runtime bindings and controls transitions
```

The normative direction says this directly:

- “Contract = vocabulary + required selections + compatibility rules”
- “Strategy = an authored and validated filing of that contract”

See [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:232).

So “filing” originally means the complete strategy’s answer to the Trainer contract. It does not establish a separate compatibility-rule object attached to every participant.

What the strategy owns is its choice:

- which participants and components are involved;
- which objective, representation, conditioning, caching, and persistence behavior it uses;
- which features and capabilities it wires;
- which bounded configuration choices it exposes;
- the typed information needed by Trainer-owned mechanics.

The contract then judges whether those choices agree. The strategy does not get to invent the meaning of compatibility independently.

The direction explicitly says validation proves that the authored arrangement is “complete and mutually compatible,” while not selecting or inventing the arrangement itself ([direction](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:141)). It also leaves open how features express what they provide, require, and accept, and which compatibility facts are structurally testable ([open questions](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:1345)).

Your own recorded clarification is even more explicit:

> “it might constrain or validate it (where this happens is not decided…)”

That is in [notes.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:577). Therefore, neither validation placement nor a participant-local validator was settled.

The code confirms that no participant-level compatibility system exists today:

- The launcher validates the configuration before constructing the strategy: [train.py](/home/imi/Projects/sd-scripts/train.py:18).
- Compatibility rules such as model/objective pairings currently live in global config validation: [config_validation.py](/home/imi/Projects/sd-scripts/library/config/config_validation.py:512).
- The strategy factory just selects a family class: [factory.py](/home/imi/Projects/sd-scripts/library/strategies/factory.py:22).
- The SDXL strategy is a large manually assembled collection of behavior facets, without a fulfillment result: [training.py](/home/imi/Projects/sd-scripts/library/strategies/sdxl/training.py:20).
- The Trainer still constructs the objective separately, outside the strategy: [trainer.py](/home/imi/Projects/sd-scripts/library/training/runners/trainer.py:130).
- Runtime component replacement currently just swaps `module: Any` by a descriptive role: [components.py](/home/imi/Projects/sd-scripts/library/models/components.py:189).

The inventory describes the actual missing boundary as one whole compatibility result covering representation, conditioning, predictor, objective, cache, precision, training mode, and persistence—not an independent validator attached to each participant ([inventory](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_inventory.md:924)).

Where I went wrong was promoting this external suggestion:

```text
ParticipantDeclaration
  authored key/address
  semantic compatibility filing
```

into a working design. That appears in [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md:197). The ensuing `SemanticCompatibilityFiling`, proposal-scoped assessment, and validator-composition machinery are extrapolations. They are not established by the normative direction or by the code. My subsequent claim that the “component owns the validator” was also unsupported: components may provide facts and local constraints, but many relevant rules span the whole training arrangement.

The clean distinction is:

- Contract compatibility asks whether the whole selected training arrangement is acceptable to the Trainer.
- Binding-transition validity asks whether materializing or replacing one participant preserves that acceptable arrangement and its logical identity.

Q1–Q5 settle that the binding authority controls identity and transitions. They do not settle that every participant carries its own semantic validator. The concrete validation mechanism remains open.

Therefore, our next step should be to correct the working exchange draft, then design the whole-strategy fulfillment exchange:

1. What the Trainer contract requires.
2. What the authored strategy declares and supplies.
3. Which facts and constraints models/features contribute.
4. What can be validated before loading.
5. What requires materialized components.
6. What must be revalidated after preparation or replacement.
7. What accepted fulfillment result the binding authority relies upon.

That gives runtime replacement a real context: a replacement is valid when the affected complete filing remains fulfilled—not merely because a participant-local callback says “yes.”

I did not modify any files during this audit. The current compatibility section should be treated as a challenged working hypothesis until we revise it.

<a id="m072"></a>
## 072 User - 2026-08-23T18:09:27.568Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 921](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:921).

lets go through that please. lets make sure to properly inspect the relevant context&#x20;


<a id="m073"></a>
## 073 Assistant (commentary) - 2026-08-23T18:09:36.729Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 925](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:925).

Yes. I’ll treat the participant-level compatibility draft as untrusted and rebuild the question from the normative direction, the chronological notes, the inventory, and the actual lifecycle in code. I’m using the OpenSpec exploration workflow to keep this in design mode, the codebase graph for verified runtime tracing, and Beads so the resulting state stays attached to `sd-scripts-syv`. I won’t change the design file until we have examined the boundary together.

<a id="m074"></a>
## 074 Assistant (commentary) - 2026-08-23T18:10:33.950Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 977](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:977).

The first important result is already clear: the repository has two compatibility moments today, even though they are not modeled as such. It rejects some selected combinations before strategy construction, then performs additional family/component checks much later during loading and preparation. I’m tracing those later checks and every place live modules are replaced so we can identify the real exchange stages rather than inventing one validation callback.

<a id="m075"></a>
## 075 Assistant (final_answer) - 2026-08-23T18:12:43.714Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1007](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1007).

I inspected the relevant direction, inventory, notes, and runtime paths. The first thing we need to resolve is not a class or datatype—it is that “validation” currently refers to three different lifecycle questions.

The documents themselves expose this tension:

- The main direction shows contract validation before the validated strategy reaches Trainer: [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:141).
- The lifecycle section says `authored definition → validated arrangement → loaded/bound runtime → prepared execution runtime`: [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:441).
- But the ordinary fine-tuning scenario says materialize first, then “validate the filing needed for this training arrangement”: [strategy_system_inventory.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_inventory.md:1716).

Those can coexist, but only if we stop treating “compatibility validation” as one undifferentiated event.

## What the current runtime actually does

Today’s path is:

```text
prepare config
  ↓
validate selected configuration combinations
  ↓
construct family strategy + separate TrainingMode
  ↓
Trainer separately constructs objective
  ↓
load model components
  ↓
cache
  ↓
create/attach trainables and apply precision policy
  ↓
Accelerate/DeepSpeed replaces or wraps live modules
  ↓
run
```

Evidence:

- Config validation precedes strategy construction: [train.py](/home/imi/Projects/sd-scripts/train.py:18).
- The factory only chooses a family strategy class: [factory.py](/home/imi/Projects/sd-scripts/library/strategies/factory.py:22).
- Trainer independently builds the objective: [trainer.py](/home/imi/Projects/sd-scripts/library/training/runners/trainer.py:130).
- Loading happens inside `Trainer.setup()`: [trainer.py](/home/imi/Projects/sd-scripts/library/training/runners/trainer.py:307).
- Deferred loading, adapter attachment, and precision changes happen later: [model_prep.py](/home/imi/Projects/sd-scripts/library/training/phases/model_prep.py:30).
- Distributed wrapping happens later still: [optimizer.py](/home/imi/Projects/sd-scripts/library/training/phases/optimizer.py:147).

The current compatibility checks are therefore scattered across lifecycle stages:

1. Early configuration checks:

   - model/objective pairing;
   - objective/prediction pairing;
   - text-encoder training versus caching/offloading;
   - broad precision and mode restrictions.

2. Materialization/loading checks:

   - checkpoint/source compatibility;
   - availability of optional dependencies such as RamTorch;
   - whether the loader can construct the expected family components.

3. Preparation checks:

   - SD3 T5 training versus cached text-encoder outputs;
   - SD3 T5 FP8 support;
   - adapter target resolution;
   - availability of required denoiser components;
   - component-specific precision and preparation behavior.

4. Replacement/wrapping currently has almost no compatibility boundary:

   - `accelerator.prepare()` results are assigned through Trainer setters;
   - those setters replace `module: Any` by the descriptive role;
   - they do not validate semantics, revisions, dependencies, or freshness.

See [adapter_mode.py](/home/imi/Projects/sd-scripts/library/training/modes/adapter_mode.py:115), [finetune_mode.py](/home/imi/Projects/sd-scripts/library/training/modes/finetune_mode.py:84), and [components.py](/home/imi/Projects/sd-scripts/library/models/components.py:189).

## The ownership split that the direction supports

| Concern | Owner |
|---|---|
| What the training pipeline requires | Trainer contract |
| Pipeline-wide compatibility and lifecycle meaning | Trainer contract |
| Compatibility semantics of a named capability | Capability contract |
| Requirements of reusable strategy behavior | Feature contract |
| Component structure, mechanics, facts, and local limitations | Model/component layer |
| Which arrangement, objective, features, capabilities, and subjects are selected | Authored strategy |
| Strategy-specific relationships and constraints | Authored strategy |
| Checking the combined arrangement | Contract-validation boundary; exact implementation placement remains open |
| Installing bindings, advancing revisions, and invalidating dependents | Binding authority |

The important correction is that the strategy supplies the arrangement being judged. It may also add genuine arrangement-specific constraints. But the strategy is not automatically the owner of every compatibility rule, and a participant is not automatically the owner of a semantic validator.

## The lifecycle questions we should separate

I think this is the correct starting model:

```text
HANDWRITTEN STRATEGY
        │
        ▼
1. Authored-arrangement conformance
        │
        ▼
   declared valid arrangement
        │
        ▼
2. Materialization acceptance
        │
        ▼
   accepted bound runtime
        │
        ▼
3. Preparation acceptance
        │
        ▼
   prepared runtime ready for requested operations
        │
        ▼
4. Later transition acceptance
```

### 1. Authored-arrangement conformance

This asks:

> Has the strategy supplied a coherent answer to the Trainer contract?

It can be checked without loading large model objects.

For an SDXL fine-tuning strategy, that might include:

- a selected representation;
- compatible conditioning and predictor behavior;
- an objective explicitly supported by that predictor behavior;
- declared training subjects;
- any selected cache behavior;
- requested sampling and persistence capabilities;
- all conditional requirements implied by those selections.

This is the primary meaning of:

```text
Contract = vocabulary + required selections + compatibility rules
Strategy = authored and validated filing
```

It is a whole-arrangement question—not one validation per denoiser, VAE, or text encoder.

### 2. Materialization acceptance

This asks:

> Did loading produce acceptable current state for the already-declared arrangement?

Inputs now include actual model/component facts that did not exist during authored-arrangement validation:

- which participants were successfully bound;
- which remain deliberately deferred;
- concrete realization/source information;
- structural facts relevant to selected behavior;
- actual relationships or target resolution;
- component limitations encountered during loading.

The binding authority owns whether the proposed state is installed. But it should not contain a built-in understanding of SDXL or “denoisers.” It evaluates the proposal against the already accepted arrangement and the applicable contract/feature/component constraints.

This is where the participant-level draft went wrong: it assumed all of that meaning could be reduced to one `SemanticCompatibilityFiling` belonging to the participant.

### 3. Preparation acceptance

This asks:

> Did preparation preserve the declared arrangement while making it executable in the requested runtime?

Examples include:

- Accelerate returning a wrapper for the same logical participant;
- several participants being jointly wrapped by DeepSpeed;
- precision changes satisfying component constraints;
- an adapter being attached to its declared targets;
- required execution routes becoming available;
- prepared results retaining valid access for execution and persistence.

This is not a second strategy definition. It is acceptance of a lifecycle transition within the already-declared strategy.

### 4. Later transition acceptance

Replacement, lazy materialization, recompilation, attachment changes, or another runtime transition asks:

> Does the proposed transition preserve a valid current arrangement, or does it actually require retiring/amending participants?

Q1–Q5 constrain the mechanics here:

- stable logical identity;
- one authority;
- typed transitions;
- revisions and freshness;
- atomic acceptance;
- dependency invalidation.

But they do not settle the semantic-validation API.

## What this changes in the draft

The problematic section currently says:

```text
ParticipantDeclaration
  authored key/address
  explicitly wired semantic compatibility filing
```

That should not remain a working decision in its present form. It collapses:

- whole-arrangement contract compatibility;
- component facts;
- feature/capability constraints;
- materialization acceptance;
- identity-preserving replacement;

into one participant-owned concept.

A safer interim understanding is:

```text
Participant declaration
  identifies one semantic participant in the authored arrangement

Arrangement filing
  states the strategy’s selected concerns, providers, relationships,
  capabilities, constraints, and bounded choices

Materialization/preparation proposals
  supply concrete facts and proposed runtime state

Validation
  applies the relevant contract, capability, feature, component,
  and arrangement constraints at the appropriate lifecycle boundary

Binding authority
  atomically accepts or rejects the resulting transition
```

We should not lock the exact participant declaration fields yet. Participant identity may require more than an authored key, but the evidence does not justify embedding a universal validator inside it.

The first design decision I recommend we settle is:

> A strategy can be validly authored before models are loaded, while materialization and preparation have separate acceptance/readiness checks that can still fail without redefining the strategy.

That distinction reconciles the direction’s early validation diagram with the inventory’s post-materialization validation scenario. Once we agree on that, we can enumerate exactly what belongs in the authored arrangement filing, using ordinary SDXL fine-tuning as the first concrete case.

No files were changed during this inspection. The graph evidence was checked against the current full index generation, with no recorded coverage issues for the 15 runtime files examined.

<a id="m076"></a>
## 076 User - 2026-08-23T18:22:23.670Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1013](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1013).

what makes the most sense from a coherence and intuitiveness standpoint? I do not want validation in random places, and we're not looking at interim solutions


<a id="m077"></a>
## 077 Assistant (final_answer) - 2026-08-23T18:24:14.799Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1022](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1022).

The most coherent target is one validation system behind the complete `TrainingStrategy`, with every lifecycle change passing through the same acceptance gate.

The key distinction is:

> Centralize where validity is decided, while keeping each rule beside the concept that gives it meaning.

That avoids both failure modes:

- validation scattered through loaders, modes, preparation hooks, and Trainer branches;
- one enormous validator full of `if model_type == "sdxl"` logic.

## Recommended target

```text
TrainingStrategy
├── authored definition
│     what is trained, selected features/capabilities, relationships,
│     bounded choices, and training semantics
│
├── applicable contract rules
│     core + selected capability/feature/component constraints
│
└── authoritative runtime state
      declarations, bindings, prepared routes, revisions, and freshness
```

Trainer receives only this complete strategy.

Every proposed state change follows one process:

```text
current strategy state
        +
typed proposed transition
        ↓
construct prospective next state
        ↓
apply all relevant contract rules
        ↓
reject, or atomically publish the next state
```

That same gate handles:

- initial strategy establishment;
- component materialization;
- deferred materialization;
- adapter attachment;
- runtime preparation;
- wrapper replacement;
- identity-preserving component replacement;
- relationship changes;
- arrangement amendments.

These are different transition types, but not different validation systems.

## Where rules belong

Rules should live with their semantic owners:

| Example rule | Where it belongs |
|---|---|
| Every trainable strategy needs a usable step exchange | Core Trainer contract |
| A selected persistence capability must support the requested product | Persistence capability contract |
| Latent caching requires compatible representation/cache behavior | Latent/cache feature contracts |
| An objective’s prediction semantics must match predictor behavior | Objective/predictor contracts |
| SD3 T5 does not support a particular FP8 preparation | SD3 T5 component integration or preparation feature |
| An adapter requires resolved compatible targets | Adapter capability contract |
| This custom teacher supervises this specific student | The authored compound strategy relationship |

The strategy explicitly wires the applicable pieces together. It does not rewrite their rules.

For example:

```text
strategy explicitly says:
  this cache behavior uses this representation provider
  this objective uses this predictor behavior
  this adapter affects these participants
```

The validator checks those explicitly authored relationships. It does not search for providers, infer dependencies, or automatically assemble anything.

## One gate, several lifecycle states

The complete lifecycle should look like this:

```text
build_training_strategy(cfg)
        ↓
validate and establish authored arrangement
        ↓
return TrainingStrategy to Trainer
        ↓
loading proposes materialized bindings
        ↓
central acceptance gate
        ↓
preparation proposes wrappers/routes/relationships
        ↓
central acceptance gate
        ↓
strategy becomes ready for execution
```

“Validated strategy” and “ready to execute” remain different:

- A strategy definition can be coherent before models are loaded.
- Its participants may still be declared but unbound.
- Loading can fail mechanically.
- A loaded candidate can be rejected as incompatible.
- A valid bound runtime may still require preparation.
- A preparation result can be rejected without corrupting the previously accepted state.

This is one lifecycle-aware invariant system—not three unrelated validators.

## What the binding authority does

The binding authority should be the single writer and commit point. It should:

1. Receive a typed transition proposal.
2. Construct the prospective complete state.
3. Invoke the strategy’s applicable contract validation.
4. Reject without changing current state, or commit atomically.
5. Advance revisions and invalidate affected derived views.
6. Return the accepted transition result.

It should not know what `denoiser`, SDXL, T5, LoRA, or latent diffusion means.

Likewise, a proposer must not choose a validator. The applicable rules were fixed by the established strategy arrangement.

## What this means for participant declarations

A participant should not carry an independent `SemanticCompatibilityFiling`.

Its meaning comes from its place in the complete authored arrangement:

```text
participant identity
+ selected behaviors that use it
+ declared relationships
+ applicable capability/feature/component constraints
```

For example, `student.denoiser` is not meaningful merely because its declaration contains a “denoiser validator.” It is meaningful because the strategy uses that participant as:

- the predictor in a particular step behavior;
- a subject of a particular objective;
- a target of an adapter relationship;
- a selected optimization subject;
- part of particular artifact products.

Identity-preserving replacement means the prospective complete arrangement still satisfies those obligations. If it does not, the authority rejects replacement and requires an explicit arrangement amendment.

## What disappears from the target architecture

This should be a real cleanup, not an additional layer over the current system.

- `validate_config()` stops owning model/strategy compatibility branches. It retains config-schema and locally intrinsic value validation.
- Trainer stops independently selecting the objective.
- Family methods stop raising late `NotImplementedError` for capabilities they nominally claim.
- `post_process_trainable()` stops hiding unrelated compatibility checks and mutations.
- Loaders return typed proposals rather than directly installing tuples.
- `Trainer.denoiser` and similar setters disappear as unrestricted replacement paths.
- `TrainingMode` dissolves as already settled.
- `accelerator.prepare()` results must be submitted as preparation transitions rather than assigned directly.
- No component-specific validator callbacks are stored in otherwise generic participant records.

Mechanical failures remain local: a corrupt checkpoint, missing dependency, filesystem error, CUDA failure, or OOM is an operation failure, not contract compatibility. The operation reports failure without publishing proposed state.

## The resulting intuitive programming model

At a high level, strategy code remains ordinary authored Python:

```python
strategy = build_training_strategy(cfg)
trainer = Trainer(strategy)
trainer.train()
```

Inside the lifecycle, the pattern stays consistent:

```text
produce proposed result
        ↓
strategy accepts or rejects it
        ↓
accepted state becomes authoritative
```

A developer looking at a feature can find its rules beside that feature. A developer looking at runtime orchestration can see exactly one path through which state becomes accepted. A developer looking at Trainer does not encounter model-family validation branches.

I think this is the strongest final design: one complete strategy, one authoritative state owner, one contract-governed acceptance path, and locally owned typed rules. The current participant-level compatibility-filing draft should be rewritten around this model rather than refined in its existing direction.

<a id="m078"></a>
## 078 User - 2026-08-23T18:45:06.330Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1030](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1030).

this still does not really align with what I want. whatever happens to the authored strategy between compilation and going to the trainer should be able to judge it. whatever it is, should know about what the trainer can do and what is acceptable to reach the trainer. this has been talked about. The contract is about what the trainer is capable of, what capabilities are available in the library, and it guides the strategy through informing the author of that. this is authoritative by default due to the nature of it being a contract FOR the trainer. whatever else "compatibility" or binding there is, this is fundamentally the same concern even if it's not explicitly defined in the contract itself, the desired result is that everything you've listed so far is able to be enforced on the authored strategy. the author of a strategy is not responsible for the details of how their strategy is valid communicated from outside of the contract or the enforcing mechanism of that contract unless they EXPLICITLY want to do something that's not covered, whatever levels there are to that scale.

we already have this in the direction

## Current Direction In One View

```text
                       TRAINING CONTRACT SYSTEM
              vocabulary + rules + lifecycle + results
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
     CORE CONTRACT        PIPELINE CAPABILITIES    FEATURE CONTRACTS
  minimum Trainer need    sampling, validation,   representation,
                          caching, persistence     conditioning, etc.
          │                       │                       │
          │              pipeline coordinates      strategy consumes
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  ▼
                   AUTHORED TRAINING STRATEGY
             explicit training intent/components/features/capabilities
             internal per-run binding authority for accepted current state
                                  │
          authored → validated → bound → prepared
                                  │
                                  ▼
                               TRAINER
                 executes core and requested capabilities
                 owns time, optimization, infrastructure,
                         triggers, and observation
```

<a id="m079"></a>
## 079 Assistant (commentary) - 2026-08-23T18:45:37.535Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1036](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1036).

You’re right. I was still treating validation as something assembled out of rules contributed by the strategy’s selected parts. That puts too much responsibility on the strategy author and weakens the contract from an authoritative acceptance system into a coordination convention. I’m checking the surrounding “Current Direction” and contract-evolution passages now so I can restate this using the direction’s actual model.

<a id="m080"></a>
## 080 Assistant (final_answer) - 2026-08-23T18:46:50.714Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1044](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1044).

Yes. Your model is more coherent, and it changes the center of the design:

> Compatibility is not a collection of rules assembled by the strategy. It is contract enforcement applied to the strategy.

The training contract system is not merely vocabulary that the strategy claims to follow. It is the authoritative definition of everything the Trainer can accept, including the known library capabilities and feature contracts used to construct acceptable strategies.

The intended path is:

```text
TRAINING CONTRACT SYSTEM
  knows:
    Trainer requirements
    pipeline capabilities
    feature contracts
    lifecycle rules
    compatible results and relationships
    known library implementations
             │
             │ supplies authoring vocabulary and guidance
             ▼
AUTHORED STRATEGY
  explicitly chooses and wires what is trained and how
             │
             │ contract compilation / enforcement
             ▼
VALIDATED TRAINING STRATEGY
  contains:
    accepted arrangement
    derived obligations
    provided capabilities
    internal binding authority governed by those obligations
             │
             ▼
TRAINER
```

Publicly, the result is still just a `TrainingStrategy`. “Validated strategy” does not need to become another object in the Trainer-facing API.

## The contract system’s real responsibilities

Whatever performs the authoring-to-Trainer transition must know:

- what the active Trainer requires;
- which pipeline capabilities it can coordinate;
- which feature contracts exist;
- which standard library implementations fulfill them;
- which combinations are valid;
- which lifecycle states and transitions are allowed;
- what results loading, preparation, optimization, execution, and persistence must produce;
- what facts must eventually be proven when they cannot be known during initial compilation.

It judges the complete authored strategy. The author does not supply validation callbacks or manually explain how ordinary library features are compatible.

The author’s ordinary responsibility is only:

```text
choose intent
choose components
choose features/capabilities
wire them explicitly
configure deliberately exposed choices
```

The contract system is responsible for understanding the consequences of those choices.

This preserves the important distinction from automatic assembly:

```text
contract system may:
  tell the author what is available
  expose typed authoring interfaces
  derive requirements from explicit choices
  reject missing or incompatible wiring
  enforce lifecycle rules

contract system may not:
  silently choose features
  silently choose implementations
  infer the intended strategy
  pull in dependencies on the author's behalf
```

Automatic validation is desirable. Automatic strategy assembly is not.

## Binding compatibility is still contract enforcement

Materialization, binding, preparation, and replacement are not a separate semantic system bolted onto the strategy.

During compilation, the contract system can derive obligations such as:

```text
this participant is used as the predictor for this objective
this representation feeds this predictor
this cache capability depends on this representation
this adapter relationship targets these participants
this artifact product covers these participants
this execution route must exist before a training step
```

Some obligations can be resolved immediately. Others require runtime evidence.

```text
compile-time knowledge
  explicit arrangement
  selected feature implementations
  declared relationships
  exposed configuration choices
  known implementation support
  lifecycle requirements

runtime evidence
  successfully materialized component facts
  resolved adapter targets
  prepared wrapper results
  precision/runtime facts
  binding and relationship revisions
```

The binding authority receives the obligations produced by the contract system. Later transitions are accepted only when runtime evidence satisfies those obligations.

Therefore:

```text
contract compilation
  derives what must remain true

binding authority
  enforces it as current state changes
```

The authority is not a second compatibility authority. It is the runtime enforcement arm of the same contract system.

## What this means for ordinary strategy authors

For a standard SDXL strategy, the author might explicitly use:

- the repository’s SDXL component integration;
- latent representation behavior;
- SDXL conditioning behavior;
- a supported predictor adapter;
- DDPM or rectified-flow objective behavior;
- selected caching, sampling, and persistence capabilities;
- a fine-tuning or adapter-based training-subject treatment.

The author should not additionally write:

- a denoiser validator;
- a VAE compatibility callback;
- a cache/predictor compatibility rule;
- a replacement validator;
- a list of facts the binding authority should inspect;
- a hand-assembled compatibility assessment.

Those meanings should already be defined by the contracts the author is using. Compilation derives the applicable obligations automatically.

If the SDXL predictor contract says it accepts a particular conditioning/result convention, and the selected conditioning feature provides another, compilation rejects the arrangement. The author should not need to know which internal validator performs that comparison.

## Explicit escape levels

The ordinary path is contract-governed by default. Extra author responsibility begins only when the author deliberately leaves the known path.

A sensible scale is:

1. **Known library composition**

   The author uses established components, features, and capabilities. The contract system knows how to validate everything. No custom validation work is required.

2. **Custom implementation of an existing feature or capability contract**

   The author explicitly provides a new implementation of a known contract. They must satisfy its established conformance/evidence boundary, but they do not redefine what the Trainer accepts.

3. **Custom strategy implementation that bypasses the standard internal decomposition**

   The author directly fulfills the existing Trainer-facing contract. They take responsibility for proving conformance at that boundary.

4. **Contract or Trainer extension**

   The experiment requires behavior the active Trainer contract does not describe. The author explicitly targets an extension/version, possibly with corresponding Trainer support.

The system should never silently treat levels 2–4 as level 1.

## Consequences for the binding draft

The participant-level semantic-filing direction is wrong because it transfers contract knowledge to the strategy author:

```text
ParticipantDeclaration
  authored key
  author-supplied semantic compatibility filing
```

The corrected direction is closer to:

```text
Authored strategy
  declares participants and explicitly wires their uses/relationships

Contract compilation
  derives the obligations implied by those uses

Validated strategy runtime
  retains those obligations behind its contract-governed authority

Materialization/preparation
  supplies evidence required by the obligations

Binding authority
  accepts or rejects transitions under the compiled contract
```

A participant declaration therefore does not need to contain a validator. Its requirements are derived from how the authored strategy uses it.

Likewise, a loader should not decide what counts as identity-preserving replacement. It returns contract-defined facts about what it produced. The strategy’s compiled obligations determine whether the result is acceptable as the same participant.

## The corrected central model

```text
                       TRAINING CONTRACT SYSTEM
           specification + authoring API + compiler/enforcer
                                  │
                     informs and constrains
                                  ▼
                       AUTHORED STRATEGY
                  explicit choices and wiring
                                  │
                         compile / validate
                                  ▼
                       TRAINING STRATEGY
             accepted arrangement + derived obligations
                 + contract-governed binding authority
                                  │
                runtime evidence and state transitions
                   continue through the same contract
                                  │
                                  ▼
                               TRAINER
```

So the next design question is no longer:

> What compatibility filing must each participant carry?

It is:

> What does the authored strategy present to the training contract system, and what accepted obligations does contract compilation establish inside the resulting strategy runtime?

That is the boundary we should design next. It follows the existing direction and keeps ordinary authors responsible for authoring a strategy—not for authoring its validator. This clarification should supersede EX-007 and the participant-scoped compatibility decisions EX-013 through EX-017.

<a id="m081"></a>
## 081 User - 2026-08-23T19:03:44.281Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1050](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1050).

a lot of this is stuff we already talked about conceptually, these are just detail questions, and I'm relying on you for them. thoughts? do we need to modify the exchange design?


<a id="m082"></a>
## 082 Assistant (commentary) - 2026-08-23T19:03:55.134Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1054](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1054).

Yes, I think the exchange design needs modification—not because the overall architecture changed, but because its participant-level compatibility section accidentally turned an unresolved implementation detail into a competing source of contract authority. I’m going to map exactly which working decisions are affected and which binding/preparation work remains valid, so we can make a surgical correction rather than restart the document. I’m using the exploration workflow here because this is still design correction, not implementation.

<a id="m083"></a>
## 083 Assistant (final_answer) - 2026-08-23T19:05:06.320Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1066](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1066).

Yes. We should modify the exchange design before continuing. The normative direction already says what you mean; the exchange document drifted away from it while answering the participant-replacement detail.

This is a correction to the downstream design, not a reopening of the architecture.

## What remains valid

Most of the binding and preparation work survives:

- stable logical participant identity;
- distinction between declaration, materialization, replacement, route rebinding, relationship transitions, and arrangement amendments;
- one internal binding authority;
- revision and freshness semantics;
- typed transition proposals;
- atomic acceptance;
- dependency invalidation;
- separation of logical participants, prepared routes, access views, and backend handles;
- preparation plans/results;
- in-place mutation coordination;
- persistence semantics.

Those concern runtime state and remain compatible with the authoritative contract system.

## What needs to change

The design currently says:

```text
ParticipantDeclaration
  authored key/address
  explicitly wired semantic compatibility filing
```

It then builds:

- participant-specific compatibility filings;
- filing-specific validator behavior;
- proposal-scoped compatibility assessments;
- validation clauses explicitly composed by the strategy author.

That is the wrong authority model. It makes the strategy author communicate how validity should be judged, when the contract system should already know that for ordinary contract-supported features and capabilities.

EX-007 and EX-013 through EX-017 should therefore be superseded, not refined.

Parts of EX-008 and EX-009 remain semantically useful, but their source changes:

- readiness and execution-route requirements remain separate from identity;
- however, the contract system derives those requirements from the authored strategy’s selected uses—it should not require the author to manually declare low-level lifecycle obligations that known contracts already imply.

## What the exchange design needs to add

The beginning of the binding exchange should become the contract-enforced strategy establishment boundary:

```text
authored strategy
+ active Trainer contract
+ known pipeline capabilities
+ feature contracts and known implementations
+ explicitly exposed configuration choices
                     │
                     ▼
          contract fulfillment/enforcement
                     │
          ┌──────────┴──────────┐
          │                     │
       rejected              accepted
                                │
                                ▼
                      TrainingStrategy
                      ├── accepted arrangement
                      ├── derived obligations
                      ├── provided capabilities
                      └── internal binding authority
                                │
                                ▼
                             Trainer
```

This does not need to become a fifth Trainer-facing exchange. It is the first part of “establish the authored arrangement” within the binding exchange.

The binding authority is created or seeded at this boundary. It receives contract-derived obligations, rather than participant-authored validators.

## How runtime evidence should work

The materialization and preparation sections should then say:

```text
contract establishment
  derives what must be true

loader / component / preparation backend
  reports what it actually produced

contract-governed authority
  checks the proposed resulting state against the derived obligations
  and atomically accepts or rejects it
```

A standard loader or feature implementation supplies facts in contract-defined forms because it conforms to a known library contract. The strategy author does not design those evidence types per strategy.

For custom work:

- a custom implementation of an existing contract must explicitly satisfy that contract’s conformance boundary;
- a genuinely new meaning requires an explicit feature/capability/Trainer-contract extension.

That is the only point where the author assumes additional validity responsibilities.

## Specific document changes

| Section | Action |
|---|---|
| Status and scope | Clarify that the exchange design is governed by the authoritative training contract system |
| Participant declaration | Remove the required semantic-compatibility filing |
| Compatibility-filing sections | Replace with contract-derived obligations and contract-defined runtime evidence |
| Establish authored arrangement | Rewrite as the contract fulfillment/strategy-establishment boundary |
| Materialization | Validate reported results against compiled obligations |
| Replacement | Judge identity preservation from the participant’s complete contract-derived use in the arrangement |
| Execution/readiness declarations | Preserve the distinctions, but make obligations derived rather than manually authored |
| Preparation projection | Expose contract-derived requirements from the validated strategy |
| EX-007 | Supersede |
| EX-008/009 | Refine their source of authority |
| EX-013/014 | Replace filings with contract-defined evidence and compiled obligations |
| EX-015 | Remove the transferable single-use compatibility-assessment concept unless later mechanics prove it necessary |
| EX-016 | Retain only the already-settled rule that the authority alone commits state |
| EX-017 | Supersede; the contract system derives ordinary validation, while custom extensions are explicit |
| Open questions | Replace filing/validator questions with contract-compilation, derived-obligation, evidence, and extension-boundary questions |
| Discussion order | Establish the authoring-to-validated-strategy boundary before returning to materialization and preparation details |

The direction document does not need conceptual repair—it already contains this model. The inventory also remains valid as evidence. After revising the exchange design, the chronological notes and Bead checkpoint should record that the participant-scoped compatibility proposal was superseded because it duplicated and weakened the training contract’s authority.

I recommend making that surgical rewrite now, then resuming with the first real detail question:

> What information must an authored strategy expose so the contract system can derive its obligations and establish the accepted strategy runtime?

That is the correct next question; designing a participant validator is not.

<a id="m084"></a>
## 084 User - 2026-08-23T19:15:31.682Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1072](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1072).

go ahead please


<a id="m085"></a>
## 085 Assistant (commentary) - 2026-08-23T19:15:41.091Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1076](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1076).

I’ll update the exchange design in three coordinated pieces: replace the participant-authored compatibility filing with contract-derived obligations, rewrite initial arrangement establishment as the contract-enforced authoring-to-Trainer boundary, and update the decision/open-question registers. Then I’ll add a chronological correction to the notes and the Bead without altering the normative direction or inventory.

<a id="m086"></a>
## 086 Assistant (commentary) - 2026-08-23T19:20:12.757Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1126](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1126).

The core rewrite is in place. The binding/preparation mechanics were preserved, while the authority drift was removed: participant declarations no longer carry authored validators, strategy establishment now begins at the authoritative contract boundary, and runtime evidence is judged against contract-derived obligations. I’m checking the document for leftover contradictory wording before recording the corrected checkpoint in Beads.


> Compaction boundary at raw line 1143, 2026-08-23T19:21:26.055Z. The generated summary is omitted.

<a id="m087"></a>
## 087 Assistant (commentary) - 2026-08-23T19:21:29.477Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1150](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1150).

The contract-authority correction is captured. I’m doing one final clean-diff check, then I’ll hand off the exact state and next design question.

<a id="m088"></a>
## 088 Assistant (final_answer) - 2026-08-23T19:21:39.311Z

Source: [01a026d5-a88b-7843-a1c0-1b51047c8182, raw line 1156](/home/imi/.codex/sessions/2026/08/22/rollout-2026-08-22T02-18-51-01a026d5-a88b-7843-a1c0-1b51047c8182.jsonl:1156).

Done. The exchange design now treats the training contract system as the authoritative judge of the complete authored strategy before Trainer receives it.

Key corrections:

- Removed participant-authored “compatibility filings” and validator logic.
- Made contract establishment the first binding operation: validate the authored strategy, derive obligations, and establish its internal runtime authority.
- Kept automatic enforcement distinct from automatic strategy assembly.
- Defined materialization, replacement, and preparation as contract-governed transitions using typed runtime evidence.
- Marked the earlier conflicting decisions as superseded and updated the open questions accordingly.
- Recorded the reasoning chronologically in [notes.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md).
- Updated bead `sd-scripts-syv` with the correction and next step.

The revised artifact is [strategy_contract_exchange_design.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md).

Q1–Q5 remain settled. The normative direction and inventory were not changed. `git diff --check` passes; only these two design documents are modified, so no tests were needed.

Next time, the concrete question is: what exactly enters contract establishment as the authored strategy, and what accepted `TrainingStrategy` comes out?

