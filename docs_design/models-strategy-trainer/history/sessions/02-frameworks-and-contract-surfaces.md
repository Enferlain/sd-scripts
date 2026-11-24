# Framework comparison and contract surfaces: 29 July 2026

Continuation of [chapter01](01-inventory-and-authoring.md). This reading starts with the Lightning/Fabric question and ends after the updated direction has been read and the next design-pressure work identified. The next chapter begins with the instruction to inspect that code.

Captured 2026-11-05T18:26:45.923335+00:00. 48 visible user/assistant messages. Bodies and historical links are preserved. Reasoning, tools, generated context and compaction summaries are omitted.

[Machine-readable provenance](02-frameworks-and-contract-surfaces.json) | [Source records](02-source-records.json) | [Conversation branch map](branch-map.md)

This passage follows the same eight-session ancestry as chapter01. Its messages occur in the selected continuation, not in the inspected earlier ancestors. This does not claim every sibling or later replay has been found.

External feedback referenced by user message 036 is available as an [exact readable copy](../snapshots/2026-07-29-contract-surfaces-feedback.md). It is imported opinion; conversation and governing adoption establish its disposition. The .md extension does not change its bytes.

The [first committed comparison](../snapshots/97a81ef4-framework_pattern_comparison.md) and [direction at that commit](../snapshots/97a81ef4-strategy_system_direction.md) predate the later three-surface/lifecycle update in this conversation. Those updates enter Git in the next direction revision; do not read the earlier snapshot as though it already contains them.

## Message index

- [001 user: wait are we just redoing lightning/fabric](#m001)
- [002 assistant: That concern is valid. The shape I just sketched is close enough to `Trainer → LightningModule` that we s](#m002)
- [003 assistant: At the highest level, yes—we are rediscovering the same separation as Lightning:](#m003)
- [004 user: we'll need to talk about this more tomorrow. we encountering more involved topics](#m004)
- [005 assistant: Agreed. This is a real decision point rather than another naming discussion. I’m recording the stopping p](#m005)
- [006 assistant: Agreed. We’ve reached a genuine architecture fork, so pausing is sensible.](#m006)
- [007 user: we might not necessarily want to be constrained by an existing system unless it allows for modification b](#m007)
- [008 assistant: Right. That changes the adoption question substantially.](#m008)
- [009 user: it looks like the route is probably good as a reference for how some things can be done](#m009)
- [010 assistant: That seems like the strongest route.](#m010)
- [011 user: Okay, so what to do now that we figured that part out?](#m011)
- [012 assistant: We have settled the stance, not the contract. The next work should be:](#m012)
- [013 user: okay, let's do this, it won't hurt](#m013)
- [014 assistant: I’ll keep this as architecture research: update the direction record, add a bounded framework-pattern com](#m014)
- [015 assistant: The repo evidence confirms why the comparison matters: the current loop already has a Lightning-like spli](#m015)
- [016 assistant: The comparison is converging on a useful three-layer distinction: Lightning’s `LightningModule` is releva](#m016)
- [017 assistant: I’ve finished the substantive comparison; I’m doing the last accuracy pass now, then I’ll record the mile](#m017)
- [018 assistant: The comparison confirms a useful split: Lightning is closest to the authored-strategy/executor relationsh](#m018)
- [019 assistant: The durable conclusion is now sharper: we are not choosing between “our system” and “Lightning.” We are s](#m019)
- [020 assistant: The framework comparison is complete, and it supports continuing our model–strategy–trainer design rather](#m020)
- [021 user: this document doesn't really explain much, don't think you can reach any conclusion without seeing the co](#m021)
- [022 assistant: You’re right. I treated public documentation as enough to classify the projects, but it is not enough to ](#m022)
- [023 assistant: The code changes the picture materially. Lightning’s public “Trainer calls LightningModule” story hides a](#m023)
- [024 assistant: The source audit now supports a different kind of conclusion than the first draft: not “these projects lo](#m024)
- [025 assistant: You were right—the first draft overreached from public documentation. I replaced it with a source-level a](#m025)
- [026 user: Okay, I read the comparison. now lets start talking about what to actually do with our repo](#m026)
- [027 assistant: Yes. The comparison has done its job; the next step is to stop talking in framework analogies and design ](#m027)
- [028 assistant: I think we should commit to this concrete target:](#m028)
- [029 user: we already knew this, you just listed some things with changed names. also it's not really clear what you](#m029)
- [030 assistant: You’re right. I turned existing intentions into new nouns without defining new mechanics.](#m030)
- [031 user: you're talking about things with uncertainty that we already looked at and talked about. I do not want to](#m031)
- [032 assistant: You’re right. I’m reopening questions the direction document already settled instead of advancing from it](#m032)
- [033 assistant: You’re right. I reread both documents fully. The inventory already completed the method/caller classifica](#m033)
- [034 user: the lightning detour was for reference and comparison and to see if it makes more sense to take how they ](#m034)
- [035 assistant: Yes—that restores the hierarchy correctly. Lightning was a subordinate implementation reference. It shoul](#m035)
- [036 user: # Files mentioned by the user:](#m036)
- [037 assistant: I’ll read the comment against the direction we just clarified and separate useful refinements from anythi](#m037)
- [038 assistant: This is a strong refinement and stays within the established direction. Its best contribution is the cons](#m038)
- [039 user: if the direction document needs updates after recent discussion and the comparisons now would be a good t](#m039)
- [040 assistant: Yes. I’m going to update the direction document now so it records the recent clarification instead of lea](#m040)
- [041 assistant: The direction update is taking shape around the actual clarification, not a new architecture: one contrac](#m041)
- [042 assistant: I’ve made the conceptual update; I’m doing a focused consistency pass now so the new three-surface contra](#m042)
- [043 assistant: The update is internally coherent so far. One distinction deserves slightly sharper wording before I fini](#m043)
- [044 assistant: The comparison’s useful findings are now folded into the direction without turning Lightning into the arc](#m044)
- [045 assistant: Updated \[strategy_system_direction.md\](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/](#m045)
- [046 user: Nice, I read the updates and it feels a bit better than before. we can probably move onto any remaining r](#m046)
- [047 assistant: I’m using the OpenSpec exploration workflow here because we’re still pressure-testing the architecture, n](#m047)
- [048 assistant: They should not be designed separately. The contract is the boundary produced by considering both sides s](#m048)

<a id="m001"></a>
## 001 User - 2026-07-29T03:03:12.015Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12193](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12193).

wait are we just redoing lightning/fabric

<a id="m002"></a>
## 002 Assistant (commentary) - 2026-07-29T03:03:41.582Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12199](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12199).

That concern is valid. The shape I just sketched is close enough to `Trainer → LightningModule` that we should treat “are we rebuilding Lightning?” as a required comparison, not wave it away. I’m checking Fabric’s exact boundary too, because Fabric and Lightning Trainer solve different layers.

<a id="m003"></a>
## 003 Assistant (final_answer) - 2026-07-29T03:04:32.272Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12207](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12207).

At the highest level, yes—we are rediscovering the same separation as Lightning:

```text
ours:       Trainer → fulfilled TrainingStrategy
Lightning:  Trainer → LightningModule
```

My previous `prepare() / training_step() / save()` sketch drifted close enough to Lightning that we should stop and compare before designing further. If we add a large lifecycle-hook interface, optimizer hooks, validation hooks, distributed preparation, checkpoint hooks, and callback machinery, then we really would be rebuilding Lightning under different names.

Fabric is different. Fabric deliberately leaves the training loop and orchestration to application code while providing distributed, precision, device, backward, and checkpoint utilities. That layer is closer to what Accelerate already provides this repository. [Lightning’s Fabric overview](https://lightning.ai/docs/fabric/stable/)

The rough comparison is:

| Concern | Lightning/Fabric | Current repository |
|---|---|---|
| Training loop | Lightning Trainer, or user-owned with Fabric | `Trainer` and phases |
| Distributed/precision mechanics | Lightning Trainer or Fabric | Accelerate |
| Model/training recipe | `LightningModule` | `TrainingStrategy` plus mode/objective |
| Component-family integration | Application-owned | `models/` and family strategies |
| Diffusion caching | Application-owned | Repository-specific strategy/data code |
| Model-format conversion and saving | Application-owned | Repository-specific models/strategies |
| Metadata/provenance | Application-owned | Central repository metadata system |

Lightning’s documented pattern is that `LightningModule` contains the model and defines `training_step()`, validation, and optimizer configuration, while `Trainer` owns the loop, devices, backward, and scheduling. That is plainly similar to where we were heading. [LightningModule documentation](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)

What Lightning does not automatically solve is our internal model integration system:

- multi-component model families;
- interchangeable representation and conditioning features;
- adapter/fine-tune/distillation arrangements;
- family-specific loading and checkpoint conversion;
- latent/text-output caching;
- multiple artifact formats;
- component provenance and metadata.

But those things could theoretically live inside a `LightningModule`. Their existence alone does not justify writing our own trainer framework.

So the honest conclusion is:

- We are not rebuilding Fabric; Accelerate already occupies much of that infrastructure layer.
- We may be approaching a specialized reimplementation of `LightningModule + Trainer`.
- The feature/component strategy work is still distinct, but it does not necessarily require a custom trainer contract.
- Before refining our contract, we should compare three real options:

```text
A. Keep our Trainer and design a deliberately narrow strategy contract.

B. Make the fulfilled strategy a LightningModule and adopt Lightning Trainer.

C. Keep our Trainer/Accelerate runtime, but borrow only the proven
   consumer-owned recipe boundary from Lightning.
```

My current leaning is C, but it needs evidence. The main reason is that our active trainer, phase system, caching, metadata, resource intelligence, and saving lifecycle are already substantial and deliberately controlled. Migrating those into Lightning may cost more and reduce clarity. But if our proposed contract starts accumulating Lightning’s hook surface, that argument collapses.

The next research should therefore be a bounded comparison of our current Trainer against Lightning Trainer/LightningModule and Fabric/Accelerate. If we keep the custom system, the design document should explicitly state what repository-specific requirements justify it and which generic framework features we refuse to reimplement.

<a id="m004"></a>
## 004 User - 2026-07-29T03:28:15.017Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12213](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12213).

we'll need to talk about this more tomorrow. we encountering more involved topics


<a id="m005"></a>
## 005 Assistant (commentary) - 2026-07-29T03:28:23.823Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12217](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12217).

Agreed. This is a real decision point rather than another naming discussion. I’m recording the stopping point in the existing Beads handoff so tomorrow we resume from the Lightning/Fabric comparison instead of accidentally continuing contract design as though that question were settled.

<a id="m006"></a>
## 006 Assistant (final_answer) - 2026-07-29T03:28:42.697Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12227](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12227).

Agreed. We’ve reached a genuine architecture fork, so pausing is sensible.

I recorded the stopping point in Beads. Tomorrow we should compare:

- A narrow custom Trainer/Strategy contract.
- Adopting Lightning Trainer/LightningModule.
- Keeping Trainer + Accelerate while borrowing only Lightning’s recipe boundary.

The comparison should separate repo-specific needs from generic training-framework behavior we should avoid rebuilding.

<a id="m007"></a>
## 007 User - 2026-07-29T17:29:12.474Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12233](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12233).

we might not necessarily want to be constrained by an existing system unless it allows for modification based on our own designs, that would be the deciding factor about whether or not we adopt something. and we also keep the code in our repo, which means we become the maintainers


<a id="m008"></a>
## 008 Assistant (final_answer) - 2026-07-29T17:30:00.586Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12238](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12238).

Right. That changes the adoption question substantially.

It is not:

> Does Lightning already implement similar concepts?

It is:

> Can Lightning serve as a substrate for our design without forcing our design to become Lightning-shaped—and does owning its code save more work than it creates?

If adopted code must live in this repository, we lose one of the main benefits of framework adoption: upstream owns maintenance. We would gain an existing implementation, but inherit:

- compatibility and dependency maintenance;
- distributed/runtime edge cases;
- internal framework complexity;
- upstream merge work if we want later fixes;
- responsibility for understanding code designed around somebody else’s assumptions.

There are several distinct choices:

| Route | Control | Maintenance |
|---|---:|---:|
| Use Lightning as an external dependency | Limited to supported extension points | Mostly upstream |
| Wrap or subclass Lightning | Moderate, potentially brittle | Shared |
| Vendor or fork Lightning | Nearly complete | Ours, with upstream merge cost |
| Vendor only small reusable machinery | High | Ours, but bounded |
| Learn from it and retain our architecture | Complete | Ours, limited to what we actually need |

The deciding test should be whether its foundational invariants are replaceable:

- Can we define our own trainer–strategy contract?
- Can we own lifecycle and phase ordering?
- Can we replace state-ownership assumptions?
- Can one strategy represent multiple components, teachers/students, and attached modules without pretending to be one ordinary model?
- Can caching, metadata, checkpoint conversion, and artifact production remain first-class?
- Can we remove or replace standard hooks without fighting hidden framework behavior?
- Can we use only the parts we value, or does everything assume the entire Lightning object model?

If the answer is “yes, but only after deeply modifying the Trainer and LightningModule internals,” then wholesale adoption probably buys us very little. We would become maintainers of a much larger system while still designing our own architecture inside it.

So my present expectation is:

- Lightning is valuable as prior art and a pressure test.
- A normal dependency is viable only if its public extension points genuinely accommodate our design.
- Vendoring the full framework is unlikely to be attractive.
- Reusing or adapting a small, separable subsystem might be worthwhile.
- Otherwise, we should keep our system deliberately narrow and make sure we are not recreating Lightning features that the repository does not need.

Our design should remain authoritative. Existing frameworks are candidates for implementation leverage, not authorities that redefine the goal.

<a id="m009"></a>
## 009 User - 2026-07-29T18:10:23.495Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12244](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12244).

it looks like the route is probably good as a reference for how some things can be done


<a id="m010"></a>
## 010 Assistant (final_answer) - 2026-07-29T18:10:36.763Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12248](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12248).

That seems like the strongest route.

Lightning becomes architectural prior art:

- How it divides trainer-owned orchestration from authored training behavior.
- How lifecycle calls and typed results can cross that boundary.
- How it handles distributed preparation, optimizers, validation, and checkpointing.
- Where its hook-heavy design becomes difficult to understand or constrain.

We can borrow patterns that survive our requirements without adopting its object model or runtime. The comparison tomorrow should therefore be a pattern audit, not an assumption that we need to integrate Lightning.

Our target remains:

```text
our contract
→ our explicitly authored strategies
→ our trainer
```

Lightning helps us avoid rediscovering solved interface problems—and shows us which framework complexity we do not want.

<a id="m011"></a>
## 011 User - 2026-07-29T18:12:28.339Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12254](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12254).

Okay, so what to do now that we figured that part out?


<a id="m012"></a>
## 012 Assistant (final_answer) - 2026-07-29T18:12:52.848Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12259](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12259).

We have settled the stance, not the contract. The next work should be:

1. Update the direction document

   Record that Lightning/Fabric are prior art, not assumed dependencies. Adoption is only reconsidered if a separable subsystem preserves our design authority and justifies its maintenance cost.

2. Perform a bounded pattern audit

   Compare only the parts relevant to our boundary:

   - loop ownership;
   - strategy/module interface;
   - state ownership;
   - optimizer and distributed preparation;
   - lifecycle hooks;
   - validation and checkpointing;
   - extension and escape mechanisms.

   The output should identify:

   ```text
   borrow this pattern
   avoid this complexity
   not applicable here
   ```

3. Derive the trainer-owned contract from our inventory

   Use a table like:

   | Trainer responsibility | What it needs from strategy | Current leak | Candidate boundary |
   |---|---|---|---|
   | Execute batch | Loss and observations | VAE/TE/denoiser arguments | `TrainingStepResult` |
   | Build optimizer | Trainable parameter groups | Component projections | Generic trainable description |
   | Distributed preparation | Modules requiring wrapping | Family-specific component lists | Generic preparation surface |
   | Save checkpoint | Run coordinates and artifact request | Whole `Trainer` passed back | Typed save request |

4. Pressure-test that candidate

   Test it conceptually against:

   - SD, SDXL, and SD3;
   - adapter and full-model training;
   - pixel versus latent representation;
   - teacher/student distillation;
   - added trainable components;
   - future non-image or LLM-containing models.

5. Then settle state ownership

   Only once we know what the trainer actually requires can we decide whether the strategy owns all loaded state, exposes generic handles, or exchanges typed preparation plans.

After those steps, we should have enough to write an OpenSpec. The immediate next task is updating the direction document and starting the small framework-pattern audit—not implementation.

<a id="m013"></a>
## 013 User - 2026-07-29T18:39:25.989Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12265](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12265).

okay, let's do this, it won't hurt


<a id="m014"></a>
## 014 Assistant (commentary) - 2026-07-29T18:39:38.232Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12269](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12269).

I’ll keep this as architecture research: update the direction record, add a bounded framework-pattern comparison under the existing working-docs folder, and connect its conclusions back to the trainer-contract questions. I’m using OpenSpec exploration for the design stance, Beads for durable handoff, and codebase-memory to keep the comparison grounded in the current Trainer/Strategy implementation.

<a id="m015"></a>
## 015 Assistant (commentary) - 2026-07-29T18:41:08.872Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12346](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12346).

The repo evidence confirms why the comparison matters: the current loop already has a Lightning-like split at one point (`process_batch()` returns loss data while the trainer owns backward/optimizer timing), but the trainer still passes the strategy its own VAE, encoders, denoiser, objective runtime, dtypes, accelerator, and configuration. I’m now isolating which external patterns help repair that boundary without importing their frameworks or hook systems.

<a id="m016"></a>
## 016 Assistant (commentary) - 2026-07-29T18:42:40.390Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12386](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12386).

The comparison is converging on a useful three-layer distinction: Lightning’s `LightningModule` is relevant prior art for the authored recipe boundary; Lightning `Trainer` is the full orchestration framework we do not currently intend to adopt; Fabric and Accelerate are infrastructure toolkits beneath a user-owned loop, with Accelerate already filling that role here. I’m capturing those separately so “borrow a boundary pattern” cannot silently become “vendor a framework.”

> Compaction boundary at selected raw line 12416 (2026-07-29T18:47:10.109Z); generated summary omitted.

<a id="m017"></a>
## 017 Assistant (commentary) - 2026-07-29T18:47:15.188Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12423](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12423).

I’ve finished the substantive comparison; I’m doing the last accuracy pass now, then I’ll record the milestone in Beads and hand back the concrete conclusions. The comparison is treating Lightning/Fabric/Accelerate as prior art, not as an adoption decision.

<a id="m018"></a>
## 018 Assistant (commentary) - 2026-07-29T18:47:30.626Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12437](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12437).

The comparison confirms a useful split: Lightning is closest to the authored-strategy/executor relationship, while Fabric and Accelerate sit lower as runtime infrastructure. I found two phrases that overstated unresolved choices, so I’m tightening those before treating the note as durable.

<a id="m019"></a>
## 019 Assistant (commentary) - 2026-07-29T18:48:00.233Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12461](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12461).

The durable conclusion is now sharper: we are not choosing between “our system” and “Lightning.” We are separating three decisions—strategy contract, trainer orchestration, and runtime infrastructure—so borrowing one boundary pattern cannot silently replace the other two. I’m recording that stopping point before moving on.

<a id="m020"></a>
## 020 Assistant (final_answer) - 2026-07-29T18:48:45.099Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12479](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12479).

The framework comparison is complete, and it supports continuing our model–strategy–trainer design rather than adopting Lightning or Fabric wholesale.

The main conclusion is a three-layer distinction:

- Fulfilled training strategy: repository-authored model/training behavior.
- Trainer: lifecycle, optimization, and orchestration.
- Accelerate: distributed/device/precision infrastructure.

Lightning provides useful precedent for the strategy–executor boundary. Fabric mostly overlaps the infrastructure role Accelerate already serves. We should borrow boundary patterns without reproducing Lightning’s hook framework or surrendering control of our architecture.

Recorded in:

- [framework_pattern_comparison.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/framework_pattern_comparison.md:1)
- [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:796)

The next design milestone is now concrete: derive the actual trainer-facing contract across construction, loaded-state ownership, distributed preparation/rebinding, trainables, batch execution, objective/loss policy, validation, persistence, and observation. That should happen before designing classes or implementing migrations.

I also updated bead `sd-scripts-syv` with this stopping point. `git diff --check` passes; no tests were needed for documentation-only work. The configured `review-mcp` tool was not available in this session, so no external review was run.

<a id="m021"></a>
## 021 User - 2026-07-29T20:47:31.823Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12485](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12485).

this document doesn't really explain much, don't think you can reach any conclusion without seeing the code


<a id="m022"></a>
## 022 Assistant (commentary) - 2026-07-29T20:47:55.879Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12489](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12489).

You’re right. I treated public documentation as enough to classify the projects, but it is not enough to judge whether their implementation could be adapted, selectively reused, or would constrain us in practice. The current “conclusion” should be downgraded to a provisional API-level observation.

I’m going back to the implementation seams themselves: how Lightning binds a module to the trainer, executes a training step, handles automatic/manual optimization, prepares/replaces modules, and serializes state; then the corresponding Fabric and installed Accelerate paths. That is the evidence needed before the comparison can support an architectural conclusion.

<a id="m023"></a>
## 023 Assistant (commentary) - 2026-07-29T20:52:11.870Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12607](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12607).

The code changes the picture materially. Lightning’s public “Trainer calls LightningModule” story hides a bidirectional object graph, dynamic hook dispatch, separate loop/strategy/precision layers, and automatic/manual optimization paths with very different ownership. Fabric likewise returns wrapper objects and contains nontrivial forwarding/rebinding machinery. I’m switching the evidence snapshot from the moving Lightning master branch to the latest stable tag before writing conclusions.

<a id="m024"></a>
## 024 Assistant (commentary) - 2026-07-29T20:56:57.379Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12686](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12686).

The source audit now supports a different kind of conclusion than the first draft: not “these projects look similar, so keep ours,” but “their implementations expose constraints our contract must model.” The biggest newly established constraint is that logical model identity and the prepared execution handle cannot be the same assumed object, and the Trainer needs a generic preparation/rebinding exchange without learning family anatomy.

<a id="m025"></a>
## 025 Assistant (final_answer) - 2026-07-29T20:58:53.134Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12731](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12731).

You were right—the first draft overreached from public documentation. I replaced it with a source-level audit covering Lightning 2.6.5, Fabric, Accelerate 1.11.0, and our active preparation/training paths.

The code establishes that:

- `LightningModule` is not a narrow strategy contract. It has broad, bidirectional coupling with Trainer, hooks, loops, distributed Strategy, wrappers, optimizers, and checkpoint connectors.
- Fabric and Accelerate can replace modules with prepared execution wrappers. This directly exposes something our design must model: stable logical component identity versus current execution binding.
- The Trainer needs a generic preparation/rebinding exchange, but it does not need family-specific knowledge such as denoiser, CLIP-G, or adapter.
- A `training_step()` method alone solves nothing. Its request/result semantics and optimization ownership are the actual contract.
- Automatic and custom/manual optimization represent different ownership arrangements.
- Runtime-resume state remains distinct from logical model-family artifacts.

The bounded conclusion is now: none is a drop-in implementation of our model-family training contract. This does not rule out adopting a separable subsystem later.

See the source-backed findings in [framework_pattern_comparison.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/framework_pattern_comparison.md:627) and the amended direction in [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:796).

Beads was updated and `git diff --check` passes. No tests were needed. `review-mcp` remains unavailable in this session.

<a id="m026"></a>
## 026 User - 2026-07-29T21:23:32.330Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12737](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12737).

Okay, I read the comparison. now lets start talking about what to actually do with our repo


<a id="m027"></a>
## 027 Assistant (commentary) - 2026-07-29T21:23:40.109Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12741](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12741).

Yes. The comparison has done its job; the next step is to stop talking in framework analogies and design the concrete boundary around our existing Trainer, modes, strategies, and Accelerate preparation. I’ll anchor this in the current inventory and start with the first decision that everything else depends on: what object/state the Trainer actually receives and owns.

<a id="m028"></a>
## 028 Assistant (final_answer) - 2026-07-29T21:25:19.753Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12752](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12752).

I think we should commit to this concrete target:

```text
handwritten strategy builder
  combines family integration + mode + objective + selected features
                            │
                            ▼
             FULFILLED TRAINING STRATEGY
              owns model-related meaning
                            │
                            ▼
                         TRAINER
        owns time, optimization, infrastructure, side effects
```

“Trainer gets one thing” should mean one model-training boundary—not literally one Python object containing the optimizer, logger, Accelerator, and everything else.

## Ownership

The fulfilled strategy should own:

- What is being trained.
- Logical model/component bindings.
- Current prepared execution bindings.
- Family behavior and selected reusable features.
- Mode behavior such as fine-tuning versus attached-adapter training.
- Objective-specific batch computation.
- Trainable parameter grouping.
- Model-family artifact persistence.
- Model-facing validation and sampling behavior.

The Trainer should own:

- Epoch and step traversal.
- Gradient accumulation timing.
- Optimizer and scheduler advancement.
- Accelerator and distributed infrastructure.
- Trigger scheduling.
- Logging, monitoring, interruption, and lifecycle reporting.
- Runtime-resume checkpoint timing.
- Calling the strategy through the contract.

The Trainer should eventually lose fields such as `vae`, `denoiser`, `text_encoders`, `adapter`, and `_primary_trainable`. Those are model topology leaking into generic orchestration.

## The concrete runtime exchange

The source audit suggests this flow:

```python
strategy = build_training_strategy(cfg)
strategy.load(load_request)

optimization_plan = strategy.build_optimization_plan()
optimizer, scheduler = trainer.build_optimization(optimization_plan)

preparation_plan = strategy.build_preparation_plan(
    optimizer=optimizer,
    scheduler=scheduler,
)
prepared = trainer.runtime.prepare(preparation_plan)
strategy.bind_prepared(prepared)

for batch in training:
    result = strategy.execute_batch(
        BatchRequest(
            batch=batch,
            coordinates=coordinates,
            phase=TrainingPhase.TRAIN,
        )
    )

    trainer.backward(result.loss)
    trainer.advance_optimizer()
```

These names are illustrative. The important mechanics are:

```text
logical model bindings
    survive for identity, metadata, inspection, and saving

prepared execution bindings
    are replaced/rebound after Accelerate preparation
    and are used for actual forward execution
```

The Trainer sees generic preparation participants with stable IDs. It does not see “SDXL denoiser,” “CLIP-G,” or “adapter.”

## Mode and objective

My current recommendation is that `TrainingMode` and objective implementations remain internally recognizable collaborators, but stop being separate objects the Trainer coordinates.

```text
FamilyIntegration
TrainingMode
Objective
Features
      │
      └── authored into FulfilledTrainingStrategy
```

That preserves the useful separation we already developed without making the Trainer call each piece individually. The strategy author still explicitly assembles everything; there is no resolver or automatic feature selection.

## What replaces the current contract

The new contract should not be another large ABC inheritance list. It needs three enforcement layers:

- A small Trainer-facing behavioral surface.
- Typed request, result, plan, and binding values.
- Shared conformance tests validating actual behavior and lifecycle transitions.

Inheritance alone cannot prove that a family implementation means the same thing by `process_batch`, supports preparation correctly, or uses prepared bindings. The conformance suite can.

## Where I would start

The first design subject should be loaded state and preparation—not `execute_batch()`.

That establishes:

1. What a logical component binding is.
2. How a strategy publishes preparation participants.
3. How optimizers and jointly prepared backends fit.
4. How prepared objects are returned and rebound.
5. What Trainer component fields can eventually disappear.
6. What metadata observes without depending on transient wrappers.

Once that is settled, we can design the batch boundary against a known state model. Starting with `training_step()` first would repeat the mistake exposed by the Lightning comparison: a deceptively simple method surrounded by undefined ownership.

<a id="m029"></a>
## 029 User - 2026-07-29T21:38:46.745Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12758](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12758).

we already knew this, you just listed some things with changed names. also it's not really clear what you mean by "fulfilled startegy" or what "build training strategy" means


<a id="m030"></a>
## 030 Assistant (final_answer) - 2026-07-29T21:39:38.415Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12764](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12764).

You’re right. I turned existing intentions into new nouns without defining new mechanics.

“Fulfilled strategy” should not be a separate object or class. I meant only:

> A `TrainingStrategy` that satisfies the Trainer contract for the configured training arrangement.

That is a validation property, not a useful runtime noun. It also conflates three different states:

```text
constructed strategy
validated configuration
loaded/prepared strategy
```

We should stop calling the object `FulfilledTrainingStrategy` and retain `TrainingStrategy`.

Likewise, `build_training_strategy(cfg)` currently has a precise, modest meaning: the factory selects an explicitly authored family strategy class and instantiates it. It does not compose features, incorporate the mode/objective, fill contract slots, or infer anything.

```python
def build_training_strategy(cfg) -> TrainingStrategy:
    strategy_cls = _STRATEGY_REGISTRY[cfg.model.type]
    return strategy_cls(cfg)
```

I incorrectly used “build” to imply a hypothetical composition stage we have not designed.

The clearer vocabulary is:

```text
Training contract
    What the Trainer requires and what strategy operations mean.

TrainingStrategy
    The concrete repository-authored implementation given to the Trainer.

Strategy construction
    Ordinary explicit Python that instantiates and wires that implementation.

Validation
    Checks that the strategy and configured arrangement satisfy the contract.

Runtime preparation
    Loads and prepares the state used by that strategy.
```

Whether mode and objective become collaborators inside `TrainingStrategy` remains undecided. I shouldn’t have presented that as settled.

The next useful discussion is therefore much less grand:

1. Open the current `TrainingStrategy`/eleven ABC contracts.
2. For every operation, classify it as:
   - genuinely required by Trainer;
   - internal strategy/feature collaboration;
   - model-layer behavior;
   - training infrastructure;
   - obsolete or misleading.
3. From the genuinely Trainer-facing operations, define what each one means and what result the Trainer expects.
4. Only then decide how one concrete strategy is constructed and where mode/objective participate.

That will tell us what `TrainingStrategy` actually is, instead of inventing another name for what we hope it becomes.

<a id="m031"></a>
## 031 User - 2026-07-29T21:42:24.480Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12770](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12770).

you're talking about things with uncertainty that we already looked at and talked about. I do not want to keep cycling through the same points when we've already went over them [strategy_system_direction.md](docs_design/models-strategy-trainer/strategy_system_direction.md) [strategy_system_inventory.md](docs_design/models-strategy-trainer/strategy_system_inventory.md) 


<a id="m032"></a>
## 032 Assistant (commentary) - 2026-07-29T21:42:37.117Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12774](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12774).

You’re right. I’m reopening questions the direction document already settled instead of advancing from its recorded stopping point. I’m using the OpenSpec exploration workflow here only to keep this as design discussion; I’ll read both documents as the source of truth, separate settled decisions from the remaining questions, and continue from the first genuinely unresolved point.

<a id="m033"></a>
## 033 Assistant (final_answer) - 2026-07-29T21:45:13.736Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12823](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12823).

You’re right. I reread both documents fully. The inventory already completed the method/caller classification and explicitly says not to repeat those call-path investigations. The direction also already establishes authored strategy construction, contract fulfillment, feature organization, Trainer ownership, and no automatic resolver.

The recorded next step is:

> Name the minimum contract concerns and their Trainer-facing result semantics.

So that is where we should continue.

The first useful distinction is that contract concerns are not automatically Trainer methods. Today the 25-method ABC collapses these together. The intended relationship should be:

```text
Contract concerns
  describe what the strategy has deliberately assembled and validated

Trainer-facing operations
  execute that completed arrangement at lifecycle boundaries
```

For the first contract version, I think we should deliberately describe the training pipeline we actually have: diffusion training, while avoiding family-specific anatomy.

The required concern filings would be:

| Concern | What must be established |
| --- | --- |
| model participation | Bound component surface and relationships |
| training subject | Which parameters/components are being optimized |
| representation | Pixels, latents, or another explicit clean-state representation |
| conditioning | How predictor conditioning is produced, including an explicit unconditional filing |
| objective | Corruption/noise-level and target semantics |
| prediction | How prepared state reaches the model and what prediction means |
| loss | How prediction and target become the optimization signal |
| runtime preparation | Precision, distributed preparation, component workarounds, and incompatibilities |
| persistence | What kind of trained result this arrangement can produce |

Caching, validation, sampling, weighted prompts, adapters, and similar items remain conditional concerns: if the authored strategy/configuration selects them, their corresponding requirements become mandatory and must validate before execution.

The Trainer should not call representation, conditioning, prediction, and loss as separate services. Those are internal strategy filings that collectively produce the current stable step result:

```text
batch + step coordinates
          │
          ▼
       strategy
  representation
  conditioning
  objective
  prediction
  loss
          │
          ▼
   BatchLossOutput
```

For the initial diffusion contract, I would preserve `BatchLossOutput` rather than rename it. Its semantics need tightening:

- Per-sample loss is the optimization input consumed by Trainer policy.
- Objective observations such as timesteps are returned for adaptive state and reporting.
- Family-specific tensors do not escape.
- Model components and `Accelerator` are no longer step arguments.
- Trainer remains responsible for modifiers, backward, accumulation, clipping, and optimizer advancement.

Other Trainer-facing results would be defined separately:

```text
training-subject filing → optimizer parameter/group description
validation filing       → evaluation result without recorder side effects
sampling filing         → generated artifact/result description
persistence filing      → saved artifact result
```

Loading and preparation are intentionally the next discussion, because that is where the bound-state ownership question belongs. We do not need to reopen mode/objective placement or construction mechanisms yet; the inventory puts those later in the discussion order.

The one point I would examine first is persistence: whether every trainable strategy must file at least one output form, even if the configured run chooses not to save, or whether persistence only enters the contract when output is selected. Everything else above appears necessary to perform the current training step at all.

<a id="m034"></a>
## 034 User - 2026-07-29T21:57:20.466Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12829](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12829).

the lightning detour was for reference and comparison and to see if it makes more sense to take how they do things if it fits under our own system, it doesn't override or get superseeded by our previous discussion and direction

I already said that I want the contract to be the minimum dictated contract by the trainer, that's what "contract" stands for, but it should also define other available capabilities and features as things the trainer is capable of, whatever that means (sampling, checkpointing etc. anything the trainer supports) AND also what the trainer doesn't care about but can be used to create something. this needs to be written in a way where it's self-explanatory and intuitive in code.

^ and this doesn't mean the current trainer mechanics have to be preserved as is either, as we already identified that it doesn't behave optimally with the current strategy system. this is where references can become useful other than just inventing things on the spot, but we want to strive for what we want to turn our repo into, not what other repos do or what we can do today, while acknowledging that we don't have to solve everything today, because new capabilities will inevitably expand the contract, even when we aim for making it comprehensive


<a id="m035"></a>
## 035 Assistant (final_answer) - 2026-07-29T21:59:10.739Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12839](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12839).

Yes—that restores the hierarchy correctly. Lightning was a subordinate implementation reference. It should inform mechanics where useful, not reopen or replace the established direction.

The contract is dictated by the intended Trainer, not mechanically extracted from today’s Trainer. We are allowed to redesign both sides until their interaction expresses the system we want.

I think the contract system has three clearly different surfaces:

```text
TRAINING CONTRACT SYSTEM
│
├── Core strategy contract
│     What every strategy must provide for this Trainer to train.
│
├── Trainer-recognized capabilities
│     Additional operations the Trainer knows how to request.
│     Sampling, validation, caching, checkpointing, generation, etc.
│
└── Strategy feature contracts
      Building blocks available to strategy authors.
      Representation, conditioning, CLIP behavior, latent diffusion,
      weighted prompts, predictor adapters, and so forth.
      The Trainer does not coordinate these.
```

That makes the terminology precise:

- A core requirement exists because the Trainer cannot perform its essential loop without it.
- A capability is understood by the Trainer but not required in every strategy. If configuration selects it, validation requires the strategy to provide it.
- A feature helps an author construct a strategy or capability. The Trainer neither discovers nor calls it.

For example:

```python
from library.strategies.contracts import TrainingStrategy
from library.strategies.capabilities import Checkpointing, Sampling
from library.strategies.features.conditioning import Clip
from library.strategies.features.diffusion import LatentDiffusion
```

Conceptually:

```python
class SdxlTrainingStrategy(TrainingStrategy):
    representation = LatentDiffusion(...)
    conditioning = Clip(...)

    sampling: Sampling = SdxlSampling(...)
    checkpointing: Checkpointing = SdxlCheckpointing(...)
```

The exact Python mechanism remains to be chosen, but it should satisfy these readability rules:

- Core obligations are obvious from `TrainingStrategy`.
- Trainer capabilities have named contracts instead of `hasattr()` checks or raising defaults.
- Features have named interfaces and discoverable package locations.
- A strategy explicitly wires its features and capabilities.
- No feature silently selects or pulls another feature.
- Configuration only chooses among arrangements the strategy author deliberately exposed.
- Compatibility validation examines the completed authored arrangement.

This also gives contract evolution a clean meaning:

```text
Trainer learns a new essential requirement
    → evolve/version the core TrainingStrategy contract

Trainer learns an additional operation
    → add a trainer-recognized capability contract

Strategy authors gain a reusable construction tool
    → add a feature contract and implementation
```

Adding `Sampling` should not force every strategy to implement sampling. Adding a new representation feature should not change the Trainer contract at all. Changing what every strategy must provide to run the Trainer is the meaningful contract evolution.

For the intended Trainer—not necessarily today’s implementation—I think the initial core is approximately:

1. Establish or bind the model arrangement for the run.
2. Cooperate with Trainer-owned runtime preparation.
3. Publish the training subjects/optimization inputs.
4. Execute a training batch and return the result required by Trainer-owned optimization.

Sampling, validation, caching, checkpointing, and artifact export then sit in the Trainer-recognized capability layer. They are still part of the overall contract vocabulary and fully validated when selected; they are simply not prerequisites for computing a training update.

Features such as latent diffusion, pixel diffusion, CLIP conditioning, weighted prompts, adapter execution, or a predictor adapter sit below both. They may be used to implement the core or a capability, but their existence does not enlarge the Trainer API.

The useful Lightning/Fabric lessons fit underneath this direction:

- Lightning shows a workable semantic batch boundary and explicit optimization-ownership variants.
- Fabric and Accelerate show that preparation can replace execution objects.
- Their hook catalogs, object ownership, and class hierarchies do not define ours.

This three-surface contract model is what I think should be established next in the direction document. Then we can design the minimum core against the Trainer we actually want, beginning with what “cooperate with runtime preparation” and “publish training subjects” mean without preserving the current Trainer hub.

<a id="m036"></a>
## 036 User - 2026-07-29T22:03:38.378Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12845](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12845).


# Files mentioned by the user:

## Yeah, this response is **back on the intended track**. It restores the Lightnin…: /home/imi/.codex/attachments/083a886d-b339-4ad7-90f9-5cec153f9360/pasted-text.txt

## My request for Codex:
any comments about this?




Readable attachment: [external contract-surfaces feedback](../snapshots/2026-07-29-contract-surfaces-feedback.md).

<a id="m037"></a>
## 037 Assistant (commentary) - 2026-07-29T22:03:43.099Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12849](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12849).

I’ll read the comment against the direction we just clarified and separate useful refinements from anything that would accidentally reintroduce a resolver, optional-method grab bag, or framework-shaped design.

<a id="m038"></a>
## 038 Assistant (final_answer) - 2026-07-29T22:04:24.922Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12859](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12859).

This is a strong refinement and stays within the established direction. Its best contribution is the consumer rule:

```text
Pipeline orchestration consumes capabilities.
Strategy implementations consume features.
```

That makes the distinction more useful than simply calling one surface “optional.”

I agree with most of the comments:

- Configuration requests use of an authored capability; it does not select or assemble its implementation.
- Lifecycle state is orthogonal to core/capability/feature classification.
- Capabilities may be composed from several internal features.
- Trainer-recognized does not mean Trainer-implemented.
- Package/class examples should remain illustrative until semantic exchanges are designed.
- Binding should not prematurely become one giant `bind()` method.
- The four exchange tables are the correct next design work.

Two nuances are important.

First, “Trainer-recognized capability” should probably mean recognized by training orchestration, not necessarily directly implemented or even directly invoked by the `Trainer` class:

```text
Trainer/pipeline recognizes caching
Data subsystem coordinates traversal and storage
Strategy capability provides family cache semantics
Features provide representation and conditioning codecs
```

Otherwise the redesigned Trainer could become another central service locator.

Second, runtime checkpointing and artifact persistence absolutely need separating, but they may not both be strategy capabilities in the same sense.

```text
Runtime checkpointing
  primarily Trainer + Accelerator infrastructure
  strategy contributes state participants and restoration semantics

Artifact persistence
  primarily strategy/model-family capability
  Trainer supplies timing, coordinates, destinations, and publishing policy
```

So I would not yet encode this:

```python
runtime_state: RuntimeCheckpointing | None
```

as though the strategy owns runtime checkpointing. The corresponding exchange table should determine whether the strategy provides a runtime-state contribution while the Trainer provides the actual capability.

The lifecycle axis is also worth adopting explicitly:

```text
                   authored   validated   bound   prepared
core contract          ●          ●         ●        ●
sampling capability    ●          ●         ●        ●
persistence capability ●          ●         ●        ●
internal features      ●          ●         ●        ●
```

Not every item necessarily needs behavior in every state, and this does not require four Python classes. It means the contract must define when an operation becomes legal and what state it may rely on.

The most important combined conclusion is:

```text
The contract system is broader than one base class.

It defines:
  minimum Trainer obligations,
  pipeline-recognized capabilities,
  strategy-authoring feature vocabulary,
  compatibility rules,
  and lifecycle validity.
```

That lets the contract aim for comprehensive vocabulary without requiring every strategy to implement every operation or pretending we can anticipate every future feature.

The proposed next work is correct: binding, runtime preparation, optimization, and step exchange tables. I would start with binding and runtime preparation together because the comparison showed they are inseparable: materialization establishes logical identity, while preparation may replace the object that must actually execute. Once those exchanges are understood, we can determine which pieces genuinely belong to the minimum Trainer contract instead of guessing method names.

<a id="m039"></a>
## 039 User - 2026-07-29T22:11:28.271Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12865](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12865).

if the direction document needs updates after recent discussion and the comparisons now would be a good time probably


<a id="m040"></a>
## 040 Assistant (commentary) - 2026-07-29T22:11:45.461Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12869](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12869).

Yes. I’m going to update the direction document now so it records the recent clarification instead of leaving the three-surface contract model, lifecycle axis, and subordinate role of framework references scattered across conversation and the comparison note. I’ll use the OpenSpec exploration guidance for the design edit and Beads for the durable stopping point; this remains design work, not production implementation.

<a id="m041"></a>
## 041 Assistant (commentary) - 2026-07-29T22:15:03.226Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12948](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12948).

The direction update is taking shape around the actual clarification, not a new architecture: one contract system with three consumer-defined surfaces, plus lifecycle as a separate axis. I also reconciled this with the earlier rejection of an “optional capability bag,” because without that explanation the document would contradict itself.

> Compaction boundary at selected raw line 12952 (2026-07-29T22:15:41.242Z); generated summary omitted.

<a id="m042"></a>
## 042 Assistant (commentary) - 2026-07-29T22:15:45.357Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12959](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12959).

I’ve made the conceptual update; I’m doing a focused consistency pass now so the new three-surface contract model, lifecycle distinction, and framework comparison don’t contradict the earlier conclusions. Then I’ll preserve the milestone in the active Bead.

<a id="m043"></a>
## 043 Assistant (commentary) - 2026-07-29T22:16:15.397Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12979](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12979).

The update is internally coherent so far. One distinction deserves slightly sharper wording before I finish: resumable checkpoints are pipeline-owned, so the strategy-side capability should be called a runtime-state contribution, not something that sounds like ownership of checkpoint orchestration.

<a id="m044"></a>
## 044 Assistant (commentary) - 2026-07-29T22:16:52.704Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13006](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13006).

The comparison’s useful findings are now folded into the direction without turning Lightning into the architecture: logical versus prepared bindings, explicit preparation exchanges, and the runtime-checkpoint/artifact split. I’m recording that boundary in the active Bead so the next session starts from the semantic exchange work instead of reopening the framework comparison.

<a id="m045"></a>
## 045 Assistant (final_answer) - 2026-07-29T22:17:54.532Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13033](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13033).

Updated [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:122) to reflect the settled direction:

- Defined the three contract surfaces: core, pipeline capabilities, and author-facing features.
- Clarified that configuration requests an already-authored capability; it does not assemble strategies.
- Separated contract classification from authored/bound/prepared lifecycle state.
- Distinguished logical components from prepared execution bindings.
- Separated runtime-state contribution from trained-artifact persistence.
- Kept Lightning/Fabric/Accelerate subordinate as implementation references, not replacement architecture.
- Recorded the next work as four concrete exchanges: binding, preparation, optimization, and step.

The active Bead now preserves these conclusions and unresolved questions. `git diff --check` passed. No tests were necessary for this design-only change. Review MCP was not available, so no review was started.

<a id="m046"></a>
## 046 User - 2026-07-29T22:56:57.103Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13039](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13039).

Nice, I read the updates and it feels a bit better than before. we can probably move onto any remaining relevant pressure testing or open questions before actually diving into coming up with the look trainer contract strategy and any downstream needs, or idk how to consider them separately when they are connected


<a id="m047"></a>
## 047 Assistant (commentary) - 2026-07-29T22:57:04.603Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13043](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13043).

I’m using the OpenSpec exploration workflow here because we’re still pressure-testing the architecture, not yet proposing an implementation. I’ll separate what can be decided independently from what has to be derived jointly across Trainer, contract, strategy, and downstream systems.

<a id="m048"></a>
## 048 Assistant (final_answer) - 2026-07-29T22:57:46.194Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13053](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13053).

They should not be designed separately. The contract is the boundary produced by considering both sides simultaneously:

```text
Trainer need
    ↕ contract exchange
Strategy fulfillment
    ↕
Model/components/features
```

Designing “the Trainer API” first would risk encoding today’s SD-shaped assumptions. Designing the strategy first would risk giving it responsibilities the Trainer should own. The right unit of design is each exchange between them.

Before defining Python classes, I think we need one final pressure-testing pass focused on scenarios—not another inventory or framework comparison.

## Pressure tests still worth doing

Use a small set of deliberately different training arrangements:

1. Standard SDXL fine-tuning  
   Establishes the normal path without treating its anatomy as universal.

2. LoRA/adapters  
   The loaded base model is not necessarily the training subject or final artifact.

3. SD3 with several independently sourced components  
   Tests multiple logical components, optional replacements, and preparation ordering.

4. Teacher/student distillation  
   Tests multiple family participants, read-only versus trainable components, and objective collaboration.

5. Pixel-space training  
   Tests that latent representation, VAE behavior, and latent caching are features rather than core requirements.

6. Prepared-module replacement  
   Tests Accelerate/FSDP/compilation cases where the logical component and executable object differ.

These scenarios should pressure the same four exchanges:

| Exchange | Questions it must answer |
|---|---|
| Binding | What arrangement exists, who materializes it, and who owns its authoritative logical state? |
| Preparation | What does infrastructure receive, what can it replace, and how are executable bindings returned? |
| Optimization | What is trainable, how are groups published, and who owns backward/step/clipping? |
| Step | What does Trainer request, what does strategy compute, and what result comes back? |

That process lets the Trainer, contract, and strategy take shape together.

## Decisions that block the first contract

These need answers during the exchange work:

- Is the bound strategy the authoritative owner of logical model/component state?
- Does the standard contract assume Trainer-owned optimization, with explicit alternative ownership profiles for research strategies?
- How do training mode and objective participate without Trainer unpacking the strategy?
- What is the minimum differentiable step result?
- How does preparation preserve stable identities while replacing execution objects?
- What must be validated before loading, after binding, and after preparation?

## Questions that do not need answers yet

These can remain open while defining the semantic contract:

- Inheritance versus composition.
- Decorators versus marker classes.
- Exact `features/` package structure.
- Attribute, registry, or query-based capability exposure.
- External plugin versioning.
- Arbitrary component assembly.
- Exact cache-codec package placement.

Those are implementation-shape questions. We first need to know what information and behavior must cross each boundary.

The most productive next discussion is therefore the binding and runtime-preparation pressure test together. They establish authoritative state and executable identity; optimization and step design depend on those answers. After all four exchanges survive the six scenarios, we should be ready to turn the result into an OpenSpec rather than continuing open-ended architecture discussion.
