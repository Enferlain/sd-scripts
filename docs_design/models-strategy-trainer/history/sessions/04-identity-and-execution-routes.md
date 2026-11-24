# Logical identity and execution routes: Q1-Q2

Continuation of [chapter03](03-training-mode-and-objective.md), beginning with the first binding-question walkthrough on 31 July and following its resumption on 20 August through the Q2 checkpoint. The next reading begins with Q3 on additions, replacements and attachments.

Captured 2026-11-05T21:38:05.698726+00:00. 12 visible user/assistant messages. Message bodies, quoted recaps, feedback, historical links and line endings are preserved. Reasoning, tools, generated context and compaction summaries are omitted.

[Machine-readable provenance](04-identity-and-execution-routes.json) | [Sources](04-source-records.json) | [Branch map](branch-map.md)

The August user recap confirms that the July answer addressed Q1 and resumes at Q2. No numbering correction is invented here. The block quoted in that recap and the subsequent text labelled Feedback retain their original attribution; text supplied in a user turn is not automatically independently authored by that user.

These early answers describe strategy-scoped keys and a normal callable route. Later adopted rules distinguish authored addresses from run incarnations, allow state-only participants with no fake route, and qualify what shared state implies about participant identity. Those later dispositions are recorded in the notepad rather than silently rewriting this conversation.

The first committed checkpoint copies are [direction](../snapshots/144faf03-strategy_system_direction.md), [inventory](../snapshots/144faf03-strategy_system_inventory.md), and [notes](../snapshots/144faf03-notes.md). They also contain later provisional Q3 material; that material has not been folded into this bounded Q1-Q2 reading.

## Message index

- [001 user: Need to walk through those questions and elaborate a bit more. starting with the first one](#m001)
- [002 assistant: I’m using the OpenSpec exploration guidance here because we’re still defining semantics, not implementing](#m002)
- [003 assistant: The first question was:](#m003)
- [004 user: There was like a 2 week break since we last got into the project, so I think we should get back into it a](#m004)
- [005 assistant: I’m going to refresh the current direction/inventory notes and the active Bead, then resume at question 2](#m005)
- [006 assistant: The written state matches the recap: question 1 is settled provisionally, and the remaining blocker is th](#m006)
- [007 assistant: Before settling question 2, I’m also using the codebase graph to refresh the current `LoadedModelComponen](#m007)
- [008 assistant: We should resume with question 2:](#m008)
- [009 user: Feedback:](#m009)
- [010 assistant: I’m happy with that resolution. It closes Q2 without making multiple routes part of every ordinary strate](#m010)
- [011 assistant: The decision is now recorded in the working direction, inventory milestone 7, and session notes. I’m also](#m011)
- [012 assistant: Agreed. Q2 is now semantically settled.](#m012)

<a id="m001"></a>
## 001 User - 2026-07-31T03:16:32.411Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13895](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13895).

Need to walk through those questions and elaborate a bit more. starting with the first one

<a id="m002"></a>
## 002 Assistant (commentary) - 2026-07-31T03:16:39.091Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13899](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13899).

I’m using the OpenSpec exploration guidance here because we’re still defining semantics, not implementing anything. I’ll focus only on the first question—logical component identity—and make the alternatives concrete against the direction we already settled.

<a id="m003"></a>
## 003 Assistant (final_answer) - 2026-07-31T03:17:31.255Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13909](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13909).

The first question was:

> What identifies a logical component within one strategy?

This is asking how the Trainer and strategy refer to “the same thing” after loading, wrapping, preparation, replacement, and persistence—without relying on a particular Python object.

## What “logical component” means

A logical component is a stable participant in the training strategy.

For example, an SDXL fine-tuning strategy might declare:

```text
denoiser
autoencoder
text_encoder.clip_l
text_encoder.clip_g
```

A distillation strategy might instead declare:

```text
student.denoiser
teacher.denoiser
student.autoencoder
conditioning.clip_l
```

These identify the component’s place and purpose in this particular strategy. They do not identify:

- the Python object currently implementing it;
- the checkpoint file it came from;
- its globally persistent model-catalog identity;
- its current device, dtype, wrapper, or compiled form.

Those are separate properties associated with the logical component.

## Why it needs an identity

A component may pass through several concrete objects:

```text
logical component: student.denoiser
        │
        ├── loaded binding:       SdxlUNet(...)
        ├── prepared binding:     DDP(SdxlUNet(...))
        ├── persistence view:     unwrapped SdxlUNet(...)
        └── metadata entity:      catalog model/component ID
```

If code identifies the component using `id(module)` or the module instance itself, the identity breaks when Accelerator, DeepSpeed, compilation, or another preparation step replaces or wraps that object.

The stable logical identity lets every part of the pipeline say:

```text
Give me the current execution binding for student.denoiser.
Clip the parameters belonging to student.denoiser.
Save the artifact contribution for student.denoiser.
Report metadata about student.denoiser.
```

The Trainer does not need to know that the component is an SDXL U-Net. It only follows the strategy’s declarations.

## My recommended answer

The strategy author explicitly assigns each logical component a stable key that is unique within that strategy instance.

Conceptually:

```python
student_denoiser = ComponentKey("student.denoiser")
teacher_denoiser = ComponentKey("teacher.denoiser")
autoencoder = ComponentKey("student.autoencoder")
```

The exact Python type and naming syntax remain undecided, but the semantics should be:

- Explicitly authored, not inferred from class names.
- Unique within one constructed strategy.
- Stable across loading, wrapping, device placement, and preparation.
- Used in typed exchanges between strategy, Trainer, optimization, metadata, and persistence.
- Opaque to generic Trainer logic—the Trainer compares and looks up keys but does not interpret `"denoiser"` as a hard-coded architecture role.
- Predictable for repository-maintained strategies so diagnostics and metadata remain understandable.
- Extensible enough for repeated roles such as teacher/student or multiple adapters.

That last part is important. A role alone is insufficient:

```text
denoiser       # ambiguous when there are two
text_encoder   # ambiguous when there are three
adapter        # ambiguous when several are attached
```

A logical address distinguishes the participants:

```text
student.denoiser
teacher.denoiser
conditioning.clip_l
conditioning.clip_g
adapter.style
adapter.character
```

## Identity is not role

We should keep these concepts separate:

| Concept | Example | Purpose |
|---|---|---|
| Logical component key | `student.denoiser` | Stable address inside the strategy |
| Capability/role | `denoising`, `trainable`, `conditioner` | What the component can be used for |
| Implementation | `SdxlUNet` | Concrete model code |
| Execution binding | `DDP(SdxlUNet(...))` | Object currently used to execute |
| Source identity | checkpoint/component provenance | Where it came from |
| Catalog identity | persistent repository-wide ID | Which model entity it represents |
| Parameter scope | selected tensors/parameters | What optimization acts upon |

This prevents names from becoming hidden contracts. The Trainer should not assume that everything named `*.denoiser` accepts a particular SD-style call. The actual interaction is defined through the strategy contract and capabilities.

## Additions and replacements

The identity rules should probably be:

- Wrapping or preparation keeps the same logical identity.
- Replacing the implementation that fills an existing declared slot keeps the logical key, while recording a new binding and source/provenance.
- Adding a genuinely new participant creates a new logical key.
- Two concurrently meaningful participants always have different keys.
- Splitting or merging components requires explicit new identities and relationships; it should not happen invisibly.

For example, replacing the checkpoint’s VAE before execution:

```text
logical key: autoencoder
old source: integrated VAE
new source: external VAE
current binding: prepared external VAE
```

The role in the strategy remains `autoencoder`; its source and concrete binding changed.

Attaching a LoRA is different:

```text
denoiser
adapter.style
```

The adapter is not merely another binding for the denoiser. It is a distinct participant related to it:

```text
adapter.style --attached_to--> denoiser
```

## The important boundary

This is an execution-and-contract identity scoped to the constructed strategy. It is not a substitute for the durable model/component identity being designed in the metadata work.

The connection is:

```text
strategy logical identity
    identifies the participant in this training definition

metadata/catalog identity
    identifies the persistent model entity or realization behind it
```

A binding record could associate them, but neither should be overloaded to mean the other.

So my current conclusion is:

> A logical component should be an explicitly declared, strategy-scoped, stable address for one training participant. The strategy defines it; generic systems use it without interpreting family anatomy; concrete objects, persistent model identities, roles, and parameter scopes are associated facts rather than the identity itself.

That gives us a reliable foundation for the next question about whether one logical component can have multiple active execution bindings.

<a id="m004"></a>
## 004 User - 2026-08-20T14:48:27.636Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13922](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13922).

There was like a 2 week break since we last got into the project, so I think we should get back into it and continue on the discussion and recording of the facts/decisions. I see that from the 5 questions that we found, your last response is addressing the first one and is looking for a follow up on the second one

"So, specifically, where were we in the binding discussion?

You had just asked:

What identifies a logical component within one strategy?

And we'd reached the provisional answer:

logical component identity
    = stable strategy-scoped identity for one participant


NOT
    Python object identity
    implementation type
    source checkpoint identity
    catalog/metadata identity
    prepared wrapper identity
    parameter scope

Something along the lines of:

student.denoiser
teacher.denoiser
conditioning.clip_l
adapter.style

with the actual representation (ComponentKey, enum-ish object, opaque token, path, etc.) still intentionally undecided.

The important semantic rule was:

logical identity survives
    loading
    source replacement
    device movement
    precision conversion
    DDP/FSDP/DeepSpeed wrapping
    compilation

while its current execution binding can change.

That lines up with the established direction doc distinction:

logical model/component binding
    identity, source, relationships, metadata, artifact semantics


prepared execution binding
    live wrapper/module that must execute

And the current LoadedModelComponent was explicitly identified as the thing to evolve, not throw away and replace with some generic god-object runtime container.

What comes next

I wouldn't spend another turn reconsidering question 1 from scratch. I think the answer is basically sound enough to carry forward:

One strategy-scoped stable logical identity denotes one semantically distinct training participant. Concrete bindings can change underneath it. Concurrently distinct participants require distinct identities.

The next question is actually where it starts getting spicy:

Can one logical component have more than one active execution binding, or must different uses receive distinct logical identities?

Because that's where we have to distinguish cases like:

one denoiser
    original module
    DDP wrapper

which intuitively feels like one component, one authoritative execution binding, from:

same underlying weights
    teacher execution
    student execution

or:

one logical model
    compiled inference path
    training/distributed path

where "one binding per component" becomes less obviously sufficient.

I'd resume exactly there. That question will force us to define what execution binding actually means, which in turn makes questions 3–5 about replacement, persistence views, and the authoritative binding map much easier instead of answering them independently."


<a id="m005"></a>
## 005 Assistant (commentary) - 2026-08-20T14:48:40.038Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13926](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13926).

I’m going to refresh the current direction/inventory notes and the active Bead, then resume at question 2 without reopening question 1. I’m using OpenSpec exploration for the architecture discussion and Beads only to preserve the resulting decisions across sessions.

<a id="m006"></a>
## 006 Assistant (commentary) - 2026-08-20T14:49:01.730Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13936](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13936).

The written state matches the recap: question 1 is settled provisionally, and the remaining blocker is the binding model rather than component naming. I’m checking the exact wording already recorded so the next conclusion extends it instead of duplicating or contradicting it.

<a id="m007"></a>
## 007 Assistant (commentary) - 2026-08-20T14:49:22.907Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13946](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13946).

Before settling question 2, I’m also using the codebase graph to refresh the current `LoadedModelComponent` and preparation behavior. That matters because the answer should distinguish a real need in this repository from hypothetical multi-binding machinery.

<a id="m008"></a>
## 008 Assistant (final_answer) - 2026-08-20T14:51:30.038Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13993](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13993).

We should resume with question 2:

> Can one logical component have more than one active execution binding, or must different uses receive distinct logical identities?

The clean answer is neither “always one” nor “arbitrarily many.” The useful invariant is:

```text
(logical component, execution purpose)
    → exactly one authoritative current execution binding
```

## Why one binding per component is too strict

Consider one denoiser used for both training and sampling:

```text
logical component
  denoiser

execution purposes
  training → distributed/precision-prepared wrapper
  sampling → compiled inference callable
```

These can still represent the same logical component if:

- they use the same authoritative model state;
- sampling is expected to observe the current trained weights;
- replacing one execution route does not create a new semantic participant;
- their different objects exist only because their execution requirements differ.

Giving them separate logical component identities would incorrectly claim there are two denoisers in the strategy.

## Why an unrestricted list of bindings is too loose

This would be dangerous:

```python
bindings["denoiser"] = [module, ddp_module, compiled_module, unwrapped_module]
```

Nothing would explain:

- which one must receive training forwards;
- which one sampling should use;
- which object owns current weights;
- whether the compiled path is fresh;
- which one may be used for persistence;
- whether bypassing the wrapper is safe.

The contract would have preserved all the ambiguity we are trying to remove.

Instead, bindings need purpose:

```text
denoiser
  training execution → DDP-wrapped module
  sampling execution → compiled inference route
  artifact view      → unwrapped/source-aware module
```

The artifact view is not necessarily an execution binding. It is an access view for state extraction and serialization.

## The normal case remains one execution binding

The most common transformation is:

```text
original module
    ↓ Accelerator/DDP/DeepSpeed/compile preparation
prepared callable
```

That is not two active execution bindings.

For example:

```text
logical component: denoiser

original/logical handle:
  SdxlUNet

training execution binding:
  DDP(SdxlUNet)
```

Forward-like training behavior must use the DDP wrapper. The original handle may remain reachable for metadata, inspection, unwrapping, or persistence, but it is not an equally valid training route.

The current code does not maintain this distinction. `LoadedModelComponent.module` initially holds the loaded module, while the mode preparation paths replace Trainer projections with prepared objects. That confirms the ambiguity identified in the direction document rather than changing our conclusion. See [components.py](/home/imi/Projects/sd-scripts/library/models/components.py:43), [adapter_mode.py](/home/imi/Projects/sd-scripts/library/training/modes/adapter_mode.py:243), and [finetune_mode.py](/home/imi/Projects/sd-scripts/library/training/modes/finetune_mode.py:234).

## When uses require distinct logical identities

Different uses need different logical identities when they represent independently meaningful participants, even if they initially come from the same file or weights.

### Teacher and student

```text
student.denoiser
teacher.denoiser
```

These are distinct because they have different:

- roles in the objective;
- update policies;
- gradient policies;
- lifecycle behavior;
- potentially diverging state.

They might share a source relationship:

```text
teacher.denoiser --initialized_from--> model X
student.denoiser --initialized_from--> model X
```

But shared origin does not make them one logical participant.

### Live weights and EMA weights

```text
denoiser
denoiser.ema
```

EMA has independently evolving state and potentially separate persistence semantics, so it is a distinct logical component.

### Several adapters

```text
adapter.style
adapter.character
```

Even if both attach to the same denoiser, they have distinct state and meaning.

## When uses should not create new identities

These are technical representations of the same participant:

- raw module versus DDP wrapper;
- wrapped versus unwrapped access;
- device or dtype movement;
- a composition such as compiled-plus-distributed wrapping;
- distributed replicas managed internally by the backend;
- train/eval mode changes on the same binding.

Infrastructure replicas should generally remain invisible to strategy-level logical identity. Each process receives the locally authoritative execution binding for the same logical component.

## The harder same-component case

A component may legitimately expose more than one prepared execution route:

```text
denoiser
  training route → distributed wrapper
  sampling route → compiled inference wrapper
```

I think the contract should permit this, but only through explicitly declared execution purposes or channels. The exact names remain undecided.

The invariant would be:

```text
one logical component
  may have several named execution bindings

one execution purpose
  must resolve to exactly one authoritative binding

unnamed competing bindings
  are invalid
```

It also creates a required freshness rule. If the sampling route is a copied or compiled representation, the system must know whether it still reflects the current trained state. Otherwise it is no longer safely just another route to the same logical component.

## Provisional decision

I recommend recording question 2 as:

> A logical component may expose multiple named execution bindings when they are technical routes to the same authoritative logical state. Exactly one current binding is authoritative for each execution purpose. Original, unwrapped, inspection, and artifact handles are typed views rather than competing execution bindings. Semantically distinct or independently evolving participants require distinct logical identities even when they share a source or underlying implementation.

In compact form:

```text
semantic difference
    → different logical identity

technical execution difference
    → same logical identity, named execution binding

inspection/persistence access
    → typed view, not an execution binding
```

This is slightly more capable than “one component, one object,” while remaining strict enough that the Trainer and strategy cannot accidentally choose whichever module reference happens to be available. The current graph generation matches the branch and reported no coverage issue for the three relevant source files, with the graph’s normal best-effort limitation.

<a id="m009"></a>
## 009 User - 2026-08-20T14:59:10.314Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 13999](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:13999).

Feedback:

I think the Q2 resolution is already basically right. I’d want to pin down four things before calling it settled:

1. **What qualifies as a distinct “execution purpose”?**
   I don’t want `training`, `validation`, `sampling`, `preview`, etc. automatically becoming separate binding slots just because callers have different intentions. A separate purpose should exist only when the component genuinely needs a different prepared/callable representation.

   So:

   ```text
   training + validation both use DDP(module)
       → one execution binding

   training uses DDP(module)
   sampling requires separately compiled route
       → potentially two execution bindings
   ```

2. **What does “same authoritative model state” mean?**
   This is the biggest semantic point. If two routes physically share parameters, easy. If one is a copy/cache/compiled representation, we need to decide what makes it still a route of the *same component* rather than independently evolving state.

   I’d distinguish:

   ```text
   shared live state
       → clearly same component

   derived state with explicit refresh/synchronization semantics
       → can still be same component

   independently evolving state
       → different logical component
   ```

   That settles cases like EMA immediately: EMA is another component, not merely `execution.ema`.

3. **Are multiple execution bindings part of the standard core contract or an explicit extension?**
   I lean toward the standard invariant being very simple:

   ```text
   component → one normal prepared execution binding
   ```

   while allowing named extra routes when a selected capability genuinely requires them.

   That would stop every normal strategy from carrying a map of execution purposes it never uses, while preserving the more general semantic model you described.

4. **What happens when an extra route becomes stale?**
   Suppose:

   ```text
   denoiser.training → live distributed model
   denoiser.sampling → separately materialized compiled model
   ```

   Training takes another optimizer step. Is sampling now:

   ```text
   invalid
   stale-but-allowed
   automatically refreshed
   refreshed on next use
   ```

   We don't need to design the mechanism yet, but the contract should say that a binding cannot remain silently authoritative once its freshness guarantee has stopped holding.

So the main thing I want to clarify is really this boundary:

```text
multiple representations of one current participant
vs.
multiple participants/state trajectories that happen to share an origin
```

Your teacher/student, EMA, adapters, DDP/unwrapped examples already draw that boundary well.

If we're happy with:

> **Distinct execution purposes are created only by materially different runtime preparation requirements, and every route under one logical component must have a defined relationship to that component's authoritative current state. Independently evolving state means a distinct logical component.**

then I’d consider Q2 semantically settled and move on.



<a id="m010"></a>
## 010 Assistant (commentary) - 2026-08-20T14:59:24.652Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 14003](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:14003).

I’m happy with that resolution. It closes Q2 without making multiple routes part of every ordinary strategy, and the freshness rule prevents “same component” from becoming a loophole for hidden state divergence.

Because this is now a settled semantic decision, the OpenSpec exploration guidance points us to record it in the direction/inventory notes before moving to Q3. I’ll capture it as a bounded invariant, while leaving representation, refresh mechanism, and storage ownership open.

<a id="m011"></a>
## 011 Assistant (commentary) - 2026-08-20T14:59:59.741Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 14014](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:14014).

The decision is now recorded in the working direction, inventory milestone 7, and session notes. I’m also updating the active Bead with the compact version so a future session can recover the conclusion without rereading the entire discussion.

<a id="m012"></a>
## 012 Assistant (final_answer) - 2026-08-20T15:00:48.239Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 14028](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:14028).

Agreed. Q2 is now semantically settled.

The central rule is:

```text
standard:
  logical component → one prepared execution binding

extension:
  selected capability may add named routes
  only for materially different runtime representations
```

The identity boundary is the state trajectory:

```text
shared live state
  → same component

derived state with explicit freshness/synchronization
  → may remain the same component

independently evolving state
  → distinct component
```

A caller’s intention alone—training, validation, preview—does not create an execution route. A stale route must become invalid, be explicitly permitted under a staleness guarantee, or be refreshed; it cannot silently continue representing current state.

I recorded this in:

- [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:1419)
- [strategy_system_inventory.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_inventory.md:1295)
- [notes.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md:127)
- Bead `sd-scripts-syv`

The following remain intentionally undecided:

- concrete component-key and route types;
- freshness/refresh implementation;
- authoritative binding-state owner;
- preparation plan/result API.

Question 3 is next:

> How are additions, replacements, and delayed materialization represented during adapter attachment or deferred loading?

That will distinguish changing an existing component’s binding from introducing a genuinely new logical participant.
