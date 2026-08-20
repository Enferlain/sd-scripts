# Notes (user and agent)

## Current continuation checkpoint (2026-07-30)

The production inventory, framework comparison, and code pressure test are
complete enough. Do not restart by proposing a graph, renaming strategy, or
repeating the Lightning/Fabric comparison.

Current direction:

```text
TRAINER
  defines the training contract and executes the training mechanism
        ↑
STRATEGY
  explicitly defines what is being trained and how
        ↑
MODELS / FEATURES / CAPABILITIES
  are deliberately used to author that strategy
```

Strategy does not mean model family. SD, SDXL, and SD3 strategies are the
maintained default definitions we author for training those standard models.
Model-family behavior is one ingredient, just like objective behavior,
representation features, training-subject treatment, and persistence support.

Do not create adapter-specific or fine-tune-specific strategy classes. That
would regress past the current mode-separated architecture to an older
combinatorial strategy design. Fine-tuning, adapter training, combined
base/adapter training, distillation, or newly attached trainables may instead
be expressed as capabilities/features deliberately selected within a
strategy.

The contract system still has three consumer-defined surfaces:

- core behavior required by the intended Trainer;
- named capabilities coordinated by Trainer or delegated pipeline systems;
- features consumed internally by strategy authors.

Capability is not synonymous with optional. The core may require a compatible
capability from a category, and a selected capability becomes a real
obligation. A capability may provide behavior, typed information for
Trainer-owned mechanics, or both.

Configuration may request use/settings of an already-authored capability. It
does not infer or assemble the strategy. Core/capability/feature classification
is independent from authored/validated/bound/prepared lifecycle state.

`TrainingMode` is not a target top-level runtime authority. Its current
responsibilities must be separated rather than moved as one block:

```text
strategy declaration
  selected training intent and subjects

Trainer / optimization mechanic
  trainable and parameter-group realization, accelerator preparation,
  backward, stepping, and temporal lifecycle

capability/domain implementation
  specialized attachment, extraction, persistence, or other behavior

strategy-internal feature
  reusable behavior used to implement the selected strategy
```

The user currently leans toward mode behavior becoming Trainer concern even
when represented as capabilities, but implementation placement is not yet
decided. It depends on the code shape. Trainer should own the mechanism without
necessarily accumulating technique-specific branches in one class.

Code pressure-test findings:

- `LoadedModelComponent`, `OptimizationPlan`, `OptimizerBuildResult`, and
  `ObjectiveRuntime` are useful typed foundations to evolve.
- the current `LoadedModelComponent.module` conflates logical/original and
  prepared execution bindings;
- `prepare_with_accelerator(trainer)` proves the need for a typed preparation
  plan/result because modes currently mutate modules, optimizer, scheduler,
  gradient-sync handle, and primary trainable through the whole Trainer;
- modes already construct logical/execution optimizer groups, while Trainer
  owns ordinary backward/step lifecycle;
- `BatchLossOutput.loss` is not the tensor used for backward:
  `per_sample_loss` passes through the Trainer-owned loss modifier first;
- diffusion `timesteps` are objective observations, not a universal training
  contract field;
- current `TrainingMode` and `ObjectiveRuntime` interactions must be
  redistributed across the one strategy/Trainer contract boundary.

Next discussion starts with binding and runtime preparation together. Define:

```text
authoritative bound state
stable logical component identity
logical/original binding
prepared execution binding
preparation participants and constraints
prepared-binding result and rebinding guarantees
```

The mode/objective classification is now recorded in
`strategy_system_inventory.md` Milestone 6. It concludes:

- `TrainingMode` dissolves rather than being renamed;
- strategy declares training subjects, objective semantics, and selected
  capabilities;
- Trainer/optimization owns generic trainable realization, precision,
  distributed preparation, optimizer creation, clipping participants, and
  ordinary lifecycle transitions;
- specialized attachment, lifecycle, and persistence behavior remains
  explicit capability/domain behavior coordinated by Trainer;
- objective mathematics remains a strategy-internal feature, while the
  current objective-owned loss modifier is better treated as an optimization
  capability;
- diagnostics should derive from authoritative binding/optimization facts,
  not reconstruct topology through mode.

Two blockers remain before choosing Python APIs:

1. the exact authoritative binding semantics and storage shape;
2. the standard Trainer-owned optimization profile and explicit research
   extension/ownership profile.

After binding/preparation, derive optimization and step exchanges. Only then
choose method names, class/package structure, and create the OpenSpec.

## Binding discussion resumed (2026-08-20)

Questions 1 and 2 now have bounded semantic answers:

- a logical component identity is stable, strategy-scoped, and denotes one
  semantically distinct participant rather than one Python object;
- the standard contract has one normal prepared execution binding per logical
  component;
- named extra routes are capability extensions justified only by materially
  different runtime preparation/callable requirements;
- every logical-component/route pair has exactly one authoritative current
  binding;
- shared or explicitly synchronized derived state may remain one component,
  while independently evolving state requires another component identity;
- original/unwrapped/artifact handles are typed access views, not competing
  execution routes; and
- stale routes cannot remain silently authoritative after their freshness
  guarantee fails.

Still open: concrete identity and route types, refresh mechanics, binding-state
storage ownership, and the full preparation plan/result API. The next question
is how additions and replacements are represented during adapter attachment or
deferred loading.

## Question 3 adapter checkpoint (2026-08-20)

The initial addition/replacement take was pressure-tested against the current
PEFT path and broader adapter shapes. The checkpoint is provisional rather than
a settled direction change.

Current LoRA proves that adding adapter-owned state and attaching its effect are
different operations: the adapter remains separately owned while selected host
`forward` methods are modified in place. VeRA further shows that one adapter
identity may own shared state plus many target-local modules. Prefix-style
state may have no independent forward route, while ControlNet/T2I-Adapter-style
side networks are independently executable participants.

Working operation taxonomy:

```text
declare
materialize/bind
execution rebind
semantic-slot replacement
effect attach/detach/activation
arrangement add/retire amendment
merge/fold transformation
```

Ordinary adapters should normally be declared by the authored strategy before
structural validation. Target resolution and attachment then fulfill that
declaration; they do not make the adapter appear as an arbitrary late list
mutation. Dynamic participant addition remains an explicit extension whose
result must name relationship/capability changes and invalidate affected
prepared routes, optimization plans, and caches.

Q2 is qualified so only execution-capable logical components require a normal
prepared execution route. State-bearing adapter components may instead expose
authoritative state, optimization, and artifact bindings.

Next pressure test: whether one logical adapter identity should correspond to
one independently managed state trajectory/training subject/persistence unit,
with its per-target injected modules treated as qualified substructure.

## User

- versioning for contracts is probably a good idea for the long run. Contracts and trainer might need to evolve over time as more models get added, but the ideal scenario is them not having to, especially the trainer, but this is only possible to accomplish via the repo growing with new capabilites and testing said contract and trainer.

### excerpts

I'd probably make it something like

folder called tools or capabilitiesor features or shared or grab box/
clip.py (a bit dubious since clip is understood as a component but it's not defined and used from a venv, might move to repo hosted definitions, we'll need to weigh the pros and cons)
diffusion/
pixel_diffusion.py
latent_diffusion.py

And such. Thoughts?

Technically, in the ultimate understanding of this system you would also grab box model components so strategies can be built aribtrarily from them, but that's like end game stuff (1 autoencoder vs the other, different text encoders, llms as vision, whatever u desire)

---

regardless of what we do the grab box thing will probaly be the future, at least that's how I feel like right now. When systems become too difficult to reason with, the easuiest method is to just organize them better. the entire strategy system is one attempted organization of the original sd-scripts repo, which was a shithole from an architecture standpoint.

---

what about using decorators instead of names like somethingsomethingfeature or somethingsomethingcomponent or whatever

@feature 
class LatentDiffusion

feels like ppl often forget they exist while they sound useful, but maybe not here, I'm not sure

---

my current stance as we discussed before is that I like the contract idea centralizing strategy (your training "plan") and handing it to the active trainer layer so it doesn't need to track or know about individual parts, but that's not how it happens today, and the contracts despite being made "strict" are not really respected in the way we imagined it when we came up with the system. and the part about being able to grab features and eventually components for your strategy is pretty attractive as a goal. Then there's also decisions (vae vs autoencoder lanugage, specific component, names, handling, separation, etc basically same with "clip") about the models folder

I didn't read through the inventory fully yet, this is just my thoughts. we should work with our design docs (direction and now inventory) in general unless we decide on other details and actions. what should we do next?

---

1 let's try to make the contracts like this:

a basic idea built on what we have today, intuition about what we might have in the future extrapolated from the direction we want to move to, what "training a neural network" means, and what is needed to be able to train something in our repo today (this can be very narrow technically so let's start from where we are at)

---

the "run" doesn't select things, these are defined in the strategy build the same as today, just under the new contract. it might constrain or validate it (where this happens is not decided from what I've read so far). the contract defines what must be fulfilled and what is available to use whether it's a slot that needs to be filled or something that you're picking for yourself. we're not going for a super smart mind reader system that automatically picks things based on implication, at least not for now. strategies should be built with intent. we can validate that the choices are correct, but that's different from a feature dragging another with it, that should still be left to whoever is putting the strategy together. (and this is us, we make the default training strategies for models aka what you see currently)

This process is manual, it's not done by a hydra replacement or a config system, at least not initially anyways. It's the same as how model code is still written by hand. strategy is essentially "the model code" in this repo. there might be a config equivalent that can do the task of putting together a strategy in the future to make things more streamlined, but that's not current goal

---

yes. there is essentially no noob mode on strategy besides "trainer needs something to train chief)" and even that can be overridden if someone wants it, or allowed for mutated/adopted/custom strategies besides the base model training strategies as that would go against the spirit of experimentation and research

if you know this repo it takes a similar approach at a base level, although it's not for training [ljleb/sd-mecha](https://github.com/ljleb/sd-mecha)

---

basically we didn't change the overall goal, because we are still the ones that put the strategies together, we just establish the mechanics around the contract (which covers the constraints as well) and the intended way to interact with the strategy system

if any updates should be made to [strategy_system_direction.md](docs_design/models-strategy-trainer/strategy_system_direction.md) now is a good time before context compacts

---

well this should be dictated by the trainer no? which also dictates the contract. idk if this means strategy passes a bundled item that the trainer then needs to unpack, or how this is handled in python/other repos normally

---

we might not necessarily want to be constrained by an existing system unless it allows for modification based on our own designs, that would be the deciding factor about whether or not we adopt something. and we also keep the code in our repo, which means we become the maintainers

---
