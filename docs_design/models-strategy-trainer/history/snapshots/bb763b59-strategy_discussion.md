The full session gives a more nuanced answer. The active-strategy design was not completely arbitrary, but it was never cleanly chosen either.

The central mistake was that we considered only two options:

```text
A. Trainer owns model state and passes it into strategy calls
B. Strategy owns model state and Trainer calls the bound strategy
```

Option A recreated the existing back-and-forth, so option B looked better.

The missing option was:

```text
C. Authored strategy produces an accepted execution arrangement
   and then Trainer executes that arrangement itself
```

That is the option you expected.

### How the drift happened

Early in the session, you were explicit:

> “the whole point of the strategy system is that it compiles everything needed for the training”

and:

> “why does trainer call things from strategies if I said it should produce one thing”

The initial code audit correctly left state ownership unresolved. It described both:

- Trainer-owned runtime passed into strategy behavior.
- A stateful bound strategy called by Trainer.

Then, after you parked the graph discussion, the agent said:

> “a deliberately stateful, bound family strategy currently appears more coherent—but we need to test that”

That was still provisional.

Shortly afterward, the discussion moved to contract vocabulary, features, CLIP, latent/pixel diffusion, and the grab-box organization. The stateful-strategy proposal was not actually tested. Nevertheless, subsequent responses began using:

```python
strategy.load(...)
strategy.prepare(...)
strategy.training_step(...)
```

as if it were the consequence of the contract discussion.

### Why it seemed reasonable at the time

There were several pressures pushing the agent toward that interpretation:

- The current Trainer stores family-specific components and passes them back to strategies.
- A bound strategy appeared to keep family anatomy behind one boundary.
- You described the strategy as a training “plan” handed to the active Trainer.
- You also said strategy is essentially “the model code” in this repository.
- You rejected `PreparedTrainingProgram` because it was merely another synonym for a method-bearing strategy.
- You wanted fewer abstractions and did not want another meaningless runtime container.

The agent combined those points into:

> One complete active strategy object owns model-facing state and exposes narrow operations to Trainer.

That does solve the back-and-forth. But it is not the only way to solve it.

Rejecting a redundant `PreparedTrainingProgram` class should have meant:

> Don’t add a second façade with the same callback methods.

It should not have meant:

> There can be no semantic distinction between the authored strategy and the accepted arrangement that Trainer executes.

### The inventory reinforced the untested assumption

After the inventory, you said:

> “I like the contract idea centralizing strategy (your training ‘plan’) and handing it to the active trainer layer so it doesn’t need to track or know about individual parts”

This statement supports a single complete handoff, but it does not say whether Trainer:

- calls the strategy as an active service, or
- consumes and executes the completed plan.

The agent selected the first interpretation and answered with a bound strategy containing `training_step()`. From there, “Trainer executes the strategy” became overloaded:

```text
Meaning 1:
Trainer invokes behavior on an active strategy object.

Meaning 2:
Trainer executes the arrangement produced by strategy authoring.
```

Nobody stopped to distinguish those meanings.

### Q5 hardened it further

The most important later turn was your question:

> “binding authority means that strategy is not what the trainer receives?”

The answer said no and proposed:

```text
Trainer
    ↓
TrainingStrategy
    ├── step behavior
    ├── capabilities
    └── BindingAuthority
```

That answer addressed whether the binding authority should become a second public object. It did **not** separately ask whether the strategy should remain an active runtime collaborator.

External feedback then favored:

```text
authored TrainingStrategy
    → active per-run strategy
        ├── behavior
        └── BindingAuthority
```

You accepted the Q5 explanation except for its conservative-v1 recommendation, and authorized the documentation update. That gave the active-strategy topology a form of tacit approval—but it was bundled inside the binding-ownership discussion, not isolated as its own architectural choice.

### Revised conclusion

The fuller history changes my previous conclusion slightly:

- The active strategy did not appear from nowhere.
- It was chosen because it cleanly solved the current Trainer/state/strategy inversion.
- Your wording sometimes allowed that interpretation.
- But the choice was never separated from the other concerns and explicitly settled.
- The simpler “strategy produces something Trainer executes” route was confused with a redundant runtime façade or a premature strategy graph and discarded alongside them.

The documents now contain both interpretations:

```text
High-level flow:
authored → validated → bound → prepared → Trainer
```

This suggests the strategy finishes its job before Trainer execution.

But detailed passages say:

```text
Trainer ↔ active TrainingStrategy
```

This makes the strategy a runtime collaborator.

That contradiction is real.

The useful design work remains valid: one binding authority, stable participant identities, revisions, routes, preparation, and narrow capability contracts. What needs reopening is only the containing runtime relationship:

> After strategy authoring and contract establishment, what exactly does the Trainer receive, and does the authored strategy remain behaviorally active?

The missing third option should now be considered directly, without turning it into a graph system or inventing another method-bearing synonym.

---

what is exactly the best version of the architecture idea? is the question. what is the best to work with, the most intuitive, and in general the one that you would want to make work in a trainer that has our goals in mind. I've treated strategy as a recipe in my mind, but clearly even in this idea it's not something that is taken and executed, and from what I've seen doing that would mean everything that it doesn't provide moves to being done somewhere else, and that's what `PreparedTrainingProgram` was

---

The version I would build is:

> The strategy is a handwritten recipe that is consumed during contract establishment. The Trainer does not execute or call the authored strategy object afterward. It executes one accepted, run-scoped arrangement produced from that recipe.

```text
HANDWRITTEN STRATEGY
chooses and wires ordinary Python behavior
(models, objective, features, capabilities, training subjects)
                  │
                  ▼
CONTRACT ESTABLISHMENT
checks completeness and compatibility
creates the run authority and accepted arrangement
                  │
                  ▼
ACCEPTED TRAINING RUN
accepted behavior + binding state + capabilities + optimization intent
                  │
        materialized / prepared
                  │
                  ▼
TRAINER
owns the loop, time, backward, optimization,
infrastructure, triggers, and observation
```

The important distinction is that the authored strategy stops participating, but the behavior it selected does not disappear.

For example, an SDXL strategy might wire together:

```text
latent representation
+ SDXL conditioning
+ chosen objective
+ denoiser invocation
+ base-loss behavior
```

Contract establishment turns that composition into an accepted training computation. During training, the Trainer invokes that computation. It does not call the original strategy and does not pass the VAE, text encoders, denoiser, objective runtime, Accelerator, configuration, and flags back into it as the current loop does.

A normal step would conceptually be:

```text
Trainer obtains batch and step coordinate
    ↓
accepted training computation uses its prepared execution routes
    ↓
returns base differentiable result and relevant observations
    ↓
Trainer applies accepted loss/optimization policy
    ↓
Trainer performs backward, clipping, optimizer advancement, triggers, logging
```

There is still a function call and a result—model-specific computation must execute somewhere—but it is not Trainer↔authored-strategy back-and-forth. It is the Trainer directly using a dependency that was already selected, validated, and bound for this run.

### Where everything belongs

| Concern | Defined by | Executed/owned by |
| --- | --- | --- |
| Model components, objective, feature choices, training subjects | Authored strategy | Established into the accepted run |
| Model/objective batch computation | Behavior explicitly wired by the strategy | Accepted computation invoked by Trainer |
| Completeness and compatibility rules | Training contract system | Contract establishment and run authority |
| Current logical and prepared bindings | Accepted run authority | Contract-governed runtime transitions |
| Backward, accumulation, clipping, optimizer timing | Trainer contract/library | Trainer |
| Sampling, caching, validation, persistence | Strategy-selected capabilities | Trainer/pipeline coordination through accepted capability endpoints |

That addresses the worry that anything omitted by the strategy simply moves to an arbitrary place. There are only three legitimate destinations:

1. Variable training meaning must be supplied or selected by the strategy.
2. Generic training machinery deliberately belongs to the Trainer.
3. Runtime realization belongs to the binding/preparation system.

If a required concern has none of those owners, the contract rejects the recipe. “Complete” means complete for the Trainer’s contract; it does not mean reimplementing the Trainer inside every strategy.

### What this means for `PreparedTrainingProgram`

The earlier proposal contained one correct idea and one bad implementation idea.

The correct idea was that a passive authored recipe needs a distinct accepted runtime product.

The bad part was giving `PreparedTrainingProgram` methods such as `train_step()`, `validate()`, and `save()`, making it another all-purpose strategy/mini-Trainer façade. That earned the criticism that it was merely a synonym.

The useful runtime product should instead be a structured package containing separate things such as:

```text
accepted declarations
binding authority
composed training computation
capability implementations
optimization intent/runtime
prepared execution routes
```

The Trainer consumes those explicit parts through narrow contracts. The package itself does not become a giant lifecycle object. Whether the eventual name is `AcceptedTrainingPlan`, `TrainingRun`, or something else should wait until its code shape is visible.

### Why I prefer this over the alternatives

I would not keep the authored strategy as the active, mutable runtime façade. It is somewhat simpler initially, but it mixes three different meanings:

```text
recipe
+ accepted contract
+ live per-run state and callbacks
```

That encourages the same lifecycle-hook growth and circular runtime relationship we disliked in the Lightning comparison.

I also would not turn the recipe into a fully declarative graph or DSL. Then a generic interpreter would need to understand every possible training computation, or we would reproduce ordinary Python calls in a much harder-to-read form.

The recommended middle is:

- ordinary Python strategy authoring;
- one genuine transition from authored recipe to accepted run;
- ordinary Python behavior implementations selected by that recipe;
- one run-scoped authority for mutable identity and bindings;
- a Trainer that remains generic without understanding SDXL anatomy.

The current documents contain both halves of this idea but currently join them incorrectly. They correctly describe handwritten composition, validation, and a shared executor, but later say the Trainer receives and repeatedly interacts with the strategy. The present `process_batch()` boundary demonstrates why that becomes inverted: the Trainer passes its own runtime back into the strategy. See [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:149), [training_loop.py](/home/imi/Projects/sd-scripts/library/training/phases/training_loop.py:461), and [diffusion.py](/home/imi/Projects/sd-scripts/library/strategies/sdxl/diffusion.py:143).

So my direct answer is: **strategy as a consumed recipe, followed by an accepted run arrangement that the Trainer executes, is the clearest and strongest architecture for our goals.** The earlier `PreparedTrainingProgram` discussion rejected a redundant façade, but it overreached by also discarding the genuine need for an accepted product between authoring and execution. The next design update should correct that topology before we attempt to settle the optimization or per-step details.

---

Need to consider all angles. Complexity, performance, extensibility. How do these interact with each model of the architecture, what happens if the strategy is not available anymore, what happens if you move something that goes into the strategy either way out of it to another place, etc. I wouldn't want per step movement between the strategy and the trainer but the solution needs to be something that's not just putting it somewhere else but solving the problem of it can be solved. Which is that all models and training plans can have unique requirements. Whats not unique, the trainer should not need from the strategy that its running. Current day code is not that.&#x20;

---

You’re right: merely replacing `strategy.process_batch(...)` with `program.step(...)` would only move the same problem. The design has to eliminate unresolved strategy decisions from the hot path.

The strongest version is:

> All variability is resolved before training starts. The accepted plan is then linked to the prepared models and turned into run-specific execution code. During training, the Trainer never queries the authored strategy and never passes its runtime state back into it.

## The central shape

```text
OPEN AUTHORING WORLD
models and training plans may have unique requirements

handwritten strategy recipe
  chooses and wires those requirements
                    │
                    ▼
contract establishment
  validates the complete arrangement
  freezes the accepted choices
                    │
                    ▼
accepted plan
  durable description of what was accepted
                    │
                    ▼
binding and preparation
  load models
  establish identities
  select prepared execution routes
  realize optimization
  build run-specific execution
                    │
                    ▼
CLOSED RUNTIME WORLD

prepared run
  direct prepared model references
  specialized training computation
  realized capabilities
  realized optimization state
                    │
                    ▼
Trainer hot loop
```

The important operation is not “move behavior from strategy to another object.” It is:

> Resolve, validate, bind, and specialize the variable behavior once so the Trainer does not have to coordinate it repeatedly.

That is similar to linking a program. The strategy is the source definition. The accepted plan records what the source means. Preparation links it to the live models and backend. The Trainer executes the linked result.

## What should happen during a step

The current call is wrong because almost everything passed to `process_batch()` is static run state:

- model components;
- objective runtime;
- dtypes;
- Accelerator;
- configuration;
- trainability flags;
- execution roles.

That state lives in the Trainer, gets handed back into the strategy, and must be interpreted again every step. See [training_loop.py](/home/imi/Projects/sd-scripts/library/training/phases/training_loop.py:461).

The target hot path should carry only things that genuinely change per step:

```text
batch
step/epoch coordinates
possibly explicit random or execution coordinates
```

The prepared run would already contain a specialized computation bound to the correct prepared execution routes:

```python
base_result = prepared_run.training(batch, step_coordinates)

final_loss = trainer.apply_optimization_policy(base_result)
trainer.backward_and_advance(final_loss)
```

That one call is not materially avoidable: some model-specific code must execute the batch. Eliminating even that call would require generating an entire specialized Trainer for every plan, with much greater complexity and negligible benefit.

What we can eliminate is the movement and rediscovery of static state. The prepared computation should already hold direct references to the appropriate wrapped models and selected objective/conditioning/representation behavior.

This is also better for performance:

- contract checks happen once;
- component lookup and compatibility resolution happen once;
- execution routes are bound once;
- no model dictionaries or family roles are interpreted per step;
- the resulting callable can potentially be compiled independently;
- Python call overhead is insignificant beside model execution;
- prepared wrappers cannot accidentally be bypassed by retrieving the wrong Trainer field.

If a binding changes, the authority invalidates the affected prepared computation and rebuilds it at a safe point. It does not introduce dynamic route resolution into every model operation.

## What remains when the strategy is gone

The authored strategy object can disappear, but its meaning cannot.

The accepted system must retain two different things:

```text
accepted plan
  frozen semantic choices
  implementation identities and settings
  participant and relationship declarations
  capability selections
  optimization intent

prepared runtime
  live models and wrappers
  current execution routes
  optimizer/scheduler state
  specialized executable computations
```

These do not necessarily require two large public classes. They are two different kinds of information:

- The accepted plan explains and reconstructs the run.
- The prepared runtime executes this particular process/backend realization.

This distinction matters for replacement. Suppose the denoiser is compatibly replaced after the original strategy object is gone:

1. The authority accepts the new binding.
2. Existing computations depending on the old route become stale.
3. Preparation produces the new execution route.
4. The frozen accepted plan still describes how the training computation is assembled.
5. A new prepared computation is linked against the new route.
6. It is published atomically for later steps.

Therefore the accepted plan must retain either reconstructable implementation descriptions or narrow factories for rebuilding runtime behavior. It cannot retain only an opaque one-shot closure if replacement and exact resume are required.

That is not secretly keeping the original strategy active. The distinction is:

```text
strategy
  may choose and alter the composition while authoring

accepted plan
  records one frozen composition and can only realize that composition
```

## Restart and resume

There are three different cases.

### Same process

The strategy object is unnecessary after acceptance. The prepared run already holds executable implementations and live bindings.

### Exact resume in another process

The snapshot should preserve:

- the accepted plan and contract version;
- stable implementation identifiers and settings;
- run authority and participant identities;
- relevant binding and optimization revisions;
- optimizer/scheduler and capability state;
- provenance/fingerprints of the code used.

Python objects and backend wrappers are reconstructed.

The corresponding implementation code still has to be installed. No architecture can reproduce arbitrary Python behavior after its code has vanished. The system should detect that situation and reject exact resume rather than silently run changed behavior.

For maintained library behavior, stable implementation identities and typed settings should make reconstruction routine. A custom research implementation that wants cross-process resume must supply the same reconstruction/state contract. Otherwise it can explicitly lack exact-resume capability.

### New run from an artifact

The old strategy is not needed. A new strategy may load the artifact into new participant incarnations, with lineage back to the artifact and source run. This is not exact runtime resume.

## Moving things out of the strategy

There are three different questions that we have been treating as one:

1. Where does the implementation code live?
2. Who selects and wires it?
3. Who invokes or owns it at runtime?

For example:

| Concern | Implementation home | Selection | Runtime use |
| --- | --- | --- | --- |
| DDPM objective mathematics | `library/objectives/` | strategy recipe | linked training computation |
| SDXL conditioning | reusable model/feature module | strategy recipe | linked training computation |
| Optimizer algorithm | `library/optimization/` | accepted policy/configuration | Trainer-owned optimization runtime |
| Accumulation and backward | Trainer/infrastructure | core contract | Trainer |
| Sampling semantics | capability implementation | strategy recipe | prepared sampling capability |
| Sampling cadence/output | pipeline configuration | run configuration | Trainer/pipeline |

Moving DDPM code out of a strategy class does not mean putting DDPM branches in the Trainer. The strategy still selects and wires DDPM. The accepted plan retains that selection. The prepared computation invokes it. Only its reusable implementation lives elsewhere.

A concern should move completely into the Trainer only when its mechanism is generic across the accepted contract. If only the settings vary, the strategy or run configuration declares settings once and the Trainer realizes them. If the behavior itself varies, it remains part of the accepted plan’s executable composition.

## How unique requirements extend the system

No Trainer can accept literally every future training algorithm without either evolving or embedding an entire programming language. The useful goal is that extensions occur at the correct level:

| New requirement | Necessary change |
| --- | --- |
| New combination of existing models/features/objectives | Write a new strategy recipe |
| New implementation of an existing behavior | Add the implementation and its conformance rules; select it in a strategy |
| New optional operation the pipeline can coordinate | Add a named capability contract and implementation |
| New policy within existing Trainer ownership | Expand the relevant typed policy |
| Fundamentally different loop or optimization ownership | Add an explicit Trainer contract/profile |

Examples of the last category could include genuinely alternating adversarial optimization, unusual multi-stage advancement, or an algorithm that must decide when backward and optimizer advancement happen. We should not disguise those as an ordinary `training_step()` result if they actually require a different execution contract.

This preserves research extensibility without making every strategy an unrestricted callback provider.

## Architecture comparison

| Model | Complexity | Runtime performance | Extensibility | Durability | Main problem |
| --- | --- | --- | --- | --- | --- |
| Active stateful strategy | Low initial, high hidden coupling | Fine computationally, poor static binding | Easy arbitrary hooks | Reconstructing mutable recipe/runtime object is awkward | Recipe, state, and executor become one object |
| Declarative plan interpreted every step | High interpreter/DSL complexity | Repeated interpretation or graph machinery | Every new behavior expands the language | Potentially excellent | Requires predicting a universal training language |
| `PreparedTrainingProgram` lifecycle façade | Medium | Fine | Method/hook surface grows easily | Better than active recipe | Mostly renames the active strategy |
| Accepted plan + linked prepared computations | Medium setup complexity, simple runtime | Best practical hot path | Ordinary Python behavior plus explicit contract evolution | Durable meaning separated from ephemeral execution | Requires carefully designed acceptance and relinking |
| Generated specialized Trainer | Very high | Marginal theoretical gain | Difficult to debug and maintain | Poor unless code generation is durable | Solves a method-call cost we do not have |

The fourth option is the one I would choose.

## The concise rule

```text
Before acceptance:
  the strategy may express unique choices.

After acceptance:
  those choices are frozen into the accepted plan.

After preparation:
  those choices are linked into direct executable behavior.

During training:
  the Trainer invokes that behavior but never consults the strategy.
```

Generic mechanics do not come from the strategy. Unique semantics do not move into the Trainer. They are compiled out of the authoring boundary and into the prepared run.

This also corrects the wording we currently have. The eventual step row should not say “strategy computes the authored training semantics.” It should say that the **prepared run’s accepted training computation** produces the model/objective result. The strategy has already completed its job.

Before treating this topology as settled, four parts deserve explicit design:

1. Exactly what accepted information must survive without the strategy object.
2. How prepared computations are built and rebuilt after binding changes.
3. Which per-step inputs and results are truly universal.
4. Where the standard Trainer contract ends and a different execution profile begins.

Those are genuine architecture questions. They determine whether this solves the problem or merely renames where it lives.

---

is that possible without the capabilities moving into the trainer? that's the most important question, because it decides whether or not strategy needs to be active or prepared

---

Yes—but only if preparation produces executable capability implementations, not merely prepared model bindings.

The real choice is not:

```text
active strategy
vs.
capabilities inside Trainer
```

It is:

```text
active authored strategy
vs.
active prepared capability set
```

I recommend the second.

## How it works

Take sampling as an example:

```text
strategy recipe
  selects and wires SDXL sampling behavior
  declares that it needs conditioning, denoising, and decoding
                    │
                    ▼
contract establishment
  validates that composition and its requirements
                    │
                    ▼
runtime preparation
  binds the selected behavior to the prepared
  text encoders, denoiser, and VAE routes
                    │
                    ▼
prepared sampling capability
  an executable SDXL-specific operation
```

When sampling is triggered:

```text
pipeline:
    result = prepared_sampling.execute(request)
```

The pipeline knows:

- when to sample;
- the generic sampling request;
- where outputs should go;
- how results enter logging and observation.

It does not know:

- that SDXL has two text encoders;
- how pooled conditioning works;
- which denoiser arguments are required;
- how latents are decoded;
- how the strategy composed those features.

That behavior remains in the prepared capability implementation.

The same pattern applies elsewhere:

| Concern | Active runtime behavior |
| --- | --- |
| Core batch computation | prepared training computation |
| Sampling | prepared sampling capability |
| Validation | prepared validation capability |
| Caching | prepared encoding/caching capability |
| Artifact production | prepared product capability |
| Runtime-state contribution | prepared state contributor |

The Trainer or pipeline only owns each capability’s generic coordination.

## Why this is not merely moving the strategy

A bad prepared object would do this:

```python
class PreparedSampling:
    def execute(self, request):
        return original_strategy.sample(request)
```

That changes nothing. The original strategy remains the real runtime object.

A genuine prepared capability instead contains the resolved behavior and direct prepared dependencies:

```text
PreparedSdxlSampling
  conditioning behavior
  prepared text-encoder routes
  prepared denoiser route
  scheduler behavior
  prepared VAE route
```

It no longer contains unresolved authoring choices and does not call back into the authored strategy. Preparation has specialized it for this run.

That is the test:

> If the prepared capability delegates to the strategy or asks the Trainer for model anatomy, we only relocated the problem. If it directly owns its accepted behavior and prepared dependencies, the problem has been resolved.

## What preparation must retain

If the original strategy disappears, the accepted run needs enough information to rebuild capabilities after replacement or resume:

```text
accepted capability definition
  implementation identity
  fixed settings
  semantic dependencies
  reconstruction behavior

current prepared capability
  live prepared routes
  ephemeral backend state
  executable operation
```

For example, after replacing a denoiser:

1. The binding authority accepts the replacement.
2. Dependent training, validation, and sampling executors become stale.
3. The accepted capability definitions remain valid.
4. Preparation rebuilds those capabilities against the new denoiser route.
5. The new prepared capability set is published.

The strategy object is unnecessary. The accepted definitions provide the frozen recipe needed to rebuild the runtime.

On cross-process resume, the implementation code must still be installed, but the original strategy instance does not need to exist. The snapshot identifies the accepted implementations and settings and reconstructs their prepared forms.

## The unavoidable active part

Capability behavior must be active somewhere. There are only three possibilities:

1. The strategy remains active.
2. Prepared capability implementations remain active.
3. The Trainer interprets a declarative language describing all possible capability behavior.

The third option is the large graph/DSL architecture we have deferred. If we do not want that, either the strategy or prepared capabilities must execute ordinary Python behavior.

Keeping the authored strategy active is simpler initially, but it combines:

```text
authoring recipe
accepted meaning
binding state
prepared runtime
capability execution
```

Producing prepared capabilities separates these cleanly.

## The exact ownership split

```text
Strategy recipe
  chooses and composes capability implementations

Contract system
  validates their requirements and compatibility

Preparation
  binds them to the current execution routes

Prepared capability
  performs the unique model/training behavior

Trainer/pipeline
  decides when to invoke it and coordinates generic infrastructure
```

A capability is, by definition, something the pipeline recognizes how to request. Therefore some consumer-side coordination must exist in the Trainer or pipeline. But knowing how to request sampling is not the same as implementing SDXL sampling.

So the answer is:

> Capabilities do not have to move into the Trainer. The authored strategy can become inactive, provided it produces accepted definitions that preparation turns into self-sufficient, executable, run-bound capabilities.

That means the architecture should have an active **prepared training runtime**, whether or not that is ultimately called a prepared strategy, prepared run, or something else. The authored recipe itself does not need to remain active.