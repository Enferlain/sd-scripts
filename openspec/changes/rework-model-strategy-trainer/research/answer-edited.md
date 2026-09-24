## Working hypothesis

The smallest common denominator is **not** a universal `training_step()`. A current candidate substrate includes:

* **values** — ephemeral representations/results;
* **persistent state** — model, optimizer, scheduler, stage/phase, data/experience state;
* **operations** — computations with declared inputs/outputs and state reads;
* **effects** — authorized mutations of persistent state or externally visible actions;
* **dependencies** — ordering/data relationships the engine must preserve;
* **regions / activities** — lifecycle scopes with explicit completion and commit/restart boundaries.

The selected pressure cases already require more than ordinary nested control flow: coherent **transition semantics** are necessary for mid-run replacement of authoritative state, and asynchronous variants require **independently progressing activities plus explicit communication/synchronization semantics**. Which of these are primitive language constructs versus derivable compositions remains open.

The engine does not need to understand diffusion, GANs, distillation, or RL as named algorithms. It needs enough visibility to preserve ordering, dependency, ownership, mutation, coordination, and restart correctness.

## 1. Ordinary diffusion

A conventional diffusion update is the easy case.

Conceptually:

```text
obtain batch
    ↓
derive/corrupt representation
    ↓
execute model
    ↓
construct objective
    ↓
backward
    ↓
optimizer update
    ↓
EMA/scheduler/progress updates
    ↓
commit advance
```

What the engine must see:

| Concern          | Required visibility                                                              |
| ---------------- | -------------------------------------------------------------------------------- |
| Ordering         | objective must exist before backward; backward before optimizer update           |
| Persistent state | model parameters, optimizer, scheduler/EMA, progress/data state, RNG as required |
| Dependencies     | model output depends on prepared batch/corruption state                          |
| Effects          | parameter update, optimizer mutation, EMA/scheduler/progress mutation            |
| Ownership        | only the accepted optimizer/update domain may mutate its parameters              |
| Restart boundary | normally the completed optimizer/update unit                                     |

The actual diffusion mathematics can remain opaque authored computation:

```text
sample t
construct x_t
predict epsilon/v/x0/flow
calculate loss
```

Nothing there requires a diffusion-specific Trainer mechanism.

So ordinary diffusion can be expressed as something roughly like:

```text
region optimize:
    compute(...)
    objective(...)
    backward(...)
    update(...)
    advance(...)
```

This is a semantic description of the case, not necessarily a claim that `backward`, `update`, and `advance` must each be separate engine primitives. Gradient accumulation, fused execution, pipeline schedules, or backend-specific realization may lower the same accepted meaning differently.

## 2. Alternating adversarial training

A GAN/VAE-GAN immediately demonstrates why one fixed optimization unit is insufficient.

```text
repeat:
    discriminator phase:
        compute D objective
        backward
        update D

    generator phase:
        compute G objective
        backward through D
        update G
```

What changes from the engine's perspective is **not** the meaning of GAN loss. It is the existence of two coordinated update domains and a persistent phase schedule.

| Concern          | Required visibility                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Ordering         | D and G phases occur in a prescribed order/frequency                                     |
| Persistent state | generator, discriminator, optimizer G, optimizer D, shared progress; phase/progress cursor only if required by the accepted commit/restart semantics |
| Dependencies     | G loss may differentiate through D while D parameters are not update-owned in that phase |
| Effects          | separate updates to G and D state                                                        |
| Ownership        | update ownership changes by phase; gradient connectivity need not                        |
| Restart boundary | either each phase or the whole D+G cycle, depending on the accepted semantics            |

That last point is important. The language should **not assume** whether:

```text
D update = independently committed advance
```

or:

```text
D update + G update = one atomic logical advance
```

The accepted training meaning has to establish that.

The generic new mechanism here is therefore something like **coordinated optimization phases**, not `GANTrainer`.

The discriminator loss remains extensible computation.

## 3. Progressive distillation with a mid-run teacher/optimizer change

This is the first case where a **plain dataflow or sequential-region model** is not enough unless it can represent coherent replacement of authoritative state and the corresponding commit/restart boundary.

Suppose the run is:

```text
stage A:
    teacher T0
    student S
    optimizer O0
    distillation updates

        ↓ milestone

transition:
    derive/replace teacher
    replace/reset/reconfigure optimizer

        ↓

stage B:
    teacher T1
    student S
    optimizer O1
    continue distillation
```

The engine must see the stage transition because it changes authoritative persistent state.

| Concern          | Required visibility                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| Ordering         | stage A must complete before transition; stage B cannot execute until transition commits                      |
| Persistent state | student, current teacher, optimizer, scheduler, stage cursor, possibly EMA/history                            |
| Dependencies     | teacher outputs feed student objective; teacher may be detached/frozen                                        |
| Effects          | student update during stages; teacher replacement and optimizer replacement/reset during transition           |
| Ownership        | teacher may execute without being update-owned; transition authority may replace which state is authoritative |
| Restart boundary | critically, before or after the transition commit—not halfway through it                                      |

The distillation computation itself can remain opaque:

```text
teacher(input)
student(input)
compare(...)
```

But this:

```text
replace current teacher
invalidate old optimizer
derive/install new optimizer
advance stage
```

is not ordinary computation.

It needs a **state-transition mechanism** with coherent publication.

This is closely analogous to the staged changes already visible in modern LLM training: Qwen3.8 replaces full attention with QSA during continued pretraining rather than treating the entire run as one static execution topology. 

I would therefore distinguish:

```text
computation region
    produces values/objectives

optimization region
    mutates existing optimization state

transition region
    replaces/reinterprets authoritative run state
```

A transition should probably behave transactionally:

```text
derive target state
validate
prepare dependent optimizer/backend state
──────── COMMIT ────────
new stage becomes authoritative
```

If the process crashes before commit, restart from the old stage. If after commit, resume the new one.

That is a genuine engine mechanism.

## 4. Rollout-based RL

Synchronous rollout RL can still fit the same general structure, but “obtain work” becomes computation inside the run:

```text
prompt
    ↓
generate rollout group
    ↓
environment / verifier / reward
    ↓
derive training experience
    ↓
advantages / targets
    ↓
policy objective
    ↓
update policy
```

What the engine needs:

| Concern          | Required visibility                                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| Ordering         | generation precedes scoring; scoring/grouping precedes advantage construction; experience precedes update            |
| Persistent state | policy, optimizer, policy version, rollout/data cursor; possibly critic/reference/reward state                       |
| Dependencies     | experience is tied to the policy state that generated it; grouped rewards may depend on sibling trajectories         |
| Effects          | policy update; possibly critic update; environment/external execution                                                |
| Ownership        | rollout realization may read the policy but not own policy mutation                                                  |
| Restart boundary | generation may be disposable/replayable or may need durable experience state; optimizer commit remains authoritative |

The important new thing is **provenance**:

```text
trajectory R
    generated by policy version P17
```

is semantically different from merely having response tokens.

The MiMo case in the research already shows how far this can go: one update can involve roughly 25K trajectories generated through harnesses/environments and then groupwise reward processing, with rollout generation and optimization running asynchronously.  Its asynchronous rollout/optimization setup also means the run may need more restoration state than weights and optimizer alone. 

For **synchronous** rollout RL, I don't think this requires an `RLTrainer`. A structured region works:

```text
region experience:
    generate(...)
    score(...)
    group(...)
    derive_targets(...)

region optimize:
    compute_objective(...)
    backward(...)
    update(...)
```

But **asynchronous** RL exposes where the minimal sequential/nested-region model fails. If asynchronous rollout training is a supported case, independently progressing activities and their communication semantics are part of the language problem now, not merely a future optimization.

Now you have something more like:

```text
rollout producers ──→ versioned experience queue ──→ optimizer consumer
       ↑                                           │
       └──────────── policy publication ───────────┘
```

That requires engine-visible concepts such as:

* concurrency;
* queues/buffers;
* policy-version provenance;
* staleness/admissibility;
* synchronization/publication;
* backpressure;
* recovery of in-flight or durable work.

You should not fake that as one opaque `GenerateRLExperience()` operation if the engine is supposed to coordinate its lifetime and recovery.

That is a genuinely new Trainer mechanism.

---

# Smallest common operation interface

I think an operation needs surprisingly little.

Conceptually:

```text
Operation

inputs:
    ValueRef*

outputs:
    ValueSpec*

state dependencies:
    StateRef*

effects:
    Effect*

requirements:
    Capability*

engine-visible execution properties:
    only those gradient / detach / replay / external-effect facts
    that cross an engine-owned boundary
```

Importantly, **writes should preferably appear as effects**, not arbitrary hidden mutation. The operation does not need to expose its internal autograd or Python implementation details unless the Trainer must coordinate them. Pure computation can stay opaque; replay/failure metadata matters mainly where external or durable effects make it relevant.

An ordinary authored operation could therefore be completely unknown to core:

```text
MyFutureMethodFoo
    inputs: A, B
    outputs: C
    reads: model X
    effects: none
```

The Trainer doesn't need to understand `Foo`.

It only needs to know how it participates in the run.

The operation implementation can contain arbitrarily sophisticated Python/PyTorch as long as no engine-relevant boundary is being hidden inside it.

---

# Smallest common region interface

I'd make regions more semantically important than nodes.

A region / activity needs something like:

```text
RegionOrActivity

inputs / outputs

contained operations or subregions

dependencies

persistent state dependencies

allowed effects

completion condition

commit / restart semantics
```

The key addition is the lifecycle boundary. A region isn't merely:

> execute these nodes.

It says:

> **this is a coherent unit of work whose effects have a defined publication/recovery boundary.**

That gives you ordinary optimization scopes, GAN phases, synchronous RL experience production, and stage-transition scopes without giving the engine any of those algorithm names.

The selected cases additionally require some representation of:

```text
ordered / repeated / conditional control
coherent transition of authoritative state
independent progress + communication when asynchronous execution is supported
```

I would **not yet assume** these must be literal primitives named `Sequence`, `Repeat`, `Conditional`, `Transition`, or `Concurrent`. That is the language-design question still under test. Arbitrary graph cycles should not be introduced merely to gain expressiveness; structured lifecycle/coordination semantics are easier to validate and restart correctly. Data dependencies inside a region can still be DAG-like where that is sufficient.

---

# Persistent state probably needs one common contract too

Something like:

```text
State

identity
authority / mutation ownership
persistence / recovery semantics

optional where semantically relevant:
    version / provenance
    dependency validity
    serialization details
```

Examples:

```text
student.parameters
optimizer_student.state
curriculum.stage
policy.version
rollout.queue
adaptive_noise_schedule.state
```

A `gan.phase_cursor` is authoritative state only if the accepted commit/restart semantics require preserving partial-cycle position; otherwise it may remain transient execution bookkeeping. Likewise, not every state object needs an exposed version merely because asynchronous policy provenance does.

The core does not need a giant enum identifying those meanings. It does need to know **who may mutate them and when a mutation becomes authoritative**.

---

# Authoritative effects should be the tightly controlled part of the language

This is where I'd try to avoid predicting the future.

Let domain computation be open-ended, but keep authoritative mutation/lifecycle effects small and explicitly governed. Possible categories include parameter/optimizer update, auxiliary-state advancement, durable publication, and coherent transition/install effects, but I would **not freeze an effect taxonomy yet**.

In particular, a catch-all such as `ExternalEffect` risks becoming an escape hatch that means anything, and separate `UpdateParameters` / `UpdateOptimizerState` effects may or may not be the right semantic split for one coherent optimizer update.

The important architecture is:

```text
OPEN:
    domain computation

TIGHTLY CONTROLLED / deliberately extensible:
    authoritative mutation, lifecycle, publication, and recovery effects
```

An extension may introduce new computation freely within its declared capabilities; it may not introduce a new form of authoritative mutation or lifecycle invisibly. That's how the engine can remain extensible without letting an extension smuggle in another Trainer.

---

# Extensible computation vs genuinely new mechanism

I'd use this test.

An addition is **extensible computation** if it can be encapsulated as:

> consume known values/state, produce values, and participate using already-understood effects/control/restart semantics.

Examples:

```text
new diffusion corruption formula
new distillation objective
new reward function
new teacher projection
new conditioning adapter computation
new verifier
```

The Trainer core should not change.

An addition is a **new Trainer mechanism** if supporting it requires a new answer to any of:

> How is authoritative state mutated?

> How are operations scheduled/coordinated?

> What persistent state must the engine track?

> What constitutes commit/restart?

> What concurrency/synchronization semantics exist?

Examples:

```text
first support for multiple coordinated optimizer phases

first transactional topology/stage replacement

first asynchronous producer/consumer experience stream

first durable in-flight external interaction that must survive restart
```

That gives a pretty clean criterion:

> **New mathematics is usually an extension. New authority, lifecycle, coordination, or recovery semantics indicate a new engine mechanism.**

That's probably the most important distinction I'd give the agent.

---

# What should strategy authors actually specify?

The original comparison suggests at least two defensible contract-first models, and I would **not choose between them prematurely**.

### A. Explicit semantic mechanism selection

The contract advertises supported semantic mechanisms/compositions, and the strategy selects them declaratively:

```text
contract supports:
    alternating update domains
    generated-experience production
    coherent state transition

strategy selects:
    the semantics this run actually requires
```

This does **not** require the strategy to instantiate engine objects or configure Trainer internals. It only means some semantics are important enough that the author must choose them explicitly rather than letting fulfillment infer them.

This is appropriate when several valid mechanisms would satisfy the same broad description but change training meaning, for example:

```text
synchronous vs asynchronous rollout production
commit every adversarial phase vs commit the whole cycle
```

### B. Requirement-driven resolution

The strategy instead states semantic relationships and constraints:

```text
two update domains: generator, discriminator

ordering:
    discriminator before generator

generator objective:
    depends differentiably on discriminator execution

update ownership:
    D objective → discriminator
    G objective → generator
```

Fulfillment may derive a realization **only where the contract makes that derivation semantically unambiguous**. It may resolve implementation freedom such as fused vs unfused execution, backend realization, placement, or sharding. It may not choose between semantically different training behaviors merely because both are executable.

So the stronger rule is:

> **The contract must distinguish what the author must select from what fulfillment is permitted to derive. Fulfillment may resolve implementation freedom; it must not invent semantic freedom.**

A hybrid is likely: some ordering/dependency facts can be derived safely, while mechanism choices that change lifecycle, provenance, concurrency, or commit semantics may require explicit author selection.

The strategy still should not assemble private Trainer objects, and fulfillment still should not guess a missing training meaning. If realization requires a semantic choice that the accepted contract did not settle, fulfillment should reject the run as underspecified or unsupported.

---

# Where the proposed interface is under pressure

Three different situations should be distinguished.

**Required extension to the candidate kernel: coherent transition semantics.**

The progressive-distillation case already requires transitions that can replace/reinterpret authoritative state and coordinate invalidation/repreparation of optimizer groups, distributed wrappers, caches, routes, compiled graphs, or other derived state. This cannot be reduced to an ordinary state-writing operation such as:

```text
state.model = new_model
```

The coherent preparation/install machinery is part of the required engine semantics for that supported case.

**Boundary beyond sequential/nested regions: independently progressing activities and communication.**

Asynchronous rollout producers and an optimizer consumer with versioned queues require more than nested `sequence/repeat/conditional` execution. If asynchronous training is supported, the engine needs explicit semantics for independent progress, communication/buffering, publication, provenance/staleness, synchronization/backpressure, and recovery of in-flight or durable work.

**Invalid encapsulation: opaque operations hiding engine-owned boundaries.**

For example this looks convenient:

```text
ProgressiveDistillationOperation.run()
```

but if inside it:

```text
optimizer.step()
replace teacher
reset optimizer
checkpoint
```

then it's a second Trainer disguised as an operation.

The rule should be:

> **An operation may remain opaque until something inside it crosses an engine-owned boundary. At that point the boundary must be represented explicitly.**

That's probably the best answer to graph granularity too.

---

# Current candidate concepts

I would not call the semantic kernel settled yet. The pressure cases currently support these candidate concepts:

```text
Value / Representation
Persistent State
Operation
Effect
Dependency
Lifecycle Region / Activity
Commit / Restart Boundary
```

The selected cases already require **coherent transition semantics**. If asynchronous rollout training is supported, they additionally require **independently progressing activities plus explicit communication/synchronization semantics**. It remains open which of these should be core primitives and which can be derived from a smaller set without losing validation or restart correctness.

The encouraging part is still that the four cases do not appear to demand four Trainers. They mostly demand different compositions of:

```text
computation
optimization
phase ordering
state transition
experience production
coordination
```

The simple nested-region abstraction genuinely bends at asynchronous RL because independent producers/consumers, policy publication, versioned experience, and in-flight recovery become part of the training meaning. That is useful evidence: the goal is not to force every case into one graph shape, but to identify exactly when a new engine mechanism is required.

The strategy/fulfillment boundary should therefore remain explicit:

> **The contract must say which training semantics are author-selected and which realization choices fulfillment may derive. Fulfillment may not infer a semantically different mechanism merely because it can execute it.**

If realization requires guessing a missing semantic choice, the contract is underspecified or the mechanism is unsupported.
