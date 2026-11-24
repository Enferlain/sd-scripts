Yeah, this expansion is better. The two distinctions you added are important enough that I’d change the shape of Q3 around them rather than treat them as adapter-specific footnotes.

The particularly good correction is:

```text
participant existence
≠
participant materialization
≠
effect attachment
```

That cleans up something my earlier answer was still muddy about.

For a normal authored LoRA strategy, this:

```text
validated arrangement
    ↓
construct LoRA
    ↓
attach LoRA
```

should not mean that `adapter.style` suddenly enters the semantic arrangement halfway through execution. It was already there; runtime work merely realizes it and establishes its attachment.

And yes, the external-VAE example should obey the same rule. If the strategy authored an external VAE from the beginning, the fact that some loader temporarily encountered an integrated VAE is implementation noise, not a contract-level replacement.

## I think the next pressure test is exactly identity granularity

Your proposed criterion:

> one independently managed state trajectory / training subject / persistence unit

is very close, but I wouldn't make any one of those properties individually decisive.

For example, suppose one VeRA runtime owns:

```text
shared projection bank
target A coefficients
target B coefficients
target C coefficients
```

Maybe optimization allows separate parameter groups:

```text
group 1 → target A coefficients
group 2 → target B coefficients
```

That alone shouldn't necessarily turn them into:

```text
adapter.target_a
adapter.target_b
```

They're still parts of one adaptation mechanism if their lifecycle and artifact semantics are inseparable.

Conversely, this could happen:

```text
one Python AdapterRuntime object
    contains:
      adapter.style
      adapter.character
```

If those can be independently:

* enabled/disabled;
* trained;
* loaded;
* saved;
* attached/detached;
* merged;
* replaced;

then the fact that one convenient runtime object stores them should **not** force one logical identity.

So I'd make the governing rule semantic:

> A logical adapter identity represents one independently addressable adaptation participant whose state and relationships are managed as one unit under the strategy contract.

Then use the other properties as evidence.

### Strong evidence for separate identities

If A and B can independently have different answers to questions like:

```text
Is it present?
Where did it come from?
Is it trainable?
Is it active?
What does it target?
Can it be detached?
Can it be saved/loaded independently?
Can it be merged independently?
Can it be retired independently?
```

then they're probably distinct logical participants.

Something like:

```text
adapter.style
adapter.character
```

even if both happen to live inside one runtime container.

### Strong evidence for one identity

If the subparts only make sense collectively:

```text
shared VeRA basis
+ target-local coefficients
+ bookkeeping required to apply them
```

and cannot meaningfully be persisted, activated, attached, or evolved as independent participants, they're implementation structure underneath:

```text
adapter.vera
```

not separate logical components.

That's very similar to why we don't make every layer of the denoiser a logical component merely because parameter scopes can address them.

---

There is another distinction I think your expanded Q3 exposes:

## Adapter identity and attachment identity are different

Consider one adapter attached twice:

```text
adapter.shared
    ├── affects → denoiser.block_A
    └── affects → denoiser.block_B
```

The adapter has one state trajectory.

But there are two attachment relationships.

Or imagine the same prefix state contributes to two conditioning paths.

We shouldn't need two adapter identities merely because there are two edges.

So conceptually:

```text
participant identity
    adapter.style

attachment/effect relationship
    adapter.style → denoiser.foo
    adapter.style → denoiser.bar
```

The relationship itself may carry substantial semantics:

```text
target
scope
effect kind
activation condition
strength/scaling
runtime requirements
```

That suggests the future bound-state model must treat relationships as first-class facts, even if we don't give every edge a public `AttachmentId`.

This becomes especially important with dynamic enable/disable:

```text
adapter.style exists
adapter.style remains bound
adapter.style remains trainable

but

attachment active = false
```

That's **not** participant retirement.

It's just a relationship/effect-state transition.

And that is a much cleaner model than having `apply_to()` implicitly determine whether the adapter “exists.”

---

## Side-network adapters pressure-test this nicely

ControlNet-like topology might be:

```text
control.depth
    state binding → ControlNet module
    execution     → prepared ControlNet

control.depth --conditions/feeds--> denoiser
```

Here the component itself is execution-capable.

LoRA-like topology:

```text
adapter.style
    state binding → LoRA runtime/state
    execution     → none

adapter.style --augments--> denoiser substructures
```

Prefix-like:

```text
prefix.task
    state binding → learned prefix tensors
    execution     → none

prefix.task --contributes_to--> conditioning
```

That strongly supports your correction to Q2.

We should stop thinking:

```text
logical component → execution binding
```

and instead think:

```text
logical component
    → authoritative bound state

execution-capable component
    → authoritative execution route(s)
```

Execution is a property/capability of some participants, not the defining property of componenthood.

That's a significant improvement.

---

## Wrapper adapters expose one extra wrinkle

Suppose:

```text
adapter.wrapper
    owns state

attachment:
    adapter.wrapper wraps denoiser execution
```

and preparation results in:

```text
denoiser.training
    old → DDP(denoiser)

then attachment/preparation composition yields

denoiser.training
    → SomeAdapterWrapper(DDP(denoiser))
```

I would **not** say the adapter suddenly owns the denoiser's execution binding.

The authoritative route still belongs to:

```text
denoiser
```

because that's the semantic participant being called.

The adapter's relationship modifies construction/composition of that route.

So we'd have something like:

```text
adapter.wrapper
    state binding → adapter state

adapter.wrapper --wraps execution of--> denoiser

denoiser
    training execution → composed prepared callable
```

This distinction prevents a nasty conceptual problem where execution ownership jumps from host to adapter merely because the adapter happens to implement the outermost Python wrapper.

---

## Merge is where identity gets really interesting

I agree merge deserves its own transformation.

Before:

```text
denoiser
adapter.style

adapter.style --augments--> denoiser
```

After destructive merge:

```text
denoiser
    state changed

adapter.style
    no longer active
```

But I'd be careful with this statement:

> the adapter may be retired as an active participant

because "retired" shouldn't imply that its semantic history disappears.

We probably want to distinguish at least:

```text
active arrangement participant
historical provenance participant
```

After merging, there may be no current bound adapter state, but persistence/provenance should still be able to state:

```text
denoiser realization Y
    derived_from:
      denoiser realization X
      adapter.style artifact Z

transformation:
    merge
```

That reinforces a distinction we already needed elsewhere:

```text
strategy-scoped logical identity
≠
durable model/artifact realization identity
```

The strategy participant can remain called `denoiser` while its concrete realization changes through merge.

---

## One thing I'd change in the five/six operations

Your list is good:

```text
1. materialize/bind
2. execution rebind
3. slot replacement
4. attach/detach effect
5. arrangement amendment
6. merge/fold
```

I'd make #4 explicitly **relationship transition**, with attachment as one important case.

Because later we may discover relationships like:

```text
initialized_from
shares_state_with
conditions
augments
wraps
feeds
teacher_of
depends_on
```

Not all of them are "attachment."

But for Q3 we don't need a universal graph vocabulary. The semantic operation is simply:

> change a declared relationship/effect without changing participant identity.

That keeps us from prematurely building the strategy graph we explicitly decided not to build.

---

So after this pressure test, I'd sharpen the adapter identity rule to:

> **A logical adapter identity corresponds to one independently addressable adaptation participant, not to a Python runtime object and not to every injected submodule. Internal state may span many tensors, target-local modules, or shared banks when those pieces form one lifecycle/persistence unit. Conversely, one runtime container may represent multiple logical adapters when those adaptations are independently selectable, trainable, attachable, persistable, mergeable, or retireable.**

And then:

```text
adapter participant
    owns adaptation state and semantic lifecycle

attachment/effect relationship
    says how that participant changes other participants

injected module
    implementation detail unless independently meaningful

execution route
    exists only if the participant is independently executable
```

With that, I think **Q3 is basically settled semantically**. The remaining naming/API questions can wait until we work through persistence, because persistence is one of the best tests of whether we've chosen the right identity granularity.
