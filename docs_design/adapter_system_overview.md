# Adapter System Overview

Date: 2026-04-17

## Purpose

This note captures the core idea for the new adapter system in simple terms.

It is not about preserving the old adapter runtime.

It is also not about making vendored adapter code the long-term runtime home
for absorbed methods.

It is about defining a clean system where:

- optimization still owns targeting and grouping policy
- the adapter system owns turning those decisions into trainable adapter state
- training/mode owns orchestration

## Core Idea

An adapter is a trainable augmentation that is attached to parts of the
original model, but is not itself part of the base model's canonical weights.

The new adapter system exists to let:

- adapter algorithms define what they do
- optimization decide what original-model parts are affected
- training run the whole thing without owning adapter internals

For absorbed methods, the intent is full repo-owned implementations.

Vendored code may still be useful as:

- reference material
- migration source material
- a temporary waypoint during implementation

But it is not the intended steady-state home for absorbed adapter method
behavior.

## High-Level Ownership

### Strategy

Strategy exposes what targetable structure exists on a specific model family.

Examples:

- which components exist
- which modules or parameters can be targeted
- stable names for those targets

Strategy does not decide training policy.

### Optimization

Optimization owns policy.

It decides:

- what original-model targets are in scope
- what effect surface the adapter should be attached to
- how returned trainables are grouped and scheduled

Optimization does not need to know adapter math.

### Adapter System

The adapter system sits in the middle.

It takes resolved original-model targets and uses them to instantiate:

- a concrete adapter type
- a concrete attached runtime instance

It then exposes whatever runtime information optimization and `AdapterMode` need
in order to keep the training flow moving.

The important point is not that every adapter type must return the exact same
tiny payload.

The important point is that every adapter type should fit into the same
overall framework and handoff shape.

### Mode / Training

Mode and training orchestrate the run.

They do not define adapter math or optimizer policy.

They coordinate the flow between strategy, optimization, and the adapter
system.

## Main Flow

The intended flow is:

```text
strategy
  -> exposes targetable model structure

optimization
  -> resolves which original-model targets are in scope

adapter system
  -> instantiates and attaches adapter runtime against those targets

optimization
  -> groups and schedules the returned trainable adapter pieces

training
  -> runs the training lifecycle
```

In short:

```text
find targets
-> choose targets
-> build adapter
-> group trainables
-> train
```

## From-Weights And Merge Flow

The same ownership split should continue to hold when adapter weights are
loaded for rank discovery, merge, or inference-style setup.

That means:

- optimization still owns resolved original-model targets
- mode still decides which flow is happening
- adapter runtime still owns method behavior

In the current repo-owned shape, loading from weights produces a loaded runtime
object that keeps any method-specific loaded state with the runtime that owns
it.

If merge is supported, mode should pass a repo-owned merge request into that
loaded runtime instead of calling built-in merge signatures directly.

Conceptually:

```text
optimization
  -> resolves targets

mode
  -> chooses from-weights / merge flow

adapter runtime
  -> builds loaded runtime from weights
  -> merges through a repo-owned merge request if supported
```

Legacy built-in wrappers can still adapt that repo-owned merge request to older
`merge_to(...)` signatures internally during migration, but that is adapter-side
translation, not the repo-owned merge contract.

## Important Rule

Target selection happens before adapter instantiation.

Optimizer grouping happens after adapter instantiation.

This means optimizer groups should not be the way the adapter learns what it
affects.

Instead:

- optimization first decides what base-model targets are affected
- the adapter system builds against those targets
- optimization then groups the trainables that come back

That round trip is intentional.

## Shared Target Refs

The current repo shape now uses a shared optimization-owned target-ref model
for targetable components, modules, and parameters.

That means:

- optimization can describe selected structure with one vocabulary
- fine-tune parameter selection can carry parameter-level provenance
- adapter module targeting can carry module-level provenance
- adapter-facing compatibility layers can keep older fields while still
  wrapping the shared target-ref model internally

The public selector surface stays the same:

- module targets use component-qualified selectors like `unet.to_q`
- parameter targets use component-qualified selectors like
  `unet.to_q.weight`

The shared target refs also carry:

- module type for module targets
- owner-module path/type for parameter targets

That gives future module-type selectors and richer adapter grouping work a
shared foundation without making the adapter runtime package own the long-term
target vocabulary.

## The Round Trip

The adapter system is part of an optimization-owned flow, but it owns the
realization step in the middle.

```text
optimization
  -> "these original-model targets are in scope"

adapter system
  -> "here are the trainable adapter parts created for those targets"

optimization
  -> "now I can build parameter groups and scheduling around them"
```

This keeps the split clean:

- optimization owns policy
- adapter owns implementation
- training owns orchestration

## What The Adapter System Must Know

The adapter system must know enough to build a concrete attached augmentation
for a specific model context.

That includes:

- which resolved targets it is attaching to
- which adapter type is being used
- any algorithm-specific settings

The adapter system should not own the full targeting DSL or optimizer policy.

For absorbed methods, the adapter system should also not stop at a thin
repo-owned wrapper around vendor algorithm classes and call that "done."

If a method is considered truly absorbed, the method behavior itself should be
owned in repo code.

## What The Algorithm Must Know

An algorithm mainly owns what it does.

That includes:

- how its trainable state is defined
- what target kinds it supports
- what structural constraints it has
- what useful metadata it can expose about its trainables

It does not need to own:

- model-family discovery
- training policy
- optimizer grouping

And for absorbed methods, that algorithm ownership should ultimately live in
repo code rather than staying as a permanent dependency on vendored runtime
classes.

## What Should Come Back From The Adapter System

The adapter system should participate in a common framework, not return
arbitrary loose objects.

At a high level, the result should include:

- the concrete attached adapter instance
- the trainable or runtime information needed for optimization and `AdapterMode`
- enough provenance to tie returned adapter state back to original-model
  targets

That provenance matters because most adapters are not meaningful as standalone
weights in a vacuum. They exist relative to the base-model targets they affect.

## Why The Provenance Matters

If optimization chooses attention-only targets, that decision must still be
visible after the adapter is instantiated.

Optimization should be able to reason about returned trainables in terms of:

- which original-model target they affect
- which component they belong to
- which algorithm produced them
- what role they play inside that algorithm

Without that, optimization loses the information it needs to group and
schedule correctly.

The exact amount of returned detail can still depend on adapter and optimizer
needs. The requirement is not "smallest possible payload." The requirement is
"predictable enough handoff for the shared framework to work."

## Design Summary

The new adapter architecture should follow this idea:

```text
strategy defines what can be targeted
optimization decides what will be affected
adapter system realizes that as trainable augmentation state
optimization groups the returned trainables
training runs the lifecycle
```

That is the core model.

## What Feels Settled

These points seem stable enough to treat as the current working direction.

- strategy exposes targetable model structure
- optimization owns targeting policy
- optimization decides what original-model parts are affected
- the adapter system realizes that decision as trainable augmentation state
- optimization groups and schedules the returned trainables
- training/mode orchestrates the whole flow
- target selection happens before adapter instantiation
- optimizer grouping happens after adapter instantiation

## What Is Still Open

These points still need more design work before they should be treated as
fully locked.

- the exact shape of resolved targets
- the exact shared framework and handoff shape used by the adapter system
- how adapter types should declare their supported target kinds and runtime
  capabilities
- how much target resolution belongs to strategy versus optimization
- the exact runtime contract for broader save/load/diagnostics capabilities
- the future config shape once the runtime boundary is explicit

## Minimal Round-Trip Shape

Without locking exact types yet, the round trip should stay conceptually small
and clear.

Optimization sends in:

- resolved original-model targets
- any already-resolved targeting options
- the adapter specification needed to instantiate the adapter runtime

The adapter system sends back:

- the concrete attached adapter instance
- the trainable or runtime information needed for grouping and orchestration
- enough provenance to tie returned adapter state back to the original-model
  targets
- any capability metadata needed by `AdapterMode` or optimization

This keeps the framework coherent without forcing every adapter type into the
same tiny internal or returned shape.

## Example Flow

Example: train a LoRA-style adapter against attention-only targets.

1. Strategy exposes the model structure and the targetable attention modules.
2. Optimization decides that only those attention targets are in scope.
3. The adapter system instantiates the chosen adapter type against those
   resolved targets.
4. The adapter system returns the created trainable adapter pieces, tied back
   to the attention targets they affect.
5. Optimization groups those returned trainables for scheduling and optimizer
   construction.
6. Training/mode runs the lifecycle.

The important point is that the adapter does not infer its targets from final
optimizer groups. It is built from already-resolved targets, and grouping
happens afterward.

## Non-Goals

This note does not try to define:

- exact config schemas
- exact runtime class names
- exact save/load API
- exact support matrix for every future adapter algorithm

Those should be decided after the main ownership and round-trip shape is
accepted.
