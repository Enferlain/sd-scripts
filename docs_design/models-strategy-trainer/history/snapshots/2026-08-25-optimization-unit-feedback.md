Yes — I think **item 1 is basically settled by this formulation**. The identity/revision distinction now survives all the actual pressure cases in the code: schedule-free wrapping, Accelerate/DeepSpeed replacement, EDM2's second optimizer, replanning, and exact resume.

There is one substantive refinement I'd make before recording it, plus a few smaller precision fixes.

### 1. Don't let “exact parameter set” mean Python parameter objects

This sentence is the only part I'd be nervous about:

> An optimization unit is one independently advanced optimization responsibility whose current revision identifies its exact parameter set and policy.

Given the preparation model you've already established, “exact parameter set” could accidentally imply concrete `nn.Parameter` identities.

Those can be affected by realization/preparation just like optimizer objects can. The semantic thing the revision should identify is the **accepted parameter membership/grouping**, qualified through the authority—not necessarily the particular runtime handles currently realizing it.

I'd make it:

> **An optimization unit is one independently advanced optimization responsibility whose current revision identifies its exact accepted parameter membership, grouping, policy, and semantic dependencies.**

Then:

```text
unit revision
    the current accepted definition of that responsibility:
    semantic parameter membership and grouping,
    optimizer/scheduler policy,
    advancement policy,
    and authority-qualified dependencies
```

The resulting runtime can separately say:

```text
unit identity + unit revision
    -> realized parameter handles
    -> concrete optimizer
    -> concrete scheduler
    -> backend/wrapper objects
    -> runtime coordination state
```

That cleanly extends the principle you've just discovered for optimizers to parameters too.

### 2. Make `unit address` explicitly non-positional and reusable

I like introducing address separately from identity. It gives you exactly the distinction that `main` and `edm2` need.

I would make one thing explicit:

```text
unit address
    stable semantic slot/key in an accepted optimization plan
    not a list index and not an incarnation identity
```

And importantly:

> **An address may be reused; an identity may not.**

For example:

```text
main -> unit U17 rev 3
retire U17

...later...

main -> unit U29 rev 1
```

Same address, different incarnation.

That makes the split/merge/retire/recreate rule completely unambiguous and closely parallels the participant-incarnation semantics you've already settled.

It also explains why the address is useful without trying to make it do identity's job.

### 3. Be precise about which dependency changes advance revision

I'd slightly narrow:

> participant/binding dependencies

because your preparation architecture has several different revisions floating around.

A backend preparation may change an execution-route revision as part of realizing exactly the same optimization definition. That **must not recursively force an optimization-unit revision**, or you've recreated the original problem one level higher.

I'd distinguish:

```text
semantic/source dependencies
    participant identities
    authoritative binding revisions where parameter resolution depends on them
    relationship/arrangement facts where relevant

runtime realization dependencies
    prepared execution routes
    wrapper/backend handles
    concrete optimizer/scheduler objects
```

A change in the former can invalidate/replan the unit and therefore produce a new unit revision.

A change in the latter can invalidate/re-realize the **runtime for the same unit revision**.

That gives you a very useful two-dimensional model:

```text
semantic unit
    U17 / revision 4

runtime realization A
    optimizer object X
    route revisions ...
    backend handles ...

runtime realization B
    optimizer object Y
    route revisions ...
    backend handles ...
```

Both can legitimately realize `U17/r4`.

That's exactly what Accelerate replacement wants.

### Your schedule-free rule is then particularly clean

This:

> Schedule-free wraps an optimizer → same unit, unchanged when merely realizing the already accepted policy

is correct, but I'd retain the qualifier because schedule-free itself **is optimization policy**.

So:

```text
accepted policy already says schedule-free
    constructing ScheduleFreeWrapper
        same identity
        same revision

replan changes normal AdamW -> schedule-free AdamW
        same identity
        new revision

different ScheduleFree configuration
        same identity
        new revision
```

Physical wrapping is realization.

Choosing schedule-free is policy.

That's a nice demonstration of the distinction.

### 4. Tiny adjustment to initial publication

I'd tweak this:

> First revision is published only after the attempt succeeds.

to:

> **The initial revision becomes current only when the complete realization/preparation attempt succeeds and is published.**

Why? Because conceptually the candidate revision already exists during resolution/preparation—you know what you're attempting to realize.

You just don't want it to become authoritative/current until final publication succeeds.

So the lifecycle becomes:

```text
resolve candidate
    address = main
    candidate identity = U17
    candidate revision = 1

attempt realization/preparation
    construct AdamW
    wrap if required
    backend prepare
    validate

success:
    publish U17/r1 as current

failure:
    no current optimization unit/runtime publication
```

Whether the internal identity `U17` gets allocated before or exactly at commit is then an implementation detail.

### 5. Optimizer mutable state definitely does not belong in revision

I strongly agree with this correction.

Otherwise you'd get the absurd result:

```text
optimizer.step()
    momentum changed
    therefore unit revision++
```

every single step.

The clean separation is:

```text
unit identity
    WHICH optimization responsibility

unit revision
    WHAT accepted definition that responsibility currently has

runtime realization
    WHICH concrete objects currently implement that definition

mutable optimization state
    WHERE execution of that definition has currently reached
```

And the checkpoint links those surfaces.

I'd only expand your checkpoint tuple slightly:

```text
unit identity
+ unit revision
+ training/checkpoint coordinate
+ unit-local advancement coordinate, where distinct
```

The last part matters once you explicitly support independently advanced units.

Suppose:

```text
main     advances every synchronized step
aux      advances every 4 synchronized steps
edm2     advances according to another accepted cadence
```

At global training step 100, their local advancement counts need not all be 100.

The optimizer/scheduler state itself obviously contains much of this, but semantically it's useful not to assume `global_step == unit step`.

### So I'd record the rule approximately as

```text
unit address
    a stable, non-positional semantic slot in the accepted
    optimization plan, such as main or edm2
    an address may be reused by a later incarnation

unit identity
    one run-scoped incarnation of an independently managed
    optimization responsibility
    identity survives revisions and runtime replacement
    identity is never reused after retirement

unit revision
    the current accepted definition of that responsibility:
    semantic parameter membership and grouping,
    optimizer/scheduler policy,
    advancement policy,
    and authority-qualified semantic dependencies

runtime realization
    the concrete parameters, optimizer, scheduler, wrappers,
    prepared objects, and backend state currently realizing
    one unit identity + revision
```

Then your transition table becomes:

| Change                                                                               | Identity                | Revision                                                        |
| ------------------------------------------------------------------------------------ | ----------------------- | --------------------------------------------------------------- |
| Initial construction/wrapping/preparation                                            | Initial identity        | Initial revision becomes current only on successful publication |
| Backend replaces concrete optimizer/scheduler                                        | Same                    | Same                                                            |
| Backend changes prepared runtime objects/routes without changing accepted definition | Same                    | Same; runtime is re-realized                                    |
| Schedule-free wrapper materializes already-declared policy                           | Same                    | Same                                                            |
| Parameter membership or optimizer-significant grouping changes                       | Same                    | New                                                             |
| Optimizer/scheduler policy changes                                                   | Same                    | New                                                             |
| Advancement policy changes                                                           | Same                    | New                                                             |
| Relevant authoritative source/binding dependency changes and requires replan         | Same                    | New                                                             |
| Split or merge units                                                                 | New identity/identities | New initial revisions                                           |
| Retire then later recreate same address                                              | New identity            | New initial revision                                            |
| Exact resume of same logical run                                                     | Restore                 | Restore                                                         |
| New run/fork from checkpoint                                                         | New identities          | New revisions, source lineage preserved                         |

And I'd replace the existing line 1364 idea—

> “Changes to its resolved parameters, policy, or concrete runtime advance an associated revision.”

—with something like:

> **Changes to a unit's accepted parameter membership/grouping, optimization policy, advancement policy, or semantic dependencies advance its revision. Recreating, wrapping, preparing, or replacing concrete runtime objects does not by itself advance that revision.**

That feels like the actual invariant.

With that refinement, I think **optimization question 1 can be marked settled rather than left as a remaining design question**. The next interesting problem really is item 2: standard non-overlap and what exactly qualifies as an explicitly accepted overlapping optimization policy.
