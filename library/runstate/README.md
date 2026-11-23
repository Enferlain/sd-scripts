# `library/runstate`: the run state

The owner pattern every owner of run state shares, generations and their
publication, and the run authority
([design](../../openspec/changes/rework-model-strategy-trainer/design.md)
D6–D8, D19, D21). It imports only the IR system, observability fact types
and base: never the libraries, training or Torch. Its types hold live
objects without importing Torch.

Section 15 builds it in two passes. The first, built so far, is what an
initial realization and one training step need. The second, after section
16's first pass, adds the remaining transitions, relationships, withdrawal,
views and each owner's part of a cut.

## Modules

| Module | Holds |
| --- | --- |
| `owners.py` | The owner pattern: `Reference` and its `Mint`, `Version`, `Dependencies`, `Evidence`, `Attempt` with its `Basis` and `Mutation`, and `Refused` |
| `generation.py` | `Generation`, `Generations` (owner registration, attempts, the current generation), `Publication`, and `ExecutableRun`: the prepared run, its owners and device adapters, and the readiness it was accepted with, run once |
| `targets.py` | `TargetRef`: whole participants (`@unet`), component-qualified and participant-qualified selectors |
| `authority/` | `RunAuthority`, its records (`Participant`, `Binding`, `Route`, `View`), its section, transitions with the evidence and evaluators that judge them, and `MaterializationAttempt` |
| `preparation.py` | `PreparationAttempt`, its `PreparationRequest` and `PreparationResult`: one coherent job, published together or not at all |
| `projections.py` | What consumers receive instead of the authority: `Execution`, the owner seam for the coordinator, and `Preparation`, the requested participants pinned to one generation |

## Realization

```text
  accepted run
     |  RunAuthority.seed          mint the run identity; publish every
     |                             declared participant, unbound (generation 1)
     |  MaterializationAttempt     run the declared materializers as one job,
     |                             with the settings the strategy gives them;
     |                             the authority judges each binding's evidence
     |  PreparationAttempt         prepare routes off to the side; publish them
     |                             with every contributor's section, together
     |  readiness(CORE_EXECUTION)  every participant bound and every use's
     |                             route prepared from its current binding;
     |                             each other owner judges its own resources
  ExecutableRun                    the coordinator over Execution(authority),
                                   the owner for participants, and the other
                                   owners; it runs once, then each owner that
                                   checks late settles (`Settles`), and a
                                   reason fails the run
```

## Publication

A generation holds one section per owner. An owner builds its next section
off to the side; `Generations.publish` checks the publication's basis (any
publication since makes it stale), lets every changed owner refuse its next
section, and installs the next generation as one store. Reading the current
generation takes no lock, publications are serialized with each other, and
published sections never change, so correctness does not depend on the GIL.
A publication of in-place work is checked by every owner with a section,
since the work may affect any of them.

## Preparation

`PreparationAttempt.derive` rejects a request that names an unbound
participant, or asks for nothing to publish, before any work. The job reads
the participants it requested through the `Preparation` projection, pinned
to the generation it was derived from, and returns a `PreparationResult`
whose parts stay apart by owner: routes for the authority, each
contributor's next section (every contributor the request names must
supply one), or failures. `publish` installs them in one
publication only while that generation is current; a stale, failed or
refused result publishes nothing, and the routes it would have replaced stay
usable. Publishing does not establish readiness: the host checks the
checkpoint once every attempt it needs has published.

In-place preparation changes the participants it requires in place, such as
setting trainability or turning on gradient checkpointing. It is offered
only where it has nothing to withdraw: before a participant's first route or
view, and while no lease on it is out (D19). The authority registers the
attempt's participants when it is derived, and at publication refuses it if
any of them was leased meanwhile. In-place work that first withdraws routes,
views and leases comes with the second pass.

A prepared route may carry what it establishes for each call through it,
such as training's module mode. A lease through that route establishes it
on the worker before the call, with the call's context and the handle the
call receives, and the call uses the participant alone (D15).

## The run authority

Producers never write records. They propose transitions with evidence from
their attempt; the authority judges every affected obligation (a
participant's `bound` and the `lease` obligations through a changed route)
and publishes its whole next section or nothing. This pass realizes
declaration, materialization and route rebinding; a run that declares
relationships is refused until they come.

At the owner seam, a lease goes through a route, the normal one unless
another is named, taken from the current generation. A version's
incarnation is the participant reference's serial number, its lineage the
binding revision and its ordinal the numeric version. The numeric version
advances only when the runtime tells an update lease that a call updated
through it. While a lease is out, the authority refuses to publish a change
to that participant.

Limits of this pass, each failing with its reason:

- A snapshot hands over the live route while nothing has updated the
  participant, and then pins it: the run may not update it afterwards. A
  strategy that both snapshots and updates a participant is refused at
  preparation and readiness, since the obligations show both uses, even
  where the two uses would never run in one execution: such a snapshot
  needs a copy of the participant's device storage, which the model
  definitions supply in section 18.
- Grants come with the second pass of section 16.
- Views, and materializers that read another participant's view, come with
  the second pass.
