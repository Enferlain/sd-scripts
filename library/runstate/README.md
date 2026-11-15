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
| `generation.py` | `Generation`, `Generations` (owner registration, attempts, the current generation) and `Publication` |
| `targets.py` | `TargetRef`: component-qualified and participant-qualified selectors |
| `authority/` | `RunAuthority`, its records (`Participant`, `Binding`, `Route`, `View`), its section, transitions with the evidence and evaluators that judge them, and `MaterializationAttempt` |
| `projections.py` | What consumers receive instead of the authority; so far `Execution`, the owner seam for the coordinator |

## Realization

```text
  accepted run
     |  RunAuthority.seed          mint the run identity; publish every
     |                             declared participant, unbound (generation 1)
     |  MaterializationAttempt     run the declared materializers as one job;
     |                             the authority judges each binding's evidence
     |  route rebinding            preparation installs prepared routes
     |  readiness(CORE_EXECUTION)  every participant bound and every use's
     |                             route prepared from its current binding
  Execution(authority)             the coordinator's owner for participants
```

## Publication

A generation holds one section per owner. An owner builds its next section
off to the side; `Generations.publish` checks the publication's basis (any
publication since makes it stale), lets every changed owner refuse its next
section, and installs the next generation as one store. Reading the current
generation takes no lock, publications are serialized with each other, and
published sections never change, so correctness does not depend on the GIL.
In-place work, which first withdraws the guarantees it affects, is refused
until withdrawal arrives.

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
  snapshot of a participant the run has updated needs a copy of its device
  storage, which the model definitions supply in section 18.
- Grants come with training's words (section 16).
- Views, and materializers that read another participant's view, come with
  the second pass.
- Materializers take no inputs; their settings come with section 18.
