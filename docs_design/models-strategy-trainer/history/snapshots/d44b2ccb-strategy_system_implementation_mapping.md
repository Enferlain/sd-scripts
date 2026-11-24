# Strategy System Implementation Mapping

## Status And Purpose

This is the deferred bridge from the settled model–strategy–trainer design to
the eventual production-code shape and governing OpenSpec.

The openspec design sequence is: (complete action, not sequential 1 2 3 4 step)

```text
architecture and exchange semantics
  -> compare with the current-code inventory
    -> derive the target code and migration path
      -> write the OpenSpec and implementation milestones
```

This document is not a source of architectural authority. Use:

- [`strategy_system_direction.md`](strategy_system_direction.md) for the
  normative architecture and settled semantics;
- [`strategy_contract_exchange_design.md`](strategy_contract_exchange_design.md)
  for the ongoing exchange design;
- [`strategy_system_inventory.md`](strategy_system_inventory.md) for evidence
  about what the code currently does; and
- [`notes.md`](notes.md) for chronological discussion history and other notes.

The items below are intentionally deferred until the relevant architecture and
exchange meanings are stable. They must not be treated as blockers merely
because their Python form has not been chosen.

## Deferred Code-Shape Decisions

### Construction and contract establishment

- Choose the Python construction API that accepts an authored definition and
  returns the one established `TrainingStrategy` accepted by Trainer.
- Make bypassing establishment or mixing contract versions impossible through
  the normal construction path.
- Decide whether construction uses a factory, builder, class method, separate
  authored-definition value, or another readable mechanism.

### Built-in and custom conformance representation

- Choose how an explicitly selected built-in implementation exposes its
  contract-conformance information.
- Compare explicit catalogs, attached typed descriptors, decorators, ordinary
  class metadata, and other code-local mechanisms after the required
  conformance information is known.
- Preserve explicit author wiring and avoid automatic assembly, method-presence
  discovery, family-name dispatch as conformance, and import-order-dependent
  availability.
- Define the concrete Python entry for explicit custom/direct conformance and
  contract extensions.

### Obligation and runtime-evidence types

- Define the concrete typed values used for obligations derived during
  establishment.
- For each exchange, encode which facts are known at establishment and which
  evidence can only be supplied by loading, materialization, or preparation.
- Choose the concrete generic/protocol/dataclass boundaries without creating a
  universal model-kind union or an unstructured fact dictionary.

### Binding state and access

- Define the concrete bound-state, execution-route, snapshot, and scoped
  access-view types that replace `LoadedModelComponent.module` and Trainer's
  family-shaped projections.
- After the durable-identity behavior is settled in the exchange design,
  choose the live `ParticipantRef` value type and its metadata/history encoding.
- Decide whether the first implementation rejects reuse of a retired authored
  key within one authority. If reuse is implemented, it must create a new
  reference and must never revive the retired incarnation.

### Transition requests and results

- Choose whether transitions use one shared envelope, several narrow
  operation-specific types, a batch/transaction container, or a combination.
- Represent atomic multi-participant and participant-plus-relationship changes
  while preserving the already settled all-or-nothing publication behavior.
- Encode expected revisions, changed revisions, invalidations, structured
  failures, and historical transition facts in readable request/result types.

## Later Current-To-Target Mapping

Once the architecture questions are settled, expand this document with a
source-backed mapping of:

```text
current production responsibility
  -> target exchange or owner
    -> retained, evolved, moved, split, or removed code
      -> dependency order and reviewable migration milestone
```

That mapping—not the preliminary vocabulary in the design record—should drive
final modules, class names, and OpenSpec implementation tasks.
