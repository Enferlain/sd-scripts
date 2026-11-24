## Purpose

Define stable semantic optimization ownership independently of concrete optimizer objects and runtime preparation details.

## ADDED Requirements

### Requirement: Units retain semantic identity
An optimization unit SHALL have run-scoped semantic identity and accepted-definition revision distinct from its authored address, parameter objects, optimizer/scheduler objects and backend handles. Changes to accepted membership, grouping, policies or semantic dependencies SHALL revise the definition. Split, merge and retire-and-recreate SHALL establish new unit identities.

#### Scenario: Wrapper replacement
- **WHEN** preparation replaces concrete handles without changing the accepted optimization definition
- **THEN** the unit retains its identity and accepted-definition revision

### Requirement: Standard parameter ownership does not overlap
Standard optimization resolution SHALL assign each selected semantic parameter to one optimization unit and one execution group, accounting for tied aliases. Deliberate overlap SHALL require an accepted capability or extension with explicit coordination semantics.

#### Scenario: Tied parameter selected twice
- **WHEN** two standard selections refer to the same tied parameter
- **THEN** resolution prevents duplicate optimizer ownership rather than treating distinct names as distinct parameters

### Requirement: Optimization realization follows accepted intent
The standard profile SHALL give Trainer-owned infrastructure responsibility for realizing accepted training subjects and optimization policies. Logical groups SHALL remain distinguishable from execution groups. Realization and preparation SHALL publish only a complete current runtime; backend constraints SHALL be allowed to determine concrete construction order within the coordinated attempt.

#### Scenario: Joint backend construction
- **WHEN** a backend requires model and optimizer preparation to be coordinated
- **THEN** realization follows that backend order while preserving the accepted semantic plan
- **AND** no partial optimizer runtime becomes current

### Requirement: Specialized behavior does not imply ownership transfer
Selection of specialized optimization-related behavior SHALL NOT implicitly transfer backward, advancement or synchronization authority away from the standard Trainer profile. A change of authority SHALL require the explicit extension boundary defined by the training contract.

#### Scenario: Auxiliary trainable feature
- **WHEN** a definition selects an auxiliary trainable feature
- **THEN** acceptance determines its supported optimization and lifecycle obligations
- **AND** selection alone neither makes it core Trainer behavior nor grants imperative control
