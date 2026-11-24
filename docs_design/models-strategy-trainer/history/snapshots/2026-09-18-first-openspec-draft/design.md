## Context

Status: initial governing design, 2026-09-18. **Production implementation is gated.** The artifacts capture settled architecture and bound the next design work; their existence is not evidence that the exchanges or migration are finished. Beads: `sd-scripts-syv`.

See [proposal.md](proposal.md) for scope. Current code predates this direction. On this pass, exact source and the code graph confirmed:

| Evidence | Current behavior | Design consequence |
| --- | --- | --- |
| [launcher](../../../train.py), `train` | Configuration validation precedes separate strategy and mode construction | Configuration validation alone cannot establish the complete authored arrangement |
| [Trainer](../../../library/training/runners/trainer.py), `train` | Coordinates setup, caching, model and optimizer preparation, evaluation, loop, and cleanup | Retain generic lifecycle responsibilities; derive narrower accepted inputs |
| [loop](../../../library/training/phases/training_loop.py), `_run_epoch_steps` | Passes family-shaped fields into `process_batch`; also owns synchronization, backward, advancement, and observation | Separate prepared domain computation from retained engine mechanics |

This is a bounded refresh, not a claim that the full historical inventory was re-audited. Graph generation `2026-09-18T00:00:47Z` reported matching file metadata and no recorded coverage issue for these paths. Graph coverage remains best effort.

### Supporting record and carry-forward rule

The following records remain available. New decisions belong here and in this change's specs. Settled decisions not yet expanded into individual scenarios remain constraints on this draft; omission from the initial delta set does not repeal them. Before implementation, task 1.1 must finish the requirement-by-requirement transfer and record its locations.

| Record | Use |
| --- | --- |
| [Direction](../../../docs_design/models-strategy-trainer/strategy_system_direction.md) | Settled architecture, Q1–Q5, contract surfaces, persistence and ownership; inherited baseline |
| [Exchange design](../../../docs_design/models-strategy-trainer/strategy_contract_exchange_design.md) | Settled exchange semantics and explicitly marked working decisions; EX-029 corrects runtime topology |
| [Inventory](../../../docs_design/models-strategy-trainer/strategy_system_inventory.md) | Production evidence and representative cases |
| [Implementation mapping](../../../docs_design/models-strategy-trainer/strategy_system_implementation_mapping.md) | Current responsibility destinations and design dependency order |
| [Notes](../../../docs_design/models-strategy-trainer/notes.md) | Chronology; later corrections supersede earlier positions |
| [Original discussion](../../../docs_design/models-strategy-trainer/strategy_discussion.md), [external discussion](../../../docs_design/models-strategy-trainer/strategy_discussion_external.md) | Raw historical context, not normative requirements |
| [Framework comparison](../../../docs_design/models-strategy-trainer/framework_pattern_comparison.md) | Subordinate prior-art evidence |
| [Isolated spike](../../../docs_design/models-strategy-trainer/executable_spike/) | Evidence about identity, freshness and publication failure; not the production API or architecture prototype |

## Goals / Non-Goals

The design must make ordinary authoring understandable, make acceptance authoritative, keep model knowledge in selected implementations, and let the Trainer execute without retaining the authoring object. It must support substantial research changes through explicit contract extensions, including changes to execution authority.

This first draft does not choose Python class names, a graph implementation, decorators, or package placement. It does not adopt another training framework, build an automatic strategy resolver, or turn SDXL anatomy into universal inputs. Metadata catalog/query expansion remains its own change; runtime identity and metadata provenance must be connected without conflating them.

## Decisions

### D1. The intended Trainer defines acceptance

Repository goals and production evidence inform the intended Trainer. Its mechanics and deliberately supported pipeline capabilities define the contract, which guides authoring before any strategy is submitted. Establishment means judging an authored definition under that existing contract and deriving obligations for later lifecycle stages.

This carries forward the contract-first correction. Neither copying today's strategy ABC nor choosing a contract after seeing an arbitrary strategy produces the intended boundary.

### D2. Strategy authoring produces an accepted arrangement

```text
Trainer and supported capabilities define the contract
                         ↓ guides
                 authored strategy
                         ↓ validated under that contract
                accepted run arrangement
                         ↓ materialization and preparation
              executable run + Trainer engine
```

The arrangement retains selected implementations, configuration, dependencies, declared effects, and one run binding authority. Ordinary execution must work after the authored strategy object is released. Capability implementations remain library behavior; removing the authoring object does not move their internals into Trainer.

The accepted execution form must express composition and dependencies. A graph, regions, schedule, lowered Python, or hybrid remains a design decision to resolve from concrete cases. A renamed broad strategy method bag does not satisfy this decision.

### D3. Trainer ownership is the starting frame

These are responsibilities, not proposed methods or a fixed call sequence.

| Trainer/pipeline responsibility | Accepted information it needs | Specialized behavior retained outside generic mechanics |
| --- | --- | --- |
| Admit a run | Successful establishment, contract/profile, deferred obligations | Authored selections and explicit wiring |
| Coordinate materialization | Participants, relationships, readiness requirements and permitted transitions | Construction, loading, adapter attachment |
| Prepare execution | Coherent current bindings, required members and backend constraints | Component-specific preparation contributions |
| Realize optimization | Accepted subjects, grouping and advancement meaning | Supported optimizer/capability implementations |
| Execute training | Prepared operations and dependencies, current routes, optimization runtime, allowed effects | Model and objective computation, custom operations and accepted imperative regions |
| Trigger capabilities | Selected capability, readiness, request/result and lifecycle meaning | Evaluation, generation, encoding and serialization semantics |
| Observe, interrupt and finish | Typed results, run coordinates and owned cleanup/state contributions | Domain-specific observations and state |

The ordinary engine owns time, infrastructure, synchronization, optimization mechanics, triggers, and observation. A recognized extension may deliberately alter execution ownership. Neither unusual cadence nor selection of an auxiliary feature automatically transfers that ownership.

### D4. One authority owns current run bindings

Carry forward Q1–Q5 without reopening them: stable participant incarnation, explicit route cardinality, distinct transitions, semantic artifact products, and one run authority. Materialization, compatible replacement and wrapping preserve participant identity. Relationships and optimization are independent axes. Metadata observes accepted state. Backend composites do not become participant identities.

The initial `run-participant-state` and `training-runtime-preparation` deltas cover the main invariants. The full settled transition and persistence baseline remains applicable while its detailed scenarios are transferred.

### D5. Optimization has semantic identity before concrete runtime

Accepted training subjects lead to semantic units and logical groups. Concrete parameters, execution groups, optimizer/scheduler objects and prepared backend handles are realizations. Standard parameter ownership does not overlap between units or execution groups, including tied aliases. Deliberate overlap needs accepted coordination through a recognized extension/capability.

Only semantic plan before realization and complete runtime after successful publication are fixed ordering constraints. Backend requirements determine construction/preparation ordering inside the coordinated attempt. Advancement policy and its connection to step execution still require design. EDM2 remains an example capability, not inherent core behavior.

### D6. Maintained and research execution share explicit acceptance

The author can compose maintained operations, supply custom structured operations, or select an imperative region with declared authority. The contract enforces supported behavior and rejects unsupported authority. An imperative executor is a runtime role; it does not require an active strategy object. The strongest research case must exercise actual control over backward or advancement, not merely a custom forward function.

## Design work required before implementation

The next discussion starts with D3: for each responsibility, specify the input, result, readiness condition, effects and failure behavior the Trainer needs. Work through SDXL fine-tune and adapter code first, then differing SD/SD3 paths and a strong research case. Existing inventory supplies navigation; exact code inspection supplies evidence for each new conclusion.

From those cases, finish minimum execution meanings, optimization realization/advancement, and step execution. Then choose concrete binding/preparation types, the accepted arrangement shape, and the authoring API. These are pending design work, not questions delegated to implementation by default.

The gate opens only when all four exchanges—binding, preparation, optimization, and step execution—have defined inputs, results, state effects, external effects, and failure conditions; standard and extension ownership are explicit; the representative cases are covered; overlapping specs are reconciled; and reviewed vertical milestones have concrete tests and no duplicate authority.

### Trace to implementation

Every material decision must connect responsibility → requirement → exchange → scenario → milestone → test. Initial anchors are below; production milestones and test paths remain deliberately unassigned until the design gate work supplies them.

| Responsibility / decision | Initial requirement | Scenario anchor | Work needed to complete trace |
| --- | --- | --- | --- |
| Admission / D1 | Contract guides authoring and acceptance | Unsupported combination | Complete obligation derivation and diagnostics |
| Execute without author object / D2 | Accepted execution survives authoring | Authoring object released | Compare cases and choose execution representation |
| Current bindings / D4 | One authority owns current participant state | Compatible replacement | Transfer remaining Q1–Q5 scenarios and map consumer cutover |
| Preparation / D4 | Preparation publishes one coherent result | Stale candidate | Define concrete backend exchange and failure tests |
| Optimization / D5 | Units retain semantic identity | Wrapper replacement | Finish advancement and runtime realization |
| Research / D6 | Custom execution declares its authority | Imperative optimization region | Define retained/delegated actions and conformance tests |

## Risks / Trade-offs

- An initial spec can look more complete than it is → keep the implementation gate explicit and leave design tasks unchecked.
- SDXL can dominate the representation → compare materially different families and strong research execution before choosing it.
- Generality can obscure readable code or add dispatch overhead → evaluate concrete authoring and execution examples; move stable selection and validation out of the hot loop and measure representative runtime overhead during implementation.
- Old and new authorities can coexist accidentally → migrate one complete run path with one authoritative state owner and remove displaced writes in the same milestone.
- The metadata change modifies overlapping loading requirements → reconcile its deltas and dependencies before implementing either overlap.
- Broad custom access can bypass invariants → define the exact authority an extension receives and which guarantees it must preserve or explicitly replace.

## Migration Plan

This is one governing change with reviewed vertical milestones. Final milestone boundaries depend on the completed exchanges and source inspection. The first production case is SDXL, but universal interfaces must already accommodate the other required cases.

Each milestone must name replaced owners and call paths, requirements and scenarios, focused regression/conformance tests, and review gates. No milestone may keep a second authoritative binding store, rename `TrainingMode` as its replacement, or use a knowingly false temporary contract. Retained domain functions can move behind narrower inputs as their consumers migrate.

Recovery is at the coherent milestone boundary: revert the affected run-path change as a unit if verification fails. Runtime preparation failure follows the specified replacement-only or destructive-mutation semantics; code rollback must not be confused with restoring already mutated runtime state.
