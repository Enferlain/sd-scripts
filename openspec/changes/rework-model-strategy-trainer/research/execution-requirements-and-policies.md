# Authored execution requirements and delegated realization choices

Date: 2026-11-26

Status: research / non-normative. Consolidates the November 26 authoring discussion (live SDXL sketch, bead `sd-scripts-syv.77`) and complements [Resource-aware execution](resource-aware-execution.md), which covers the November 24 runtime and scheduling research. This is not an amendment to OpenSpec requirements, an authoring API proposal or approval of a particular scheduler. Example requirements describe possible intent to investigate, not fields to add to the sketch.

## 1. Research question and motivation

How can the author express a requirement on execution when ordinary strategy composition does not already convey it, and what can a requirement sensibly be about? The system should understand authored behavior and choose its execution wherever that decision has been delegated. A requirement should constrain relevant entities or relationships the system already understands, rather than introduce a special flag for each pair of activities.

The question arose while distinguishing a reader's selected weight revision (which changes the meaning of a read) from how that revision is preserved (e.g. through a detached copy or protected live state). The same issue arises for placement, capacity, overlap, prefetch, rematerialization and other physical decisions. In discussion an illustrative phrase about validation not pausing training became a literal `validation_never_pauses_training` flag in the sketch. That is not the intended design: the phrase is underspecified, and nothing establishes that this SDXL strategy needs such a requirement. Do not add it or analogous flags.

This is the authoring-facing side of the resource scheduling problem. Existing code stays independently usable. The author selects ordinary or custom code and describes its role and relationships in the strategy; the IR system understands and executes it. Knowledge about code and hardware may help interpretation and feasibility checks but is not a separate source of authored intent. Configuration can expose choices that change the whole authored run. The decision to split or combine strategies belongs to authors, not a complexity restriction in the configuration system.

## 2. Separate behavior, obligations, requirements and optimization

The working distinction is conceptual rather than a prescribed schema:

| Kind | Meaning | Example |
| --- | --- | --- |
| Authored behavior | What computations and relationships the author chooses. | Read the revision selected at the request, compute a loss, update a unit from its contributions. |
| Derived obligation | What the system must ensure to fulfill that behavior and its contract. | Keep the requested state available until the last physical consumer finishes. |
| Additional requirement | A condition the author or launch configuration actually needs to hold, not already implied. | Bound a resource, require a particular realization property, or constrain an observable progress condition. |
| Preference / objective | A ranking among valid execution choices, not a mandatory correctness condition. | Prefer less transfer traffic or minimize time to completion. |
| Delegated decision | Something the system may determine among realizations satisfying the above. | Retain, copy or otherwise preserve a revision using a supported mechanism. |

Do not use “only things that change numbers belong in a strategy” as a rule. Latency, isolation, placement, resource use and delivery behavior can matter to an author even if mathematical results are unchanged. Nor should the author have to prescribe every such decision. A requirement that can already be derived should not be duplicated manually.

A strategy can expose execution requirements as configurable choices; requirements need not be machine-independent to appear in a strategy. Configuration and strategy may supply constraints with different scopes, but the meaning must be determined by what is expressed, not a blanket restriction based on where it is written. Hydra remains the configuration composition mechanism; neither this note nor the resource note settles a new configuration schema or mapping API.

## 3. Existing system concepts that can support the research

**Observed:** the current design already includes region/activity/request lifecycles, ownership, revision selection, leases, protected/retained products, stream flow bounds, device admission, physical completion, setup scope, and rank coordination. In particular:

- [D14](https://github.com/Enferlain/sd-scripts/blob/9055aa20/openspec/changes/rework-model-strategy-trainer/design.md#L2798-L2865) separates freshness from physical access realization, and separates version selection, placement and admission. A request-time revision is not a handler-start revision.
- [D15](https://github.com/Enferlain/sd-scripts/blob/9055aa20/openspec/changes/rework-model-strategy-trainer/design.md#L3250-L3385) distinguishes live activities from actual device overlap. Its current default is no overlap without an explicit strategy permission; whether this needs changing is open.
- [D16](https://github.com/Enferlain/sd-scripts/blob/9055aa20/openspec/changes/rework-model-strategy-trainer/design.md#L3896-L3948) defines retention until last use and coordinated access over resources. D17 specifies activity coordination. These already express many correctness conditions that should not need new annotations.
- [D20](https://github.com/Enferlain/sd-scripts/blob/9055aa20/openspec/changes/rework-model-strategy-trainer/design.md#L5454-L5478) distinguishes module training/evaluation mode from gradient participation; frozen code may carry gradients without switching to train mode. It is a training-domain concern expressed through the IR's existing system mechanisms, not a generic scheduling heuristic.
- The current contract's refusal principle is relevant: a refusal should name the actual failure or undeclared departure it prevents. Lack of prior ratings or tests is not itself grounds to forbid an experiment. If a target cannot enforce a required property, explain that target limitation; do not silently ignore it.

The authoring discussion is revisiting the previously documented implementation declarations and permission mechanisms; this note cites the existing design as evidence of available concepts and historical decisions, not approval of the disputed parts. Changes to specs and code belong after the design question is worked through.

## 4. External precedents

### 4.1 Legion mapping and Bishop: policies over existing entities

**Observed:** Legion distinguishes logical tasks/data access from physical mapping. Its mapper interface can choose processors, physical instances, variants and priorities and receive profiling results. Bishop is a higher-level mapping DSL demonstrated for Regent: selector-like rules identify tasks/regions and assign processor or memory targets. It is an example of controlling execution policy without changing computation code.

Sources: [Legion mapper API](https://legion.stanford.edu/mapper/index.html), [custom mapper tutorial](https://legion.stanford.edu/tutorial/custom_mappers.html), [Bishop tutorial notebook](https://notebook.community/StanfordLegion/jupyter-regent/notebooks/Session%203%20Part%201).

**Inference:** a generic relation or scoped selection over existing IR entities is more promising than a Boolean flag whose name embeds two domain activities. The Kuroiro author need not learn Legion's region annotations or its mapping DSL. Study how policy scope, matching, composition and precedence work. Bishop's demonstration is not evidence of modern integration readiness or a reason to introduce a second language into Kuroiro.

### 4.2 Constraint programming: interval, resource and alternative relationships

**Observed:** MiniZinc's global scheduling constraints and CP-SAT's interval constraints provide reusable operators rather than a domain-specific option for each pair of tasks.

| Constraint | Meaning (simplified) | Possible question for Kuroiro |
| --- | --- | --- |
| Precedence | An event or interval endpoint precedes another. | Is the ordering already implied by dataflow or ownership? |
| `disjunctive` / no-overlap | Selected intervals do not run simultaneously. | At what granularity and on what resource does exclusion apply? |
| `cumulative` | Total concurrent demands stay below a capacity. | Is demand known, estimated, or dynamically reserved? |
| `alternative` | Select an eligible realization for an optional task. | Which physical mechanisms satisfy the same authored behavior? |
| `span` | A containing interval covers selected component tasks. | Does existing scope/lifetime already express this? |

Sources: [MiniZinc scheduling constraints](https://docs.minizinc.dev/en/stable/lib-globals-scheduling.html), [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver/), [CP-SAT constraint definitions](https://github.com/google/or-tools/blob/stable/ortools/sat/cp_model.proto).

**Inference:** the algebra is useful even if no solver is embedded. Static interval optimization can help with bounded repeatable regions and candidate placement. A dynamic, unbounded, effectful run with unknown durations is not naturally one finite scheduling instance. Online admission, existing lifecycle enforcement and monitoring may be more appropriate. A solver timeout/UNKNOWN is not proof of infeasibility. Do not make this a general-purpose author-facing constraint language unless actual requirements justify one.

### 4.3 Halide and MLIR Transform: control over realization without changing computation

**Observed:** Halide separates the algorithm from schedule choices. MLIR Transform provides a representation for controlling transformations over existing IR, and its tutorial reproduces a Halide schedule.

Sources: [Halide](https://halide-lang.org/), [MLIR Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/), [reproducing Halide schedules](https://mlir.llvm.org/docs/Tutorials/transform/ChH/).

**Inference:** useful precedents for expressing transformations/constraints on accepted program entities instead of editing numerical code. Neither implies that Kuroiro should expose low-level scheduling or all compiler passes to authors. Their granularity and assumptions about compiler-visible computations differ from opaque Python calls, stateful requests and long-lived run activities.

### 4.4 Hard requirements, preferences and operational guarantees

**PyTorch checkpoint policies (observed).** Selective activation checkpointing distinguishes `MUST_*` and `PREFER_*` policies for saving/recomputing and, in current documentation, CPU offload. This demonstrates a difference between a binding choice and an optimization hint, over selected operations. Exact support varies with execution mode and version; inspect the selected Torch version before relying on offload behavior. Source: [torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html).

**Ray placement groups (observed).** Resource bundles and packing/spreading strategies express allocation relationships over generic tasks/actors; placement groups can reserve bundles together. Logical resource reservations are not proof of physical memory isolation. Source: [Ray placement groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html), [scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/index.html).

**Linux SCHED_DEADLINE (observed).** Runtime, deadline and period express a particular timing contract, backed by admission control and CPU scheduler mechanisms. It is a useful reference for precision of terms and for refusing unsupported promises, not a guarantee for GPU work. Source: [Linux deadline scheduling](https://kernel.org/doc/html/latest/scheduler/sched-deadline.html).

**CUDA stream priority (observed).** Stream priorities influence scheduling but do not guarantee execution order or preempt already-running work. A high-priority training stream is not evidence that auxiliary work cannot delay it. Source: [CUDA asynchronous execution guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

**Inference across these:** distinguish a requirement from a preference and an objective, define what backend mechanism actually provides it, and expose the scope and limits of the promise. A policy can guide the system without claiming guarantees its runtime cannot enforce. The appropriate granularity for Kuroiro is still an open question.

## 5. Potential subjects of requirements

The following inventory is a research aid, not an API vocabulary or list of fields that must exist. Begin with concrete authored cases, determine what the existing IR already expresses, and add a representation only if there is an unmet need.

| Subject | Potential requirement or objective | Questions before supporting it |
| --- | --- | --- |
| Placement and residency | Use compatible device types; preserve residency in a memory domain. | Physical vs logical location, scope, movement of derived storage, supported target mechanisms. |
| Capacity | Bound live device/host memory or another scarce resource. | What counts, how reserved/observed, external allocations, enforcement and failure behavior. |
| Ordering and freshness | Observe a specific revision or a publication-before-consumption relationship. | Often already expressed by ordinary strategy structure; avoid duplicates. |
| Concurrency / exclusion | Selected work must not overlap; or must be independently admissible. | Overlap on what resource, during which lifetime, what counts as progress? |
| Lifetime and ownership | Preserve state or an artifact until its consumers settle. | Derivable from leases and retention? Exact transfer/release conditions? |
| Progress | Eligible pending work is not indefinitely starved. | Eligibility, fairness assumptions, finite work, competing requests and failure. |
| Latency and rate | Bound waiting time, completion latency or throughput. | Measurement window, workload assumptions, hard vs best effort, supported guarantees. |
| Realization | Constrain which properties an execution alternative must preserve. | What is semantically equivalent, and under what scope/evidence? |
| Numerical/replay | Require a specified comparison or recovery guarantee. | Observable values, state/effects, reference, tolerances, hardware, time horizon. |
| Optimization objective | Improve completion time, memory, energy or transfer cost. | Tradeoffs, scope, measurements, preference strength; never silently violate a requirement. |

Requirements can concern sets of activities/resources or invocation instances, not just source-level names. The exact way authors refer to those entities, how scope works, and how configurable choices expose them are open questions. Prefer existing references/relationships over inventing a separate naming hierarchy. Do not require every category in the first implementation or in the SDXL sketch.

## 6. Worked ambiguity: “validation never pauses training”

Assume the author already expressed that a validation request reads the model revision current when requested. Several feasible realizations may exist: protect live state and defer conflicting updates; capture a detached revision; transfer a copy to another device; or a supported alternative preserving that same state. Capture itself may need synchronization, and detached execution may still compete for capacity and bandwidth.

| Possible intended meaning | What is actually constrained | Why not equivalent to the others |
| --- | --- | --- |
| Training need not wait for validation to finish. | A dependency/protection lifetime relationship. | Training may still wait for capture, memory or device admission. |
| Give training scheduling preference. | Priority or optimization policy. | Priority is not isolation or a hard timing guarantee. |
| Both activities keep progressing. | Liveness under stated eligibility/resource assumptions. | Neither needs to run literally at the same instant. |
| Validation changes throughput by at most an agreed amount. | Measured performance bound and reference workload. | Needs a defined measurement window and enforceable mechanism; a copy alone proves nothing. |
| Use independent devices for auxiliary work. | Placement and resource isolation. | Shared buses, CPU, allocator or communication may still affect performance. |

These are different requirements and may not all be necessary or even satisfiable. The authored revision requirement and the existing dependencies should stand on their own. Only if the author wants a further observable guarantee should a general means of expressing that guarantee be investigated. Do not translate this table into five API flags. No change to the live SDXL example is warranted solely because this illustrative case was discussed.

## 7. Equivalence is an eligibility condition, not a claim about all mechanisms

Delegated choice means the system chooses among alternatives that preserve the requirements of the actual authored behavior. Examples of subtle differences:

- A detached copy that omits buffers, scale/amax history, adapter state or another dependency may not represent the requested revision.
- Recomputation can repeat mutation, draw different random values, use different module/precision state or observe a newer weight version.
- Producer prefetch can change the input observed by a consumer if production depends on training progress or mutable state.
- Offload and restoration must preserve required storage semantics, including aliasing and format; quantization is not automatically equivalent to an exact copy.
- Concurrent execution can change RNG consumption, reduction order, kernel choice or collective ordering.

The system should establish applicability with evidence appropriate to the guarantee, rather than assume a mathematical resemblance is sufficient. A single small-input check is evidence for its tested scope, not a universal proof. The evidence may come from static analysis, a supported mechanism's contract, qualified implementation properties, known backend guarantees and observations. Keep correctness evidence distinct from performance estimates and recommendations. Nothing here requires that facts be stored in library files or that every library callable carry an IR declaration.

There need not be a privileged “plain plan” that the system always falls back to. An unsupported candidate can be skipped while another eligible candidate is considered. Where an explicit author choice rules out an otherwise equivalent realization, honor the choice. If no supported eligible realization is found, distinguish the planner's limitation from a contradictory or impossible requirement.

### Numerical and reproducibility scope

Mathematical equivalence, tolerance-based numerical comparison and bitwise equality are useful distinctions to investigate, not a complete settled hierarchy of run guarantees. Define the scope (operation, region, step, state transition, whole run), compared values/effects, reference, tolerance metric, allowed nondeterminism and relevant environment. A local comparison does not bound accumulated divergence over training. A different kernel may be admissible for bitwise comparison if the required equality can be established; a recorded identical kernel does not by itself establish reproducibility.

Independent random streams, per-sample keys and counter-based generators are promising candidates for isolating unrelated draws, preserving behavior across batching/rank changes, and reducing recorded RNG state. They are separate research questions: identity stability, repeated sample occurrences, draw indexing, integration with stateful third-party generators, replay and backend capabilities still matter. Do not assign stream identity from a variable name without defining the consequences of renames and edits. These issues intersect with realization choice but this note does not resolve the RNG design.

## 8. Where a guarantee can be established and enforced

| Stage | Typical work |
| --- | --- |
| Verification | Derive logical obligations and identify structural contradictions, invalid scopes and known dependency/wait cycles. |
| Preparation | Match target capabilities, compare candidate realizations, establish feasible static arrangements where possible. |
| Admission | Reserve actual resources, check readiness and protection, enforce bounds and coordinate ranks before work starts. |
| Execution | Observe actual completion, resource consumption, progress, timing and effects; adapt only within accepted guarantees. |
| Settlement/reporting | Check outcome-level guarantees and record actual choices, violations, unresolved effects and cleanup. |

These are existing system responsibilities, not a separate requirement interpreter. A monitor is not enforcement: observing a missed deadline after it occurs cannot supply a hard latency guarantee. A requirement with no supported enforcement mechanism must be represented honestly (unsupported, or an explicitly accepted best-effort objective), not silently downgraded.

Distinguish contradictory intent, unsupported target capability, insufficient evidence, currently unavailable resources, no solution found by the planner within its search bounds, and actual runtime violation. These are different outcomes. In particular, “no solution found” does not mean “impossible”; a resource limit observed at one instant need not be a permanent incompatibility. Dynamic failures and recovery must respect ownership and uncertain effects.

Plan records should capture relevant evidence and actual decisions, including changes during execution. Hard requirements, objectives and achieved results must not be conflated. Preferences can guide selection, but reporting should make important departures explainable rather than generate arbitrary noise for every minor heuristic. Exactly what is persisted and how replay constrains new planning remain open.

## 9. Authoring mechanisms to compare, not adopt yet

Four families are worth studying against actual strategies:

1. Ordinary structure: express ordering, dataflow, scopes, revision selection and activity lifetimes through existing constructs. Prefer derived requirements over duplicate annotations.
2. Scoped information: associate a genuine extra requirement with an existing activity, resource, region or reference when structure alone is insufficient.
3. Declarative relationships: express a generic relation among existing entities when scope/structure alone cannot convey it. MiniZinc and Bishop provide precedents at different levels, but no need to adopt their syntax.
4. Reusable execution policies: group requirements/objectives where authors or launch configurations want to reuse them. A policy must not redefine the authored computation behind the strategy's back.

These approaches are not mutually exclusive. The next step is to identify what authors need to express in realistic configurable strategies and whether the current system can already derive or enforce it. A new API such as `iro.require(...)`, special selectors, a DSL, an annotation framework or a global solver is not a conclusion of this research.

## 10. Research cases and questions

Use the live sketch and existing representative training cases (`teacher-student-distillation.md`, `training-data-pipeline.md`, `distributed-training-execution.md`, `preference-and-rl-post-training.md`) as stressors for the general mechanics, not restrictions on how broad each strategy can be.

| Case | Question to answer |
| --- | --- |
| Concurrent requests for a particular revision | Is the intended freshness and lifetime already expressible? What additional constraint, if any, does the author actually need? |
| One-GPU execution under memory pressure | Can run-wide capacity and valid realizations suffice without per-activity flags? How does admission maintain progress? |
| Live producer consuming evolving state | Does an extra requirement exist beyond the authored dependency/flow policy? Can prefetch be changed safely? |
| Several optimized components, frozen conduits and retained gradients | Which obligations are derived from training meaning through IR, rather than from physical scheduling preference? |
| Heterogeneous rank placement and collective execution | What global guarantees are necessary while allowing different local plans? |
| Stochastic or effectful region with alternative realization | Can the alternative preserve observed behavior, and what must be checked before allowing it? |
| An author genuinely requires progress/latency/isolation | Write the observable property precisely; identify mechanism, scope, eligibility conditions and enforcement stage; demonstrate a violated and a fulfilled case. |

For each case, first describe intended behavior with ordinary authoring mechanics. Then identify a requirement not conveyed by that description, if any. Analyze whether it is a semantic obligation, binding constraint, preference or optimization objective; derive what can be derived. Compare possible representations only for the remaining need. Use the existing lifecycle reference and failure tests, not just successful numeric outputs.

**Open questions:** which generic relationships are already sufficient; what genuine unmet requirements appear; how requirements refer to dynamic invocation instances; how scoped/configured constraints compose; how to establish hard guarantees with opaque code and shared resources; whether currently explicit permissions can be replaced by derived eligibility without losing a deliberate author choice; how runtime adaptation/replay interacts with requirements. See the companion note for candidate execution mechanisms and measurement experiments.

**Research conclusion:** the author should be able to express precise requirements over relevant execution entities or relationships when necessary, while the system decides among valid realizations wherever that authority is delegated. Evidence from prior runs can help choose an execution plan but cannot override intent or safety obligations. The external precedents support this separation; they do not settle an author-facing syntax or establish a need for additional flags in the current SDXL sketch.
