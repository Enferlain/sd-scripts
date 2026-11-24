Pass 1: architecture, authoring and ownership.

Apply the shared general audit protocol. This is a read-only concern audit,
not another orientation pass or a production implementation task.

Determine whether the recorded architecture consistently assigns responsibility
across:

training contract → authored strategy → accepted arrangement
→ governed realization → executable run + Trainer engine.

These are semantic dependencies, not a mandatory universal execution sequence.
Verify the specification actually realizes the governing goals; do not assume
coherence because the intended topology is stated.

Sources:
- Read the governing design's source-authority rules and relevant decisions,
  especially D1–D7, D10 and D12, including the composed-run-state discussion.
- Read complete relevant requirements and scenarios in training-contract,
  run-participant-state and accepted-training-execution.
- Follow ownership across training-optimization, runtime preparation,
  capability coordination, persistence, metadata/observability and PEFT deltas.
- Include relevant companion metadata requirements and unchanged main-spec
  obligations wherever they affect these boundaries.
- Consult the normative direction, explicit historical corrections, research
  cases and experiments where needed. Use current production only as evidence
  for a migration claim, not as the target ownership model.

Use the Pass 0 report as navigation, not authority. Its investigation items
are leads, not confirmed findings or the complete audit checklist.

Examine these questions:

1. Contract and fulfillment
   Does the contract exist before authoring and describe supported consumer
   behavior without being assumed code-owned by a Trainer instance?
   Are core, recognized capabilities and strategy features distinguished by
   their consumers?
   Does fulfillment establish authoritative run-specific meaning, obligations
   and permissions without automatic assembly or silent repair?
   Do later evidence checks enforce those obligations rather than independently
   reinterpret the strategy?
   Distinguish semantic acceptance from executable readiness without creating
   competing acceptance authorities.

2. What survives authoring
   Does the accepted arrangement retain selected executable behavior, state,
   dependencies, policies and capabilities without ordinary callbacks into the
   broad authoring interface?
   Check both complete release of the authoring object and an implementation
   object retained through a separately accepted runtime role.
   Do not assume role separation requires separate Python objects or classes.

3. Dynamic behavior and whole-run composition
   Verify that acceptance does not freeze execution: batch-dependent decisions,
   schedules, adaptive feedback, stateful algorithms, bounded choices and
   permitted structural changes must retain their accepted meaning.
   Check ownership of independently progressing activities, lifecycle stages,
   handoffs and cross-owner dependencies.
   An action-local computation graph or fixed input–action–optimizer sequence
   is not evidence that hierarchical whole-run coordination is covered.

4. State and coordination ownership
   Examine who establishes, changes, coordinates and observes:
   participant/relationship/binding state; activity and input state;
   optimization state; selected-operation state; capability state;
   and metadata/observation facts.
   One coherent run aggregate must not silently become one unrestricted writer.
   Conversely, local ownership must not permit bypassing coordinated changes
   that affect other owners.
   Distinguish ordinary accepted state evolution from changes requiring a
   governed transition or new fulfillment.

5. Trainer, capabilities and the research path
   Verify that generic coordination remains training-owned while specialized
   algorithms remain selected implementations—not family branches in Trainer.
   Check capability availability, strategy provision, runtime request and
   readiness as distinct concerns.
   Verify explicit alternative authority can genuinely transfer the supported
   mechanics it declares, with an identified executor and retained duties.
   Do not reduce the research path to cosmetic customization, or turn it into
   an unqualified “strategy takeover.”
   Distinguish declared/trusted Python behavior from guarantees the framework
   can actually enforce. Do not infer a mutation sandbox.

6. Domain boundaries and legacy displacement
   Check that independently managed participants need not become artificial
   family component slots, while targeting preserves applicable provenance.
   Verify PEFT feature/method behavior, target resolution, optimization grouping
   and artifact choice have coherent owners.
   Confirm TrainingMode dissolution does not recreate a runtime mode axis or
   leave old strategy/Trainer fields as competing authorities.
   Metadata and observers must consume owner-established facts, not establish
   live identity or reconstruct topology independently.

Use concrete recorded cases to test these boundaries:
- ordinary maintained training;
- stateful/adaptive selected behavior;
- one participant used in several roles, including a frozen gradient path;
- an independently managed learned participant and direct-plus-PEFT training;
- independent production alongside training or capability work;
- a stage change affecting several owners;
- an explicitly granted imperative region alongside retained Trainer duties.

These are architectural source checks. Do not demand a new integrated
implementation, universal backend support or new experiments merely because
the existing prototypes are narrow.

Investigate P0-INV-02, P0-INV-03 and P0-INV-07 explicitly. For each, determine
whether the sources establish a coherent distinction, a real ambiguity or
conflict, or a concrete choice deliberately deferred to G5. Follow other
Pass 0 items only where they intersect this concern.

Keep detailed scheduling, backend publication, derivative protocols, recovery
formats and performance measurements with their assigned later passes.
Inspect enough to establish ownership consistency; route remaining questions
without silently assuming their answers.

Return:
- a concise source-backed responsibility map for these boundaries;
- findings using the general protocol and P1-prefixed IDs;
- verified areas and concrete case reasoning;
- dispositions of the three named Pass 0 investigation items;
- coverage limitations and questions routed to later passes.

Do not repeat the entire Pass 0 register, propose a new architecture, apply
corrections, or issue an overall G5-readiness verdict.
