# Revision brief: authoring-model design changes

Parked, 2026-11-26. Kept for reference, not an active task or an agreed design. It mixes authoring questions with earlier design questions. The decisions it lists as pending are still undecided.
Date: 2026-11-26. Prepared by Claude from two reviews and their follow-ups. Basis: `9055aa20`.

**Task: revise the proposal for another review. Don't edit the design, delta specs or code.**

This brief replaces the verdicts and requested edits in `authoring-model-design-changes-review.md`. In particular, it changes that file on A1 evidence, A3, A7 and C3. Its file locations remain useful starting points. Agreement between reviewers is not the user's approval, and the later passes were not independent of earlier findings. Code was read, not run.

Paths are under `openspec/changes/rework-model-strategy-trainer/` unless they start with `library/`.

## Governing rules for the revision

1. **How the system runs a read must not change gradient participation or any other authored behavior.**
2. Identify what depends on current mechanisms so required meaning and guarantees survive. This does not require keeping the current representation.
3. Derive what can be established from the authored strategy. Don't add a mandatory annotation merely because a mechanism is being removed.
4. Recording the intended direction does not mean the implementation supports it.

## Access modes: investigate before rewriting A4 or A7

Trace every use of access mode in design, specs, IR, training and run state. Classify each by concern; one use may involve several, and these need not become five constructs.

| Concern | Question |
| --- | --- |
| Version selection | Which state is observed? |
| Gradient participation | How does state take part in differentiation? |
| Access authority | What may be read or changed? |
| Lifetime dependency | What stays available and protected, and until when? |
| Physical realization | How are those requirements satisfied? |

Also trace consumers: verification, normalized form, obligations, reproducibility class and accepted-run loading. For each, state what information determines its result once realization is delegated.

**Test for lost meaning:** can two strategies differ only in access mode and intend different behavior? If so, show how existing authored structure carries that distinction, or record what is missing.

One case to examine, inferred from code: a training strategy makes a live prediction from trained weights and uses a second read of the same weights as a target. `training/units/derivation.py:94` treats a live read as a gradient source and a snapshot read as not one. First check whether the whole strategy passes the other checks. Run-state support for a snapshot of updated weights is currently missing. The design's existing no-derivative-edge cut may carry the intent.

Known dependencies:

- Gradient sources: `design.md` 3207; accepted-training-execution spec 167; `library/training/units/derivation.py:94`.
- `library/iro/verify.py`: retained protection (`held`), nested access (`_nested`, `iro-access-002`), lease obligations and reproducibility.
- workflow-composition spec 446-475: reproducibility wording uses access modes.
- `library/iro/accepted.py`: loading compares normalized form, obligations revision and reproducibility class.
- `library/runstate/authority/`: current snapshot limits for updated participants.

Keep the rule against gradients crossing activities, and required state protection through backward, unless a replacement is shown to preserve the intended behavior and guarantees. The analysis mechanism remains open.

## Revisions by item

### A1: library boundary and code knowledge

- Don't move the old declaration architecture into a catalog under new names. Review the assumption throughout that an independently declared, installed implementation is the unit of selection, including D16's selection-by-name rule.
- Don't drop declaration evidence globally. It covers inputs, outputs and named type structure, and word definitions also use declarations. Source evidence covers only the defining module; `library/iro/evidence.py` says pieces do not carry fixtures yet. Preserve the needed coverage when removing library declarations.
- Separate author intent, facts about code, and execution choices. An assertion does not change what code does. Keep its provenance and level of verification.
- Resolve the assertion conflict in decision 1 below.
- Add the missed decorator paragraph and example (`design.md` about 4834-4850), and check other declaration passages listed in the earlier review.
- Move the run-state dependency question out of settled corrections. First distinguish ordinary domain objects from authoritative records of a running execution, including the placement of current types.

### A2 and C3: configuration and run requirements

- Keep Hydra and author-exposed choices. Don't require catalog membership for usability or specify selection by name.
- Say the strategy exposes everything configuration may change **about the strategy**. Launch settings are separate.
- Proposed C3 principle: configuration cannot silently invalidate a fixed requirement; it may select an exposed choice that changes or removes one. Additional run requirements may constrain execution. Report conflicts. How requirements are expressed remains open.

### A3: precision

A participant-level recipe may be a convenient authoring form. Preserve separate storage, computation, optimizer, process-wide setting and serialization responsibilities. Don't imply that all routes or uses share one precision context.

### A5: overlap and residency

- Direction: authors need not manually authorize an optimization. A way to constrain one remains open when there is a genuine need.
- Don't claim authors can express arbitrary execution requirements or invent replacement flags.
- Keep current operational mechanisms until their replacements are implemented. The design says overlap permission syntax has not been settled (about 3281), and work runs in turn. Don't invent an interim syntax.
- Include the permission-dependent rejection near 3304. Skip an optional candidate that cannot be established. The current "strategy does not update it" offload check alone does not prove there are no unseen writes.

### A6: delegated execution

- Keep the existing equivalent-execution obligations: associations and admission; computation and gradients under the numerical policy; dynamic decisions and observation order; state and external effects; authority and current-use dependencies; completion, failure and restoration.
- Put the shared delegation rule at workflow level. Training adds its own obligations.
- Keep **"Numerical choices are fixed for the run" in force** until explicitly changed. An open investigation does not authorize adaptive numerical choices. Check the effect on overlap and kernel selection.
- Preserve allowed behaviors and required within-run consistency, not necessarily one earlier execution. An approximation needs authored permission.

### A7: unchanged behavior

Module mode remains under D20. Don't say only the source of gradient facts changes. Preserve intended gradient behavior and review A7 with A4.

### Five conceptual categories

Include authored behavior, derived obligation, additional requirement, preference and delegated decision. Label them conceptual, not a schema or mandatory types. They may overlap; delegation concerns who decides, within the constraints. Don't require authors to restate derivable obligations.

### Open work

- **B1:** record the current lack of a general way to express execution constraints.
- **B8:** link `research/mock-a-execution-mechanism.md` and `docs_design/models-strategy-trainer/history/sessions/17-connected-candidate-and-ir-machinery.md` (about 2885-2895 and 6191-6204). The first session passage is an assistant proposal to defer supervisor versus cooperating flows; don't call it user agreement from that excerpt. Mock A calls its supervisor a candidate. D15 now specifies one coordinator per process. State which mechanisms remain open.
- Scheduling research can proceed independently of the overhead pass. Existing acceptance gates still apply. Identify a purpose before integrating an external runtime.
- **B9:** investigate intentional gradient boundaries generally, including the existing no-edge cut, not primarily `iro.train.fixed`.

## C2: separate discussion

Discuss the Trainer's host role before rewriting D17. Examine which fulfillment, preparation, coordination and lifecycle responsibilities are generic, while units, advancement and gradient meaning remain training-specific. This does not decide code placement. Data-only and filtering conformance cases with stand-ins provide limited evidence of generic execution, not production hosting.

## Implementation gaps

Keep the target design distinct from current support. Track code that still depends on old mechanisms in status notes or tasks, without presenting two competing target contracts. Check dependencies before applying any subset of A. Don't claim delegation works merely because its direction is written.

## Decisions for the user

1. **May correctness rely on an unchecked assertion?** The proposal says facts relied on for correctness are checked by the system. The review follow-ups allow supplied information or assertions with provenance. Choose one:
   - **No:** assertions can guide investigation, but an unverified one cannot establish correctness. Use a verified safe execution or report the limit.
   - **Yes, explicitly:** the run may rely on a stated assumption. Record its source and affected guarantees; don't report those guarantees as system-verified. Define who may accept it and any limits.
2. **Confirm A2 and A6** as intended direction, with the revisions above.
3. **C1 cautious handling:** may the system choose it without asking when it is established to cover every possibility of the unknown? This would change the earlier topic 1 wording. If no safe cover or accepted basis exists, report the limit.
4. **Confirm the C3 principle** above.

Keep these marked pending until the user answers.

## Return

Return a revised proposal with: the access-mode trace and lost-meaning test; target corrections; implementation gaps; open investigations; and pending decisions. Mark claims as observed, inferred, proposed or user-confirmed.
