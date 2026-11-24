**We should make the next milestone a matched comparison, with a fixed endpoint and explicit decision criteria.** That means filling the coverage gaps, aligning transformation policies, and comparing the machinery each implementation actually needs.

I would not finish the production language first. Both frameworks should still be allowed to inform that design.

**1. Agree on the comparison cases before writing more implementation.**

Use the cases already developed. Define their meaning independently of either candidate’s classes, syntax, or prepared-plan format.

| Case | What it tests | Remaining comparison work |
|---|---|---|
| B: independent production and feedback | Data-only work, feedback causality, admission, shutdown | Already present in both; retain as a control |
| A: correlation and accumulation | Multiple input associations, model-assisted preparation, requested evaluation | Implement the same case in Pliron |
| C: temporary optimization authority | Scoped transfer, restoration, verified handback, retained advancement | Implement the same case in Pliron |
| Mixed feedback/model ownership | Production reads a changing model; learning updates it; feedback affects selection | Implement the same case in Pliron |
| Independent payload/resource extension | Extensibility, typed boundaries, frozen resources, domain outcomes | Align grouping policy and analysis exercise |
| Different-lifetime composition | Pending requests, independent producer lifetime, separate group/run settlement | Implement the same case in xDSL |

This would establish comparable **semantic coverage**. It does not require copying every existing test into both packages.

The existing [comparison assessment](https://github.com/Enferlain/sd-scripts/blob/2f4c9353940d5f467e8fb7ceb2233fd88243baa4/docs_design/models-strategy-trainer/candidate_comparison.md) already provides much of the case and oracle inventory.

**2. Share expectations and execution services; keep the IR implementations independent.**

Reuse:

- Numerical fixtures and selected Python implementations.
- Existing owner engine and physical-worker services.
- Independent arithmetic and causal checks.
- Agreed positive and negative scenarios.

Allow each candidate its own operations, regions, interfaces, analyses, serialization, and lowering.

Avoid introducing a shared executable graph that already resolves all the interesting semantics before either framework sees it. That would mostly compare two encodings of decisions made elsewhere.

Each candidate should independently establish the relevant dependencies, resource requirements, lifetime obligations, and prepared correspondence. Compare permitted outcomes and causal relationships—not identical scheduler traces.

**3. Align one meaningful transformation exercise.**

The payload/resource extension is the best existing vehicle:

- Permit grouping through compatible typed stages and frozen reads.
- Preserve an opaque-effect barrier.
- Preserve every stage’s validation, outcomes, and physical completion.
- Reject a forged saved group crossing the barrier.
- Exercise analysis reuse and invalidation where applicable.

Currently the candidates use different grouping policies, so dispatch counts cannot fairly compare them. Align the intended behavior while allowing each framework its natural implementation.

I would leave SSA-changing rewriting outside this milestone unless we identify a transformation the actual system needs.

**4. Compare development and maintenance directly.**

For each matched case, record:

| Dimension | Evidence to examine |
|---|---|
| Semantic expression | Which obligations are explicit, and which remain hardcoded in profile logic? |
| Framework contribution | What verification, traversal, rewriting, equivalence, or analysis machinery is actually reused? |
| Extensibility | What existing modules must change to add a domain operation or resource contract? |
| Diagnostics | Can an invalid program identify the source, violated obligation, and expected/provided contracts? |
| Python integration | How are bindings installed, failures returned, and representations inspected? |
| Maintenance | How many separate places must change when one semantic contract changes? |
| Performance | Acceptance, verification, preparation, serialization, startup, and memory under matched workloads |

Keep **implementation cost, machinery ergonomics, and Python integration** as separate judgments. That lets us exclude initial implementation cost from the recommendation if desired, without hiding maintenance consequences.

Also distinguish framework effects from language and boundary choices. Pliron’s subprocess startup is a property of the current integration; Rust alone does not prove better performance or backend reuse.

**The immediate implementation order I recommend is:**

1. **Port the lifetime composition case to xDSL.**
2. **Port C’s temporary-authority case to Pliron.**
3. Align the resource-aware transformation exercise.
4. Complete Pliron’s A and mixed cases to close the remaining semantic coverage gaps.

The first two give us a useful comparison quickly: one case originating from each candidate, covering different architectural demands. The remaining work makes the broader comparison defensible.

End the milestone with a decision record: **preferred framework, reasons, material weaknesses, and conditions that would reverse the choice**. No new pressure cases should be added unless an implementation exposes a concrete unanswered question. That gives us a route to a decision rather than another open-ended feasibility phase.