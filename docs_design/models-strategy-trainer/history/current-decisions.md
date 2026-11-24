# Carry-forward at the end of the historical walkthrough

This is a navigation and continuity record, not another specification. The
walkthrough reaches the last design discussion before this audit began on
5 November 2026. Its code frontier is `f984c41a`, the scoped-product experiment.
The [governing design](../../../openspec/changes/rework-model-strategy-trainer/design.md)
and delta specs retain authority under their existing carry-forward rule.

The earlier source documents now live in the
[historical archive](../../archive/models-strategy-trainer/README.md).
Their authority and bytes are preserved; the
[relocation map](archive-relocations.json) resolves original capture paths.

Choosing xDSL did not restart the design. The intended system's meaning governs
the representation, composition interfaces and runtime machinery. The work now
is to give that established system concrete production shapes, carrying its
requirements and the experiments' actual limitations forward.

## What remains settled

The notepad links retain the original rationale, adoption or correction, and
the consequence for G5. They should be read before interpreting an older sketch
as a current requirement.

| Subject | Carry forward | Decision trail and governing home |
| --- | --- | --- |
| System scope | Stateful, heterogeneous data and model workflows. Training is a demanding domain whose already researched requirements remain in scope; SD-shaped fixtures do not define the envelope. | [H061](decision-notepad.md#h061), [H078](decision-notepad.md#h078); design D1, D10, D14 and representative cases. |
| Contract and authoring | Intended consumers define the contract. Python authoring makes explicit selections and connections under that contract; fulfillment checks them using maintained shared knowledge. Ordinary users normally select/configure provided behavior. | [H001](decision-notepad.md#h001), [H007](decision-notepad.md#h007), [H031](decision-notepad.md#h031), [H046](decision-notepad.md#h046), [H081](decision-notepad.md#h081); design D1–D5, D7 and `training-contract`. |
| Accepted meaning | One accepted whole-run arrangement preserves selected behavior, policies, relationships and obligations. It outlives the authoring recipe and determines execution and coordination; it is more than a visualization. | [H040](decision-notepad.md#h040), [H049](decision-notepad.md#h049), [H059](decision-notepad.md#h059), [H062](decision-notepad.md#h062), [H072](decision-notepad.md#h072); design D4, D10 and `accepted-training-execution`. |
| Runtime representations | Semantic, normalized, prepared and runtime forms may differ, with checked correspondence. Preparation may specialize suitable work without creating another independently authored run or another source of semantic authority. | [H072](decision-notepad.md#h072), [H076](decision-notepad.md#h076), [H084](decision-notepad.md#h084); design D14 and `workflow-composition`. |
| State and identity | Participant and relationship identity, authored addresses, definition revisions, numerical versions, physical realizations and optimization progress are distinct. One participant-state authority coexists with narrower optimizer, backend, input, operation and capability owners. A composed run root does not grant every component unrestricted mutation. | [H021–H025](decision-notepad.md#h021), [H035](decision-notepad.md#h035), [H038](decision-notepad.md#h038), [H054](decision-notepad.md#h054), [H059](decision-notepad.md#h059); design D6–D9 and `run-participant-state`. |
| Preparation and change | One revisioned obligation model serves initial fulfillment, preparation and governed transitions. Acceptance and executable readiness differ. Fallible preparation precedes publication; destructive attempts withdraw affected guarantees, and stale results cannot restore them. | [H032](decision-notepad.md#h032), [H034](decision-notepad.md#h034), [H047](decision-notepad.md#h047); design D7–D8 and `training-runtime-preparation`. |
| Domain computation and optimization | Selected implementations own domain behavior. Semantic optimization units, contributions/windows, gradient destinations and per-unit outcomes are explicit; optimizer/backend objects are realizations. Unusual cadence alone does not transfer authority. Authorized imperative work needs explicit grants and handback. | [H018–H019](decision-notepad.md#h018), [H038–H041](decision-notepad.md#h038), [H064](decision-notepad.md#h064), [H067](decision-notepad.md#h067); design D9–D10 and `training-optimization`. |
| PEFT and products | Shared maintained PEFT composes with direct training without a replacement mode axis. Semantic adapter targeting, governed host resolution, method realization and optimizer parameter grouping have different owners. Artifact choice is independent of training treatment. | [H017](decision-notepad.md#h017), [H050–H052](decision-notepad.md#h050); design D12, `adapter-system`, `adapter-module-targeting` and persistence deltas. |
| Lifetimes and data | Independent producers, requested activities, admitted work and completion/stop relationships belong in inspectable contracts. Pending requested work keeps its enclosing group alive. Producing an item is distinct from admitting it to a consumer. Filtering an admitted item to zero outputs still requires an input disposition. | [H062–H064](decision-notepad.md#h062), [H073](decision-notepad.md#h073), [H083–H085](decision-notepad.md#h083); design D14 and `workflow-composition`. |
| Completion and failure | Call return, validation, logical completion, actual effects, physical settlement and later retained use differ. Cancellation closes admission at its declared scope while entered work and cleanup settle. Partial and uncertain outcomes survive; no universal rollback or atomic training step is promised. | [H027](decision-notepad.md#h027), [H063](decision-notepad.md#h063), [H067](decision-notepad.md#h067), [H086](decision-notepad.md#h086); execution, optimization and composition deltas. |
| Effects and transformations | Binding requirements, invocation authority, reported effects/outcomes and physical settlement remain separate. Shared declared effects and actual wiring justify legal transformations; unknown effects are conservative barriers. Purity alone does not erase failure or completion boundaries. Selected Python is trusted, not sandboxed by a declaration. | [H065](decision-notepad.md#h065), [H085](decision-notepad.md#h085); design D14, `training-contract` and `workflow-composition`. |
| Storage-dependent products | A result may retain storage/state or a derivative graph beyond its producing call. Dependencies must remain protected through dependent physical uses and release. The present scoped, read-only profile rejects escape and does not establish general alias discovery or cross-region ownership transfer. | [H086](decision-notepad.md#h086); [product contract and evidence limits](../_deprecated/xdsl_language/products.md), composition/preparation deltas. |
| Persistence and restoration | Semantic products/members/resources and actual output status differ from runtime restoration. Same-run restoration preserves required logical continuation, not necessarily bit-identical transient replay. Missing required facts leave restoration incomplete; creating a new run is a separate explicit request with new identity and lineage. | [H016](decision-notepad.md#h016), [H026–H027](decision-notepad.md#h026), [H068–H069](decision-notepad.md#h068); design D11 and artifact/capability deltas. |
| Machinery and backends | xDSL is the accepted working substrate; Python authoring is independent of that choice. Project semantics and runtime ownership remain ours. Multi-level lowering and MLIR-related opportunities matter, but the CPU experiments do not qualify native/GPU integration. Pliron remains credible; neither Pliron nor shared Rust dependencies mandate Burn. | [H075](decision-notepad.md#h075), [H077–H082](decision-notepad.md#h077); design D14. |
| Evidence | Independent reference semantics, numerical/causal expectations, corrupted-claim controls, cold artifacts, archive erasure and held physical work test different things. Frontends sharing owners are not independent ownership implementations. Historical provider failures remain failures, and measurements retain their environment and scope. | [H037](decision-notepad.md#h037), [H070](decision-notepad.md#h070), [H074](decision-notepad.md#h074), [H080](decision-notepad.md#h080), [H084](decision-notepad.md#h084); qualification records and design D13–D14. |

The user's latest concern about “a declared current set” of PEFT methods is
preserved under [H050](decision-notepad.md#h050). It does not establish a new
whitelist or maintenance mechanism. Whether the implementation makes ordinary
feature composition unnecessarily burdensome is deferred until there is a
concrete system to assess.

## What changed, rather than disappeared

| Earlier position or artifact | Later disposition |
| --- | --- |
| Active strategy as the ordinary runtime enclosure/peer | Superseded by accepted-run arrangement and narrower runtime owners. The same object may implement a narrow accepted role; authoring itself does not remain the runtime authority. H028, H040 and design D4/D7. |
| Ordinary participant-authored compatibility validators or transferable assessment authority | Superseded by contract-derived shared conformance and governed revision checks. Explicit new/custom semantic extension is a separate matter. H031 and the governing EX-007/015/017 dispositions. |
| `TrainingMode` as a permanent axis and owner | Responsibilities are distributed into training-subject intent, integration, preparation, capabilities and optimization. H017, H050–H052 and D12. |
| Missing restore facts could silently mean a new run | Explicitly corrected in October: restoration remains incomplete; a new run requires its own request. H069. |
| A model-family plugin catalogue, provisional package tree, surface mocks or custom-Python baseline determines the final architecture | These remain historical suggestions, evidence or sketches. Later adopted semantics and xDSL selection control their use. H060, H071–H072, H077–H082. |
| xDSL lacked regions/graph structure compared with Pliron | Corrected during the upstream capability discussion. Both had been demonstrated viable; the later working choice concerns machinery and lowering opportunities. H079–H082. |
| A clean review, passing fixture or completed experimental milestone makes production ready | These qualify only the stated scope. The connected-system prerequisite and G5 gate remain. H048, H053, H070–H071, H084. |
| “Small language” permits dropping prior distinctions | Rejected framing. Bounded implementation scope is useful evidence control; system meaning determines which distinctions must be represented. H087. |

Letters are overloaded in the history. October's surface mocks A/B/C, later
execution pressure cases A/B/C, and the external report's architecture categories
are different naming schemes. A rejection of one mock is not rejection of the
later pressure case with the same letter.

## What the last five chapters establish

| Historical phase | Result and boundary |
| --- | --- |
| [15: research and whole-run language](sessions/15-research-and-whole-run-language.md) | Broader recipes expose independent activities, stateful inputs, structured/adaptive computation and non-universal optimization cadence. The hierarchical whole-run direction is selected; the first executable experiments remain probes. |
| [16: exchange design, audits and G5 correction](sessions/16-exchanges-audits-and-g5-correction.md) | G3/G4 specify optimization, capability and recovery exchanges. Audits reconcile real contradictions. The user then requires an assembled concrete execution mechanism before following the inherited G5 list mechanically. |
| [17: connected candidate and IR investigation](sessions/17-connected-candidate-and-ir-machinery.md) | A project-owned candidate derives execution and coordination from the accepted graph and develops shared owners through pressure cases. Reference learning and independent checks inform it. Its asyncio policy, Python storage shapes and numerical targets remain implementation evidence. |
| [18: matched frameworks and xDSL choice](sessions/18-framework-comparison-and-xdsl-choice.md) | xDSL and Pliron 0.20.0 both demonstrate credible IR machinery. Matched comparisons disclose shared runtime and cost boundaries. On 2 November, the user accepts xDSL as working substrate, with a specific reason to reconsider if an actual obstruction appears. |
| [19: shared language to retained products](sessions/19-shared-language-and-current-frontier.md) | D14 adopts bounded contracts and explicit amendments to supplied research. Shared verification, independent reference, owner preparation, effects/filtering/fusion and retained products advance reusable machinery without selecting production APIs or completing G5. |

## What is still concrete design or implementation work

The [current G5 tasks](../../../openspec/changes/rework-model-strategy-trainer/tasks.md)
remain unchecked. Experimental milestones 6–11 feed production choices; they
are not the production migration. The remaining work includes concrete identity,
authority, binding, preparation, optimization and publication types; the complete
accepted/prepared/runtime model; the contract and Python authoring boundary;
module/dependency placement; and traced vertical migration milestones with
displaced authority, acceptance evidence and review boundaries.

Some research implementation limits remain separate: general cross-region/flow
retained-product transfer, mutable aliases, broader language composition and
targets, distributed/device protocols, and bounded retirement of historical
bookkeeping. These are not reasons to redo the architecture survey, and they
are not all prerequisites for every G5 design task. The governing G5 text
explicitly allows 5.1/5.2 to develop independently, with reconciliation before
production placement and migration.

This historical side quest changes none of those statuses. The
[placement plan](placement-plan.md) organizes existing records; it does not
choose the production package tree or begin the main implementation.
