# Review: Pre-OpenSpec → Governing OpenSpec Transition Records

I verified the diff against the final state of all four focus files and cross-checked the surrounding `docs_design/` records for residual contradictory gating wording. Overall the transition is recorded coherently and consistently. Findings below are organized by the seven verification points, followed by residual issues.

## Verification of the Seven Claims

**(1) OpenSpec creation is the next step; production remains gated — ✅ consistent across all four files.**
- `notes.md` §"Governing OpenSpec transition decision (2026-08-27)": "The next step is to create one governing model–strategy–Trainer OpenSpec change" + "Implementation must not begin merely because the change exists."
- `strategy_contract_exchange_design.md` §"Scope And OpenSpec Design Gate": "The governing OpenSpec **may now be created**… Creating it does not declare the design implementation-ready… complete enough to **begin production work** only when:" — the gate was correctly re-targeted from change creation (old "complete enough to propose a governing change when") to production readiness.
- `strategy_system_direction.md` line ~1974: implementation tasks "must not be treated as ready until the four exchanges and optimization ownership are defined and pressure-tested."
- The three gate statements (notes / exchange / direction) are substantively aligned: four exchanges, standard + extension optimization boundaries, multi-family and imperative/research pressure cases, and migration milestones without dual current-state authorities or false temporary contracts.

**(2) Intended Trainer has authority; contract constrains strategies — ✅.**
- Direction's authority/evidence flow (`repository goals + current-code evidence → intended Trainer design → training contract → …`) and "It constrains authored filings; it does not force the Trainer to adopt behavior merely because a current strategy performs it."
- The main direction diagram was correctly rewritten to `PRE-EXISTING TRAINING CONTRACT ↓ guides authoring and defines acceptance`, and the mapping's target-path diagram now starts at `INTENDED TRAINER REQUIREMENTS + MAINTAINED CAPABILITY/FEATURE SURFACES`.
- The "contract establishment ≠ choosing/creating a contract" clarification is now stated in three places (direction ~1076, exchange §"Establish accepted run arrangement", mapping §"Target Production Path"), which closes the previous ambiguity. The EX-020 rewording ("active during establishment") is consistent.

**(3) Current code is evidence/pressure testing, not contract source — ✅.**
- Mapping: "SDXL is evidence about one maintained filing of the contract; it is not the source or shape of the contract itself"; `process_batch()` is "one maintained implementation case, not a template for the universal contract"; step 3 requires SD/SD3 and one strong imperative/research case to be compared so that "no one case defines the core."
- The preservation bullet was correctly widened from SDXL-only to "SD, SDXL, and SD3 … using SDXL only as the first detailed migration slice," and the exchange gate's scenario list was extended with custom-operation / imperative-region / research execution-authority cases. This fixes the previous SDXL-as-template skew.

**(4) Supporting roles, no competing authority — ✅ with two gaps (see findings).** Direction, exchange record, mapping, inventory, framework comparison, and spike all now declare subordinate/evidentiary roles with an explicit before/after-creation handoff (exchange §"Compaction Handoff" is particularly clear). The exchange record's "it must not become a parallel normative design after the OpenSpec exists" is the key sentence and is present.

**(5) One change ≠ big-bang — ✅.** Identical "several focused specifications / numbered milestones / vertical slices … not a big-bang implementation" statements in notes, exchange, and direction; mapping's artifact-job breakdown (`proposal/design/specs/tasks/tests`) and migration-slicing rules match.

**(6) Traceability chain — ✅ coherent.** The chain `intended Trainer responsibility -> contract requirement -> accepted-arrangement exchange -> representative scenario -> implementation milestone -> acceptance test` is stated identically in three places (direction, notes, mapping §"OpenSpec Design Dependency Order"), mapping step 10 requires tasks to reference the governing requirement *and* scenarios, and tests "prove the scenarios." Ordering is sound (scenario sits between exchange and milestone; milestone implements requirement+scenario; test verifies).

**(7) No active wording requiring all exchanges before creation — ✅ in the normative records; three residuals (below).** Repo-wide searches for `pre-OpenSpec`, `complete enough`, `four exchanges`, `propose a governing`, `Completion Gate`, `all exchanges` confirm the old creation gate is gone from the focus files; remaining `pre-OpenSpec` mentions all describe the transition itself.

## Findings (all low severity — documentation polish, no substantive contradiction)

1. **Stale sequencing entry in `notes.md` (~lines 929–932), unmarked as superseded.** The 2026-08-25 entry still says "Only after those [optimization and step exchanges] are coherent should the current-to-target implementation mapping and governing OpenSpec be derived from the full design." This directly contradicts the 2026-08-27 decision to create the change now. It is defensible under notes' declared "chronological, may contain superseded positions" role, and the mapping already exists despite it — but a one-line "(superseded by the transition decision of 2026-08-27 below)" would prevent a reader scanning for sequencing guidance from stopping at the wrong entry.

2. **`strategy_system_inventory.md:1074` — "Before an OpenSpec or production edits, discuss these in order:"** (Milestone 5 "Recommended Discussion Order"). This still gates OpenSpec creation on further discussion. The listed discussions were in fact subsequently held (Milestones 6–10 + the exchange design), and the inventory is declared evidentiary/subordinate — so it is not an active normative contradiction — but the sentence is now stale in the same way as finding 1 and is the only remaining line in the corpus that literally places something "before an OpenSpec." A small correction or completion note would close claim (7) fully.

3. **Ambiguous modifier in the mapping's design-sequence diagram (lines ~9–16).** `settled architecture + exchange semantics + current-code evidence -> create the governing OpenSpec` — it is unclear whether "settled" distributes over "exchange semantics." Read strictly, it implies exchange semantics must be settled before creation, which conflicts with the transition decision (optimization/step exchanges are deliberately *not* complete). Suggest "settled architecture + exchange semantics to date + current-code evidence" or "…+ exchange work so far." Note the neighboring prose (line ~38) gets this right by using "before … governing OpenSpec **design is finalized**," not "created."

4. **Role declarations omit `strategy_discussion.md` / `strategy_discussion_external.md`** (minor, claim 4 completeness). Both role lists (exchange §"Status And Authority", direction §"Transition To The Governing OpenSpec") enumerate inventory, mapping, notes, framework comparison, and the spike, but not the two raw transcript files, which have no status header of their own. They are evidently conversation logs and low-risk, but a one-line status note (raw transcript, superseded by the direction record) — or folding them under "chronological notes" in the lists — would make the "no competing authority" claim exhaustive.

5. **`notes.md` opens with a stale "Current continuation checkpoint (2026-07-30)"** whose "next discussion" pointers (binding/preparation) are long since completed. The final transition section corrects this, but since `notes.md` is the file most likely to be entered top-down (and has no top-of-file role statement of its own — its role is declared by sibling documents and only in its last section), consider a short header noting that entries are dated snapshots and the final section is current.

## Checks With No Issues

- No statement anywhere authorizes starting production implementation; every implementation-readiness statement is gated.
- The mapping's "combine" fate term is defined and used (launcher row) consistently with the added establishment clarification.
- The spike (`executable_spike/exchange.py`) carries an explicit isolation docstring and is referenced only as evidence.
- The direction's "Research Completed Before The Governing Change" retitle is coherent: the body items are framed as pre-implementation research, and the closing paragraph correctly routes further design into the change rather than extending exploration.
- Decision-marker semantics (`WORKING DECISION … refined before the corresponding OpenSpec design or requirement is finalized`) and the register note ("No working entry becomes an OpenSpec requirement merely because it appears in this table") correctly prevent the exchange record from self-promoting to spec authority.

**Verdict:** the transition is recorded accurately and satisfies all seven verification points; the five findings above are minor hygiene items, with findings 1–3 being the only ones that touch claim (7)'s letter and worth fixing before the OpenSpec change is created.

---
## Review Trace
- Workspace: /home/imi/Projects/sd-scripts
- Diff target: unstaged
- Context entries requested: 2
- Context files loaded: 2
- Files selected for initial diff: 4
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 39693 prompt tokens, 2536 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: search_repository
- Model call: API usage 42395 prompt tokens, 97 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 43858 prompt tokens, 731 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 46240 prompt tokens, 1219 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 48751 prompt tokens, 382 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 51470 prompt tokens, 266 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_files
- Model call: API usage 56688 prompt tokens, 577 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 58845 prompt tokens, 296 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 61240 prompt tokens, 4055 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 65673 prompt tokens, 231 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 66642 prompt tokens, 3275 completion tokens
- Review complete after 11 model call(s)