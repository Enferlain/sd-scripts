## 1. Reconcile Resource-System Direction

- [x] 1.1 Review the new resource-intelligence fact classes against every current resource-monitor event, metadata fact, startup estimate, and report-derived value.
- [x] 1.2 Record the disposition of the historical `phase-resource-accounting` change and migrate any still-valid requirements into this change.
- [x] 1.3 Update the active resource-system bead and roadmap language so this OpenSpec change is the governing design.

## 2. Define Canonical Resource Facts

- [x] 2.1 Add accepted metadata dataclasses for resource observations with explicit measurement semantics, scope, source, and runtime identity.
- [x] 2.2 Add accepted metadata dataclasses for structural resource facts with owner identity and estimation/measurement basis.
- [x] 2.3 Add accepted metadata dataclasses for versioned resource profiles with source-run/fact relationships.
- [x] 2.4 Add accepted metadata dataclasses for evidence-constrained accounting statements and explicit accounting gaps.
- [x] 2.5 Add central validation and emitter routing for the resource fact catalog.
- [x] 2.6 Add focused metadata tests proving resource fact classes, relationships, and semantic distinctions survive filing and storage.

## 3. Support Telemetry-Volume Metadata Ingestion

- [x] 3.1 Measure current bundled-event, scalar-observation, and observation-frame `MetadataRuntime` overhead and memory growth for representative sampled resource runs.
- [x] 3.2 Define and implement accepted observation-frame/batch plus `MetadataRuntime` buffer ingestion for sampled resource facts without exposing backend/storage calls to resource-domain code.
- [x] 3.3 Define retention and drop/degraded-observability policies for sampled resource facts.
- [x] 3.4 Verify sampled ingestion cannot block training indefinitely and that dropped/degraded telemetry is recorded.
- [x] 3.5 Add SQLite-storage and in-memory-backend tests for representative resource telemetry volumes.

## 4. Introduce Resource-Domain Fact Production

- [x] 4.1 Add a resource-domain fact-production seam behind the existing `ResourceMonitor` lifecycle facade.
- [x] 4.2 Convert current session, phase, step, sampled, and deep-counter paths to produce canonical typed observation frames once.
- [x] 4.3 Project the existing resource JSONL shape from canonical facts as a compatibility export.
- [x] 4.4 Preserve current console summaries from canonical facts or resource-domain summaries without making console output canonical.
- [x] 4.5 Add end-to-end equivalence tests for current JSONL, metadata, and console behavior during migration.

## 5. Add Resource Identity And Query Views

- [x] 5.1 Define resource relationships for runs, processes/ranks, hosts, devices, phases/events, steps, components/groups, collectors, and artifacts.
- [x] 5.2 Implement a resource-run query/view interface over metadata-owned public snapshot/query/projection APIs without exposing raw storage.
- [x] 5.3 Prove the resource-run view preserves original observations alongside profiles and accounting statements.
- [x] 5.4 Add query tests for multi-rank, multi-device, phase, component, and artifact-linked facts.

## 6. Migrate Reports And Exports

- [x] 6.1 Move benchmark report resource input from canonical JSONL parsing to the resource-run view.
- [x] 6.2 Preserve existing report resource summaries and per-device/debug output during migration.
- [x] 6.3 Implement metadata-owned projections for JSONL, report, profile, and accounting export schemas, then register produced artifacts with links to their source run.
- [x] 6.4 Remove duplicate hand-authored resource schema assembly after compatibility and performance verification.

## 7. Decompose Collection And Operational Policy

Implementation note: this section should be handled as a vertical collector
slice, not as unused contract scaffolding. The contracts, extracted collectors,
mode policy, and tests should preserve collection-cycle truth, exact fallback
provenance, rank/device scope separation, degradation records, and diagnostic
window side effects while keeping existing user-facing behavior stable.

- [x] 7.1 Define collector capability, cost, scope, cadence, availability, and degraded-behavior contracts.
- [ ] 7.2 Extract current process-memory, CUDA allocator, sampled GPU-used, and deep allocator collection behind those contracts.
- [x] 7.3 Map current `off`, `basic`, `sampled`, and `deep` configuration onto collector/cost policy without changing user behavior.
- [ ] 7.4 Add collector failure, fallback, budget, and bounded-diagnostic-window tests.
- [ ] 7.5 Evaluate additional collectors only against explicit resource questions and retention policy.

## 8. Produce Structural Facts And Profiles

- [ ] 8.1 Migrate startup component parameter/trainable estimates into canonical structural resource facts.
- [ ] 8.2 Add structural facts for optimizer, gradient, cache, worker, or artifact state only where the owning domain can provide trustworthy sizes.
- [ ] 8.3 Define and implement the first versioned run resource profile from accepted observations and structural facts.
- [ ] 8.4 Add comparison/regression queries and tests over stored profiles.
- [ ] 8.5 Render report profile views as projections of durable profiles rather than report-only calculations.

## 9. Implement Evidence-Constrained Accounting

- [ ] 9.1 Define the first trustworthy accounting questions and the accepted source facts required to answer them.
- [ ] 9.2 Implement structural accounting statements for known resource-bearing state.
- [ ] 9.3 Implement explicitly bounded operation/window accounting only where runtime evidence supports it.
- [ ] 9.4 Calculate and preserve accounting gaps without assigning them to fallback owners.
- [ ] 9.5 Add report/query views that present observed, profiled, accounted, and gap values without conflating them.
- [ ] 9.6 Add tests preventing unsupported phase-delta or operation-local observations from becoming persistent ownership claims.

## 10. Validate And Consolidate

- [ ] 10.1 Define migration acceptance thresholds, then benchmark runtime overhead, ingestion throughput, memory growth, query performance, and artifact size across collection policies.
- [ ] 10.2 Verify typed facts remain authoritative during migration and that compatibility JSONL/report outputs remain equivalent within declared acceptance criteria.
- [ ] 10.3 Verify profile/accounting source relationships and derivation versions fail validation when required evidence is missing.
- [ ] 10.4 Verify failure/OOM cleanup preserves the latest accepted resource context without masking the original failure.
- [ ] 10.5 Update resource, metadata, observability, and report documentation with the settled ownership model.
- [ ] 10.6 Update changelog and roadmap, archive superseded resource design artifacts, and close completed beads after verification.
