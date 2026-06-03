## 1. Define the provenance model and vocabulary

- [ ] 1.1 Introduce a repo-owned attribution record shape separate from `ResourceMonitorFacts` for future debug/report-oriented ownership claims.
- [ ] 1.2 Define the initial evidence-level vocabulary, caveat model, and unknown/unattributed remainder semantics in the implementation-facing docs and tests.
- [ ] 1.3 Define the initial coarse runtime-owner categories the repo can support honestly without pretending to know allocator-internal ownership.

## 2. Add explicit attribution instrumentation seams

- [ ] 2.1 Add narrow repo-owned attribution window hooks around a first supported set of runtime operations such as startup/model loading, optimizer creation, accelerator preparation, or checkpoint save.
- [ ] 2.2 Capture before/after observational snapshots and supporting runtime-trace linkage for those windows without changing the always-on live resource schema.
- [ ] 2.3 Keep unsupported or ambiguous cases explicit by recording unknown remainder instead of forcing all deltas into named owners.

## 3. Surface attribution separately from live resource facts

- [ ] 3.1 Thread attribution records into benchmark-report/debug payloads as a distinct surface from the existing `resource_monitor` live facts.
- [ ] 3.2 Render evidence levels, caveats, and unknown remainder explicitly in human-facing summaries so users can tell observation from best-effort attribution.
- [ ] 3.3 Add focused verification that attribution output never reclassifies inferred owner claims as raw observed counters.

## 4. Revisit wider semantics after the first local path

- [ ] 4.1 Evaluate how the attribution model should adapt to distributed, offload, or multi-device paths after the initial local/single-process flow is credible.
- [ ] 4.2 Decide whether future attribution records should also file through observability metadata as a distinct fact type once the report/debug contract is stable.
