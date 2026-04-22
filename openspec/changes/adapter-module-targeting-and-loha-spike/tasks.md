## 1. Module Targeting Foundation

- [x] 1.1 Extend the repo-owned adapter target payload and optimization-side adapter target selection so adapter runs can resolve concrete module targets instead of only component roots
- [x] 1.2 Keep the shared adapter target contract minimal and module-centered, including stable component/path provenance needed by adapter runtimes
- [x] 1.3 Add focused tests for optimization-owned adapter module targeting and resolved-target provenance

## 2. Repo-Native Loha Runtime

- [x] 2.1 Add a repo-native `loha` runtime under `library/adapters/methods/peft/loha/` that consumes `AdapterBuildRequest` and realizes adapter state only from resolved adapter targets
- [x] 2.2 Add repo-owned trainable-parameter-ref exposure for the `loha` runtime so optimization grouping continues to consume parameter-native refs with provenance
- [x] 2.3 Register `loha` as a repo-owned adapter method without making the vendor LyCORIS wrapper the repo contract

## 3. Loaded Runtime, Export, And Validation

- [x] 3.1 Add `loha` from-weights reconstruction and loaded-runtime merge behavior through the existing repo-owned `LoadedAdapterRuntime` and merge request seams
- [x] 3.2 Route `loha` export/save-load behavior through the existing repo-owned adapter persistence seams
- [x] 3.3 Add focused PEFT mode, runtime registry, and optimization-grouping coverage for the `loha` build, trainable-ref, from-weights, merge, and export flows
