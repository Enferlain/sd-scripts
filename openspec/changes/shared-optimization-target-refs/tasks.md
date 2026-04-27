## 1. Shared Target Model

- [x] 1.1 Add `library/optimization/targets.py` with shared target-ref types for component, module, and parameter targets, leaving `library/adapters/runtime/targets.py` as the adapter-facing target construction layer
- [x] 1.2 Add helper constructors for component, module, and parameter target refs using component-qualified selectors
- [x] 1.3 Add owner-module path/type resolution for parameter targets
- [x] 1.4 Add focused unit coverage for target-ref construction, selector names, module type, and owner-module provenance

## 2. Fine-Tune Path Integration

- [x] 2.1 Update `NamedParameterRef` or its replacement in `library/optimization/grouping.py` to carry parameter target refs
- [x] 2.2 Update `_collect_component_named_parameters(...)` to build shared parameter target refs without changing current selector strings
- [x] 2.3 Keep `resolve_finetune_selection(...)` and `build_finetune_grouping(...)` behavior stable while consuming the shared target provenance internally
- [x] 2.4 Extend focused fine-tune grouping tests to assert parameter target selectors and owner-module provenance

## 3. Adapter Target Integration

- [x] 3.1 Update adapter module target construction in `library/adapters/runtime/targets.py` to build shared module target refs with module type provenance
- [x] 3.2 Keep existing `AdapterResolvedTarget` fields available while adding or wrapping the shared target ref
- [x] 3.3 Update `AdapterTrainableParameterRef` to preserve source target provenance for adapter-created trainables
- [x] 3.4 Update repo-owned LoHa trainable-ref exposure to propagate the source module target ref
- [x] 3.5 Update the LoRA wrapper path where feasible, while keeping compatibility behavior explicit if exact module provenance cannot be recovered from the legacy wrapper

## 4. Grouping Behavior Preservation

- [x] 4.1 Verify fine-tune learning-rate group matching still uses existing component-qualified parameter selector strings
- [x] 4.2 Verify adapter grouping still groups returned trainable refs by component learning-rate policy
- [x] 4.3 Add adapter grouping coverage for source target module type provenance without introducing adapter `groups` support yet
- [x] 4.4 Keep `PeftMode` and `FineTuneMode` responsible only for lifecycle sequencing around target selection, trainable realization, and grouping

## 5. Documentation And Validation

- [x] 5.1 Update `docs_design/adapter_system_overview.md` and/or `docs_design/optimization_layer_plan.md` to describe shared target refs across components, modules, and parameters
- [x] 5.2 Update `CHANGELOG.md` and `ROADMAP.md` with the target-ref foundation and remaining selector/grouping follow-up
- [x] 5.3 Run focused tests for optimization grouping, adapter runtime registry, PEFT mode, and model parameter dump helpers
- [x] 5.4 Run lint on touched optimization, adapter, docs-adjacent test files
