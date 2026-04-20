## 1. Lock The Architecture

- [x] 1.1 Review the adapter-system proposal, design, and spec with the current repo direction and confirm the ownership split between strategy, optimization, adapter system, and `PeftMode`
- [x] 1.2 Identify the runtime concepts and adapter-to-optimization handoff that the implementation will use
- [x] 1.3 Decide the initial `library/adapters/` package shape and which parts of the package sketch should exist from the first implementation pass
- [x] 1.4 Decide the initial migration boundary between compatibility-shaped adapter surfaces and the new adapter-system architecture

## 2. Define The Adapter Runtime Path

- [x] 2.1 Create the initial `library/adapters/` scaffolding for the agreed package layout
- [x] 2.2 Introduce the repo-owned adapter runtime/framework surface that `PeftMode` will use for adapter training
- [x] 2.3 Rework the adapter path so adapter instantiation consumes resolved original-model targets instead of owning targeting policy
- [x] 2.4 Replace compatibility-era optimizer handoff behavior with a handoff that lets optimization remain the owner of grouping and scheduling

## 3. Rework Config And Persistence Direction

- [x] 3.1 Define the first adapter-type-specific config surfaces needed for the new adapter architecture
- [x] 3.2 Treat the current `PeftConfig` as a compatibility surface and route it toward the new adapter-type-specific direction
- [x] 3.3 Define the adapter-training save/load flow so `PeftMode` remains the training-side owner while adapter runtime behavior participates where needed

## 4. Migrate And Validate

- [x] 4.1 Migrate the current built-in adapter path onto the new adapter-system architecture without moving training-side ownership out of `PeftMode`, including aligning base-weight merge and inference-style setup with the same optimization-owned resolved-target handoff used by the main adapter training path
- [x] 4.2 Validate that optimization still owns original-model targeting and grouping behavior for adapter runs
- [x] 4.3 Add or update focused tests and design/docs coverage for the new adapter architecture and migration behavior
