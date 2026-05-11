## 1. Define the loaded-component contract

- [x] 1.1 Add the repo-owned loaded-component data model and family-declaration seam under `library/models/`
- [x] 1.2 Update current model families to declare their top-level loaded components, public labels, order, and generic roles/capabilities through the new seam
- [x] 1.3 Add focused tests proving SD, SDXL, and SD3 expose the expected declared component order and identity through the new contract

## 2. Replace the loader and trainer root contract

- [ ] 2.1 Change `ModelLoadingStrategy.load_target_model()` to return the loaded-component surface instead of the fixed `text_encoders/vae/denoiser` tuple
- [ ] 2.2 Update `Trainer.setup()` and trainer state to treat the loaded-component collection as the primary top-level model representation
- [ ] 2.3 Remove the old tuple-oriented assumptions from trainer-facing helpers that currently rebuild top-level component structure from `text_encoders`, `vae`, and `denoiser`

## 3. Migrate diagnostics, selectors, and tooling

- [ ] 3.1 Update training-mode diagnostics hooks and startup-summary builders to consume loaded components or filtered views of them
- [ ] 3.2 Update resource-monitor startup component accounting to consume the new component surface while preserving family-declared order
- [ ] 3.3 Update component-qualified selector generation and related tests to derive selector prefixes from declared loaded components
- [ ] 3.4 Update the model-inspection / parameter-dump tool to load and render top-level components through the new contract

## 4. Migrate optimization and adapter consumers

- [ ] 4.1 Update shared optimization target refs to derive top-level component identity from declared loaded components
- [ ] 4.2 Update fine-tune grouping and selection helpers to consume declared components instead of rebuilding the old trio-based component map
- [ ] 4.3 Update adapter runtime context and adapter target-resolution helpers to derive component scope from declared loaded components and their targeting-relevant semantics
- [ ] 4.4 Add focused tests covering non-trivial family shapes such as multiple same-role components and non-slot-based component ordering in targeting/grouping paths

## 5. Remove the old assumption and document the new boundary

- [ ] 5.1 Remove the remaining repo-owned `text_encoders/vae/denoiser` contract assumptions from affected runtime seams once all consumers are migrated
- [ ] 5.2 Update architecture docs, changelog, and any relevant roadmap/spec references to describe family-declared loaded components as the new top-level contract
- [ ] 5.3 Run focused verification for trainer setup, observability, selector tooling, optimization grouping, and adapter targeting to confirm the old tuple contract is no longer required
