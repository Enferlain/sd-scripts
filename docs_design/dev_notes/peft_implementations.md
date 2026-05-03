# PEFT Implementation Notes

This file is for implementation-specific notes, one-off observations, and
method-local commentary that may be useful later but should not shape the
general "how to implement future PEFT methods" guidance in
`library/adapters/methods/peft/README.md`.

## LoHa / LoKr / LoCon

- Plain `dropout` remains intentionally unsupported for LoHa/LoKr because the
  absorbed LyCORIS behavior did not apply it consistently in rebuilt-weight
  mode. Keep `rank_dropout` and `module_dropout` as the active supported knobs
  unless that implementation direction changes later.
- LoCon is the counterexample that makes the dropout rule method-local rather
  than family-global: its absorbed behavior still uses plain `dropout`, but
  only in bypass-mode output application and the DoRA input path.
- Scalar-mode export currently folds `scalar` into the first exported factor
  and resets scalar to identity on load.
- LoKr full-matrix mode is not just a layout toggle: it uses unit scaling by
  overriding alpha to rank, matching the absorbed LyCORIS intent that fully
  materialized Kronecker factors should not get an extra rank scale.

## OFT

- OFT is the clearest example so far that not every absorbed method wants a
  fake LoRA-shaped `rank + alpha` surface. The repo-owned OFT path uses
  method-local `factor` and `constraint` names because those are the real
  algorithm knobs in the LyCORIS-derived implementation.
- Repo-owned OFT intentionally absorbs the LyCORIS Diag-OFT direction, not the
  older built-in `oft_deprecated` path. Treat `oft_deprecated` as dead code
  waiting for deletion, not as a compatibility contract to preserve.
- The absorbed OFT bypass path needed real fixes before it was safe to own:
  the vendor/legacy logic had a broken diff branch and a rescale shape bug.
- OFT plain `dropout` should stay a block-transform concern. Reapplying a
  second output-shaped mask in bypass mode just recreates the broken vendor
  broadcast path and makes the semantics harder to reason about.
- Repo-owned OFT now has a good reason to prefer a compact native parameter
  layout over the older full-square block storage: the method is known to be
  heavy, and the skew-symmetric blocks only need the independent upper-triangle
  values to reconstruct the effective transform.

## BOFT

- BOFT should keep its own method-local factorization language instead of being
  flattened into a fake LoRA-style rank surface. The active knob is still a
  LyCORIS-style factorization hint, and the runtime/module layer owns the
  butterfly-specific block/stage derivation from there.
- BOFT plain `dropout` is a real method-local semantic, but it belongs on the
  butterfly transforms themselves by replacing selected stage/block matrices
  with identity during training. Do not expose a fake `rank_dropout` surface
  just because the vendor inheritance chain accepts the argument; the absorbed
  BOFT path does not use it meaningfully.
- Repo-owned BOFT should start with the compact upper-triangle storage choice
  from day one instead of inheriting the older full-square vendor layout.
  Like OFT, the effective skew-symmetric blocks only need their independent
  upper-triangle values, and BOFT is also known to be a comparatively heavy
  method.
- The repo-owned BOFT surface now has an explicit partial butterfly-depth knob
  through `num_stages`. Keep that as a method-local BOFT concept layered on top
  of the factorized block layout instead of forcing the repo config to mirror
  Hugging Face's `boft_n_butterfly_factor` name exactly.
- The `FastBlockDiag` idea from Hugging Face PEFT is currently not a strong fit
  for this repo-owned BOFT implementation because our path already applies
  butterfly stages directly to weights/output-axis tensors without materializing
  full block-diagonal matrices. Revisit only if a future BOFT implementation
  starts assembling explicit block-diagonal operators again or profiling shows a
  different hotspot than the current direct stage application.

## DyLoRA

- DyLoRA is better treated as a LoRA-shaped training trick than as a separate
  factorization family. The repo-owned path keeps the surface small: `rank`,
  `alpha`, `block_size`, optional `module_dropout`, and optional
  `bypass_mode`.
- The absorbed vendor DyLoRA path had several real rough edges: `get_weight()`
  rebuilt factors from `.data`, `load_state_dict()` was effectively disabled,
  and the bypass path had a `scale` / `gamma` mix-up. The repo-owned version
  should stay explicit and test those semantics directly rather than assuming
  the vendor path is already a safe contract.
- Repo-owned DyLoRA exports standard LoRA-shaped `lora_up.weight` /
  `lora_down.weight` tensors plus an explicit `block_size` tensor so the
  dynamic training layout round-trips exactly without forcing the main adapter
  artifact shape to become DyLoRA-specific.
- The current repo-owned DyLoRA scaling policy keeps full-rank export/merge on
  normal `alpha / rank` semantics while scaling sampled training prefixes by
  `sqrt(full_rank / active_rank)` so smaller active prefixes do not collapse in
  magnitude as aggressively as plain truncated LoRA updates.
- Behavior difference versus the current LyCORIS DyLoRA is worth remembering:
  LyCORIS samples a random prefix and scales it by active block count, but its
  current rebuild path concatenates `.data` tensors and its local
  `load_state_dict()` path is effectively disabled. The repo-owned DyLoRA path
  instead rebuilds the sampled prefix from real parameters, explicitly trains
  only the sampled block while still using the whole prefix for the effective
  update, keeps full-rank merge/export on standard LoRA-shaped weights, and
  saves `block_size` alongside `lora_up.weight` / `lora_down.weight` / `alpha`
  so the dynamic training layout round-trips exactly.

## GLoRA

- GLoRA is another reminder that not every LyCORIS method should be flattened
  into plain LoRA terms. The repo-owned path keeps the actual method shape:
  branch A contributes `W A`, branch B contributes `B`, and bypass mode applies
  `W(X + A(X)) + B(X)` directly instead of pretending the method is only one
  low-rank branch.
- The absorbed vendor GLoRA path had at least two important rough edges worth
  remembering: the Tucker branch on the B path was effectively unreachable as
  written, and bypass-mode scaling had a mismatch where local multiplier
  handling did not cleanly line up with the rebuilt-weight path.
- Repo-owned GLoRA keeps plain `dropout` as a real method-local semantic, but
  currently preserves the unusual LyCORIS behavior where non-bypass mode drops
  the forward input rather than the rebuilt branch weights themselves. That is
  worth revisiting later if rebuilt-weight GLoRA becomes a serious workflow.

## IA3

- IA3 is another method that should keep its own real config language instead
  of being squeezed into fake `rank` / `alpha` knobs. The useful user-facing
  choices are axis selection (`train_on_input`), optional module dropout, and
  optional bypass mode.
- The vendor IA3 path had two correctness hazards worth fixing in the
  repo-owned version: output-side scaling left the original bias untouched in
  merged-weight mode, and state-dict reconstruction ignored the saved
  `on_input` flag even though that changes the meaning of the learned scale
  tensor.
- The LyCORIS IA3 preset is worth remembering because it is not globally
  uniform: attention `k_proj` / `v_proj` use output-side scaling, while
  feed-forward `fc2` / `ff.net.2` use input-side scaling. The current
  repo-owned IA3 path now auto-selects that axis per target by the current
  IA3 target-name patterns when `train_on_input` is left unset, while still
  allowing an explicit global override when needed.
- Legacy or compatibility checkpoints that omit `on_input` are only safely
  inferable when input and output dimensions differ. Square layers need the
  saved axis flag to avoid ambiguous reconstruction.

## ABBA

- ABBA is still most naturally exposed through one repo-owned `rank` plus the
  LyCORIS-style half-and-half split into `r1` / `r2`, rather than widening the
  active config surface to separate pair ranks before there is a real user
  need for that.
- The absorbed vendor ABBA path had several correctness hazards worth fixing in
  the repo-owned version: the linear bypass path cached Khatri-Rao factors
  without any invalidation after parameter updates, the convolution bypass diff
  path added the original bias a second time, and the non-DoRA merged-weight
  path multiplied the adapter contribution by `multiplier` twice.
- Repo-owned ABBA keeps plain `dropout` as a real method-local semantic like
  the absorbed LyCORIS path, but it should stay scoped to the forward delta
  output or the weight-decompose input path rather than being reinvented as a
  fake rank-only knob.
- Export should not depend on `sqrt(scalar)` symmetry tricks. Folding scalar
  into one exported factor and resetting runtime scalar to identity on load is
  simpler and avoids invalid values if training drives the scalar negative.

## TLora

- TLora is another method that should keep its own real config language
  instead of pretending it is just "LoRA plus one extra flag." The useful
  user-facing choices are still `rank` / `alpha`, but the method-specific
  behavior also needs singular-vector selection, data-vs-random SVD init, and
  timestep-mask schedule controls such as `min_rank` and `mask_alpha`.
- The absorbed vendor TLora path had one especially important architecture
  hazard: timestep masks lived in a module-global singleton that training code
  had to mutate out-of-band before forward. The current repo-owned TLora slice
  intentionally does not solve that by quietly editing unrelated training or
  strategy layers. Instead it keeps the mask helpers method-local, documents
  the missing integration seam explicitly, and leaves proper repo-owned
  timestep-mask wiring as separate follow-up design work.
- Per-sample timestep masks are still the interesting TLora behavior, but they
  also mean rebuilt-weight mode cannot represent the whole batch with one
  merged delta. The repo-owned module still fails clearly if a caller tries to
  collapse a batched mask into one diff weight for merge-style math.
- Plain `dropout` remains a real method-local semantic for TLora, but it is
  only meaningful on the active bypass delta path. Until the repo has a proper
  timestep-mask seam, treat TLora's schedule fields as documented method-local
  intent rather than a guarantee that the shared training path is driving them.
- Reference points for the missing wiring are worth keeping explicit:
  `library/vendor/lycoris/lycoris/modules/tlora.py` defines the original mask
  helpers and global-state contract, while
  `library/vendor/lycoris/docs/Algo-Details.md` and
  `library/vendor/lycoris/docs/Network-Args.md` describe the intended training
  behavior and config knobs.
