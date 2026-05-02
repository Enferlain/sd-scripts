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
