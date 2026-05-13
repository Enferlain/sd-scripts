# Adapter Layer Notes

This folder should stay small, generic, and boring.

## Ownership

- `library/adapters/` owns generic adapter runtime lookup/build concerns.
- Shared adapter target helpers consume the repo-level loaded-component
  contract from `library.models`; they should take `loaded_components` and
  preserve component provenance instead of accepting or reconstructing the old
  `text_encoders` / `vae` / `denoiser` bundle.
- It should not own family-specific config interpretation, method branch
  selection, legacy method normalization, or method-local validation rules.
- Family-specific resolution belongs under
  `library/adapters/methods/<family>/`.
- Method-specific behavior belongs under
  `library/adapters/methods/<family>/<method>/`.

## Runtime Boundary

- `AdapterModelContext` carries loaded top-level model components and exposes
  role-based projection helpers for adapter runtimes.
- Absorbed adapter runtimes may still need concrete diffusion-shaped arguments
  such as text encoder modules and denoiser modules at their edge, but that
  projection should happen at the mode/runtime boundary, not inside shared
  target expansion helpers.
- Generic adapter code should use declared component keys, public labels, roles,
  and capabilities. It should not infer behavior from placeholder names like
  `text_encoder1` or assume that every future model family has the same fixed
  component trio.
- Deprecated PEFT compatibility packages should be removed rather than kept as
  alternate runtime paths.

## Smell Test

If code under `library/adapters/` needs to know names like `lora`, `loha`,
`lokr`, or a family-specific config branch shape, that logic probably belongs
lower in the family or method layer instead.

If shared adapter code accepts both `loaded_components` and the old
`text_encoders` / `vae` / `denoiser` shape, it is probably rebuilding a
transition shim in the wrong layer.

## Notes

- `methods/` may get a different name in the future.
