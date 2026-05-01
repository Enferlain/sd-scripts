# Adapter Layer Notes

This folder should stay small, generic, and boring.

## Ownership

- `library/adapters/` owns generic adapter runtime lookup/build concerns.
- It should not own family-specific config interpretation, method branch
  selection, legacy method normalization, or method-local validation rules.
- Family-specific resolution belongs under
  `library/adapters/methods/<family>/`.
- Method-specific behavior belongs under
  `library/adapters/methods/<family>/<method>/`.

## Smell Test

If code under `library/adapters/` needs to know names like `lora`, `loha`,
`lokr`, or a family-specific config branch shape, that logic probably belongs
lower in the family or method layer instead.


## Notes

- methods folder will probably have adifferent name in the future