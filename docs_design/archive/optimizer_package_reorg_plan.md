# Optimizer Package Reorg Plan

This note is the working guide for reorganizing `library/optimization/optimizers/` with PyCharm refactor tools.

The goal is:

- reduce the number of giant top-level optimizer files
- group obvious optimizer families together
- keep one public optimizer class per file where practical
- avoid backend-based buckets like `torchao/`, `schedulefree/`, or `8bit/`
- preserve the shared registry/factory surface while moving files around

## Ground Rules

- Organize by optimizer family first, backend variant second.
- Keep `AO`, `8bit`, `bnb`, `schedulefree`, `v2`, `v3`, and `exmachina` variants inside the family package they belong to.
- Keep truly standalone optimizers as top-level files.
- Do not introduce a generic `misc/` or `experimental/` package.
- Leave `library/optimization/wrappers/` as its own layer. Do not move wrapper-style optimizers like `ScheduleFreeWrapper`, `CPUOffloadOptimizerWrapper`, or `SNOOASGD` into `optimizers/`.
- When in doubt, prefer a small family package over another 1000+ line module.

## Recommended End State

This is the proposed target layout after the package reorg settles.

```text
library/optimization/optimizers/
  README.md
  __init__.py

  adopt/
    __init__.py
    adopt.py
    adoptmars.py
    fadoptmars.py
    schedulefree.py
    schedulefree_ao.py

  ademamix/
    __init__.py
    ademamix.py
    simplified.py
    simplified_exm.py

  adamw/
    __init__.py
    low_bit_ao.py
    kahan_8bit.py

  compass/
    __init__.py
    compass.py
    compass_plus.py
    compass_adopt.py
    compass_adoptmars.py
    compass_ao.py
    compass_8bit_bnb.py
    fcompass.py
    fcompass_plus.py
    fcompass_adopt.py
    fcompass_adoptmars.py

  farmscrop/
    __init__.py
    farmscrop.py
    farmscrop_v2.py

  fmarscrop/
    __init__.py
    fmarscrop.py
    fmarscrop_v2.py
    fmarscrop_v2_exmachina.py
    fmarscrop_v3.py
    fmarscrop_v3_exmachina.py

  rmsprop/
    __init__.py
    rmsprop.py
    rmsprop_adopt.py
    rmsprop_adoptmars.py

  scorn/
    __init__.py
    scorn.py
    scornmachina.py

  abmog.py
  adabelief.py
  adai.py
  adammini.py
  adan.py
  alice.py
  bcos.py
  came.py
  cstableadamw.py
  dehaze.py
  fftdescent.py
  fira.py
  fishmonger.py
  galore.py
  glyph.py
  gooddog.py
  grokfast.py
  lamb.py
  laprop.py
  lpf_adamw.py
  momentus_caution.py
  mythical.py
  oagopt.py
  ocgopt.py
  projective_adam.py
  racs.py
  ranger21.py
  remaster.py
  scgopt.py
  scion.py
  sgd_sai.py
  shampoo.py
  singstate.py
  soap.py
  spam.py
  talon.py
  vsgd.py
  wiwiopt.py

  utils/
    ...
```

## Families To Package

These are the packages worth creating because the family boundaries are already real.

### `adopt/`

Move these here:

- `adopt.py`
- `adopt_schedulefree.py`
- `adopt_schedulefree_ao.py`

Target classes:

- `ADOPT`
- `ADOPTMARS`
- `FADOPTMARS`
- `ADOPTEMAMixScheduleFree`
- `ADOPTMARSScheduleFree`
- `ADOPTNesterovScheduleFree`
- `ADOPTScheduleFree`
- `FADOPTEMAMixScheduleFree`
- `FADOPTMARSScheduleFree`
- `FADOPTNesterovScheduleFree`
- `FADOPTScheduleFree`
- `ADOPTAOScheduleFree`

Note:

- This family is the strongest candidate for a real package because it mixes plain, schedulefree, and AO variants of the same core family.

### `ademamix/`

Move these here:

- `ademamix.py`

Target classes:

- `AdEMAMix`
- `SimplifiedAdEMAMix`
- `SimplifiedAdEMAMixExM`

Note:

- This can start as one file moved under the package and be split later, or be split immediately into three files.

### `adamw/`

Move these here:

- `adamw_low_bit.py`
- `adamw_8bit_kahan.py`

Target classes:

- `AdamW4bitAO`
- `AdamW8bitAO`
- `AdamWfp8AO`
- `AdamW8bitKahan`

Note:

- `AdamMini` stays flat. It is Adam-related, but not a normal `AdamW` family variant.

### `compass/`

Move these here:

- `compass.py`
- `fcompass.py`

Target classes:

- `Compass`
- `CompassPlus`
- `CompassADOPT`
- `CompassADOPTMARS`
- `CompassAO`
- `Compass8BitBNB`
- `FCompass`
- `FCompassPlus`
- `FCompassADOPT`
- `FCompassADOPTMARS`

Note:

- This is the highest-value structural cleanup because the current family is the most obviously oversized.

### `farmscrop/`

Move these here:

- `farmscrop.py`
- `farmscrop_v2.py`

### `fmarscrop/`

Move these here:

- `fmarscrop.py`
- `fmarscrop_v2.py`
- `fmarscrop_v2_exmachina.py`
- `fmarscrop_v3.py`
- `fmarscrop_v3_exmachina.py`

### `rmsprop/`

Move these here:

- `rmsprop.py`

Target classes:

- `RMSProp`
- `RMSPropADOPT`
- `RMSPropADOPTMARS`

### `scorn/`

Move these here:

- `scorn.py`
- `scornmachina.py`

## Optimizers That Should Stay Flat

These should remain top-level files unless a new strong family pattern appears later.

- `abmog.py`
- `adabelief.py`
- `adai.py`
- `adammini.py`
- `adan.py`
- `alice.py`
- `bcos.py`
- `came.py`
- `cstableadamw.py`
- `dehaze.py`
- `fftdescent.py`
- `fira.py`
- `fishmonger.py`
- `galore.py`
- `glyph.py`
- `gooddog.py`
- `grokfast.py`
- `lamb.py`
- `laprop.py`
- `lpf_adamw.py`
- `momentus_caution.py`
- `mythical.py`
- `oagopt.py`
- `ocgopt.py`
- `projective_adam.py`
- `racs.py`
- `ranger21.py`
- `remaster.py`
- `scgopt.py`
- `scion.py`
- `sgd_sai.py`
- `shampoo.py`
- `singstate.py`
- `soap.py`
- `spam.py`
- `talon.py`
- `vsgd.py`
- `wiwiopt.py`

## Recommended Refactor Order

Do not move everything in one pass. Use this order so the highest-value cleanup lands first and the risk stays manageable.

### Pass 1: Big obvious families

1. `compass/`
2. `adopt/`
3. `fmarscrop/`

### Pass 2: Medium families

1. `ademamix/`
2. `adamw/`
3. `rmsprop/`

### Pass 3: Small families

1. `farmscrop/`
2. `scorn/`

## Two-Phase Strategy For Each Family

Use this pattern to reduce risk.

### Phase A: Package move only

Move the current file into a family package without splitting every class yet.

Example:

- move `compass.py` to `optimizers/compass/compass.py`
- move `fcompass.py` to `optimizers/compass/fcompass.py`
- add `optimizers/compass/__init__.py`
- update imports/registry/tests

This gets the package structure in place quickly.

### Phase B: Split by public optimizer class

After the family package is stable, split the big modules into one-class-per-file where that improves readability.

Example:

- `compass/compass.py` -> `compass.py`, `compass_plus.py`, `compass_adopt.py`, `compass_adoptmars.py`, `compass_ao.py`, `compass_8bit_bnb.py`

Use this second phase only when it clearly improves maintainability.

## PyCharm Refactor Workflow

Use PyCharm move refactor rather than manual file moves whenever possible.

For each family:

1. Create the package directory with `__init__.py`.
2. Use PyCharm `Refactor -> Move` on the source file into that package.
3. Fix the package `__init__.py` so it re-exports the public classes.
4. Update [optimizers/__init__.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/__init__.py) imports to point at the new module paths.
5. Update [registry.py](/mnt/d/Projects/sd-scripts/library/optimization/registry.py) target strings to the new module paths.
6. Run focused tests before doing more moves.
7. Only then split multi-class files further inside the package.

## Invariants To Preserve

These are the things that must keep working after every move.

- Public optimizer names must not change.
- Registry keys must not change.
- Registry target strings must point to the new module paths.
- `library.optimization.optimizers.__all__` must still expose the same public classes.
- Existing config names like `CompassAO`, `ADOPTScheduleFree`, `FMARSCropV2ExMachina`, and `AdamW8bitKahan` must still resolve through the shared factory.
- Wrapper-style optimizers stay under `library/optimization/wrappers/`.

## After-Move Checklist

Run this after each package move.

### Code updates

- update `library/optimization/optimizers/__init__.py`
- update `library/optimization/registry.py`
- update any intra-family imports
- update doc references if a path is mentioned directly in `README.md`, `ROADMAP.md`, or `CHANGELOG.md`

### Verification

Use focused checks rather than full-suite runs after each family move.

```bash
uv run ruff check <moved files> library/optimization/optimizers/__init__.py library/optimization/registry.py
uv run pytest tests/unit/optimizers/test_registry.py -v -k "<family names>"
uv run pytest tests/unit/optimizers/test_absorbed_integrations.py -v -k "<family names>"
```

Examples:

```bash
uv run pytest tests/unit/optimizers/test_absorbed_integrations.py -v -k "Compass"
uv run pytest tests/unit/optimizers/test_absorbed_integrations.py -v -k "ADOPT"
uv run pytest tests/unit/optimizers/test_absorbed_integrations.py -v -k "FMARSCrop"
```

## Family-Specific Notes

### `compass/`

- Do this first.
- Start with package move only.
- Split into one-class-per-file only after the package import surface is stable.

### `adopt/`

- Keep schedulefree and AO variants in the same family package.
- Do not create separate global `schedulefree/` or `torchao/` optimizer buckets.

### `adamw/`

- Keep low-bit AO and Kahan variants together.
- `AdamMini` should not be folded into this family.

### `fmarscrop/`

- This family already wants to live as its own package.
- The versioned and `ExMachina` naming is enough; do not invent deeper subpackages.

### `rmsprop/`

- The current single file can move into a package first.
- Splitting the three variants can happen later if the file still feels too heavy.

### `scorn/`

- This is a natural small package.
- `SCORNMachina` is close enough to `SCORN` to keep with it.

## Things To Avoid

- Do not group files by backend only.
- Do not create `optimizers/torchao/`, `optimizers/bnb/`, or `optimizers/schedulefree/` as top-level organization buckets.
- Do not mix wrappers into the optimizer package just because they have optimizer-like names.
- Do not try to split every standalone optimizer into a package. That creates noise without helping navigation.
- Do not change registry names while moving modules.

## Suggested First Concrete Pass

If you want the cleanest first chunk, do this exact sequence:

1. Create `optimizers/compass/`
2. Move `compass.py` and `fcompass.py` into it
3. Repoint package exports and registry targets
4. Verify `Compass`, `CompassAO`, `Compass8BitBNB`, `CompassPlus`, `CompassADOPT`, `CompassADOPTMARS`, `FCompass`, `FCompassPlus`, `FCompassADOPT`, `FCompassADOPTMARS`
5. Create `optimizers/adopt/`
6. Move `adopt.py`, `adopt_schedulefree.py`, and `adopt_schedulefree_ao.py`
7. Repoint package exports and registry targets
8. Verify all `ADOPT*` and `FADOPT*` paths

That gets the two biggest family wins without trying to solve the whole package in one jump.
