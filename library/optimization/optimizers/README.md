By raw top-level vendor-file count, there are `26` unmatched donor modules left. Two of those are effectively already handled in other homes:
- `lpfadamw.py` is already absorbed as [lpf_adamw.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/lpf_adamw.py)
- `snoo_asgd.py` is already repo-owned under the wrapper path

So the practical remaining top-level queue is closer to `24`, plus the backend-heavy remainder still sitting inside the donor [compass.py](/mnt/d/Projects/sd-scripts/library/vendor/LoRA_Easy_Training_scripts_Backend/custom_scheduler/LoraEasyCustomOptimizer/compass.py).

**Still ahead**
- `Compass8BitBNB` and `CompassAO` inside the donor [compass.py](/mnt/d/Projects/sd-scripts/library/vendor/LoRA_Easy_Training_scripts_Backend/custom_scheduler/LoraEasyCustomOptimizer/compass.py)
- `scorn.py`
- `scornmachina.py`
- `abmog.py`
- `adagc.py`
- `adam.py`
- `adammini.py`
- `bcos.py`
- `came.py`
- `cstableadamw.py`
- `farmscrop.py`
- `fftdescent.py`
- `fishmonger.py`
- `fmarscrop.py`
- `glyph.py`
- `grokfast.py`
- `oagopt.py`
- `ocgopt.py`
- `projective_adam.py`
- `scgopt.py`
- `sgd.py`
- `singstate.py`
- `talon.py`
- `wiwiopt.py`
- `clybius_experiments.py` as a likely lower-priority/unclear one

**What I’d call the real remaining heavy bucket**
- `CompassAO` / `Compass8BitBNB`
- `SCORN`
- `SCORNMachina`

Those are the ones most likely to need the same careful treatment `Compass` just needed.

**Also still untouched if we ever want it**
- CPU/offload runtime wrapper in [cpu_offload.py](/mnt/d/Projects/sd-scripts/library/vendor/LoRA_Easy_Training_scripts_Backend/custom_scheduler/LoraEasyCustomOptimizer/low_bit_optim/cpu_offload.py)

With `dehaze.py`, `gooddog.py`, and `mythical.py` absorbed, `scorn.py` is still probably the cleanest next serious target in the same orthogonalized bucket, with the remaining Compass backend slice still among the messier follow-ups.
