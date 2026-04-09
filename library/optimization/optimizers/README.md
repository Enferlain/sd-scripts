Source-of-truth donor note: treat [LoraEasyCustomOptimizer](/mnt/d/Projects/sd-scripts/library/vendor/LoRA_Easy_Training_scripts_Backend/custom_scheduler/LoraEasyCustomOptimizer) as the authoritative optimizer donor tree for this audit/absorption work. The older `custom_optimizer/` copy is stale and should not drive queue or provenance decisions.

There are still several top-level donor modules left to absorb. A chunk of that surface is already effectively handled in other homes:
- `lpfadamw.py` is already absorbed as [lpf_adamw.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/lpf_adamw.py)
- `snoo_asgd.py` is already repo-owned under the wrapper path
- `adam.py` is already effectively absorbed across [adamw_low_bit.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/adamw_low_bit.py) and [adamw_8bit_kahan.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/adamw_8bit_kahan.py)
- `adagc.py` is already effectively absorbed as [adagc.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/utils/adagc.py)
- `sgd.py` is already effectively absorbed as [sgd_sai.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/sgd_sai.py)

So the practical remaining queue is mostly the smaller leaf modules below.

**Still ahead**
- `adammini.py`
- remaining heavier `fmarscrop.py` siblings (`FMARSCropV2ExMachina`, `FMARSCropV3`, `FMARSCropV3ExMachina`)
- `clybius_experiments.py` as a likely lower-priority/unclear one

`bcos.py` is now absorbed as [bcos.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/bcos.py), `projective_adam.py` is now absorbed as [projective_adam.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/projective_adam.py), `wiwiopt.py` is now absorbed as [wiwiopt.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/wiwiopt.py), `oagopt.py` / `ocgopt.py` / `scgopt.py` are now absorbed into [oagopt.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/oagopt.py), [ocgopt.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/ocgopt.py), and [scgopt.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/scgopt.py), `fftdescent.py` is now absorbed as [fftdescent.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/fftdescent.py), `farmscrop.py` is now absorbed across [farmscrop.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/farmscrop.py) and [farmscrop_v2.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/farmscrop_v2.py), the plain `fmarscrop.py` path is now absorbed across [fmarscrop.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/fmarscrop.py) and [fmarscrop_v2.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/fmarscrop_v2.py), `fishmonger.py` is now absorbed across [fishmonger.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/fishmonger.py), `abmog.py` is now absorbed as [abmog.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/abmog.py), `singstate.py` is now absorbed as [singstate.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/singstate.py), `talon.py` is now absorbed as [talon.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/talon.py), `glyph.py` is now absorbed as [glyph.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/glyph.py), and the repo-owned [compass.py](/mnt/d/Projects/sd-scripts/library/optimization/optimizers/compass.py) now covers the remaining vendor `Compass8BitBNB` / `CompassAO` family members too, so the next target can stay focused on the smaller donor leaves.

**Also still untouched if we ever want it**
- CPU/offload runtime wrapper in [cpu_offload.py](/mnt/d/Projects/sd-scripts/library/vendor/LoRA_Easy_Training_scripts_Backend/custom_scheduler/LoraEasyCustomOptimizer/low_bit_optim/cpu_offload.py)

With `came.py`, `cstableadamw.py`, `grokfast.py`, `bcos.py`, `projective_adam.py`, `wiwiopt.py`, `abmog.py`, `singstate.py`, `talon.py`, `glyph.py`, `farmscrop.py`, the plain `fmarscrop.py` pair, and the full `compass.py` family absorbed too, the remaining work is mostly the awkward leaf modules rather than one large backend-heavy bucket.

# interesting optimizers according tp agent

## Interesting

- ProjectiveAdam: "It has a clear idea, the implementation shape is readable once you trace the projection/inverse-projection loop, and it was a nice fit for the repo-owned optimizer layer after smoothing out the CPU/offload edges."
- OAGOpt: "It feels like the cleanest “there’s a real optimizer idea here” one of the three: orthogonalized update shaping, adaptive scaling, cautious masking, FFT-side filtering, and a fairly coherent full-step construction. It’s definitely busy, but it has an identity."
- ABMOG: "It has a stronger identity than the other two: AB/AM history correction, spectral shaping, cautious masking, dual-norm scaling, and that odd little offloaded-history path. It’s busy, but it feels like it’s trying to do a specific thing rather than just remixing optimizer tropes."
- FMARSCrop: "It feels more coherent: MARS correction, Fisher scaling, optional stable weight decay, and the Compass-style momentum amplification all hang together reasonably well."

 ## Honorable mentions

- WiwiOpt: "There’s a lot going on, but it feels like someone genuinely experimenting with update geometry instead of just renaming Adam variants. It’s messy in a fun way."
- SCORNMachina: "...the one I respect the most from an engineering angle, mostly because it forced the most serious offloaded-state/runtime thinking. It’s not the prettiest, but it exposed useful integration pressure on the framework."
- OCGOpt: "The centralized/full-momentum split is the main hook, and the optional Kahan path makes it feel a bit more practical for lower-precision training. I like it less than OAGOpt conceptually, but more than a pure gimmick."
- TALON: "The sign-vs-magnitude split is a real idea, and it has a cleaner conceptual core than a lot of these. It still gets pretty kitchen-sink once the spectral/frequency options pile on, but I can see the shape."
- Glyph: "LMO-style normalization, oscillation smoothing, adaptive EMA masking, and a pretty distinct update shape. It feels more intentional than a lot of the random optimizer mashups.
- FMARSCropV2: "...is okay, but it reads more like a trimmed/practical variant of FMARSCROP than a stronger idea."

## Meh

- BCOS: "...the least exciting conceptually, but I like that it was easy to make safer without changing its basic shape. It felt like a very practical absorption."
- SCGOpt: "It’s not bad, but it feels more like a sign-based variant remixing ideas we’ve already seen elsewhere, rather than introducing a notably sharper new shape."
- SingState: "...the least compelling of the three. It’s not bad, but it feels more like a stripped variant living in the same neighborhood as TALON rather than a sharper standalone idea."
