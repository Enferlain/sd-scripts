Fork of sd-scripts from [sd3](https://github.com/kohya-ss/sd-scripts/tree/sd3), specifically another fork of that sd3 from here [sd3-upstream](https://github.com/67372a/sd-scripts/tree/sd3-upstream). For use with the lora easy trainer ui for the time being, until I make a new in-repo frontend.
The library has been reorganized for modularity and it's highly opinionated. There can be inconsistencies or questionable choices until further notice, or no notice at all. readme and docs will be properly updated to reflect the state of the repo in time.

---

This repository contains scripts for training, generation, and utilities focused on Stable Diffusion (SD and SDXL) models.

The scripts support:

- Fine-tuning
- LoRA (network) training
- Textual Inversion training
- Image generation (might have been deleted)
- Model conversion (untested)

## Refactoring and New Structure

This repository has recently undergone a significant refactoring to improve modularity and maintainability. The key changes are:

*   **New Directory Structure:**
    *   All training scripts are now located in the `scripts/` directory.
    *   Core logic, modules, and utilities have been moved into the `library/` directory. This includes the functionality from the previous 7600-line `train_util.py` file, which has been broken down into smaller, more manageable modules.

*   **Model Support:**
    *   To streamline the codebase, support is currently focused exclusively on Stable Diffusion (SD) and SDXL models.
    *   Support for other model types has been temporarily removed, but may be reintroduced in the future as the new structure stabilizes.

## Directory Structure

```
.
├── scripts/            # Main training scripts
├── library/            # Core logic and modules
│   ├── config/         # Argument parsing and configuration
│   ├── data/           # Dataset handling, caching, and image/prompt utilities
│   ├── losses/         # Loss functions for training
│   ├── models/         # Model definitions (U-Net, Text Encoders, etc.)
│   ├── networks/       # LoRA and other network customizations
│   ├── optimizations/  # Optimizations like DeepSpeed
│   ├── optimizers/     # Custom optimizer implementations
│   ├── pipelines/      # Stable Diffusion pipelines
│   ├── strategies/     # Training strategies for different model types
│   ├── timestep_samplers/ # Samplers for the diffusion process
│   ├── training/       # Core training loop, checkpointing, and sample generation
│   ├── utils/          # Miscellaneous utility functions
│   └── vendor/         # Third-party code
├── tools/              # Utility scripts for data processing, model management, etc. (untested but unchanged)
└── docs/               # Documentation files (pending rewrite)
```

## Change History

Check the original repo and [Releases](https://github.com/kohya-ss/sd-scripts/releases) page for more details and a history of original changes.
Also worth looking up [machina's fork](https://github.com/67372a/sd-scripts/tree/sd3-upstream), specifically the sd3 and sd3-upstream branches for the stuff he did there, which there are tons of, to have an idea for how these repos diverge from the original.

---
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Enferlain/sd-scripts)
