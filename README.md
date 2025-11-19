This repository contains scripts for training, generation, and utilities focused on Stable Diffusion (SD and SDXL) models.

The scripts support:

- DreamBooth training
- Fine-tuning
- LoRA training
- Textual Inversion training
- Image generation
- Model conversion

[日本語版READMEはこちら](./README-ja.md)

The development version is in the `dev` branch. Please check the dev branch for the latest changes.

FLUX.1 and SD3/SD3.5 support is done in the `sd3` branch. If you want to train them, please use the sd3 branch.


For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

### Sponsors

We are grateful to the following companies for their generous sponsorship:

### Support the Project

If you find this project helpful, please consider supporting its development via [GitHub Sponsors](https://github.com/sponsors/kohya-ss/). Your support is greatly appreciated!

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
├── tools/              # Utility scripts for data processing, model management, etc.
└── docs/               # Documentation files
```

## About requirements.txt

The file does not contain requirements for PyTorch. Because the version of PyTorch depends on the environment, it is not included in the file. Please install PyTorch first according to the environment. See installation instructions below.

The scripts are tested with Pytorch 2.1.2. PyTorch 2.2 or later will work. Please install the appropriate version of PyTorch and xformers.

## Links to usage documentation

Most of the documents are written in Japanese.

[English translation by darkstorm2150 is here](https://github.com/darkstorm2150/sd-scripts#links-to-usage-documentation). Thanks to darkstorm2150!

* [Training guide - common](./docs/train_README-ja.md) : data preparation, options etc... 
  * [Chinese version](./docs/train_README-zh.md)
* [SDXL training](./docs/train_SDXL-en.md) (English version)
* [Dataset config](./docs/config_README-ja.md) 
  * [English version](./docs/config_README-en.md)
* [DreamBooth training guide](./docs/train_db_README-ja.md)
* [Step by Step fine-tuning guide](./docs/fine_tune_README_ja.md):
* [Training LoRA](./docs/train_network_README-ja.md)
* [Training Textual Inversion](./docs/train_ti_README-ja.md)
* [Image generation](./docs/gen_img_README-ja.md)
* note.com [Model conversion](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Python 3.10.x, 3.11.x, and 3.12.x will work but not tested.

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Windows Installation

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

accelerate config
```

If `python -m venv` shows only `python`, change `python` to `py`.

Note: Now `bitsandbytes==0.44.0`, `prodigyopt==1.0` and `lion-pytorch==0.0.6` are included in the requirements.txt. If you'd like to use the another version, please install it manually.

This installation is for CUDA 11.8. If you use a different version of CUDA, please install the appropriate version of PyTorch and xformers. For example, if you use CUDA 12, please install `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121` and `pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121`.

If you use PyTorch 2.2 or later, please change `torch==2.1.2` and `torchvision==0.16.2` and `xformers==0.0.23.post1` to the appropriate version.

<!-- 
cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
-->
Answers to accelerate config:

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

If you'd like to use bf16, please answer `bf16` to the last question.

Note: Some user reports ``ValueError: fp16 mixed precision requires a GPU`` is occurred in training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`` 

(Single GPU with id `0` will be used.)

## Usage

All training scripts are located in the `scripts/` directory. To run a script, you'll need to use `accelerate launch`, for example:

```powershell
accelerate launch scripts/sdxl_train.py --network_module=networks.lora ...
```

For more detailed usage instructions and examples, please refer to the [SDXL training documentation](./docs/train_SDXL-en.md).

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

### Upgrade PyTorch

If you want to upgrade PyTorch, you can upgrade it with `pip install` command in [Windows Installation](#windows-installation) section. `xformers` is also required to be upgraded when PyTorch is upgraded.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause

## Change History

Please see the [Releases](https://github.com/kohya-ss/sd-scripts/releases) page for a history of changes.
