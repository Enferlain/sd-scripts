# Third-party notices

This repository is licensed under the Apache License 2.0 (`LICENSE`). Some of
its files are derived from, or follow the structure of, code from the projects
below. Each such file names its sources in its header; this file lists the
sources, their licenses and the files that derive from them.

## kohya-ss/sd-scripts

- Source: https://github.com/kohya-ss/sd-scripts
- License: Apache License 2.0 (`LICENSE` in this repository). Copyright 2022 kohya-ss.
- Files derived from it:
  - `library/models/denoisers/sd_unet.py`, from `library/sdxl_original_unet.py`
  - `library/models/sdxl/assembly.py`, following `library/sdxl_model_util.py`

## Hugging Face Diffusers

- Source: https://github.com/huggingface/diffusers
- License: Apache License 2.0 (`LICENSE` in this repository). Copyright 2022-2023 The HuggingFace Team. All rights reserved. (The repository's `LICENSE` has no copyright line; its source files carry this one.)
- kohya-ss's `sdxl_original_unet.py` credits Diffusers' UNet as its base. Files derived through it:
  - `library/models/denoisers/sd_unet.py`

## openai/guided-diffusion

- Source: https://github.com/openai/guided-diffusion
- License: MIT. Copyright (c) 2021 OpenAI.
- The UNet structure (`openaimodel.py`) that the latent diffusion UNet follows. Files following it:
  - `library/models/denoisers/sd_unet.py`

## CompVis/latent-diffusion

- Source: https://github.com/CompVis/latent-diffusion
- License: MIT. Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich.
- The latent diffusion UNet and its spatial transformer. Files following it:
  - `library/models/denoisers/sd_unet.py`

## Stability-AI/generative-models

- Source: https://github.com/Stability-AI/generative-models
- License: MIT (`LICENSE-CODE`). Copyright (c) 2023 Stability AI.
- SDXL's UNet configuration and attention (`sgm/modules`). Files following it:
  - `library/models/denoisers/sd_unet.py`
  - `library/models/sdxl/assembly.py`

## MIT License text

The MIT-licensed sources above are used under these terms, with each one's
copyright line as given above:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
