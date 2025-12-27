# Padding Handling in SD Scripts: SD vs SDXL

This document summarizes the findings regarding how padding and tokenization are handled in `sd-scripts`, specifically comparing Stable Diffusion (SD) 1.x/2.x and SDXL.

## Core Findings

1.  **Implements Strategies:** The `library.strategies` module (specifically `strategy_base.py`) centralizes the logic for padding and tokenization, dynamically adapting based on the tokenizer's properties.
2.  **Legacy Code in Pipelines:**
    - **SD Pipeline (`lpw_stable_diffusion.py`):** Uses an internal, legacy implementation of padding logic. It does _not_ use `library.strategies`.
    - **SDXL Pipeline (`sdxl_lpw_stable_diffusion.py`):** Uses the modern `library.strategies` system. However, it still contains dead code (unused functions) copied from the SD pipeline.

## Detailed Comparison

### 1. Tokenizer Behavior (The "Why")

The core difference stems from the tokenizers used by the respective models:

| Model          | Tokenizer        | Pad Token | EOS Token | Relation       |
| :------------- | :--------------- | :-------- | :-------- | :------------- |
| **SD 1.x**     | CLIP (ViT-L/14)  | `49407`   | `49407`   | **Pad == EOS** |
| **SD 2.x**     | CLIP (ViT-L/14)  | `0`       | `49407`   | **Pad != EOS** |
| **SDXL (TE1)** | CLIP (ViT-L/14)  | `49407`   | `49407`   | **Pad == EOS** |
| **SDXL (TE2)** | OpenCLIP (ViT-G) | `0`       | `49407`   | **Pad != EOS** |

### 2. Implementation Logic

#### Strategy System (`library/strategies/strategy_base.py`)

The class `TokenizeStrategy` contains the method `_get_input_ids`. It detects the padding style by checking:

```python
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    # V1 Style Logic
else:
    # V2/SDXL Style Logic
```

- **V1 Style (Pad == EOS):**

  - Padding is treated effectively as repeating the EOS token.
  - Chunks are constructed as `[BOS] ... tokens ... [EOS] [EOS] ...`.
  - When splitting long prompts (>77 tokens), it converts the sequence into chunks of `[BOS] ... [EOS]`.

- **V2/SDXL Style (Pad != EOS):**
  - Padding (`0`) is distinct from EOS.
  - The model expects a clear termination.
  - Chunks are constructed ensuring valid termination: `[BOS] ... tokens ... [EOS] [PAD] [PAD] ...`.
  - Logic ensures that even if a chunk is full or padded, the `<EOS>` token is correctly placed before any zero-padding begins.

#### SD Pipeline (`library/pipelines/lpw_stable_diffusion.py`)

- **Status:** Legacy / Independent.
- **Logic:** Implements similar logic to the strategy system but manually within `pad_tokens_and_weights` and `get_unweighted_text_embeddings`.
- **Observation:** It is older code that has likely not been refactored to use the shared strategies to avoid breaking changes or regression in inference.

#### SDXL Pipeline (`library/pipelines/sdxl_lpw_stable_diffusion.py`)

- **Status:** Modern.
- **Logic:**
  - Imports and uses `library.strategies.strategy_sdxl` and `library.strategies.strategy_base`.
  - In `__call__`, it retrieves the active strategy:
    ```python
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
    ```
  - It delegates the actual tokenization and weighting to these strategy classes.
- **Issues:** Contains unused global functions (`pad_tokens_and_weights`, etc.) that were likely copy-pasted from the SD pipeline during initial development but superseded by the strategy system.

## Summary Table

| Feature             | SD Pipeline                | SDXL Pipeline                      |
| :------------------ | :------------------------- | :--------------------------------- |
| **Source File**     | `lpw_stable_diffusion.py`  | `sdxl_lpw_stable_diffusion.py`     |
| **Logic Source**    | Local Functions            | `library.strategies`               |
| **V1 Support**      | Native (Local)             | Via Strategy (TE1)                 |
| **V2/SDXL Support** | Native (Local)             | Via Strategy (TE2)                 |
| **Code Quality**    | Functional, Self-Contained | Good (Modular), contains dead code |
