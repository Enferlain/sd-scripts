# AUDIT 3: Validation Dataset

## Findings

| Question                                            | Status    | Notes                        |
| --------------------------------------------------- | --------- | ---------------------------- |
| How is validation dataloader currently created?     | ✅ Legacy | `DatasetGroup` pattern       |
| Should validation use `prepare_validation_epoch()`? | ✅ Yes    | Wire to `TrainingDataset`    |
| Does validation need per-epoch manifest?            | ❌ No     | Deterministic/Static         |

### 1. How is validation dataloader currently created?

The current validation dataloader is created using the **legacy data pipeline** (`DatasetGroup`).

*   **Mechanism:** `scripts/sdxl_peft.py` calls `library/data/_deprecated/dataset_setup.py:prepare_datasets`.
*   **Construction:** `prepare_datasets` utilizes `BlueprintGenerator` and `config_util.generate_dataset_group_by_blueprint` to instantiate a `val_dataset_group` (which is a collection of legacy `BaseDataset` instances).
*   **Loader:** The script then creates a standard PyTorch `DataLoader` wrapping this `val_dataset_group` with `shuffle=False`.
*   **Code Reference:**
    ```python
    # scripts/sdxl_peft.py
    train_dataset_group, val_dataset_group, collator, current_epoch, current_step = dataset_result
    # ...
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset_group if val_dataset_group is not None else [],
        shuffle=False,
        # ...
    )
    ```

### 2. Should validation use `prepare_validation_epoch()`?

**Yes.** To migrate validation to the new Phase 4 data pipeline, `prepare_validation_epoch()` should be used.

*   **Functionality:** `library/data/pipeline/epoch_preparation.py:prepare_validation_epoch` generates an `EpochManifest` specifically for validation. It groups images by bucket and sorts them deterministically (unlike the random shuffle in `prepare_epoch`).
*   **Integration Path:** The training script (`sdxl_peft.py`) needs to be updated to:
    1.  Call `prepare_validation_epoch(manifest, ...)` to get the validation manifest.
    2.  Instantiate the new `TrainingDataset` (from `library/data/pipeline/dataloader.py`) using this manifest.
    3.  Create the dataloader via `create_training_dataloader` (or manually), ensuring `shuffle=False`.

### 3. Does validation need per-epoch manifest?

**No.** A single static manifest is sufficient for validation.

*   **Reasoning:**
    *   **Determinism:** Validation data should remain constant across epochs to ensure comparable metrics. It does not require the per-epoch reshuffling or dynamic caption dropout used in training.
    *   **Implementation:** `prepare_validation_epoch` sets `epoch=0` and explicitly comments "Validation doesn't have epochs". It performs a deterministic sort of the buckets.
    *   **Efficiency:** Since the order and content don't change, the `EpochManifest` generated at the start of training can be reused for every validation loop, unlike the training manifest which must be regenerated (or re-seeded) every epoch to ensure proper shuffling and aspect ratio bucket stability.
