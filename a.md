# Data Loading Performance Investigation

## Executive Summary

The significant performance gap observed (4it/s vs 300it/s) and the "hotwired" feeling of the data loading process are caused by a fundamental design choice: **the dataset is responsible for batch construction, effectively bypassing PyTorch's native parallelization mechanisms.**

## Current Architecture Analysis

### 1. The "Batch-in-Dataset" Pattern

In `scripts/sd_finetune.py`, the DataLoader is initialized as follows:

```python
train_dataloader = torch.utils.data.DataLoader(
    train_dataset_group,
    batch_size=1,  # <--- CRITICAL
    shuffle=True,
    collate_fn=collator,
    num_workers=n_workers,
    ...
)
```

Although `batch_size` is set to 1, the `BaseDataset.__getitem__` method in `library/data/dataset.py` does not return a single image. Instead, it:

1.  Retrieves a pre-calculated bucket index.
2.  Iterates through `bucket_batch_size` (e.g., 32) images.
3.  Sequentially loads, resizes, and processes each image/latent.
4.  Stacks them into tensors.
5.  Returns a dictionary containing the **entire batch** of data.

To the PyTorch DataLoader, this looks like loading "1 item". In reality, that "1 item" is a heavy stack of 32 images.

### 2. Bucketing Custom Logic

This design exists primarily to support aspect-ratio bucketing. The `BucketManager` groups images by resolution. To ensure a batch only contains images of the same resolution, the dataset takes control of grouping items, rather than letting the DataLoader randomly sample items.

## Performance Bottlenecks

### 1. Sequential Processing within Workers

Standard PyTorch DataLoaders achieve high throughput by having multiple workers optionally fetch **single** samples in parallel.

- **Ideal (Standard)**: 8 workers fetch 8 individual images simultaneously.
- **Current Implementation**: 1 worker fetches 1 "item" (which is actually a loop of 32 images). Inside that worker, the loading of those 32 images is **sequential**.

While you can increase `num_workers`, the granularity is too coarse. If a batch takes 2 seconds to prepare (due to I/O or resizing), that worker is blocked for the full 2 seconds.

### 2. High Overhead in `__getitem__`

The `__getitem__` method contains significant logic that runs on the CPU for every batch generation:

- Checking `image_info` state.
- Resolving paths.
- Handling caption processing (dropping, shuffling, wildcards).
- Checking caching strategies.

When combined with the sequential loop, this "garbage" logic accumulates, becoming a CPU bottleneck that starves the GPU.

### 3. Caching Inefficiency

Even when using cached latents (`.npz` files), the `__getitem__` loop still runs. Instead of a direct map from `Index -> Load File`, the code performs logic checks for every item in the batch structure before loading.

## Recommended Alternative Architecture

To achieve the "300it/s" performance of a raw loader while maintaining bucketing capabilities, the architecture should be refactored to:

1.  **Use a Custom `BatchSampler`**: Move the bucketing logic here. The sampler yields lists of indices that belong to the same bucket.
2.  **Standard `Dataset`**: The dataset's `__getitem__` should simply take an index and return **one** simple image/latent.
3.  **Native `DataLoader` Batching**: Initialize the DataLoader with the custom BatchSampler.
    - This allows PyTorch to spawn workers that fetch individual images in parallel.
    - The `collate_fn` would then stack the images (which are guaranteed to be the same size by the BatchSampler) into a batch.

This separation of concerns allows PyTorch's multiprocessing to saturate the I/O and CPU bandwidth effectively.

---

"not doing all the absolute garbage it does just to make 1 batch contain everything you need
like caching shit
it should just be a normal data loader that either loads off disk or calls some func to cache but it's a hotwired mess
I can cache like 300it/s on a 4090 by not using the dataloader :FubukiWheeze:
dataloader is 4it/s"

"also do note that new_cache_latents on the dataset group (dreamboothdataset, typically) has code that was explicitly commented out for multiprocessing, which can 10x latent caching speed :FubukiWheeze: 

if you're gonna rewrite it however, heed this architecture design:

1 thread for loading images
1 thread for resizing them
1 thread for VAE encoding
1 thread for saving to disk

four threads to do "two things" may seem excessive, but even if you just break out the loading and VAE encoding, you can easily get 50+it/s on like, a 3090, I was hitting 120it/s on 1x4090, and can cache all of danbooru/e6 in like, 4-5 hours on eight cards :FubukiWheeze: "