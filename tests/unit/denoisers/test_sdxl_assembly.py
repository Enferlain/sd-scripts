"""SDXL as released, as plain code: its size embedding and its conditioning layout."""

import math

import pytest
import torch

from library.models.sdxl.assembly import SIZE_EMBEDDING, conditioning, size_embedding


def reference(numbers: list[float]) -> torch.Tensor:
    """The reference's embedding of each number, cosines then sines, in float32, one row."""
    half = SIZE_EMBEDDING // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    angles = torch.tensor(numbers, dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).reshape(1, -1)


def test_sizes_are_embedded_exactly_whatever_dtype_the_embedding_is_made_in() -> None:
    """Made in bfloat16 first, 1417 would be 1416; the highest frequency turns that one unit into a whole radian."""
    embedded = size_embedding([[1417, 1645]], [[2, 0]], [[1024, 1024]], torch.device("cpu"), torch.bfloat16)
    assert embedded.dtype == torch.bfloat16 and embedded.shape == (1, 6 * SIZE_EMBEDDING)
    original = embedded[:, : 2 * SIZE_EMBEDDING]
    assert torch.equal(original, reference([1417.0, 1645.0]).to(torch.bfloat16))
    assert not torch.equal(original, reference([1416.0, 1648.0]).to(torch.bfloat16)), "the sizes bfloat16 would round them to"


def test_sdxl_conditioning_refuses_inputs_whose_batches_differ() -> None:
    sizes = torch.zeros(2, 1536)
    with pytest.raises(ValueError, match="2 size rows, and 1 first encoder's states"):
        conditioning(torch.zeros(1, 77, 768), torch.zeros(2, 77, 1280), torch.zeros(2, 1280), sizes, torch.float32)
    context, vector = conditioning(torch.zeros(2, 77, 768), torch.zeros(2, 77, 1280), torch.zeros(2, 1280), sizes, torch.float32)
    assert context.shape == (2, 77, 2048) and vector.shape == (2, 2816)
