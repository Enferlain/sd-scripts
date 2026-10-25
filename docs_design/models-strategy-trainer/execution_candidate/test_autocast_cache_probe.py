"""Qualify reuse evidence without changing the conservative worker default."""

import pytest
import torch

from .autocast_cache_probe import cast_nodes, fixture, forward, gradient, overlap


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("policies", (("cached", "cached"), ("cached", "uncached"), ("uncached", "cached"), ("explicit", "explicit")))
def test_overlapping_cache_ownership(policies, dtype):
    result = overlap(policies[0], policies[1], dtype)["regions"]
    if policies == ("cached", "cached"):
        assert "same dtype" in result["second"]["error"]
    else:
        assert all(row["values_and_gradient"] == "exact" for row in result.values())
        # An uncached reader's exit clears the global cache, even while the
        # cached reader is still in its region. Explicit local reuse survives.
        expected = {("cached", "uncached"): (2, 2), ("uncached", "cached"): (3, 1), ("explicit", "explicit"): (1, 1)}
        assert tuple(result[name]["weight_cast_nodes"] for name in ("first", "second")) == expected[policies]


@pytest.mark.parametrize("source", ("fp32_trainable", "fp32_frozen", "bf16_trainable"))
def test_matched_region_results_and_cast_reuse(source):
    weight, inputs, target = fixture(source, 16)
    reference = None
    for policy in ("cached", "uncached", "explicit"):
        values = forward(weight, inputs, 4, policy)
        if weight.requires_grad:
            expected = (4 if policy == "uncached" else 1) if source == "fp32_trainable" else 0
            assert cast_nodes(values) == expected
        actual = gradient(values, target)
        if reference is None:
            reference = (values, actual)
        else:
            assert all(torch.equal(left, right) for left, right in zip(reference[0], values, strict=True))
            assert torch.equal(reference[1], actual)


@pytest.mark.parametrize("policy", ("cached", "uncached", "explicit"))
def test_reuse_ends_before_a_weight_update(policy):
    weight, inputs, _ = fixture("fp32_trainable", 16)
    values = forward(weight, inputs, 4, policy)
    before = values[0].detach().clone()
    gradient(values, weight)
    with torch.no_grad():
        weight.mul_(2)
    # A fresh cast/graph after the update, not a persistent low-precision copy
    # or cast grad_fn carried across steps.
    values = forward(weight, inputs, 4, policy)
    assert torch.equal(values[0], before * 2)
    assert torch.equal(gradient(values, weight), torch.full_like(weight, 4))


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_local_cast_matches_native_cached_random_computation(dtype):
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(32, 32, generator=generator, requires_grad=True)
    inputs = torch.randn(8, 32, generator=generator, dtype=dtype)
    cached = forward(weight, inputs, 4, "cached", dtype)
    cached_gradient = gradient(cached, weight)
    explicit = forward(weight, inputs, 4, "explicit", dtype)
    assert all(torch.equal(left, right) for left, right in zip(cached, explicit, strict=True))
    assert torch.equal(cached_gradient, gradient(explicit, weight))
