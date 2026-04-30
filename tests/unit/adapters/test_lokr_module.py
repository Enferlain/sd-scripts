import math

import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.lokr.module import LokrConfig, LokrModule, factorization


def test_lokr_factorization_preserves_explicit_factor_orientation():
    assert factorization(128, 32) == (32, 4)


def test_lokr_module_rejects_non_positive_rank():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="rank must be positive"):
        LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=0))


def test_lokr_module_rejects_none_rank():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="rank must be positive"):
        LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=None))


def test_lokr_module_rejects_plain_dropout():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="plain dropout is disabled"):
        LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=2, dropout=0.1))


def test_lokr_module_rejects_weight_decompose_with_bypass_mode():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="weight_decompose is incompatible with bypass_mode"):
        LokrModule.from_target_module(
            "lokr_linear",
            target,
            config=LokrConfig(lora_dim=2, weight_decompose=True, bypass_mode=True),
        )


def test_lokr_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 4, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=2, alpha=2.0))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert module.use_w2 is True
    assert torch.count_nonzero(module.lokr_w2).item() == 0


def test_lokr_module_scalar_init_preserves_base_forward_with_zero_scalar():
    torch.manual_seed(19)
    target = torch.nn.Linear(4, 4, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=2, alpha=2.0, use_scalar=True))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert isinstance(module.scalar, torch.nn.Parameter)
    assert torch.allclose(module.scalar.detach(), torch.zeros_like(module.scalar))
    assert not torch.allclose(module.lokr_w2, torch.zeros_like(module.lokr_w2))


def test_lokr_module_random_nonzero_init_starts_active():
    torch.manual_seed(22)
    target = torch.nn.Linear(4, 4, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LokrModule.from_target_module(
        "lokr_linear",
        target,
        config=LokrConfig(lora_dim=2, alpha=2.0, init_mode="random_nonzero"),
    )
    actual = module(inputs)

    assert not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(module.lokr_w2, torch.zeros_like(module.lokr_w2))


def test_lokr_module_merged_weight_matches_forward_output_for_linear_target():
    torch.manual_seed(7)
    target = torch.nn.Linear(4, 4, bias=False)
    module = LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=2, alpha=2.0))
    module.eval()

    with torch.no_grad():
        module.lokr_w1.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=target.weight.dtype))
        module.lokr_w2.copy_(torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=target.weight.dtype))

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)

    actual = module(inputs)
    expected = F.linear(inputs, merged_weight.to(inputs.dtype), merged_bias)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lokr_module_forward_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(13)
    target = torch.nn.Linear(4, 4, bias=False)
    module = LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=2, alpha=2.0))
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lokr_module_bypass_mode_adds_diff_path_on_top_of_base_forward():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 4, bias=False)
    module = LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=2, alpha=2.0, bypass_mode=True))
    module.eval()

    with torch.no_grad():
        module.lokr_w1.fill_(0.2)
        module.lokr_w2.fill_(0.3)

    inputs = torch.randn(5, 4)
    actual = module(inputs)
    expected = target(inputs) + module.bypass_forward_diff(inputs, scale=module.multiplier)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lokr_module_bypass_mode_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(29)
    target = torch.nn.Linear(4, 4, bias=False)
    module = LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=2, alpha=2.0, bypass_mode=True))
    module.eval()

    with torch.no_grad():
        module.lokr_w1.fill_(0.2)
        module.lokr_w2.fill_(0.3)

    inputs = torch.randn(5, 4, dtype=torch.float64)
    actual = module(inputs)

    base = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias)
    diff = F.linear(inputs.to(module.lokr_w1.dtype), module.get_weight(module.shape).to(module.lokr_w1.dtype) * module.multiplier)
    expected = (base + diff).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lokr_module_rank_dropout_scaling_preserves_mask_but_changes_kept_rows():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=False)
    module = LokrModule.from_target_module("lokr_linear", target, config=LokrConfig(lora_dim=2, alpha=2.0, rank_dropout=0.5))
    module_scaled = LokrModule.from_target_module(
        "lokr_linear_scaled",
        target,
        config=LokrConfig(lora_dim=2, alpha=2.0, rank_dropout=0.5, rank_dropout_scale=True),
    )

    with torch.no_grad():
        module.lokr_w1.fill_(0.2)
        module.lokr_w2.fill_(0.3)
        module_scaled.lokr_w1.copy_(module.lokr_w1)
        module_scaled.lokr_w2.copy_(module.lokr_w2)

    module.train()
    module_scaled.train()
    torch.manual_seed(123)
    unscaled = module.get_weight(module.shape)
    torch.manual_seed(123)
    scaled = module_scaled.get_weight(module_scaled.shape)

    zero_rows_unscaled = unscaled.reshape(unscaled.shape[0], -1).abs().sum(dim=1) == 0
    zero_rows_scaled = scaled.reshape(scaled.shape[0], -1).abs().sum(dim=1) == 0

    assert torch.equal(zero_rows_unscaled, zero_rows_scaled)
    assert not torch.allclose(unscaled, scaled)


def test_lokr_module_rs_lora_uses_sqrt_rank_scaling_for_decomposed_weight():
    torch.manual_seed(37)
    target = torch.nn.Linear(36, 36, bias=False)
    standard = LokrModule.from_target_module(
        "lokr_standard",
        target,
        config=LokrConfig(lora_dim=2, alpha=2.0, decompose_both=True),
    )
    rs_scaled = LokrModule.from_target_module(
        "lokr_rs",
        target,
        config=LokrConfig(lora_dim=2, alpha=2.0, decompose_both=True, rs_lora=True),
    )

    with torch.no_grad():
        standard.lokr_w1_a.fill_(0.2)
        standard.lokr_w1_b.fill_(0.3)
        standard.lokr_w2_a.fill_(0.4)
        standard.lokr_w2_b.fill_(0.5)
        rs_scaled.lokr_w1_a.copy_(standard.lokr_w1_a)
        rs_scaled.lokr_w1_b.copy_(standard.lokr_w1_b)
        rs_scaled.lokr_w2_a.copy_(standard.lokr_w2_a)
        rs_scaled.lokr_w2_b.copy_(standard.lokr_w2_b)

    expected_ratio = math.sqrt(standard.lora_dim)
    assert torch.allclose(rs_scaled.get_weight(rs_scaled.shape), standard.get_weight(standard.shape) * expected_ratio)


def test_lokr_module_export_round_trip_preserves_decomposed_weight():
    torch.manual_seed(11)
    target = torch.nn.Linear(16, 16, bias=False)
    module = LokrModule.from_target_module(
        "lokr_linear",
        target,
        config=LokrConfig(lora_dim=1, alpha=4.0, decompose_both=True),
    )

    with torch.no_grad():
        module.lokr_w1_a.normal_(mean=0.0, std=0.2)
        module.lokr_w1_b.normal_(mean=0.0, std=0.2)
        module.lokr_w2_a.normal_(mean=0.0, std=0.2)
        module.lokr_w2_b.normal_(mean=0.0, std=0.2)

    state_dict = module.export_state_dict()
    restored = LokrModule.make_module_from_state_dict(
        "lokr_linear",
        target,
        *(state_dict.get(key) for key in LokrModule.export_weight_keys),
    )

    assert restored.use_w1 is False
    assert restored.use_w2 is False
    assert torch.allclose(restored.get_weight(restored.shape), module.get_weight(module.shape), atol=1e-6, rtol=1e-5)


def test_lokr_module_merged_weight_matches_forward_output_for_tucker_conv2d_target():
    torch.manual_seed(43)
    target = torch.nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
    module = LokrModule.from_target_module(
        "lokr_conv2d",
        target,
        config=LokrConfig(lora_dim=1, alpha=2.0, use_tucker=True),
    )
    module.eval()

    with torch.no_grad():
        module.lokr_w1.fill_(0.2)
        module.lokr_t2.fill_(0.3)
        module.lokr_w2_a.fill_(0.4)
        module.lokr_w2_b.fill_(0.5)

    inputs = torch.randn(2, 8, 5, 5)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)

    actual = module(inputs)
    expected = F.conv2d(inputs, merged_weight.to(inputs.dtype), merged_bias, stride=target.stride, padding=target.padding)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
