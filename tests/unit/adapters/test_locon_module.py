import math

import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.locon.module import LoconConfig, LoconModule


def test_locon_module_rejects_non_positive_rank():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="rank must be positive"):
        LoconModule.from_target_module("locon_linear", target, config=LoconConfig(lora_dim=0))


def test_locon_module_rejects_weight_decompose_with_bypass_mode():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="weight_decompose is incompatible with bypass_mode"):
        LoconModule.from_target_module(
            "locon_linear",
            target,
            config=LoconConfig(lora_dim=2, weight_decompose=True, bypass_mode=True),
        )


def test_locon_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 3, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LoconModule.from_target_module("locon_linear", target, config=LoconConfig(lora_dim=2, alpha=2.0))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.lora_up.weight).item() == 0


def test_locon_module_scalar_init_preserves_base_forward_with_zero_scalar():
    torch.manual_seed(19)
    target = torch.nn.Linear(4, 3, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LoconModule.from_target_module("locon_linear", target, config=LoconConfig(lora_dim=2, alpha=2.0, use_scalar=True))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert isinstance(module.scalar, torch.nn.Parameter)
    assert torch.allclose(module.scalar.detach(), torch.zeros_like(module.scalar))
    assert not torch.allclose(module.lora_up.weight, torch.zeros_like(module.lora_up.weight))


def test_locon_module_random_nonzero_init_starts_active():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 3, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LoconModule.from_target_module(
        "locon_linear",
        target,
        config=LoconConfig(lora_dim=2, alpha=2.0, init_mode="random_nonzero"),
    )
    actual = module(inputs)

    assert not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_locon_module_forward_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(29)
    target = torch.nn.Linear(4, 3, bias=False)
    module = LoconModule.from_target_module("locon_linear", target, config=LoconConfig(lora_dim=2, alpha=2.0))
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_locon_module_bypass_mode_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 3, bias=False)
    module = LoconModule.from_target_module("locon_linear", target, config=LoconConfig(lora_dim=2, alpha=2.0, bypass_mode=True))
    module.eval()

    with torch.no_grad():
        module.lora_down.weight.fill_(0.2)
        module.lora_up.weight.fill_(0.3)

    inputs = torch.randn(5, 4, dtype=torch.float64)
    actual = module(inputs)

    base = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias)
    down = F.linear(inputs.to(module.dtype), module.lora_down.weight.to(module.dtype))
    up = F.linear(down, module.lora_up.weight.to(module.dtype))
    expected = (base + up * module.scale).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_locon_module_plain_dropout_is_applied_in_bypass_mode():
    torch.manual_seed(37)
    target = torch.nn.Linear(4, 3, bias=False)
    module = LoconModule.from_target_module(
        "locon_linear",
        target,
        config=LoconConfig(lora_dim=2, alpha=2.0, bypass_mode=True, dropout=0.5, init_mode="random_nonzero"),
    )
    module.train()
    inputs = torch.randn(5, 4)

    module.dropout = 0.0
    torch.manual_seed(101)
    without_plain_dropout = module.bypass_forward_diff(inputs, scale=module.multiplier)
    module.dropout = 0.5
    torch.manual_seed(101)
    with_plain_dropout = module.bypass_forward_diff(inputs, scale=module.multiplier)

    assert not torch.allclose(with_plain_dropout, without_plain_dropout)


def test_locon_module_export_round_trip_preserves_tucker_weight():
    torch.manual_seed(41)
    target = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
    module = LoconModule.from_target_module(
        "locon_conv",
        target,
        config=LoconConfig(lora_dim=2, alpha=4.0, use_tucker=True),
    )

    with torch.no_grad():
        module.lora_down.weight.normal_(mean=0.0, std=0.2)
        module.lora_up.weight.normal_(mean=0.0, std=0.2)
        assert module.lora_mid is not None
        module.lora_mid.weight.normal_(mean=0.0, std=0.2)

    state_dict = module.export_state_dict()
    restored = LoconModule.make_module_from_state_dict(
        "locon_conv",
        target,
        *(state_dict.get(key) for key in LoconModule.export_weight_keys),
    )

    assert restored.tucker is True
    assert torch.allclose(restored.get_weight(restored.shape), module.get_weight(module.shape), atol=1e-6, rtol=1e-5)


def test_locon_module_rs_lora_uses_sqrt_rank_scaling():
    torch.manual_seed(43)
    target = torch.nn.Linear(4, 4, bias=False)
    standard = LoconModule.from_target_module("locon_standard", target, config=LoconConfig(lora_dim=4, alpha=4.0))
    rs_scaled = LoconModule.from_target_module("locon_rs", target, config=LoconConfig(lora_dim=4, alpha=4.0, rs_lora=True))

    with torch.no_grad():
        standard.lora_down.weight.fill_(0.2)
        standard.lora_up.weight.fill_(0.3)
        rs_scaled.lora_down.weight.copy_(standard.lora_down.weight)
        rs_scaled.lora_up.weight.copy_(standard.lora_up.weight)

    expected_ratio = math.sqrt(standard.lora_dim)
    assert torch.allclose(rs_scaled.get_weight(rs_scaled.shape), standard.get_weight(standard.shape) * expected_ratio)
