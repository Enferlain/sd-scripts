import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.abba.module import AbbaConfig, AbbaModule


def _fill_nonzero_weights(module: AbbaModule) -> None:
    with torch.no_grad():
        module.lora_down1.weight.copy_(torch.randn_like(module.lora_down1.weight) * 0.1)
        module.lora_up1.weight.copy_(torch.randn_like(module.lora_up1.weight) * 0.1)
        module.lora_down2.weight.copy_(torch.randn_like(module.lora_down2.weight) * 0.1)
        module.lora_up2.weight.copy_(torch.randn_like(module.lora_up2.weight) * 0.1)


def test_abba_module_rejects_rank_below_two():
    target = torch.nn.Linear(4, 4, bias=False)

    with pytest.raises(ValueError, match="rank must be at least 2"):
        AbbaModule.from_target_module("abba_linear", target, config=AbbaConfig(lora_dim=1))


def test_abba_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 4, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = AbbaModule.from_target_module("abba_linear", target, config=AbbaConfig(lora_dim=4, alpha=2.0))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.lora_up2.weight).item() == 0


def test_abba_module_merged_weight_matches_forward_output_for_linear_target():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 4, bias=False)
    module = AbbaModule.from_target_module("abba_linear", target, config=AbbaConfig(lora_dim=4, alpha=2.0))
    module.eval()
    _fill_nonzero_weights(module)

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)
    actual = module(inputs)
    expected = F.linear(inputs.to(merged_weight.dtype), merged_weight, merged_bias).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_abba_module_merged_weight_applies_multiplier_once():
    torch.manual_seed(29)
    target = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        target.weight.zero_()
    module = AbbaModule.from_target_module("abba_linear", target, config=AbbaConfig(lora_dim=4, alpha=2.0))
    _fill_nonzero_weights(module)

    diff_unit, _ = module.get_diff_weight(multiplier=1.0, shape=target.weight.shape, device=target.weight.device)
    merged_weight, _ = module.get_merged_weight(multiplier=2.0, shape=target.weight.shape, device=target.weight.device)

    assert torch.allclose(merged_weight, diff_unit * 2.0, atol=1e-6, rtol=1e-5)


def test_abba_module_bypass_linear_rebuilds_current_khatri_rao_factors():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=False)
    module = AbbaModule.from_target_module("abba_linear", target, config=AbbaConfig(lora_dim=4, alpha=2.0, bypass_mode=True))
    module.eval()
    inputs = torch.randn(5, 4)

    _fill_nonzero_weights(module)
    first = module.bypass_forward_diff(inputs, scale=module.multiplier)
    with torch.no_grad():
        module.lora_up1.weight.zero_()
        module.lora_up1.weight[0, 0] = 1.0
        module.lora_down1.weight.zero_()
        module.lora_down1.weight[0, 0] = 1.0
        module.lora_up2.weight.zero_()
        module.lora_up2.weight[1, 1] = 1.0
        module.lora_down2.weight.zero_()
        module.lora_down2.weight[1, 1] = 1.0
    second = module.bypass_forward_diff(inputs, scale=module.multiplier)

    assert not torch.allclose(first, second)


def test_abba_module_bypass_conv_diff_omits_original_bias():
    torch.manual_seed(37)
    target = torch.nn.Conv2d(2, 3, kernel_size=1, bias=True)
    module = AbbaModule.from_target_module("abba_conv", target, config=AbbaConfig(lora_dim=4, alpha=2.0, bypass_mode=True))
    module.eval()
    _fill_nonzero_weights(module)
    inputs = torch.randn(2, 2, 4, 4)

    diff = module.bypass_forward_diff(inputs, scale=module.multiplier)
    base = module._apply_base_op(inputs)
    combined = module.bypass_forward(inputs, scale=module.multiplier)

    assert torch.allclose(combined, base + diff, atol=1e-6, rtol=1e-5)


def test_abba_module_forward_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(41)
    target = torch.nn.Linear(4, 4, bias=False)
    module = AbbaModule.from_target_module("abba_linear", target, config=AbbaConfig(lora_dim=4))
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_abba_module_bypass_mode_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(43)
    target = torch.nn.Linear(4, 4, bias=False)
    module = AbbaModule.from_target_module("abba_linear", target, config=AbbaConfig(lora_dim=4, alpha=2.0, bypass_mode=True))
    module.eval()
    _fill_nonzero_weights(module)
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = module.bypass_forward(inputs, scale=module.multiplier)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_abba_module_export_round_trip_preserves_merged_weight():
    torch.manual_seed(47)
    target = torch.nn.Linear(4, 4, bias=False)
    module = AbbaModule.from_target_module(
        "abba_linear",
        target,
        config=AbbaConfig(lora_dim=4, alpha=2.0, use_scalar=True),
    )
    _fill_nonzero_weights(module)
    with torch.no_grad():
        module.scalar.fill_(1.75)

    state_dict = module.export_state_dict()
    restored = AbbaModule.make_module_from_state_dict(
        "abba_linear",
        target,
        state_dict["lora_up1.weight"],
        state_dict["lora_down1.weight"],
        state_dict["lora_up2.weight"],
        state_dict["lora_down2.weight"],
        state_dict["alpha"],
        state_dict.get("dora_scale"),
    )

    original_merged, _ = module.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    restored_merged, _ = restored.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert torch.allclose(restored_merged, original_merged, atol=1e-6, rtol=1e-5)


def test_abba_module_module_dropout_can_skip_adapter_path():
    torch.manual_seed(53)
    target = torch.nn.Linear(4, 4, bias=False)
    module = AbbaModule.from_target_module(
        "abba_linear",
        target,
        config=AbbaConfig(lora_dim=4, alpha=2.0, module_dropout=1.0),
    )
    module.train()
    _fill_nonzero_weights(module)
    inputs = torch.randn(5, 4)

    assert torch.allclose(module(inputs), target(inputs), atol=1e-6, rtol=1e-5)
