import math

import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.loha.module import LohaConfig, LohaModule


def test_loha_module_rejects_non_positive_rank():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="rank must be positive"):
        LohaModule.from_target_module("loha_linear", target, config=LohaConfig(lora_dim=0))


def test_loha_module_rejects_negative_rank():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="rank must be positive"):
        LohaModule.from_target_module("loha_linear", target, config=LohaConfig(lora_dim=-1))


def test_loha_module_rejects_none_rank():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="rank must be positive"):
        LohaModule.from_target_module("loha_linear", target, config=LohaConfig(lora_dim=None))


def test_loha_module_rejects_weight_decompose_with_bypass_mode():
    target = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(ValueError, match="weight_decompose is incompatible with bypass_mode"):
        LohaModule.from_target_module(
            "loha_linear",
            target,
            config=LohaConfig(lora_dim=2, weight_decompose=True, bypass_mode=True),
        )


def test_loha_module_zeroed_parameters_produce_zero_diff_weight():
    target = torch.nn.Linear(4, 3, bias=False)
    module = LohaModule.from_target_module("loha_linear", target, config=LohaConfig(lora_dim=2, alpha=2.0))

    with torch.no_grad():
        module.hada_w1_a.zero_()
        module.hada_w1_b.zero_()
        module.hada_w2_a.zero_()
        module.hada_w2_b.zero_()

    diff_weight, _ = module.get_diff_weight(shape=target.weight.shape)

    assert torch.allclose(diff_weight, torch.zeros_like(target.weight))


def test_loha_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 3, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LohaModule.from_target_module("loha_linear", target, config=LohaConfig(lora_dim=2, alpha=2.0))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.hada_w2_a).item() == 0


def test_loha_module_scalar_init_preserves_base_forward_with_zero_scalar():
    torch.manual_seed(19)
    target = torch.nn.Linear(4, 3, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LohaModule.from_target_module("loha_linear", target, config=LohaConfig(lora_dim=2, alpha=2.0, use_scalar=True))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert isinstance(module.scalar, torch.nn.Parameter)
    assert torch.allclose(module.scalar.detach(), torch.zeros_like(module.scalar))
    assert not torch.allclose(module.hada_w2_a, torch.zeros_like(module.hada_w2_a))


def test_loha_module_zero_delta_he_init_preserves_base_forward():
    torch.manual_seed(21)
    target = torch.nn.Linear(4, 3, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LohaModule.from_target_module(
        "loha_linear",
        target,
        config=LohaConfig(lora_dim=2, alpha=2.0, init_mode="zero_delta_he"),
    )
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.hada_w2_a).item() == 0
    assert not torch.allclose(module.hada_w1_a, torch.zeros_like(module.hada_w1_a))
    assert not torch.allclose(module.hada_w1_b, torch.zeros_like(module.hada_w1_b))
    assert not torch.allclose(module.hada_w2_b, torch.zeros_like(module.hada_w2_b))


def test_loha_module_random_nonzero_init_starts_active():
    torch.manual_seed(22)
    target = torch.nn.Linear(4, 3, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LohaModule.from_target_module(
        "loha_linear",
        target,
        config=LohaConfig(lora_dim=2, alpha=2.0, init_mode="random_nonzero"),
    )
    actual = module(inputs)

    assert not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(module.hada_w2_a, torch.zeros_like(module.hada_w2_a))


def test_loha_module_random_nonzero_scalar_mode_starts_active():
    torch.manual_seed(24)
    target = torch.nn.Linear(4, 3, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LohaModule.from_target_module(
        "loha_linear",
        target,
        config=LohaConfig(lora_dim=2, alpha=2.0, init_mode="random_nonzero", use_scalar=True),
    )
    actual = module(inputs)

    assert not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(module.scalar.detach(), torch.ones_like(module.scalar))


def test_loha_module_merged_weight_matches_forward_output_for_linear_target():
    torch.manual_seed(7)
    target = torch.nn.Linear(4, 3, bias=False)
    module = LohaModule.from_target_module("loha_linear", target, config=LohaConfig(lora_dim=2, alpha=2.0))
    module.eval()

    with torch.no_grad():
        module.hada_w1_a.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=target.weight.dtype))
        module.hada_w1_b.copy_(torch.tensor([[0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4]], dtype=target.weight.dtype))
        module.hada_w2_a.copy_(torch.tensor([[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]], dtype=target.weight.dtype))
        module.hada_w2_b.copy_(torch.tensor([[1.4, 1.3, 1.2, 1.1], [1.0, 0.9, 0.8, 0.7]], dtype=target.weight.dtype))

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)

    actual = module(inputs)
    expected = F.linear(inputs, merged_weight.to(inputs.dtype), merged_bias)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_loha_module_bypass_mode_adds_diff_path_on_top_of_base_forward():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 3, bias=False)
    module = LohaModule.from_target_module("loha_linear", target, config=LohaConfig(lora_dim=2, alpha=2.0, bypass_mode=True))
    module.eval()

    with torch.no_grad():
        module.hada_w1_a.fill_(0.2)
        module.hada_w1_b.fill_(0.3)
        module.hada_w2_a.fill_(0.4)
        module.hada_w2_b.fill_(0.5)

    inputs = torch.randn(5, 4)
    actual = module(inputs)
    expected = target(inputs) + module.bypass_forward_diff(inputs, scale=module.multiplier)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_loha_module_module_dropout_can_skip_adapter_path():
    torch.manual_seed(29)
    target = torch.nn.Linear(4, 3, bias=False)
    module = LohaModule.from_target_module("loha_linear", target, config=LohaConfig(lora_dim=2, alpha=2.0, module_dropout=1.0))
    module.train()

    with torch.no_grad():
        module.hada_w1_a.fill_(0.2)
        module.hada_w1_b.fill_(0.3)
        module.hada_w2_a.fill_(0.4)
        module.hada_w2_b.fill_(0.5)

    inputs = torch.randn(5, 4)

    assert torch.allclose(module(inputs), target(inputs), atol=1e-6, rtol=1e-5)


def test_loha_module_rank_dropout_scaling_preserves_mask_but_changes_kept_rows():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=False)
    base_config = LohaConfig(lora_dim=4, alpha=4.0, rank_dropout=0.5)
    module = LohaModule.from_target_module("loha_linear", target, config=base_config)
    module_scaled = LohaModule.from_target_module(
        "loha_linear_scaled",
        target,
        config=LohaConfig(lora_dim=4, alpha=4.0, rank_dropout=0.5, rank_dropout_scale=True),
    )

    with torch.no_grad():
        module.hada_w1_a.fill_(0.2)
        module.hada_w1_b.fill_(0.3)
        module.hada_w2_a.fill_(0.4)
        module.hada_w2_b.fill_(0.5)
        module_scaled.hada_w1_a.copy_(module.hada_w1_a)
        module_scaled.hada_w1_b.copy_(module.hada_w1_b)
        module_scaled.hada_w2_a.copy_(module.hada_w2_a)
        module_scaled.hada_w2_b.copy_(module.hada_w2_b)

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


def test_loha_module_rs_lora_uses_sqrt_rank_scaling():
    torch.manual_seed(37)
    target = torch.nn.Linear(4, 4, bias=False)
    standard = LohaModule.from_target_module("loha_standard", target, config=LohaConfig(lora_dim=4, alpha=4.0))
    rs_scaled = LohaModule.from_target_module("loha_rs", target, config=LohaConfig(lora_dim=4, alpha=4.0, rs_lora=True))

    with torch.no_grad():
        standard.hada_w1_a.fill_(0.2)
        standard.hada_w1_b.fill_(0.3)
        standard.hada_w2_a.fill_(0.4)
        standard.hada_w2_b.fill_(0.5)
        rs_scaled.hada_w1_a.copy_(standard.hada_w1_a)
        rs_scaled.hada_w1_b.copy_(standard.hada_w1_b)
        rs_scaled.hada_w2_a.copy_(standard.hada_w2_a)
        rs_scaled.hada_w2_b.copy_(standard.hada_w2_b)

    expected_ratio = math.sqrt(standard.lora_dim)
    assert torch.allclose(rs_scaled.get_weight(rs_scaled.shape), standard.get_weight(standard.shape) * expected_ratio)


def test_loha_module_export_round_trip_preserves_tucker_weight():
    torch.manual_seed(11)
    target = torch.nn.Conv2d(2, 3, kernel_size=3, bias=False)
    module = LohaModule.from_target_module(
        "loha_conv",
        target,
        config=LohaConfig(lora_dim=2, alpha=4.0, use_tucker=True),
    )

    with torch.no_grad():
        module.hada_t1.normal_(mean=0.0, std=0.2)
        module.hada_t2.normal_(mean=0.0, std=0.2)
        module.hada_w1_a.normal_(mean=0.0, std=0.2)
        module.hada_w1_b.normal_(mean=0.0, std=0.2)
        module.hada_w2_a.normal_(mean=0.0, std=0.2)
        module.hada_w2_b.normal_(mean=0.0, std=0.2)

    state_dict = module.export_state_dict()
    restored = LohaModule.make_module_from_state_dict(
        "loha_conv",
        target,
        *(state_dict.get(key) for key in LohaModule.export_weight_keys),
    )

    assert restored.tucker is True
    assert torch.allclose(restored.get_weight(restored.shape), module.get_weight(module.shape), atol=1e-6, rtol=1e-5)


def test_loha_module_wd_on_output_changes_merged_weight_behavior():
    torch.manual_seed(41)
    target = torch.nn.Linear(4, 3, bias=False)
    output_wd = LohaModule.from_target_module(
        "loha_output_wd",
        target,
        config=LohaConfig(lora_dim=2, alpha=2.0, weight_decompose=True, wd_on_output=True),
    )
    input_wd = LohaModule.from_target_module(
        "loha_input_wd",
        target,
        config=LohaConfig(lora_dim=2, alpha=2.0, weight_decompose=True, wd_on_output=False),
    )

    with torch.no_grad():
        output_wd.hada_w1_a.fill_(0.2)
        output_wd.hada_w1_b.fill_(0.3)
        output_wd.hada_w2_a.fill_(0.4)
        output_wd.hada_w2_b.fill_(0.5)
        input_wd.hada_w1_a.copy_(output_wd.hada_w1_a)
        input_wd.hada_w1_b.copy_(output_wd.hada_w1_b)
        input_wd.hada_w2_a.copy_(output_wd.hada_w2_a)
        input_wd.hada_w2_b.copy_(output_wd.hada_w2_b)
        output_wd.dora_scale.fill_(1.0)
        input_wd.dora_scale.fill_(1.0)

    merged_output, _ = output_wd.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    merged_input, _ = input_wd.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert output_wd.dora_scale.shape != input_wd.dora_scale.shape
    assert not torch.allclose(merged_output, merged_input)


def test_loha_module_merged_weight_matches_forward_output_for_conv1d_target():
    torch.manual_seed(43)
    target = torch.nn.Conv1d(2, 3, kernel_size=1, bias=False)
    module = LohaModule.from_target_module("loha_conv1d", target, config=LohaConfig(lora_dim=2, alpha=2.0))
    module.eval()

    with torch.no_grad():
        module.hada_w1_a.fill_(0.2)
        module.hada_w1_b.fill_(0.3)
        module.hada_w2_a.fill_(0.4)
        module.hada_w2_b.fill_(0.5)

    inputs = torch.randn(4, 2, 5)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)

    actual = module(inputs)
    expected = F.conv1d(inputs, merged_weight.to(inputs.dtype), merged_bias, stride=target.stride, padding=target.padding)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_loha_module_merged_weight_matches_forward_output_for_conv3d_target():
    torch.manual_seed(47)
    target = torch.nn.Conv3d(2, 3, kernel_size=1, bias=False)
    module = LohaModule.from_target_module("loha_conv3d", target, config=LohaConfig(lora_dim=2, alpha=2.0))
    module.eval()

    with torch.no_grad():
        module.hada_w1_a.fill_(0.2)
        module.hada_w1_b.fill_(0.3)
        module.hada_w2_a.fill_(0.4)
        module.hada_w2_b.fill_(0.5)

    inputs = torch.randn(2, 2, 3, 3, 3)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)

    actual = module(inputs)
    expected = F.conv3d(inputs, merged_weight.to(inputs.dtype), merged_bias, stride=target.stride, padding=target.padding)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
