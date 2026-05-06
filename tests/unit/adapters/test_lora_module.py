import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.lora.module import LoraConfig, LoraModule


def _fill_nonzero_weights(module: LoraModule) -> None:
    with torch.no_grad():
        module.lora_down.weight.copy_(torch.randn_like(module.lora_down.weight) * 0.1)
        module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)


def test_lora_module_rejects_non_positive_rank():
    target = torch.nn.Linear(4, 4, bias=False)

    with pytest.raises(ValueError, match="rank must be positive"):
        LoraModule.from_target_module("lora_linear", target, config=LoraConfig(lora_dim=0))


def test_lora_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 4, bias=True)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = LoraModule.from_target_module("lora_linear", target, config=LoraConfig(lora_dim=4, alpha=2.0))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.lora_up.weight).item() == 0


def test_lora_module_merged_weight_matches_forward_output_for_linear_target():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 4, bias=True)
    module = LoraModule.from_target_module("lora_linear", target, config=LoraConfig(lora_dim=2, alpha=4.0))
    module.eval()
    _fill_nonzero_weights(module)

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)
    actual = module(inputs)
    expected = F.linear(inputs.to(merged_weight.dtype), merged_weight, merged_bias).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lora_module_merged_weight_matches_forward_output_for_conv2d_target():
    torch.manual_seed(29)
    target = torch.nn.Conv2d(4, 6, kernel_size=3, padding=1, bias=False)
    module = LoraModule.from_target_module("lora_conv", target, config=LoraConfig(lora_dim=3, alpha=3.0))
    module.eval()
    _fill_nonzero_weights(module)

    inputs = torch.randn(2, 4, 8, 8)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)
    actual = module(inputs)
    expected = F.conv2d(inputs.to(merged_weight.dtype), merged_weight, merged_bias, stride=target.stride, padding=target.padding).to(
        inputs.dtype
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lora_module_forward_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=True)
    module = LoraModule.from_target_module("lora_linear", target, config=LoraConfig(lora_dim=2, alpha=2.0))
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lora_module_uses_hot_path_under_autocast_even_when_adapter_dtype_differs():
    target = torch.nn.Linear(4, 4, bias=True)
    module = LoraModule.from_target_module("lora_linear", target, config=LoraConfig(lora_dim=2, alpha=2.0))
    module.to(dtype=torch.float32)
    inputs = torch.randn(5, 4, dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert module._needs_explicit_dtype_fallback(inputs) is False


def test_lora_module_hot_path_matches_manual_reference_under_autocast():
    torch.manual_seed(33)
    target = torch.nn.Linear(4, 4, bias=True)
    module = LoraModule.from_target_module("lora_linear", target, config=LoraConfig(lora_dim=2, alpha=2.0))
    module.eval()
    _fill_nonzero_weights(module)
    inputs = torch.randn(5, 4, dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = module(inputs)

    org = F.linear(inputs.float(), target.weight.float(), target.bias.float())
    lora_hidden = F.linear(inputs.float(), module.lora_down.weight.float())
    lora_out = F.linear(lora_hidden, module.lora_up.weight.float())
    expected = (org + lora_out * (module.multiplier * module.scale)).to(actual.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_lora_module_rank_dropout_uses_last_axis_for_linear_outputs():
    target = torch.nn.Linear(4, 4, bias=False)
    module = LoraModule.from_target_module("lora_linear", target, config=LoraConfig(lora_dim=2, rank_dropout=0.5))
    module.train()

    rank_activations = torch.ones(3, 5, 2)
    torch.manual_seed(7)
    dropped, scale = module._apply_rank_dropout(rank_activations)

    assert dropped.shape == rank_activations.shape
    assert scale == pytest.approx(module.scale * 2.0)

    # Rank dropout should broadcast over the sequence axis for Linear outputs
    # shaped like [batch, seq, rank].
    assert torch.equal(dropped[:, 0, :], dropped[:, 1, :])
    assert torch.equal(dropped[:, 1, :], dropped[:, 2, :])


def test_lora_module_export_round_trip_preserves_merged_weight():
    torch.manual_seed(37)
    target = torch.nn.Conv1d(4, 6, kernel_size=1, bias=False)
    module = LoraModule.from_target_module("lora_conv1d", target, config=LoraConfig(lora_dim=2, alpha=2.0))
    _fill_nonzero_weights(module)

    state_dict = module.export_state_dict()
    restored = LoraModule.make_module_from_state_dict(
        "lora_conv1d",
        target,
        state_dict["lora_up.weight"],
        state_dict["lora_down.weight"],
        state_dict["alpha"],
    )

    original_merged_weight, _ = module.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    restored_merged_weight, _ = restored.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert torch.allclose(restored_merged_weight, original_merged_weight, atol=1e-6, rtol=1e-5)


def test_lora_module_module_dropout_can_skip_adapter_path():
    torch.manual_seed(41)
    target = torch.nn.Linear(4, 4, bias=True)
    module = LoraModule.from_target_module("lora_linear", target, config=LoraConfig(lora_dim=2, module_dropout=1.0))
    module.train()
    _fill_nonzero_weights(module)
    inputs = torch.randn(5, 4)

    assert torch.allclose(module(inputs), target(inputs), atol=1e-6, rtol=1e-5)
