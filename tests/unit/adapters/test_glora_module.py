import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.glora.module import GloraConfig, GloraModule


def _fill_nonzero_weights(module: GloraModule) -> None:
    with torch.no_grad():
        module.a1.weight.copy_(torch.randn_like(module.a1.weight) * 0.1)
        module.a2.weight.copy_(torch.randn_like(module.a2.weight) * 0.1)
        module.b1.weight.copy_(torch.randn_like(module.b1.weight) * 0.1)
        module.b2.weight.copy_(torch.randn_like(module.b2.weight) * 0.1)
        if module.tucker and module.bm is not None:
            module.bm.weight.copy_(torch.randn_like(module.bm.weight) * 0.1)


def test_glora_module_rejects_non_positive_rank():
    target = torch.nn.Linear(4, 4, bias=False)

    with pytest.raises(ValueError, match="rank must be positive"):
        GloraModule.from_target_module("glora_linear", target, config=GloraConfig(lora_dim=0))


def test_glora_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 4, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = GloraModule.from_target_module("glora_linear", target, config=GloraConfig(lora_dim=4, alpha=2.0))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.a2.weight).item() == 0
    assert torch.count_nonzero(module.b2.weight).item() == 0


def test_glora_module_merged_weight_matches_forward_output_for_linear_target():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 4, bias=False)
    module = GloraModule.from_target_module("glora_linear", target, config=GloraConfig(lora_dim=4, alpha=2.0))
    module.eval()
    _fill_nonzero_weights(module)

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)

    actual = module(inputs)
    expected = F.linear(inputs.to(merged_weight.dtype), merged_weight, merged_bias).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_glora_module_bypass_mode_multiplier_scales_diff_output():
    torch.manual_seed(29)
    target = torch.nn.Linear(4, 4, bias=False)
    module = GloraModule.from_target_module(
        "glora_linear",
        target,
        config=GloraConfig(lora_dim=4, alpha=2.0, bypass_mode=True),
    )
    module.eval()
    _fill_nonzero_weights(module)
    inputs = torch.randn(5, 4)

    diff_unit = module.bypass_forward_diff(inputs, scale=1.0)
    diff_double = module.bypass_forward_diff(inputs, scale=2.0)

    assert torch.allclose(diff_double, diff_unit * 2.0, atol=1e-5, rtol=1e-4)


def test_glora_module_forward_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=False)
    module = GloraModule.from_target_module("glora_linear", target, config=GloraConfig(lora_dim=4))
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_glora_module_bypass_mode_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(37)
    target = torch.nn.Linear(4, 4, bias=False)
    module = GloraModule.from_target_module(
        "glora_linear",
        target,
        config=GloraConfig(lora_dim=4, alpha=2.0, bypass_mode=True),
    )
    module.eval()
    _fill_nonzero_weights(module)

    inputs = torch.randn(5, 4, dtype=torch.float64)
    actual = module(inputs)
    expected = module.bypass_forward(inputs, scale=module.multiplier)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_glora_module_tucker_path_is_reachable_for_conv_targets():
    torch.manual_seed(41)
    target = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
    module = GloraModule.from_target_module(
        "glora_conv",
        target,
        config=GloraConfig(lora_dim=4, alpha=2.0, use_tucker=True),
    )

    assert module.tucker is True
    assert module.bm is not None
    _fill_nonzero_weights(module)

    state_dict = module.export_state_dict()
    restored = GloraModule.make_module_from_state_dict(
        "glora_conv",
        target,
        state_dict["a1.weight"],
        state_dict["a2.weight"],
        state_dict["b1.weight"],
        state_dict["b2.weight"],
        state_dict["bm.weight"],
        state_dict["alpha"],
    )

    original_merged, _ = module.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    restored_merged, _ = restored.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert torch.allclose(restored_merged, original_merged, atol=1e-6, rtol=1e-5)


def test_glora_module_export_round_trip_preserves_merged_weight():
    torch.manual_seed(43)
    target = torch.nn.Linear(4, 4, bias=False)
    module = GloraModule.from_target_module(
        "glora_linear",
        target,
        config=GloraConfig(lora_dim=4, alpha=2.0, use_scalar=True),
    )
    _fill_nonzero_weights(module)

    state_dict = module.export_state_dict()
    restored = GloraModule.make_module_from_state_dict(
        "glora_linear",
        target,
        state_dict["a1.weight"],
        state_dict["a2.weight"],
        state_dict["b1.weight"],
        state_dict["b2.weight"],
        state_dict.get("bm.weight"),
        state_dict["alpha"],
    )

    original_merged, _ = module.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    restored_merged, _ = restored.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert torch.allclose(restored_merged, original_merged, atol=1e-6, rtol=1e-5)


def test_glora_module_module_dropout_can_skip_adapter_path():
    torch.manual_seed(47)
    target = torch.nn.Linear(4, 4, bias=False)
    module = GloraModule.from_target_module(
        "glora_linear",
        target,
        config=GloraConfig(lora_dim=4, alpha=2.0, module_dropout=1.0),
    )
    module.train()
    _fill_nonzero_weights(module)
    inputs = torch.randn(5, 4)

    assert torch.allclose(module(inputs), target(inputs), atol=1e-6, rtol=1e-5)
