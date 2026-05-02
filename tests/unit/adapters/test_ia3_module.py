import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.ia3.module import Ia3Config, Ia3Module


def _fill_nonzero_weight(module: Ia3Module, value: float = 0.25) -> None:
    with torch.no_grad():
        module.weight.fill_(value)


def test_ia3_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 4, bias=True)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = Ia3Module.from_target_module("ia3_linear", target, config=Ia3Config())
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.weight).item() == 0


def test_ia3_module_merged_weight_matches_forward_output_for_output_scaling_biasful_target():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 4, bias=True)
    module = Ia3Module.from_target_module("ia3_linear", target, config=Ia3Config())
    _fill_nonzero_weight(module)

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)
    actual = module(inputs)
    expected = F.linear(inputs.to(merged_weight.dtype), merged_weight, merged_bias).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert merged_bias is not None
    assert not torch.allclose(merged_bias, target.bias)


def test_ia3_module_merged_weight_matches_forward_output_for_input_scaling_biasful_target():
    torch.manual_seed(29)
    target = torch.nn.Linear(6, 4, bias=True)
    module = Ia3Module.from_target_module("ia3_linear", target, config=Ia3Config(train_on_input=True))
    _fill_nonzero_weight(module)

    inputs = torch.randn(5, 6)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)
    actual = module(inputs)
    expected = F.linear(inputs.to(merged_weight.dtype), merged_weight, merged_bias).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert merged_bias is not None
    assert torch.allclose(merged_bias, target.bias.to(merged_bias.dtype), atol=1e-6, rtol=1e-5)


def test_ia3_module_forward_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=True)
    module = Ia3Module.from_target_module("ia3_linear", target, config=Ia3Config())
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_ia3_module_bypass_mode_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(37)
    target = torch.nn.Linear(4, 4, bias=True)
    module = Ia3Module.from_target_module("ia3_linear", target, config=Ia3Config(bypass_mode=True))
    _fill_nonzero_weight(module)
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = module.bypass_forward(inputs, scale=module.multiplier)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_ia3_module_bypass_diff_omits_bias_for_input_scaling():
    torch.manual_seed(41)
    target = torch.nn.Linear(6, 4, bias=True)
    module = Ia3Module.from_target_module("ia3_linear", target, config=Ia3Config(train_on_input=True, bypass_mode=True))
    _fill_nonzero_weight(module)
    inputs = torch.randn(5, 6)

    actual = module.bypass_forward_diff(inputs, scale=module.multiplier)
    scale_delta = module.weight.detach() * module.multiplier
    expected = F.linear(
        inputs.to(target.weight.dtype) * scale_delta.view(1, -1),
        target.weight,
        None,
    ).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_ia3_module_export_round_trip_preserves_weight_bias_and_axis_selection():
    torch.manual_seed(43)
    target = torch.nn.Conv2d(4, 6, kernel_size=3, padding=1, bias=True)
    module = Ia3Module.from_target_module("ia3_conv", target, config=Ia3Config(train_on_input=True))
    _fill_nonzero_weight(module)

    state_dict = module.export_state_dict()
    restored = Ia3Module.make_module_from_state_dict(
        "ia3_conv",
        target,
        state_dict["weight"],
        state_dict["on_input"],
    )

    original_merged_weight, original_merged_bias = module.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    restored_merged_weight, restored_merged_bias = restored.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert restored.train_on_input is True
    assert torch.allclose(restored_merged_weight, original_merged_weight, atol=1e-6, rtol=1e-5)
    assert restored_merged_bias is not None
    assert original_merged_bias is not None
    assert torch.allclose(restored_merged_bias, original_merged_bias, atol=1e-6, rtol=1e-5)


def test_ia3_module_load_accepts_vendor_style_conv_weight_shape():
    target = torch.nn.Conv2d(4, 6, kernel_size=1, bias=False)
    module = Ia3Module.from_target_module("ia3_conv", target, config=Ia3Config(train_on_input=False))

    weights = {
        "weight": torch.full((1, 6, 1, 1), 0.125),
        "on_input": torch.tensor(0, dtype=torch.int64),
    }
    module.load_export_state_dict(weights)

    assert torch.allclose(module.weight, torch.full_like(module.weight, 0.125))


def test_ia3_module_load_rejects_train_on_input_mismatch():
    target = torch.nn.Linear(4, 6, bias=False)
    module = Ia3Module.from_target_module("ia3_linear", target, config=Ia3Config(train_on_input=False))

    with pytest.raises(ValueError, match="train_on_input mismatch"):
        module.load_export_state_dict(
            {
                "weight": torch.full((6,), 0.125),
                "on_input": torch.tensor(1, dtype=torch.int64),
            }
        )


def test_ia3_module_make_module_from_state_dict_rejects_ambiguous_square_layer_without_flag():
    target = torch.nn.Linear(4, 4, bias=False)

    with pytest.raises(ValueError, match="must store 'on_input' when input and output dimensions are ambiguous"):
        Ia3Module.make_module_from_state_dict(
            "ia3_linear",
            target,
            torch.full((4,), 0.125),
            None,
        )


def test_ia3_module_load_rejects_incompatible_conv_weight_shape():
    target = torch.nn.Conv2d(4, 6, kernel_size=1, bias=False)
    module = Ia3Module.from_target_module("ia3_conv", target, config=Ia3Config(train_on_input=False))

    with pytest.raises(ValueError, match="runtime expects 6 values, weights store 7"):
        module.load_export_state_dict(
            {
                "weight": torch.full((7,), 0.125),
                "on_input": torch.tensor(0, dtype=torch.int64),
            }
        )


def test_ia3_module_module_dropout_can_skip_adapter_path():
    torch.manual_seed(47)
    target = torch.nn.Linear(4, 4, bias=True)
    module = Ia3Module.from_target_module(
        "ia3_linear",
        target,
        config=Ia3Config(module_dropout=1.0),
    )
    module.train()
    _fill_nonzero_weight(module)
    inputs = torch.randn(5, 4)

    assert torch.allclose(module(inputs), target(inputs), atol=1e-6, rtol=1e-5)
