import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.oft.module import OftConfig, OftModule


def _fill_active_oft_blocks(module: OftModule) -> None:
    assert module.block_num == 2
    assert module.block_size == 2
    with torch.no_grad():
        module.oft_blocks.copy_(
            torch.tensor(
                [
                    [[0.0, 0.2], [-0.1, 0.0]],
                    [[0.0, -0.3], [0.4, 0.0]],
                ],
                dtype=module.oft_blocks.dtype,
            )
        )


def test_oft_module_rejects_non_positive_factor():
    target = torch.nn.Linear(4, 4, bias=False)

    with pytest.raises(ValueError, match="factor must be positive"):
        OftModule.from_target_module("oft_linear", target, config=OftConfig(factor=0))


def test_oft_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 4, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = OftModule.from_target_module("oft_linear", target, config=OftConfig(factor=2, constraint=1e-3))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.oft_blocks).item() == 0


def test_oft_module_merged_weight_matches_forward_output_for_linear_target():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 4, bias=False)
    module = OftModule.from_target_module("oft_linear", target, config=OftConfig(factor=2, constraint=0.5))
    module.eval()
    _fill_active_oft_blocks(module)

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)

    actual = module(inputs)
    expected = F.linear(inputs.to(merged_weight.dtype), merged_weight, merged_bias).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_oft_module_forward_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(29)
    target = torch.nn.Linear(4, 4, bias=False)
    module = OftModule.from_target_module("oft_linear", target, config=OftConfig(factor=2, constraint=1e-3))
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_oft_module_bypass_mode_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=False)
    module = OftModule.from_target_module("oft_linear", target, config=OftConfig(factor=2, constraint=0.5, bypass_mode=True))
    module.eval()
    _fill_active_oft_blocks(module)

    inputs = torch.randn(5, 4, dtype=torch.float64)
    actual = module(inputs)

    base = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias)
    diff = module.bypass_forward_diff(inputs, scale=module.multiplier)
    expected = (base + diff.to(base.dtype)).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_oft_module_plain_dropout_changes_bypass_diff():
    torch.manual_seed(37)
    target = torch.nn.Linear(4, 4, bias=False)
    module = OftModule.from_target_module(
        "oft_linear",
        target,
        config=OftConfig(factor=2, constraint=0.5, bypass_mode=True, dropout=0.5),
    )
    module.train()
    _fill_active_oft_blocks(module)
    inputs = torch.randn(5, 4)

    module.dropout = 0.0
    torch.manual_seed(101)
    without_plain_dropout = module.bypass_forward_diff(inputs, scale=module.multiplier)
    module.dropout = 0.5
    torch.manual_seed(101)
    with_plain_dropout = module.bypass_forward_diff(inputs, scale=module.multiplier)

    assert not torch.allclose(with_plain_dropout, without_plain_dropout)


def test_oft_module_export_round_trip_preserves_rescaled_merged_weight():
    torch.manual_seed(41)
    target = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
    module = OftModule.from_target_module(
        "oft_conv",
        target,
        config=OftConfig(factor=2, constraint=0.5, rescaled=True),
    )
    _fill_active_oft_blocks(module)

    with torch.no_grad():
        assert module.rescale is not None
        module.rescale.normal_(mean=1.0, std=0.1)

    state_dict = module.export_state_dict()
    restored = OftModule.make_module_from_state_dict(
        "oft_conv",
        target,
        *(state_dict.get(key) for key in OftModule.export_weight_keys),
    )

    original_merged, _ = module.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    restored_merged, _ = restored.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert torch.allclose(restored_merged, original_merged, atol=1e-6, rtol=1e-5)


def test_oft_module_module_dropout_can_skip_adapter_path():
    torch.manual_seed(43)
    target = torch.nn.Linear(4, 4, bias=False)
    module = OftModule.from_target_module(
        "oft_linear",
        target,
        config=OftConfig(factor=2, constraint=0.5, module_dropout=1.0),
    )
    module.train()
    _fill_active_oft_blocks(module)
    inputs = torch.randn(5, 4)

    assert torch.allclose(module(inputs), target(inputs), atol=1e-6, rtol=1e-5)
