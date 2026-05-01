import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.dylora.module import DyloraConfig, DyloraModule


def _fill_nonzero_weights(module: DyloraModule) -> None:
    with torch.no_grad():
        module.lora_down_blocks.copy_(torch.randn_like(module.lora_down_blocks) * 0.1)
        module.lora_up_blocks.copy_(torch.randn_like(module.lora_up_blocks) * 0.1)


def test_dylora_module_rejects_invalid_block_size():
    target = torch.nn.Linear(4, 4, bias=False)

    with pytest.raises(ValueError, match="must be divisible by block_size"):
        DyloraModule.from_target_module("dylora_linear", target, config=DyloraConfig(lora_dim=4, block_size=3))


def test_dylora_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 4, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = DyloraModule.from_target_module("dylora_linear", target, config=DyloraConfig(lora_dim=4, alpha=2.0))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.lora_up_blocks).item() == 0


def test_dylora_module_merged_weight_matches_forward_output_for_linear_target():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 4, bias=False)
    module = DyloraModule.from_target_module("dylora_linear", target, config=DyloraConfig(lora_dim=4, alpha=2.0))
    module.eval()
    _fill_nonzero_weights(module)

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)

    actual = module(inputs)
    expected = F.linear(inputs.to(merged_weight.dtype), merged_weight, merged_bias).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_dylora_module_forward_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(29)
    target = torch.nn.Linear(4, 4, bias=False)
    module = DyloraModule.from_target_module("dylora_linear", target, config=DyloraConfig(lora_dim=4))
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_dylora_module_bypass_mode_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=False)
    module = DyloraModule.from_target_module(
        "dylora_linear",
        target,
        config=DyloraConfig(lora_dim=4, alpha=2.0, bypass_mode=True),
    )
    module.eval()
    _fill_nonzero_weights(module)

    inputs = torch.randn(5, 4, dtype=torch.float64)
    actual = module(inputs)

    base = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias)
    diff = module.bypass_forward_diff(inputs, scale=module.multiplier)
    expected = (base + diff.to(base.dtype)).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_dylora_module_training_only_updates_current_block(monkeypatch):
    torch.manual_seed(37)
    target = torch.nn.Linear(4, 4, bias=False)
    module = DyloraModule.from_target_module(
        "dylora_linear",
        target,
        config=DyloraConfig(lora_dim=4, alpha=2.0, block_size=2),
    )
    _fill_nonzero_weights(module)
    module.train()
    monkeypatch.setattr(module, "_sample_active_blocks", lambda: 2)
    inputs = torch.randn(5, 4)

    output = module(inputs)
    output.sum().backward()

    assert module.lora_up_blocks.grad is not None
    assert module.lora_down_blocks.grad is not None
    assert torch.count_nonzero(module.lora_up_blocks.grad[0]).item() == 0
    assert torch.count_nonzero(module.lora_down_blocks.grad[0]).item() == 0
    assert torch.count_nonzero(module.lora_up_blocks.grad[1]).item() > 0
    assert torch.count_nonzero(module.lora_down_blocks.grad[1]).item() > 0


def test_dylora_module_export_round_trip_preserves_merged_weight_and_block_size():
    torch.manual_seed(41)
    target = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
    module = DyloraModule.from_target_module(
        "dylora_conv",
        target,
        config=DyloraConfig(lora_dim=4, alpha=2.0, block_size=2),
    )
    _fill_nonzero_weights(module)

    state_dict = module.export_state_dict()
    restored = DyloraModule.make_module_from_state_dict(
        "dylora_conv",
        target,
        state_dict["lora_up.weight"],
        state_dict["lora_down.weight"],
        state_dict["alpha"],
        state_dict["block_size"],
    )

    original_merged, _ = module.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    restored_merged, _ = restored.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert restored.block_size == 2
    assert torch.allclose(restored_merged, original_merged, atol=1e-6, rtol=1e-5)


def test_dylora_module_export_uses_standard_lora_weight_shapes():
    target = torch.nn.Linear(4, 4, bias=False)
    module = DyloraModule.from_target_module("dylora_linear", target, config=DyloraConfig(lora_dim=4, block_size=2))
    _fill_nonzero_weights(module)

    exported = module.export_state_dict()

    assert exported["lora_up.weight"].shape == (4, 4)
    assert exported["lora_down.weight"].shape == (4, 4)
    assert int(exported["block_size"].item()) == 2


def test_dylora_module_module_dropout_can_skip_adapter_path():
    torch.manual_seed(43)
    target = torch.nn.Linear(4, 4, bias=False)
    module = DyloraModule.from_target_module(
        "dylora_linear",
        target,
        config=DyloraConfig(lora_dim=4, alpha=2.0, module_dropout=1.0),
    )
    module.train()
    _fill_nonzero_weights(module)
    inputs = torch.randn(5, 4)

    assert torch.allclose(module(inputs), target(inputs), atol=1e-6, rtol=1e-5)
