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
