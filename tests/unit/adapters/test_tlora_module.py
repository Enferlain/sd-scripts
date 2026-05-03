import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.tlora.module import TloraConfig, TloraModule, compute_timestep_mask_batch


def _perturb_tlora_module(module: TloraModule) -> None:
    with torch.no_grad():
        module.q_layer.weight.add_(torch.randn_like(module.q_layer.weight) * 0.05)
        module.p_layer.weight.add_(torch.randn_like(module.p_layer.weight) * 0.05)
        module.lambda_layer.add_(0.25)


def test_tlora_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 4, bias=False)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = TloraModule.from_target_module("tlora_linear", target, config=TloraConfig(lora_dim=4, alpha=4.0))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_tlora_module_batched_mask_forces_bypass_and_restores_input_dtype():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 4, bias=False)
    module = TloraModule.from_target_module("tlora_linear", target, config=TloraConfig(lora_dim=4, alpha=4.0))
    module.eval()
    _perturb_tlora_module(module)
    module.set_timestep_mask(torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))

    inputs = torch.randn(2, 4, dtype=torch.float64)
    expected = module.bypass_forward(inputs, scale=module.multiplier)
    actual = module(inputs)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    module.clear_timestep_mask()


def test_tlora_module_get_diff_weight_rejects_batched_mask():
    target = torch.nn.Linear(4, 4, bias=False)
    module = TloraModule.from_target_module("tlora_linear", target, config=TloraConfig(lora_dim=4))
    module.set_timestep_mask(torch.ones(2, 4))

    with pytest.raises(ValueError, match="batched timestep masks require bypass-mode execution"):
        module.get_diff_weight()

    module.clear_timestep_mask()


def test_tlora_module_export_round_trip_preserves_merged_weight():
    torch.manual_seed(29)
    target = torch.nn.Linear(4, 4, bias=False)
    module = TloraModule.from_target_module(
        "tlora_linear",
        target,
        config=TloraConfig(lora_dim=4, alpha=4.0, use_scalar=True),
    )
    _perturb_tlora_module(module)
    with torch.no_grad():
        module.scalar.fill_(1.75)

    state_dict = module.export_state_dict()
    restored = TloraModule.make_module_from_state_dict(
        "tlora_linear",
        target,
        state_dict["q_layer.weight"],
        state_dict["p_layer.weight"],
        state_dict["lambda_layer"],
        state_dict["alpha"],
        state_dict["base_q"],
        state_dict["base_p"],
        state_dict["base_lambda"],
    )

    original_merged, _ = module.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    restored_merged, _ = restored.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert torch.allclose(restored_merged, original_merged, atol=1e-6, rtol=1e-5)


def test_tlora_module_merged_weight_matches_forward_output_for_linear_target():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=False)
    module = TloraModule.from_target_module("tlora_linear", target, config=TloraConfig(lora_dim=4, alpha=4.0))
    module.eval()
    _perturb_tlora_module(module)

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)
    actual = module(inputs)
    expected = F.linear(inputs.to(merged_weight.dtype), merged_weight, merged_bias).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_compute_timestep_mask_batch_respects_endpoint_ranks():
    timesteps = torch.tensor([1000, 0], dtype=torch.long)

    mask = compute_timestep_mask_batch(
        timesteps,
        max_timestep=1000,
        max_rank=4,
        min_rank=2,
        alpha=1.0,
    )

    assert mask.shape == (2, 4)
    assert mask[0].sum().item() == 2
    assert mask[1].sum().item() == 4
