import pytest
import torch
import torch.nn.functional as F

from library.adapters.methods.peft.vera.module import VeraConfig, VeraModule, VeraSharedProjectionBank


def _fill_nonzero_lambda_weights(module: VeraModule) -> None:
    with torch.no_grad():
        module.vera_lambda_b.copy_(torch.randn_like(module.vera_lambda_b) * 0.1)
        module.vera_lambda_d.copy_(torch.randn_like(module.vera_lambda_d) * 0.1)


def test_vera_module_rejects_non_positive_rank():
    target = torch.nn.Linear(4, 4, bias=False)
    shared_bank = VeraSharedProjectionBank(rank=1, max_in_features=4, max_out_features=4)

    with pytest.raises(ValueError, match="rank must be positive"):
        VeraModule.from_target_module("vera_linear", target, shared_bank=shared_bank, config=VeraConfig(rank=0))


def test_vera_module_default_init_preserves_base_forward():
    torch.manual_seed(17)
    target = torch.nn.Linear(4, 4, bias=True)
    shared_bank = VeraSharedProjectionBank(rank=3, max_in_features=4, max_out_features=4)
    inputs = torch.randn(5, 4)

    expected = target(inputs)
    module = VeraModule.from_target_module("vera_linear", target, shared_bank=shared_bank, config=VeraConfig(rank=3, dropout=0.2))
    actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(module.vera_lambda_b).item() == 0


def test_vera_module_merged_weight_matches_forward_output():
    torch.manual_seed(23)
    target = torch.nn.Linear(4, 3, bias=True)
    shared_bank = VeraSharedProjectionBank(rank=2, max_in_features=6, max_out_features=5)
    module = VeraModule.from_target_module("vera_linear", target, shared_bank=shared_bank, config=VeraConfig(rank=2))
    module.eval()
    _fill_nonzero_lambda_weights(module)

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)
    actual = module(inputs)
    expected = F.linear(inputs.to(merged_weight.dtype), merged_weight, merged_bias).to(inputs.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_vera_module_export_round_trip_preserves_merged_weight():
    torch.manual_seed(29)
    target = torch.nn.Linear(4, 3, bias=False)
    shared_bank = VeraSharedProjectionBank(rank=2, max_in_features=4, max_out_features=3)
    module = VeraModule.from_target_module("vera_linear", target, shared_bank=shared_bank, config=VeraConfig(rank=2))
    _fill_nonzero_lambda_weights(module)

    shared_state = shared_bank.export_state_dict()
    restored_bank = VeraSharedProjectionBank.from_state_dict(shared_state["vera_A"], shared_state["vera_B"])
    state_dict = module.export_state_dict()
    restored = VeraModule.make_module_from_state_dict(
        "vera_linear",
        target,
        restored_bank,
        state_dict["vera_lambda_b"],
        state_dict["vera_lambda_d"],
    )

    original_merged_weight, _ = module.get_merged_weight(shape=target.weight.shape, device=target.weight.device)
    restored_merged_weight, _ = restored.get_merged_weight(shape=target.weight.shape, device=target.weight.device)

    assert torch.allclose(restored_merged_weight, original_merged_weight, atol=1e-6, rtol=1e-5)


def test_vera_shared_bank_skips_projection_export_when_disabled():
    shared_bank = VeraSharedProjectionBank(
        rank=2,
        max_in_features=4,
        max_out_features=3,
        projection_prng_key=11,
        save_projection=False,
    )

    assert shared_bank.export_state_dict() == {}


def test_vera_shared_bank_reset_from_prng_key_restores_deterministic_projections():
    shared_bank = VeraSharedProjectionBank(
        rank=2,
        max_in_features=4,
        max_out_features=3,
        projection_prng_key=11,
        save_projection=False,
    )
    expected_vera_A = shared_bank.vera_A.detach().clone()
    expected_vera_B = shared_bank.vera_B.detach().clone()

    shared_bank.reset_from_prng_key(19)
    assert not torch.allclose(shared_bank.vera_A, expected_vera_A)
    assert not torch.allclose(shared_bank.vera_B, expected_vera_B)

    shared_bank.reset_from_prng_key(11)
    assert torch.allclose(shared_bank.vera_A, expected_vera_A)
    assert torch.allclose(shared_bank.vera_B, expected_vera_B)


def test_vera_module_forward_accepts_mixed_input_dtype_and_restores_original_dtype():
    torch.manual_seed(31)
    target = torch.nn.Linear(4, 4, bias=True)
    shared_bank = VeraSharedProjectionBank(rank=2, max_in_features=4, max_out_features=4)
    module = VeraModule.from_target_module("vera_linear", target, shared_bank=shared_bank, config=VeraConfig(rank=2))
    inputs = torch.randn(5, 4, dtype=torch.float64)

    actual = module(inputs)
    expected = F.linear(inputs.to(target.weight.dtype), target.weight, target.bias).to(inputs.dtype)

    assert actual.dtype == inputs.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_vera_module_supports_transformers_conv1d():
    transformers_pytorch_utils = pytest.importorskip("transformers.pytorch_utils")
    Conv1D = transformers_pytorch_utils.Conv1D

    torch.manual_seed(37)
    target = Conv1D(3, 4)
    shared_bank = VeraSharedProjectionBank(rank=2, max_in_features=4, max_out_features=3)
    module = VeraModule.from_target_module("vera_conv1d", target, shared_bank=shared_bank, config=VeraConfig(rank=2))
    module.eval()
    _fill_nonzero_lambda_weights(module)

    inputs = torch.randn(5, 4)
    merged_weight, merged_bias = module.get_merged_weight(shape=target.weight.shape, device=inputs.device)
    actual = module(inputs)
    expected = torch.matmul(inputs.to(merged_weight.dtype), merged_weight)
    if merged_bias is not None:
        expected = expected + merged_bias
    expected = expected.to(actual.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
