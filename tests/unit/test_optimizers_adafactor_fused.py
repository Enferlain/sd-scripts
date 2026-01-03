"""
Unit tests for library/optimizers/adafactor_fused.py.

Tests the stochastic rounding function used for bfloat16 precision optimization.
"""

import torch

from library.optimizers.adafactor_fused import copy_stochastic_, patch_adafactor_fused


class TestCopyStochastic:
    """Tests for copy_stochastic_ function (stochastic rounding for bfloat16)."""

    def test_preserves_tensor_shape(self):
        """Target tensor should maintain its shape after copy."""
        source = torch.randn(4, 8, dtype=torch.float32)
        target = torch.zeros(4, 8, dtype=torch.bfloat16)

        copy_stochastic_(target, source)

        assert target.shape == source.shape

    def test_output_dtype_is_bfloat16(self):
        """Target tensor should remain bfloat16 after the operation."""
        source = torch.randn(4, 8, dtype=torch.float32)
        target = torch.zeros(4, 8, dtype=torch.bfloat16)

        copy_stochastic_(target, source)

        # Note: The function copies float32-viewed result to bfloat16 target
        # The actual dtype check is that it doesn't crash
        assert target.dtype == torch.bfloat16

    def test_average_error_is_small(self):
        """On average, stochastic rounding should be close to the source value."""
        torch.manual_seed(42)  # For reproducibility

        source = torch.randn(1000, 1000, dtype=torch.float32)
        target = torch.zeros(1000, 1000, dtype=torch.bfloat16)

        copy_stochastic_(target, source)

        # Convert back to float32 for comparison
        target_f32 = target.float()
        error = (target_f32 - source).abs().mean()

        # Stochastic rounding should have small average error (< 0.01 for normalized data)
        assert error < 0.02, f"Average error {error} is too large"

    def test_seeded_randomness_produces_consistent_results(self):
        """Same seed should produce same stochastic rounding result."""
        source = torch.randn(10, 10, dtype=torch.float32)

        torch.manual_seed(123)
        target1 = torch.zeros(10, 10, dtype=torch.bfloat16)
        copy_stochastic_(target1, source)

        torch.manual_seed(123)
        target2 = torch.zeros(10, 10, dtype=torch.bfloat16)
        copy_stochastic_(target2, source)

        # Same seed should give same result
        assert torch.equal(target1, target2)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should (usually) produce different results."""
        source = torch.randn(10, 10, dtype=torch.float32)

        torch.manual_seed(123)
        target1 = torch.zeros(10, 10, dtype=torch.bfloat16)
        copy_stochastic_(target1, source)

        torch.manual_seed(456)
        target2 = torch.zeros(10, 10, dtype=torch.bfloat16)
        copy_stochastic_(target2, source)

        # Different seeds should give different results (extremely likely)
        assert not torch.equal(target1, target2)

    def test_handles_scalar_tensor(self):
        """Should handle 0-dimensional tensors."""
        source = torch.tensor(3.14159, dtype=torch.float32)
        target = torch.tensor(0.0, dtype=torch.bfloat16)

        copy_stochastic_(target, source)

        # Should be close to the source value
        assert abs(target.item() - 3.14159) < 0.1

    def test_handles_large_tensor(self):
        """Should handle larger tensors without issues."""
        source = torch.randn(100, 100, 10, dtype=torch.float32)
        target = torch.zeros(100, 100, 10, dtype=torch.bfloat16)

        # Should not raise
        copy_stochastic_(target, source)

        assert target.shape == source.shape

    def test_extreme_values(self):
        """Should handle extreme but valid float32 values."""
        source = torch.tensor([1e10, -1e10, 1e-10, -1e-10], dtype=torch.float32)
        target = torch.zeros(4, dtype=torch.bfloat16)

        copy_stochastic_(target, source)

        # Large values should be preserved approximately
        assert target[0].item() > 1e9
        assert target[1].item() < -1e9


class TestPatchAdafactorFused:
    """Tests for patch_adafactor_fused function."""

    def test_patches_step_and_step_param(self):
        """Should patch the optimizer's step and step_param methods."""
        from transformers import Adafactor

        # Create a minimal parameter to avoid issues
        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = Adafactor([param], lr=1e-3, scale_parameter=False, relative_step=False)

        original_step = optimizer.step

        patch_adafactor_fused(optimizer)

        # Methods should be patched
        assert hasattr(optimizer, "step_param")
        # The step method should have changed
        assert optimizer.step != original_step

    def test_patched_optimizer_can_step(self):
        """Patched optimizer should be able to perform a step without error."""
        from transformers import Adafactor

        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = Adafactor([param], lr=1e-3, scale_parameter=False, relative_step=False)

        patch_adafactor_fused(optimizer)

        # Create a dummy gradient
        param.grad = torch.randn_like(param)

        # Step should not raise
        optimizer.step()

    def test_patched_optimizer_with_bfloat16_uses_stochastic_rounding(self):
        """When param is bfloat16, patched step should use stochastic rounding."""
        from transformers import Adafactor

        # Create bfloat16 parameter
        param = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
        optimizer = Adafactor([param], lr=1e-3, scale_parameter=False, relative_step=False)

        patch_adafactor_fused(optimizer)

        original_data = param.data.clone()
        param.grad = torch.randn(4, 4, dtype=torch.bfloat16)

        # Step should complete without error
        optimizer.step()

        # Parameter should have changed
        assert not torch.equal(param.data, original_data)
        # Should still be bfloat16
        assert param.dtype == torch.bfloat16
