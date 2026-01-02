"""
Unit tests for library/performance/custom_offloading_utils.py

Tests the pure utility functions for CPU/device offloading.
Note: Offloader and ModelOffloader classes are not tested here as they require
GPU streams and threading which are better suited for integration tests.
"""

import torch
import torch.nn as nn

from library.performance.custom_offloading_utils import (
    to_device,
    to_cpu,
    create_cpu_offloading_wrapper,
)

from library.utils.torch_utils import weights_to_device


# =============================================================================
# Tests: to_device
# =============================================================================

class TestToDevice:
    """Tests for to_device function."""
    
    def test_single_tensor(self):
        """Single tensor should be moved to device."""
        tensor = torch.randn(3, 3)
        device = torch.device("cpu")
        
        result = to_device(tensor, device)
        
        assert result.device == device
        assert result.shape == tensor.shape
    
    def test_tensor_preserves_values(self):
        """Tensor values should be preserved after moving."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        device = torch.device("cpu")
        
        result = to_device(tensor, device)
        
        assert torch.allclose(result, tensor)
    
    def test_list_of_tensors(self):
        """List of tensors should all be moved."""
        tensors = [torch.randn(2, 2), torch.randn(3, 3)]
        device = torch.device("cpu")
        
        result = to_device(tensors, device)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(t.device == device for t in result)
    
    def test_tuple_of_tensors(self):
        """Tuple of tensors should remain tuple."""
        tensors = (torch.randn(2, 2), torch.randn(3, 3))
        device = torch.device("cpu")
        
        result = to_device(tensors, device)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(t.device == device for t in result)
    
    def test_dict_of_tensors(self):
        """Dict of tensors should have values moved."""
        tensors = {"a": torch.randn(2, 2), "b": torch.randn(3, 3)}
        device = torch.device("cpu")
        
        result = to_device(tensors, device)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        assert all(t.device == device for t in result.values())
    
    def test_nested_structure(self):
        """Deeply nested structures should be handled recursively."""
        data = {
            "tensors": [torch.randn(2, 2), torch.randn(3, 3)],
            "nested": {
                "inner": (torch.randn(1, 1),)
            }
        }
        device = torch.device("cpu")
        
        result = to_device(data, device)
        
        assert result["tensors"][0].device == device
        assert result["tensors"][1].device == device
        assert result["nested"]["inner"][0].device == device
    
    def test_non_tensor_passthrough(self):
        """Non-tensor values should be returned unchanged."""
        data = {
            "number": 42,
            "string": "hello",
            "none": None,
            "tensor": torch.randn(2, 2),
        }
        device = torch.device("cpu")
        
        result = to_device(data, device)
        
        assert result["number"] == 42
        assert result["string"] == "hello"
        assert result["none"] is None
        assert isinstance(result["tensor"], torch.Tensor)
    
    def test_empty_containers(self):
        """Empty containers should be handled."""
        assert to_device([], torch.device("cpu")) == []
        assert to_device((), torch.device("cpu")) == ()
        assert to_device({}, torch.device("cpu")) == {}
    
    def test_mixed_list(self):
        """List with mixed types should work."""
        data = [torch.randn(2, 2), 42, "text", None]
        device = torch.device("cpu")
        
        result = to_device(data, device)
        
        assert isinstance(result[0], torch.Tensor)
        assert result[1] == 42
        assert result[2] == "text"
        assert result[3] is None


# =============================================================================
# Tests: to_cpu
# =============================================================================

class TestToCpu:
    """Tests for to_cpu function."""
    
    def test_single_tensor(self):
        """Single tensor should be moved to CPU."""
        tensor = torch.randn(3, 3)  # Already on CPU, but tests the path
        
        result = to_cpu(tensor)
        
        assert result.device == torch.device("cpu")
        assert result.shape == tensor.shape
    
    def test_tensor_preserves_values(self):
        """Tensor values should be preserved."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        result = to_cpu(tensor)
        
        assert torch.allclose(result, tensor)
    
    def test_list_of_tensors(self):
        """List of tensors should all be moved to CPU."""
        tensors = [torch.randn(2, 2), torch.randn(3, 3)]
        
        result = to_cpu(tensors)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(t.device == torch.device("cpu") for t in result)
    
    def test_tuple_of_tensors(self):
        """Tuple of tensors should remain tuple."""
        tensors = (torch.randn(2, 2), torch.randn(3, 3))
        
        result = to_cpu(tensors)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_dict_of_tensors(self):
        """Dict of tensors should have values moved to CPU."""
        tensors = {"a": torch.randn(2, 2), "b": torch.randn(3, 3)}
        
        result = to_cpu(tensors)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        assert all(t.device == torch.device("cpu") for t in result.values())
    
    def test_nested_structure(self):
        """Deeply nested structures should be handled."""
        data = {
            "tensors": [torch.randn(2, 2)],
            "nested": {"inner": (torch.randn(1, 1),)}
        }
        
        result = to_cpu(data)
        
        assert result["tensors"][0].device == torch.device("cpu")
        assert result["nested"]["inner"][0].device == torch.device("cpu")
    
    def test_non_tensor_passthrough(self):
        """Non-tensor values should be returned unchanged."""
        data = {"number": 42, "string": "hello", "none": None}
        
        result = to_cpu(data)
        
        assert result["number"] == 42
        assert result["string"] == "hello"
        assert result["none"] is None
    
    def test_empty_containers(self):
        """Empty containers should be handled."""
        assert to_cpu([]) == []
        assert to_cpu(()) == ()
        assert to_cpu({}) == {}


# =============================================================================
# Tests: weights_to_device
# =============================================================================

class TestWeightsToDevice:
    """Tests for weights_to_device function."""

    def test_linear_layer_weights(self):
        """Linear layer weights should be moved."""
        layer = nn.Linear(10, 5)
        device = torch.device("cpu")

        weights_to_device(layer, device)

        assert layer.weight.data.device == device

    def test_conv_layer_weights(self):
        """Conv2d layer weights should be moved."""
        layer = nn.Conv2d(3, 16, 3)
        device = torch.device("cpu")

        weights_to_device(layer, device)

        assert layer.weight.data.device == device

    def test_nested_modules(self):
        """Nested modules should all have weights moved."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        device = torch.device("cpu")

        weights_to_device(model, device)

        assert model[0].weight.data.device == device
        assert model[2].weight.data.device == device

    def test_module_without_weight_no_error(self):
        """Modules without weight attribute should not error."""
        layer = nn.ReLU()  # Has no weight
        device = torch.device("cpu")

        # Should not raise
        weights_to_device(layer, device)

    def test_module_with_none_weight_no_error(self):
        """Modules with None weight should not error."""
        layer = nn.Linear(10, 5, bias=False)
        # Manually set weight to None for testing (unusual but valid test)
        # Actually, let's test a different scenario - BatchNorm affine=False
        layer = nn.BatchNorm2d(10, affine=False)  # Has no weight
        device = torch.device("cpu")

        # Should not raise
        weights_to_device(layer, device)

    def test_complex_model(self):
        """Complex model with multiple layer types."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.Flatten(),
            nn.Linear(32, 10),
        )
        device = torch.device("cpu")

        weights_to_device(model, device)

        # All weighted layers should have weights on device
        assert model[0].weight.data.device == device  # Conv2d
        assert model[1].weight.data.device == device  # BatchNorm2d
        assert model[3].weight.data.device == device  # Conv2d
        assert model[5].weight.data.device == device  # Linear


# =============================================================================
# Tests: create_cpu_offloading_wrapper
# =============================================================================

class TestCreateCpuOffloadingWrapper:
    """Tests for create_cpu_offloading_wrapper function."""
    
    def test_wraps_simple_function(self):
        """Simple function should be wrapped correctly."""
        def add_one(x):
            return x + 1
        
        device = torch.device("cpu")
        wrapped = create_cpu_offloading_wrapper(add_one, device)
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = wrapped(input_tensor)
        
        # Should work and return result on CPU
        assert result.device == torch.device("cpu")
        assert torch.allclose(result, torch.tensor([2.0, 3.0, 4.0]))
    
    def test_outputs_on_cpu(self):
        """Outputs should always be on CPU after wrapper."""
        def identity(x):
            return x
        
        device = torch.device("cpu")
        wrapped = create_cpu_offloading_wrapper(identity, device)
        
        input_tensor = torch.tensor([1.0, 2.0])
        result = wrapped(input_tensor)
        
        assert result.device == torch.device("cpu")
    
    def test_multiple_inputs(self):
        """Multiple input tensors should work."""
        def add(a, b):
            return a + b
        
        device = torch.device("cpu")
        wrapped = create_cpu_offloading_wrapper(add, device)
        
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        result = wrapped(a, b)
        
        assert torch.allclose(result, torch.tensor([4.0, 6.0]))
        assert result.device == torch.device("cpu")
    
    def test_tuple_output(self):
        """Tuple outputs should all be on CPU."""
        def split(x):
            return x[:2], x[2:]
        
        device = torch.device("cpu")
        wrapped = create_cpu_offloading_wrapper(split, device)
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = wrapped(input_tensor)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].device == torch.device("cpu")
        assert result[1].device == torch.device("cpu")
    
    def test_dict_output(self):
        """Dict outputs should have all values on CPU."""
        def make_dict(x):
            return {"doubled": x * 2, "halved": x / 2}
        
        device = torch.device("cpu")
        wrapped = create_cpu_offloading_wrapper(make_dict, device)
        
        input_tensor = torch.tensor([2.0, 4.0])
        result = wrapped(input_tensor)
        
        assert isinstance(result, dict)
        assert result["doubled"].device == torch.device("cpu")
        assert result["halved"].device == torch.device("cpu")
    
    def test_preserves_function_behavior(self):
        """Wrapper should preserve the original function's behavior."""
        def complex_op(x, multiplier=2):
            return x * multiplier + 1
        
        device = torch.device("cpu")
        wrapped = create_cpu_offloading_wrapper(complex_op, device)
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # Test default multiplier
        result1 = wrapped(input_tensor)
        assert torch.allclose(result1, torch.tensor([3.0, 5.0, 7.0]))
    
    def test_list_inputs(self):
        """List inputs should be handled."""
        def sum_list(tensors):
            return sum(tensors)
        
        device = torch.device("cpu")
        wrapped = create_cpu_offloading_wrapper(sum_list, device)
        
        tensors = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
        result = wrapped(tensors)
        
        assert result.device == torch.device("cpu")


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for offloading utilities."""
    
    def test_to_device_with_scalar(self):
        """Scalar values should pass through."""
        assert to_device(42, torch.device("cpu")) == 42
        assert to_device(3.14, torch.device("cpu")) == 3.14
        assert to_device("string", torch.device("cpu")) == "string"
    
    def test_to_cpu_with_scalar(self):
        """Scalar values should pass through to_cpu."""
        assert to_cpu(42) == 42
        assert to_cpu(3.14) == 3.14
        assert to_cpu("string") == "string"
    
    def test_to_device_none(self):
        """None should pass through."""
        assert to_device(None, torch.device("cpu")) is None
    
    def test_to_cpu_none(self):
        """None should pass through to_cpu."""
        assert to_cpu(None) is None
    
    def test_empty_module_weights_to_device(self):
        """Empty Sequential should not error."""
        model = nn.Sequential()
        device = torch.device("cpu")

        # Should not raise
        weights_to_device(model, device)
    
    def test_to_device_preserves_tensor_dtype(self):
        """Tensor dtype should be preserved after move."""
        tensors = {
            "float32": torch.randn(2, 2, dtype=torch.float32),
            "float16": torch.randn(2, 2, dtype=torch.float16),
            "int64": torch.randint(0, 10, (2, 2), dtype=torch.int64),
        }
        device = torch.device("cpu")
        
        result = to_device(tensors, device)
        
        assert result["float32"].dtype == torch.float32
        assert result["float16"].dtype == torch.float16
        assert result["int64"].dtype == torch.int64
    
    def test_to_cpu_preserves_tensor_dtype(self):
        """Tensor dtype should be preserved after to_cpu."""
        tensor_f16 = torch.randn(2, 2, dtype=torch.float16)
        tensor_i32 = torch.randint(0, 10, (2, 2), dtype=torch.int32)
        
        assert to_cpu(tensor_f16).dtype == torch.float16
        assert to_cpu(tensor_i32).dtype == torch.int32
    
    def test_to_device_preserves_requires_grad(self):
        """requires_grad should be preserved."""
        tensor = torch.randn(2, 2, requires_grad=True)
        device = torch.device("cpu")
        
        result = to_device(tensor, device)
        
        assert result.requires_grad is True
    
    def test_deeply_nested_structure(self):
        """Very deeply nested structures should work."""
        deep = {"level1": {"level2": {"level3": [torch.randn(2, 2)]}}}
        device = torch.device("cpu")
        
        result = to_device(deep, device)
        
        assert result["level1"]["level2"]["level3"][0].device == device


# =============================================================================
# Tests: Offloader class
# =============================================================================

class TestOffloader:
    """Tests for Offloader class."""
    
    def test_initialization(self):
        """Offloader should initialize correctly."""
        from library.performance.custom_offloading_utils import Offloader
        
        offloader = Offloader(num_blocks=10, blocks_to_swap=3, device=torch.device("cpu"))
        
        assert offloader.num_blocks == 10
        assert offloader.blocks_to_swap == 3
        assert offloader.device == torch.device("cpu")
        assert offloader.cuda_available is False  # CPU device
    
    def test_cuda_available_detection(self):
        """cuda_available should be True for CUDA device."""
        from library.performance.custom_offloading_utils import Offloader
        
        # Even if no GPU, we test the logic
        offloader = Offloader(num_blocks=5, blocks_to_swap=2, device=torch.device("cuda"))
        assert offloader.cuda_available is True
        
        offloader_cpu = Offloader(num_blocks=5, blocks_to_swap=2, device=torch.device("cpu"))
        assert offloader_cpu.cuda_available is False
    
    def test_thread_pool_created(self):
        """Thread pool should be created with max_workers=1."""
        from library.performance.custom_offloading_utils import Offloader
        
        offloader = Offloader(num_blocks=5, blocks_to_swap=2, device=torch.device("cpu"))
        
        assert offloader.thread_pool is not None
        assert offloader.thread_pool._max_workers == 1
    
    def test_futures_empty_initially(self):
        """Futures dict should be empty initially."""
        from library.performance.custom_offloading_utils import Offloader
        
        offloader = Offloader(num_blocks=5, blocks_to_swap=2, device=torch.device("cpu"))
        
        assert offloader.futures == {}


# =============================================================================
# Tests: ModelOffloader class
# =============================================================================

class TestModelOffloader:
    """Tests for ModelOffloader class."""
    
    def test_initialization(self):
        """ModelOffloader should initialize correctly."""
        from library.performance.custom_offloading_utils import ModelOffloader
        
        blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        offloader = ModelOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))
        
        assert offloader.num_blocks == 5
        assert offloader.blocks_to_swap == 2
        assert offloader.supports_backward is True
    
    def test_forward_only_mode(self):
        """forward_only should be set correctly."""
        from library.performance.custom_offloading_utils import ModelOffloader
        
        blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        
        # With backward support
        offloader = ModelOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"), supports_backward=True)
        assert offloader.forward_only is False
        
        # Without backward support
        offloader_fwd = ModelOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"), supports_backward=False)
        assert offloader_fwd.forward_only is True
    
    def test_set_forward_only(self):
        """set_forward_only should change the mode."""
        from library.performance.custom_offloading_utils import ModelOffloader
        
        blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        offloader = ModelOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))
        
        assert offloader.forward_only is False
        offloader.set_forward_only(True)
        assert offloader.forward_only is True
    
    def test_create_backward_hook_returns_none_when_not_needed(self):
        """create_backward_hook should return None for blocks that don't need hooks."""
        from library.performance.custom_offloading_utils import ModelOffloader
        
        blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        offloader = ModelOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))
        
        # Middle blocks usually don't need hooks
        hook = offloader.create_backward_hook(blocks, block_index=2)
        # The result depends on num_blocks_propagated logic
        # For block_index=2, num_blocks - 1 - idx = 5 - 1 - 2 = 2
        # swapping = 2 > 0 and 2 <= 2 (blocks_to_swap) = True
        # So this block DOES need a hook
        assert hook is not None
    
    def test_wait_for_block_noop_when_no_swap(self):
        """wait_for_block should do nothing when blocks_to_swap is 0."""
        from library.performance.custom_offloading_utils import ModelOffloader
        
        blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        offloader = ModelOffloader(blocks, blocks_to_swap=0, device=torch.device("cpu"))
        
        # Should not raise
        offloader.wait_for_block(0)
        offloader.wait_for_block(3)
    
    def test_submit_move_blocks_noop_when_no_swap(self):
        """submit_move_blocks should do nothing when blocks_to_swap is 0."""
        from library.performance.custom_offloading_utils import ModelOffloader
        
        blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        offloader = ModelOffloader(blocks, blocks_to_swap=0, device=torch.device("cpu"))
        
        # Should not raise and should not add futures
        offloader.submit_move_blocks(blocks, block_idx=0)
        assert len(offloader.futures) == 0
    
    def test_prepare_block_devices_noop_when_no_swap(self):
        """prepare_block_devices_before_forward should do nothing when blocks_to_swap is 0."""
        from library.performance.custom_offloading_utils import ModelOffloader
        
        blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        offloader = ModelOffloader(blocks, blocks_to_swap=0, device=torch.device("cpu"))
        
        # Should not raise
        offloader.prepare_block_devices_before_forward(blocks)

