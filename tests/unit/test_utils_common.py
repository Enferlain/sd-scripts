import torch
import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from library.utils.common_utils import (
    str_to_dtype,
    resize_image,
    pil_resize,
    GradualLatent,
    validate_interpolation_fn,
    swap_weight_devices,
    EulerAncestralDiscreteSchedulerGL
)

class TestStrToDtype:
    def test_float32(self):
        assert str_to_dtype("float32") == torch.float32
        assert str_to_dtype("fp32") == torch.float32
        assert str_to_dtype("float") == torch.float32

    def test_float16(self):
        assert str_to_dtype("float16") == torch.float16
        assert str_to_dtype("fp16") == torch.float16

    def test_bfloat16(self):
        assert str_to_dtype("bfloat16") == torch.bfloat16
        assert str_to_dtype("bf16") == torch.bfloat16
    
    def test_float8(self):
        # Depending on torch version, these might be available or not
        # The function returns specific types if available or defaults to e4m3fn
        try:
            val = str_to_dtype("fp8")
            assert val == torch.float8_e4m3fn
        except AttributeError:
            pass # torch version might be old

    def test_none_default(self):
        assert str_to_dtype(None, default_dtype=torch.float16) == torch.float16
        assert str_to_dtype(None) is None

    def test_invalid(self):
        with pytest.raises(ValueError):
            str_to_dtype("invalid_type")

class TestResizeImage:
    @pytest.fixture
    def sample_image_cv2(self):
        # Create a 100x100 BGR image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Red square in top left
        img[0:50, 0:50] = [0, 0, 255] 
        return img

    def test_resize_area(self, sample_image_cv2):
        # 100x100 -> 50x50
        resized = resize_image(sample_image_cv2, 100, 100, 50, 50, resize_interpolation="area")
        assert resized.shape == (50, 50, 3)
        
    def test_resize_lanczos_pil(self, sample_image_cv2):
        # 100x100 -> 50x50
        resized = resize_image(sample_image_cv2, 100, 100, 50, 50, resize_interpolation="lanczos")
        assert resized.shape == (50, 50, 3)
        
    def test_resize_auto_interpolation(self, sample_image_cv2):
        # Downscale -> should use area
        resized = resize_image(sample_image_cv2, 100, 100, 50, 50, resize_interpolation=None)
        assert resized.shape == (50, 50, 3)

        # Upscale -> should use lanczos
        resized = resize_image(sample_image_cv2, 100, 100, 200, 200, resize_interpolation=None)
        assert resized.shape == (200, 200, 3)

    def test_resize_alpha_channel(self):
        # 4-channel image (BGRA)
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :, 3] = 255 # Opaque
        
        # Resize using PIL backend (e.g. lanczos checks for alpha)
        resized = resize_image(img, 100, 100, 50, 50, resize_interpolation="lanczos")
        assert resized.shape == (50, 50, 4)
        
        # Resize using CV2 backend (e.g. area)
        resized_cv = resize_image(img, 100, 100, 50, 50, resize_interpolation="area")
        assert resized_cv.shape == (50, 50, 4)

class TestGradualLatent:
    def test_initialization(self):
        gl = GradualLatent(0.5, 10, 2, 0.1)
        assert gl.ratio == 0.5
        assert gl.start_timesteps == 10
        assert gl.every_n_steps == 2
        
    def test_interpolate(self):
        gl = GradualLatent(0.5, 10, 2, 0.1)
        x = torch.randn(1, 4, 16, 16)
        resized = gl.interpolate(x, (32, 32), unsharp=False)
        assert resized.shape == (1, 4, 32, 32)
        
    def test_unsharp_mask_active(self):
        # Ensure ksize is set so unsharp mask actually runs
        gl = GradualLatent(0.5, 10, 2, 0.1, gaussian_blur_ksize=3)
        x = torch.rand(1, 4, 32, 32)
        
        # Run with and without unsharp
        res_true = gl.interpolate(x, (32, 32), unsharp=True)
        res_false = gl.interpolate(x, (32, 32), unsharp=False)
        
        # Should differ because of Gaussian blur unmasking
        assert not torch.allclose(res_true, res_false)
        assert res_true.shape == res_false.shape

def test_validate_interpolation_fn():
    assert validate_interpolation_fn("lanczos") is True
    assert validate_interpolation_fn("area") is True
    assert validate_interpolation_fn("box") is True
    assert validate_interpolation_fn("invalid") is False

def test_swap_weight_devices_mock():
    # Mock logic that typically requires CUDA
    # We patch 'library.utils.common_utils.torch' so that torch.cuda calls inside usage are mocked
    with patch("library.utils.common_utils.torch") as mock_torch:
        m1 = MagicMock()
        m2 = MagicMock()
        
        # Mock modules yield self
        m1.modules.return_value = [m1]
        m2.modules.return_value = [m2]
        
        # Mock weight attributes
        m1.weight = MagicMock()
        m2.weight = MagicMock()
        
        # Ensure they look like the same class
        m2.__class__ = m1.__class__
        
        # Run function
        swap_weight_devices(m1, m2)
        
        # Verify calls
        mock_torch.cuda.current_stream.assert_called()
        mock_torch.cuda.Stream.assert_called()

class TestEulerSchedulerGL:
    def test_step_with_gradual_latent(self):
        # We need to suppress the parent class __init__ if it does complex stuff
        with patch("library.utils.common_utils.EulerAncestralDiscreteScheduler.__init__", return_value=None):
            # Also patch sigmas property on the class using create=True since it might not exist yet
            with patch.object(EulerAncestralDiscreteSchedulerGL, 'sigmas', new_callable=PropertyMock, create=True) as mock_sigmas, \
                 patch.object(EulerAncestralDiscreteSchedulerGL, 'step_index', new_callable=PropertyMock, create=True) as mock_step_index, \
                 patch.object(EulerAncestralDiscreteSchedulerGL, 'config', new_callable=PropertyMock, create=True) as mock_config, \
                 patch.object(EulerAncestralDiscreteSchedulerGL, 'is_scale_input_called', new_callable=PropertyMock, create=True) as mock_scale_input:
                
                scheduler = EulerAncestralDiscreteSchedulerGL(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
                
                # FIX: Manually initialize the attribute that __init__ would have set
                scheduler._step_index = 0  # Initialize it to 0 so += 1 works

                # Setup necessary attributes for step()
                # Use the property mock to return tensor
                mock_sigmas.return_value = torch.tensor([1.0, 0.5, 0.0])
                mock_step_index.return_value = 0 # Simulate step 0
                
                # Setup config mock
                conf = MagicMock()
                conf.prediction_type = "epsilon"
                mock_config.return_value = conf
                
                mock_scale_input.return_value = True

                # scheduler.step_index = 0 # Cannot set property, handled by mock above
                # scheduler.config = MagicMock() # Cannot set property
                
                # Setup GradualLatent
                gl = MagicMock()
                gl.s_noise = 1.0
                gl.unsharp_target_x = False
                # interpolate returns a tensor of correct shape
                gl.interpolate.return_value = torch.zeros((1, 4, 32, 32))
                
                scheduler.set_gradual_latent_params((32, 32), gl)
                
                model_output = torch.randn(1, 4, 16, 16) # Original size
                sample = torch.randn(1, 4, 16, 16)       # Original size
                
                # We mock the internal random generator and torch in common_utils to avoid issues
                # Note: internal code uses 'diffusers.schedulers.scheduling_euler_ancestral_discrete.randn_tensor'
                with patch("library.utils.common_utils.diffusers.schedulers.scheduling_euler_ancestral_discrete.randn_tensor") as mock_randn:
                    mock_randn.return_value = torch.zeros((1, 4, 32, 32))
                    
                    # step returns a scheduler output object or tuple
                    # We expect it to call gl.interpolate because resized_size is set
                    # The timestamp must be a tensor or float, not int index if strict checks are on, 
                    # but typically explicitly passing tensor avoids ambiguous index vs time
                    
                    # NOTE: Diffusers error: "Passing integer indices... is not supported." 
                    # This happens when timestep is int and scheduler thinks it's an index.
                    # We pass a tensor to be safe/correct for modern diffusers.
                    t = torch.tensor(0.0, dtype=torch.float32)

                    output = scheduler.step(model_output, t, sample, return_dict=False)
                    
                    # Verify interpolate was called
                    assert gl.interpolate.called
                    # It calls it for sample and derivative (because unsharp_target_x=False default in mock setup above)
                    assert gl.interpolate.call_count >= 2
