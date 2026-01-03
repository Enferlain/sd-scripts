# These classes were used only for standalone inference scripts, not during training sampling
import torch

from diffusers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteSchedulerOutput,
    randn_tensor,
)
from torchvision.transforms import functional as TF


class GradualLatent:
    def __init__(
        self,
        ratio: float,
        start_timesteps: int,
        every_n_steps: int,
        ratio_step: float,
        s_noise: float = 1.0,
        gaussian_blur_ksize: list[int] | None = None,
        gaussian_blur_sigma: list[float] | None = None,
        gaussian_blur_strength: float = 0.5,
        unsharp_target_x: bool = True,
    ) -> None:
        """
        Initializes the GradualLatent object.

        Args:
            ratio (float): The ratio of the resolution change.
            start_timesteps (int): The timesteps to start the gradual latent process.
            every_n_steps (int): The frequency of steps to apply the gradual latent process.
            ratio_step (float): The step size for the ratio change.
            s_noise (float, optional): The noise scale factor. Defaults to 1.0.
            gaussian_blur_ksize (list[int], optional): The kernel size for Gaussian blur. Defaults to None.
            gaussian_blur_sigma (list[float], optional): The sigma value for Gaussian blur. Defaults to None.
            gaussian_blur_strength (float, optional): The strength of the Gaussian blur. Defaults to 0.5.
            unsharp_target_x (bool, optional): Whether to apply unsharp mask to the target x. Defaults to True.
        """
        self.ratio = ratio
        self.start_timesteps = start_timesteps
        self.every_n_steps = every_n_steps
        self.ratio_step = ratio_step
        self.s_noise = s_noise
        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.gaussian_blur_strength = gaussian_blur_strength
        self.unsharp_target_x = unsharp_target_x

    def __str__(self) -> str:
        """
        Returns a string representation of the GradualLatent object.
        """
        return (
            f"GradualLatent(ratio={self.ratio}, start_timesteps={self.start_timesteps}, "
            + f"every_n_steps={self.every_n_steps}, ratio_step={self.ratio_step}, s_noise={self.s_noise}, "
            + f"gaussian_blur_ksize={self.gaussian_blur_ksize}, gaussian_blur_sigma={self.gaussian_blur_sigma}, gaussian_blur_strength={self.gaussian_blur_strength}, "
            + f"unsharp_target_x={self.unsharp_target_x})"
        )

    def apply_unsharp_mask(self, x: torch.Tensor):
        """
        Applies an unsharp mask to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be sharpened.

        Returns:
            torch.Tensor: The sharpened tensor.
        """
        if self.gaussian_blur_ksize is None:
            return x
        blurred = TF.gaussian_blur(x, self.gaussian_blur_ksize, self.gaussian_blur_sigma)
        # mask = torch.sigmoid((x - blurred) * self.gaussian_blur_strength)
        mask = (x - blurred) * self.gaussian_blur_strength
        sharpened = x + mask
        return sharpened

    def interpolate(self, x: torch.Tensor, resized_size, unsharp=True):
        """
        Interpolates the input tensor to a new size, optionally applying an unsharp mask.

        Args:
            x (torch.Tensor): The input tensor to be interpolated.
            resized_size (tuple): The target size for interpolation.
            unsharp (bool, optional): Whether to apply an unsharp mask after interpolation. Defaults to True.

        Returns:
            torch.Tensor: The interpolated (and optionally sharpened) tensor.
        """
        org_dtype = x.dtype
        if org_dtype == torch.bfloat16:
            x = x.float()

        x = torch.nn.functional.interpolate(x, size=resized_size, mode="bicubic", align_corners=False).to(dtype=org_dtype)

        # apply unsharp mask / アンシャープマスクを適用する
        if unsharp and self.gaussian_blur_ksize:
            x = self.apply_unsharp_mask(x)

        return x


class EulerAncestralDiscreteSchedulerGL(EulerAncestralDiscreteScheduler):
    """
    Euler Ancestral Discrete Scheduler with Gradual Latent support.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the EulerAncestralDiscreteSchedulerGL.
        """
        super().__init__(*args, **kwargs)
        self.resized_size = None
        self.gradual_latent = None

    def set_gradual_latent_params(self, size, gradual_latent: GradualLatent):
        """
        Sets the parameters for the gradual latent process.

        Args:
            size (tuple): The target size for the latent.
            gradual_latent (GradualLatent): The GradualLatent configuration object.
        """
        self.resized_size = size
        self.gradual_latent = gradual_latent

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> EulerAncestralDiscreteSchedulerOutput | tuple:
        """
        Predict the sample from the previous timesteps by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timesteps in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timesteps."
                ),
            )

        if not self.is_scale_input_called:
            # logger.warning(
            print(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise TODO: Unresolved attribute reference 'prediction_type' for class 'dict'
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`")

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        device = model_output.device
        if self.resized_size is None:
            prev_sample = sample + derivative * dt

            noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
            s_noise = 1.0
        else:
            print("resized_size", self.resized_size, "model_output.shape", model_output.shape, "sample.shape", sample.shape)
            s_noise = self.gradual_latent.s_noise

            if self.gradual_latent.unsharp_target_x:
                prev_sample = sample + derivative * dt
                prev_sample = self.gradual_latent.interpolate(prev_sample, self.resized_size)
            else:
                sample = self.gradual_latent.interpolate(sample, self.resized_size)
                derivative = self.gradual_latent.interpolate(derivative, self.resized_size, unsharp=False)
                prev_sample = sample + derivative * dt

            noise = randn_tensor(
                (model_output.shape[0], model_output.shape[1], self.resized_size[0], self.resized_size[1]),
                dtype=model_output.dtype,
                device=device,
                generator=generator,
            )

        prev_sample = prev_sample + noise * sigma_up * s_noise

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
