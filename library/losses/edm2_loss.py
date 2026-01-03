import torch
import torch.nn as nn
import numpy as np
import os
import logging

from diffusers import DDPMScheduler

from library.utils.hash_utils import precalculate_safetensors_hashes

logger = logging.getLogger(__name__)


def normalize(x: torch.Tensor, dim=None, eps=1e-4, dtype=torch.float32) -> torch.Tensor:
    """
    Normalizes a tensor along specified dimensions.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int or list, optional): Dimension(s) to normalize. Defaults to None (all dimensions except batch).
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-4.
        dtype (torch.dtype, optional): Data type for computation. Defaults to torch.float32.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=dtype)  # type: torch.Tensor
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class FourierFeatureExtractor(torch.nn.Module):
    """
    Extracts Fourier features from input.
    """

    def __init__(self, num_channels, bandwidth=1, dtype=torch.float32):
        """
        Initializes the FourierFeatureExtractor.

        Args:
            num_channels (int): Number of output channels (frequencies).
            bandwidth (int, optional): Bandwidth scaling factor. Defaults to 1.
            dtype (torch.dtype, optional): Data type for weights. Defaults to torch.float32.
        """
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes Fourier features.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with Fourier features.
        """
        y = x.to(self.dtype)
        y = y.ger(self.freqs.to(self.dtype))
        y = y + self.phases.to(self.dtype)  # type: torch.Tensor
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


class NormalizedLinearLayer(torch.nn.Module):
    """
    Linear layer with weight normalization.
    """

    def __init__(self, in_channels, out_channels, kernel=(), dtype=torch.float32):
        """
        Initializes the NormalizedLinearLayer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel (tuple, optional): Kernel size (not used in implementation, but kept for signature). Defaults to ().
            dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
        """
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))
        self.dtype = dtype

    def forward(self, x: torch.Tensor, gain=1) -> torch.Tensor:
        """
        Performs the forward pass with normalized weights.

        Args:
            x (torch.Tensor): Input tensor.
            gain (int, optional): Gain factor. Defaults to 1.

        Returns:
            torch.Tensor: Output tensor.
        """
        w = self.weight.to(self.dtype)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w, dtype=self.dtype))  # forced weight normalization
        w = normalize(w, dtype=self.dtype)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # type: torch.Tensor # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


class AdaptiveLossWeightMLP(nn.Module):
    """
    MLP for adaptive loss weighting based on EDM2.
    """

    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        logvar_channels: int = 128,
        lambda_weights: torch.Tensor | None = None,
        device="cuda",
        dtype=torch.float32,
        use_importance_weights: bool = True,
        importance_weights_max_weight: float = 10.0,
        importance_weights_min_snr_gamma: float = 1.0,
        importance_weights: torch.Tensor | None = None,
    ):
        """
        Initializes the AdaptiveLossWeightMLP.

        Args:
            noise_scheduler (DDPMScheduler): The noise scheduler.
            logvar_channels (int, optional): Channels for log variance features. Defaults to 128.
            lambda_weights (torch.Tensor, optional): Precomputed lambda weights. Defaults to None.
            device (str, optional): Device to run on. Defaults to 'cuda'.
            dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
            use_importance_weights (bool, optional): Whether to use importance weighting. Defaults to True.
            importance_weights_max_weight (float, optional): Maximum weight for importance weighting. Defaults to 10.0.
            importance_weights_min_snr_gamma (float, optional): Min-SNR gamma for importance weighting. Defaults to 1.0.
            importance_weights (torch.Tensor, optional): Explicit importance weights. Defaults to None.
        """
        super().__init__()
        self.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        # self.a_bar_mean = noise_scheduler.alphas_cumprod.mean()
        # self.a_bar_std = noise_scheduler.alphas_cumprod.std()
        self.a_bar_mean = self.alphas_cumprod.mean()
        self.a_bar_std = self.alphas_cumprod.std()
        self.logvar_fourier = FourierFeatureExtractor(logvar_channels, dtype=dtype)
        self.logvar_linear = NormalizedLinearLayer(
            logvar_channels, 1, kernel=[], dtype=dtype
        )  # kernel = []? (not in code given, added matching edm2)
        self.lambda_weights = (
            lambda_weights.to(device=device, dtype=dtype) if lambda_weights is not None else torch.ones(1000, device=device)
        )
        self.noise_scheduler = noise_scheduler
        self.dtype = dtype

        self.use_importance_weights = (use_importance_weights,)
        self.importance_weights = (
            importance_weights.to(device=device, dtype=dtype)
            if importance_weights is not None
            else torch.ones(1000, device=device, dtype=dtype)
        )

        if self.use_importance_weights:
            # min snr importance weights
            all_timesteps = torch.arange(noise_scheduler.config.num_train_timesteps).to(device=device)  # type: ignore[union-attr]
            snr = torch.stack([noise_scheduler.all_snr[t] for t in all_timesteps])

            min_snr_gamma = (importance_weights_max_weight * (1 + 1 / importance_weights_min_snr_gamma)) * torch.minimum(
                snr, torch.full_like(snr, importance_weights_min_snr_gamma)
            )  # multiply the torch.minimum by the max weight you want * 2 (i.e multiply by 40 and it'll cap off at 20 loss)
            min_snr_gamma = torch.div(min_snr_gamma, snr + 1).to(dtype=dtype, device=device)
            self.importance_weights = torch.where(
                self.importance_weights > min_snr_gamma,
                self.importance_weights,
                min_snr_gamma,
            )

    def _forward(self, timesteps: torch.Tensor):
        """
        Internal forward pass to compute adaptive weights from timesteps.
        """
        # a_bar = self.noise_scheduler.alphas_cumprod[timesteps]
        a_bar = self.alphas_cumprod[timesteps]
        c_noise = a_bar.sub(self.a_bar_mean).div_(self.a_bar_std)
        return self.logvar_linear(self.logvar_fourier(c_noise)).squeeze()

    def forward(self, loss: torch.Tensor, timesteps):
        """
        Applies adaptive weighting to the loss.

        Args:
            loss (torch.Tensor): Original loss.
            timesteps: Timesteps associated with the loss.

        Returns:
            tuple: (Weighted loss, Scaled loss)
        """
        timesteps = timesteps.long()
        adaptive_loss_weights = self._forward(timesteps)
        loss_scaled = loss * (self.lambda_weights[timesteps] / torch.exp(adaptive_loss_weights))  # type: torch.Tensor
        loss = loss_scaled + (self.importance_weights[timesteps] * adaptive_loss_weights)  # type: torch.Tensor

        return loss, loss_scaled

    def get_trainable_params(self):
        """
        Returns parameters to be optimized.
        """
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        """
        Saves the model weights to a file.

        Args:
            file (str): Path to save the file.
            dtype: Data type to save as.
            metadata (dict): Metadata to save with the weights (for safetensors).
        """
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        """
        Loads weights from a file.

        Args:
            file (str): Path to the weight file.

        Returns:
            The result of load_state_dict.
        """
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info


def create_weight_MLP(
    noise_scheduler: DDPMScheduler,
    logvar_channels: int = 128,
    lambda_weights: torch.Tensor | None = None,
    optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW,
    lr: float = 2e-2,
    optimizer_args: dict | None = None,
    dtype=torch.float32,
    device="cuda",
    use_importance_weights: bool = True,
    importance_weights_max_weight: float = 10.0,
    importance_weights_min_snr_gamma: float = 1.0,
):
    """
    Creates an instance of AdaptiveLossWeightMLP and its optimizer.

    Args:
        noise_scheduler (DDPMScheduler): The noise scheduler.
        logvar_channels (int, optional): Channels for log variance. Defaults to 128.
        lambda_weights (torch.tensor, optional): Lambda weights. Defaults to None.
        optimizer (torch.optim.Optimizer, optional): Optimizer class. Defaults to AdamW.
        lr (float, optional): Learning rate. Defaults to 2e-2.
        optimizer_args (dict, optional): Arguments for the optimizer. Defaults to {'weight_decay': 0, 'betas': (0.9, 0.99)}.
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
        device (str, optional): Device. Defaults to 'cuda'.
        use_importance_weights (bool, optional): Enable importance weighting. Defaults to True.
        importance_weights_max_weight (float, optional): Max importance weight. Defaults to 10.0.
        importance_weights_min_snr_gamma (float, optional): Min-SNR gamma. Defaults to 1.0.

    Returns:
        tuple: (AdaptiveLossWeightMLP instance, Optimizer instance)
    """
    logger.info("creating weight MLP")
    if optimizer_args is None:
        optimizer_args = {"weight_decay": 0, "betas": (0.9, 0.99)}
    lossweightMLP = AdaptiveLossWeightMLP(
        noise_scheduler,
        logvar_channels,
        lambda_weights,
        device,
        dtype=dtype,
        importance_weights_max_weight=importance_weights_max_weight,
        importance_weights_min_snr_gamma=importance_weights_min_snr_gamma,
        use_importance_weights=use_importance_weights,
    )
    MLP_optim = optimizer(lossweightMLP.parameters(), lr=lr, **optimizer_args)
    return lossweightMLP, MLP_optim
