# Based on fdl_loss_gpt_o4_mini_high.py
import torch
import torch.nn as nn


class FDLossLatent(nn.Module):
    """
    Frequency Distribution Loss (FDL) for Latents (e.g., 4-channel SDXL):
      L(U, V) = SWD(Amplitude(FFT2(F(U))), Amplitude(FFT2(F(V))))
              + lambda_phase * SWD(Phase(FFT2(F(U))), Phase(FFT2(F(V))))

    Where:
      - F is an optional feature extractor (default: Identity).
      - FFT2 is the 2D Fast Fourier Transform.
      - Amplitude and Phase are derived from the complex FFT output.
      - SWD is the Sliced Wasserstein Distance, approximated by:
        1. Projecting feature vectors (across channels) onto random directions.
        2. Comparing the 1D distributions of projected values across spatial locations.

    This module works on arbitrary BxCxHxW inputs.
    """

    def __init__(self,
                 lambda_phase: float = 0.1,
                 num_projections: int = 256,
                 feature_extractor: nn.Module = None):
        """
        Args:
          lambda_phase (float): Weight for the phase component SWD loss.
          num_projections (int): Number of random projections for SWD approximation.
          feature_extractor (nn.Module, optional): A module mapping input -> features.
                                                  If None, uses nn.Identity().
        """
        super().__init__()
        self.lambda_phase = lambda_phase
        self.num_projections = num_projections
        self.feature_extractor = feature_extractor if feature_extractor is not None else nn.Identity()

    def _compute_swd_per_sample(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        Compute Sliced Wasserstein Distance. Returns loss per batch item.

        Args:
            data1 (torch.Tensor): Features of shape (B, C, H, W).
            data2 (torch.Tensor): Features of shape (B, C, H, W).

        Returns:
            torch.Tensor: SWD loss for each sample in the batch (Shape B,).
        """
        B, C, H, W = data1.shape
        N = H * W

        data1_flat = data1.permute(0, 2, 3, 1).reshape(B, N, C)
        data2_flat = data2.permute(0, 2, 3, 1).reshape(B, N, C)

        thetas = torch.randn(self.num_projections, C, device=data1.device, dtype=data1.dtype)
        thetas = thetas / (thetas.norm(dim=1, keepdim=True) + 1e-8)  # Shape: (P, C)

        proj1 = data1_flat @ thetas.t()  # Shape: (B, N, P)
        proj2 = data2_flat @ thetas.t()  # Shape: (B, N, P)

        proj1_sorted, _ = torch.sort(proj1, dim=1)
        proj2_sorted, _ = torch.sort(proj2, dim=1)

        # L1 distance averaged over N and P, but kept per-sample (B)
        loss_per_sample = torch.mean(torch.abs(proj1_sorted - proj2_sorted), dim=(1, 2))  # Shape: (B,)

        return loss_per_sample  # Return per-sample loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        feat_x = self.feature_extractor(x)
        feat_y = self.feature_extractor(y)

        fft_x = torch.fft.fft2(feat_x, dim=(-2, -1), norm="ortho")
        fft_y = torch.fft.fft2(feat_y, dim=(-2, -1), norm="ortho")

        amp_x = torch.abs(fft_x)
        amp_y = torch.abs(fft_y)
        phase_x = torch.angle(fft_x)
        phase_y = torch.angle(fft_y)

        # Get per-sample SWD
        swd_amplitude = self._compute_swd_per_sample(amp_x, amp_y)  # Shape: (B,)
        swd_phase = self._compute_swd_per_sample(phase_x, phase_y)  # Shape: (B,)

        # Combine losses per sample
        loss = swd_amplitude + self.lambda_phase * swd_phase  # Shape: (B,)

        return loss  # Return per-sample loss tensor


class ChannelMixerExtractor(nn.Module):
    """
    Learns a linear combination of input channels using 1x1 Convolutions.
    Outputs the same number of channels as the input (4).
    """

    def __init__(self, in_channels=4, mid_channels=16, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
            # You could add another ReLU here if desired
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x (B, 4, H, W)
        Output: (B, 4, H, W)
        """
        return self.mixer(x)


class ShallowConvExtractor(nn.Module):
    """
    Applies a few convolutional layers to extract local spatial features.
    Maintains spatial dimensions.
    """

    def __init__(self, in_channels=4, num_features=16, num_layers=2, dtype=torch.float32, device="cpu"):
        super().__init__()
        layers = []
        current_channels = in_channels
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(current_channels, num_features, kernel_size=3, stride=1, padding=1, dtype=dtype,
                          device=device)
            )
            layers.append(nn.ReLU(inplace=True))
            current_channels = num_features

        # Optional: Add a final 1x1 conv to project back to 4 channels if desired
        # layers.append(nn.Conv2d(num_features, in_channels, kernel_size=1))

        self.extractor = nn.Sequential(*layers)
        self.output_channels = current_channels  # Or in_channels if projecting back

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x (B, 4, H, W)
        Output: (B, num_features, H, W) or (B, 4, H, W) if projected back
        """
        return self.extractor(x)


class MultiScaleConvExtractor(nn.Module):
    """
    Uses parallel convolutions with different kernel sizes to capture multi-scale features.
    Concatenates the outputs. Maintains spatial dimensions.
    """

    def __init__(self, in_channels=4, features_per_scale=8, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, features_per_scale, kernel_size=1, stride=1, padding=0, dtype=dtype,
                               device=device)
        self.conv3 = nn.Conv2d(in_channels, features_per_scale, kernel_size=3, stride=1, padding=1, dtype=dtype,
                               device=device)
        self.conv5 = nn.Conv2d(in_channels, features_per_scale, kernel_size=5, stride=1, padding=2, dtype=dtype,
                               device=device)
        self.relu = nn.ReLU(inplace=True)
        self.output_channels = features_per_scale * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x (B, 4, H, W)
        Output: (B, features_per_scale * 3, H, W)
        """
        f1 = self.relu(self.conv1(x))
        f3 = self.relu(self.conv3(x))
        f5 = self.relu(self.conv5(x))
        # Concatenate along the channel dimension
        out = torch.cat((f1, f3, f5), dim=1)
        return out
