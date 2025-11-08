# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False.
                             If True, the loss is computed on the average spectrum
                             and then broadcasted to all samples in the batch.
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False,
                 batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        # (N, C, H, W) -> (N, P, P, C, patch_h, patch_w)
        x = x.reshape(x.shape[0], x.shape[1],
                      patch_factor, patch_h,
                      patch_factor, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(x.shape[0], patch_factor * patch_factor,
                      x.shape[3], x.shape[4], x.shape[5])  # (N, P*P, C, patch_h, patch_w)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')  # FFT on last two dims
        freq = torch.stack([freq.real, freq.imag], -1)  # (N, P*P, C, patch_h, patch_w, 2)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # Calculate the loss based on the provided frequencies.
        # Input shape: (N_eff, P*P, C, H', W', 2), where N_eff is 1 if ave_spectrum=True, else N
        # Output shape: (N_eff,)

        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()  # (N_eff, P*P, C, H', W')
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2  # (N_eff, P*P, C, H', W', 2)
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[
                ..., 1] + 1e-12) ** self.alpha  # Add eps for stability, (N_eff, P*P, C, H', W')

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                # Normalize across the entire batch's spectral magnitudes (or the single averaged spectrum)
                matrix_tmp = matrix_tmp / (matrix_tmp.max() + 1e-7)  # Add eps for stability
            else:
                # Normalize per sample (or the single avg sample), per patch, per channel
                max_vals = matrix_tmp.amax(dim=(-1, -2), keepdim=True)  # Shape: (N_eff, P*P, C, 1, 1)
                matrix_tmp = matrix_tmp / (max_vals + 1e-7)  # Add eps for stability

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()  # (N_eff, P*P, C, H', W')

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1.0 + 1e-6, (
        # Allow for slight float inaccuracies
                'The values of spectrum weight matrix should be in the range [0, 1], '
                'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2  # (N_eff, P*P, C, H', W', 2)
        freq_distance = tmp[..., 0] + tmp[..., 1]  # (N_eff, P*P, C, H', W')

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance  # (N_eff, P*P, C, H', W')

        # Average over patch, channel, freq_h, freq_w dims, keep N_eff dim
        # Dimensions to average over: 1 (P*P), 2 (C), 3 (H'), 4 (W')
        loss_per_sample_or_avg = torch.mean(loss, dim=(1, 2, 3, 4))  # (N_eff,)

        return loss_per_sample_or_avg

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix
                of shape (N or 1, P*P, C, H', W'). If ave_spectrum is True
                and matrix is provided, it should have N=1.
                Default: None (If set to None: calculated online, dynamic).

        Returns:
            torch.Tensor: The focal frequency loss for each sample in the batch,
                          shape (N,). If ave_spectrum is True, the loss value
                          computed on the average spectrum is broadcasted to all samples.
        """
        batch_size = pred.shape[0]
        pred_freq = self.tensor2freq(pred)  # (N, P*P, C, H', W', 2)
        target_freq = self.tensor2freq(target)  # (N, P*P, C, H', W', 2)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            # Average frequencies across the batch dimension
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)  # (1, P*P, C, H', W', 2)
            target_freq = torch.mean(target_freq, 0, keepdim=True)  # (1, P*P, C, H', W', 2)
            # If a matrix is provided, ensure it's compatible (N=1)
            if matrix is not None and matrix.shape[0] != 1:
                raise ValueError(
                    f"Provided matrix must have batch size 1 when ave_spectrum is True, but got shape {matrix.shape}")

        # calculate focal frequency loss
        # loss_val will be shape (N,) if ave_spectrum=False
        # loss_val will be shape (1,) if ave_spectrum=True
        loss_val = self.loss_formulation(pred_freq, target_freq, matrix)

        # If using average spectrum, broadcast the single loss value to all samples
        if self.ave_spectrum:
            loss_val = loss_val.expand(batch_size)  # Shape (1,) -> (N,)

        # Apply overall loss weight
        return loss_val * self.loss_weight  # (N,)
