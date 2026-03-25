from __future__ import annotations

import logging
import os

import matplotlib
import numpy as np
import torch

from library.config.dataclasses.loss import EDM2Config
from library.config.dataclasses.training import TrainingConfig

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

plt.ioff()

logger = logging.getLogger(__name__)


def plot_edm2_loss_weighting_check(edm2_config: EDM2Config, training_config: TrainingConfig, global_step: int) -> bool:
    """Return whether the EDM2 weighting plot should be emitted at this step."""
    return (
        edm2_config.enabled
        and edm2_config.visualization.enabled
        and (
            global_step
            % (
                int(edm2_config.visualization.every_n_steps)
                if edm2_config.visualization.every_n_steps
                else 20
            )
            == 0
            or global_step >= training_config.max_train_steps
        )
    )


def plot_edm2_loss_weighting(edm2_config: EDM2Config, output_name, step: int, model, num_timesteps: int = 1000, device="cpu"):
    """Plot EDM2 loss weighting across timesteps using the learned parameters."""
    with torch.inference_mode():
        model.train(False)
        timesteps = torch.arange(0, 1000, device=device, dtype=torch.long)
        learnedweights = model._forward(timesteps).cpu().numpy()
        lambdas = model.lambda_weights.cpu().numpy()
        learnedweights = lambdas / np.exp(learnedweights)
        model.train(True)

        plt.figure(figsize=(10, 6))
        plt.plot(timesteps.cpu().numpy(), learnedweights, label=f"Dynamic Loss Weight\nStep: {step}")
        plt.xlabel("Timesteps")
        plt.ylabel("Weight")
        plt.title("Dynamic Loss Weighting vs Timesteps")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)
        if edm2_config.visualization.y_limit is not None:
            plt.ylim(top=int(edm2_config.visualization.y_limit))
        plt.xlim(left=0, right=num_timesteps)
        plt.xticks(np.arange(0, num_timesteps + 1, 100))

        try:
            os.makedirs(edm2_config.visualization.output_dir, exist_ok=True)
            output_dir = os.path.join(edm2_config.visualization.output_dir, output_name)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"weighting_step_{str(step).zfill(7)}.png"))
        except Exception as exc:
            logger.warning(f"Failed to save weighting graph image. Due to: {exc}")

        plt.close()
