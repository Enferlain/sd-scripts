import importlib
import ast
import logging
import os
import numpy as np
import torch
import math

import matplotlib

matplotlib.use("Agg")  # Set the backend to 'Agg', non-interactive backend

import matplotlib.pyplot as plt

plt.ioff()  # Explicitly turn off interactive mode

from library.utils.common_utils import setup_logging  # noqa: E402
from library.optimizers.scheduler import get_dummy_scheduler  # noqa: E402
from library.losses import edm2_loss  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def prepare_edm2_loss_weighting(loss_config, training_config, noise_scheduler, accelerator):
    """
    Prepares the EDM2 loss weighting model, optimizer, and scheduler based on the configuration.

    Args:
        loss_config: Configuration object containing loss settings.
        training_config: Configuration object containing training settings.
        noise_scheduler: The noise scheduler used in training.
        accelerator: The accelerator object for distributed training.

    Returns:
        tuple: A tuple containing (edm2_model, edm2_optimizer, edm2_lr_scheduler).
               If EDM2 loss weighting is not enabled, returns (None, None, None).
    """
    if loss_config.edm2_loss_weighting:
        values = loss_config.edm2_loss_weighting_optimizer.split(".")
        optimizer_module = importlib.import_module(".".join(values[:-1]))
        case_sensitive_optimizer_type = values[-1]
        opti_args = ast.literal_eval(loss_config.edm2_loss_weighting_optimizer_args)
        opti_lr = float(loss_config.edm2_loss_weighting_optimizer_lr) if loss_config.edm2_loss_weighting_optimizer_lr else 2e-2

        edm2_model, edm2_optimizer = edm2_loss.create_weight_MLP(
            noise_scheduler,
            logvar_channels=int(loss_config.edm2_loss_weighting_num_channels) if loss_config.edm2_loss_weighting_num_channels else 128,
            optimizer=getattr(optimizer_module, case_sensitive_optimizer_type),
            lr=opti_lr,
            optimizer_args=opti_args,
            device=accelerator.device,
            dtype=torch.float32,
            use_importance_weights=loss_config.edm2_loss_weighting_importance_weighting,
            importance_weights_max_weight=float(loss_config.edm2_loss_weighting_importance_weighting_max)
            if loss_config.edm2_loss_weighting_importance_weighting_max is not None
            else 10.0,
            importance_weights_min_snr_gamma=float(loss_config.edm2_loss_weighting_importance_min_snr_gamma)
            if loss_config.edm2_loss_weighting_importance_min_snr_gamma is not None
            else 1.0,
        )
        if loss_config.edm2_loss_weighting_initial_weights:
            edm2_model.load_weights(loss_config.edm2_loss_weighting_initial_weights)

        if loss_config.edm2_loss_weighting_lr_scheduler:

            def InverseSqrt(
                wrap_optimizer: torch.optim.Optimizer,
                warmup_steps: int = 0,
                constant_steps: int = 0,
                decay_scaling: float = 1.0,
            ):
                def lr_lambda(current_step: int):
                    if current_step <= warmup_steps:
                        return current_step / max(1, warmup_steps)
                    else:
                        return 1 / math.sqrt(max(current_step / max(constant_steps + warmup_steps, 1), 1) ** decay_scaling)

                return torch.optim.lr_scheduler.LambdaLR(optimizer=wrap_optimizer, lr_lambda=lr_lambda)

            edm2_lr_scheduler = InverseSqrt(
                edm2_optimizer,
                warmup_steps=int(
                    training_config.max_train_steps * float(loss_config.edm2_loss_weighting_lr_scheduler_warmup_percent)
                    if loss_config.edm2_loss_weighting_lr_scheduler_warmup_percent is not None
                    else 0.05
                ),
                constant_steps=int(
                    training_config.max_train_steps * float(loss_config.edm2_loss_weighting_lr_scheduler_constant_percent)
                    if loss_config.edm2_loss_weighting_lr_scheduler_constant_percent is not None
                    else 0.15
                ),
                decay_scaling=float(loss_config.edm2_loss_weighting_lr_scheduler_decay_scaling)
                if loss_config.edm2_loss_weighting_lr_scheduler_decay_scaling is not None
                else 1.0,
            )  # FIXME: CONFIG USAGE
        else:
            edm2_lr_scheduler = get_dummy_scheduler(edm2_optimizer)

        edm2_lr_scheduler = accelerator.prepare(edm2_lr_scheduler)

        edm2_model, edm2_optimizer = accelerator.prepare(edm2_model, edm2_optimizer)
    else:
        edm2_optimizer = None
        edm2_lr_scheduler = None
        edm2_model = None

    return edm2_model, edm2_optimizer, edm2_lr_scheduler


def handle_conflicting_configuration(loss_config):
    """
    Checks for and handles conflicting configurations related to EDM2 loss weighting.

    Specifically, it resolves conflicts between EDM2 importance weighting,
    debiased estimation loss, and Min-SNR gamma.

    Args:
        loss_config: Configuration object containing loss settings.
    """
    # Check for the critical conflicting settings
    if (
        loss_config.edm2_loss_weighting
        and loss_config.edm2_loss_weighting_importance_weighting
        and not loss_config.edm2_loss_weighting_importance_weighting_safety_override
    ):
        # --- Debiased Estimation Check ---
        if loss_config.debiased_estimation_loss:
            loss_config.debiased_estimation_loss = False
            logger.warning(
                "Debiased estimation loss AND EDM2 loss weighting with importance weighting are enabled. "
                "It is not advised to use both, as there is a possibility of loss curving to 0 as SNR approaches 0, "
                "as such, **Debiased estimation loss has been DISABLED**. "
                "You may override this behavior by setting edm2_loss_weighting_importance_weighting_safety_override=True."
            )

        # --- Min SNR Gamma Check ---
        if loss_config.min_snr_gamma:
            logger.warning(
                "Min snr gamma AND EDM2 loss weighting with importance weighting are enabled. "
                "It is not advised to use both, as there is a possibility of loss curving to 0 as SNR approaches 0, "
                "as such, **min snr gamma has been DISABLED**. "
                "You may override this behavior by setting edm2_loss_weighting_importance_weighting_safety_override=True."
            )
            loss_config.min_snr_gamma = None


def plot_edm2_loss_weighting_check(loss_config, training_config, global_step):
    """
    Determines whether to plot the EDM2 loss weighting graph at the current step.

    Args:
        loss_config: Configuration object containing loss settings.
        training_config: Configuration object containing training settings.
        global_step (int): The current global training step.

    Returns:
        bool: True if the graph should be plotted, False otherwise.
    """
    return (
        loss_config.edm2_loss_weighting
        and loss_config.edm2_loss_weighting_generate_graph
        and (
            global_step
            % (
                int(loss_config.edm2_loss_weighting_generate_graph_every_x_steps)
                if loss_config.edm2_loss_weighting_generate_graph_every_x_steps
                else 20
            )
            == 0
            or global_step >= training_config.max_train_steps
        )
    )


def plot_edm2_loss_weighting(loss_config, output_name, step: int, model, num_timesteps: int = 1000, device="cpu"):
    """
    Plot the edm2 loss weighting across timesteps using the learned parameters.

    Args:
        loss_config: Configuration object containing loss settings.
        output_name (str): Name of the output directory/file prefix.
        step (int): Current training step.
        model: The edm2 model instance (after training).
        num_timesteps (int): Total number of timesteps to plot.
        device: Device to run computations on.
    """
    with torch.inference_mode():
        model.train(False)
        timesteps = torch.arange(0, 1000, device=device, dtype=torch.long)
        learnedweights = model._forward(timesteps).cpu().numpy()
        lambdas = model.lambda_weights.cpu().numpy()
        learnedweights = lambdas / np.exp(learnedweights)
        model.train(True)

        # Plot the dynamic loss weights over time
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps.cpu().numpy(), learnedweights, label=f"Dynamic Loss Weight\nStep: {step}")
        plt.xlabel("Timesteps")
        plt.ylabel("Weight")
        plt.title("Dynamic Loss Weighting vs Timesteps")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)
        if loss_config.edm2_loss_weighting_generate_graph_y_limit is not None:
            plt.ylim(top=int(loss_config.edm2_loss_weighting_generate_graph_y_limit))
        plt.xlim(left=0, right=num_timesteps)
        plt.xticks(np.arange(0, num_timesteps + 1, 100))
        # plt.show()

        try:
            os.makedirs(loss_config.edm2_loss_weighting_generate_graph_output_dir, exist_ok=True)
            output_dir = os.path.join(loss_config.edm2_loss_weighting_generate_graph_output_dir, output_name)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"weighting_step_{str(step).zfill(7)}.png"))
        except Exception as e:
            logger.warning(f"Failed to save weighting graph image. Due to: {e}")

        plt.close()
