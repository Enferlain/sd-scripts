from __future__ import annotations

import ast
import importlib
import math

import torch

from library.config.dataclasses.loss import EDM2Config
from library.config.dataclasses.training import TrainingConfig
from library.losses.edm2 import edm2_loss
from library.losses.edm2.edm2_modifier import EDM2LossModifier
from library.optimization.scheduler import get_dummy_scheduler


def prepare_edm2_loss_weighting(edm2_config: EDM2Config, training_config: TrainingConfig, noise_scheduler, accelerator):
    """Prepare the EDM2 model, optimizer, and scheduler for the active run."""
    if edm2_config.enabled:
        values = edm2_config.optimizer.type.split(".")
        optimizer_module = importlib.import_module(".".join(values[:-1]))
        case_sensitive_optimizer_type = values[-1]
        opti_args = ast.literal_eval(edm2_config.optimizer.args)
        opti_lr = float(edm2_config.optimizer.lr) if edm2_config.optimizer.lr else 2e-2

        edm2_model, edm2_optimizer = edm2_loss.create_weight_MLP(
            noise_scheduler,
            logvar_channels=int(edm2_config.num_channels) if edm2_config.num_channels else 128,
            optimizer=getattr(optimizer_module, case_sensitive_optimizer_type),
            lr=opti_lr,
            optimizer_args=opti_args,
            device=accelerator.device,
            dtype=torch.float32,
            use_importance_weights=edm2_config.importance.enabled,
            importance_weights_max_weight=float(edm2_config.importance.max_weight)
            if edm2_config.importance.max_weight is not None
            else 10.0,
            importance_weights_min_snr_gamma=float(edm2_config.importance.min_snr_gamma)
            if edm2_config.importance.min_snr_gamma is not None
            else 1.0,
        )
        if edm2_config.initial_weights:
            edm2_model.load_weights(edm2_config.initial_weights)

        if edm2_config.optimizer.use_scheduler:

            def inverse_sqrt(
                wrap_optimizer: torch.optim.Optimizer,
                warmup_steps: int = 0,
                constant_steps: int = 0,
                decay_scaling: float = 1.0,
            ):
                def lr_lambda(current_step: int):
                    if current_step <= warmup_steps:
                        return current_step / max(1, warmup_steps)
                    return 1 / math.sqrt(max(current_step / max(constant_steps + warmup_steps, 1), 1) ** decay_scaling)

                return torch.optim.lr_scheduler.LambdaLR(optimizer=wrap_optimizer, lr_lambda=lr_lambda)

            edm2_lr_scheduler = inverse_sqrt(
                edm2_optimizer,
                warmup_steps=int(
                    training_config.max_train_steps * float(edm2_config.optimizer.warmup_percent)
                    if edm2_config.optimizer.warmup_percent is not None
                    else 0.05
                ),
                constant_steps=int(
                    training_config.max_train_steps * float(edm2_config.optimizer.constant_percent)
                    if edm2_config.optimizer.constant_percent is not None
                    else 0.15
                ),
                decay_scaling=float(edm2_config.optimizer.decay_scaling)
                if edm2_config.optimizer.decay_scaling is not None
                else 1.0,
            )
        else:
            edm2_lr_scheduler = get_dummy_scheduler(edm2_optimizer)

        edm2_lr_scheduler = accelerator.prepare(edm2_lr_scheduler)
        edm2_model, edm2_optimizer = accelerator.prepare(edm2_model, edm2_optimizer)
    else:
        edm2_optimizer = None
        edm2_lr_scheduler = None
        edm2_model = None

    return edm2_model, edm2_optimizer, edm2_lr_scheduler


def create_edm2_modifier(edm2_config: EDM2Config, training_config: TrainingConfig, noise_scheduler, accelerator) -> EDM2LossModifier:
    """Create the EDM2 loss modifier for the active training run."""
    edm2_model, edm2_optimizer, edm2_lr_scheduler = prepare_edm2_loss_weighting(
        edm2_config,
        training_config,
        noise_scheduler,
        accelerator,
    )
    return EDM2LossModifier(
        config=edm2_config,
        training_config=training_config,
        model=edm2_model,
        optimizer=edm2_optimizer,
        lr_scheduler=edm2_lr_scheduler,
    )
