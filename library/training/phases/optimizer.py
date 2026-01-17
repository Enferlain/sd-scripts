"""
Optimizer Phase - Optimizer and scheduler setup.

These functions handle optimizer creation, LR scheduler configuration,
and training step calculations.
"""

from __future__ import annotations

import logging
import math


logger = logging.getLogger(__name__)


def calculate_max_train_steps(
    max_train_epochs: int,
    num_batches_per_epoch: int,
    num_processes: int,
    gradient_accumulation_steps: int,
) -> int:
    """
    Calculate the total number of training steps from epoch count.

    Args:
        max_train_epochs: Number of epochs to train
        num_batches_per_epoch: Batches per epoch (based on dataset size / batch_size)
        num_processes: Number of distributed processes (accelerator.num_processes)
        gradient_accumulation_steps: Gradient accumulation steps

    Returns:
        Total number of optimization steps
    """
    return max_train_epochs * math.ceil(num_batches_per_epoch / num_processes / gradient_accumulation_steps)
