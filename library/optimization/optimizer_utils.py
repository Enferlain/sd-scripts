import ast
import importlib
import inspect
import logging
import types
import torch

from collections.abc import Callable
from torch.optim import Optimizer

from library.config.dataclasses.optimizer import OptimizerConfig, LearningRatesConfig
from library.config.dataclasses.peft import PeftConfig
from library.constants import int_pattern, float_pattern
from library.optimization.arguments import parse_key_value_args
from library.optimization.optimizer_factory import get_optimizer
from library.optimization.registry import (
    OPT_CAP_TRAIN_EVAL_TOGGLE,
    get_configured_optimizer_name,
    get_optimizer_registration,
)
from library.optimization.types import materialize_parameter_groups


logger = logging.getLogger(__name__)


# =============================================================================
# LR-based training control helpers
# =============================================================================


def should_train_text_encoder(learning_rates: LearningRatesConfig) -> bool:
    """
    Check if any text encoder should be trained based on learning rates.

    Returns True if:
    - text_encoders LR is None (will use base LR)
    - text_encoders LR is a positive number
    - text_encoders LR is a list with any positive values
    """
    te_lr = learning_rates.text_encoders
    if te_lr is None:
        return True  # Default: train TE with base LR
    if isinstance(te_lr, (int, float)):
        return te_lr > 0
    # List of LRs - train if any are positive
    return any(lr > 0 for lr in te_lr)


def should_train_denoiser(learning_rates: LearningRatesConfig) -> bool:
    """
    Check if the denoiser should be trained based on learning rates.

    Returns True if:
    - denoiser LR is None (will use base LR)
    - denoiser LR is a positive number
    """
    denoiser_lr = learning_rates.denoiser
    return denoiser_lr is None or denoiser_lr > 0


def get_text_encoders_train_flags(learning_rates: LearningRatesConfig, text_encoders: list[object]) -> list[bool]:
    """
    Resolve per-text-encoder train flags from the configured learning rates.

    Args:
        learning_rates: Learning-rate configuration.
        text_encoders: Text encoders in the current model family.

    Returns:
        A list of booleans indicating whether each text encoder should train.
    """
    num_text_encoders = len(text_encoders)
    te_lr = learning_rates.text_encoders
    if te_lr is None:
        return [should_train_text_encoder(learning_rates)] * num_text_encoders
    if isinstance(te_lr, int | float):
        return [te_lr > 0] * num_text_encoders

    te_flags = [lr_val > 0 for lr_val in te_lr]
    while len(te_flags) < num_text_encoders:
        te_flags.append(False)
    return te_flags[:num_text_encoders]


def prepare_optimizer(optimizer_config: OptimizerConfig, learning_rates: LearningRatesConfig, adapter_config: PeftConfig, adapter):
    """
    Prepares the optimizer for training, including handling adapter-specific logic and learning rate setup.

    Args:
        optimizer_config (OptimizerConfig): Configuration for the optimizer.
        learning_rates (LearningRatesConfig): Configuration for learning rates.
        adapter_config (PeftConfig): Configuration for PEFT/Adapters.
        adapter: The adapter module or object handling the specific adapter logic.

    Returns:
        tuple: A tuple containing the optimizer name, optimizer arguments string, optimizer instance,
               train function, eval function, and learning rate descriptions.
    """
    if isinstance(adapter_config.orthograd_targets, str):
        orthograd_targets = ast.literal_eval(adapter_config.orthograd_targets)
    else:
        orthograd_targets = [
            "lora_down.weight",
            "lora_up.weight",
            "lora_down1.weight",
            "lora_up1.weight",
            "lora_down2.weight",
            "lora_up2.weight",
            "a1.weight",
            "a2.weight",
            "b1.weight",
            "b2.weight",
            "c1.weight",
        ]

    optimizer_kwargs = parse_key_value_args(optimizer_config.optimizer_args)

    try:
        # Check optimizer defaults
        case_sensitive_optimizer_type = optimizer_config.optimizer_type  # not lower

        if "." not in case_sensitive_optimizer_type:  # from torch.optim
            optimizer_module = torch.optim
        else:  # from other library
            values = case_sensitive_optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            case_sensitive_optimizer_type = values[-1]

        # Need to handle base optimizer
        if case_sensitive_optimizer_type.lower() == "schedulefreewrapper" or optimizer_config.optimizer_type.lower().endswith(
            "snoo_asgd".lower()
        ):
            case_sensitive_full_base_optimizer_name = optimizer_kwargs.get("base_optimizer_type")
            if case_sensitive_full_base_optimizer_name is None:
                raise ValueError("base_optimizer_type is required in optimizer_args for ScheduleFreeWrapper/snoo_asgd optimizers")
            base_optimizer_values = case_sensitive_full_base_optimizer_name.split(".")  # Not None - line 116 raises if None
            base_optimizer_module = importlib.import_module(".".join(base_optimizer_values[:-1]))
            case_sensitive_base_optimizer_type = base_optimizer_values[-1]
            optimizer_class = getattr(base_optimizer_module, case_sensitive_base_optimizer_type)
        else:
            optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)

        sig = inspect.signature(optimizer_class.__init__)

        optimizer_init_sig_parameters = sig.parameters
    except Exception as e:
        logger.warning(f"Encountered an error while trying to determine default orthograd from optimizer init signature. {e}")
        optimizer_init_sig_parameters = {}

    apply_orthograd = any(
        optimizer_kwargs.get(key, getattr(optimizer_init_sig_parameters.get(key, types.SimpleNamespace()), "default", False)) is True
        for key in ["use_orthograd", "orthograd"]
    )

    # learning_rates is now passed explicitly as a parameter

    # Check if peft supports multiple text encoder learning rates
    support_multiple_lrs = hasattr(adapter, "prepare_optimizer_params_with_multiple_te_lrs")

    try:
        if support_multiple_lrs:
            # only flux atm via Kohya's - still uses old signature for now
            raw_te_lr = learning_rates.text_encoders
            if raw_te_lr is None or isinstance(raw_te_lr, (float, int)):
                text_encoder_lr = raw_te_lr
            else:
                text_encoder_lr = raw_te_lr  # Keep as list
            results = adapter.prepare_optimizer_params_with_multiple_te_lrs(
                text_encoder_lr=text_encoder_lr,
                unet_lr=learning_rates.denoiser,
                learning_rate=learning_rates.base,
                apply_orthograd=apply_orthograd,
                orthograd_targets=orthograd_targets,
            )
        else:
            # New signature: pass LearningRatesConfig directly
            results = adapter.prepare_optimizer_params(
                learning_rates=learning_rates, apply_orthograd=apply_orthograd, orthograd_targets=orthograd_targets
            )
        if type(results) is tuple:
            trainable_params, lr_descriptions = results
        else:
            trainable_params = results
            lr_descriptions = None
    except TypeError:
        # Fallback for adapters that don't yet support new signature (e.g., LyCORIS)
        raw_te_lr = learning_rates.text_encoders
        if raw_te_lr is None or isinstance(raw_te_lr, (float, int)):
            text_encoder_lr = raw_te_lr
        else:
            text_encoder_lr = raw_te_lr[0] if len(raw_te_lr) > 0 else None
        results = adapter.prepare_optimizer_params(
            text_encoder_lr=text_encoder_lr,
            unet_lr=learning_rates.denoiser,
            learning_rate=learning_rates.base,
            apply_orthograd=apply_orthograd,
            orthograd_targets=orthograd_targets,
        )
        if type(results) is tuple:
            trainable_params, lr_descriptions = results
        else:
            trainable_params = results
            lr_descriptions = None

    optimizer_name, optimizer_args, optimizer = get_optimizer(
        optimizer_config,
        learning_rates,
        optimizer_config.scheduler,
        materialize_parameter_groups(trainable_params),
        optimizer_kwargs,
    )
    # Cast to Optimizer - get_optimizer returns object but we know it's an Optimizer
    optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(optimizer, optimizer_config)  # type: ignore[arg-type]

    return optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn, lr_descriptions


def get_optimizer_train_eval_fn(optimizer: Optimizer, optimizer_config: OptimizerConfig) -> tuple[Callable, Callable]:
    """
    Returns the train and eval functions for the optimizer if it is schedule-free.

    Args:
        optimizer (Optimizer): The optimizer instance.
        optimizer_config (OptimizerConfig): Configuration for the optimizer.

    Returns:
        Tuple[Callable, Callable]: A tuple containing the train function and the eval function.
                                   Returns dummy no-op functions if not schedule-free.
    """
    if not is_schedulefree_optimizer(optimizer, optimizer_config) or getattr(optimizer_config, "fused_optimizer_groups", False):
        # return dummy func
        return lambda: None, lambda: None

    # get train and eval functions from optimizer (schedule-free optimizers have these)
    train_fn = getattr(optimizer, "train", lambda: None)
    eval_fn = getattr(optimizer, "eval", lambda: None)

    return train_fn, eval_fn


def is_schedulefree_optimizer(optimizer: Optimizer, optimizer_config: OptimizerConfig) -> bool:
    """
    Checks if the optimizer is a schedule-free optimizer.

    Args:
        optimizer (Optimizer): The optimizer instance.
        optimizer_config (OptimizerConfig): Configuration for the optimizer.

    Returns:
        bool: True if the optimizer is schedule-free, False otherwise.
    """
    optimizer_name = get_configured_optimizer_name(optimizer_config)
    registration = get_optimizer_registration(optimizer_name)
    if registration is not None:
        return registration.supports(OPT_CAP_TRAIN_EVAL_TOGGLE)

    optimizer_name = optimizer_name.lower()
    return optimizer_name.endswith("schedulefree".lower()) or optimizer_name.endswith("schedulefreewrapper".lower())


def is_wrapper_optimizer(optimizer_config: OptimizerConfig) -> bool:
    """
    Checks if the optimizer is a wrapper type optimizer.

    Args:
        optimizer_config (OptimizerConfig): Configuration for the optimizer.

    Returns:
        bool: True if the optimizer is a wrapper optimizer, False otherwise.
    """
    optimizer_name = get_configured_optimizer_name(optimizer_config)
    registration = get_optimizer_registration(optimizer_name)
    if registration is not None:
        return registration.kind == "wrapper"

    optimizer_name = optimizer_name.lower()
    return optimizer_name.endswith("schedulefreewrapper".lower()) or optimizer_name.endswith("snoo_asgd".lower())


def parse_string_to_type(s):
    """
    Parses a string into a specific type (int, float) if possible.

    Args:
        s: The string to parse.

    Returns:
        The parsed value as int, float, or the original string.
    """
    if s is not None:
        if isinstance(s, (float, int)):
            return s
        elif float_pattern.match(s):
            return float(s)
        elif int_pattern.match(s):
            return int(s)
        else:
            return s
    else:
        return None
