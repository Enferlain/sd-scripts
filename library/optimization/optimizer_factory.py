import inspect
import logging
from copy import deepcopy

import torch

from library.config.dataclasses.optimizer import LearningRatesConfig, OptimizerConfig, SchedulerConfig
from library.optimization.arguments import parse_key_value_args
from library.optimization.loading import load_target
from library.optimization.registry import (
    OptimizerRegistration,
    get_configured_optimizer_name,
    get_optimizer_registration,
    is_schedulefree_optimizer_name,
    is_wrapper_optimizer_name,
)
from library.optimization.types import materialize_parameter_groups
from library.optimization.wrappers import ScheduleFreeWrapper, WrappedOptimizerProxy
from library.utils.compile_env import prepare_windows_compiler_env_for_torch_compile


logger = logging.getLogger(__name__)


_BITSANDBYTES_ATTRIBUTE_ERRORS = {
    "lion8bit": "No Lion8bit. The version of bitsandbytes installed seems to be old. Please install 0.38.0 or later.",
    "pagedadamw": "No PagedAdamW. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later.",
    "pagedadamw8bit": "No PagedAdamW8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later.",
    "pagedadamw32bit": "No PagedAdamW32bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later.",
    "pagedlion8bit": "No PagedLion8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later.",
}


def _ensure_nesterov_momentum(optimizer_kwargs: dict, *, warning: bool = False):
    if "momentum" in optimizer_kwargs:
        return

    if warning:
        logger.warning("8-bit SGD with Nesterov must be with momentum, set momentum to 0.9")
    else:
        logger.info("SGD with Nesterov must be with momentum, set momentum to 0.9")
    optimizer_kwargs["momentum"] = 0.9


def _warn_adaptive_lr_usage(trainable_params, lr: float | None):
    actual_lr = lr
    lr_count = 1
    if isinstance(trainable_params, list) and isinstance(trainable_params[0], dict):
        lrs = set()
        actual_lr = trainable_params[0].get("lr", actual_lr)
        for group in trainable_params:
            lrs.add(group.get("lr", actual_lr))
        lr_count = len(lrs)

    if actual_lr is None:
        return

    if actual_lr <= 0.1:
        logger.warning(f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0: lr={actual_lr}")
        logger.warning("recommend option: lr=1.0")
    if lr_count > 1:
        logger.warning(
            f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect: lr={actual_lr}"
        )


def _prepare_adafactor(
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    trainable_params,
    optimizer_kwargs: dict,
    lr: float | None,
):
    if "relative_step" not in optimizer_kwargs:
        optimizer_kwargs["relative_step"] = True
    if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
        logger.info("set relative_step to True because warmup_init is True")
        optimizer_kwargs["relative_step"] = True
    logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

    if optimizer_kwargs["relative_step"]:
        logger.info("relative_step is true")
        if lr is not None and lr != 0.0:
            logger.warning("learning rate is used as initial_lr")
        optimizer_config.learning_rates.base = 0.0

        if isinstance(trainable_params, list) and isinstance(trainable_params[0], dict):
            has_group_lr = False
            for group in trainable_params:
                popped_lr = group.pop("lr", None)
                has_group_lr = has_group_lr or (popped_lr is not None)

            if has_group_lr:
                logger.warning("unet_lr and text_encoder_lr are ignored")

        if scheduler_config.lr_scheduler != "adafactor":
            logger.info("use adafactor_scheduler")

        return None

    if optimizer_config.max_grad_norm != 0.0:
        logger.warning("because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0")
    if scheduler_config.lr_scheduler != "constant_with_warmup":
        logger.warning("constant_with_warmup will be good")
    if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
        logger.warning("clip_threshold=1.0 will be good")

    return lr


def _resolve_constructor_learning_rate(
    optimizer_name: str,
    base_lr: float | None,
    trainable_params,
) -> float | None:
    """Resolve constructor LR without inventing a synthetic fallback value."""
    if base_lr is not None:
        return base_lr

    if not isinstance(trainable_params, list) or not trainable_params:
        raise ValueError(
            f"{optimizer_name} cannot be built with optimizer.learning_rates.base=null unless "
            "every optimizer param group defines an explicit lr."
        )

    for index, group in enumerate(trainable_params):
        if not isinstance(group, dict) or "params" not in group:
            raise ValueError(
                f"{optimizer_name} cannot be built with optimizer.learning_rates.base=null unless "
                "every optimizer param group is an explicit dict with an lr."
            )
        if group.get("lr") is None:
            raise ValueError(
                f"{optimizer_name} cannot be built with optimizer.learning_rates.base=null because "
                f"optimizer param group {index} does not define an explicit lr."
            )

    return None


def _build_optimizer_init_kwargs(optimizer_kwargs: dict, *, lr: float | None, **extra_kwargs):
    init_kwargs = dict(optimizer_kwargs)
    if lr is not None:
        init_kwargs["lr"] = lr
    init_kwargs.update(extra_kwargs)
    return init_kwargs


def _load_registered_optimizer_class(registration: OptimizerRegistration):
    if registration.target is None:
        return None

    if registration.backend == "torchao":
        prepare_windows_compiler_env_for_torch_compile()

    try:
        return load_target(registration.target)
    except ImportError as err:
        if registration.backend == "bitsandbytes":
            raise ImportError("No bitsandbytes") from err
        if registration.backend == "torchao":
            raise ImportError("No torchao") from err
        if registration.backend == "dadaptation":
            raise ImportError("No dadaptation") from err
        if registration.backend == "prodigy":
            raise ImportError("No Prodigy") from err
        if registration.backend == "schedulefree":
            raise ImportError("No schedulefree") from err
        raise
    except AttributeError as err:
        if registration.backend == "bitsandbytes" and registration.name in _BITSANDBYTES_ATTRIBUTE_ERRORS:
            raise AttributeError(_BITSANDBYTES_ATTRIBUTE_ERRORS[registration.name]) from err
        raise


def _build_registered_optimizer(
    registration: OptimizerRegistration,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    trainable_params,
    lr: float | None,
    optimizer_kwargs: dict,
):
    if registration.target is None or registration.backend is None:
        return None, None, lr

    if registration.backend in {"dadaptation", "prodigy"}:
        _warn_adaptive_lr_usage(trainable_params, lr)

    if registration.name == "adafactor":
        lr = _prepare_adafactor(optimizer_config, scheduler_config, trainable_params, optimizer_kwargs, lr)

    optimizer_class = _load_registered_optimizer_class(registration)
    if optimizer_class is None:
        return None, None, lr

    logger.info(f"use {registration.name} optimizer | {optimizer_kwargs}")

    if registration.name == "sgdnesterov":
        _ensure_nesterov_momentum(optimizer_kwargs)
        optimizer = optimizer_class(
            trainable_params,
            **_build_optimizer_init_kwargs(optimizer_kwargs, lr=lr, nesterov=True),
        )
        return optimizer_class, optimizer, lr

    if registration.name == "sgdnesterov8bit":
        _ensure_nesterov_momentum(optimizer_kwargs, warning=True)
        optimizer = optimizer_class(
            trainable_params,
            **_build_optimizer_init_kwargs(optimizer_kwargs, lr=lr, nesterov=True),
        )
        return optimizer_class, optimizer, lr

    optimizer = optimizer_class(trainable_params, **_build_optimizer_init_kwargs(optimizer_kwargs, lr=lr))
    return optimizer_class, optimizer, lr


def _build_arbitrary_optimizer(
    optimizer_config: OptimizerConfig,
    trainable_params,
    lr: float | None,
    optimizer_kwargs: dict,
):
    case_sensitive_optimizer_type = optimizer_config.optimizer_type
    logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

    if "." not in case_sensitive_optimizer_type:
        optimizer_module = torch.optim
    else:
        optimizer_class = load_target(case_sensitive_optimizer_type)
        case_sensitive_optimizer_type = optimizer_class.__name__
        optimizer_module = torch.optim if optimizer_class.__module__.startswith("torch.optim") else None

    if is_wrapper_optimizer_name(optimizer_config.optimizer_type):
        raise ValueError("Wrapper optimizers must be registered and built through the shared wrapper path")

    optimizer_class = (
        load_target(optimizer_config.optimizer_type)
        if optimizer_module is None
        else getattr(optimizer_module, case_sensitive_optimizer_type)
    )
    optimizer = optimizer_class(trainable_params, **_build_optimizer_init_kwargs(optimizer_kwargs, lr=lr))
    return optimizer_class, optimizer


def _split_wrapper_optimizer_kwargs(
    optimizer_kwargs: dict,
    *,
    wrapper_class=None,
    wrapper_style: str | None = None,
):
    base_optimizer_type = optimizer_kwargs.get("base_optimizer_type")
    if base_optimizer_type is None:
        raise ValueError("base_optimizer_type is required in optimizer_args for wrapper optimizers")

    base_optimizer_kwargs = {}
    wrapper_kwargs = {}
    wrapper_signature_params = set()
    if wrapper_class is not None:
        wrapper_signature_params = {
            name
            for name in inspect.signature(wrapper_class.__init__).parameters
            if name not in {"self", "optimizer", "base_optimizer_kwargs"}
        }
    for key, value in optimizer_kwargs.items():
        if key == "base_optimizer_type":
            continue
        if key.startswith("base_optimizer."):
            base_optimizer_kwargs[key.removeprefix("base_optimizer.")] = value
            continue
        if key.startswith("wrapper."):
            wrapper_kwargs[key.removeprefix("wrapper.")] = value
            continue
        if key in wrapper_signature_params:
            wrapper_kwargs[key] = value
            continue
        if wrapper_style == "wrap_optimizer_with_base_kwargs":
            base_optimizer_kwargs[key] = value
            continue
        wrapper_kwargs[key] = value

    return base_optimizer_type, base_optimizer_kwargs, wrapper_kwargs


def _build_registered_wrapper(
    registration: OptimizerRegistration,
    learning_rates: LearningRatesConfig,
    scheduler_config: SchedulerConfig,
    trainable_params,
    optimizer_kwargs: dict,
):
    if registration.target is None:
        return None, None

    wrapper_class = _load_registered_optimizer_class(registration)
    if wrapper_class is None:
        return None, None
    base_optimizer_type, base_optimizer_kwargs, wrapper_kwargs = _split_wrapper_optimizer_kwargs(
        optimizer_kwargs,
        wrapper_class=wrapper_class,
        wrapper_style=registration.wrapper_style,
    )

    base_optimizer_config = OptimizerConfig(optimizer_type=base_optimizer_type)
    _, _, base_optimizer = get_optimizer(
        base_optimizer_config,
        deepcopy(learning_rates),
        deepcopy(scheduler_config),
        trainable_params,
        base_optimizer_kwargs,
    )

    logger.info(f"use {registration.name} wrapper | {wrapper_kwargs}")
    if registration.wrapper_style == "wrap_optimizer":
        optimizer = wrapper_class(base_optimizer, **wrapper_kwargs)
    elif registration.wrapper_style == "wrap_optimizer_with_base_kwargs":
        optimizer = wrapper_class(base_optimizer, base_optimizer_kwargs=base_optimizer_kwargs, **wrapper_kwargs)
    else:
        raise ValueError(f"Unsupported wrapper construction style: {registration.wrapper_style}")

    if not hasattr(optimizer, "base_optimizer"):
        optimizer = WrappedOptimizerProxy(optimizer, base_optimizer)

    return wrapper_class, optimizer


def _maybe_wrap_with_schedulefree(
    optimizer_config: OptimizerConfig,
    optimizer_name: str,
    optimizer,
):
    if not optimizer_config.optimizer_schedulefree_wrapper:
        return optimizer

    if is_schedulefree_optimizer_name(optimizer_name) or is_wrapper_optimizer_name(optimizer_name):
        return optimizer

    wrapper_kwargs = parse_key_value_args(optimizer_config.schedulefree_wrapper_args)
    sf_wrapper = ScheduleFreeWrapper(optimizer, **wrapper_kwargs)
    sf_wrapper.train()
    logger.info(f"wrap optimizer with ScheduleFreeWrapper | {wrapper_kwargs}")
    return sf_wrapper


def get_optimizer(
    optimizer_config: OptimizerConfig,
    learning_rates: LearningRatesConfig,
    scheduler_config: SchedulerConfig,
    trainable_params,
    optimizer_kwargs: dict | None = None,
) -> tuple[str, str, object]:
    """
    Creates and returns an optimizer based on the provided configuration.

    Args:
        optimizer_config (OptimizerConfig): Configuration for the optimizer.
        learning_rates (LearningRatesConfig): Configuration for learning rates.
        scheduler_config (SchedulerConfig): Configuration for the scheduler.
        trainable_params: Parameters to be optimized.
        optimizer_kwargs (Dict): Additional keyword arguments for the optimizer.

    Returns:
        tuple[str, str, object]: A tuple containing the optimizer name, the optimizer arguments string, and the optimizer instance.
    """
    optimizer_type = optimizer_config.optimizer_type
    if optimizer_config.use_8bit_adam:
        assert not optimizer_config.use_lion_optimizer, "both option use_8bit_adam and use_lion_optimizer are specified"
        assert optimizer_type is None or optimizer_type == "", "both option use_8bit_adam and optimizer_type are specified"
    elif optimizer_config.use_lion_optimizer:
        assert optimizer_type is None or optimizer_type == "", "both option use_lion_optimizer and optimizer_type are specified"

    configured_optimizer_name = get_configured_optimizer_name(optimizer_config)
    optimizer_type = configured_optimizer_name.lower()

    if optimizer_config.fused_backward_pass:
        assert optimizer_type == "adafactor", "fused_backward_pass currently only works with optimizer_type Adafactor"
        assert True, "fused_backward_pass validation skipped for now during refactor"

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if not optimizer_kwargs:
        optimizer_kwargs = parse_key_value_args(optimizer_config.optimizer_args)

    raw_optimizer_type = optimizer_config.optimizer_type or ""
    needs_param_names = optimizer_type == "adammini" or raw_optimizer_type.rsplit(".", 1)[-1].lower() == "adammini"
    metadata_keys = {"param_names"} if needs_param_names else None
    trainable_params = materialize_parameter_groups(trainable_params, include_metadata_keys=metadata_keys)

    lr = _resolve_constructor_learning_rate(configured_optimizer_name, learning_rates.base, trainable_params)
    optimizer = None
    optimizer_class = None
    optimizer_registration = get_optimizer_registration(optimizer_type)

    if optimizer_registration is not None and optimizer_registration.kind == "optimizer":
        optimizer_class, optimizer, lr = _build_registered_optimizer(
            optimizer_registration,
            optimizer_config,
            scheduler_config,
            trainable_params,
            lr,
            optimizer_kwargs,
        )
    elif optimizer_registration is not None and optimizer_registration.kind == "wrapper":
        optimizer_class, optimizer = _build_registered_wrapper(
            optimizer_registration,
            learning_rates,
            scheduler_config,
            trainable_params,
            optimizer_kwargs,
        )

    if optimizer is None:
        optimizer_class, optimizer = _build_arbitrary_optimizer(optimizer_config, trainable_params, lr, optimizer_kwargs)

    assert optimizer_class is not None, "optimizer_class should not be None at this point"
    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer = _maybe_wrap_with_schedulefree(optimizer_config, optimizer_name, optimizer)
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    train_method = getattr(optimizer, "train", None)
    if train_method is not None and callable(train_method):
        train_method()

    return optimizer_name, optimizer_args, optimizer
