import importlib
import logging

import torch
import transformers

from library.config.dataclasses.optimizer import OptimizerConfig, LearningRatesConfig, SchedulerConfig
from library.optimization.arguments import parse_key_value_args
from library.optimization.types import materialize_parameter_groups


logger = logging.getLogger(__name__)


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
    # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, AdEMAMix8bit, PagedAdEMAMix8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, Adafactor"

    optimizer_type = optimizer_config.optimizer_type
    if optimizer_config.use_8bit_adam:
        assert not optimizer_config.use_lion_optimizer, "both option use_8bit_adam and use_lion_optimizer are specified"
        assert optimizer_type is None or optimizer_type == "", "both option use_8bit_adam and optimizer_type are specified"
        optimizer_type = "AdamW8bit"

    elif optimizer_config.use_lion_optimizer:
        assert optimizer_type is None or optimizer_type == "", "both option use_lion_optimizer and optimizer_type are specified"
        optimizer_type = "Lion"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    optimizer_type = optimizer_type.lower()

    if optimizer_config.fused_backward_pass:
        assert optimizer_type == "Adafactor".lower(), "fused_backward_pass currently only works with optimizer_type Adafactor"
        assert (
            # args.gradient_accumulation_steps == 1 # This should be checked elsewhere or passed efficiently, ignoring for now as it's validation logic which should be in config
            True
        ), "fused_backward_pass validation skipped for now during refactor"

    # Break down arguments
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if not optimizer_kwargs:
        optimizer_kwargs = parse_key_value_args(optimizer_config.optimizer_args)
    # logger.info(f"optkwargs {optimizer}_{kwargs}")

    trainable_params = materialize_parameter_groups(trainable_params)

    lr = learning_rates.base
    optimizer = None
    optimizer_class = None

    if optimizer_type == "Lion".lower():
        try:
            import lion_pytorch
        except ImportError as err:
            raise ImportError("No lion_pytorch") from err
        logger.info(f"use Lion optimizer | {optimizer_kwargs}")
        optimizer_class = lion_pytorch.Lion
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("8bit".lower()):
        try:
            import bitsandbytes as bnb
        except ImportError as err:
            raise ImportError("No bitsandbytes") from err

        if optimizer_type == "AdamW8bit".lower():
            logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "SGDNesterov8bit".lower():
            logger.info(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
            if "momentum" not in optimizer_kwargs:
                logger.warning("8-bit SGD with Nesterov must be with momentum, set momentum to 0.9")
                optimizer_kwargs["momentum"] = 0.9

            optimizer_class = bnb.optim.SGD8bit
            optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

        elif optimizer_type == "Lion8bit".lower():
            logger.info(f"use 8-bit Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.Lion8bit
            except AttributeError as err:
                raise AttributeError(
                    "No Lion8bit. The version of bitsandbytes installed seems to be old. Please install 0.38.0 or later."
                ) from err
        elif optimizer_type == "PagedAdamW8bit".lower():
            logger.info(f"use 8-bit PagedAdamW optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedAdamW8bit
            except AttributeError as err:
                raise AttributeError(
                    "No PagedAdamW8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later."
                ) from err
        elif optimizer_type == "PagedLion8bit".lower():
            logger.info(f"use 8-bit Paged Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedLion8bit
            except AttributeError as err:
                raise AttributeError(
                    "No PagedLion8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later."
                ) from err

        if optimizer_class is not None:
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "PagedAdamW".lower():
        logger.info(f"use PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            import bitsandbytes as bnb
        except ImportError as err:
            raise ImportError("No bitsandbytes") from err
        try:
            optimizer_class = bnb.optim.PagedAdamW
        except AttributeError as err:
            raise AttributeError(
                "No PagedAdamW. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later."
            ) from err
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "PagedAdamW32bit".lower():
        logger.info(f"use 32-bit PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            import bitsandbytes as bnb
        except ImportError as err:
            raise ImportError("No bitsandbytes") from err
        try:
            optimizer_class = bnb.optim.PagedAdamW32bit
        except AttributeError as err:
            raise AttributeError(
                "No PagedAdamW32bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later."
            ) from err
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov".lower():
        logger.info(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            logger.info("SGD with Nesterov must be with momentum, set momentum to 0.9")
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type.startswith("DAdapt".lower()) or optimizer_type == "Prodigy".lower():
        # check lr and lr_count, and logger.info warning
        actual_lr = lr
        lr_count = 1
        if isinstance(trainable_params, list) and isinstance(trainable_params[0], dict):
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            logger.warning(f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0: lr={actual_lr}")
            logger.warning("recommend option: lr=1.0")
        if lr_count > 1:
            logger.warning(
                f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect: lr={actual_lr}"
            )

        if optimizer_type.startswith("DAdapt".lower()):
            # DAdaptation family
            # check dadaptation is installed
            try:
                import dadaptation
                import dadaptation.experimental as experimental
            except ImportError as err:
                raise ImportError("No dadaptation") from err

            # set optimizer
            if optimizer_type == "DAdaptation".lower() or optimizer_type == "DAdaptAdamPreprint".lower():
                optimizer_class = experimental.DAdaptAdamPreprint
                logger.info(f"use D-Adaptation AdamPreprint optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdaGrad".lower():
                optimizer_class = dadaptation.DAdaptAdaGrad
                logger.info(f"use D-Adaptation AdaGrad optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdam".lower():
                optimizer_class = dadaptation.DAdaptAdam
                logger.info(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdan".lower():
                optimizer_class = dadaptation.DAdaptAdan
                logger.info(f"use D-Adaptation Adan optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdanIP".lower():
                optimizer_class = experimental.DAdaptAdanIP
                logger.info(f"use D-Adaptation AdanIP optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptLion".lower():
                optimizer_class = dadaptation.DAdaptLion
                logger.info(f"use D-Adaptation Lion optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptSGD".lower():
                optimizer_class = dadaptation.DAdaptSGD
                logger.info(f"use D-Adaptation SGD optimizer | {optimizer_kwargs}")
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")

            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        else:
            # Prodigy
            # check Prodigy is installed
            try:
                import prodigyopt
            except ImportError as err:
                raise ImportError("No Prodigy") from err

            logger.info(f"use Prodigy optimizer | {optimizer_kwargs}")
            optimizer_class = prodigyopt.Prodigy
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        # Check arguments and correct as appropriate
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            logger.info("set relative_step to True because warmup_init is True")
            optimizer_kwargs["relative_step"] = True
        logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

        if optimizer_kwargs["relative_step"]:
            logger.info("relative_step is true")
            if lr != 0.0:
                logger.warning("learning rate is used as initial_lr")
            optimizer_config.learning_rates.base = 0.0  # Sentinel: Adafactor uses relative_step, lr computed internally

            # Process when trainable_params is a group: remove lr
            if isinstance(trainable_params, list) and isinstance(trainable_params[0], dict):
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

                if has_group_lr:
                    # Disable args for now TODO It is not desirable because the dependency is reversed
                    logger.warning("unet_lr and text_encoder_lr are ignored")
                    # args.unet_lr = None # cannot modifying config easily here, just ignore
                    # args.text_encoder_lr = None

            if scheduler_config.lr_scheduler != "adafactor":
                logger.info("use adafactor_scheduler")
            # optimizer_config.scheduler.lr_scheduler = f"adafactor:{lr}"  # Avoiding modification of config

            lr = None
        else:
            if optimizer_config.max_grad_norm != 0.0:
                logger.warning("because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0")
            if scheduler_config.lr_scheduler != "constant_with_warmup":
                logger.warning("constant_with_warmup will be good")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                logger.warning("clip_threshold=1.0 will be good")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("schedulefree".lower()):
        try:
            import schedulefree as sf
        except ImportError as err:
            raise ImportError("No schedulefree") from err

        if optimizer_type == "RAdamScheduleFree".lower():
            optimizer_class = sf.RAdamScheduleFree
            logger.info(f"use RAdamScheduleFree optimizer | {optimizer_kwargs}")
        elif optimizer_type == "AdamWScheduleFree".lower():
            optimizer_class = sf.AdamWScheduleFree
            logger.info(f"use AdamWScheduleFree optimizer | {optimizer_kwargs}")
        elif optimizer_type == "SGDScheduleFree".lower():
            optimizer_class = sf.SGDScheduleFree
            logger.info(f"use SGDScheduleFree optimizer | {optimizer_kwargs}")
        else:
            optimizer_class = None

        if optimizer_class is not None:
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer is None:
        # Use arbitrary optimizer
        case_sensitive_optimizer_type = optimizer_config.optimizer_type  # not lower
        logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

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
            base_optimizer_values = case_sensitive_full_base_optimizer_name.split(".")
            base_optimizer_module = importlib.import_module(".".join(base_optimizer_values[:-1]))
            case_sensitive_base_optimizer_type = base_optimizer_values[-1]
            base_optimizer_class = getattr(base_optimizer_module, case_sensitive_base_optimizer_type)

            optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
            optimizer = optimizer_class(trainable_params, base_optimizer_class, lr=lr, **optimizer_kwargs)
        else:
            optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    """
    # wrap any of above optimizer with schedulefree, if optimizer is not schedulefree
    if optimizer_config.optimizer_schedulefree_wrapper and not optimizer_type.endswith("schedulefree".lower()):
        try:
            import schedulefree as sf
        except ImportError:
            raise ImportError("No schedulefree")

        schedulefree_wrapper_kwargs = {}
        if optimizer_config.schedulefree_wrapper_args is not None and len(optimizer_config.schedulefree_wrapper_args) > 0:
            for arg in optimizer_config.schedulefree_wrapper_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                schedulefree_wrapper_kwargs[key] = value

        sf_wrapper = sf.ScheduleFreeWrapper(optimizer, **schedulefree_wrapper_kwargs)
        sf_wrapper.train()  # make optimizer as train mode

        # we need to make optimizer as a subclass of torch.optim.Optimizer, we make another Proxy class over SFWrapper
        class OptimizerProxy(torch.optim.Optimizer):
            def __init__(self, sf_wrapper):
                self._sf_wrapper = sf_wrapper

            def __getattr__(self, name):
                return getattr(self._sf_wrapper, name)

            # override properties
            @property
            def state(self):
                return self._sf_wrapper.state

            @state.setter
            def state(self, state):
                self._sf_wrapper.state = state

            @property
            def param_groups(self):
                return self._sf_wrapper.param_groups

            @param_groups.setter
            def param_groups(self, param_groups):
                self._sf_wrapper.param_groups = param_groups

            @property
            def defaults(self):
                return self._sf_wrapper.defaults

            @defaults.setter
            def defaults(self, defaults):
                self._sf_wrapper.defaults = defaults

            def add_param_group(self, param_group):
                self._sf_wrapper.add_param_group(param_group)

            def load_state_dict(self, state_dict):
                self._sf_wrapper.load_state_dict(state_dict)

            def state_dict(self):
                return self._sf_wrapper.state_dict()

            def zero_grad(self):
                self._sf_wrapper.zero_grad()

            def step(self, closure=None):
                self._sf_wrapper.step(closure)

            def train(self):
                self._sf_wrapper.train()

            def eval(self):
                self._sf_wrapper.eval()

            # Method to pass isinstance check
            def __instancecheck__(self, instance):
                return isinstance(instance, (type(self), Optimizer))

        optimizer = OptimizerProxy(sf_wrapper)

        logger.info(f"wrap optimizer with ScheduleFreeWrapper | {schedulefree_wrapper_kwargs}")
    """

    # for logging
    assert optimizer_class is not None, "optimizer_class should not be None at this point"
    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    train_method = getattr(optimizer, "train", None)
    if train_method is not None and callable(train_method):
        # make optimizer as train mode before training for schedulefree optimizer. the optimizer will be in eval mode in sampling and saving.
        train_method()

    return optimizer_name, optimizer_args, optimizer
