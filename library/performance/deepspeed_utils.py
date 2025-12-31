import os
import torch
import logging
from typing import Optional

from accelerate import DeepSpeedPlugin

from library.config.dataclasses.data import LoaderConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.performance import DeepSpeedConfig, PrecisionConfig
from library.utils.common_utils import setup_logging
from library.utils.device_utils import get_preferred_device

setup_logging()
logger = logging.getLogger(__name__)


def prepare_deepspeed_config(deepspeed_config: DeepSpeedConfig, loader_config: LoaderConfig = None):
    """Modify training config for deepspeed if enabled."""
    if not deepspeed_config.deepspeed:
        return
    
    if loader_config is not None:
        loader_config.max_workers = 1


def prepare_deepspeed_plugin(
    deepspeed_config: DeepSpeedConfig,
    precision_config: PrecisionConfig,
    training_config: TrainingConfig = None,
) -> Optional[DeepSpeedPlugin]:
    """Create deepspeed plugin from configs.
    
    Args:
        deepspeed_config: DeepSpeed settings (zero_stage, offload options, etc.)
        precision_config: Precision settings (mixed_precision, full_fp16)
        training_config: Optional training settings (gradient_accumulation_steps, train_batch_size)
    """
    if not deepspeed_config.deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    # Get values from training config if available, otherwise use defaults
    gradient_accumulation_steps = training_config.gradient_accumulation_steps if training_config else 1
    train_batch_size = training_config.train_batch_size if training_config else 1
    
    # Get max_grad_norm - typically on optimizer config, default to 1.0
    # Note: This may need to be passed in separately
    max_grad_norm = 1.0

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=deepspeed_config.zero_stage,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping=max_grad_norm,
        offload_optimizer_device=deepspeed_config.offload_optimizer_device,
        offload_optimizer_nvme_path=deepspeed_config.offload_optimizer_nvme_path,
        offload_param_device=deepspeed_config.offload_param_device,
        offload_param_nvme_path=deepspeed_config.offload_param_nvme_path,
        zero3_init_flag=deepspeed_config.zero3_init_flag,
        zero3_save_16bit_model=deepspeed_config.zero3_save_16bit_model,
    )
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_batch_size
    deepspeed_plugin.deepspeed_config["train_batch_size"] = (
            train_batch_size * gradient_accumulation_steps * int(os.environ.get("WORLD_SIZE", 1))
    )

    deepspeed_plugin.set_mixed_precision(precision_config.mixed_precision)
    if precision_config.mixed_precision.lower() == "fp16":
        deepspeed_plugin.deepspeed_config["fp16"]["initial_scale_power"] = 0
    if precision_config.full_fp16 or deepspeed_config.fp16_master_weights_and_gradients:
        if deepspeed_config.offload_optimizer_device == "cpu" and deepspeed_config.zero_stage == 2:
            deepspeed_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] = True
            logger.info("[DeepSpeed] full fp16 enable.")
        else:
            logger.info(
                "[DeepSpeed]full fp16, fp16_master_weights_and_grads currently only supported using ZeRO-Offload with DeepSpeedCPUAdam on ZeRO-2 stage."
            )

    if deepspeed_config.offload_optimizer_device is not None:
        logger.info("[DeepSpeed] start to manually build cpu_adam.")
        deepspeed.ops.op_builder.CPUAdamBuilder().load()
        logger.info("[DeepSpeed] building cpu_adam done.")

    return deepspeed_plugin


def prepare_deepspeed_model(precision_config: PrecisionConfig, **models):
    """Wrap models for DeepSpeed training.
    
    Args:
        precision_config: PrecisionConfig with mixed_precision setting
        **models: Named model arguments to wrap
    """
    models = {k: v for k, v in models.items() if v is not None}

    class DeepSpeedWrapper(torch.nn.Module):
        def __init__(self, **kw_models) -> None:
            super().__init__()

            self.models = torch.nn.ModuleDict()

            wrap_model_forward_with_torch_autocast = precision_config.mixed_precision != "no"

            for key, model in kw_models.items():
                if isinstance(model, list):
                    model = torch.nn.ModuleList(model)

                if wrap_model_forward_with_torch_autocast:
                    model = self.__wrap_model_with_torch_autocast(model)

                assert isinstance(
                    model, torch.nn.Module
                ), f"model must be an instance of torch.nn.Module, but got {key} is {type(model)}"

                self.models.update(torch.nn.ModuleDict({key: model}))

        def __wrap_model_with_torch_autocast(self, model):
            if isinstance(model, torch.nn.ModuleList):
                model = torch.nn.ModuleList([self.__wrap_model_forward_with_torch_autocast(m) for m in model])
            else:
                model = self.__wrap_model_forward_with_torch_autocast(model)
            return model

        def __wrap_model_forward_with_torch_autocast(self, model):

            assert hasattr(model, "forward"), f"model must have a forward method."

            forward_fn = model.forward

            def forward(*args, **kwargs):
                try:
                    device_type = model.device.type
                except AttributeError:
                    logger.warning(
                        "[DeepSpeed] model.device is not available. Using get_preferred_device() "
                        "to determine the device_type for torch.autocast()."
                    )
                    device_type = get_preferred_device().type

                with torch.autocast(device_type=device_type):
                    return forward_fn(*args, **kwargs)

            model.forward = forward
            return model

        def get_models(self):
            return self.models

    ds_model = DeepSpeedWrapper(**models)
    return ds_model
