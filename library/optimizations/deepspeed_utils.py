import os
import torch
import logging

from accelerate import DeepSpeedPlugin

from library.config.dataclasses.training import TrainingConfig
from library.utils.common_utils import setup_logging
from library.utils.device_utils import get_preferred_device

setup_logging()
logger = logging.getLogger(__name__)

def prepare_deepspeed_args(cfg: TrainingConfig):
    if not cfg.deepspeed.deepspeed:
        return

    cfg.max_data_loader_n_workers = 1

def prepare_deepspeed_plugin(cfg: TrainingConfig):
    if not cfg.deepspeed.deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=cfg.deepspeed.zero_stage,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_clipping=cfg.max_grad_norm,
        offload_optimizer_device=cfg.deepspeed.offload_optimizer_device,
        offload_optimizer_nvme_path=cfg.deepspeed.offload_optimizer_nvme_path,
        offload_param_device=cfg.deepspeed.offload_param_device,
        offload_param_nvme_path=cfg.deepspeed.offload_param_nvme_path,
        zero3_init_flag=cfg.deepspeed.zero3_init_flag,
        zero3_save_16bit_model=cfg.deepspeed.zero3_save_16bit_model,
    )
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = cfg.train_batch_size
    deepspeed_plugin.deepspeed_config["train_batch_size"] = (
            cfg.train_batch_size * cfg.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])
    )

    deepspeed_plugin.set_mixed_precision(cfg.mixed_precision)
    if cfg.mixed_precision.lower() == "fp16":
        deepspeed_plugin.deepspeed_config["fp16"]["initial_scale_power"] = 0
    if cfg.full_fp16 or cfg.deepspeed.fp16_master_weights_and_gradients:
        if cfg.deepspeed.offload_optimizer_device == "cpu" and cfg.deepspeed.zero_stage == 2:
            deepspeed_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] = True
            logger.info("[DeepSpeed] full fp16 enable.")
        else:
            logger.info(
                "[DeepSpeed]full fp16, fp16_master_weights_and_grads currently only supported using ZeRO-Offload with DeepSpeedCPUAdam on ZeRO-2 stage."
            )

    if cfg.deepspeed.offload_optimizer_device is not None:
        logger.info("[DeepSpeed] start to manually build cpu_adam.")
        deepspeed.ops.op_builder.CPUAdamBuilder().load()
        logger.info("[DeepSpeed] building cpu_adam done.")

    return deepspeed_plugin

def prepare_deepspeed_model(cfg: TrainingConfig, **models):
    models = {k: v for k, v in models.items() if v is not None}

    class DeepSpeedWrapper(torch.nn.Module):
        def __init__(self, **kw_models) -> None:
            super().__init__()

            self.models = torch.nn.ModuleDict()

            wrap_model_forward_with_torch_autocast = cfg.mixed_precision != "no"

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
