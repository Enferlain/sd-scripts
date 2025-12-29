import time
import os
from typing import Optional

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import TorchDynamoPlugin

import library.performance.deepspeed_utils as deepspeed_utils
from omegaconf import OmegaConf

from library.config.dataclasses.performance import (
    PrecisionConfig,
    CompilationConfig,
    DistributedConfig,
    DeepSpeedConfig,
)
from library.config.dataclasses.output import LoggingConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.validation import ValidationConfig


def prepare_accelerator(
    precision_config: PrecisionConfig,
    compilation_config: CompilationConfig,
    distributed_config: DistributedConfig,
    deepspeed_config: DeepSpeedConfig,
    logging_config: LoggingConfig = None,
    training_config: TrainingConfig = None,
):
    """
    Prepare accelerator with optional deepspeed plugin.
    
    Args:
        precision_config: Precision settings (mixed_precision)
        compilation_config: Torch compile settings
        distributed_config: DDP settings (gradient_as_bucket_view, static_graph)
        deepspeed_config: DeepSpeed settings
        logging_config: Optional logging settings (logging_dir, log_with, wandb settings)
        training_config: Optional training settings (gradient_accumulation_steps)
    """

    # Handle logging directory
    if logging_config is None or logging_config.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if logging_config.log_prefix is None else logging_config.log_prefix
        logging_dir = logging_config.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    # Handle log_with setting
    if logging_config is None or logging_config.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = logging_config.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if logging_config.wandb_api_key is not None:
                wandb.login(key=logging_config.wandb_api_key)

    # torch.compile options
    if compilation_config.torch_compile:
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=True,
            use_regional_compilation=True,
        )
    else:
        dynamo_plugin = None

    # DDP kwargs
    kwargs_handlers = [
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=distributed_config.ddp_gradient_as_bucket_view,
                static_graph=distributed_config.ddp_static_graph
            )
            if distributed_config.ddp_gradient_as_bucket_view or distributed_config.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]
    
    # Deepspeed plugin
    deepspeed_plugin = deepspeed_utils.prepare_deepspeed_plugin(
        deepspeed_config, precision_config, training_config
    )

    # Gradient accumulation steps
    gradient_accumulation_steps = training_config.gradient_accumulation_steps if training_config else 1

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=precision_config.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_plugin=dynamo_plugin,
        deepspeed_plugin=deepspeed_plugin,
    )
    return accelerator


def init_trackers(accelerator: Accelerator, logging_config: LoggingConfig, default_tracker_name: str):
    """
    Initialize experiment trackers with tracker specific behaviors.
    
    Args:
        accelerator: Accelerator instance
        logging_config: LoggingConfig with tracker settings
        default_tracker_name: Default name for the tracker
    """
    if accelerator.is_main_process:
        init_kwargs = {}
        if hasattr(logging_config, 'wandb_run_name') and logging_config.wandb_run_name:
            init_kwargs["wandb"] = {"name": logging_config.wandb_run_name}
        if hasattr(logging_config, 'log_tracker_config') and logging_config.log_tracker_config is not None:
            init_kwargs = logging_config.log_tracker_config

        # sanitize config for logging - convert to dict if needed
        if hasattr(logging_config, '__dataclass_fields__'):
            from dataclasses import asdict
            config_to_log = asdict(logging_config)
        else:
            config_to_log = OmegaConf.to_container(logging_config, resolve=True)
        
        sensitive_keys = ["wandb_api_key", "huggingface_token"]
        for key in sensitive_keys:
            if key in config_to_log:
                config_to_log[key] = "*****"

        tracker_name = logging_config.log_tracker_name if hasattr(logging_config, 'log_tracker_name') and logging_config.log_tracker_name else default_tracker_name
        accelerator.init_trackers(
            tracker_name,
            config=config_to_log,
            init_kwargs=init_kwargs,
        )


def calculate_val_loss_check(
    validation_config: ValidationConfig,
    training_config: TrainingConfig,
    global_step: int,
    epoch_step: int,
    val_dataloader,
    train_dataloader,
) -> bool:
    """Check if validation should be run at this step.
    
    Args:
        validation_config: ValidationConfig object with validation settings
        training_config: TrainingConfig object for max_train_steps
        global_step: Current global training step
        epoch_step: Current step within the epoch
        val_dataloader: Validation dataloader (if None, returns False)
        train_dataloader: Training dataloader for epoch length check
    """
    if val_dataloader is None:
        return False

    if global_step != 0 and global_step < training_config.max_train_steps:
        if validation_config.validate_every_n_steps is not None:
            if global_step % int(validation_config.validate_every_n_steps) != 0:
                return False
        else:
            if epoch_step != len(train_dataloader) - 1:
                return False
    return True


def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True):
    names = []
    if including_unet:
        names.append("unet")
    names.append("text_encoder1")
    names.append("text_encoder2")

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names):
    lrs = lr_scheduler.get_last_lr()

    for lr_index in range(len(lrs)):
        name = names[lr_index]
        logs["lr/" + name] = float(lrs[lr_index])

        if optimizer_type.lower().startswith("DAdapt".lower()) or optimizer_type.lower() == "Prodigy".lower():
            logs["lr/d*lr/" + name] = (
                    lr_scheduler.optimizers[-1].param_groups[lr_index]["d"] *
                    lr_scheduler.optimizers[-1].param_groups[lr_index]["lr"]
            )


def determine_grad_sync_context(precision_config: Optional[PrecisionConfig], accelerator, sync_gradients, training_model, edm2_model=None):
    # TODO: Investigate why this was considered and update signature maybe?
    # if precision_config and precision_config.full_bf16:
    #    if not sync_gradients and accelerator.num_processes > 1:
    #        if edm2_model is not None:
    #            return accelerator.no_sync(training_model, edm2_model)
    #        else:
    #            return accelerator.no_sync(training_model)
    #    else:
    #        return contextlib.nullcontext()
    # else:
    if edm2_model is not None:
        return accelerator.accumulate(training_model, edm2_model)
    else:
        return accelerator.accumulate(training_model)
