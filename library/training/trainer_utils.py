import time
import os

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import TorchDynamoPlugin

import library.optimizations.deepspeed_utils as deepspeed_utils
from omegaconf import OmegaConf
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.logging import LoggingConfig
from typing import Union, Any

def prepare_accelerator(cfg: PerformanceConfig):
    """
    this function also prepares deepspeed plugin
    """

    # logging_dir and log_prefix are in LoggingConfig, but logging_dir is required for accelerator if log_with is tensorboard.
    # log_with is in LoggingConfig.
    # However, prepare_accelerator is usually called with 'args' which contained everything.
    # We should assume cfg is a config object that has these attributes, or we need to pass both configs.
    # Since deepspeed_utils.prepare_deepspeed_plugin(args) also expects args, we likely need a combined object or access attributes.

    logging_dir = getattr(cfg, "logging_dir", None)
    log_prefix = getattr(cfg, "log_prefix", "")
    if log_prefix is None:
        log_prefix = ""

    if logging_dir is not None:
        logging_dir = logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    log_with = getattr(cfg, "log_with", None)

    if log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
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
            if getattr(cfg, "wandb_api_key", None) is not None:
                wandb.login(key=cfg.wandb_api_key)

    # torch.compile のオプション。 NO の場合は torch.compile は使わない
    # torch.compile のオプション。 NO の場合は torch.compile は使わない
    if getattr(cfg, "torch_compile", False):
        # Configure the compilation backend
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="default",  # Options: "default", "reduce-overhead", "max-autotune"
            fullgraph=False,
            dynamic=True,
            use_regional_compilation=True,
        )
    else:
        dynamo_plugin = None

    #    (
    #        InitProcessGroupKwargs(
    #            backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
    #            init_method=(
    #                "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
    #            ),
    #            timeout=datetime.timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
    #        )
    #        if torch.cuda.device_count() > 1
    #        else None
    #    ),

    kwargs_handlers = [
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=cfg.ddp_gradient_as_bucket_view, static_graph=cfg.ddp_static_graph
            )
            if cfg.ddp_gradient_as_bucket_view or cfg.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]
    deepspeed_plugin = deepspeed_utils.prepare_deepspeed_plugin(cfg)

    accelerator = Accelerator(
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 1), # In TrainingConfig usually
        mixed_precision=cfg.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_plugin=dynamo_plugin,
        deepspeed_plugin=deepspeed_plugin,
    )
    return accelerator


def init_trackers(accelerator: Accelerator, cfg: Any, default_tracker_name: str):
    """
    Initialize experiment trackers with tracker specific behaviors
    """
    # Assuming cfg is FullConfig or similar that can be converted to container

    if accelerator.is_main_process:
        init_kwargs = {}
        # Accessing logging config. Usually cfg.logging if it's FullConfig
        logging_cfg = getattr(cfg, "logging", cfg)

        wandb_run_name = getattr(logging_cfg, "wandb_run_name", None)
        if wandb_run_name:
            init_kwargs["wandb"] = {"name": wandb_run_name}

        log_tracker_config = getattr(logging_cfg, "log_tracker_config", None)
        if log_tracker_config is not None:
            init_kwargs = log_tracker_config

        # sanitize config for logging
        if hasattr(cfg, "to_container"): # Omegaconf
             config_to_log = OmegaConf.to_container(cfg, resolve=True)
        elif hasattr(cfg, "__dataclass_fields__"): # Dataclass
             from dataclasses import asdict
             config_to_log = asdict(cfg)
        else:
             config_to_log = vars(cfg) # argparse Namespace or simple object

        sensitive_keys = ["wandb_api_key", "huggingface_token"]

        # Recursive cleaning might be needed if config is nested
        def clean_recursive(d):
            if isinstance(d, dict):
                for key in sensitive_keys:
                    if key in d:
                        d[key] = "*****"
                for v in d.values():
                    clean_recursive(v)

        clean_recursive(config_to_log)

        log_tracker_name = getattr(logging_cfg, "log_tracker_name", None)
        accelerator.init_trackers(
            default_tracker_name if log_tracker_name is None else log_tracker_name,
            config=config_to_log,
            init_kwargs=init_kwargs,
        )


def calculate_val_loss_check(cfg, global_step, epoch_step, val_dataloader, train_dataloader) -> bool:
    if val_dataloader is None:
        return False

    max_train_steps = getattr(cfg, "max_train_steps", 1000000000) # TrainingConfig
    if global_step != 0 and global_step < max_train_steps:
        validation_every_n_step = getattr(cfg, "validation_every_n_step", None) # Not in dataclasses? check source code
        if validation_every_n_step is None:
             validation_every_n_step = getattr(cfg, "validate_every_n_steps", None) # TrainingConfig has validate_every_n_steps

        if validation_every_n_step is not None:
            if global_step % int(validation_every_n_step) != 0:
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


def determine_grad_sync_context(args, accelerator, sync_gradients, training_model, edm2_model=None):
    # TODO
    # if args.full_bf16:
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
