import os
import asyncio
import json
import shutil
import logging
import safetensors.torch

from typing import Optional
from huggingface_hub import hf_hub_download

from library.utils import huggingface_util
from library.config.dataclasses.output import SavingConfig
from library.config.dataclasses.output import HuggingFaceConfig

from library.constants import (
    SS_METADATA_KEY_ADAPTER_MODULE,
    SS_METADATA_KEY_ADAPTER_RANK,
    SS_METADATA_KEY_ADAPTER_ALPHA,
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_ADAPTER_ARGS,
    DEFAULT_EPOCH_NAME,
    EPOCH_FILE_NAME,
    DEFAULT_STEP_NAME,
    STEP_FILE_NAME,
    DEFAULT_LAST_OUTPUT_NAME,
    EPOCH_DIFFUSERS_DIR_NAME,
    STEP_DIFFUSERS_DIR_NAME,
    EPOCH_STATE_NAME,
    STEP_STATE_NAME,
    LAST_STATE_NAME
)
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


def build_minimum_adapter_metadata(
        v2: Optional[str],
        base_model: Optional[str],
        adapter_module: str,
        adapter_rank: str,
        adapter_alpha: str,
        adapter_args: Optional[dict],
):
    # old LoRA doesn't have base_model
    metadata = {
        SS_METADATA_KEY_ADAPTER_MODULE: adapter_module,
        SS_METADATA_KEY_ADAPTER_RANK: adapter_rank,
        SS_METADATA_KEY_ADAPTER_ALPHA: adapter_alpha,
    }
    if v2 is not None:
        metadata[SS_METADATA_KEY_V2] = v2
    if base_model is not None:
        metadata[SS_METADATA_KEY_BASE_MODEL_VERSION] = base_model
    if adapter_args is not None:
        metadata[SS_METADATA_KEY_ADAPTER_ARGS] = json.dumps(adapter_args)
    return metadata


# NOTE: Legacy get_model_metadata() and get_sai_model_spec_dataclass() removed.
# Use library.utils.model_metadata.get_model_metadata_from_config() instead.


def resume_from_local_or_hf_if_specified(accelerator, saving_config: SavingConfig, hf_config: Optional[HuggingFaceConfig] = None):
    if not saving_config.resume:
        return

    if hf_config is None or not hf_config.resume_from_huggingface:
        logger.info(f"resume training from local state: {saving_config.resume}")
        accelerator.load_state(saving_config.resume)
        return

    logger.info(f"resume training from huggingface state: {saving_config.resume}")
    repo_id = saving_config.resume.split("/")[0] + "/" + saving_config.resume.split("/")[1]
    path_in_repo = "/".join(saving_config.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    logger.info(f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}")

    list_files = huggingface_util.list_dir(
        repo_id=repo_id,
        subfolder=path_in_repo,
        revision=revision,
        token=hf_config.huggingface_token,  # CONFIG ERROR
        repo_type=repo_type,
    )

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=hf_config.huggingface_token,  # CONFIG ERROR
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)  # FIXME: Parameter 'args' unfilled, expected '*tuple[]'

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
    if len(results) == 0:
        raise ValueError(
            "No files found in the specified repo id/path/revision / 指定されたリポジトリID/パス/リビジョンにファイルが見つかりませんでした"
        )
    dirname = os.path.dirname(results[0])
    accelerator.load_state(dirname)


def default_if_none(value, default):
    return default if value is None else value


def get_epoch_ckpt_name(saving_config: SavingConfig, ext: str, epoch_no: int, output_name_append: str = ""):
    model_name = default_if_none(saving_config.output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name + output_name_append, epoch_no) + ext


def get_step_ckpt_name(saving_config: SavingConfig, ext: str, step_no: int, output_name_append: str = ""):
    model_name = default_if_none(saving_config.output_name, DEFAULT_STEP_NAME)
    return STEP_FILE_NAME.format(model_name + output_name_append, step_no) + ext


def get_last_ckpt_name(saving_config: SavingConfig, ext: str, output_name_append: str = ""):
    model_name = default_if_none(saving_config.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + output_name_append + ext


def get_remove_epoch_no(saving_config: SavingConfig, epoch_no: int):
    if saving_config.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - saving_config.save_every_n_epochs * saving_config.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(saving_config: SavingConfig, step_no: int):
    if saving_config.save_last_n_steps is None:
        return None

    # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
    # save_every_n_steps=10, save_last_n_steps=30の場合、50step目には30step分残し、10step目を削除する
    remove_step_no = step_no - saving_config.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % saving_config.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


# NOTE: SD1.5/2-specific save_sd_model_on_epoch_end_or_stepwise moved to sd_checkpointing.py


def save_sd_model_on_epoch_end_or_stepwise_common(
        saving_config: SavingConfig,
        on_epoch_end: bool,
        accelerator,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        epoch: int,
        num_train_epochs: int,
        global_step: int,
        sd_saver,
        diffusers_saver,
        hf_config: Optional[HuggingFaceConfig] = None,
):
    if on_epoch_end:
        epoch_no = epoch + 1
        saving = epoch_no % saving_config.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        if not saving:
            return

        model_name = default_if_none(saving_config.output_name, DEFAULT_EPOCH_NAME)
        remove_no = get_remove_epoch_no(saving_config, epoch_no)
    else:
        # 保存するか否かは呼び出し側で判断済み

        model_name = default_if_none(saving_config.output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch  # 例: 最初のepochの途中で保存したら0になる、SDモデルに保存される
        remove_no = get_remove_step_no(saving_config, global_step)

    os.makedirs(saving_config.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(saving_config, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(saving_config, ext, global_step)

        ckpt_file = os.path.join(saving_config.output_dir, ckpt_name)
        logger.info("")
        logger.info(f"saving checkpoint: {ckpt_file}")
        sd_saver(ckpt_file, epoch_no, global_step)

        # Upload to HuggingFace if configured
        if hf_config is not None and hf_config.huggingface_repo_id is not None:
            huggingface_util.upload(hf_config, ckpt_file, "/" + ckpt_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_ckpt_name = get_epoch_ckpt_name(saving_config, ext, remove_no)
            else:
                remove_ckpt_name = get_step_ckpt_name(saving_config, ext, remove_no)

            remove_ckpt_file = os.path.join(saving_config.output_dir, remove_ckpt_name)
            if os.path.exists(remove_ckpt_file):
                logger.info(f"removing old checkpoint: {remove_ckpt_file}")
                os.remove(remove_ckpt_file)

    else:
        if on_epoch_end:
            out_dir = os.path.join(saving_config.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no))
        else:
            out_dir = os.path.join(saving_config.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step))

        logger.info("")
        logger.info(f"saving model: {out_dir}")
        diffusers_saver(out_dir)

        # Upload to HuggingFace if configured
        if hf_config is not None and hf_config.huggingface_repo_id is not None:
            huggingface_util.upload(hf_config, out_dir, "/" + model_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_out_dir = os.path.join(saving_config.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, remove_no))
            else:
                remove_out_dir = os.path.join(saving_config.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, remove_no))

            if os.path.exists(remove_out_dir):
                logger.info(f"removing old model: {remove_out_dir}")
                shutil.rmtree(remove_out_dir)

    if saving_config.save_state:
        if on_epoch_end:
            save_and_remove_state_on_epoch_end(saving_config, accelerator, epoch_no)
        else:
            save_and_remove_state_stepwise(saving_config, accelerator, global_step)


def save_and_remove_state_on_epoch_end(saving_config: SavingConfig, accelerator, epoch_no,
                                       hf_config: Optional[HuggingFaceConfig] = None):
    model_name = default_if_none(saving_config.output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(saving_config.output_dir, exist_ok=True)

    state_dir = os.path.join(saving_config.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)

    # Upload state to HuggingFace if configured
    if hf_config is not None and hf_config.save_state_to_huggingface and hf_config.huggingface_repo_id is not None:
        huggingface_util.upload(hf_config, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))
    
    last_n_epochs = saving_config.save_last_n_epochs_state if saving_config.save_last_n_epochs_state else saving_config.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - saving_config.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(saving_config.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(saving_config: SavingConfig, accelerator, step_no,
                                   hf_config: Optional[HuggingFaceConfig] = None):
    model_name = default_if_none(saving_config.output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(saving_config.output_dir, exist_ok=True)

    state_dir = os.path.join(saving_config.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)

    # Upload state to HuggingFace if configured
    if hf_config is not None and hf_config.save_state_to_huggingface and hf_config.huggingface_repo_id is not None:
        huggingface_util.upload(hf_config, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

    last_n_steps = saving_config.save_last_n_steps_state if saving_config.save_last_n_steps_state else saving_config.save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % saving_config.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(saving_config.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(saving_config: SavingConfig, accelerator, hf_config: Optional[HuggingFaceConfig] = None):
    model_name = default_if_none(saving_config.output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(saving_config.output_dir, exist_ok=True)

    state_dir = os.path.join(saving_config.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)

    # Upload state to HuggingFace if configured
    if hf_config is not None and hf_config.save_state_to_huggingface and hf_config.huggingface_repo_id is not None:
        huggingface_util.upload(hf_config, state_dir, "/" + LAST_STATE_NAME.format(model_name))


# NOTE: SD1.5/2-specific save_sd_model_on_train_end moved to sd_checkpointing.py


def save_sd_model_on_train_end_common(
        saving_config: SavingConfig,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        epoch: int,
        global_step: int,
        sd_saver,
        diffusers_saver,
        hf_config: Optional[HuggingFaceConfig] = None,
):
    model_name = default_if_none(saving_config.output_name, DEFAULT_LAST_OUTPUT_NAME)

    if save_stable_diffusion_format:
        os.makedirs(saving_config.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(saving_config.output_dir, ckpt_name)

        logger.info(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        sd_saver(ckpt_file, epoch, global_step)

        # Upload to HuggingFace if configured
        if hf_config is not None and hf_config.huggingface_repo_id is not None:
            huggingface_util.upload(hf_config, ckpt_file, "/" + ckpt_name)
    else:
        out_dir = os.path.join(saving_config.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"save trained model as Diffusers to {out_dir}")
        diffusers_saver(out_dir)

        # Upload to HuggingFace if configured
        if hf_config is not None and hf_config.huggingface_repo_id is not None:
            huggingface_util.upload(hf_config, out_dir, "/" + model_name)


def register_adapter_state_hooks(accelerator, adapter, cfg, current_epoch, current_step):
    """
    Register save/load hooks for peft-only checkpointing.

    These hooks ensure that only the PEFT peft weights (LoRA/LyCORIS) are saved/loaded
    during checkpointing, not the full base model weights.

    Args:
        accelerator: HuggingFace Accelerator
        adapter: The PEFT peft to save/load
        cfg: Training configuration (needs cfg.performance.deepspeed)
        current_epoch: Shared Value for current epoch tracking
        current_step: Shared Value for current step tracking

    Returns:
        Callable that returns steps_from_state (or None if not resumed)
    """
    # Container for steps loaded from state (nonlocal workaround)
    state_container = {"steps_from_state": None}

    def save_model_hook(models, weights, output_dir):
        # pop weights of other models than peft to save only peft weights
        # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
        if accelerator.is_main_process or cfg.performance.deepspeed:
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(adapter))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                if len(weights) > i:
                    weights.pop(i)

        # save current epoch and step
        train_state_file = os.path.join(output_dir, "train_state.json")
        # +1 is needed because the state is saved before current_step is set from global_step
        logger.info(
            f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value + 1}")
        with open(train_state_file, "w", encoding="utf-8") as f:
            json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

    def load_model_hook(models, input_dir):
        # remove models except peft
        remove_indices = []
        for i, model in enumerate(models):
            if not isinstance(model, type(accelerator.unwrap_model(adapter))):
                remove_indices.append(i)
        for i in reversed(remove_indices):
            models.pop(i)

        # load current epoch and step
        train_state_file = os.path.join(input_dir, "train_state.json")
        if os.path.exists(train_state_file):
            with open(train_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            state_container["steps_from_state"] = data["current_step"]
            logger.info(f"load train state from {train_state_file}: {data}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    def get_steps_from_state():
        return state_container["steps_from_state"]

    return get_steps_from_state
