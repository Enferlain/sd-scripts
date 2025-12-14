import os
import asyncio
import json
import hashlib
import shutil
import subprocess
import time
import logging
import safetensors.torch
import torch

from typing import Optional, Any
from io import BytesIO
from huggingface_hub import hf_hub_download

from library.utils import sai_model_spec, huggingface_util
from library.models import model_util

from library.constants import (
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_ARGS,
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

logger = logging.getLogger(__name__)


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode(
            "ascii").strip()
    except:
        return "(unknown)"


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


def build_minimum_network_metadata(
        v2: Optional[str],
        base_model: Optional[str],
        network_module: str,
        network_dim: str,
        network_alpha: str,
        network_args: Optional[dict],
):
    # old LoRA doesn't have base_model
    metadata = {
        SS_METADATA_KEY_NETWORK_MODULE: network_module,
        SS_METADATA_KEY_NETWORK_DIM: network_dim,
        SS_METADATA_KEY_NETWORK_ALPHA: network_alpha,
    }
    if v2 is not None:
        metadata[SS_METADATA_KEY_V2] = v2
    if base_model is not None:
        metadata[SS_METADATA_KEY_BASE_MODEL_VERSION] = base_model
    if network_args is not None:
        metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(network_args)
    return metadata


def get_sai_model_spec(
        state_dict: dict,
        args: Any,
        sdxl: bool,
        lora: bool,
        textual_inversion: bool,
        is_stable_diffusion_ckpt: Optional[bool] = None,  # None for TI and LoRA
        flux: str = None,  # "dev", "schnell" or "chroma"
        lumina: str = None,
        optional_metadata: dict[str, str] | None = None,
):
    timestamp = time.time()

    v2 = getattr(args, "v2", False)
    v_parameterization = getattr(args, "v_parameterization", False)
    reso = getattr(args, "resolution", None)

    metadata_title = getattr(args, "metadata_title", None)
    output_name = getattr(args, "output_name", None)
    title = metadata_title if metadata_title is not None else output_name

    min_timestep = getattr(args, "min_timestep", None)
    max_timestep = getattr(args, "max_timestep", None)

    if min_timestep is not None or max_timestep is not None:
        min_time_step = min_timestep if min_timestep is not None else 0
        max_time_step = max_timestep if max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    # Convert individual model parameters to model_config dict
    # TODO: Update calls to this function to pass in the model config
    model_config = {}
    if flux is not None:
        model_config["flux"] = flux
    if lumina is not None:
        model_config["lumina"] = lumina

    # Extract metadata_* fields from args and merge with optional_metadata
    extracted_metadata = {}

    # Extract all metadata_* attributes from args/config
    # Supports both object attributes and dictionary-like access
    if hasattr(args, "__dict__"):
        iterator = dir(args)
        getter = getattr
    elif isinstance(args, dict):
        iterator = args.keys()
        getter = lambda obj, key: obj[key]
    else:
        iterator = []
        getter = lambda obj, key: None

    for attr_name in iterator:
        if attr_name.startswith("metadata_") and not attr_name.startswith("metadata___"):
            value = getter(args, attr_name)
            if value is not None:
                # Remove metadata_ prefix and exclude already handled fields
                field_name = attr_name[9:]  # len("metadata_") = 9
                if field_name not in ["title", "author", "description", "license", "tags"]:
                    extracted_metadata[field_name] = value

    # Merge extracted metadata with provided optional_metadata
    all_optional_metadata = {**extracted_metadata}
    if optional_metadata:
        all_optional_metadata.update(optional_metadata)

    metadata_author = getattr(args, "metadata_author", None)
    metadata_description = getattr(args, "metadata_description", None)
    metadata_license = getattr(args, "metadata_license", None)
    metadata_tags = getattr(args, "metadata_tags", None)
    clip_skip = getattr(args, "clip_skip", None)

    metadata = sai_model_spec.build_metadata(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=metadata_author,
        description=metadata_description,
        license=metadata_license,
        tags=metadata_tags,
        timesteps=timesteps,
        clip_skip=clip_skip,  # None or int
        model_config=model_config,
        optional_metadata=all_optional_metadata if all_optional_metadata else None,
    )
    return metadata


def get_sai_model_spec_dataclass(
        state_dict: dict,
        args: Any,
        sdxl: bool,
        lora: bool,
        textual_inversion: bool,
        is_stable_diffusion_ckpt: Optional[bool] = None,
        flux: str = None,
        lumina: str = None,
        hunyuan_image: str = None,
        optional_metadata: dict[str, str] | None = None,
) -> sai_model_spec.ModelSpecMetadata:
    """
    Get ModelSpec metadata as a dataclass - preferred for new code.
    Automatically extracts metadata_* fields from args.
    """
    timestamp = time.time()

    v2 = getattr(args, "v2", False)
    v_parameterization = getattr(args, "v_parameterization", False)
    reso = getattr(args, "resolution", None)

    metadata_title = getattr(args, "metadata_title", None)
    output_name = getattr(args, "output_name", None)
    title = metadata_title if metadata_title is not None else output_name

    min_timestep = getattr(args, "min_timestep", None)
    max_timestep = getattr(args, "max_timestep", None)

    if min_timestep is not None or max_timestep is not None:
        min_time_step = min_timestep if min_timestep is not None else 0
        max_time_step = max_timestep if max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    # Convert individual model parameters to model_config dict
    model_config = {}
    if flux is not None:
        model_config["flux"] = flux
    if lumina is not None:
        model_config["lumina"] = lumina
    if hunyuan_image is not None:
        model_config["hunyuan_image"] = hunyuan_image

    metadata_author = getattr(args, "metadata_author", None)
    metadata_description = getattr(args, "metadata_description", None)
    metadata_license = getattr(args, "metadata_license", None)
    metadata_tags = getattr(args, "metadata_tags", None)
    clip_skip = getattr(args, "clip_skip", None)

    # Use the dataclass function directly
    return sai_model_spec.build_metadata_dataclass(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=metadata_author,
        description=metadata_description,
        license=metadata_license,
        tags=metadata_tags,
        timesteps=timesteps,
        clip_skip=clip_skip,
        model_config=model_config,
        optional_metadata=optional_metadata,
    )


def resume_from_local_or_hf_if_specified(accelerator, args):
    resume = getattr(args, "resume", None)
    if not resume:
        return

    resume_from_huggingface = getattr(args, "resume_from_huggingface", False)
    if not resume_from_huggingface:
        logger.info(f"resume training from local state: {resume}")
        accelerator.load_state(resume)
        return

    logger.info(f"resume training from huggingface state: {resume}")
    repo_id = resume.split("/")[0] + "/" + resume.split("/")[1]
    path_in_repo = "/".join(resume.split("/")[2:])
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

    huggingface_token = getattr(args, "huggingface_token", None)
    list_files = huggingface_util.list_dir(
        repo_id=repo_id,
        subfolder=path_in_repo,
        revision=revision,
        token=huggingface_token,
        repo_type=repo_type,
    )

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

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


def get_epoch_ckpt_name(args: Any, ext: str, epoch_no: int, output_name_append: str = ""):
    output_name = getattr(args, "output_name", None)
    model_name = default_if_none(output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name + output_name_append, epoch_no) + ext


def get_step_ckpt_name(args: Any, ext: str, step_no: int, output_name_append: str = ""):
    output_name = getattr(args, "output_name", None)
    model_name = default_if_none(output_name, DEFAULT_STEP_NAME)
    return STEP_FILE_NAME.format(model_name + output_name_append, step_no) + ext


def get_last_ckpt_name(args: Any, ext: str, output_name_append: str = ""):
    output_name = getattr(args, "output_name", None)
    model_name = default_if_none(output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + output_name_append + ext


def get_remove_epoch_no(args: Any, epoch_no: int):
    save_last_n_epochs = getattr(args, "save_last_n_epochs", None)
    if save_last_n_epochs is None:
        return None

    save_every_n_epochs = getattr(args, "save_every_n_epochs", 1)
    remove_epoch_no = epoch_no - save_every_n_epochs * save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: Any, step_no: int):
    save_last_n_steps = getattr(args, "save_last_n_steps", None)
    if save_last_n_steps is None:
        return None

    # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
    # save_every_n_steps=10, save_last_n_steps=30の場合、50step目には30step分残し、10step目を削除する
    save_every_n_steps = getattr(args, "save_every_n_steps", 1)
    remove_step_no = step_no - save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
        args: Any,
        on_epoch_end: bool,
        accelerator,
        src_path: str,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        save_dtype: torch.dtype,
        epoch: int,
        num_train_epochs: int,
        global_step: int,
        text_encoder,
        unet,
        vae,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True)
        model_util.save_stable_diffusion_checkpoint(
            getattr(args, "v2", False), ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, sai_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        model_util.save_diffusers_checkpoint(
            getattr(args, "v2", False), out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )

    save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        save_stable_diffusion_format,
        use_safetensors,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        diffusers_saver,
    )


def save_sd_model_on_epoch_end_or_stepwise_common(
        args: Any,
        on_epoch_end: bool,
        accelerator,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        epoch: int,
        num_train_epochs: int,
        global_step: int,
        sd_saver,
        diffusers_saver,
):
    output_name = getattr(args, "output_name", None)
    output_dir = getattr(args, "output_dir", ".")
    save_every_n_epochs = getattr(args, "save_every_n_epochs", 1)

    if on_epoch_end:
        epoch_no = epoch + 1
        saving = epoch_no % save_every_n_epochs == 0 and epoch_no < num_train_epochs
        if not saving:
            return

        model_name = default_if_none(output_name, DEFAULT_EPOCH_NAME)
        remove_no = get_remove_epoch_no(args, epoch_no)
    else:
        # 保存するか否かは呼び出し側で判断済み

        model_name = default_if_none(output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch  # 例: 最初のepochの途中で保存したら0になる、SDモデルに保存される
        remove_no = get_remove_step_no(args, global_step)

    os.makedirs(output_dir, exist_ok=True)
    huggingface_repo_id = getattr(args, "huggingface_repo_id", None)

    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(args, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(args, ext, global_step)

        ckpt_file = os.path.join(output_dir, ckpt_name)
        logger.info("")
        logger.info(f"saving checkpoint: {ckpt_file}")
        sd_saver(ckpt_file, epoch_no, global_step)

        if huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_ckpt_name = get_epoch_ckpt_name(args, ext, remove_no)
            else:
                remove_ckpt_name = get_step_ckpt_name(args, ext, remove_no)

            remove_ckpt_file = os.path.join(output_dir, remove_ckpt_name)
            if os.path.exists(remove_ckpt_file):
                logger.info(f"removing old checkpoint: {remove_ckpt_file}")
                os.remove(remove_ckpt_file)

    else:
        if on_epoch_end:
            out_dir = os.path.join(output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no))
        else:
            out_dir = os.path.join(output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step))

        logger.info("")
        logger.info(f"saving model: {out_dir}")
        diffusers_saver(out_dir)

        if huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_out_dir = os.path.join(output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, remove_no))
            else:
                remove_out_dir = os.path.join(output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, remove_no))

            if os.path.exists(remove_out_dir):
                logger.info(f"removing old model: {remove_out_dir}")
                shutil.rmtree(remove_out_dir)

    save_state = getattr(args, "save_state", False)
    if save_state:
        if on_epoch_end:
            save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        else:
            save_and_remove_state_stepwise(args, accelerator, global_step)


def save_and_remove_state_on_epoch_end(args: Any, accelerator, epoch_no):
    output_name = getattr(args, "output_name", None)
    output_dir = getattr(args, "output_dir", ".")
    model_name = default_if_none(output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(output_dir, exist_ok=True)

    state_dir = os.path.join(output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)

    save_state_to_huggingface = getattr(args, "save_state_to_huggingface", False)
    if save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))

    save_last_n_epochs_state = getattr(args, "save_last_n_epochs_state", None)
    save_last_n_epochs = getattr(args, "save_last_n_epochs", None)
    save_every_n_epochs = getattr(args, "save_every_n_epochs", 1)

    last_n_epochs = save_last_n_epochs_state if save_last_n_epochs_state else save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: Any, accelerator, step_no):
    output_name = getattr(args, "output_name", None)
    output_dir = getattr(args, "output_dir", ".")
    model_name = default_if_none(output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(output_dir, exist_ok=True)

    state_dir = os.path.join(output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)

    save_state_to_huggingface = getattr(args, "save_state_to_huggingface", False)
    if save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

    save_last_n_steps_state = getattr(args, "save_last_n_steps_state", None)
    save_last_n_steps = getattr(args, "save_last_n_steps", None)
    save_every_n_steps = getattr(args, "save_every_n_steps", 1)

    last_n_steps = save_last_n_steps_state if save_last_n_steps_state else save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: Any, accelerator):
    output_name = getattr(args, "output_name", None)
    output_dir = getattr(args, "output_dir", ".")
    model_name = default_if_none(output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(output_dir, exist_ok=True)

    state_dir = os.path.join(output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)

    save_state_to_huggingface = getattr(args, "save_state_to_huggingface", False)
    if save_state_to_huggingface:
        logger.info("uploading last state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))


def save_sd_model_on_train_end(
        args: Any,
        src_path: str,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        save_dtype: torch.dtype,
        epoch: int,
        global_step: int,
        text_encoder,
        unet,
        vae,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True)
        model_util.save_stable_diffusion_checkpoint(
            getattr(args, "v2", False), ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, sai_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        model_util.save_diffusers_checkpoint(
            getattr(args, "v2", False), out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )

    save_sd_model_on_train_end_common(
        args, save_stable_diffusion_format, use_safetensors, epoch, global_step, sd_saver, diffusers_saver
    )


def save_sd_model_on_train_end_common(
        args: Any,
        save_stable_diffusion_format: bool,
        use_safetensors: bool,
        epoch: int,
        global_step: int,
        sd_saver,
        diffusers_saver,
):
    output_name = getattr(args, "output_name", None)
    output_dir = getattr(args, "output_dir", ".")
    model_name = default_if_none(output_name, DEFAULT_LAST_OUTPUT_NAME)
    huggingface_repo_id = getattr(args, "huggingface_repo_id", None)

    if save_stable_diffusion_format:
        os.makedirs(output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(output_dir, ckpt_name)

        logger.info(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        sd_saver(ckpt_file, epoch, global_step)

        if huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=True)
    else:
        out_dir = os.path.join(output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"save trained model as Diffusers to {out_dir}")
        diffusers_saver(out_dir)

        if huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name, force_sync_upload=True)
