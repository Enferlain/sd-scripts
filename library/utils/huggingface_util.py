import os
import logging

from typing import Union, BinaryIO
from huggingface_hub import HfApi
from pathlib import Path

from library.utils.common_utils import fire_in_thread, setup_logging
from library.config.dataclasses.output import HuggingFaceConfig

setup_logging()
logger = logging.getLogger(__name__)


def exists_repo(repo_id: str, repo_type: str, revision: str = "main", token: str = None):
    """
    Checks if a HuggingFace repository exists.

    Args:
        repo_id: The ID of the repository (e.g., "username/repo_name").
        repo_type: The type of the repository (e.g., "model", "dataset", "space").
        revision: The revision to check (default is "main").
        token: The HuggingFace API token.

    Returns:
        bool: True if the repository exists, False otherwise.
    """
    api = HfApi(
        token=token,
    )
    try:
        api.repo_info(repo_id=repo_id, revision=revision, repo_type=repo_type)
        return True
    except:
        return False


def upload(
        hf_config: HuggingFaceConfig,
        src: Union[str, Path, bytes, BinaryIO],
        dest_suffix: str = "",
        force_sync_upload: bool = False,
):
    """
    Upload a file or folder to HuggingFace Hub.
    
    Args:
        hf_config: HuggingFaceConfig dataclass with repo settings
        src: Source file/folder path or file object
        dest_suffix: Suffix to append to path_in_repo
        force_sync_upload: Force synchronous upload even if async_upload is True
    """
    repo_id = hf_config.huggingface_repo_id
    repo_type = hf_config.huggingface_repo_type
    token = hf_config.huggingface_token
    path_in_repo = hf_config.huggingface_path_in_repo + dest_suffix if hf_config.huggingface_path_in_repo is not None else None
    private = hf_config.huggingface_repo_visibility is None or hf_config.huggingface_repo_visibility != "public"
    api = HfApi(token=token)
    if not exists_repo(repo_id=repo_id, repo_type=repo_type, token=token):
        try:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
        except Exception as e:
            logger.error("===========================================")
            logger.error(f"failed to create HuggingFace repo / HuggingFaceのリポジトリの作成に失敗しました : {e}")
            logger.error("===========================================")

    is_folder = (type(src) == str and os.path.isdir(src)) or (isinstance(src, Path) and src.is_dir())

    def uploader():
        try:
            if is_folder:
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    folder_path=src,
                    path_in_repo=path_in_repo,
                )
            else:
                api.upload_file(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    path_or_fileobj=src,
                    path_in_repo=path_in_repo,
                )
        except Exception as e:
            logger.error("===========================================")
            logger.error(f"failed to upload to HuggingFace / HuggingFaceへのアップロードに失敗しました : {e}")
            logger.error("===========================================")

    if hf_config.async_upload and not force_sync_upload:
        fire_in_thread(uploader)
    else:
        uploader()


def list_dir(
        repo_id: str,
        subfolder: str,
        repo_type: str,
        revision: str = "main",
        token: str = None,
):
    """
    Lists files in a subdirectory of a HuggingFace repository.

    Args:
        repo_id: The ID of the repository.
        subfolder: The subdirectory to list files from.
        repo_type: The type of the repository.
        revision: The revision to list from (default is "main").
        token: The HuggingFace API token.

    Returns:
        list: A list of RepoFile objects in the specified subdirectory.
    """
    api = HfApi(
        token=token,
    )
    repo_info = api.repo_info(repo_id=repo_id, revision=revision, repo_type=repo_type)
    file_list = [file for file in repo_info.siblings if file.rfilename.startswith(subfolder)]
    return file_list
