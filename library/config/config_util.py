"""
DEPRECATED: This module is used by legacy training scripts (sdxl_finetune.py, etc.)
New scripts should use library/data/ modules directly (scanners.py, manifest.py, etc.)
"""
import logging
import random

from typing import Any, Protocol, runtime_checkable
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent, indent
from dataclasses import asdict, dataclass

from library.config.dataclasses.data import DataConfig
from library.data._deprecated.controlnet_dataset import ControlNetDataset
from library.data._deprecated.data_structures import ControlNetSubset, DreamBoothSubset, FineTuningSubset
from library.data._deprecated.dataset_group import DatasetGroup
from library.data._deprecated.dreambooth_dataset import DreamBoothDataset
from library.data._deprecated.finetuning_dataset import FineTuningDataset
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@runtime_checkable
class RootConfig(Protocol):
    """Protocol defining the expected structure for any training config passed to BlueprintGenerator.

    All script-specific root configs (SDFineTuneConfig, SDPeftConfig, etc.)
    should satisfy this protocol.
    """

    data: DataConfig


# --- Dataclass Definitions for Blueprint ---
# These dataclasses define the structure of the "blueprint" used to build the datasets.
# They are populated by the BlueprintGenerator from the Hydra config.


@dataclass
class BaseSubsetParams:
    image_dir: str | None = None
    num_repeats: int = 1
    shuffle_caption: bool = False
    caption_separator: str = ","
    keep_tokens: int = 0
    keep_tokens_separator: str | None = None
    secondary_separator: str | None = None
    enable_wildcard: bool = False
    color_aug: bool = False
    flip_aug: bool = False
    face_crop_aug_range: tuple[float, float] | None = None
    random_crop: bool = False
    random_crop_padding_percent: float = 0.05
    caption_prefix: str | None = None
    caption_suffix: str | None = None
    caption_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0
    caption_tag_dropout_rate: float = 0.0
    token_warmup_min: int = 1
    token_warmup_step: float = 0
    custom_attributes: dict[str, Any] | None = None
    validation_seed: int = 0
    validation_split: float = 0.0
    resize_interpolation: str | None = None


@dataclass
class DreamBoothSubsetParams(BaseSubsetParams):
    is_reg: bool = False
    class_tokens: str | None = None
    caption_extension: str = ".caption"
    cache_info: bool = False
    alpha_mask: bool = False


@dataclass
class FineTuningSubsetParams(BaseSubsetParams):
    metadata_file: str | None = None
    alpha_mask: bool = False


@dataclass
class ControlNetSubsetParams(BaseSubsetParams):
    conditioning_data_dir: str | None = None
    caption_extension: str = ".caption"
    cache_info: bool = False


@dataclass
class BaseDatasetParams:
    resolution: tuple[int, int] | list[int] | None = None  # Accept list from hydra
    adapter_multiplier: float = 1.0
    debug_dataset: bool = False
    validation_seed: int | None = None
    validation_split: float = 0.0
    resize_interpolation: str | None = None


@dataclass
class DreamBoothDatasetParams(BaseDatasetParams):
    batch_size: int = 1
    enable_bucket: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = False
    prior_loss_weight: float = 1.0


@dataclass
class FineTuningDatasetParams(BaseDatasetParams):
    batch_size: int = 1
    enable_bucket: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = False


@dataclass
class ControlNetDatasetParams(BaseDatasetParams):
    batch_size: int = 1
    enable_bucket: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = False


@dataclass
class SubsetBlueprint:
    params: DreamBoothSubsetParams | FineTuningSubsetParams | ControlNetSubsetParams


@dataclass
class DatasetBlueprint:
    is_dreambooth: bool
    is_controlnet: bool
    params: DreamBoothDatasetParams | FineTuningDatasetParams | ControlNetDatasetParams
    subsets: Sequence[SubsetBlueprint]


@dataclass
class DatasetGroupBlueprint:
    datasets: Sequence[DatasetBlueprint]


@dataclass
class Blueprint:
    dataset_group: DatasetGroupBlueprint


class BlueprintGenerator:
    def __init__(self):
        pass

    def generate(self, cfg: "RootConfig") -> Blueprint:
        dataset_blueprints = []

        # Access nested data config
        data_config = cfg.data

        # Determine dataset type from the configuration of its subsets
        is_finetuning_type = any(hasattr(s, "metadata_file") and s.metadata_file for s in data_config.source.subsets)
        is_controlnet_type = any(hasattr(s, "conditioning_data_dir") and s.conditioning_data_dir for s in data_config.source.subsets)

        if is_controlnet_type:
            is_dreambooth = False
            is_controlnet = True
            subset_params_klass = ControlNetSubsetParams
            dataset_params_klass = ControlNetDatasetParams
        elif is_finetuning_type:
            is_dreambooth = False
            is_controlnet = False
            subset_params_klass = FineTuningSubsetParams
            dataset_params_klass = FineTuningDatasetParams
        else:
            is_dreambooth = True
            is_controlnet = False
            subset_params_klass = DreamBoothSubsetParams
            dataset_params_klass = DreamBoothDatasetParams

        # Sub-configs to search for field values
        SUB_CONFIGS = ["preprocessing", "caption", "bucketing", "caching", "source"]

        subset_blueprints = []
        for subset_cfg in data_config.source.subsets:
            params_dict = {}

            # Populate params from the data config sub-configs as defaults,
            # using the correct subset parameter class to get all possible keys.
            for key in asdict(subset_params_klass()):
                # Search through sub-configs to find where the field lives
                for config_name in SUB_CONFIGS:
                    sub_config = getattr(data_config, config_name, None)
                    if sub_config and hasattr(sub_config, key):
                        params_dict[key] = getattr(sub_config, key)
                        break  # Found it, stop searching

            # Overwrite with subset-specific values
            # Convert subset_cfg to dict to iterate
            if hasattr(subset_cfg, "__dataclass_fields__"):
                subset_cfg_dict = asdict(subset_cfg)  # type: ignore[type-var]
            elif isinstance(subset_cfg, dict):
                subset_cfg_dict = subset_cfg
            else:
                # If it's a DictConfig or similar
                subset_cfg_dict = dict(subset_cfg)

            for key, value in subset_cfg_dict.items():
                if value is not None:
                    params_dict[key] = value

            params = subset_params_klass(**params_dict)
            subset_blueprints.append(SubsetBlueprint(params=params))

        # Create dataset-level parameters
        dataset_params_dict = {}
        for key in asdict(dataset_params_klass()):
            # Search through sub-configs to find where the field lives
            for config_name in SUB_CONFIGS:
                sub_config = getattr(data_config, config_name, None)
                if sub_config and hasattr(sub_config, key):
                    value = getattr(sub_config, key)
                    if value is not None:
                        dataset_params_dict[key] = value
                    break  # Found it, stop searching

        # Convert resolution list to tuple if necessary
        resolution = dataset_params_dict.get("resolution")
        if isinstance(resolution, list):
            dataset_params_dict["resolution"] = tuple(resolution)

        params = dataset_params_klass(**dataset_params_dict)

        dataset_blueprints.append(DatasetBlueprint(is_dreambooth, is_controlnet, params, subset_blueprints))

        dataset_group_blueprint = DatasetGroupBlueprint(dataset_blueprints)
        return Blueprint(dataset_group_blueprint)


def generate_dataset_group_by_blueprint(
    dataset_group_blueprint: DatasetGroupBlueprint,
) -> tuple[DatasetGroup, DatasetGroup | None]:
    datasets: list[DreamBoothDataset | FineTuningDataset | ControlNetDataset] = []

    for dataset_blueprint in dataset_group_blueprint.datasets:
        extra_dataset_params = {}

        if dataset_blueprint.is_controlnet:
            subset_klass = ControlNetSubset
            dataset_klass = ControlNetDataset
        elif dataset_blueprint.is_dreambooth:
            subset_klass = DreamBoothSubset
            dataset_klass = DreamBoothDataset
            extra_dataset_params = {"is_training_dataset": True}
        else:
            subset_klass = FineTuningSubset
            dataset_klass = FineTuningDataset

        subsets = [subset_klass(**asdict(subset_blueprint.params)) for subset_blueprint in dataset_blueprint.subsets]
        dataset = dataset_klass(subsets=subsets, **asdict(dataset_blueprint.params), **extra_dataset_params)
        datasets.append(dataset)

    val_datasets: list[DreamBoothDataset | FineTuningDataset | ControlNetDataset] = []
    for dataset_blueprint in dataset_group_blueprint.datasets:
        dataset_blueprint.params.validation_split = (
            float(dataset_blueprint.params.validation_split) if dataset_blueprint.params.validation_split is not None else 0.0
        )

        if not (0.0 <= dataset_blueprint.params.validation_split <= 1.0):
            logging.warning(
                f"Dataset param `validation_split` ({dataset_blueprint.params.validation_split}) is not a valid number between 0.0 and 1.0, skipping validation split..."
            )
            continue

        if dataset_blueprint.params.validation_split == 0.0:
            continue

        extra_dataset_params = {}
        if dataset_blueprint.is_controlnet:
            subset_klass = ControlNetSubset
            dataset_klass = ControlNetDataset
        elif dataset_blueprint.is_dreambooth:
            subset_klass = DreamBoothSubset
            dataset_klass = DreamBoothDataset
            extra_dataset_params = {"is_training_dataset": False}
        else:
            subset_klass = FineTuningSubset
            dataset_klass = FineTuningDataset

        subsets = [subset_klass(**asdict(subset_blueprint.params)) for subset_blueprint in dataset_blueprint.subsets]
        dataset = dataset_klass(subsets=subsets, **asdict(dataset_blueprint.params), **extra_dataset_params)
        val_datasets.append(dataset)

    def print_info(_datasets, dataset_type: str):
        info = ""
        for i, dataset in enumerate(_datasets):
            is_dreambooth = isinstance(dataset, DreamBoothDataset)
            is_controlnet = isinstance(dataset, ControlNetDataset)
            info += dedent(
                f"""\
                [{dataset_type} {i}]
                  batch_size: {dataset.batch_size}
                  resolution: {(dataset.width, dataset.height)}
                  resize_interpolation: {dataset.resize_interpolation}
                  enable_bucket: {dataset.enable_bucket}
            """
            )

            if dataset.enable_bucket:
                info += indent(
                    dedent(
                        f"""\
                  min_bucket_reso: {dataset.min_bucket_reso}
                  max_bucket_reso: {dataset.max_bucket_reso}
                  bucket_reso_steps: {dataset.bucket_reso_steps}
                  bucket_no_upscale: {dataset.bucket_no_upscale}
                \n"""
                    ),
                    "  ",
                )
            else:
                info += "\n"

            for j, subset in enumerate(dataset.subsets):
                info += indent(
                    dedent(
                        f"""\
                  [Subset {j} of {dataset_type} {i}]
                    image_dir: "{subset.image_dir}"
                    image_count: {subset.img_count}
                    num_repeats: {subset.num_repeats}
                    shuffle_caption: {subset.shuffle_caption}
                    keep_tokens: {subset.keep_tokens}
                    caption_dropout_rate: {subset.caption_dropout_rate}
                    caption_dropout_every_n_epochs: {subset.caption_dropout_every_n_epochs}
                    caption_tag_dropout_rate: {subset.caption_tag_dropout_rate}
                    caption_prefix: {subset.caption_prefix}
                    caption_suffix: {subset.caption_suffix}
                    color_aug: {subset.color_aug}
                    flip_aug: {subset.flip_aug}
                    face_crop_aug_range: {subset.face_crop_aug_range}
                    random_crop: {subset.random_crop}
                    random_crop_padding_percent: {float(getattr(subset, "random_crop_padding_percent", 0.05))}
                    token_warmup_min: {subset.token_warmup_min},
                    token_warmup_step: {subset.token_warmup_step},
                    alpha_mask: {subset.alpha_mask}
                    resize_interpolation: {subset.resize_interpolation}
                    custom_attributes: {subset.custom_attributes}
                """
                    ),
                    "  ",
                )

                if is_dreambooth:
                    info += indent(
                        dedent(
                            f"""\
                        is_reg: {subset.is_reg}
                        class_tokens: {subset.class_tokens}
                        caption_extension: {subset.caption_extension}
                    \n"""
                        ),
                        "    ",
                    )
                elif not is_controlnet:
                    info += indent(
                        dedent(
                            f"""\
                        metadata_file: {subset.metadata_file}
                    \n"""
                        ),
                        "    ",
                    )

        logger.info(info)

    print_info(datasets, "Dataset")

    if len(val_datasets) > 0:
        print_info(val_datasets, "Validation Dataset")

    # make buckets first because it determines the length of dataset
    # and set the same seed for all datasets
    seed = random.randint(0, 2**31)  # actual seed is seed + epoch_no

    for i, dataset in enumerate(datasets):
        logger.info(f"[Prepare dataset {i}]")
        dataset.make_buckets()
        dataset.set_seed(seed)

    for i, dataset in enumerate(val_datasets):
        logger.info(f"[Prepare validation dataset {i}]")
        dataset.make_buckets()
        dataset.set_seed(seed)

    return (DatasetGroup(datasets), DatasetGroup(val_datasets) if val_datasets else None)  # type: ignore[arg-type]


def generate_dreambooth_subsets_config_by_subdirs(train_data_dir: str | None = None, reg_data_dir: str | None = None):
    def extract_dreambooth_params(name: str) -> tuple[int, str]:
        tokens = name.split("_")
        try:
            n_repeats = int(tokens[0])
        except ValueError:
            logger.warning(f"ignore directory without repeats: {name}")
            return 0, ""
        caption_by_folder = "_".join(tokens[1:])
        return n_repeats, caption_by_folder

    def generate(base_dir: str | None, is_reg: bool):
        if base_dir is None:
            return []

        base_dir: Path = Path(base_dir)
        if not base_dir.is_dir():
            return []

        subsets_config = []
        for subdir in base_dir.iterdir():
            if not subdir.is_dir():
                continue

            num_repeats, class_tokens = extract_dreambooth_params(subdir.name)
            if num_repeats < 1:
                continue

            subset_config = {
                "image_dir": str(subdir),
                "num_repeats": num_repeats,
                "is_reg": is_reg,
                "class_tokens": class_tokens,
            }
            subsets_config.append(subset_config)

        return subsets_config

    subsets_config = []
    subsets_config += generate(train_data_dir, False)
    subsets_config += generate(reg_data_dir, True)

    return subsets_config


def generate_user_config_from_dataset(cfg) -> dict:
    """
    Generate user_config from root config for DreamBooth subdirectory parsing.
    """
    source_config = cfg.data.source
    if source_config.dataset_class is None:
        user_config = {
            "datasets": [
                {"subsets": generate_dreambooth_subsets_config_by_subdirs(source_config.train_data_dir, source_config.reg_data_dir)}
            ]
        }
    else:
        # For arbitrary dataset, we don't need subsets config in the same way.
        # BlueprintGenerator logic for arbitrary dataset is handled differently (by not calling it or handling it upstream).
        # If dataset_class is present, BlueprintGenerator might not be used or used differently.
        user_config = {"datasets": []}  # Empty or handled otherwise

    return user_config
