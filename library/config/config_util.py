import logging
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from pathlib import Path
from textwrap import dedent, indent
from dataclasses import asdict, dataclass

# We need the FullConfig type hint, but can't import it directly without creating a circular dependency
# if config.py imports this file. Using a string hint is fine.
# from library.config.dataclasses.config import FullConfig

from library.data.data_structures import ControlNetSubset, DreamBoothSubset, FineTuningSubset
from library.data.dataset import DatasetGroup, DreamBoothDataset, FineTuningDataset, ControlNetDataset
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# --- Dataclass Definitions for Blueprint ---
# These dataclasses define the structure of the "blueprint" used to build the datasets.
# They are populated by the BlueprintGenerator from the Hydra config.


@dataclass
class BaseSubsetParams:
    image_dir: Optional[str] = None
    num_repeats: int = 1
    shuffle_caption: bool = False
    caption_separator: str = ","
    keep_tokens: int = 0
    keep_tokens_separator: Optional[str] = None
    secondary_separator: Optional[str] = None
    enable_wildcard: bool = False
    color_aug: bool = False
    flip_aug: bool = False
    face_crop_aug_range: Optional[Tuple[float, float]] = None
    random_crop: bool = False
    random_crop_padding_percent: float = 0.05
    caption_prefix: Optional[str] = None
    caption_suffix: Optional[str] = None
    caption_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0
    caption_tag_dropout_rate: float = 0.0
    token_warmup_min: int = 1
    token_warmup_step: float = 0
    custom_attributes: Optional[Dict[str, Any]] = None
    validation_seed: int = 0
    validation_split: float = 0.0
    resize_interpolation: Optional[str] = None


@dataclass
class DreamBoothSubsetParams(BaseSubsetParams):
    is_reg: bool = False
    class_tokens: Optional[str] = None
    caption_extension: str = ".caption"
    cache_info: bool = False
    alpha_mask: bool = False


@dataclass
class FineTuningSubsetParams(BaseSubsetParams):
    metadata_file: Optional[str] = None
    alpha_mask: bool = False


@dataclass
class ControlNetSubsetParams(BaseSubsetParams):
    conditioning_data_dir: str = None
    caption_extension: str = ".caption"
    cache_info: bool = False


@dataclass
class BaseDatasetParams:
    resolution: Optional[Union[Tuple[int, int], List[int]]] = None  # Accept list from hydra
    network_multiplier: float = 1.0
    debug_dataset: bool = False
    validation_seed: Optional[int] = None
    validation_split: float = 0.0
    resize_interpolation: Optional[str] = None


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
    params: Union[DreamBoothSubsetParams, FineTuningSubsetParams, ControlNetSubsetParams]


@dataclass
class DatasetBlueprint:
    is_dreambooth: bool
    is_controlnet: bool
    params: Union[DreamBoothDatasetParams, FineTuningDatasetParams, ControlNetDatasetParams]
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

    def generate(self, cfg: "FullConfig") -> Blueprint:
        dataset_blueprints = []

        dataset_config = cfg.dataset

        # Determine dataset type from the configuration of its subsets
        is_finetuning_type = any(hasattr(s, "metadata_file") and s.metadata_file for s in dataset_config.subsets)
        is_controlnet_type = any(
            hasattr(s, "conditioning_data_dir") and s.conditioning_data_dir for s in dataset_config.subsets
        )

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

        subset_blueprints = []
        for subset_cfg in dataset_config.subsets:
            params_dict = {}

            # Populate params from the dataset config as a base,
            # using the correct subset parameter class to get all possible keys.
            for key in asdict(subset_params_klass()):
                if hasattr(dataset_config, key):
                    params_dict[key] = getattr(dataset_config, key)

            # Overwrite with subset-specific values
            # Convert subset_cfg to dict to iterate
            if hasattr(subset_cfg, "__dataclass_fields__"):
                subset_cfg_dict = asdict(subset_cfg)
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
            if hasattr(dataset_config, key):
                dataset_params_dict[key] = getattr(dataset_config, key)
            elif hasattr(cfg, "buckets") and hasattr(cfg.buckets, key):
                dataset_params_dict[key] = getattr(cfg.buckets, key)

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
) -> Tuple[DatasetGroup, Optional[DatasetGroup]]:
    datasets: List[Union[DreamBoothDataset, FineTuningDataset, ControlNetDataset]] = []

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

    val_datasets: List[Union[DreamBoothDataset, FineTuningDataset, ControlNetDataset]] = []
    for dataset_blueprint in dataset_group_blueprint.datasets:
        dataset_blueprint.params.validation_split = (
            float(dataset_blueprint.params.validation_split)
            if dataset_blueprint.params.validation_split is not None
            else 0.0
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

    return (DatasetGroup(datasets), DatasetGroup(val_datasets) if val_datasets else None)


def generate_dreambooth_subsets_config_by_subdirs(
    train_data_dir: Optional[str] = None, reg_data_dir: Optional[str] = None
):
    def extract_dreambooth_params(name: str) -> Tuple[int, str]:
        tokens = name.split("_")
        try:
            n_repeats = int(tokens[0])
        except ValueError as e:
            logger.warning(f"ignore directory without repeats / 繰り返し回数のないディレクトリを無視します: {name}")
            return 0, ""
        caption_by_folder = "_".join(tokens[1:])
        return n_repeats, caption_by_folder

    def generate(base_dir: Optional[str], is_reg: bool):
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

def generate_user_config_from_args(args) -> dict:
    """
    Generate user_config from args.
    This is for backward compatibility.
    """
    # Assuming args is config.dataset (or compatible object)
    if args.dataset_class is None:
        user_config = {
            "datasets": [
                {
                    "subsets": generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)
                }
            ]
        }
    else:
        # For arbitrary dataset, we don't need subsets config in the same way,
        # but we need to structure it if needed.
        # However, BlueprintGenerator logic for arbitrary dataset is handled differently (by not calling it or handling it upstream).
        # If dataset_class is present, BlueprintGenerator might not be used or used differently.
        user_config = {"datasets": []} # Empty or handled otherwise

    return user_config
