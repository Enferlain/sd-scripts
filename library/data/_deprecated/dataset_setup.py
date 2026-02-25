from multiprocessing import Value
import logging

from library.config import config_util as config_util
from library.config.config_util import BlueprintGenerator
from library.data._deprecated.dataset_utils import load_arbitrary_dataset, collator_class, debug_dataset



logger = logging.getLogger(__name__)


def prepare_datasets(cfg, strategies):
    """
    Prepare train and validation dataset groups.

    Handles:
    - User config generation from train_data_dir/reg_data_dir
    - Blueprint generation
    - Dataset group creation
    - Collator setup
    - Latent cacheability validation

    Args:
        cfg: Training configuration (SDPeftConfig or SDXLPeftConfig)
        strategies: PEFT strategy instance for validation

    Returns:
        Tuple of (train_dataset_group, val_dataset_group, collator, current_epoch, current_step)
        Returns None for the tuple if debug_dataset mode is active or no data found.
    """
    cache_latents = cfg.data.caching.cache_latents

    # Prepare datasets
    if cfg.data.source.dataset_class is None:
        # Check if we have manually provided subsets via train_data_dir/reg_data_dir
        if (cfg.data.source.train_data_dir is not None or cfg.data.source.reg_data_dir is not None) and len(cfg.data.source.subsets) == 0:
            # Generate subsets config from dirs
            user_config = config_util.generate_user_config_from_dataset(cfg)
            # We need to inject this into cfg.data.source.subsets
            # cfg.data.source.subsets is a List[dict] (or ListConfig)
            # user_config['datasets'][0]['subsets'] is the list we want
            if user_config["datasets"]:
                cfg.data.source.subsets = user_config["datasets"][0]["subsets"]

        blueprint_generator = BlueprintGenerator()
        blueprint = blueprint_generator.generate(cfg)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        # use arbitrary dataset class
        train_dataset_group = load_arbitrary_dataset(cfg.data, cfg.training.max_token_length)
        val_dataset_group = None  # placeholder until validation dataset supported for arbitrary

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if cfg.data.loader.max_workers == 0 else None
    collator = collator_class(current_epoch, current_step, ds_for_collator)

    if cfg.data.preprocessing.debug_dataset:
        train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
        debug_dataset(train_dataset_group)

        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            debug_dataset(val_dataset_group)
        return None  # Signal to caller to exit early

    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
        )
        return None  # Signal to caller to exit early

    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), (
            "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
        )
        if val_dataset_group is not None:
            assert val_dataset_group.is_latent_cacheable(), (
                "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
            )

    strategies.validate_extra_config(cfg, train_dataset_group, val_dataset_group)

    return train_dataset_group, val_dataset_group, collator, current_epoch, current_step
