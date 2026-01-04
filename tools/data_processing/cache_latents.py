# latentsのdiskへの事前キャッシュを行う / cache latents to disk

import argparse
import torch
import logging

# TODO add back missing pipes
# from library import flux_train_utils, flux_utils, strategy_flux


from library.data._deprecated.dataset import load_arbitrary_dataset
from library.strategies import strategy_sdxl, strategy_sd, strategy_base
from library.training.trainer_utils import prepare_accelerator
from library.utils import config_util
from library.utils.common_utils import setup_logging, add_logging_arguments
from library.models.sd_model_prep import load_target_model
from library.models.sdxl_model_prep import load_target_model as load_target_model_sdxl

from library.utils.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)

from library.config.arguments import (
    prepare_dataset_args,
    enable_high_vram,
    add_sd_models_arguments,
    add_training_arguments,
    add_dataset_arguments,
    add_masked_loss_arguments,
    add_dit_training_arguments,
    read_config_from_file,
)
from library.utils.torch_utils import set_seed_from_config, prepare_dtype

setup_logging()
logger = logging.getLogger(__name__)


def set_tokenize_strategy(is_sd: bool, is_sdxl: bool, is_flux: bool, args: argparse.Namespace) -> None:
    if is_flux:
        _, is_schnell, _ = flux_utils.check_flux_state_dict_diffusers_schnell(args.pretrained_model_name_or_path)
    else:
        is_schnell = False

    if is_sd:
        tokenize_strategy = strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)
    elif is_sdxl:
        tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)
    else:
        if args.t5xxl_max_token_length is None:
            if is_schnell:
                t5xxl_max_token_length = 256
            else:
                t5xxl_max_token_length = 512
        else:
            t5xxl_max_token_length = args.t5xxl_max_token_length

        logger.info(f"t5xxl_max_token_length: {t5xxl_max_token_length}")
        tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length, args.tokenizer_cache_dir)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)


def cache_to_disk(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    prepare_dataset_args(args, True)
    enable_high_vram(args)

    # assert args.cache_latents_to_disk, "cache_latents_to_disk must be True / cache_latents_to_diskはTrueである必要があります"
    args.cache_latents = True
    args.cache_latents_to_disk = True

    use_dreambooth_method = args.in_json is None

    set_seed_from_config(args)

    is_sd = not args.sdxl and not args.flux
    is_sdxl = args.sdxl
    is_flux = args.flux

    set_tokenize_strategy(is_sd, is_sdxl, is_flux, args)

    if is_sd or is_sdxl:
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(is_sd, True, args.vae_batch_size, args.skip_cache_check)
    else:
        latents_caching_strategy = strategy_flux.FluxLatentsCachingStrategy(True, args.vae_batch_size, args.skip_cache_check)
    strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    use_user_config = args.dataset_config is not None
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if use_user_config:
            logger.info(f"Loading dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)}
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        # use arbitrary dataset class
        train_dataset_group = load_arbitrary_dataset(args)
        val_dataset_group = None

    # acceleratorを準備する
    logger.info("prepare accelerator")
    args.deepspeed = False
    accelerator = prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, _ = prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    logger.info("load model")
    if is_sd:
        _, vae, _, _ = load_target_model(
            args,
            weight_dtype,
            accelerator,
        )
    elif is_sdxl:
        (_, _, _, vae, _, _, _) = load_target_model_sdxl(args, accelerator, "sdxl", weight_dtype)
    else:
        vae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)

    if is_sd or is_sdxl:
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()

    # cache latents with dataset
    # TODO use DataLoader to speed up
    train_dataset_group.new_cache_latents(vae, accelerator)

    accelerator.wait_for_everyone()
    accelerator.print("Finished caching latents to disk.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    add_sd_models_arguments(parser)

    add_training_arguments(parser, True)
    add_dataset_arguments(parser, True, True, True)
    add_masked_loss_arguments(parser)
    config_util.add_config_arguments(parser)
    add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)

    parser.add_argument("--sdxl", action="store_true", help="Use SDXL model / SDXLモデルを使用する")
    parser.add_argument("--flux", action="store_true", help="Use FLUX model / FLUXモデルを使用する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="[Deprecated] This option does not work. Existing .npz files are always checked. Use `--skip_cache_check` to skip the check."
        " / [非推奨] このオプションは機能しません。既存の .npz は常に検証されます。`--skip_cache_check` で検証をスキップできます。",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    cache_to_disk(args)
