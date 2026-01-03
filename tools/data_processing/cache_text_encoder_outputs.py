# text encoder出力のdiskへの事前キャッシュを行う / cache text encoder outputs to disk in advance

import argparse
import torch
import logging

from library.constants import MODEL_VERSION_SDXL_BASE_V1_0
from library.training.trainer_utils import prepare_accelerator
from tools.data_processing.cache_latents import set_tokenize_strategy
from library.strategies import strategy_sdxl, strategy_base
from library.utils.torch_utils import set_seed_from_config, prepare_dtype, str_to_dtype
from library.utils import config_util
from library.utils.config_util import ConfigSanitizer, BlueprintGenerator
from library.utils.common_utils import setup_logging, add_logging_arguments
from library.data.dataset import load_arbitrary_dataset
from library.models.sdxl_model_prep import load_target_model as load_target_model_sdxl

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

# TODO add back missing pipes
# from library import (
#     flux_train_utils,
#     flux_utils,
#     strategy_flux,
# )
# import library.utils.sai_model_spec as sai_model_spec

setup_logging()
logger = logging.getLogger(__name__)


def cache_to_disk(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    prepare_dataset_args(args, True)
    enable_high_vram(args)

    args.cache_text_encoder_outputs = True
    args.cache_text_encoder_outputs_to_disk = True

    use_dreambooth_method = args.in_json is None

    set_seed_from_config(args)

    is_sd = not args.sdxl and not args.flux
    is_sdxl = args.sdxl
    is_flux = args.flux

    assert is_sdxl or is_flux, (
        "Cache text encoder outputs to disk is only supported for SDXL and FLUX models / テキストエンコーダ出力のディスクキャッシュはSDXLまたはFLUXでのみ有効です"
    )
    assert is_sdxl or args.weighted_captions is None, (
        "Weighted captions are only supported for SDXL models / 重み付きキャプションはSDXLモデルでのみ有効です"
    )

    set_tokenize_strategy(is_sd, is_sdxl, is_flux, args)

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
    t5xxl_dtype = str_to_dtype(args.t5xxl_dtype, weight_dtype)

    # モデルを読み込む
    logger.info("load model")
    if is_sdxl:
        _, text_encoder1, text_encoder2, _, _, _, _ = load_target_model_sdxl(args, accelerator, MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)
        text_encoder1.to(accelerator.device, weight_dtype)
        text_encoder2.to(accelerator.device, weight_dtype)
        text_encoders = [text_encoder1, text_encoder2]
    else:
        clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, accelerator.device, disable_mmap=args.disable_mmap_load_safetensors)

        t5xxl = flux_utils.load_t5xxl(args.t5xxl, None, accelerator.device, disable_mmap=args.disable_mmap_load_safetensors)

        if t5xxl.dtype == torch.float8_e4m3fnuz or t5xxl.dtype == torch.float8_e5m2 or t5xxl.dtype == torch.float8_e5m2fnuz:
            raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
        elif t5xxl.dtype == torch.float8_e4m3fn:
            logger.info("Loaded fp8 T5XXL model")

        if t5xxl_dtype != t5xxl_dtype:
            if t5xxl.dtype == torch.float8_e4m3fn and t5xxl_dtype.itemsize() >= 2:
                logger.warning(
                    "The loaded model is fp8, but the specified T5XXL dtype is larger than fp8.  This may cause a performance drop."
                    " / ロードされたモデルはfp8ですが、指定されたT5XXLのdtypeがfp8より高精度です。精度低下が発生する可能性があります。"
                )
            logger.info(f"Casting T5XXL model to {t5xxl_dtype}")
            t5xxl.to(t5xxl_dtype)

        text_encoders = [clip_l, t5xxl]

    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    # build text encoder outputs caching strategy
    if is_sdxl:
        text_encoder_outputs_caching_strategy = strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, None, args.skip_cache_check, is_weighted=args.weighted_captions
        )
    else:
        text_encoder_outputs_caching_strategy = strategy_flux.FluxTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            is_partial=False,
            apply_t5_attn_mask=args.apply_t5_attn_mask,
        )
    strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)

    # build text encoding strategy
    if is_sdxl:
        text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()
    else:
        text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # cache text encoder outputs
    train_dataset_group.new_cache_text_encoder_outputs(text_encoders, accelerator)

    accelerator.wait_for_everyone()
    accelerator.print("Finished caching text encoder outputs to disk.")


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
        "--t5xxl_dtype",
        type=str,
        default=None,
        help="T5XXL model dtype, default: None (use mixed precision dtype) / T5XXLモデルのdtype, デフォルト: None (mixed precisionのdtypeを使用)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="[Deprecated] This option does not work. Existing .npz files are always checked. Use `--skip_cache_check` to skip the check."
        " / [非推奨] このオプションは機能しません。既存の .npz は常に検証されます。`--skip_cache_check` で検証をスキップできます。",
    )
    parser.add_argument(
        "--weighted_captions",
        action="store_true",
        default=False,
        help="Enable weighted captions in the standard style (token:1.3). No commas inside parens, or shuffle/dropout may break the decoder. / 「[token]」、「(token)」「(token:1.3)」のような重み付きキャプションを有効にする。カンマを括弧内に入れるとシャッフルやdropoutで重みづけがおかしくなるので注意",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    cache_to_disk(args)
