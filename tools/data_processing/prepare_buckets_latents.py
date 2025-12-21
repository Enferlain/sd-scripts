import argparse
import os
import json
import numpy as np
import torch
import logging

from pathlib import Path
from typing import List
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

import library.models.model_util as model_util

from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex, get_preferred_device
from library.data.caching import cache_batch_latents, is_disk_cached_latents_is_expected
from library.data.data_structures import BucketManager, ImageInfo
from library.data.dataset import ImageLoadingDataset
from library.data.image_utils import glob_images_pathlib

init_ipex()

setup_logging()
logger = logging.getLogger(__name__)

DEVICE = get_preferred_device()


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def get_npz_filename(data_dir, image_key, is_full_path, recursive):
    if is_full_path:
        base_name = os.path.splitext(os.path.basename(image_key))[0]
        relative_path = os.path.relpath(os.path.dirname(image_key), data_dir)
    else:
        base_name = image_key
        relative_path = ""

    if recursive and relative_path:
        return os.path.join(data_dir, relative_path, base_name) + ".npz"
    else:
        return os.path.join(data_dir, base_name) + ".npz"


def main(args):
    # assert args.bucket_reso_steps % 8 == 0, f"bucket_reso_steps must be divisible by 8"
    if args.bucket_reso_steps % 8 > 0:
        logger.warning(f"resolution of buckets in training time is a multiple of 8")
    if args.bucket_reso_steps % 32 > 0:
        logger.warning(
            f"WARNING: bucket_reso_steps is not divisible by 32. It is not working with SDXL"
        )

    train_data_dir_path = Path(args.train_data_dir)
    image_paths: List[str] = [str(p) for p in glob_images_pathlib(train_data_dir_path, args.recursive)]
    logger.info(f"found {len(image_paths)} images.")

    if os.path.exists(args.in_json):
        logger.info(f"loading existing metadata: {args.in_json}")
        with open(args.in_json, "rt", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        logger.error(f"no metadata: {args.in_json}")
        return

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = model_util.load_vae(args.model_name_or_path, weight_dtype)
    vae.eval()
    vae.to(DEVICE, dtype=weight_dtype)

    # Calculate bucket size
    max_reso = tuple([int(t) for t in args.max_resolution.split(",")])
    assert (
        len(max_reso) == 2
    ), f"illegal resolution (not 'width,height'): {args.max_resolution}"

    bucket_manager = BucketManager(
        args.bucket_no_upscale, max_reso, args.min_bucket_reso, args.max_bucket_reso, args.bucket_reso_steps
    )
    if not args.bucket_no_upscale:
        bucket_manager.make_buckets()
    else:
        logger.warning(
            "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically"
        )

    # Assign images to appropriate buckets one by one and calculate latents
    img_ar_errors = []

    def process_batch(is_last):
        for bucket in bucket_manager.buckets:
            if (is_last and len(bucket) > 0) or len(bucket) >= args.batch_size:
                cache_batch_latents(vae, True, bucket, args.flip_aug, args.alpha_mask, False, random_crop_padding_percent=0.05)
                bucket.clear()

    # Option to use DataLoader for faster loading
    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    bucket_counts = {}
    for data_entry in tqdm(data, smoothing=0.0):
        if data_entry[0] is None:
            continue

        img_tensor, image_path = data_entry[0]
        if img_tensor is not None:
            image = transforms.functional.to_pil_image(img_tensor)
        else:
            try:
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception as e:
                logger.error(f"Could not load image path: {image_path}, error: {e}")
                continue

        image_key = image_path if args.full_path else os.path.splitext(os.path.basename(image_path))[0]
        if image_key not in metadata:
            metadata[image_key] = {}

        # Actually, moving the following part to DataSet would speed up, but it's a lot of work

        reso, resized_size, ar_error = bucket_manager.select_bucket(image.width, image.height)
        img_ar_errors.append(abs(ar_error))
        bucket_counts[reso] = bucket_counts.get(reso, 0) + 1

        # Resolution recorded in metadata is in latent units, so truncate in units of 8
        metadata[image_key]["train_resolution"] = (reso[0] - reso[0] % 8, reso[1] - reso[1] % 8)

        if not args.bucket_no_upscale:
            # When upscaling is not performed, confirm that the resized size matches the bucket size in either width or height
            assert (
                resized_size[0] == reso[0] or resized_size[1] == reso[1]
            ), f"internal error, resized size not match: {reso}, {resized_size}, {image.width}, {image.height}"
            assert (
                resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
            ), f"internal error, resized size too small: {reso}, {resized_size}, {image.width}, {image.height}"

        assert (
            resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
        ), f"internal error resized size is small: {resized_size}, {reso}"

        # If the file already exists, check shape etc. and skip if same
        npz_file_name = get_npz_filename(args.train_data_dir, image_key, args.full_path, args.recursive)
        if args.skip_existing:
            if is_disk_cached_latents_is_expected(reso, npz_file_name, args.flip_aug):
                continue

        # Add to batch
        image_info = ImageInfo(image_key, 1, "", False, image_path)
        image_info.latents_npz = npz_file_name
        image_info.bucket_reso = reso
        image_info.resized_size = resized_size
        image_info.image = image
        bucket_manager.add_image(reso, image_info)

        # Determine whether to infer batch and infer
        process_batch(False)

        # Process the rest
    process_batch(True)

    bucket_manager.sort()
    for i, reso in enumerate(bucket_manager.resos):
        count = bucket_counts.get(reso, 0)
        if count > 0:
            logger.info(f"bucket {i} {reso}: {count}")
    img_ar_errors = np.array(img_ar_errors)
    logger.info(f"mean ar error: {np.mean(img_ar_errors)}")

    # Write metadata and finish
    logger.info(f"writing metadata: {args.out_json}")
    with open(args.out_json, "wt", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images")
    parser.add_argument("in_json", type=str, help="metadata file to input")
    parser.add_argument("out_json", type=str, help="metadata file to output")
    parser.add_argument("model_name_or_path", type=str, help="model name or path to encode latents")
    parser.add_argument(
        "--v2", action="store_true", help="not used (for backward compatibility)"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster)",
    )
    parser.add_argument(
        "--max_resolution",
        type=str,
        default="512,512",
        help="max resolution in fine tuning (width,height)",
    )
    parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets")
    parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets")
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="make bucket for each image without upscaling",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision",
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="use full path as image-key in metadata (supports multiple directories)",
    )
    parser.add_argument(
        "--flip_aug",
        action="store_true",
        help="flip augmentation, save latents for flipped images",
    )
    parser.add_argument(
        "--alpha_mask",
        type=str,
        default="",
        help="save alpha mask for images for loss calculation",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip images if npz already exists (both normal and flipped exists if flip_aug is enabled)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively look for training tags in all child folders of train_data_dir",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
