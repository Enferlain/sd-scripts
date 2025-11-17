import math
import random
import cv2
import torch
import numpy as np

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from PIL import Image

from library.models import model_util


class ImageInfo:
    def __init__(self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: Optional[torch.Tensor] = None
        self.latents_flipped: Optional[torch.Tensor] = None
        self.latents_npz: Optional[str] = None  # set in cache_latents
        self.latents_original_size: Optional[Tuple[int, int]] = None  # original image size, not latents size
        self.latents_crop_ltrb: Optional[Tuple[int, int]] = (
            None  # crop left top right bottom in original pixel size, not latents size
        )
        self.cond_img_path: Optional[str] = None
        self.image: Optional[Image.Image] = None  # optional, original PIL Image
        self.text_encoder_outputs_npz: Optional[str] = None  # set in cache_text_encoder_outputs

        # new
        self.text_encoder_outputs: Optional[List[torch.Tensor]] = None
        # old
        self.text_encoder_outputs1: Optional[torch.Tensor] = None
        self.text_encoder_outputs2: Optional[torch.Tensor] = None
        self.text_encoder_pool2: Optional[torch.Tensor] = None

        self.alpha_mask: Optional[torch.Tensor] = None  # alpha mask can be flipped in runtime
        self.resize_interpolation: Optional[str] = None


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "the max_size should be larger than the width of max_reso"
                assert max_size >= max_reso[1], "the max_size should be larger than the height of max_reso"
            if min_size is not None:
                assert max_size >= min_size, "the max_size should be larger than the min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []  # 前処理時は (image_key, image, original size, crop left/top)、学習時は image_key

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        resos = model_util.make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # logger.info(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # 拡大および縮小を行う
            # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            # logger.info(f"use predef, {image_width}, {image_height}, {reso}, {resized_size}")
        else:
            # 縮小のみを行う
            if image_width * image_height > self.max_area:
                # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
                # 元のbucketingと同じロジック
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # logger.info(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # logger.info(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
                # logger.info(resized_size)
            else:
                resized_size = (image_width, image_height)  # リサイズは不要

            # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # logger.info(f"use arbitrary {image_width}, {image_height}, {resized_size}, {bucket_width}, {bucket_height}")

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Stability AIの前処理に合わせてcrop left/topを計算する。crop rightはflipのaugmentationのために求める
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucketのほうが横長→縦を合わせる
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


class AugHelper:
    # albumentationsへの依存をなくしたがとりあえず同じinterfaceを持たせる

    def __init__(self):
        pass

    def color_aug(self, image: np.ndarray):
        # self.color_aug_method = albu.OneOf(
        #     [
        #         albu.HueSaturationValue(8, 0, 0, p=0.5),
        #         albu.RandomGamma((95, 105), p=0.5),
        #     ],
        #     p=0.33,
        # )
        hue_shift_limit = 8

        # remove dependency to albumentations
        if random.random() <= 0.33:
            if random.random() > 0.5:
                # hue shift
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
                if hue_shift < 0:
                    hue_shift = 180 + hue_shift
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            else:
                # random gamma
                gamma = random.uniform(0.95, 1.05)
                image = np.clip(image ** gamma, 0, 255).astype(np.uint8)

        return {"image": image}

    def get_augmentor(self, use_color_aug: bool):  # -> Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]]:
        return self.color_aug if use_color_aug else None


class BaseSubset:
    def __init__(
            self,
            image_dir: Optional[str],
            alpha_mask: Optional[bool],
            num_repeats: int,
            shuffle_caption: bool,
            caption_separator: str,
            keep_tokens: int,
            keep_tokens_separator: str,
            secondary_separator: Optional[str],
            enable_wildcard: bool,
            color_aug: bool,
            flip_aug: bool,
            face_crop_aug_range: Optional[Tuple[float, float]],
            random_crop: bool,
            random_crop_padding_percent: float,
            caption_dropout_rate: float,
            caption_dropout_every_n_epochs: int,
            caption_tag_dropout_rate: float,
            caption_prefix: Optional[str],
            caption_suffix: Optional[str],
            token_warmup_min: int,
            token_warmup_step: Union[float, int],
            custom_attributes: Optional[Dict[str, Any]] = None,
            validation_seed: Optional[int] = None,
            validation_split: Optional[float] = 0.0,
            resize_interpolation: Optional[str] = None,
    ) -> None:
        self.image_dir = image_dir
        self.alpha_mask = alpha_mask if alpha_mask is not None else False
        self.num_repeats = num_repeats
        self.shuffle_caption = shuffle_caption
        self.caption_separator = caption_separator
        self.keep_tokens = keep_tokens
        self.keep_tokens_separator = keep_tokens_separator
        self.secondary_separator = secondary_separator
        self.enable_wildcard = enable_wildcard
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.random_crop_padding_percent = random_crop_padding_percent
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.caption_prefix = caption_prefix
        self.caption_suffix = caption_suffix

        self.token_warmup_min = token_warmup_min  # step=0におけるタグの数
        self.token_warmup_step = token_warmup_step  # N（N<1ならN*max_train_steps）ステップ目でタグの数が最大になる

        self.custom_attributes = custom_attributes if custom_attributes is not None else {}

        self.img_count = 0

        self.validation_seed = int(validation_seed) if validation_seed is not None else None
        self.validation_split = float(validation_split) if validation_split is not None else 0.0

        self.resize_interpolation = resize_interpolation


class DreamBoothSubset(BaseSubset):
    def __init__(
            self,
            image_dir: str,
            is_reg: bool,
            class_tokens: Optional[str],
            caption_extension: str,
            cache_info: bool,
            alpha_mask: bool,
            num_repeats,
            shuffle_caption,
            caption_separator: str,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            random_crop_padding_percent,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes: Optional[Dict[str, Any]] = None,
            validation_seed: Optional[int] = None,
            validation_split: Optional[float] = 0.0,
            resize_interpolation: Optional[str] = None,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            random_crop_padding_percent,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir


class FineTuningSubset(BaseSubset):
    def __init__(
            self,
            image_dir,
            metadata_file: str,
            alpha_mask: bool,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            random_crop_padding_percent,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes: Optional[Dict[str, Any]] = None,
            validation_seed: Optional[int] = None,
            validation_split: Optional[float] = 0.0,
            resize_interpolation: Optional[str] = None,
    ) -> None:
        assert metadata_file is not None, "metadata_file must be specified / metadata_fileは指定が必須です"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            random_crop_padding_percent,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.metadata_file = metadata_file

    def __eq__(self, other) -> bool:
        if not isinstance(other, FineTuningSubset):
            return NotImplemented
        return self.metadata_file == other.metadata_file


class ControlNetSubset(BaseSubset):
    def __init__(
            self,
            image_dir: str,
            conditioning_data_dir: str,
            caption_extension: str,
            cache_info: bool,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            random_crop_padding_percent,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes: Optional[Dict[str, Any]] = None,
            validation_seed: Optional[int] = None,
            validation_split: Optional[float] = 0.0,
            resize_interpolation: Optional[str] = None,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            False,  # alpha_mask
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            random_crop_padding_percent,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.conditioning_data_dir = conditioning_data_dir
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, ControlNetSubset):
            return NotImplemented
        return self.image_dir == other.image_dir and self.conditioning_data_dir == other.conditioning_data_dir
