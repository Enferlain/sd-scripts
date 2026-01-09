import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

import library.data.bucketing
from library.data._deprecated.dataset import BaseDataset, ImageInfo


@pytest.mark.unit
class TestDatasetBucketing:
    @pytest.fixture
    def mock_dataset(self):
        """Creates a BaseDataset with mocked init to avoid overhead."""
        # We patch __init__ so we can instantiate without real dependencies
        with patch("library.data.dataset.BaseDataset.__init__", return_value=None):
            dataset = BaseDataset(None, 1.0, False)
            # Manually set attributes needed for the tests
            dataset.enable_bucket = True
            dataset.bucket_manager = None
            dataset.width = 512
            dataset.height = 512
            dataset.min_bucket_reso = 256
            dataset.max_bucket_reso = 1024
            dataset.bucket_reso_steps = 64
            dataset.bucket_no_upscale = False
            dataset.image_data = {}
            dataset.bucket_info = None
            dataset.batch_size = 1

            # Missing attributes causing failures
            dataset.seed = 42
            dataset.current_epoch = 0
            dataset.max_train_steps = 100

            return dataset

    # ==========================================
    # adjust_min_max_bucket_reso_by_steps Tests
    # ==========================================

    def test_adjust_reso_perfect_match(self, mock_dataset):
        # 256 and 1024 are divisible by 64
        min_r, max_r = mock_dataset.adjust_min_max_bucket_reso_by_steps(
            resolution=(512, 512), min_bucket_reso=256, max_bucket_reso=1024, bucket_reso_steps=64
        )
        assert min_r == 256
        assert max_r == 1024

    def test_adjust_reso_needs_adjustment(self, mock_dataset):
        # 200 -> rounds down to 192 (64*3)
        # 1000 -> rounds up to 1024 (64*16)
        min_r, max_r = mock_dataset.adjust_min_max_bucket_reso_by_steps(
            resolution=(512, 512), min_bucket_reso=200, max_bucket_reso=1000, bucket_reso_steps=64
        )
        assert min_r == 192
        assert max_r == 1024

    def test_adjust_reso_assertion_error(self, mock_dataset):
        # min_bucket_reso (1024) > resolution (512) -> Should raise Assertion Error
        with pytest.raises(AssertionError):
            mock_dataset.adjust_min_max_bucket_reso_by_steps(
                resolution=(512, 512), min_bucket_reso=1024, max_bucket_reso=2048, bucket_reso_steps=64
            )

    # ==========================================
    # make_buckets Tests
    # ==========================================

    def test_make_buckets_basic_flow(self, mock_dataset):
        # Setup Image Data
        img1 = MagicMock(spec=ImageInfo)
        img1.image_key = "img1"
        img1.image_size = (512, 512)
        img1.num_repeats = 1
        img1.absolute_path = "/tmp/img1.png"

        mock_dataset.image_data = {"img1": img1}

        # Mock BucketManager class
        with patch("library.data.dataset.BucketManager") as MockBucketManager:
            # Setup BucketManager instance
            bm = MockBucketManager.return_value
            library.data.bucketing.select_bucket.return_value = ((512, 512), (512, 512), 0.0)
            bm.buckets = [["img1"]]  # Mock one bucket with one image (list of lists)
            bm.resos = [(512, 512)]
            bm.sort = MagicMock()
            bm.shuffle = MagicMock()

            # Run make_buckets
            mock_dataset.make_buckets()

            # Verifications
            MockBucketManager.assert_called_once()  # Initialized
            bm.make_buckets.assert_called_once()  # Logic called
            library.data.bucketing.select_bucket.assert_called_with(512, 512)  # Selection called

            # Verify bucket_reso update and add_image call
            assert img1.bucket_reso == (512, 512)
            bm.add_image.assert_called_with(img1.bucket_reso, "img1")

            # Verify bucket_info was populated
            assert mock_dataset.bucket_info is not None
            assert 0 in mock_dataset.bucket_info["buckets"]

            # Verify sort and shuffle
            bm.sort.assert_called_once()
            bm.shuffle.assert_called_once()

    def test_make_buckets_no_upscale(self, mock_dataset):
        mock_dataset.bucket_no_upscale = True

        # Setup Image Data to avoid empty loop skipping logic if any
        img1 = MagicMock(spec=ImageInfo)
        img1.image_key = "img1"
        img1.image_size = (512, 512)
        img1.num_repeats = 1
        mock_dataset.image_data = {"img1": img1}

        with patch("library.data.dataset.BucketManager") as MockBucketManager:
            bm = MockBucketManager.return_value
            # select_bucket still needs to return something
            library.data.bucketing.select_bucket.return_value = ((512, 512), (512, 512), 0.0)
            bm.buckets = []
            bm.shuffle = MagicMock()  # Needed for shuffle_buckets

            mock_dataset.make_buckets()

            # Should NOT call make_buckets on manager if no_upscale is True
            bm.make_buckets.assert_not_called()

    def test_make_buckets_fetch_image_size(self, mock_dataset):
        # Image info with NO size initially
        img1 = MagicMock(spec=ImageInfo)
        img1.image_key = "img1"  # explicitly set image_key
        img1.image_size = None
        img1.absolute_path = "/tmp/img1.png"
        img1.num_repeats = 1
        mock_dataset.image_data = {"img1": img1}

        # Mock get_image_size method on the dataset itself
        mock_dataset.get_image_size = MagicMock(return_value=(300, 300))

        with patch("library.data.dataset.BucketManager") as MockBucketManager:
            bm = MockBucketManager.return_value
            # We need to ensure select_bucket doesn't fail return logic
            library.data.bucketing.select_bucket.return_value = ((320, 320), (300, 300), 0.0)
            bm.buckets = []
            bm.shuffle = MagicMock()

            mock_dataset.make_buckets()

            # Verify size was fetched
            mock_dataset.get_image_size.assert_called_with("/tmp/img1.png")
            assert img1.image_size == (300, 300)

    def test_make_buckets_disable_bucket_uses_fixed_reso(self, mock_dataset):
        """Test the path where enable_bucket is False."""
        mock_dataset.enable_bucket = False

        img1 = MagicMock(spec=ImageInfo)
        img1.image_key = "img1"
        img1.image_size = (640, 480)
        img1.num_repeats = 1
        mock_dataset.image_data = {"img1": img1}

        with patch("library.data.dataset.BucketManager") as MockBucketManager:
            bm = MockBucketManager.return_value
            bm.set_predefined_resos = MagicMock()
            library.data.bucketing.select_bucket.return_value = ((512, 512), (512, 512), 0.0)
            bm.resos = [(512, 512)]
            bm.buckets = [["img1"]]
            bm.shuffle = MagicMock()

            mock_dataset.make_buckets()

            # Should call set_predefined_resos with global resolution
            bm.set_predefined_resos.assert_called_once_with([(512, 512)])
            # Should NOT call make_buckets
            bm.make_buckets.assert_not_called()


# =============================================================================
# __getitem__ Tests
# =============================================================================


@pytest.mark.unit
class TestDatasetGetItem:
    """Tests for BaseDataset.__getitem__ method."""

    @pytest.fixture
    def mock_dataset_for_getitem(self):
        """Creates a BaseDataset with mocked init and all dependencies for __getitem__."""
        with patch("library.data.dataset.BaseDataset.__init__", return_value=None):
            dataset = BaseDataset(None, 1.0, False)

            # Core attributes
            dataset.caching_mode = None  # Not in caching mode
            dataset.enable_bucket = True
            dataset.prior_loss_weight = 0.5
            dataset.adapter_multiplier = 1.0
            dataset.debug_dataset = False
            dataset.batch_size = 1
            dataset.width = 512
            dataset.height = 512

            # Strategy mocks
            dataset.text_encoder_output_caching_strategy = None
            dataset.tokenize_strategy = MagicMock()
            # tokenize returns list of [tensor per tokenizer]; each tensor is [batch x seq_len]
            dataset.tokenize_strategy.tokenize.return_value = [[torch.tensor([1, 2, 3])]]
            dataset.latents_caching_strategy = MagicMock()

            # Augmentation
            dataset.aug_helper = MagicMock()
            dataset.aug_helper.get_augmentor.return_value = None  # No augmentation

            # Image transforms
            dataset.image_transforms = MagicMock(return_value=torch.randn(3, 512, 512))

            # Caption processing
            dataset.process_caption = MagicMock(side_effect=lambda s, c: c)  # Return caption as-is

            return dataset

    def _create_bucket_index(self, bucket_index=0, bucket_batch_size=1, batch_index=0):
        """Helper to create bucket index mock."""
        bi = MagicMock()
        bi.bucket_index = bucket_index
        bi.bucket_batch_size = bucket_batch_size
        bi.batch_index = batch_index
        return bi

    def _create_image_info(self, image_key, has_latents=False, has_npz=False, caption="test caption"):
        """Helper to create ImageInfo mock."""
        info = MagicMock(spec=ImageInfo)
        info.image_key = image_key
        info.caption = caption
        info.is_reg = False
        info.bucket_reso = (512, 512)
        info.resized_size = (512, 512)
        info.resize_interpolation = 1
        info.absolute_path = f"/tmp/{image_key}.png"
        info.text_encoder_outputs = None
        info.text_encoder_outputs_npz = None

        if has_latents:
            info.latents = torch.randn(4, 64, 64)
            info.latents_flipped = torch.randn(4, 64, 64)
            info.latents_original_size = (512, 512)
            info.latents_crop_ltrb = (0, 0, 0, 0)
            info.alpha_mask = None
            info.latents_npz = None
        elif has_npz:
            info.latents = None
            info.latents_npz = "/tmp/latents.npz"
        else:
            info.latents = None
            info.latents_npz = None

        return info

    def _create_subset(self, flip_aug=False, random_crop=False, color_aug=False, alpha_mask=False):
        """Helper to create subset mock."""
        subset = MagicMock()
        subset.flip_aug = flip_aug
        subset.random_crop = random_crop
        subset.color_aug = color_aug
        subset.alpha_mask = alpha_mask
        subset.custom_attributes = {}
        subset.random_crop_padding_percent = 0.0
        return subset

    def test_getitem_with_cached_latents(self, mock_dataset_for_getitem):
        """Test __getitem__ with latents cached in memory."""
        dataset = mock_dataset_for_getitem

        # Setup bucket manager and indices
        dataset.bucket_manager = MagicMock()
        dataset.bucket_manager.buckets = [["img1"]]
        dataset.buckets_indices = [self._create_bucket_index()]

        # Setup image data with cached latents
        img_info = self._create_image_info("img1", has_latents=True)
        dataset.image_data = {"img1": img_info}
        dataset.image_to_subset = {"img1": self._create_subset()}

        # Call __getitem__
        with patch("random.random", return_value=0.9):  # No flip
            example = dataset[0]

        # Verify output structure
        assert "latents" in example
        assert example["latents"] is not None
        assert example["latents"].shape == (1, 4, 64, 64)  # Batched
        assert example["images"] is None  # No images when latents are cached
        assert "loss_weights" in example
        assert "original_sizes_hw" in example
        assert "crop_top_lefts" in example
        assert "target_sizes_hw" in example

    def test_getitem_with_disk_latents(self, mock_dataset_for_getitem):
        """Test __getitem__ with latents loaded from disk."""
        dataset = mock_dataset_for_getitem

        # Setup bucket manager and indices
        dataset.bucket_manager = MagicMock()
        dataset.bucket_manager.buckets = [["img1"]]
        dataset.buckets_indices = [self._create_bucket_index()]

        # Setup image data with disk latents
        img_info = self._create_image_info("img1", has_npz=True)
        dataset.image_data = {"img1": img_info}
        dataset.image_to_subset = {"img1": self._create_subset()}

        # Mock latents loading from disk
        mock_latents = np.random.randn(4, 64, 64).astype(np.float32)
        dataset.latents_caching_strategy.load_latents_from_disk.return_value = (
            mock_latents,  # latents
            (512, 512),  # original_size
            (0, 0, 0, 0),  # crop_ltrb
            mock_latents,  # flipped_latents
            None,  # alpha_mask
        )

        with patch("random.random", return_value=0.9):  # No flip
            example = dataset[0]

        # Verify latents were loaded
        dataset.latents_caching_strategy.load_latents_from_disk.assert_called_once()
        assert example["latents"] is not None
        assert example["images"] is None

    def test_getitem_with_image_loading(self, mock_dataset_for_getitem):
        """Test __getitem__ with image loading from file."""
        dataset = mock_dataset_for_getitem

        # Setup bucket manager and indices
        dataset.bucket_manager = MagicMock()
        dataset.bucket_manager.buckets = [["img1"]]
        dataset.buckets_indices = [self._create_bucket_index()]

        # Setup image data without latents (needs image loading)
        img_info = self._create_image_info("img1", has_latents=False, has_npz=False)
        dataset.image_data = {"img1": img_info}
        dataset.image_to_subset = {"img1": self._create_subset()}

        # Mock image loading
        mock_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        dataset.load_image_with_face_info = MagicMock(return_value=(mock_img, 0, 0, 0, 0))

        with patch("random.random", return_value=0.9):  # No flip
            with patch("library.data.dataset.trim_and_resize_if_required") as mock_trim:
                mock_trim.return_value = (mock_img, [512, 512], (0, 0, 0, 0))
                example = dataset[0]

        # Verify image was loaded and transformed
        dataset.load_image_with_face_info.assert_called_once()
        dataset.image_transforms.assert_called_once()
        assert example["images"] is not None
        assert example["latents"] is None

    def test_getitem_with_flip_augmentation(self, mock_dataset_for_getitem):
        """Test __getitem__ with flip augmentation enabled."""
        dataset = mock_dataset_for_getitem

        # Setup with flip_aug enabled
        dataset.bucket_manager = MagicMock()
        dataset.bucket_manager.buckets = [["img1"]]
        dataset.buckets_indices = [self._create_bucket_index()]

        img_info = self._create_image_info("img1", has_latents=True)
        dataset.image_data = {"img1": img_info}
        dataset.image_to_subset = {"img1": self._create_subset(flip_aug=True)}

        with patch("random.random", return_value=0.3):  # Will flip (< 0.5)
            example = dataset[0]

        # Verify flipped latents were used
        assert example["flippeds"] == [True]

    def test_getitem_caching_mode_early_return(self, mock_dataset_for_getitem):
        """Test __getitem__ returns early when in caching mode."""
        dataset = mock_dataset_for_getitem
        dataset.caching_mode = "latent"  # Enable caching mode

        # Setup minimal bucket data
        dataset.bucket_manager = MagicMock()
        dataset.bucket_manager.buckets = [["img1"]]
        dataset.buckets_indices = [self._create_bucket_index()]

        # Mock get_item_for_caching
        dataset.get_item_for_caching = MagicMock(return_value={"cached": True})

        example = dataset[0]

        # Should call caching method and return early
        dataset.get_item_for_caching.assert_called_once()
        assert example == {"cached": True}

    def test_getitem_loss_weight_for_reg_image(self, mock_dataset_for_getitem):
        """Test that regularization images get prior_loss_weight."""
        dataset = mock_dataset_for_getitem

        dataset.bucket_manager = MagicMock()
        dataset.bucket_manager.buckets = [["img1"]]
        dataset.buckets_indices = [self._create_bucket_index()]

        # Create reg image
        img_info = self._create_image_info("img1", has_latents=True)
        img_info.is_reg = True  # This is a regularization image
        dataset.image_data = {"img1": img_info}
        dataset.image_to_subset = {"img1": self._create_subset()}

        with patch("random.random", return_value=0.9):
            example = dataset[0]

        # Loss weight should be prior_loss_weight for reg images
        assert example["loss_weights"][0].item() == 0.5  # prior_loss_weight

    def test_getitem_batch_of_two(self, mock_dataset_for_getitem):
        """Test __getitem__ with batch size of 2."""
        dataset = mock_dataset_for_getitem

        dataset.bucket_manager = MagicMock()
        dataset.bucket_manager.buckets = [["img1", "img2"]]
        dataset.buckets_indices = [self._create_bucket_index(bucket_batch_size=2)]

        # Setup two images with latents
        img_info1 = self._create_image_info("img1", has_latents=True)
        img_info2 = self._create_image_info("img2", has_latents=True)
        dataset.image_data = {"img1": img_info1, "img2": img_info2}
        dataset.image_to_subset = {"img1": self._create_subset(), "img2": self._create_subset()}

        with patch("random.random", return_value=0.9):
            example = dataset[0]

        # Should have batched results
        assert example["latents"].shape[0] == 2
        assert example["loss_weights"].shape[0] == 2
        assert len(example["captions"]) == 2
