
import pytest
from unittest.mock import MagicMock, patch
from library.data.dataset import BaseDataset, ImageInfo

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
            resolution=(512, 512), 
            min_bucket_reso=256, 
            max_bucket_reso=1024, 
            bucket_reso_steps=64
        )
        assert min_r == 256
        assert max_r == 1024

    def test_adjust_reso_needs_adjustment(self, mock_dataset):
        # 200 -> rounds down to 192 (64*3)
        # 1000 -> rounds up to 1024 (64*16)
        min_r, max_r = mock_dataset.adjust_min_max_bucket_reso_by_steps(
            resolution=(512, 512), 
            min_bucket_reso=200, 
            max_bucket_reso=1000, 
            bucket_reso_steps=64
        )
        assert min_r == 192
        assert max_r == 1024

    def test_adjust_reso_assertion_error(self, mock_dataset):
        # min_bucket_reso (1024) > resolution (512) -> Should raise Assertion Error
        with pytest.raises(AssertionError):
            mock_dataset.adjust_min_max_bucket_reso_by_steps(
                resolution=(512, 512), 
                min_bucket_reso=1024, 
                max_bucket_reso=2048, 
                bucket_reso_steps=64
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
            bm.select_bucket.return_value = ((512, 512), (512, 512), 0.0)
            bm.buckets = [["img1"]] # Mock one bucket with one image (list of lists)
            bm.resos = [(512, 512)]
            bm.sort = MagicMock()
            bm.shuffle = MagicMock()
            
            # Run make_buckets
            mock_dataset.make_buckets()
            
            # Verifications
            MockBucketManager.assert_called_once() # Initialized
            bm.make_buckets.assert_called_once() # Logic called
            bm.select_bucket.assert_called_with(512, 512) # Selection called
            
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
            bm.select_bucket.return_value = ((512, 512), (512, 512), 0.0)
            bm.buckets = [] 
            bm.shuffle = MagicMock() # Needed for shuffle_buckets

            mock_dataset.make_buckets()
            
            # Should NOT call make_buckets on manager if no_upscale is True
            bm.make_buckets.assert_not_called()

    def test_make_buckets_fetch_image_size(self, mock_dataset):
        # Image info with NO size initially
        img1 = MagicMock(spec=ImageInfo)
        img1.image_key = "img1" # explicitly set image_key
        img1.image_size = None
        img1.absolute_path = "/tmp/img1.png"
        img1.num_repeats = 1
        mock_dataset.image_data = {"img1": img1}
        
        # Mock get_image_size method on the dataset itself
        mock_dataset.get_image_size = MagicMock(return_value=(300, 300))

        with patch("library.data.dataset.BucketManager") as MockBucketManager:
            bm = MockBucketManager.return_value
            # We need to ensure select_bucket doesn't fail return logic
            bm.select_bucket.return_value = ((320, 320), (300, 300), 0.0)
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
            bm.select_bucket.return_value = ((512, 512), (512, 512), 0.0)
            bm.resos = [(512, 512)]
            bm.buckets = [["img1"]]
            bm.shuffle = MagicMock()
    
            mock_dataset.make_buckets()
    
            # Should call set_predefined_resos with global resolution
            bm.set_predefined_resos.assert_called_once_with([(512, 512)])
            # Should NOT call make_buckets
            bm.make_buckets.assert_not_called()
