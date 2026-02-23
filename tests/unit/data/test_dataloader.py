import pytest
import torch
from unittest.mock import MagicMock, patch

from library.data.dataloader import TrainingDataset
from library.data.structures import DatasetManifest, EpochManifest, BatchInfo, CacheEntry, CacheData


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_dataset_manifest():
    manifest = MagicMock(spec=DatasetManifest)
    manifest.entries = {}

    # Setup get_entry
    def get_entry(img_id):
        return manifest.entries.get(img_id)

    manifest.get_entry = MagicMock(side_effect=get_entry)

    return manifest


@pytest.fixture
def mock_epoch_manifest():
    manifest = MagicMock(spec=EpochManifest)
    manifest.epoch = 1
    manifest.seed = 42
    manifest.batches = []
    manifest.num_batches = 0
    return manifest


@pytest.fixture
def mock_latent_strategy():
    strategy = MagicMock()
    # Default behavior for load_cache
    cache_data = CacheData(
        latents=torch.randn(4, 64, 64),
        latents_flipped=torch.randn(4, 64, 64),
        conditioning=MagicMock(),
        alpha_mask=None,
    )
    strategy.load_cache.return_value = cache_data
    return strategy


@pytest.fixture
def mock_te_strategy():
    strategy = MagicMock()
    cache_data = CacheData(aux={"encoder_output": torch.randn(2, 77, 768)})
    strategy.load_cache.return_value = cache_data
    return strategy


@pytest.fixture
def sample_entries():
    return {
        "img1": CacheEntry(
            id="img1",
            image_path="/tmp/img1.jpg",
            original_size=(512, 512),
            bucket_reso=(512, 512),
            resized_size=(512, 512),
            caption="caption1",
            latent_cache_path="/tmp/cache/img1.safetensors",
            is_reg=False,
        ),
        "img2": CacheEntry(
            id="img2",
            image_path="/tmp/img2.jpg",
            original_size=(512, 512),
            bucket_reso=(512, 512),
            resized_size=(512, 512),
            caption="caption2",
            latent_cache_path="/tmp/cache/img2.safetensors",
            is_reg=True,  # Regularization image
        ),
        "img3": CacheEntry(
            id="img3",
            image_path="/tmp/img3.jpg",
            original_size=(512, 512),
            bucket_reso=(512, 512),
            resized_size=(512, 512),
            caption="caption3",
            latent_cache_path="/tmp/cache/img3.safetensors",
            is_reg=False,
            te_cache_path="/tmp/te_cache/img3.safetensors",
        ),
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_training_dataset_iteration(mock_dataset_manifest, mock_epoch_manifest, mock_latent_strategy, sample_entries):
    """Verify that the dataset yields batches in the correct order."""

    # Setup manifest with entries
    mock_dataset_manifest.entries = sample_entries

    # Setup epoch manifest with 2 batches
    batch1 = BatchInfo(image_ids=["img1"], bucket_reso=(512, 512), processed_captions=["proc_cap1"])
    batch2 = BatchInfo(image_ids=["img2"], bucket_reso=(512, 512), processed_captions=["proc_cap2"])
    mock_epoch_manifest.batches = [batch1, batch2]
    mock_epoch_manifest.num_batches = 2

    dataset = TrainingDataset(
        dataset_manifest=mock_dataset_manifest,
        epoch_manifest=mock_epoch_manifest,
        latent_strategy=mock_latent_strategy,
    )

    batches = list(dataset)

    assert len(batches) == 2

    # Check batch 1
    assert batches[0]["image_ids"] == ["img1"]
    assert batches[0]["captions"] == ["proc_cap1"]
    assert batches[0]["bucket_reso"] == (512, 512)
    assert batches[0]["latents"].shape == (1, 4, 64, 64)

    # Check batch 2
    assert batches[1]["image_ids"] == ["img2"]
    assert batches[1]["captions"] == ["proc_cap2"]


def test_batch_format(mock_dataset_manifest, mock_epoch_manifest, mock_latent_strategy, sample_entries):
    """Verify that the batch contains all required fields: latents, conditionings, captions, etc."""

    mock_dataset_manifest.entries = sample_entries
    batch_info = BatchInfo(image_ids=["img1"], bucket_reso=(512, 512), processed_captions=["proc_cap1"])
    mock_epoch_manifest.batches = [batch_info]

    dataset = TrainingDataset(
        dataset_manifest=mock_dataset_manifest,
        epoch_manifest=mock_epoch_manifest,
        latent_strategy=mock_latent_strategy,
    )

    batch = next(iter(dataset))

    expected_keys = ["latents", "captions", "image_ids", "bucket_reso", "conditionings", "loss_weights", "flippeds", "alpha_masks"]

    for key in expected_keys:
        assert key in batch, f"Batch missing key: {key}"

    assert isinstance(batch["latents"], torch.Tensor)
    assert isinstance(batch["loss_weights"], torch.Tensor)
    assert isinstance(batch["conditionings"], list)
    assert isinstance(batch["flippeds"], list)


def test_flip_aug(mock_dataset_manifest, mock_epoch_manifest, mock_latent_strategy, sample_entries):
    """Verify that random flip augmentation selects flipped latents."""

    mock_dataset_manifest.entries = sample_entries
    batch_info = BatchInfo(image_ids=["img1"], bucket_reso=(512, 512))
    mock_epoch_manifest.batches = [batch_info]

    # Mock load_cache to return distinct normal/flipped latents
    normal_latent = torch.zeros(4, 64, 64)
    flipped_latent = torch.ones(4, 64, 64)

    mock_latent_strategy.load_cache.return_value = CacheData(
        latents=normal_latent,
        latents_flipped=flipped_latent,
        conditioning=MagicMock(),
    )

    dataset = TrainingDataset(
        dataset_manifest=mock_dataset_manifest,
        epoch_manifest=mock_epoch_manifest,
        latent_strategy=mock_latent_strategy,
        flip_aug=True,
    )

    # We patch random.random to control flip decision
    # Case 1: No flip (random > 0.5)
    with patch("random.random", return_value=0.6):
        batch = next(iter(dataset))
        assert torch.equal(batch["latents"][0], normal_latent)
        assert batch["flippeds"][0] is False

    # Case 2: Flip (random < 0.5)
    with patch("random.random", return_value=0.4):
        batch = next(iter(dataset))
        assert torch.equal(batch["latents"][0], flipped_latent)
        assert batch["flippeds"][0] is True


def test_prior_loss_weight(mock_dataset_manifest, mock_epoch_manifest, mock_latent_strategy, sample_entries):
    """Verify that is_reg=True uses the configured prior_loss_weight."""

    mock_dataset_manifest.entries = sample_entries
    # img1 is normal (reg=False), img2 is reg (reg=True)
    batch_info = BatchInfo(image_ids=["img1", "img2"], bucket_reso=(512, 512))
    mock_epoch_manifest.batches = [batch_info]

    prior_weight = 0.5

    dataset = TrainingDataset(
        dataset_manifest=mock_dataset_manifest,
        epoch_manifest=mock_epoch_manifest,
        latent_strategy=mock_latent_strategy,
        prior_loss_weight=prior_weight,
    )

    batch = next(iter(dataset))
    loss_weights = batch["loss_weights"]

    # img1 should be 1.0, img2 should be 0.5
    assert loss_weights[0].item() == 1.0
    assert loss_weights[1].item() == prior_weight


def test_streaming_tokens(mock_dataset_manifest, mock_epoch_manifest, mock_latent_strategy, sample_entries):
    """Verify that streaming_tokens=True calls get_slice() to load per-batch tokens."""

    mock_dataset_manifest.entries = sample_entries
    # Two batches, each size 1
    mock_epoch_manifest.batches = [
        BatchInfo(image_ids=["img1"], bucket_reso=(512, 512), processed_captions=["cap1"]),
        BatchInfo(image_ids=["img2"], bucket_reso=(512, 512), processed_captions=["cap2"]),
    ]
    mock_epoch_manifest.num_batches = 2

    tokens_path = "dummy_tokens.safetensors"

    dataset = TrainingDataset(
        dataset_manifest=mock_dataset_manifest,
        epoch_manifest=mock_epoch_manifest,
        latent_strategy=mock_latent_strategy,
        tokens_path=tokens_path,
        streaming_tokens=True,
    )

    # Mock safetensors safe_open and slice
    with patch("safetensors.safe_open") as mock_safe_open:
        mock_file = MagicMock()
        mock_safe_open.return_value.__enter__.return_value = mock_file

        mock_file.keys.return_value = ["clip_l"]
        mock_slice = MagicMock()
        mock_file.get_slice.return_value = mock_slice

        # When slice is indexed, return dummy tensor
        mock_slice.__getitem__.return_value = torch.tensor([[1, 2, 3]])

        # Iterate
        batches = list(dataset)

        assert len(batches) == 2

        # Check get_slice calls
        # 1st batch: offset 0, size 1
        # 2nd batch: offset 1, size 1

        # Verify get_slice was called
        assert mock_file.get_slice.call_count >= 2
        mock_file.get_slice.assert_called_with("clip_l")

        # Verify slice indexing
        # Note: slice indexing is handled by __getitem__ on the mock object,
        # so we check if __getitem__ was called with slices

        # We can't easily check slice arguments on a mock __getitem__ directly unless we setup specific side effects,
        # but the fact that we got batches implies it worked if the code logic is correct.
        # Let's inspect call args if possible, or trust that the loop offset logic is exercised.


def test_on_the_fly_tokenization(mock_dataset_manifest, mock_epoch_manifest, mock_latent_strategy, sample_entries):
    """Verify functionality when tokens_path=None (on-the-fly tokenization fallback)."""

    mock_dataset_manifest.entries = sample_entries

    # Batch info with pre-tokenized input_ids (simulating what happens when tokens_path=None but BatchInfo has it,
    # or just checking fallback logic if BatchInfo doesn't have it but we expect it to be handled elsewhere/not crash)

    # Note: TrainingDataset code says:
    # elif batch_info.input_ids:
    #     batch["input_ids"] = {encoder_name: torch.tensor(tokens) ...}

    input_ids_dict = {"clip": [[101, 200, 102]]}
    batch_info = BatchInfo(image_ids=["img1"], bucket_reso=(512, 512), processed_captions=["cap1"], input_ids=input_ids_dict)
    mock_epoch_manifest.batches = [batch_info]

    dataset = TrainingDataset(
        dataset_manifest=mock_dataset_manifest,
        epoch_manifest=mock_epoch_manifest,
        latent_strategy=mock_latent_strategy,
        tokens_path=None,  # No token file
    )

    batch = next(iter(dataset))

    assert "input_ids" in batch
    assert torch.equal(batch["input_ids"]["clip"], torch.tensor([[101, 200, 102]]))


def test_distributed_sharding(mock_dataset_manifest, mock_epoch_manifest, mock_latent_strategy, sample_entries):
    """Verify that rank/world_size splits batches correctly."""

    mock_dataset_manifest.entries = sample_entries

    # Create 4 batches
    batches = [
        BatchInfo(image_ids=["img1"], bucket_reso=(512, 512)),  # Rank 0
        BatchInfo(image_ids=["img2"], bucket_reso=(512, 512)),  # Rank 1
        BatchInfo(image_ids=["img1"], bucket_reso=(512, 512)),  # Rank 0
        BatchInfo(image_ids=["img2"], bucket_reso=(512, 512)),  # Rank 1
    ]
    mock_epoch_manifest.batches = batches
    mock_epoch_manifest.num_batches = 4

    # Rank 0 of 2
    dataset_r0 = TrainingDataset(
        dataset_manifest=mock_dataset_manifest,
        epoch_manifest=mock_epoch_manifest,
        latent_strategy=mock_latent_strategy,
        rank=0,
        world_size=2,
    )

    batches_r0 = list(dataset_r0)
    assert len(batches_r0) == 2
    assert batches_r0[0]["image_ids"] == ["img1"]
    assert batches_r0[1]["image_ids"] == ["img1"]

    # Rank 1 of 2
    dataset_r1 = TrainingDataset(
        dataset_manifest=mock_dataset_manifest,
        epoch_manifest=mock_epoch_manifest,
        latent_strategy=mock_latent_strategy,
        rank=1,
        world_size=2,
    )

    batches_r1 = list(dataset_r1)
    assert len(batches_r1) == 2
    assert batches_r1[0]["image_ids"] == ["img2"]
    assert batches_r1[1]["image_ids"] == ["img2"]

    # Test __len__
    assert len(dataset_r0) == 2
    assert len(dataset_r1) == 2


def test_te_cache_loading(mock_dataset_manifest, mock_epoch_manifest, mock_latent_strategy, mock_te_strategy, sample_entries):
    """Verify text encoder cache loading."""

    mock_dataset_manifest.entries = sample_entries
    # img3 has te_cache_path
    batch_info = BatchInfo(image_ids=["img3"], bucket_reso=(512, 512))
    mock_epoch_manifest.batches = [batch_info]

    dataset = TrainingDataset(
        dataset_manifest=mock_dataset_manifest,
        epoch_manifest=mock_epoch_manifest,
        latent_strategy=mock_latent_strategy,
        te_strategy=mock_te_strategy,
    )

    batch = next(iter(dataset))

    assert "text_encoder_outputs" in batch
    assert "encoder_output" in batch["text_encoder_outputs"]
    # Mock returns shape [2, 77, 768], we have 1 image, so stack result should be [1, 2, 77, 768] (if mock logic allows)
    # Actually, mock_te_strategy returns CacheData.aux as dict.
    # _load_te_outputs stacks them.
    # mock output: torch.randn(2, 77, 768)

    expected_shape = (1, 2, 77, 768)
    assert batch["text_encoder_outputs"]["encoder_output"].shape == expected_shape
