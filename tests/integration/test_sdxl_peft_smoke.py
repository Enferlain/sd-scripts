"""
Smoke tests for SDXL PEFT data pipeline integration.

Tests the new data pipeline components integrated into sdxl_peft.py:
- create_manifest_from_config
- CachingEngine with SDXL strategies
- Per-epoch DataLoader creation
- training metadata backbone generation
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from library.data import (
    CaptionConfig,
    CachingEngine,
    DatasetManifest,
    create_training_dataloader,
    prepare_epoch,
    prepare_validation_epoch,
)
from library.data import create_manifest_from_config, compute_tag_frequency
from library.strategies.sdxl.caching import (
    SdxlLatentsPipelineStrategy,
    SdxlTextEncoderPipelineStrategy,
)
from library.training.metadata import TrainingMetadataBuildContext, build_training_metadata_bundle


# Path to test assets
TEST_IMAGES_DIR = Path(__file__).parent.parent / "assets" / "images"
TEST_VAE_PATH = Path(__file__).parent.parent / "assets" / "sdxl_vae.safetensors"

# Count actual image files in test directory
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".jxl"}
EXPECTED_IMAGE_COUNT = (
    len([f for f in TEST_IMAGES_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]) if TEST_IMAGES_DIR.exists() else 0
)


@pytest.fixture
def mock_cfg():
    """Create a mock config matching the shared PEFT config structure."""
    cfg = MagicMock()

    # Data config
    cfg.data.source.train_data_dir = str(TEST_IMAGES_DIR)
    cfg.data.source.reg_data_dir = None
    cfg.data.source.val_data_dir = None
    cfg.data.source.in_json = None
    cfg.data.source.subsets = []
    cfg.data.source.dataset_repeats = 1

    cfg.data.preprocessing.resolution = "1024,1024"
    cfg.data.preprocessing.flip_aug = False
    cfg.data.preprocessing.color_aug = False
    cfg.data.preprocessing.random_crop = False
    cfg.data.preprocessing.face_crop_aug_range = None
    cfg.data.preprocessing.resize_interpolation = "LANCZOS"

    cfg.data.bucketing.enable_bucket = True
    cfg.data.bucketing.min_bucket_reso = 256
    cfg.data.bucketing.max_bucket_reso = 2048
    cfg.data.bucketing.bucket_reso_steps = 64
    cfg.data.bucketing.bucket_no_upscale = False

    cfg.data.caption.caption_extension = ".txt"
    cfg.data.caption.caption_separator = ","
    cfg.data.caption.shuffle_caption = False
    cfg.data.caption.keep_tokens = 0
    cfg.data.caption.caption_dropout_rate = 0.0
    cfg.data.caption.caption_dropout_every_n_epochs = 0
    cfg.data.caption.caption_tag_dropout_rate = 0.0
    cfg.data.caption.caption_prefix = None
    cfg.data.caption.caption_suffix = None
    cfg.data.caption.weighted_captions = False
    cfg.data.caption.token_warmup_min = 1
    cfg.data.caption.token_warmup_step = 0.0
    cfg.data.caption.keep_tokens_separator = ""
    cfg.data.caption.secondary_separator = None
    cfg.data.caption.enable_wildcard = False

    cfg.data.caching.cache_dir = None
    cfg.data.caching.cache_latents = True
    cfg.data.caching.cache_latents_to_disk = True
    cfg.data.caching.cache_text_encoder_outputs = False
    cfg.data.caching.vae_batch_size = 2
    cfg.data.caching.skip_cache_check = False

    cfg.data.loader.max_workers = 2
    cfg.data.loader.persistent_workers = False

    # Training config
    cfg.training.train_batch_size = 1
    cfg.training.seed = 42
    cfg.training.max_token_length = 77
    cfg.training.clip_skip = 1
    cfg.training.max_train_steps = 100
    cfg.training.gradient_accumulation_steps = 1

    # Objective config
    cfg.objective.path = "ddpm"
    cfg.objective.prediction = "epsilon"

    # Validation config
    cfg.validation.validation_split = 0.0
    cfg.validation.validation_seed = 42

    # Performance config
    cfg.performance.precision.no_half_vae = False
    cfg.performance.precision.mixed_precision = "fp16"
    cfg.performance.precision.full_fp16 = False
    cfg.performance.precision.fp8_base = False
    cfg.performance.precision.fp8_base_unet = False
    cfg.performance.memory.gradient_checkpointing = False
    cfg.performance.memory.lowram = False

    # Loss config
    cfg.loss.prior_loss_weight = 1.0
    cfg.loss.regularization.noise_offset = None
    cfg.loss.regularization.multires_noise_iterations = None
    cfg.loss.regularization.multires_noise_discount = None
    cfg.loss.regularization.adaptive_noise_scale = None
    cfg.loss.regularization.zero_terminal_snr = False
    cfg.loss.regularization.ip_noise_gamma = None
    cfg.loss.regularization.noise_offset_random_strength = None
    cfg.loss.regularization.ip_noise_gamma_random_strength = None
    cfg.loss.snr.min_snr_gamma = None
    cfg.loss.snr.debiased_estimation_loss = False
    cfg.loss.loss_type = "l2"
    cfg.loss.huber.huber_schedule = None
    cfg.loss.huber.huber_scale = None
    cfg.loss.huber.huber_c = None

    # PEFT config
    cfg.adapter.peft.lora.rank = 4
    cfg.adapter.peft.lora.alpha = 1.0
    cfg.adapter.peft.lora.dropout = 0.0
    cfg.adapter.peft.continue_from = None
    cfg.adapter.peft.continue_mode = "strict"
    cfg.adapter.peft.scale_weight_norms = None

    # Optimizer config
    cfg.optimizer.learning_rates.base = 1e-4
    cfg.optimizer.learning_rates.text_encoders = None
    cfg.optimizer.learning_rates.denoiser = None
    cfg.optimizer.scheduler.lr_warmup_steps = 0
    cfg.optimizer.scheduler.lr_scheduler = "constant"
    cfg.optimizer.max_grad_norm = 1.0

    # Output config
    cfg.output.saving.output_name = "test_model"
    cfg.output.saving.hash_algorithm = "sha256"
    cfg.output.metadata.training_comment = None

    # Model config
    cfg.model.pretrained_model_name_or_path = None
    cfg.model.vae = None
    cfg.model.model_type = "sdxl"

    return cfg


@pytest.fixture
def mock_vae():
    """Create a mock VAE that returns properly shaped latents."""
    vae = MagicMock()
    vae.device = torch.device("cpu")
    vae.dtype = torch.float32

    def mock_encode(images):
        b, c, h, w = images.shape
        latent_h, latent_w = h // 8, w // 8
        mock_output = MagicMock()
        mock_output.latent_dist.sample.return_value = torch.randn(b, 4, latent_h, latent_w)
        return mock_output

    vae.encode = mock_encode
    return vae


@pytest.fixture
def mock_accelerator():
    """Create a mock accelerator for single-GPU testing."""
    accelerator = MagicMock()
    accelerator.process_index = 0
    accelerator.num_processes = 1
    accelerator.device = torch.device("cpu")
    accelerator.wait_for_everyone = MagicMock()
    return accelerator


class TestManifestCreation:
    """Test manifest creation from config."""

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_create_manifest_from_config_basic(self, mock_cfg):
        """Test creating manifest using create_manifest_from_config."""
        manifest = create_manifest_from_config(
            data_config=mock_cfg.data,
            latent_dtype="fp16",
        )

        assert isinstance(manifest, DatasetManifest)
        assert len(manifest.entries) == EXPECTED_IMAGE_COUNT
        assert len(manifest.buckets) > 0

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_manifest_entries_have_captions(self, mock_cfg):
        """Test that all manifest entries have captions."""
        manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16")

        for entry_id, entry in manifest.entries.items():
            assert entry.caption, f"Entry {entry_id} missing caption"
            assert entry.image_path, f"Entry {entry_id} missing image path"

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_validation_split(self, mock_cfg):
        """Test validation split creates val entries."""
        mock_cfg.validation.validation_split = 0.4  # 40% split

        manifest = create_manifest_from_config(
            mock_cfg.data,
            latent_dtype="fp16",
            validation_split=mock_cfg.validation.validation_split,
            validation_seed=mock_cfg.validation.validation_seed,
        )

        train_entries = [e for e in manifest.entries.values() if e.split == "train"]
        val_entries = [e for e in manifest.entries.values() if e.split == "val"]

        assert len(train_entries) >= 1, "Should have at least 1 train entry"
        assert len(val_entries) >= 1, "Should have at least 1 val entry"


class TestCachingIntegration:
    """Test caching with SDXL strategies."""

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_sdxl_latent_caching(self, mock_cfg, mock_vae, mock_accelerator):
        """Test SDXL latent caching creates proper cache files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            # Create manifest with cache_dir
            manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", cache_dir=str(cache_dir))

            # Create caching engine with SDXL strategy
            strategy = SdxlLatentsPipelineStrategy(
                flip_aug=mock_cfg.data.preprocessing.flip_aug,
                dtype="fp16",
            )
            engine = CachingEngine(
                backend=strategy,
                batch_size=mock_cfg.data.caching.vae_batch_size,
            )

            # Cache the dataset
            updated_manifest = engine.cache_dataset(
                manifest=manifest,
                model=mock_vae,
                accelerator=mock_accelerator,
                cache_dir=cache_dir,
                show_progress=False,
            )

            # Verify cache files created
            cache_files = list(cache_dir.glob("*.safetensors"))
            assert len(cache_files) == EXPECTED_IMAGE_COUNT, f"Expected {EXPECTED_IMAGE_COUNT} cache files, found {len(cache_files)}"

            # Verify entries updated with cache paths
            for entry in updated_manifest.entries.values():
                assert entry.latent_cache_path is not None

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_latent_cache_roundtrip(self, mock_cfg, mock_vae, mock_accelerator):
        """Prove full disk roundtrip: encode → save to disk → load from disk → verify data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            # Create manifest with cache_dir
            manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", cache_dir=str(cache_dir))

            # Create caching engine and cache
            strategy = SdxlLatentsPipelineStrategy(flip_aug=False, dtype="fp16")
            engine = CachingEngine(backend=strategy, batch_size=2)
            manifest = engine.cache_dataset(manifest, mock_vae, mock_accelerator, cache_dir, show_progress=False)

            # Load each cache file and verify contents
            for entry in manifest.entries.values():
                cache_path = Path(entry.latent_cache_path)
                assert cache_path.exists(), f"Cache file missing: {cache_path}"

                # Load from disk
                loaded = strategy.load_cache(cache_path)

                # Verify latents
                assert loaded.latents is not None, "Latents not loaded"
                assert loaded.latents.ndim == 3, f"Expected 3D tensor [C,H,W], got {loaded.latents.ndim}D"
                assert loaded.latents.shape[0] == 4, f"Expected 4 channels, got {loaded.latents.shape[0]}"

                # Verify conditioning metadata
                assert loaded.conditioning is not None, "Conditioning not loaded"
                assert hasattr(loaded.conditioning, "original_size_hw"), "Missing original_size_hw"
                assert hasattr(loaded.conditioning, "crop_top_left"), "Missing crop_top_left"
                assert hasattr(loaded.conditioning, "target_size_hw"), "Missing target_size_hw"


class TestDataLoaderCreation:
    """Test per-epoch DataLoader creation."""

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_prepare_epoch(self, mock_cfg, mock_vae, mock_accelerator):
        """Test prepare_epoch creates valid epoch manifest."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            # Create and cache manifest
            manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", cache_dir=str(cache_dir))
            strategy = SdxlLatentsPipelineStrategy(flip_aug=False, dtype="fp16")
            engine = CachingEngine(backend=strategy, batch_size=2)
            manifest = engine.cache_dataset(manifest, mock_vae, mock_accelerator, cache_dir, show_progress=False)

            # Prepare epoch
            caption_config = CaptionConfig(
                shuffle_caption=mock_cfg.data.caption.shuffle_caption,
                caption_dropout_rate=mock_cfg.data.caption.caption_dropout_rate,
            )
            epoch_manifest = prepare_epoch(
                manifest=manifest,
                epoch=0,
                seed=mock_cfg.training.seed,
                batch_size=mock_cfg.training.train_batch_size,
                caption_config=caption_config,
            )

            assert epoch_manifest is not None
            assert len(epoch_manifest.batches) > 0

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_create_dataloader(self, mock_cfg, mock_vae, mock_accelerator):
        """Test DataLoader yields proper batches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            # Setup
            manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", cache_dir=str(cache_dir))
            strategy = SdxlLatentsPipelineStrategy(flip_aug=False, dtype="fp16")
            engine = CachingEngine(backend=strategy, batch_size=2)
            manifest = engine.cache_dataset(manifest, mock_vae, mock_accelerator, cache_dir, show_progress=False)

            caption_config = CaptionConfig()
            epoch_manifest = prepare_epoch(manifest, 0, 42, 1, caption_config)

            # Create DataLoader
            dataloader = create_training_dataloader(
                dataset_manifest=manifest,
                epoch_manifest=epoch_manifest,
                latent_cache_backend=strategy,
                flip_aug=False,
                prior_loss_weight=1.0,
                rank=0,
                world_size=1,
            )

            # Get first batch
            batch = next(iter(dataloader))

            # Verify batch format
            assert "latents" in batch
            assert "conditionings" in batch
            assert "captions" in batch
            assert "loss_weights" in batch

            # Verify shapes
            assert batch["latents"].ndim == 4  # [B, C, H, W]
            assert batch["latents"].shape[1] == 4  # 4 latent channels


class TestTrainingMetadata:
    """Test training metadata generation from manifests."""

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_build_training_metadata_bundle(self, mock_cfg):
        """Test metadata generation from manifest."""
        manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16")

        bundle = build_training_metadata_bundle(
            TrainingMetadataBuildContext(
                cfg=mock_cfg,
                manifest=manifest,
                val_manifest=None,
                session_id=12345,
                training_started_at=1704067200.0,
                model_version="sdxl_base_v1-0",
                num_train_epochs=10,
                optimizer_name="AdamW",
                optimizer_args="",
                num_batches_per_epoch=100,
                total_batch_size=1,
            )
        )
        metadata = bundle.full.metadata

        assert "num_train_images" in metadata
        assert metadata["num_train_images"] == str(EXPECTED_IMAGE_COUNT)  # EXPECTED_IMAGE_COUNT images * 1 repeat
        assert "session_id" in metadata

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_tag_frequency_computation(self, mock_cfg):
        """Test tag frequency helper."""
        manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16")

        tag_freq = compute_tag_frequency(manifest, ",")

        assert isinstance(tag_freq, dict)
        # Should have entries for each directory
        assert len(tag_freq) > 0


class TestValidationPipeline:
    """Test validation data handling."""

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_prepare_validation_epoch(self, mock_cfg, mock_vae, mock_accelerator):
        """Test validation epoch preparation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            # Create manifest WITH validation split so we have val entries
            manifest = create_manifest_from_config(
                data_config=mock_cfg.data,
                latent_dtype="fp16",
                validation_split=0.4,  # ~40% go to val
                validation_seed=42,
            )
            strategy = SdxlLatentsPipelineStrategy(flip_aug=False, dtype="fp16")
            engine = CachingEngine(backend=strategy, batch_size=2)
            manifest = engine.cache_dataset(manifest, mock_vae, mock_accelerator, cache_dir, show_progress=False)

            # Prepare validation epoch
            val_epoch = prepare_validation_epoch(
                manifest=manifest,
                batch_size=1,
                seed=42,
            )

            assert val_epoch is not None
            assert len(val_epoch.batches) > 0, "Should have at least 1 validation batch"

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_val_data_dir_explicit(self, mock_cfg, mock_vae, mock_accelerator, tmp_path):
        """Test that val_data_dir creates separate validation entries from explicit directory."""
        import shutil

        # Create separate train and val directories
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        # Copy first 3 images to train, next 2 to val
        # Only use images that have caption files (to ensure predictable counts)
        all_images_with_captions = [
            img for img in sorted(TEST_IMAGES_DIR.iterdir()) if img.suffix.lower() in IMAGE_EXTENSIONS and img.with_suffix(".txt").exists()
        ][:5]  # Limit to 5 images for test

        for img in all_images_with_captions[:3]:
            shutil.copy(img, train_dir / img.name)
            caption = img.with_suffix(".txt")
            shutil.copy(caption, train_dir / caption.name)

        for img in all_images_with_captions[3:5]:
            shutil.copy(img, val_dir / img.name)
            caption = img.with_suffix(".txt")
            shutil.copy(caption, val_dir / caption.name)

        # Update config to use explicit directories
        mock_cfg.data.source.train_data_dir = str(train_dir)
        mock_cfg.data.source.val_data_dir = str(val_dir)

        # Create manifest - should have entries from both dirs with correct splits
        # First create training manifest, then validation manifest, and merge entries
        train_manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16")
        val_manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", validation=True)

        train_entries = list(train_manifest.entries.values())
        val_entries = list(val_manifest.entries.values())

        assert len(train_entries) == 3, f"Expected 3 train entries, got {len(train_entries)}"
        assert len(val_entries) == 2, f"Expected 2 val entries, got {len(val_entries)}"


class TestTextEncoderCaching:
    """Test text encoder output caching."""

    @pytest.fixture
    def mock_text_encoders(self):
        """Create mock CLIP text encoders that return proper tensor shapes."""
        # CLIP-L (text_encoder1) - 768 hidden dim
        te1 = MagicMock()
        te1.device = torch.device("cpu")
        te1.dtype = torch.float32

        def te1_forward(input_ids, **kwargs):
            batch_size, seq_len = input_ids.shape
            mock_output = MagicMock()
            # encode_batch accesses .hidden_states[-2], so provide a list
            hidden = torch.randn(batch_size, seq_len, 768)
            mock_output.hidden_states = [hidden, hidden, hidden]  # Fake layer outputs
            return mock_output

        te1.side_effect = te1_forward

        # CLIP-G (text_encoder2) - 1280 hidden dim with pooled output
        te2 = MagicMock()
        te2.device = torch.device("cpu")
        te2.dtype = torch.float32

        def te2_forward(input_ids, **kwargs):
            batch_size, seq_len = input_ids.shape
            mock_output = MagicMock()
            hidden = torch.randn(batch_size, seq_len, 1280)
            mock_output.hidden_states = [hidden, hidden, hidden]  # Fake layer outputs
            mock_output.text_embeds = torch.randn(batch_size, 1280)  # Pooled output
            return mock_output

        te2.side_effect = te2_forward

        return te1, te2

    @pytest.fixture
    def mock_tokenizers(self):
        """Create mock tokenizers for CLIP - return objects with .input_ids like HuggingFace."""

        def make_tokenizer():
            tokenizer = MagicMock()
            tokenizer.model_max_length = 77

            def tokenize_call(captions, **kwargs):
                batch_size = len(captions) if isinstance(captions, list) else 1
                result = MagicMock()
                result.input_ids = torch.zeros(batch_size, 77, dtype=torch.long)
                return result

            tokenizer.side_effect = tokenize_call
            return tokenizer

        return make_tokenizer(), make_tokenizer()

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_sdxl_te_caching_creates_files(self, mock_cfg, mock_text_encoders, mock_tokenizers, mock_accelerator):
        """Test SDXL TE caching creates proper cache files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            te1, te2 = mock_text_encoders
            tok1, tok2 = mock_tokenizers

            # Create manifest
            manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", cache_dir=str(cache_dir))

            # Create TE caching strategy
            strategy = SdxlTextEncoderPipelineStrategy(dtype="fp16", max_token_length=77)

            engine = CachingEngine(backend=strategy, batch_size=2)

            # Cache TE outputs - encode_batch expects 4-tuple: (te1, te2, tok1, tok2)
            updated_manifest = engine.cache_dataset(
                manifest=manifest,
                model=(te1, te2, tok1, tok2),
                accelerator=mock_accelerator,
                cache_dir=cache_dir,
                show_progress=False,
            )

            # Verify cache files created
            cache_files = list(cache_dir.glob("*.safetensors"))
            assert len(cache_files) == EXPECTED_IMAGE_COUNT, f"Expected {EXPECTED_IMAGE_COUNT} TE cache files, found {len(cache_files)}"

            # Verify entries updated (CachingEngine uses latent_cache_path for all strategies)
            for entry in updated_manifest.entries.values():
                assert entry.latent_cache_path is not None

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_te_cache_roundtrip(self, mock_cfg, mock_text_encoders, mock_tokenizers, mock_accelerator):
        """Test TE cache save → load roundtrip: verify loaded data has correct structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            te1, te2 = mock_text_encoders
            tok1, tok2 = mock_tokenizers

            manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", cache_dir=str(cache_dir))
            strategy = SdxlTextEncoderPipelineStrategy(dtype="fp16", max_token_length=77)

            engine = CachingEngine(backend=strategy, batch_size=2)
            # Pass 4-tuple: (te1, te2, tok1, tok2)
            manifest = engine.cache_dataset(manifest, (te1, te2, tok1, tok2), mock_accelerator, cache_dir, show_progress=False)

            # Load each TE cache and verify contents
            for entry in manifest.entries.values():
                # TE strategy uses te_cache_path, not latent_cache_path
                cache_path = Path(entry.te_cache_path)
                assert cache_path.exists(), f"TE cache file missing: {cache_path}"

                # Load from disk
                loaded = strategy.load_cache(cache_path)

                # Verify aux contains TE outputs
                assert loaded.aux is not None, "aux dict not loaded"
                assert "hidden_state1" in loaded.aux, "hidden_state1 missing"
                assert "hidden_state2" in loaded.aux, "hidden_state2 missing"
                assert "pool2" in loaded.aux, "pool2 missing"

                # Verify tensor shapes
                assert loaded.aux["hidden_state1"].ndim == 2, "hidden_state1 should be [seq_len, dim]"
                assert loaded.aux["hidden_state2"].ndim == 2, "hidden_state2 should be [seq_len, dim]"
                assert loaded.aux["pool2"].ndim == 1, "pool2 should be [dim]"


class TestResumeSupport:
    """Test resume batch skipping with itertools.islice."""

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_islice_skips_correct_batches(self, mock_cfg, mock_vae, mock_accelerator):
        """Test that islice correctly skips batches for resume."""
        import itertools

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            # Setup: create manifest and cache
            manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", cache_dir=str(cache_dir))
            strategy = SdxlLatentsPipelineStrategy(flip_aug=False, dtype="fp16")
            engine = CachingEngine(backend=strategy, batch_size=2)
            manifest = engine.cache_dataset(manifest, mock_vae, mock_accelerator, cache_dir, show_progress=False)

            # Create epoch with batch_size=1 so we get EXPECTED_IMAGE_COUNT batches
            caption_config = CaptionConfig()
            epoch_manifest = prepare_epoch(manifest, 0, 42, 1, caption_config)
            total_batches = len(epoch_manifest.batches)

            # Create dataloader
            dataloader = create_training_dataloader(
                dataset_manifest=manifest,
                epoch_manifest=epoch_manifest,
                latent_cache_backend=strategy,
                flip_aug=False,
                prior_loss_weight=1.0,
                rank=0,
                world_size=1,
                num_workers=0,  # Single-threaded for predictable behavior
            )

            # Skip first 2 batches using islice
            skip_count = 2
            dataloader_iter = iter(dataloader)
            skipped_iter = itertools.islice(dataloader_iter, skip_count, None)

            # Count remaining batches
            remaining_batches = list(skipped_iter)
            expected_remaining = total_batches - skip_count

            assert len(remaining_batches) == expected_remaining, (
                f"Expected {expected_remaining} batches after skipping {skip_count}, got {len(remaining_batches)}"
            )

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_skip_all_batches_yields_empty(self, mock_cfg, mock_vae, mock_accelerator):
        """Test skipping all batches yields nothing."""
        import itertools

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", cache_dir=str(cache_dir))
            strategy = SdxlLatentsPipelineStrategy(flip_aug=False, dtype="fp16")
            engine = CachingEngine(backend=strategy, batch_size=2)
            manifest = engine.cache_dataset(manifest, mock_vae, mock_accelerator, cache_dir, show_progress=False)

            caption_config = CaptionConfig()
            epoch_manifest = prepare_epoch(manifest, 0, 42, 1, caption_config)
            total_batches = len(epoch_manifest.batches)

            dataloader = create_training_dataloader(
                dataset_manifest=manifest,
                epoch_manifest=epoch_manifest,
                latent_cache_backend=strategy,
                flip_aug=False,
                prior_loss_weight=1.0,
                rank=0,
                world_size=1,
                num_workers=0,
            )

            # Skip ALL batches
            dataloader_iter = iter(dataloader)
            skipped_iter = itertools.islice(dataloader_iter, total_batches, None)

            remaining = list(skipped_iter)
            assert len(remaining) == 0, f"Expected 0 batches after skipping all, got {len(remaining)}"


class TestConfigIntegration:
    """Test config → behavior wiring."""

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_cache_dir_from_config_is_used(self, mock_cfg, mock_vae, mock_accelerator, tmp_path):
        """Test that cfg.data.caching.cache_dir is respected."""
        custom_cache_dir = tmp_path / "custom_cache_location"
        custom_cache_dir.mkdir()

        # Set cache_dir in config
        mock_cfg.data.caching.cache_dir = str(custom_cache_dir)

        manifest = create_manifest_from_config(mock_cfg.data, latent_dtype="fp16", cache_dir=str(custom_cache_dir))
        strategy = SdxlLatentsPipelineStrategy(flip_aug=False, dtype="fp16")
        engine = CachingEngine(backend=strategy, batch_size=2)

        # Use config cache_dir explicitly (as sdxl_peft.py does)
        engine.cache_dataset(
            manifest=manifest,
            model=mock_vae,
            accelerator=mock_accelerator,
            cache_dir=custom_cache_dir,  # Use the config value
            show_progress=False,
        )

        # Verify cache files are in custom_cache_location
        cache_files = list(custom_cache_dir.glob("*.safetensors"))
        assert len(cache_files) == EXPECTED_IMAGE_COUNT, (
            f"Expected {EXPECTED_IMAGE_COUNT} cache files in custom dir, found {len(cache_files)}"
        )
