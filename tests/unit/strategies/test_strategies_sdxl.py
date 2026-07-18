"""Unit tests for the active SDXL strategy concern files."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from library.config.dataclasses.timestep import TimestepConfig
from library.losses.loss_modifiers import NoOpLossModifier
from library.objectives.ddpm import DDPMObjectiveRuntime
from library.objectives.rectified_flow import RectifiedFlowObjectiveRuntime
from library.strategies.base.context import StrategyContext, StrategyPhase, current_strategy_context
from library.strategies.sdxl.checkpointing import SdxlCheckpointingStrategy
from library.strategies.sdxl.denoiser import SdxlDenoiserCallingStrategy
from library.strategies.sdxl.diffusion import SdxlDiffusionTrainingStrategy, build_sdxl_flow_target
from library.strategies.sdxl.encoding import SdxlTextEncodingStrategy
from library.strategies.sdxl.tokenization import SdxlTokenizeStrategy
from library.strategies.sdxl.training import SdxlTrainingStrategy
from library.strategies.sdxl.validation import SdxlValidationStrategy


class _TestSdxlDenoiserStrategy(SdxlDiffusionTrainingStrategy, SdxlDenoiserCallingStrategy):
    pass


@pytest.mark.unit
def test_full_model_checkpoint_uses_centrally_projected_metadata_unchanged(tmp_path):
    strategy = SdxlCheckpointingStrategy()
    strategy.ckpt_info = object()
    strategy.logit_scale = object()
    metadata = {
        "kuro.schema_version": "1",
        "modelspec.title": "Typed Full Model",
        "ss_adapter_module": "lora",
    }
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            output=SimpleNamespace(
                saving=SimpleNamespace(
                    save_model_as="safetensors",
                    output_dir=str(tmp_path),
                ),
                huggingface=None,
            ),
            model=SimpleNamespace(pretrained_model_name_or_path="source"),
        ),
        denoiser=object(),
        text_encoders=[object(), object()],
        vae=object(),
        accelerator=Mock(unwrap_model=lambda model: model),
    )

    with (
        patch("library.models.sdxl.conversion.save_stable_diffusion_checkpoint") as save_checkpoint,
        patch("library.strategies.sdxl.checkpointing.get_model_metadata_from_config") as legacy_builder,
    ):
        strategy.save_model_checkpoint(
            trainer,
            ckpt_name="model.safetensors",
            step=12,
            epoch=3,
            metadata=metadata,
            save_dtype=torch.float16,
        )

    assert save_checkpoint.call_args.args[9] is metadata
    legacy_builder.assert_not_called()


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_clip_tokenizer1():
    """Create a mock CLIPTokenizer for SDXL tokenizer 1 (CLIP-L)."""
    tokenizer = Mock()
    tokenizer.model_max_length = 77
    tokenizer.bos_token_id = 49406
    tokenizer.eos_token_id = 49407
    tokenizer.pad_token_id = 49407
    tokenizer.eos_token = 49407

    def tokenizer_call(text, **kwargs):
        if not text.strip():
            return Mock(input_ids=[49406, 49407] + [49407] * 75)
        return Mock(input_ids=[49406] + [100] * 5 + [49407] + [49407] * 70)

    tokenizer.__call__ = tokenizer_call
    tokenizer.side_effect = tokenizer_call
    return tokenizer


@pytest.fixture
def mock_clip_tokenizer2():
    """Create a mock CLIPTokenizer for SDXL tokenizer 2 (OpenCLIP)."""
    tokenizer = Mock()
    tokenizer.model_max_length = 77
    tokenizer.bos_token_id = 49406
    tokenizer.eos_token_id = 49407
    tokenizer.pad_token_id = 0  # SDXL uses 0 for tokenizer2
    tokenizer.eos_token = 49407

    def tokenizer_call(text, **kwargs):
        if not text.strip():
            return Mock(input_ids=[49406, 49407] + [0] * 75)
        return Mock(input_ids=[49406] + [200] * 5 + [49407] + [0] * 70)

    tokenizer.__call__ = tokenizer_call
    tokenizer.side_effect = tokenizer_call
    return tokenizer


@pytest.fixture
def mock_clip_text_encoder1():
    """Create a mock CLIP-L text encoder (CLIPTextModel)."""
    encoder = Mock()
    encoder.device = torch.device("cpu")

    # Mock parameters() to return an iterator with a tensor that has .device
    mock_param = Mock()
    mock_param.device = torch.device("cpu")
    encoder.parameters = Mock(return_value=iter([mock_param]))

    def encoder_call(tokens, output_hidden_states=False, return_dict=False):
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]
        hidden_dim = 768  # CLIP-L hidden dim

        hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(13)]
        return {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states[-1],
        }

    encoder.__call__ = encoder_call
    encoder.side_effect = encoder_call
    return encoder


@pytest.fixture
def mock_clip_text_encoder2():
    """Create a mock OpenCLIP text encoder (CLIPTextModelWithProjection)."""
    encoder = Mock()
    encoder.device = torch.device("cpu")

    # Mock parameters() to return an iterator with a tensor that has .device
    mock_param = Mock()
    mock_param.device = torch.device("cpu")
    encoder.parameters = Mock(return_value=iter([mock_param]))

    # Mock text_projection for pool workaround
    encoder.text_projection = Mock()
    encoder.text_projection.weight = Mock(dtype=torch.float32)
    encoder.text_projection.side_effect = lambda x: x

    def encoder_call(tokens, output_hidden_states=False, return_dict=False):
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]
        hidden_dim = 1280  # OpenCLIP hidden dim

        hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(13)]
        return {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states[-1],
            "text_embeds": torch.randn(batch_size, 1280),
        }

    encoder.__call__ = encoder_call
    encoder.side_effect = encoder_call
    return encoder


@pytest.fixture
def sdxl_strategy_cfg():
    """Create a minimal SDXL strategy config mock."""
    cfg = Mock()
    cfg.training.max_token_length = 75
    cfg.data.caching.tokenizer_cache_dir = None
    return cfg


# =============================================================================
# SdxlTokenizeStrategy Tests
# =============================================================================


@pytest.mark.unit
class TestSdxlTokenizeStrategy:
    """Test SdxlTokenizeStrategy with mocked dual tokenizers."""

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_init_loads_dual_tokenizers(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test initialization loads both tokenizers."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]

        strategy = SdxlTokenizeStrategy(max_length=None)

        assert strategy.tokenizer1 is mock_clip_tokenizer1
        assert strategy.tokenizer2 is mock_clip_tokenizer2
        assert mock_load_tokenizer.call_count == 2

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_init_sets_tokenizer2_pad_to_zero(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test that strategy keeps the loader-provided tokenizer2."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]

        strategy = SdxlTokenizeStrategy(max_length=None)

        assert strategy.tokenizer2.pad_token_id == 0

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_init_custom_max_length(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test custom max_length adds 2 for BOS/EOS."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]

        strategy = SdxlTokenizeStrategy(max_length=150)

        assert strategy.max_length == 152

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_tokenize_returns_tuple_of_two_tensors(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test tokenize returns tuple of two token tensors."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        strategy = SdxlTokenizeStrategy(max_length=None)

        with patch("library.strategies.sdxl.tokenization.get_clip_input_ids") as mock_get_ids:
            mock_get_ids.return_value = torch.randint(0, 1000, (1, 77))

            result = strategy.tokenize("a photo of a cat")

            assert isinstance(result, list)
            assert len(result) == 2
            assert mock_get_ids.call_count == 2  # Called for both tokenizers

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_tokenize_with_weights_returns_dual_tokens_and_weights(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test tokenize_with_weights returns tokens and weights for both tokenizers."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        strategy = SdxlTokenizeStrategy(max_length=None)

        with patch("library.strategies.sdxl.tokenization.get_clip_input_ids") as mock_get_ids:
            mock_get_ids.return_value = (torch.randint(0, 1000, (1, 77)), torch.ones(1, 77))

            tokens_list, weights_list = strategy.tokenize_with_weights("(emphasized:1.5)")

            assert len(tokens_list) == 2
            assert len(weights_list) == 2


# =============================================================================
# SdxlTextEncodingStrategy Tests
# =============================================================================


@pytest.mark.unit
class TestSdxlTextEncodingStrategy:
    """Test SdxlTextEncodingStrategy with mocked dual text encoders."""

    def test_init(self):
        """Test initialization."""
        strategy = SdxlTextEncodingStrategy([Mock(), Mock()])
        # No-op init, just verify it doesn't error
        assert strategy is not None

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_encode_tokens_returns_three_outputs(
        self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2
    ):
        """Test encode_tokens returns [hidden1, hidden2, pool2]."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        tokenize_strategy = SdxlTokenizeStrategy(max_length=None)
        encoding_strategy = SdxlTextEncodingStrategy([tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2])

        with patch.object(tokenize_strategy, "tokenize") as mock_tokenize:
            mock_tokenize.return_value = [torch.randint(0, 1000, (1, 1, 77)), torch.randint(0, 1000, (1, 1, 77))]
            result = encoding_strategy.encode_tokens([mock_clip_text_encoder1, mock_clip_text_encoder2], list(mock_tokenize.return_value))

        assert len(result) == 3  # hidden1, hidden2, pool2

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_encode_tokens_with_unwrapped_encoder(
        self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2
    ):
        """Test encode_tokens with 3 models (wrapped encoder2 case)."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        tokenize_strategy = SdxlTokenizeStrategy(max_length=None)
        encoding_strategy = SdxlTextEncodingStrategy([tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2])

        with patch.object(tokenize_strategy, "tokenize") as mock_tokenize:
            mock_tokenize.return_value = [torch.randint(0, 1000, (1, 1, 77)), torch.randint(0, 1000, (1, 1, 77))]

            # Pass 3 models: encoder1, encoder2, unwrapped_encoder2
            result = encoding_strategy.encode_tokens(
                [mock_clip_text_encoder1, mock_clip_text_encoder2, mock_clip_text_encoder2],
                list(mock_tokenize.return_value),
            )

        assert len(result) == 3

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_encode_tokens_with_weights_applies_dual_weights(
        self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2
    ):
        """Test that encode_tokens_with_weights applies weights to both encoders."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        tokenize_strategy = SdxlTokenizeStrategy(max_length=None)
        encoding_strategy = SdxlTextEncodingStrategy([tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2])

        with patch.object(tokenize_strategy, "tokenize_with_weights") as mock_tokenize:
            tokens = [torch.randint(0, 1000, (1, 1, 77)), torch.randint(0, 1000, (1, 1, 77))]
            weights = [torch.ones(1, 1, 77) * 1.5, torch.ones(1, 1, 77) * 1.5]
            mock_tokenize.return_value = (tokens, weights)

            result = encoding_strategy.encode_tokens_with_weights([mock_clip_text_encoder1, mock_clip_text_encoder2], tokens, weights)

        assert len(result) == 3


@pytest.mark.unit
def test_sdxl_process_batch_resolves_conditioning_with_keywords() -> None:
    strategy = SdxlDiffusionTrainingStrategy()
    strategy.vae_latent_scale = 1.0
    strategy.resolve_conditioning = Mock(return_value=("cond1", "cond2", "pool"))
    strategy.get_noise_pred_and_target = Mock(
        return_value=(
            torch.zeros(1, 4, 8, 8),
            torch.zeros(1, 4, 8, 8),
            torch.zeros(1, dtype=torch.long),
            None,
        )
    )

    cfg = SimpleNamespace(
        data=SimpleNamespace(caching=SimpleNamespace()),
        loss=SimpleNamespace(
            loss_type="l2",
            loss_scale=1.0,
            loss_multiplier=None,
            masked=SimpleNamespace(masked_loss=False),
            huber=SimpleNamespace(schedule="constant", c=0.1),
        ),
    )
    batch = {"loss_weights": torch.ones(1)}
    text_encoders = [Mock(), Mock()]
    objective_runtime = DDPMObjectiveRuntime(
        name="ddpm",
        num_train_timesteps=1000,
        timestep_runtime=None,
        loss_modifier=NoOpLossModifier(),
        noise_scheduler=Mock(),
        alphas_cumprod=torch.ones(1000),
    )
    accelerator = Mock()
    accelerator.device = torch.device("cpu")

    with patch("library.strategies.sdxl.diffusion.prepare_latents", return_value=torch.zeros(1, 4, 8, 8)):
        strategy.process_batch(
            batch=batch,
            text_encoders=text_encoders,
            unet=Mock(),
            trainable_model=Mock(),
            vae=Mock(),
            objective_runtime=objective_runtime,
            vae_dtype=torch.float32,
            weight_dtype=torch.float32,
            accelerator=accelerator,
            cfg=cfg,
            is_train=False,
        )

    strategy.resolve_conditioning.assert_called_once_with(
        batch=batch,
        text_encoders=text_encoders,
        accelerator=accelerator,
        cfg=cfg,
        train_text_encoder=True,
        is_train=False,
        weight_dtype=torch.float32,
    )


@pytest.mark.unit
def test_sdxl_validation_resolves_conditioning_with_keywords() -> None:
    strategy = SdxlValidationStrategy()
    strategy.vae_latent_scale = 1.0
    strategy.resolve_conditioning = Mock(return_value=("cond1", "cond2", "pool"))
    strategy.get_noise_pred_and_target = Mock(
        return_value=(
            torch.zeros(1, 4, 8, 8),
            torch.zeros(1, 4, 8, 8),
            torch.zeros(1, dtype=torch.long),
            None,
        )
    )

    cfg = SimpleNamespace(data=SimpleNamespace(caching=SimpleNamespace()))
    batch = {}
    text_encoders = [Mock(), Mock()]
    accelerator = Mock()
    accelerator.device = torch.device("cpu")

    with patch("library.strategies.sdxl.validation.prepare_latents", return_value=torch.zeros(1, 4, 8, 8)):
        strategy.process_val_batch(
            batch=batch,
            text_encoders=text_encoders,
            unet=Mock(),
            trainable_model=Mock(),
            vae=Mock(),
            objective_runtime=Mock(),
            vae_dtype=torch.float32,
            weight_dtype=torch.float32,
            accelerator=accelerator,
            cfg=cfg,
            train_text_encoder=False,
            train_denoiser=True,
            timesteps_list=[50],
        )

    strategy.resolve_conditioning.assert_called_once_with(
        batch=batch,
        text_encoders=text_encoders,
        accelerator=accelerator,
        cfg=cfg,
        train_text_encoder=False,
        is_train=False,
        weight_dtype=torch.float32,
    )


@pytest.mark.unit
def test_sdxl_flow_target_matches_direct_velocity_direction() -> None:
    latents = torch.tensor([[[[1.0]]], [[[2.0]]]])
    noise = torch.tensor([[[[4.0]]], [[[7.0]]]])

    target = build_sdxl_flow_target(latents, noise)

    assert torch.equal(target, noise - latents)


@pytest.mark.unit
def test_sdxl_get_noise_pred_and_target_rectified_flow_uses_rf_target_and_weighting() -> None:
    strategy = SdxlDiffusionTrainingStrategy()
    latents = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ]
    )
    noise = torch.tensor(
        [
            [[[8.0, 7.0], [6.0, 5.0]]],
            [[[4.0, 3.0], [2.0, 1.0]]],
        ]
    )
    weighting = torch.full((2, 1, 1, 1), 2.5)
    timesteps = torch.tensor([100, 200], dtype=torch.long)
    noisy_model_input = torch.zeros_like(latents)
    noise_pred = torch.ones_like(latents)

    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sdxl"),
        objective=SimpleNamespace(prediction="flow"),
        performance=SimpleNamespace(memory=SimpleNamespace(gradient_checkpointing=False)),
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=lambda: nullcontext())
    objective_runtime = RectifiedFlowObjectiveRuntime(
        name="rectified_flow",
        num_train_timesteps=1000,
        timestep_runtime=None,
        loss_modifier=NoOpLossModifier(),
        timestep_config=TimestepConfig(),
        loss_weighting_scheme="none",
    )
    objective_runtime.build_training_batch_state = Mock(
        return_value=SimpleNamespace(
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            sigmas=torch.full((2, 1, 1, 1), 0.5),
            loss_weighting=weighting,
        )
    )
    strategy.call_denoiser = Mock(return_value=noise_pred)

    result_noise_pred, target, result_timesteps, result_weighting = strategy.get_noise_pred_and_target(
        cfg=cfg,
        accelerator=accelerator,
        objective_runtime=objective_runtime,
        latents=latents,
        batch={},
        text_encoder_conds=("cond1", "cond2", "pool"),
        unet=Mock(),
        trainable_model=Mock(),
        weight_dtype=torch.float32,
        train_denoiser=True,
        is_train=False,
    )

    assert torch.equal(result_noise_pred, noise_pred)
    assert torch.equal(target, noise - latents)
    assert torch.equal(result_timesteps, timesteps)
    assert torch.equal(result_weighting, weighting)


@pytest.mark.unit
def test_sdxl_get_noise_pred_and_target_publishes_denoiser_forward_contexts() -> None:
    observed_contexts: list[StrategyContext] = []
    latents = torch.zeros(3, 1, 2, 2)
    noise = torch.ones_like(latents)
    timesteps = torch.tensor([100, 200, 300], dtype=torch.long)
    noisy_model_input = torch.full_like(latents, 0.5)

    class _RecordingDenoiser:
        def __call__(self, model_input, denoiser_timesteps, text_embedding, vector_embedding):
            del denoiser_timesteps, text_embedding, vector_embedding
            context = current_strategy_context()
            assert context is not None
            observed_contexts.append(context)
            return torch.ones_like(model_input)

    strategy = _TestSdxlDenoiserStrategy()

    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sdxl"),
        objective=SimpleNamespace(prediction="flow"),
        performance=SimpleNamespace(memory=SimpleNamespace(gradient_checkpointing=False)),
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=lambda: nullcontext())
    objective_runtime = RectifiedFlowObjectiveRuntime(
        name="rectified_flow",
        num_train_timesteps=1000,
        timestep_runtime=None,
        loss_modifier=NoOpLossModifier(),
        timestep_config=TimestepConfig(),
        loss_weighting_scheme="none",
    )
    objective_runtime.build_training_batch_state = Mock(
        return_value=SimpleNamespace(
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            sigmas=torch.full((3, 1, 1, 1), 0.5),
            loss_weighting=None,
        )
    )

    trainable_model = SimpleNamespace(set_multiplier=Mock())
    batch = {
        "conditionings": [
            SimpleNamespace(original_size_hw=(1024, 1024), crop_top_left=(0, 0), target_size_hw=(1024, 1024)),
            SimpleNamespace(original_size_hw=(1024, 1024), crop_top_left=(0, 0), target_size_hw=(1024, 1024)),
            SimpleNamespace(original_size_hw=(1024, 1024), crop_top_left=(0, 0), target_size_hw=(1024, 1024)),
        ],
        "custom_attributes": [
            {"diff_output_preservation": True},
            {},
            {"diff_output_preservation": True},
        ],
    }

    strategy.get_noise_pred_and_target(
        cfg=cfg,
        accelerator=accelerator,
        objective_runtime=objective_runtime,
        latents=latents,
        batch=batch,
        text_encoder_conds=(
            torch.ones(3, 77, 1280),
            torch.ones(3, 77, 1280),
            torch.ones(3, 1280),
        ),
        unet=_RecordingDenoiser(),
        trainable_model=trainable_model,
        weight_dtype=torch.float32,
        train_denoiser=True,
        is_train=True,
        global_step=42,
    )

    assert len(observed_contexts) == 2
    normal_context, indexed_context = observed_contexts
    assert normal_context.phase is StrategyPhase.TRAIN
    assert normal_context.model_family == "sdxl"
    assert normal_context.training is not None
    assert normal_context.training.global_step == 42
    assert normal_context.training.is_train is True
    assert normal_context.denoiser is not None
    assert torch.equal(normal_context.denoiser.timesteps, timesteps)
    assert normal_context.denoiser.sample_indices is None
    assert normal_context.denoiser.batch_size == 3

    assert indexed_context.phase is StrategyPhase.TRAIN
    assert indexed_context.training is not None
    assert indexed_context.training.global_step == 42
    assert indexed_context.training.is_train is True
    assert indexed_context.denoiser is not None
    assert torch.equal(indexed_context.denoiser.timesteps, torch.tensor([100, 300]))
    assert indexed_context.denoiser.sample_indices == (0, 2)
    assert indexed_context.denoiser.batch_size == 2
    assert current_strategy_context() is None


@pytest.mark.unit
def test_sdxl_checkpoint_metadata_omits_ddpm_prediction_for_rf() -> None:
    strategy = SdxlCheckpointingStrategy()
    cfg = SimpleNamespace(
        objective=SimpleNamespace(path="rectified_flow", prediction="flow"),
        output=SimpleNamespace(metadata=SimpleNamespace()),
        data=SimpleNamespace(preprocessing=SimpleNamespace(resolution=1024)),
        timestep=SimpleNamespace(min_timestep=None, max_timestep=None),
        training=SimpleNamespace(clip_skip=None),
    )

    with patch("library.strategies.sdxl.checkpointing.get_model_metadata_from_config", return_value={}) as mock_metadata:
        strategy.get_model_metadata(cfg)

    assert mock_metadata.call_args.kwargs["prediction_type"] is None
    assert mock_metadata.call_args.kwargs["v_parameterization"] is False


@pytest.mark.unit
class TestSdxlTrainingStrategyComposition:
    """Tests for SDXL training-strategy composition across concern files."""

    def test_training_strategy_uses_split_concern_modules(self):
        """TrainingStrategy should inherit facet methods from the split SDXL files."""
        assert SdxlTrainingStrategy.tokenize_captions.__module__ == "library.strategies.sdxl.tokenization"
        assert SdxlTrainingStrategy.encode_te_outputs_in_memory.__module__ == "library.strategies.sdxl.encoding"
        assert SdxlTrainingStrategy.create_te_caching_strategy.__module__ == "library.strategies.sdxl.caching"
        assert SdxlTrainingStrategy.resolve_conditioning.__module__ == "library.strategies.sdxl.conditioning"

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_tokenize_captions_returns_dual_chunked_tensors(
        self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2, sdxl_strategy_cfg
    ):
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        strategy = SdxlTrainingStrategy(sdxl_strategy_cfg)

        result = strategy.tokenize_captions([mock_clip_tokenizer1, mock_clip_tokenizer2], ["caption 1", "caption 2"], 75)

        assert len(result) == 2
        assert all(isinstance(tokens, torch.Tensor) for tokens in result)
        assert all(tokens.shape[0] == 2 for tokens in result)
        assert all(tokens.ndim == 3 for tokens in result)

    def test_encode_te_outputs_in_memory_returns_cpu_te_outputs(
        self, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2, sdxl_strategy_cfg
    ):
        with patch("library.strategies.sdxl.tokenization.load_tokenizer") as mock_load_tokenizer:
            mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
            strategy = SdxlTrainingStrategy(sdxl_strategy_cfg)

            with patch("library.strategies.sdxl.encoding.encode_input_ids_sdxl") as mock_encode:
                mock_encode.return_value = (
                    torch.randn(1, 77, 768),
                    torch.randn(1, 77, 1280),
                    torch.randn(1, 1280),
                )

                result = strategy.encode_te_outputs_in_memory(
                    text_encoders=[mock_clip_text_encoder1, mock_clip_text_encoder2],
                    tokenizers=[mock_clip_tokenizer1, mock_clip_tokenizer2],
                    caption="a photo of a cat",
                    max_token_length=75,
                    device=torch.device("cpu"),
                )

        assert set(result) == {"hidden_state1", "hidden_state2", "pool2"}
        assert all(t.device.type == "cpu" for t in result.values())

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_get_text_cond_fallback_tokenizes_captions_with_strategy_helper(
        self,
        mock_load_tokenizer,
        mock_clip_tokenizer1,
        mock_clip_tokenizer2,
        mock_clip_text_encoder1,
        mock_clip_text_encoder2,
        sdxl_strategy_cfg,
    ):
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        strategy = SdxlTrainingStrategy(sdxl_strategy_cfg)
        cfg = Mock()
        cfg.data.caption.weighted_captions = False
        cfg.performance.precision.full_fp16 = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")
        batch = {"captions": ["caption 1", "caption 2"]}

        token_tensors = [torch.randint(0, 1000, (2, 1, 77)), torch.randint(0, 1000, (2, 1, 77))]
        encoded_outputs = [torch.randn(2, 77, 768), torch.randn(2, 77, 1280), torch.randn(2, 1280)]
        models = [mock_clip_text_encoder1, mock_clip_text_encoder2, mock_clip_text_encoder2]

        with (
            patch.object(strategy, "tokenize", return_value=token_tensors) as mock_tokenize,
            patch.object(strategy, "get_models_for_text_encoding", return_value=models) as mock_get_models,
            patch.object(strategy, "encode_tokens", return_value=encoded_outputs) as mock_encode,
        ):
            result = strategy.resolve_conditioning(
                cfg=cfg,
                accelerator=accelerator,
                batch=batch,
                text_encoders=[mock_clip_text_encoder1, mock_clip_text_encoder2],
                weight_dtype=torch.float32,
                train_text_encoder=False,
                is_train=False,
            )

        mock_tokenize.assert_called_once_with(batch["captions"])
        mock_get_models.assert_called_once_with(cfg, accelerator, [mock_clip_text_encoder1, mock_clip_text_encoder2])
        mock_encode.assert_called_once_with(models, token_tensors)
        assert len(result) == 3

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_resolve_conditioning_weighted_captions_use_strategy_weight_helpers(
        self,
        mock_load_tokenizer,
        mock_clip_tokenizer1,
        mock_clip_tokenizer2,
        mock_clip_text_encoder1,
        mock_clip_text_encoder2,
        sdxl_strategy_cfg,
    ):
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        strategy = SdxlTrainingStrategy(sdxl_strategy_cfg)
        cfg = Mock()
        cfg.data.caption.weighted_captions = True
        cfg.performance.precision.full_fp16 = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")
        batch = {"captions": ["(caption:1.2)"]}

        token_tensors = [torch.randint(0, 1000, (1, 1, 77)), torch.randint(0, 1000, (1, 1, 77))]
        weight_tensors = [torch.ones(1, 1, 77) * 1.2, torch.ones(1, 1, 77) * 1.2]
        encoded_outputs = [torch.randn(1, 77, 768), torch.randn(1, 77, 1280), torch.randn(1, 1280)]
        models = [mock_clip_text_encoder1, mock_clip_text_encoder2, mock_clip_text_encoder2]

        with (
            patch.object(strategy, "tokenize_with_weights", return_value=(token_tensors, weight_tensors)) as mock_tokenize,
            patch.object(strategy, "get_models_for_text_encoding", return_value=models) as mock_get_models,
            patch.object(strategy, "encode_tokens_with_weights", return_value=encoded_outputs) as mock_encode,
            patch.object(strategy, "encode_tokens") as mock_plain_encode,
        ):
            result = strategy.resolve_conditioning(
                cfg=cfg,
                accelerator=accelerator,
                batch=batch,
                text_encoders=[mock_clip_text_encoder1, mock_clip_text_encoder2],
                weight_dtype=torch.float32,
                train_text_encoder=False,
                is_train=False,
            )

        mock_tokenize.assert_called_once_with(batch["captions"])
        mock_get_models.assert_called_once_with(cfg, accelerator, [mock_clip_text_encoder1, mock_clip_text_encoder2])
        mock_encode.assert_called_once_with(models, token_tensors, weight_tensors)
        mock_plain_encode.assert_not_called()
        assert len(result) == 3

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_resolve_conditioning_uses_cached_outputs_without_live_reencode(
        self,
        mock_load_tokenizer,
        mock_clip_tokenizer1,
        mock_clip_tokenizer2,
        mock_clip_text_encoder1,
        mock_clip_text_encoder2,
        sdxl_strategy_cfg,
    ):
        """Cached TE outputs should be returned directly when live re-encoding is unnecessary."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        strategy = SdxlTrainingStrategy(sdxl_strategy_cfg)
        cfg = Mock()
        cfg.data.caption.weighted_captions = False
        cfg.performance.precision.full_fp16 = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")
        batch = {
            "text_encoder_outputs": {
                "hidden_state1": torch.randn(2, 77, 768),
                "hidden_state2": torch.randn(2, 77, 1280),
                "pool2": torch.randn(2, 1280),
            }
        }

        with (
            patch.object(strategy, "encode_tokens") as mock_encode,
            patch.object(strategy, "encode_tokens_with_weights") as mock_weighted_encode,
        ):
            result = strategy.resolve_conditioning(
                cfg=cfg,
                accelerator=accelerator,
                batch=batch,
                text_encoders=[mock_clip_text_encoder1, mock_clip_text_encoder2],
                weight_dtype=torch.float32,
                train_text_encoder=False,
                is_train=False,
            )

        mock_encode.assert_not_called()
        mock_weighted_encode.assert_not_called()
        assert len(result) == 3
