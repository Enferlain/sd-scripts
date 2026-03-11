"""
Unit tests for library/training/phases/caching.py

Tests the caching phase functions with mocked trainer state.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.training
@pytest.mark.unit
class TestRunCaching:
    """Test run_caching orchestration function."""

    def test_calls_latent_and_te_caching(self, mock_trainer):
        """Test that run_caching calls both sub-functions."""
        with (
            patch("library.training.phases.caching.run_latent_caching") as mock_latent,
            patch("library.training.phases.caching.run_te_caching") as mock_te,
        ):
            from library.training.phases.caching import run_caching

            run_caching(mock_trainer)

            mock_latent.assert_called_once_with(mock_trainer)
            mock_te.assert_called_once_with(mock_trainer)

    def test_offloads_text_encoders_when_not_cached(self, mock_trainer):
        """Test TE offloading when cache_text_encoder_outputs=False and offload_text_encoders=True."""
        mock_trainer.cfg.data.caching.cache_text_encoder_outputs = False
        mock_trainer.cfg.performance.memory.offload_text_encoders = True

        with (
            patch("library.training.phases.caching.run_latent_caching"),
            patch("library.training.phases.caching.run_te_caching"),
            patch("library.training.phases.caching.clean_memory_on_device"),
        ):
            from library.training.phases.caching import run_caching

            run_caching(mock_trainer)

            # Each text encoder should be moved to CPU
            for te in mock_trainer.text_encoders:
                te.to.assert_called_with("cpu")

    def test_skips_offload_when_te_cached(self, mock_trainer):
        """Test TE offloading is skipped when TE outputs are cached."""
        mock_trainer.cfg.data.caching.cache_text_encoder_outputs = True
        mock_trainer.cfg.performance.memory.offload_text_encoders = True

        with patch("library.training.phases.caching.run_latent_caching"), patch("library.training.phases.caching.run_te_caching"):
            from library.training.phases.caching import run_caching

            # Reset call counts
            for te in mock_trainer.text_encoders:
                te.to.reset_mock()

            run_caching(mock_trainer)

            # Text encoders should NOT be moved to CPU here (handled in run_te_caching)
            for te in mock_trainer.text_encoders:
                # Check that .to("cpu") was not called in run_caching
                cpu_calls = [call for call in te.to.call_args_list if call.args == ("cpu",)]
                assert len(cpu_calls) == 0


@pytest.mark.training
@pytest.mark.unit
class TestRunLatentCaching:
    """Test run_latent_caching function."""

    def test_skips_when_disabled(self, mock_trainer):
        """Test early return when cache_latents=False."""
        mock_trainer.cfg.data.caching.cache_latents = False

        from library.training.phases.caching import run_latent_caching

        run_latent_caching(mock_trainer)

        # VAE should not be touched
        mock_trainer.vae.to.assert_not_called()

    def test_creates_latent_cache_handler_via_strategies(self, mock_trainer):
        """Test that latent strategy is created via trainer.strategies.create_latent_caching_strategy."""
        mock_strategy = MagicMock()
        mock_trainer.strategies.create_latent_caching_strategy.return_value = mock_strategy

        with (
            patch("library.training.phases.caching.CachingEngine") as MockEngine,
            patch("library.training.phases.caching.clean_memory_on_device"),
        ):
            mock_engine = MagicMock()
            mock_engine.cache_dataset = MagicMock(return_value=mock_trainer.train_manifest)
            MockEngine.return_value = mock_engine

            from library.training.phases.caching import run_latent_caching

            run_latent_caching(mock_trainer)

            mock_trainer.strategies.create_latent_caching_strategy.assert_called_once_with(mock_trainer.cfg)
            assert mock_trainer.latent_cache_handler == mock_strategy

    def test_moves_vae_to_device_and_back(self, mock_trainer):
        """Test VAE is moved to device for caching, then back to CPU."""
        mock_trainer.strategies.create_latent_caching_strategy.return_value = MagicMock()

        with (
            patch("library.training.phases.caching.CachingEngine") as MockEngine,
            patch("library.training.phases.caching.clean_memory_on_device"),
        ):
            mock_engine = MagicMock()
            mock_engine.cache_dataset = MagicMock(return_value=mock_trainer.train_manifest)
            MockEngine.return_value = mock_engine

            from library.training.phases.caching import run_latent_caching

            run_latent_caching(mock_trainer)

            # VAE should be moved to accelerator device first
            calls = mock_trainer.vae.to.call_args_list
            assert len(calls) >= 2
            # Last call should be "cpu"
            assert calls[-1].args[0] == "cpu"

    def test_emits_resource_monitor_phase_hooks(self, mock_trainer):
        """Latent caching should emit phase_start/phase_end hooks."""
        mock_trainer.strategies.create_latent_caching_strategy.return_value = MagicMock()

        with (
            patch("library.training.phases.caching.CachingEngine") as MockEngine,
            patch("library.training.phases.caching.clean_memory_on_device"),
        ):
            mock_engine = MagicMock()
            mock_engine.cache_dataset = MagicMock(return_value=mock_trainer.train_manifest)
            MockEngine.return_value = mock_engine

            from library.training.phases.caching import run_latent_caching

            run_latent_caching(mock_trainer)

            mock_trainer._resource_monitor.phase_start.assert_called_once_with("latent_caching")
            mock_trainer._resource_monitor.phase_end.assert_called_once_with("latent_caching")


@pytest.mark.training
@pytest.mark.unit
class TestRunTECaching:
    """Test run_te_caching function."""

    def test_skips_when_disabled(self, mock_trainer):
        """Test early return when cache_text_encoder_outputs=False."""
        mock_trainer.cfg.data.caching.cache_text_encoder_outputs = False

        from library.training.phases.caching import run_te_caching

        # Reset mocks
        for te in mock_trainer.text_encoders:
            te.to.reset_mock()

        run_te_caching(mock_trainer)

        # Text encoders should not be touched
        for te in mock_trainer.text_encoders:
            te.to.assert_not_called()

    def test_disk_mode_uses_caching_engine(self, mock_trainer):
        """Test disk-based TE caching uses CachingEngine via strategy factory."""
        mock_trainer.cfg.data.caching.cache_text_encoder_outputs_to_disk = True
        mock_te_cache_handler = MagicMock()
        mock_trainer.strategies.create_te_caching_strategy.return_value = mock_te_cache_handler

        with (
            patch("library.training.phases.caching.CachingEngine") as MockEngine,
            patch("library.training.phases.caching.clean_memory_on_device"),
        ):
            mock_engine = MagicMock()
            mock_engine.cache_dataset = MagicMock(return_value=mock_trainer.train_manifest)
            MockEngine.return_value = mock_engine

            from library.training.phases.caching import run_te_caching

            run_te_caching(mock_trainer)

            mock_trainer.strategies.create_te_caching_strategy.assert_called_once_with(mock_trainer.cfg)
            mock_trainer.strategies.build_te_cache_model_bundle.assert_called_once_with(
                mock_trainer.cfg,
                mock_trainer.accelerator,
                mock_trainer.text_encoders,
                mock_trainer.tokenizers,
            )
            MockEngine.assert_called_once()
            mock_engine.cache_dataset.assert_called()

            first_call = mock_engine.cache_dataset.call_args_list[0]
            assert first_call.kwargs["model"] == mock_trainer.strategies.build_te_cache_model_bundle.return_value

    def test_moves_text_encoders_to_cpu_after(self, mock_trainer):
        """Test text encoders are moved to CPU after caching."""
        mock_trainer.cfg.data.caching.cache_text_encoder_outputs_to_disk = True
        mock_trainer.strategies.create_te_caching_strategy.return_value = MagicMock()

        with (
            patch("library.training.phases.caching.CachingEngine") as MockEngine,
            patch("library.training.phases.caching.clean_memory_on_device"),
        ):
            mock_engine = MagicMock()
            mock_engine.cache_dataset = MagicMock(return_value=mock_trainer.train_manifest)
            MockEngine.return_value = mock_engine

            from library.training.phases.caching import run_te_caching

            run_te_caching(mock_trainer)

            # Each TE should have final call to("cpu")
            for te in mock_trainer.text_encoders:
                calls = te.to.call_args_list
                assert calls[-1].args[0] == "cpu"

    def test_emits_resource_monitor_phase_hooks(self, mock_trainer):
        """TE caching should emit phase_start/phase_end hooks."""
        mock_trainer.cfg.data.caching.cache_text_encoder_outputs_to_disk = True
        mock_trainer.strategies.create_te_caching_strategy.return_value = MagicMock()

        with (
            patch("library.training.phases.caching.CachingEngine") as MockEngine,
            patch("library.training.phases.caching.clean_memory_on_device"),
        ):
            mock_engine = MagicMock()
            mock_engine.cache_dataset = MagicMock(return_value=mock_trainer.train_manifest)
            MockEngine.return_value = mock_engine

            from library.training.phases.caching import run_te_caching

            run_te_caching(mock_trainer)

            mock_trainer._resource_monitor.phase_start.assert_called_with("te_caching")
            mock_trainer._resource_monitor.phase_end.assert_called_with("te_caching")
