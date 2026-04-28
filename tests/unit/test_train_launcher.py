"""Unit tests for the active config-driven train entrypoint and its factories."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from library.training.modes import FineTuneMode, AdapterMode
from library.training.modes.factory import build_training_mode


def test_train_module_imports() -> None:
    """The root train launcher should import cleanly."""
    import train

    assert train is not None
    assert callable(train.train)
    assert callable(train.main)


def test_build_training_mode_peft() -> None:
    """mode=adapter should build the PEFT mode."""
    cfg = SimpleNamespace(mode="adapter")

    mode = build_training_mode(cfg)

    assert isinstance(mode, AdapterMode)


def test_build_training_mode_finetune() -> None:
    """mode=finetune should build the fine-tune mode."""
    cfg = SimpleNamespace(mode="finetune")

    mode = build_training_mode(cfg)

    assert isinstance(mode, FineTuneMode)


def test_build_training_mode_textual_inversion_not_implemented() -> None:
    """The generic launcher should fail fast for inactive textual inversion mode."""
    cfg = SimpleNamespace(mode="textual_inversion")

    with pytest.raises(NotImplementedError, match="textual_inversion"):
        build_training_mode(cfg)


@patch("library.strategies.sd.tokenization.load_tokenizer")
def test_build_training_strategy_sd(mock_load_tokenizer) -> None:
    """SD-family model types should resolve to the shared SD strategy."""
    from library.strategies.factory import build_training_strategy
    from library.strategies.sd.training import SdTrainingStrategy

    mock_load_tokenizer.return_value = Mock(model_max_length=77)
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sd15"),
        data=SimpleNamespace(caching=SimpleNamespace(tokenizer_cache_dir=None)),
        training=SimpleNamespace(max_token_length=75, clip_skip=None),
    )

    strategy = build_training_strategy(cfg)

    assert isinstance(strategy, SdTrainingStrategy)


@patch("library.strategies.sdxl.tokenization.load_tokenizer")
def test_build_training_strategy_sdxl(mock_load_tokenizer) -> None:
    """SDXL model type should resolve to the SDXL strategy."""
    from library.strategies.factory import build_training_strategy
    from library.strategies.sdxl.training import SdxlTrainingStrategy

    mock_load_tokenizer.side_effect = [Mock(model_max_length=77), Mock(model_max_length=77, pad_token_id=0)]
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sdxl"),
        data=SimpleNamespace(caching=SimpleNamespace(tokenizer_cache_dir=None)),
        training=SimpleNamespace(max_token_length=75),
    )

    strategy = build_training_strategy(cfg)

    assert isinstance(strategy, SdxlTrainingStrategy)


def test_build_training_strategy_unknown_model_type_raises() -> None:
    """Unknown model types should fail fast with a clear error."""
    from library.strategies.factory import build_training_strategy

    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="flux"),
        data=SimpleNamespace(caching=SimpleNamespace(tokenizer_cache_dir=None)),
        training=SimpleNamespace(max_token_length=75, clip_skip=None),
    )

    with pytest.raises(ValueError, match="Unsupported model\\.model_type"):
        build_training_strategy(cfg)


@patch("train.Trainer")
@patch("train.build_training_mode")
@patch("train.build_training_strategy")
@patch("train.validate_config")
@patch("train.prepare_config")
def test_train_wires_mode_strategy_and_trainer(
    mock_prepare_config,
    mock_validate_config,
    mock_build_training_strategy,
    mock_build_training_mode,
    mock_trainer_cls,
) -> None:
    """train() should prepare config, build objects, and start the trainer."""
    import train

    cfg = SimpleNamespace(mode="adapter", model=SimpleNamespace(model_type="sdxl"))
    strategies = Mock()
    mode = Mock()
    trainer = mock_trainer_cls.return_value
    mock_build_training_strategy.return_value = strategies
    mock_build_training_mode.return_value = mode

    train.train(cfg)

    mock_prepare_config.assert_called_once_with(cfg)
    mock_validate_config.assert_called_once_with(cfg)
    mock_build_training_strategy.assert_called_once_with(cfg)
    mock_build_training_mode.assert_called_once_with(cfg)
    mock_trainer_cls.assert_called_once_with(cfg, strategies, mode)
    trainer.train.assert_called_once_with()
