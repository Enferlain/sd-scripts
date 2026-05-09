from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from library.logging.console import MainProcessConsole


def test_log_block_renders_directly_to_console(capsys):
    logger = MagicMock()
    console = MainProcessConsole(is_main_process=True, logger=logger)

    console.log_block("training run\n  mode: AdapterMode")

    captured = capsys.readouterr()
    assert "training run" in captured.err
    logger.log.assert_not_called()


def test_log_uses_logger_with_tag():
    logger = MagicMock()
    console = MainProcessConsole(is_main_process=True, logger=logger)

    console.log("prepared epoch 0", tag="epoch", stacklevel=5)

    logger.log.assert_called_once()
    level, message = logger.log.call_args.args[:2]
    assert level > 0
    assert message == "[epoch] prepared epoch 0"
    assert logger.log.call_args.kwargs["stacklevel"] == 5


def test_log_startup_summary_renders_summary_block(capsys):
    logger = MagicMock()
    console = MainProcessConsole(is_main_process=True, logger=logger)
    summary = SimpleNamespace(
        runtime_rows=[("mode", "AdapterMode")],
        dataset_rows=[],
        schedule_rows=[],
        component_rows=[],
        optimizer_name="AdamW8bit",
        optimizer_rows=[],
        aliases=[],
    )

    console.log_startup_summary(summary)

    captured = capsys.readouterr()
    assert "training run" in captured.err
    assert "mode: AdapterMode" in captured.err
    logger.log.assert_not_called()


def test_log_external_uses_logger_inside_external_write_mode():
    logger = MagicMock()
    console = MainProcessConsole(is_main_process=True, logger=logger)

    with patch("library.logging.console.tqdm.external_write_mode", return_value=nullcontext()) as mock_external:
        console.log_external("prepared epoch 0", tag="epoch", stacklevel=5)

    mock_external.assert_called_once()
    logger.log.assert_called_once()
    level, message = logger.log.call_args.args[:2]
    assert level > 0
    assert message == "[epoch] prepared epoch 0"
    assert logger.log.call_args.kwargs["stacklevel"] == 6


def test_print_external_uses_external_write_mode(capsys):
    logger = MagicMock()
    console = MainProcessConsole(is_main_process=True, logger=logger)

    with patch("library.logging.console.tqdm.external_write_mode", return_value=nullcontext()) as mock_external:
        console.print_external("Epoch 1/1")

    captured = capsys.readouterr()
    assert "Epoch 1/1" in captured.err
    mock_external.assert_called_once()
    logger.log.assert_not_called()
