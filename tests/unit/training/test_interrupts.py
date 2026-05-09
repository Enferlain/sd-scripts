from __future__ import annotations

import signal
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from library.training.interrupts import install_double_ctrl_c_guard


def test_install_double_ctrl_c_guard_warns_on_first_sigint():
    trainer = SimpleNamespace(_console=MagicMock())

    with install_double_ctrl_c_guard(trainer):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)

    trainer._console.print_external.assert_called_once_with(
        "[interrupt] press Ctrl+C again within 5s to interrupt training"
    )


def test_install_double_ctrl_c_guard_interrupts_on_second_sigint():
    trainer = SimpleNamespace(_console=MagicMock())

    with install_double_ctrl_c_guard(trainer):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)


def test_install_double_ctrl_c_guard_resets_after_cooldown():
    trainer = SimpleNamespace(_console=MagicMock())

    with (
        patch("library.training.interrupts.time.monotonic", side_effect=[100.0, 106.0]),
        install_double_ctrl_c_guard(trainer),
    ):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)

    assert trainer._console.print_external.call_count == 2


def test_install_double_ctrl_c_guard_falls_back_to_stderr_without_console(capsys):
    trainer = SimpleNamespace(_console=None)

    with install_double_ctrl_c_guard(trainer):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)

    captured = capsys.readouterr()
    assert "[interrupt] press Ctrl+C again within 5s to interrupt training" in captured.err
