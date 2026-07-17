from __future__ import annotations

import signal
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from library.training.interrupts import install_double_ctrl_c_guard


def test_install_double_ctrl_c_guard_warns_on_first_sigint():
    trainer = SimpleNamespace(_console=MagicMock())

    with patch("library.training.interrupts.os.write") as write, install_double_ctrl_c_guard(trainer):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)

    write.assert_called_once_with(
        2,
        b"[interrupt] press Ctrl+C again within 5s to interrupt training\n",
    )
    trainer._console.print_external.assert_not_called()


def test_install_double_ctrl_c_guard_interrupts_on_second_sigint():
    trainer = SimpleNamespace(_console=MagicMock())
    previous_handler = signal.getsignal(signal.SIGINT)

    with patch("library.training.interrupts.os.write") as write, install_double_ctrl_c_guard(trainer):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
        assert signal.getsignal(signal.SIGINT) is previous_handler

    assert write.call_count == 2
    assert b"interruption accepted; shutting down cleanly" in write.call_args_list[1].args[1]


def test_install_double_ctrl_c_guard_resets_after_cooldown():
    trainer = SimpleNamespace(_console=MagicMock())

    with (
        patch("library.training.interrupts.time.monotonic", side_effect=[100.0, 106.0]),
        patch("library.training.interrupts.os.write") as write,
        install_double_ctrl_c_guard(trainer),
    ):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)

    assert write.call_count == 2


def test_install_double_ctrl_c_guard_falls_back_to_stderr_when_direct_write_fails(capsys):
    trainer = SimpleNamespace(_console=None)

    with (
        patch("library.training.interrupts.os.write", side_effect=OSError("closed")),
        install_double_ctrl_c_guard(trainer),
    ):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)

    captured = capsys.readouterr()
    assert "[interrupt] press Ctrl+C again within 5s to interrupt training" in captured.err
