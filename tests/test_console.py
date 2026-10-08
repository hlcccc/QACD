# -*- coding: utf-8 -*-
"""Printing Chinese on a console that cannot encode it.

`scripts/demo_offline.py` prints Chinese section labels. On a Windows console
whose code page is cp1252 -- the default on an English install, and on the
Windows CI runner -- `print` does not degrade, it raises::

    UnicodeEncodeError: 'charmap' codec can't encode characters in position 6-8

and the process dies with a traceback. Every Linux leg passed, so the suite
looked healthy while the Windows leg had been red for two commits.

The subprocess test at the bottom is the real regression: it runs the exact
command CI runs, with the exact encoding that broke it.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qacd.console import make_console_safe  # noqa: E402


# ---------------------------------------------------------------------------
# the helper itself
# ---------------------------------------------------------------------------

class _Reconfigurable(io.StringIO):
    def __init__(self):
        super().__init__()
        self.calls = []

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)


def test_it_asks_the_stream_to_replace_rather_than_raise(monkeypatch):
    out, err = _Reconfigurable(), _Reconfigurable()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    make_console_safe()

    assert out.calls == [{"errors": "replace"}]
    assert err.calls == [{"errors": "replace"}]


def test_it_does_not_change_the_encoding(monkeypatch):
    """Forcing UTF-8 onto a cp1252 console gives mojibake, which is worse than '?'.

    Mojibake is silent; `?` at least says something was dropped.
    """
    out = _Reconfigurable()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", _Reconfigurable())

    make_console_safe()

    assert "encoding" not in out.calls[0]


def test_a_stream_without_reconfigure_is_left_alone(monkeypatch):
    """pytest's capture objects and plain pipes do not all have it."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    make_console_safe()  # must not raise


def test_a_stream_whose_reconfigure_rejects_errors_is_left_alone(monkeypatch):
    class Stubborn(io.StringIO):
        def reconfigure(self, **kwargs):
            raise TypeError("reconfigure() got an unexpected keyword argument 'errors'")

    monkeypatch.setattr(sys, "stdout", Stubborn())
    monkeypatch.setattr(sys, "stderr", Stubborn())

    make_console_safe()  # must not raise


def test_calling_it_twice_is_fine(monkeypatch):
    out = _Reconfigurable()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", _Reconfigurable())

    make_console_safe()
    make_console_safe()

    assert len(out.calls) == 2


# ---------------------------------------------------------------------------
# the regression, as CI runs it
# ---------------------------------------------------------------------------

def _run(script: str, encoding: str):
    env = dict(os.environ, PYTHONIOENCODING=encoding)
    return subprocess.run(
        [sys.executable, script],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


@pytest.mark.parametrize("script", [
    "scripts/demo_offline.py",
    "examples/smoke_test_stub_model.py",
])
def test_the_documented_scripts_survive_a_cp1252_console(script):
    """The exact failure, reproduced and then asserted fixed.

    Run without the fix, both scripts exit 1 with a UnicodeEncodeError at their
    first Chinese label.
    """
    result = _run(script, "cp1252")

    assert result.returncode == 0, (
        f"{script} exited {result.returncode} under a cp1252 console\n"
        f"stderr tail:\n{result.stderr[-1500:]}"
    )
    assert "UnicodeEncodeError" not in result.stderr


def test_the_console_script_survives_a_cp1252_console():
    """`qacd demo` is the last Windows CI step, and it prints the same labels."""
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    result = subprocess.run(
        [sys.executable, "-m", "qacd.cli", "demo"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300,
    )

    assert result.returncode == 0, result.stderr[-1500:]
    assert "UnicodeEncodeError" not in result.stderr
