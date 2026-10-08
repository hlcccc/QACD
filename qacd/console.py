# -*- coding: utf-8 -*-
"""Make console output survive a terminal that cannot encode it.

A Windows console defaults to a legacy code page -- cp1252 on an en-US box,
cp936 on a Chinese one -- and printing a Chinese label there does not degrade,
it raises::

    UnicodeEncodeError: 'charmap' codec can't encode characters in position 6-8

Nothing is printed at all, and the process dies with a traceback. That is how
the Windows CI leg went red while all four Linux legs passed, on a script whose
only offence was `print("--- 技术点3 mechanical OCR channel ---")`, and it would
do the same to an integrator running the documented demo on an English Windows
machine.

Reconfiguring the stream with ``errors="replace"`` turns an unencodable glyph
into ``?`` and leaves everything else alone. On a UTF-8 terminal nothing changes.
It is deliberately not an encoding *change*: forcing UTF-8 onto a cp1252 console
would produce mojibake rather than an error, which is harder to notice.
"""

from __future__ import annotations

import sys
from typing import Any

__all__ = ["make_console_safe"]


def make_console_safe() -> None:
    """Stop ``print`` from raising on a console that cannot encode the text.

    Safe to call more than once, and safe when the streams are not reconfigurable
    (a ``StringIO`` under pytest, a pipe on an old interpreter): the call is a
    no-op rather than an error, because it exists to remove a crash, not add one.
    """
    for stream in (sys.stdout, sys.stderr):
        _reconfigure(stream)


def _reconfigure(stream: Any) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(errors="replace")
    except (TypeError, ValueError, OSError):
        # A capture object that takes no `errors`, a closed stream, or a detached
        # one. None of them is this function's problem to solve.
        pass
