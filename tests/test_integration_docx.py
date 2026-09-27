"""The integration table must never advertise a field the service does not return.

An earlier revision of the 出参 column promised ``calibrated_confidence`` while
``POST /v1/qacd/risk`` returned no such key at all -- the string was hardcoded in
the generator and had silently drifted from the code. ``response_signature()``
now reads the field list out of ``ResponseRisk`` and fails the build on drift.
These tests pin both halves of that guarantee.

The generator needs python-docx, which is not part of the documented quick start
(``pip install -r requirements.txt``), so the whole module skips without it --
before importing it, so a NumPy-only install degrades to a clean skip rather than
a collection error.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("docx", reason="the integration-table generator needs python-docx")

_spec = importlib.util.spec_from_file_location(
    "make_integration_docx", ROOT / "scripts" / "make_integration_docx.py"
)
assert _spec is not None and _spec.loader is not None
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)


def _sample_response() -> dict:
    from qacd.types import ResponseRisk

    return ResponseRisk(
        risk_score=0.25, is_high_risk=False, threshold=0.5, num_claims=1
    ).to_dict()


def test_every_advertised_field_really_exists():
    signature = generator.response_signature()
    actual = _sample_response()
    for name in generator.RESPONSE_FIELD_ORDER:
        assert f'"{name}"' in signature
        assert name in actual, f"{name} is advertised but the service does not return it"


def test_calibrated_confidence_is_advertised_and_present():
    """The field whose absence motivated this guard."""
    assert "calibrated_confidence" in generator.RESPONSE_FIELD_ORDER
    assert "calibrated_confidence" in _sample_response()
    assert '"calibrated_confidence"' in generator.response_signature()


def test_drift_is_a_hard_failure(monkeypatch):
    """A name the code does not produce must abort, not be written out quietly."""
    monkeypatch.setattr(
        generator, "RESPONSE_FIELD_ORDER", ["risk_score", "not_a_real_field"]
    )
    with pytest.raises(SystemExit) as excinfo:
        generator.response_signature()
    assert "not_a_real_field" in str(excinfo.value)


def test_the_default_owner_is_rejected_as_a_placeholder():
    """An empty 对接负责人 is the most likely reason for the table to come back."""
    assert any(marker in generator.OWNER for marker in generator.PLACEHOLDER_MARKERS)


def test_topic2_row_is_consistent_with_the_module_contract():
    row = generator.TOPIC2_ROWS[0]
    # Ten values for columns 2..11; 序号 (col 0) is already in the template and
    # 所属子课题 (col 1) is prepended by the writer.
    assert len(row) == 10
    assert row[3] == generator.response_signature()
    assert row[-1] == generator.OWNER
