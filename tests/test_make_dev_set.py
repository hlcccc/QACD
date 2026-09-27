"""The development-set builder: labelling rule, normalisation, and grouping.

`examples/make_dev_set.py` is the step every integrator has to take before QACD
does anything useful -- fitting needs labels, and the labels come from their data.
The README documents the field list; these tests pin the rule that produces it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "make_dev_set", ROOT / "examples" / "make_dev_set.py"
)
assert _spec is not None and _spec.loader is not None
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def _answers(matches: int) -> list[str]:
    """Ten annotator answers, `matches` of which agree with the model."""
    return ["Dakota Digital"] * matches + ["Nikon Coolpix"] * (10 - matches)


# ---------------------------------------------------------------------------
# The VQA rule: min(matches / 3, 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "matches,expected_score",
    [(0, 0.0), (1, pytest.approx(1 / 3)), (2, pytest.approx(2 / 3)), (3, 1.0), (10, 1.0)],
)
def test_vqa_agreement_score(matches, expected_score):
    record = {"answers": _answers(matches)}
    _, score = builder.decide_label("Dakota Digital", record, threshold=0.5)
    assert score == expected_score


def test_three_agreeing_annotators_score_one():
    """The divisor is 3, so 3 of 10 is already full credit -- not 3 of 10."""
    _, score = builder.decide_label("Dakota Digital", {"answers": _answers(3)}, 0.5)
    assert score == 1.0


def test_threshold_decides_the_partial_band():
    """2 of 10 scores 0.667: correct at 0.5, wrong at 1.0."""
    record = {"answers": _answers(2)}
    assert builder.decide_label("Dakota Digital", record, threshold=0.5)[0] == 0
    assert builder.decide_label("Dakota Digital", record, threshold=1.0)[0] == 1


def test_a_single_gold_string_is_an_exact_match():
    assert builder.decide_label("blue", {"gold": "blue"}, 0.5)[0] == 0
    assert builder.decide_label("red", {"gold": "blue"}, 0.5)[0] == 1


@pytest.mark.parametrize(
    "model_answer",
    ["Blue", "BLUE", "  blue  ", "blue.", "the blue", "blue!"],
)
def test_normalisation_absorbs_case_punctuation_and_articles(model_answer):
    assert builder.decide_label(model_answer, {"gold": "blue"}, 0.5)[0] == 0


def test_a_record_with_neither_answers_nor_gold_is_rejected():
    with pytest.raises(ValueError):
        builder.decide_label("a", {"question": "q"}, 0.5)


# ---------------------------------------------------------------------------
# End to end, through main()
# ---------------------------------------------------------------------------

def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_group_is_the_image(tmp_path, monkeypatch):
    """Several questions about one photograph must share a group.

    The calibration split is taken by image; if group were the question, two
    questions about the same photograph could land on opposite sides of the split
    and the calibration set would contain a near-duplicate of the training set.
    """
    rows = [
        {"question": f"q{i}", "answer": "Dakota Digital", "image": f"/img{i % 3}.jpg",
         "answers": _answers(10)}
        for i in range(9)
    ]
    src = _write(tmp_path / "raw.jsonl", rows)
    out = tmp_path / "dev.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["make_dev_set.py", "--data", str(src), "--out", str(out)]
    )
    assert builder.main() == 0

    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 9
    assert {r["group"] for r in records} == {"/img0.jpg", "/img1.jpg", "/img2.jpg"}
    assert all(r["group"] == r["image"] for r in records)


def test_output_is_readable_by_the_cli_loader(tmp_path, monkeypatch):
    """The whole point: the file feeds `qacd fit` without further editing."""
    from qacd.cli import _load_records

    rows = [
        {"question": f"q{i}", "answer": "Dakota Digital",
         "image": f"/img{i}.jpg", "answers": _answers(10 if i % 2 else 0)}
        for i in range(6)
    ]
    src = _write(tmp_path / "raw.jsonl", rows)
    out = tmp_path / "dev.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["make_dev_set.py", "--data", str(src), "--out", str(out)]
    )
    assert builder.main() == 0

    records = _load_records(str(out))
    assert len(records) == 6
    assert {r["failed"] for r in records} == {0, 1}
    assert all(set(r) == {"question", "answer", "image", "failed", "group"} for r in records)


def test_missing_fields_name_the_line(tmp_path, monkeypatch):
    src = _write(tmp_path / "raw.jsonl", [{"question": "q", "answer": "a"}])
    monkeypatch.setattr(
        sys, "argv",
        ["make_dev_set.py", "--data", str(src), "--out", str(tmp_path / "d.jsonl")],
    )
    with pytest.raises(SystemExit) as excinfo:
        builder.main()
    assert ":1" in str(excinfo.value) and "image" in str(excinfo.value)


def test_a_single_class_dev_set_warns(tmp_path, monkeypatch, capsys):
    """A calibrator fitted on one class learns nothing; say so rather than fit."""
    rows = [
        {"question": f"q{i}", "answer": "Dakota Digital", "image": f"/img{i}.jpg",
         "gold": "Dakota Digital"}
        for i in range(5)
    ]
    src = _write(tmp_path / "raw.jsonl", rows)
    monkeypatch.setattr(
        sys, "argv",
        ["make_dev_set.py", "--data", str(src), "--out", str(tmp_path / "d.jsonl")],
    )
    assert builder.main() == 0
    assert "every answer was labelled correct" in capsys.readouterr().err


def test_the_utf8_bom_windows_writes_is_tolerated(tmp_path, monkeypatch):
    """PowerShell's `Out-File -Encoding utf8` prepends a BOM."""
    src = tmp_path / "raw.jsonl"
    src.write_text(
        json.dumps({"question": "q", "answer": "a", "image": "/i.jpg", "gold": "a"}) + "\n",
        encoding="utf-8-sig",
    )
    assert src.read_bytes()[:3] == b"\xef\xbb\xbf"
    out = tmp_path / "dev.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["make_dev_set.py", "--data", str(src), "--out", str(out)]
    )
    assert builder.main() == 0
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["failed"] == 0
