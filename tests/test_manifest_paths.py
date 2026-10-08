# -*- coding: utf-8 -*-
"""Taking the machine's directory layout out of a shipped manifest.

The manifests are provenance records, and the manifests travel: they sit in the
submission package, and ``run_frozen_evaluation.py`` copies their ``sources``
block verbatim into ``results/manifest.json``. Recording the absolute path of
every input therefore publishes a research server's directory layout in a file
that was never meant to describe a machine -- while ``NOTICE.md`` promised the
opposite.

What makes an input checkable is its hash. These tests pin that the hash, the
byte count and every other field survive the rewrite untouched, and that the
rewrite is idempotent, because a "fix" that quietly dropped a hash would be
worse than the leak it removed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "tools" / "strip_manifest_paths.py"
    spec = importlib.util.spec_from_file_location("strip_manifest_paths", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strip_manifest_paths"] = module
    spec.loader.exec_module(module)
    return module


strip = _load()


# ---------------------------------------------------------------------------
# what counts as absolute
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [
    "/home/HLC/project/x.csv",
    "/mnt/data/HLC/qacd/mechanical_features/dev.csv",
    r"C:\Users\someone\secret\x.csv",
    "C:/Users/someone/secret/x.csv",
    r"\\server\share\x.csv",
])
def test_absolute_paths_are_recognised(value):
    assert strip.is_absolute(value)


@pytest.mark.parametrize("value", [
    "x.csv", "dev_features.csv.gz", "../sibling/x.csv", "", "relative/dir/x.csv",
])
def test_relative_names_are_left_alone(value):
    assert not strip.is_absolute(value)


# ---------------------------------------------------------------------------
# the rewrite
# ---------------------------------------------------------------------------

def test_a_path_becomes_a_file_and_keeps_its_hash():
    original = {
        "dev_features": {
            "path": "/home/HLC/project/hallucination_calibration/outputs/x/y.csv",
            "sha256": "abc123",
        }
    }
    cleaned, changed = strip.relativise(original)

    assert changed == 1
    assert cleaned["dev_features"] == {"file": "y.csv", "sha256": "abc123"}
    assert "path" not in cleaned["dev_features"], "the key no longer claims to be a path"


def test_every_other_field_survives():
    """Counts, parameters and shapes are the point of the manifest."""
    original = {
        "created_utc": "2026-09-24T15:36:50+00:00",
        "bundle_bytes": 852003,
        "protocol": {"seed": 20260920, "k": 5, "alphas": [0.05, 0.1]},
        "shapes": {"dev_matrix": [3025, 112]},
        "sources": {
            "dev": {
                "path": "/mnt/data/HLC/x/dev.csv",
                "sha256": "deadbeef",
                "rows": 3025,
                "target_columns_dropped": ["label"],
            }
        },
        "exports": {"dev.csv.gz": {"bytes": 139552, "sha256": "cafe"}},
    }
    cleaned, changed = strip.relativise(json.loads(json.dumps(original)))

    assert changed == 1
    assert cleaned["bundle_bytes"] == 852003
    assert cleaned["protocol"] == original["protocol"]
    assert cleaned["shapes"] == original["shapes"]
    assert cleaned["exports"] == original["exports"]
    assert cleaned["created_utc"] == original["created_utc"]
    assert cleaned["sources"]["dev"]["rows"] == 3025
    assert cleaned["sources"]["dev"]["target_columns_dropped"] == ["label"]
    assert cleaned["sources"]["dev"]["sha256"] == "deadbeef"


def test_a_relative_path_is_not_touched():
    """Nothing to hide, and rewriting it would lose information."""
    original = {"sources": {"a": {"path": "already/relative.csv", "sha256": "x"}}}
    cleaned, changed = strip.relativise(json.loads(json.dumps(original)))
    assert changed == 0
    assert cleaned == original


def test_running_it_twice_changes_nothing_the_second_time(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps({"sources": {"a": {"path": "/mnt/data/HLC/a.csv", "sha256": "h"}}}),
        encoding="utf-8",
    )
    assert strip.strip_file(path) == 1
    once = path.read_text(encoding="utf-8")
    assert strip.strip_file(path) == 0
    assert path.read_text(encoding="utf-8") == once


def test_check_mode_reports_without_writing(tmp_path):
    path = tmp_path / "m.json"
    original = {"sources": {"a": {"path": "/mnt/data/HLC/a.csv", "sha256": "h"}}}
    path.write_text(json.dumps(original), encoding="utf-8")

    assert strip.strip_file(path, check=True) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == original, "check must not write"


# ---------------------------------------------------------------------------
# the manifests that actually ship
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["artifacts/bundle_manifest.json",
                                  "artifacts/raw/raw_manifest.json"])
def test_the_shipped_manifests_hold_no_absolute_paths(name):
    """This is the regression. It failed for both files when it was written."""
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} is not present in this checkout")

    raw = path.read_text(encoding="utf-8")
    _, changed = strip.relativise(json.loads(raw))
    assert changed == 0, (
        f"{name} names {changed} absolute path(s). NOTICE.md says absolute paths "
        f"are omitted; run tools/strip_manifest_paths.py to make that true."
    )


def test_the_bundle_hash_matches_the_bundle_next_to_it():
    """The rewrite must not have disturbed the one hash consumers actually check."""
    manifest_path = ROOT / "artifacts" / "bundle_manifest.json"
    bundle_path = ROOT / "artifacts" / "evidence_bundle.npz"
    if not (manifest_path.is_file() and bundle_path.is_file()):
        pytest.skip("artifacts are not present in this checkout")

    import hashlib

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert digest == manifest["bundle_sha256"]
    assert bundle_path.stat().st_size == manifest["bundle_bytes"]
