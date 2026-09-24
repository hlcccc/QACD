"""Load and verify the frozen evidence bundle.

The bundle is the *input* to the scoring stage: cached model evidence assembled
by the research pipeline, together with per-claim labels, the image-disjoint
split and the MVR stability frames. It contains no fitted parameters — the
calibrators are fitted here, by this repository's code.

Two rules the loader enforces:

1. **Nothing is silently defaulted.** A missing array, a shape mismatch or a
   non-finite value raises. (The reference pipeline used to zero-fill missing
   features; that hid problems instead of surfacing them.)
2. **Everything is hashed.** The bundle's SHA256 is checked against the manifest
   before any array is used, so a report can state exactly which evidence it was
   computed from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

__all__ = ["EvidenceBundle", "load_bundle", "sha256_file"]

REQUIRED_ARRAYS = (
    "dev_matrix",
    "test_matrix",
    "dev_claim_labels",
    "dev_claim_train_mask",
    "dev_claim_response_index",
    "dev_response_ids",
    "dev_response_y",
    "dev_response_images",
    "test_claim_response_index",
    "test_response_ids",
    "test_response_y",
    "test_response_images",
    "dev_mvr",
    "test_mvr",
    "lm_names",
    "mechanical_names",
    "mvr_names",
    "baseline_names",
    "baseline_dev",
    "baseline_test",
    "seed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class EvidenceBundle:
    """Validated contents of ``evidence_bundle.npz`` plus its manifest."""

    arrays: Dict[str, np.ndarray]
    manifest: Dict[str, object]
    path: Path

    # -- convenience accessors ------------------------------------------
    @property
    def lm_names(self) -> List[str]:
        return [str(x) for x in self.arrays["lm_names"]]

    @property
    def mechanical_names(self) -> List[str]:
        return [str(x) for x in self.arrays["mechanical_names"]]

    @property
    def mvr_names(self) -> List[str]:
        return [str(x) for x in self.arrays["mvr_names"]]

    @property
    def baseline_names(self) -> List[str]:
        return [str(x) for x in self.arrays["baseline_names"]]

    @property
    def feature_names(self) -> List[str]:
        return self.lm_names + self.mechanical_names

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def select(self, names: List[str]) -> np.ndarray:
        """Column subset of the dev/test matrices, in the requested order."""
        index = {name: i for i, name in enumerate(self.feature_names)}
        missing = [name for name in names if name not in index]
        if missing:
            raise KeyError(f"bundle does not contain features: {missing[:5]}")
        columns = [index[name] for name in names]
        return columns

    def describe(self) -> str:
        shapes = self.manifest.get("shapes", {})
        return (
            f"bundle {self.path.name} sha256={str(self.manifest.get('bundle_sha256'))[:16]}... "
            f"dev_matrix={shapes.get('dev_matrix')} test_matrix={shapes.get('test_matrix')} "
            f"LM={shapes.get('lm_features')} mech={shapes.get('mechanical_features')}"
        )


def load_bundle(path: str | Path, manifest_path: str | Path | None = None, verify: bool = True) -> EvidenceBundle:
    """Load the evidence bundle and verify it against its manifest."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"evidence bundle not found: {path}\n"
            "Export it from the research host with tools/export_bundle.py, or pass --bundle."
        )
    manifest_path = Path(manifest_path) if manifest_path else path.with_name("bundle_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify:
        expected = manifest.get("bundle_sha256")
        actual = sha256_file(path)
        if expected and actual != expected:
            raise ValueError(
                f"bundle hash mismatch: manifest says {expected}, file is {actual}"
            )
        manifest["_verified_sha256"] = actual

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    missing = [key for key in REQUIRED_ARRAYS if key not in arrays]
    if missing:
        raise ValueError(f"bundle is missing required arrays: {missing}")

    bundle = EvidenceBundle(arrays=arrays, manifest=manifest, path=path)
    _validate(bundle)
    return bundle


def _validate(bundle: EvidenceBundle) -> None:
    a = bundle.arrays
    n_lm, n_mech = len(a["lm_names"]), len(a["mechanical_names"])
    n_features = n_lm + n_mech
    if a["dev_matrix"].shape[1] != n_features or a["test_matrix"].shape[1] != n_features:
        raise ValueError(
            f"feature count mismatch: matrices have {a['dev_matrix'].shape[1]}/"
            f"{a['test_matrix'].shape[1]} columns but names describe {n_features}"
        )
    for key in ("dev_matrix", "test_matrix", "dev_mvr", "test_mvr"):
        if not np.isfinite(a[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if a["dev_matrix"].shape[0] != a["dev_claim_labels"].shape[0]:
        raise ValueError("dev_matrix and dev_claim_labels disagree on claim count")
    if a["dev_matrix"].shape[0] != a["dev_claim_train_mask"].shape[0]:
        raise ValueError("dev_matrix and dev_claim_train_mask disagree on claim count")
    if a["test_matrix"].shape[0] != a["test_claim_response_index"].shape[0]:
        raise ValueError("test_matrix and test_claim_response_index disagree on claim count")
    if a["dev_response_y"].shape[0] != a["dev_mvr"].shape[0]:
        raise ValueError("dev responses and dev MVR rows disagree")
    if a["test_response_y"].shape[0] != a["test_mvr"].shape[0]:
        raise ValueError("test responses and test MVR rows disagree")
    if not a["dev_claim_train_mask"].any():
        raise ValueError("no claim rows are marked for score-training")
    if a["baseline_dev"].shape[1] != a["dev_response_y"].shape[0]:
        raise ValueError("baseline_dev is not aligned to the development responses")
    if a["baseline_test"].shape[1] != a["test_response_y"].shape[0]:
        raise ValueError("baseline_test is not aligned to the test responses")
