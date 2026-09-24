#!/usr/bin/env python
"""Export a verifiable evidence bundle for the QACD repository.

Runs on the research server. It reuses the *exact* loading and feature-assembly
functions of the frozen protocol script (imported, not re-implemented) so that
the exported matrices are identical to what produced the frozen scorers.

It performs no model inference: it only assembles already-cached evidence.
It writes nothing outside /mnt/data/HLC.

Output: /mnt/data/HLC/qacd_eval_bundle/
    evidence_bundle.npz    claim matrices, labels, splits, MVR frames
    bundle_manifest.json   SHA256 of every source input and of the bundle
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path("/home/HLC/project")
EXPERIMENTS = PROJECT / "hallucination_calibration" / "experiments"
PROTOCOL = EXPERIMENTS / "run_conformal_selective_prediction_strict.py"
OUT_DIR = Path("/mnt/data/HLC/qacd_eval_bundle")

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol():
    spec = importlib.util.spec_from_file_location("qacd_frozen_protocol", PROTOCOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {PROTOCOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["qacd_frozen_protocol"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    P = load_protocol()

    print("[1/6] assembling development claim frame ...")
    dev_frame, dev_hashes = P.load_feature_frame(P.DEV)
    print(f"      dev claims: {len(dev_frame)}  columns: {len(dev_frame.columns)}")

    print("[2/6] assembling frozen-test claim frame ...")
    test_frame, test_hashes = P.load_feature_frame(P.TEST)
    print(f"      test claims: {len(test_frame)}  columns: {len(test_frame.columns)}")

    print("[3/6] labels and image-disjoint split ...")
    dev_claim_labels, dev_ids, dev_y, dev_images, dev_claim_index = P.development_labels(
        P.DEV["claims"], dev_frame["unit_id"]
    )
    _, test_ids, test_y, test_images, test_claim_index = P.development_labels(
        P.TEST["claims"], test_frame["unit_id"]
    )
    score_train_images, calibration_images = P.image_split(dev_images)

    dev_claim_image = np.asarray(
        [
            str(dev_images[i]) if 0 <= i < len(dev_images) else ""
            for i in np.asarray(dev_claim_index, dtype=int)
        ],
        dtype=object,
    ).astype(str)
    claim_train_mask = np.array(
        [image in score_train_images for image in dev_claim_image], dtype=bool
    )
    if not claim_train_mask.any():
        raise RuntimeError("no claim rows fall in the score-training images")

    print(
        f"      dev responses {len(dev_ids)} | test responses {len(test_ids)} | "
        f"score-train images {len(score_train_images)} | calibration images {len(calibration_images)} | "
        f"train claim rows {int(claim_train_mask.sum())}/{len(dev_claim_image)}"
    )

    print("[4/6] feature selection + score-training-only imputation ...")
    dev_aligned = P.add_direction_aligned_features(dev_frame)
    test_aligned = P.add_direction_aligned_features(test_frame)
    lm_names, mech_names = P.select_feature_names(dev_aligned, test_aligned, claim_train_mask)
    all_names = lm_names + mech_names
    print(f"      LM features {len(lm_names)} | mechanical features {len(mech_names)}")

    medians = P.fit_imputer(dev_aligned, all_names, claim_train_mask)
    dev_matrix = P.apply_imputer(dev_aligned, all_names, medians)
    test_matrix = P.apply_imputer(test_aligned, all_names, medians)

    print("[5/7] MVR frames from the cached K=5 sampling ...")
    dev_mvr, test_mvr = P.load_mvr_frames(dev_ids, dev_images, test_ids, test_images)
    print(f"      dev MVR {dev_mvr.shape} | test MVR {test_mvr.shape}")

    print("[6/7] reference scores from the frozen replay (baselines + QACD arms) ...")
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    scores_dev = pd.read_csv(P.DEFAULT_OUT / "development_scores.csv")
    scores_test = pd.read_csv(P.DEFAULT_OUT / "test_scores_unlabeled.csv")

    reference_names = []
    reference_dev = []
    reference_test = []
    for method in sorted(set(scores_test["method"]) | set(scores_dev["method"])):
        d = scores_dev[scores_dev["method"] == method].drop_duplicates("sample_id")
        t = scores_test[scores_test["method"] == method].drop_duplicates("sample_id")
        if d.empty or t.empty:
            continue
        dv = d.set_index("sample_id")["score"].reindex(dev_ids).to_numpy(dtype=float)
        tv = t.set_index("sample_id")["score"].reindex(test_ids).to_numpy(dtype=float)
        if not (np.isfinite(dv).all() and np.isfinite(tv).all()):
            print(f"      skip {method}: alignment produced non-finite values")
            continue
        print(f"      {method:34s} test AUROC={roc_auc_score(test_y, tv):.4f}")
        reference_names.append(method)
        reference_dev.append(dv)
        reference_test.append(tv)

    print("[7/7] writing bundle ...")
    bundle_path = OUT_DIR / "evidence_bundle.npz"
    np.savez_compressed(
        bundle_path,
        dev_matrix=dev_matrix,
        test_matrix=test_matrix,
        dev_claim_labels=np.asarray(dev_claim_labels, dtype=np.int64),
        dev_claim_train_mask=np.asarray(claim_train_mask, dtype=bool),
        dev_claim_response_index=np.asarray(dev_claim_index, dtype=np.int64),
        dev_response_ids=np.asarray(dev_ids, dtype=np.int64),
        dev_response_y=np.asarray(dev_y, dtype=np.int64),
        dev_response_images=np.asarray(dev_images, dtype=object).astype(str),
        test_claim_response_index=np.asarray(test_claim_index, dtype=np.int64),
        test_response_ids=np.asarray(test_ids, dtype=np.int64),
        test_response_y=np.asarray(test_y, dtype=np.int64),
        test_response_images=np.asarray(test_images, dtype=object).astype(str),
        dev_mvr=dev_mvr.to_numpy(dtype=np.float64),
        test_mvr=test_mvr.to_numpy(dtype=np.float64),
        lm_names=np.asarray(lm_names, dtype=object).astype(str),
        mechanical_names=np.asarray(mech_names, dtype=object).astype(str),
        mvr_names=np.asarray(list(P.MVR_FEATURES), dtype=object).astype(str),
        imputer_medians=np.asarray([medians[n] for n in all_names], dtype=np.float64),
        score_train_images=np.asarray(sorted(score_train_images), dtype=object).astype(str),
        calibration_images=np.asarray(sorted(calibration_images), dtype=object).astype(str),
        seed=np.asarray([int(P.SEED)], dtype=np.int64),
        calibration_fraction=np.asarray([float(P.CALIBRATION_FRACTION)], dtype=np.float64),
        k=np.asarray([int(P.K)], dtype=np.int64),
        baseline_names=np.asarray(reference_names, dtype=object).astype(str),
        baseline_dev=np.asarray(reference_dev, dtype=np.float64),
        baseline_test=np.asarray(reference_test, dtype=np.float64),
        baseline_sign_flipped=np.asarray([False] * len(reference_names), dtype=bool),
    )

    sources = {}
    for split, config in (("dev", P.DEV), ("test", P.TEST)):
        for key, path in config.items():
            sources[f"{split}_{key}"] = {"path": str(path), "sha256": sha256_file(Path(path))}
    sources["sampling"] = {"path": str(P.SAMPLES), "sha256": sha256_file(Path(P.SAMPLES))}
    sources["protocol_script"] = {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)}

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": bundle_path.name,
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "protocol": {
            "seed": int(P.SEED),
            "calibration_fraction": float(P.CALIBRATION_FRACTION),
            "k": int(P.K),
            "alphas": [float(a) for a in P.ALPHAS],
            "feature_frame_hashes": {
                **dev_hashes,
                **{f"test_{k}": v for k, v in test_hashes.items()},
            },
        },
        "shapes": {
            "dev_matrix": list(dev_matrix.shape),
            "test_matrix": list(test_matrix.shape),
            "lm_features": len(lm_names),
            "mechanical_features": len(mech_names),
            "dev_responses": int(len(dev_ids)),
            "test_responses": int(len(test_ids)),
            "score_train_images": len(score_train_images),
            "calibration_images": len(calibration_images),
        },
        "sources": sources,
    }
    (OUT_DIR / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print(f"bundle   : {bundle_path}  ({manifest['bundle_bytes'] / 1e6:.1f} MB)")
    print(f"sha256   : {manifest['bundle_sha256']}")
    print(f"manifest : {OUT_DIR / 'bundle_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
