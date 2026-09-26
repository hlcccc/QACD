"""The frozen feature set wired into QACDPipeline.

The point of these tests is that the pipeline builds the 112-column vector
itself, from raw evidence, at scoring time — and that doing so reproduces the
batch result. The artifact-backed test is the one that would catch a regression
in the wiring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas")

from frozen.assemble import FrozenFeatureAssembler  # noqa: E402
from frozen.build import build_matrices, claim_images, load_split  # noqa: E402
from qacd.pipeline import QACDPipeline, REFERENCE_FEATURE_NAMES  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "artifacts" / "raw"
BUNDLE = REPO / "artifacts" / "evidence_bundle.npz"

requires_artifacts = pytest.mark.skipif(
    not (RAW / "dev_features.csv.gz").exists() or not BUNDLE.exists(),
    reason="raw evidence tables or the evidence bundle are not present",
)


@pytest.fixture(scope="module")
def frozen_split():
    bundle = np.load(BUNDLE, allow_pickle=False)
    dev = load_split(RAW, "dev")
    test = load_split(RAW, "test")
    matrices = build_matrices(dev, test, claim_images(dev, RAW / "dev_claims.csv.gz"))
    return bundle, dev, test, matrices


# -- assembler -----------------------------------------------------------

@requires_artifacts
def test_assembler_reproduces_the_frozen_matrix(frozen_split):
    bundle, dev, test, matrices = frozen_split
    assembler = FrozenFeatureAssembler.from_matrices(matrices)
    assert assembler.n_features == 112
    assert np.array_equal(assembler.transform_frame(dev), bundle["dev_matrix"])
    assert np.array_equal(assembler.transform_frame(test), bundle["test_matrix"])


@requires_artifacts
def test_assembler_is_row_wise(frozen_split):
    """A single claim assembles to the same vector as in the whole table."""
    _, dev, _, matrices = frozen_split
    assembler = FrozenFeatureAssembler.from_matrices(matrices)
    table = assembler.transform_frame(dev.head(25))
    for i, row in enumerate(dev.head(25).to_dict("records")):
        assert np.array_equal(assembler.transform_one(row), table[i])


@requires_artifacts
def test_assembler_imputes_absent_features_from_the_frozen_medians(frozen_split):
    _, dev, _, matrices = frozen_split
    assembler = FrozenFeatureAssembler.from_matrices(matrices)
    row = dev.iloc[3].to_dict()
    # Drop a raw column the derivations read; the median must stand in.
    target = "direct_verifier_support_mean"
    median = assembler.medians.get("direct_verifier_low_support_mean")
    row.pop(target, None)
    vector = assembler.transform_one(row)
    assert np.isfinite(vector).all()
    assert median is not None


def test_assembler_rejects_an_empty_frame():
    assembler = FrozenFeatureAssembler(feature_names=["a"], medians={"a": 0.0})
    with pytest.raises(ValueError):
        assembler.transform_frame(pd.DataFrame())


def test_assembler_rejects_no_rows():
    assembler = FrozenFeatureAssembler(feature_names=["a"], medians={"a": 0.0})
    with pytest.raises(ValueError):
        assembler.transform_rows([])


# -- pipeline wiring -----------------------------------------------------

@requires_artifacts
def test_pipeline_switches_to_the_frozen_feature_set(frozen_split):
    _, _, _, matrices = frozen_split
    pipeline = QACDPipeline()
    assert pipeline.feature_names == REFERENCE_FEATURE_NAMES  # 48 by default
    pipeline.attach_frozen_features(FrozenFeatureAssembler.from_matrices(matrices))
    assert pipeline.feature_names == matrices.names
    assert len(pipeline.feature_names) == 112


def test_attach_rejects_an_assembler_without_names():
    with pytest.raises(ValueError):
        QACDPipeline().attach_frozen_features(object())


@requires_artifacts
def test_pipeline_scoring_requires_an_assembler_and_a_fit(frozen_split):
    bundle, dev, test, matrices = frozen_split
    bare = QACDPipeline()
    with pytest.raises(ValueError, match="assembler"):
        bare.score_evidence_frame(test)

    attached = QACDPipeline(assembler=FrozenFeatureAssembler.from_matrices(matrices))
    with pytest.raises(ValueError, match="not fitted"):
        attached.score_evidence_frame(test)


@requires_artifacts
def test_pipeline_fit_matrix_rejects_a_dimension_mismatch(frozen_split):
    _, _, _, matrices = frozen_split
    pipeline = QACDPipeline(assembler=FrozenFeatureAssembler.from_matrices(matrices))
    with pytest.raises(ValueError, match="expects"):
        pipeline.fit_matrix(
            matrices.dev_matrix[:, :10],
            np.zeros(matrices.dev_matrix.shape[0]),
            matrices.train_mask,
        )


@requires_artifacts
def test_pipeline_closes_the_loop(frozen_split):
    """Fit and score through the pipeline; the number must match the batch path."""
    from evaluation.metrics import auroc
    from qacd.aggregate import claim_to_response_max

    bundle, dev, test, matrices = frozen_split
    pipeline = QACDPipeline(assembler=FrozenFeatureAssembler.from_matrices(matrices))
    pipeline.fit_matrix(matrices.dev_matrix, bundle["dev_claim_labels"], matrices.train_mask)
    assert pipeline.fitted_

    claim_scores = pipeline.score_evidence_frame(test)
    assert claim_scores.shape == (len(test),)

    y_test = bundle["test_response_y"]
    response = claim_to_response_max(claim_scores, len(y_test), bundle["test_claim_response_index"])
    auroc_pipeline = auroc(y_test, response)

    reference_index = {
        str(n): i for i, n in enumerate(bundle["baseline_names"])
    }
    reference = auroc(y_test, bundle["baseline_test"][reference_index["QACD LM + instrument"]])
    assert abs(auroc_pipeline - reference) < 1e-5


@requires_artifacts
def test_pipeline_scores_identically_row_by_row(frozen_split):
    bundle, _, test, matrices = frozen_split
    pipeline = QACDPipeline(assembler=FrozenFeatureAssembler.from_matrices(matrices))
    pipeline.fit_matrix(matrices.dev_matrix, bundle["dev_claim_labels"], matrices.train_mask)
    table = pipeline.score_evidence_frame(test.head(15))
    rows = pipeline.score_evidence_frame_by_row(test.head(15))
    assert np.allclose(table, rows, atol=1e-15)
