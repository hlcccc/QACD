"""Aggregation and conformal selection."""

import numpy as np
import pytest

from qacd.aggregate import claim_to_response_max, response_risk
from qacd.conformal import (
    VALIDATED,
    benjamini_hochberg,
    benjamini_yekutieli,
    conformal_pvalues,
    conformal_select,
)


# -- aggregation ---------------------------------------------------------

def test_max_operator_picks_the_worst_claim():
    scores = [0.1, 0.9, 0.2, 0.3]
    index = [0, 0, 1, 1]
    out = claim_to_response_max(scores, 2, index)
    assert out.tolist() == [0.9, 0.3]


def test_response_without_claims_gets_the_default():
    out = claim_to_response_max([0.5], 3, [0], default=0.0)
    assert out.tolist() == [0.5, 0.0, 0.0]


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        claim_to_response_max([0.1, 0.2], 1, [0])


def test_out_of_range_index_is_rejected():
    with pytest.raises(ValueError):
        claim_to_response_max([0.1], 1, [5])


def test_response_risk_thresholds():
    scores, flags = response_risk([0.2, 0.8], [0, 0], 1, threshold=0.5)
    assert scores.tolist() == [0.8]
    assert flags.tolist() == [True]


def test_empty_input():
    assert claim_to_response_max([], 0, []).shape == (0,)


# -- conformal -----------------------------------------------------------

def test_p_values_are_in_the_half_open_unit_interval():
    cal = [0.1, 0.2, 0.3, 0.4, 0.5]
    p = conformal_pvalues(cal, [0.0, 0.25, 0.9, 1.0])
    # p = 1 exactly when the test score is below every null score: the
    # observation is entirely unremarkable under the null.
    assert np.all(p > 0.0) and np.all(p <= 1.0)


def test_a_riskier_test_item_gets_a_smaller_p_value():
    cal = [0.1, 0.2, 0.3, 0.4, 0.5]
    p = conformal_pvalues(cal, [0.35, 0.15])
    assert p[0] < p[1], "a higher test score is less surprising under the null"


def test_p_value_of_a_maximal_score_is_pinned_to_the_floor():
    cal = [0.1, 0.2, 0.3]
    p = conformal_pvalues(cal, [1.0])
    assert p[0] == pytest.approx(1.0 / (len(cal) + 1))


def test_empty_calibration_is_rejected():
    with pytest.raises(ValueError):
        conformal_pvalues([], [0.5])


def test_bh_accepts_clearly_null_items():
    p = [0.001, 0.002, 0.9, 0.95]
    mask = benjamini_hochberg(p, alpha=0.10)
    assert mask.tolist() == [True, True, False, False]


def test_alpha_too_small_accepts_nothing():
    # This is the empirically observed situation on the frozen TextVQA replay.
    p = [0.05, 0.2, 0.4, 0.6, 0.8]
    assert benjamini_hochberg(p, alpha=0.05).sum() == 0
    assert benjamini_yekutieli(p, alpha=0.05).sum() == 0


def test_by_is_at_least_as_conservative_as_bh():
    p = [0.001, 0.01, 0.02, 0.03, 0.2, 0.5]
    bh = benjamini_hochberg(p, alpha=0.1).sum()
    by = benjamini_yekutieli(p, alpha=0.1).sum()
    assert by <= bh


def test_selection_reports_coverage_and_fdp():
    cal = list(np.linspace(0.0, 1.0, 200))
    test = [0.01, 0.02, 0.99]
    result = conformal_select(cal, test, alpha=0.10, procedure="BH", test_labels=[0, 0, 1])
    assert result.num_items == 3
    assert result.coverage == pytest.approx(result.num_accepted / 3)
    if result.num_accepted:
        assert result.realized_fdp is not None
    assert result.validated is VALIDATED is False


def test_selection_carries_a_status_flag():
    result = conformal_select([0.1, 0.2, 0.3], [0.05], alpha=0.2, procedure="BY")
    assert result.validated is VALIDATED is False
    assert isinstance(result.notes, list)


def test_zero_acceptance_is_reported_not_silent():
    # A calibration set that supports no selection must say so.
    result = conformal_select([0.05, 0.2, 0.4, 0.6, 0.8], [0.9], alpha=0.05, procedure="BY")
    if result.num_accepted == 0:
        assert any("No item was accepted" in n for n in result.notes)


def test_invalid_procedure_and_alpha_are_rejected():
    with pytest.raises(ValueError):
        conformal_select([0.1], [0.2], procedure="bonferroni")
    with pytest.raises(ValueError):
        conformal_select([0.1], [0.2], alpha=1.5)
