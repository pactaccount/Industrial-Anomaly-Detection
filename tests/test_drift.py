"""Tests for drift detection.

Drift detection is the early-warning system for the failure mode that
kills most production deployments: a good model whose threshold no longer
fits the line.
"""

from __future__ import annotations

import numpy as np

from argus.services.drift import check_drift, ks_critical_value, ks_statistic


def test_ks_statistic_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    # Two samples from the same distribution: a small but non-zero statistic.
    assert ks_statistic(a, b) < 0.15
    assert ks_statistic(a, a) == 0.0


def test_ks_statistic_detects_a_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(3, 1, size=500)
    # A three-sigma shift separates the distributions almost entirely, so
    # the KS statistic approaches 1. The exact value depends on the draw.
    assert ks_statistic(a, b) > 0.8


def test_critical_value_decreases_with_stricter_alpha():
    assert ks_critical_value(0.01) > ks_critical_value(0.05)
    assert ks_critical_value(0.05) > ks_critical_value(0.10)


def test_stable_distribution_is_not_flagged():
    rng = np.random.default_rng(2)
    baseline = list(rng.normal(3.0, 0.3, size=300))
    current = list(rng.normal(3.0, 0.3, size=300))
    report = check_drift(baseline, current, threshold=4.0, min_samples=20)
    assert not report.drifted
    assert report.ks_statistic < report.critical_value


def test_shifted_distribution_is_flagged():
    rng = np.random.default_rng(3)
    baseline = list(rng.normal(3.0, 0.3, size=300))
    current = list(rng.normal(4.5, 0.3, size=300))
    report = check_drift(baseline, current, threshold=4.0, min_samples=20)
    assert report.drifted
    assert report.mean_shift > 1.0
    assert report.notes, "a drift must come with an explanation"


def test_drift_projects_a_higher_false_reject_rate():
    rng = np.random.default_rng(4)
    baseline = list(rng.normal(3.0, 0.2, size=300))
    current = list(rng.normal(4.2, 0.2, size=300))
    report = check_drift(baseline, current, threshold=3.8, min_samples=20)
    assert report.projected_false_reject_rate > report.baseline_false_reject_rate
    assert report.projected_false_reject_rate > 0.5


def test_small_sample_is_reported_as_unreliable():
    """Below the minimum sample size we must not raise a false alarm."""
    rng = np.random.default_rng(5)
    baseline = list(rng.normal(3.0, 0.3, size=300))
    current = list(rng.normal(9.0, 0.3, size=5))
    report = check_drift(baseline, current, threshold=4.0, min_samples=20)
    assert not report.drifted
    assert any("not reliable" in n for n in report.notes)


def test_empty_baseline_is_handled():
    report = check_drift([], [1.0] * 50, threshold=2.0, min_samples=20)
    assert not report.drifted
    assert any("No baseline" in n for n in report.notes)
