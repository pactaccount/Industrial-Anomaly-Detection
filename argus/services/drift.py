"""Drift detection: notice when the line has changed before the FPR does.

The hidden cost that kills most AI inspection pilots is not training — it is
recalibration as the process drifts. Nobody bills for it, and it happens
silently: the lighting changes, or a supplier changes material, or a
process tweak shifts the part slightly. The scores move up, more good parts
get rejected, and the operator starts ignoring the alerts.

The useful signal is not "the score is high" — the model will say that about
a defective part, which is correct. The signal is **"the score
distribution of the incoming good parts has moved"** relative to the
distribution the model was calibrated on.

We implement this with the two-sample Kolmogorov-Smirnov statistic, which
compares two empirical distributions without assuming normality, and
compare it against a critical value so we report a rate rather than a
vague "it looks different".

The critical values are the standard asymptotic Kolmogorov distribution:

    c(alpha) = sqrt(-0.5 * ln(alpha/2))

and the test rejects at significance ``alpha`` when the KS statistic
exceeds it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic.

    The statistic is the largest difference between the two empirical
    cumulative distribution functions.
    """
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return 0.0
    all_values = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_values, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_critical_value(alpha: float = 0.05, n: int | None = None) -> float:
    """The KS critical value at significance ``alpha``.

    Two regimes:

    * **Asymptotic** (large samples): ``sqrt(-0.5 * ln(alpha/2))``. This is
      the standard large-sample approximation.
    * **Finite-sample** (small samples): the asymptotic value is far too
      large to be useful. With n=300 and alpha=0.05 the asymptotic value
      is 1.358, but a KS statistic of 1.0 is a massive, unambiguous
      distribution shift. The exact critical value for finite samples is
      approximately ``c(alpha) / sqrt(n)`` (the asymptotic form scaled by
      the effective sample size), with the usual ``sqrt(n_eff)`` where
      ``n_eff = n1*n2/(n1+n2)``.

    Using the asymptotic value for realistic sample sizes means drift is
    never detected, which is worse than not having the check at all.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    base = math.sqrt(-0.5 * math.log(alpha / 2.0))
    if n is None or n <= 0:
        return base
    return base / math.sqrt(n)


def ks_critical_value_for_sizes(
    alpha: float, n1: int, n2: int
) -> float:
    """The KS critical value for two samples of the given sizes."""
    if n1 <= 0 or n2 <= 0:
        return ks_critical_value(alpha)
    n_eff = (n1 * n2) / (n1 + n2)
    return ks_critical_value(alpha, n=int(round(n_eff)))


@dataclass
class DriftReport:
    """The result of a drift check."""

    baseline_n: int = 0
    current_n: int = 0
    ks_statistic: float = 0.0
    critical_value: float = 0.0
    drifted: bool = False
    alpha: float = 0.05
    baseline_mean: float = 0.0
    current_mean: float = 0.0
    mean_shift: float = 0.0
    # How many standard deviations the mean moved, using the baseline spread.
    shift_in_sigma: float = 0.0
    # The share of current parts that would now be rejected.
    projected_false_reject_rate: float = 0.0
    baseline_false_reject_rate: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "DRIFTED" if self.drifted else "stable"
        lines = [
            f"Drift check: {verdict}",
            f"  KS statistic:    {self.ks_statistic:.4f} "
            f"(critical {self.critical_value:.4f} at alpha={self.alpha})",
            f"  Baseline mean:   {self.baseline_mean:.3f} (n={self.baseline_n})",
            f"  Current mean:    {self.current_mean:.3f} (n={self.current_n})",
            f"  Mean shift:      {self.mean_shift:+.3f} "
            f"({self.shift_in_sigma:+.2f} baseline sigma)",
            f"  Projected FPR:   {self.projected_false_reject_rate * 100:.2f}%"
            f" (was {self.baseline_false_reject_rate * 100:.2f}%)",
        ]
        if self.notes:
            lines.append("  Notes:")
            lines.extend(f"    - {n}" for n in self.notes)
        return "\n".join(lines)


def check_drift(
    baseline_scores: list[float],
    current_scores: list[float],
    threshold: float | None = None,
    alpha: float = 0.05,
    min_samples: int = 20,
) -> DriftReport:
    """Compare the current score distribution against the calibrated one.

    Args:
        baseline_scores: the scores the model was calibrated on — the
            held-out good-part scores stored with the model.
        current_scores: the scores of recent good parts.
        threshold: the model's operating threshold, used to project what
            the false reject rate has become.
        alpha: significance level for the KS test.
        min_samples: below this many current samples the test is not
            reliable, and we say so rather than reporting a false alarm.

    Returns:
        A :class:`DriftReport`.
    """
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    current = np.asarray(current_scores, dtype=np.float64)

    report = DriftReport(
        baseline_n=len(baseline),
        current_n=len(current),
        alpha=alpha,
        critical_value=ks_critical_value_for_sizes(alpha, len(baseline), len(current)),
    )

    if len(current) < min_samples:
        report.notes.append(
            f"Only {len(current)} samples (need {min_samples}); the KS test is "
            "not reliable at this size. Collect more parts before acting."
        )
        return report

    if len(baseline) == 0:
        report.notes.append("No baseline scores available to compare against.")
        return report

    report.ks_statistic = ks_statistic(baseline, current)
    report.drifted = report.ks_statistic > report.critical_value
    report.baseline_mean = float(baseline.mean())
    report.current_mean = float(current.mean())
    report.mean_shift = report.current_mean - report.baseline_mean

    baseline_std = float(baseline.std())
    if baseline_std > 0:
        report.shift_in_sigma = report.mean_shift / baseline_std

    if threshold is not None:
        report.baseline_false_reject_rate = float((baseline > threshold).mean())
        report.projected_false_reject_rate = float((current > threshold).mean())
        if report.projected_false_reject_rate > report.baseline_false_reject_rate + 0.02:
            report.notes.append(
                "The false reject rate has risen materially. Recalibrate the "
                "threshold, and check the lighting and process before "
                "assuming the parts themselves have changed."
            )

    if report.drifted and not report.notes:
        report.notes.append(
            "The score distribution has shifted. Recalibrate the threshold, "
            "and check whether the optics, recipe or supplier changed."
        )

    return report
