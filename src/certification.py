"""Distribution-free threshold certification (Section 4.1, Theorem 1).

Candidate thresholds are chosen without the calibration data and sorted in
decreasing order. For each candidate and calibration family the exact
binomial p-value is ``P(Bin(n_e, alpha) <= k_e)``, where ``k_e`` counts the
exceedances; the intersection--union p-value is the maximum over families.

* Fixed-sequence rule: certify candidates in order until the first p-value
  above ``beta`` and deploy the last certified threshold.
* Bonferroni rules: certify every candidate whose p-value is at most
  ``beta / (L J)`` or ``beta / L``.

An infinite threshold means that no candidate was certified (abstention).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def threshold_at_sensitivity(scores: np.ndarray, target_sensitivity: float) -> float:
    """Largest threshold whose empirical exceedance rate on ``scores`` is at least ``target_sensitivity``."""

    positives = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must lie in (0, 1]")
    allowed_misses = int(np.floor((1.0 - target_sensitivity) * positives.size + 1e-12))
    index = min(allowed_misses, positives.size - 1)
    return float(np.nextafter(positives[index], -np.inf))


def candidates_from_target_scores(target_scores: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
    """Candidate thresholds at the given empirical detection levels of a separate target split."""

    return np.unique([threshold_at_sensitivity(target_scores, s) for s in sensitivities])


def exceedances(sorted_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return sorted_scores.size - np.searchsorted(sorted_scores, thresholds, side="right")


def binomial_risk_pvalue(exceedance_count: int, n: int, alpha: float) -> float:
    """Exact p-value of ``H: risk > alpha`` from ``exceedance_count`` exceedances in ``n`` observations."""

    return float(binom.cdf(int(exceedance_count), int(n), float(alpha)))


def fixed_sequence_threshold(
    candidate_thresholds: np.ndarray,
    calibration_families: list[np.ndarray],
    alpha: float,
    beta: float = 0.05,
) -> tuple[float, int]:
    """Fixed-sequence intersection--union rule. Returns ``(threshold, number certified)``."""

    candidates = np.unique(np.asarray(candidate_thresholds, dtype=float).reshape(-1))[::-1]
    if candidates.size == 0 or not np.isfinite(candidates).all():
        raise ValueError("candidate thresholds must be nonempty and finite")
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("alpha and beta must lie in (0, 1)")
    families = [np.sort(np.asarray(scores, dtype=float).reshape(-1)) for scores in calibration_families]
    if not families or any(scores.size == 0 for scores in families):
        raise ValueError("at least one nonempty calibration family is required")
    selected = float("inf")
    certified = 0
    for threshold in candidates:
        p_value = max(
            binomial_risk_pvalue(scores.size - int(np.searchsorted(scores, threshold, side="right")), scores.size, alpha)
            for scores in families
        )
        if p_value > beta:
            break
        selected = float(threshold)
        certified += 1
    return selected, certified


def deploy_rules(candidates: np.ndarray, sorted_families: list[np.ndarray], alpha: float, beta: float) -> dict[str, float]:
    """Deployed threshold of the two Bonferroni rules and the fixed-sequence rule on one candidate grid."""

    descending = np.sort(candidates)[::-1]
    per_family = np.asarray([binom.cdf(exceedances(fam, descending), fam.size, alpha) for fam in sorted_families])
    iut = per_family.max(axis=0)
    size = descending.size
    out = {}
    certified = descending[iut <= beta / (size * len(sorted_families))]
    out["bonferroni_LJ"] = float(certified.min()) if certified.size else float("inf")
    certified = descending[iut <= beta / size]
    out["bonferroni_L"] = float(certified.min()) if certified.size else float("inf")
    failures = np.flatnonzero(iut > beta)
    stop = int(failures[0]) if failures.size else size
    out["fixed_sequence"] = float(descending[stop - 1]) if stop > 0 else float("inf")
    return out
