"""Exact tests and intervals used in the analyses."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_law


def paired_sign_flip_pvalue(differences: np.ndarray, alternative: str = "greater") -> float:
    """Exact sign-flip p-value over at most 20 nonzero paired differences."""

    differences = np.asarray(differences, dtype=float).reshape(-1)
    if not np.isfinite(differences).all() or differences.size == 0:
        raise ValueError("differences must be nonempty and finite")
    differences = differences[differences != 0.0]
    if differences.size == 0:
        return 1.0
    if differences.size > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 nonzero differences")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("unsupported alternative")
    observed = float(np.mean(differences))
    masks = np.arange(1 << differences.size, dtype=np.uint64)[:, None]
    bits = (masks >> np.arange(differences.size, dtype=np.uint64)) & 1
    signs = 2.0 * bits.astype(float) - 1.0
    statistics = np.mean(signs * np.abs(differences)[None, :], axis=1)
    tolerance = 1e-15
    if alternative == "greater":
        return float(np.mean(statistics >= observed - tolerance))
    if alternative == "less":
        return float(np.mean(statistics <= observed + tolerance))
    return float(np.mean(np.abs(statistics) >= abs(observed) - tolerance))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values in the original order."""

    p_values = np.asarray(p_values, dtype=float).reshape(-1)
    if p_values.size == 0 or np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("p-values must be nonempty and lie in [0, 1]")
    order = np.argsort(p_values, kind="mergesort")
    adjusted_sorted = np.empty(p_values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (p_values.size - rank) * p_values[index])
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(p_values)
    adjusted[order] = adjusted_sorted
    return adjusted


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper--Pearson interval for a binomial proportion."""

    a = 1.0 - level
    lower = 0.0 if k == 0 else float(beta_law.ppf(a / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta_law.ppf(1 - a / 2, k + 1, n - k))
    return lower, upper
