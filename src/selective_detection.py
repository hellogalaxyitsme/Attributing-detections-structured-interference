"""Mismatch-selective adaptive detectors and their large-sample mimic limits (Proposition 3).

Detectors: ABORT (Pulsone and Rader, 2001), W-ABORT (Bandiera, Besson and
Ricci, 2008), the adaptive Rao test (De Maio, 2007) and Kelly's GLRT. All are
written through the whitened energies ``n_K = y'S^-1 H (H'S^-1 H)^-1 H'S^-1 y``
and ``c_K = y'S^-1 y`` with the unnormalized scatter matrix ``S``; for one
target column they coincide with the published rank-one statistics.
"""

from __future__ import annotations

import numpy as np

# Large-sample null limits of the statistics (S/K -> Sigma, K -> infinity).
NULL_LIMITS = {"kelly": 0.0, "rao": 0.0, "abort": 0.5, "w_abort": 1.0}


def statistics_from_energies(numerator: np.ndarray, quadratic: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """AMF ``K n``, ACE ``n / c``, Kelly ``n / (1 + c)``, ABORT ``(1 + n) / (2 + c)``,
    W-ABORT ``(1 + c) / (1 + c - n)^2`` and Rao ``n / ((1 + c)(1 + c - n))``."""

    n = np.asarray(numerator, dtype=float)
    c = np.asarray(quadratic, dtype=float)
    tiny = np.finfo(float).tiny
    orthogonal = np.maximum(1.0 + c - n, tiny)
    return {
        "amf": float(k) * n,
        "ace": n / np.maximum(c, tiny),
        "kelly": n / (1.0 + c),
        "abort": (1.0 + n) / (2.0 + c),
        "w_abort": (1.0 + c) / orthogonal**2,
        "rao": n / ((1.0 + c) * orthogonal),
    }


def adaptive_energy_grams(columns: np.ndarray, secondary: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial Gram matrices of ``n_K`` and ``c_K`` over a set of columns.

    ``columns`` has shape ``(n, m, J)``. For ``y = X w`` the energies are
    ``n_K = w'A w`` and ``c_K = w'Q w``, so every mimic amplitude reuses the same
    noise and secondary data.
    """

    columns = np.asarray(columns, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    target = np.asarray(target, dtype=float)
    scatter = np.matmul(secondary.transpose(0, 2, 1), secondary)
    weighted = np.linalg.solve(scatter, columns)
    weighted_h = np.linalg.solve(scatter, np.broadcast_to(target, (columns.shape[0],) + target.shape))
    gram = np.matmul(target.T[None, :, :], weighted_h)
    projected = np.matmul(target.T[None, :, :], weighted)
    numerator = np.matmul(projected.transpose(0, 2, 1), np.linalg.solve(gram, projected))
    quadratic = np.matmul(columns.transpose(0, 2, 1), weighted)
    return 0.5 * (numerator + numerator.transpose(0, 2, 1)), 0.5 * (quadratic + quadratic.transpose(0, 2, 1))


def eigencanceler_grams(columns: np.ndarray, secondary: np.ndarray, target: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Gram matrices of the eigencanceler numerator and residual energy after annihilating
    the dominant ``rank`` eigenvectors of ``S_K``."""

    columns = np.asarray(columns, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    target = np.asarray(target, dtype=float)
    k = secondary.shape[1]
    scatter = np.matmul(secondary.transpose(0, 2, 1), secondary) / float(k)
    _, vectors = np.linalg.eigh(scatter)
    basis = vectors[:, :, -int(rank):]
    residual_x = columns - np.matmul(basis, np.matmul(basis.transpose(0, 2, 1), columns))
    residual_h = target[None, :, :] - np.matmul(basis, np.matmul(basis.transpose(0, 2, 1), target))
    gram = np.matmul(target.T[None, :, :], residual_h)
    projected = np.matmul(target.T[None, :, :], residual_x)
    numerator = np.matmul(projected.transpose(0, 2, 1), np.linalg.solve(gram, projected))
    total = np.matmul(residual_x.transpose(0, 2, 1), residual_x)
    return 0.5 * (numerator + numerator.transpose(0, 2, 1)), total


def combine(gram: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``w' G_n w`` for every trial ``n``."""

    weights = np.asarray(weights, dtype=float)
    return np.einsum("nij,i,j->n", gram, weights, weights)


def large_sample_mimic_limit(detector: str, coherence: float, energy: float) -> float:
    """Limit of a statistic on a mimic with coherence ``kappa_f^2`` and normalized energy ``E``."""

    k2, e = float(coherence), float(energy)
    n, c = k2 * e, e
    orthogonal = 1.0 + c - n
    if detector == "kelly":
        return n / (1.0 + c)
    if detector == "abort":
        return (1.0 + n) / (2.0 + c)
    if detector == "w_abort":
        return (1.0 + c) / orthogonal**2
    if detector == "rao":
        return n / ((1.0 + c) * orthogonal)
    raise ValueError(f"no large-sample limit for {detector}")


def w_abort_acceptance_window(coherence: float) -> tuple[float, float]:
    """Normalized energies at which the W-ABORT limit exceeds its null limit.

    The window ``(0, (1 - 2 beta) / beta^2)`` with ``beta = 1 - kappa_f^2`` is
    nonempty if and only if ``kappa_f^2 > 1/2``.
    """

    beta = 1.0 - float(coherence)
    if beta >= 0.5:
        return (0.0, 0.0)
    return (0.0, (1.0 - 2.0 * beta) / beta**2)
