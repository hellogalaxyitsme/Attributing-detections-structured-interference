"""Canonical target/interference geometry and declaration-error measures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatorPair:
    """Target operator ``H`` and declared interference basis ``U`` (orthonormal columns)."""

    target: np.ndarray
    interference: np.ndarray
    overlap: float
    seed: int


def controlled_operator_pair(sensor_dim: int, target_dim: int, interference_rank: int, overlap: float, seed: int) -> OperatorPair:
    """Operators whose principal-angle cosines all equal ``overlap`` (Supplement S3).

    ``U`` is the first ``r`` columns of a random orthogonal matrix ``Q`` and
    ``H = gamma Q[:, :p] + sqrt(1 - gamma^2) Q[:, r:r+p]``.
    """

    if not (1 <= target_dim <= interference_rank < sensor_dim and sensor_dim - interference_rank >= target_dim):
        raise ValueError("require 1 <= p <= r < m and m - r >= p")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(sensor_dim, sensor_dim)))
    interference = q[:, :interference_rank]
    inside = interference[:, :target_dim]
    outside = q[:, interference_rank : interference_rank + target_dim]
    target = overlap * inside + np.sqrt(max(0.0, 1.0 - overlap**2)) * outside
    return OperatorPair(target=target, interference=interference, overlap=overlap, seed=seed)


def _column_basis(matrix: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0:
        return np.empty((matrix.shape[0], 0))
    threshold = tolerance * max(matrix.shape) * singular[0]
    return u[:, singular > threshold]


def subspace_projector(matrix: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto ``col(matrix)``."""

    basis = _column_basis(np.asarray(matrix, dtype=float))
    return basis @ basis.T


def declaration_error(target: np.ndarray, true_interference: np.ndarray, declared_interference: np.ndarray) -> dict[str, float]:
    """Subspace error ``delta = ||P_hat - U U'||_2`` and target-specific leakage ``eta_s = ||P_hat_S U U'||_2``."""

    target = np.asarray(target, dtype=float)
    true_projector = subspace_projector(true_interference)
    declared_projector = subspace_projector(declared_interference)
    delta = float(np.linalg.norm(true_projector - declared_projector, ord=2))
    declared_residual = np.eye(target.shape[0]) - declared_projector
    declared_target_projector = subspace_projector(declared_residual @ target)
    leakage = float(np.linalg.norm(declared_target_projector @ true_projector, ord=2))
    return {"projector_distance": delta, "target_specific_leakage": leakage}
