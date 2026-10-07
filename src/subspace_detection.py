"""Subspace detectors, exact laws of Gaussian quadratic forms, and the simulation scene.

Detectors (classical, credited in the paper):

* matched-subspace GLR and MSD-F (Scharf and Friedlander, 1994);
* clairvoyant and adaptive AMF, ACE and Kelly subspace detectors (Kelly, 1986;
  Robey et al., 1992; Kraut, Scharf and McWhorter, 2001);
* the eigencanceler, which estimates the interference subspace from secondary
  data (Haimovich, 1996).

Exact operating characteristics use Imhof's (1961) inversion formula and
Ruben's (1962) series for Gaussian quadratic forms (Proposition 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.stats import chi2

# A residual is treated as zero when its squared norm is at most this fraction
# of the squared norm of the observation. Floating-point residuals of exact
# declared-subspace fields are below 1e-28 in relative terms.
ZERO_RESIDUAL_TOLERANCE = 1e-20


# ---------------------------------------------------------------------------
# Exact laws of Gaussian quadratic forms
# ---------------------------------------------------------------------------


def _symmetric_sqrt(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    if values.min() < -1e-10 * max(1.0, abs(values.max())):
        raise ValueError("covariance must be positive semidefinite")
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def quadratic_form_eigenvalues(form: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Eigenvalues ``lambda`` such that ``y'By = sum lambda_j Z_j^2`` for ``y ~ N(0, C)``."""

    form = np.asarray(form, dtype=float)
    form = 0.5 * (form + form.T)
    root = _symmetric_sqrt(covariance)
    values = np.linalg.eigvalsh(root @ form @ root)
    scale = max(1.0, float(np.max(np.abs(values)))) if values.size else 1.0
    return values[np.abs(values) > 1e-11 * scale]


def imhof_tail(eigenvalues: np.ndarray, threshold: float = 0.0) -> float:
    """Exact ``P(sum_j lambda_j Z_j^2 > threshold)`` for i.i.d. standard normals.

    One eigenvalue: closed form. Two eigenvalues: one-dimensional conditioning.
    Definite forms with at least three eigenvalues and spread below 1e3:
    Ruben's chi-squared mixture series. Otherwise: Imhof's inversion formula,
    truncated where Imhof's tail bound falls below 1e-10.
    """

    lam = np.asarray(eigenvalues, dtype=float).reshape(-1)
    lam = lam[lam != 0.0]
    threshold = float(threshold)
    if lam.size == 0:
        return 1.0 if threshold < 0.0 else 0.0
    if np.all(lam > 0.0) and threshold <= 0.0:
        return 1.0
    if np.all(lam < 0.0) and threshold >= 0.0:
        return 0.0
    if lam.size == 1:
        ratio = threshold / lam[0]
        tail = float(chi2.sf(ratio, 1)) if ratio > 0.0 else 1.0
        return tail if lam[0] > 0.0 else 1.0 - tail
    spread = float(np.max(np.abs(lam)) / np.min(np.abs(lam)))
    if lam.size >= 3 and spread < 1e3 and (np.all(lam > 0.0) or np.all(lam < 0.0)):
        if lam[0] > 0.0:
            return _ruben_tail(lam, threshold)
        return 1.0 - _ruben_tail(-lam, -threshold)
    if lam.size == 2:

        def conditional(z: float) -> float:
            return imhof_tail(lam[:1], threshold - lam[1] * z * z) * np.exp(-0.5 * z * z)

        value, _ = integrate.quad(conditional, 0.0, np.inf, limit=400, epsabs=1e-13)
        return float(min(1.0, max(0.0, 2.0 * value / np.sqrt(2.0 * np.pi))))
    k = lam.size
    abs_lam = np.abs(lam)

    def integrand(u: float) -> float:
        if u == 0.0:
            return 0.5 * float(np.sum(lam)) - 0.5 * threshold
        theta = 0.5 * float(np.sum(np.arctan(lam * u))) - 0.5 * threshold * u
        log_rho = 0.25 * float(np.sum(np.log1p((lam * u) ** 2)))
        return float(np.sin(theta) / (u * np.exp(log_rho)))

    # Imhof's truncation bound: tail <= 2 / (pi k U^{k/2} prod |lambda|^{1/2}).
    log_prod = 0.5 * float(np.sum(np.log(abs_lam)))
    target = 1e-10
    log_upper = (np.log(2.0 / (np.pi * k * target)) - log_prod) / (0.5 * k)
    upper = float(np.exp(min(log_upper, 700.0)))
    # Segments start at the eigenvalue scale and double, but never span more
    # than twenty periods of the threshold-induced oscillation.
    base = 1.0 / float(np.max(abs_lam))
    max_width = 40.0 * np.pi / abs(threshold) if threshold != 0.0 else np.inf
    edges = [0.0]
    edge = 0.0
    width = base
    while edge < upper and len(edges) < 200_000:
        edge = min(upper, edge + min(width, max_width))
        edges.append(edge)
        width *= 2.0
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        value, _ = integrate.quad(integrand, left, right, limit=200, epsabs=1e-14, epsrel=1e-10)
        total += value
    return float(min(1.0, max(0.0, 0.5 + total / np.pi)))


def _ruben_tail(lam: np.ndarray, threshold: float, tolerance: float = 1e-12) -> float:
    """``P(sum lambda_j Z_j^2 > t)`` for positive weights via Ruben's (1962) series."""

    lam = np.asarray(lam, dtype=float)
    if threshold <= 0.0:
        return 1.0
    reference = float(lam.min())
    ratios = 1.0 - reference / lam
    k = lam.size
    coefficients = [float(np.prod(np.sqrt(reference / lam)))]
    powers = np.ones_like(ratios)
    g_values: list[float] = []
    cdf = coefficients[0] * float(chi2.cdf(threshold / reference, k))
    mass = coefficients[0]
    index = 0
    while 1.0 - mass > tolerance and index < 200_000:
        index += 1
        powers = powers * ratios
        g_values.append(float(np.sum(powers)))
        # c_i = (1/(2i)) sum_{r=1}^{i} g_r c_{i-r}
        coefficient = float(np.dot(g_values, coefficients[::-1])) / (2.0 * index)
        coefficients.append(coefficient)
        mass += coefficient
        cdf += coefficient * float(chi2.cdf(threshold / reference, k + 2 * index))
        if coefficient == 0.0 and np.max(ratios) == 0.0:
            break
    return float(min(1.0, max(0.0, 1.0 - cdf)))


def gaussian_quadratic_tail(form: np.ndarray, covariance: np.ndarray, threshold: float = 0.0) -> float:
    """Exact ``P(y'By > threshold)`` for ``y ~ N(0, covariance)``."""

    return imhof_tail(quadratic_form_eigenvalues(form, covariance), threshold)


# ---------------------------------------------------------------------------
# Declared geometry and the soft/hard information split
# ---------------------------------------------------------------------------


def orthonormal_basis(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] <= tolerance:
        return np.empty((matrix.shape[0], 0))
    return left[:, singular > tolerance * singular[0]]


@dataclass(frozen=True)
class DeclaredGeometry:
    """Bases of col(RH) (dimension p) and of its complement in range(R) (dimension q)."""

    target_basis: np.ndarray
    complement_basis: np.ndarray

    @property
    def p(self) -> int:
        return int(self.target_basis.shape[1])

    @property
    def q(self) -> int:
        return int(self.complement_basis.shape[1])


def declared_geometry(target: np.ndarray, declared_interference: np.ndarray) -> DeclaredGeometry:
    target = np.asarray(target, dtype=float)
    m = target.shape[0]
    interference_basis = orthonormal_basis(declared_interference)
    residual = np.eye(m) - interference_basis @ interference_basis.T
    target_basis = orthonormal_basis(residual @ target)
    residual_basis = orthonormal_basis(residual)
    remainder = residual_basis - target_basis @ (target_basis.T @ residual_basis)
    complement = orthonormal_basis(remainder)
    if target_basis.shape[1] == 0 or complement.shape[1] == 0:
        raise ValueError("declared geometry requires rank(RH) > 0 and q > 0")
    return DeclaredGeometry(target_basis=target_basis, complement_basis=complement)


def soft_hard_information(
    target: np.ndarray,
    interference: np.ndarray,
    interference_covariance: np.ndarray,
    noise_variance: float,
) -> dict[str, np.ndarray | float]:
    """Split ``G = H' Sigma^-1 H`` into ``G_h + G_s`` (Proposition 2).

    For white noise ``sigma^2 I`` and ``Sigma = U Lambda U' + sigma^2 I`` with
    orthonormal ``U``, ``Sigma^-1 = R / sigma^2 + U (Lambda + sigma^2 I)^-1 U'``.
    Returns both parts and the worst-case mimic coherence
    ``kappa^2 = lambda_max(G^-1 G_s)``.
    """

    target = np.asarray(target, dtype=float)
    basis = orthonormal_basis(interference)
    if basis.shape[1] != np.asarray(interference).shape[1]:
        raise ValueError("interference operator must have full column rank")
    interference = np.asarray(interference, dtype=float)
    coordinates = basis.T @ interference
    lam = coordinates @ np.asarray(interference_covariance, dtype=float) @ coordinates.T
    sigma2 = float(noise_variance)
    m = target.shape[0]
    residual = np.eye(m) - basis @ basis.T
    soft_core = np.linalg.inv(lam + sigma2 * np.eye(basis.shape[1]))
    g_hard = target.T @ residual @ target / sigma2
    g_soft = target.T @ basis @ soft_core @ basis.T @ target
    g_total = g_hard + g_soft
    values = np.linalg.eigvals(np.linalg.solve(g_total, g_soft))
    return {
        "g_hard": g_hard,
        "g_soft": g_soft,
        "g_total": g_total,
        "max_mimic_coherence": float(np.max(np.real(values))),
    }


def canonical_mimic_coherence(overlap: float, inr: float) -> float:
    """``kappa^2 = gamma^2 / (1 + rho (1 - gamma^2))`` in the canonical geometry (7)."""

    gamma2 = float(overlap) ** 2
    return gamma2 / (1.0 + float(inr) * (1.0 - gamma2))


# ---------------------------------------------------------------------------
# Detector statistics on data arrays (rows are observations)
# ---------------------------------------------------------------------------


def zero_residual_mask(measurements: np.ndarray, geometry: DeclaredGeometry) -> np.ndarray:
    """Observations whose residual after annihilation of the declared subspace is zero."""

    values = np.asarray(measurements, dtype=float)
    residual = np.sum((values @ geometry.target_basis) ** 2, axis=1) + np.sum((values @ geometry.complement_basis) ** 2, axis=1)
    return residual <= ZERO_RESIDUAL_TOLERANCE * np.sum(values**2, axis=1)


def glr_statistic(measurements: np.ndarray, geometry: DeclaredGeometry, noise_variance: float) -> np.ndarray:
    """Known-noise GLR ``||P_S y||^2 / sigma^2``; zero for observations with ``Ry = 0``."""

    coordinates = np.asarray(measurements, dtype=float) @ geometry.target_basis
    scores = np.sum(coordinates**2, axis=1) / float(noise_variance)
    scores[zero_residual_mask(measurements, geometry)] = 0.0
    return scores


def f_statistic(measurements: np.ndarray, geometry: DeclaredGeometry) -> np.ndarray:
    """MSD-F statistic (4); observations with ``Ry = 0`` are non-detections (``F = 0``)."""

    values = np.asarray(measurements, dtype=float)
    numerator = np.sum((values @ geometry.target_basis) ** 2, axis=1) / geometry.p
    denominator = np.sum((values @ geometry.complement_basis) ** 2, axis=1) / geometry.q
    scores = numerator / np.maximum(denominator, np.finfo(float).tiny)
    scores[zero_residual_mask(values, geometry)] = 0.0
    return scores


def _whitened_target_projector(precision: np.ndarray, target: np.ndarray) -> np.ndarray:
    weighted = precision @ target
    gram = target.T @ weighted
    return weighted @ np.linalg.solve(gram, weighted.T)


def clairvoyant_amf(measurements: np.ndarray, covariance: np.ndarray, target: np.ndarray) -> np.ndarray:
    precision = np.linalg.inv(covariance)
    form = _whitened_target_projector(precision, np.asarray(target, dtype=float))
    values = np.asarray(measurements, dtype=float)
    return np.einsum("ni,ij,nj->n", values, form, values)


def clairvoyant_ace(measurements: np.ndarray, covariance: np.ndarray, target: np.ndarray) -> np.ndarray:
    precision = np.linalg.inv(covariance)
    form = _whitened_target_projector(precision, np.asarray(target, dtype=float))
    values = np.asarray(measurements, dtype=float)
    numerator = np.einsum("ni,ij,nj->n", values, form, values)
    denominator = np.einsum("ni,ij,nj->n", values, precision, values)
    return numerator / np.maximum(denominator, np.finfo(float).tiny)


def amf_quadratic_form(covariance: np.ndarray, target: np.ndarray) -> np.ndarray:
    return _whitened_target_projector(np.linalg.inv(covariance), np.asarray(target, dtype=float))


def adaptive_subspace_statistics(
    measurements: np.ndarray, secondary: np.ndarray, target: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-trial adaptive AMF, ACE and Kelly statistics.

    ``measurements`` has shape ``(n, m)`` and ``secondary`` shape ``(n, K, m)``,
    so every trial has independent secondary data. ``S_K`` is the
    unnormalized scatter matrix: AMF ``K n_K``, ACE ``n_K / c_K`` and Kelly
    ``n_K / (1 + c_K)``.
    """

    values = np.asarray(measurements, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    target = np.asarray(target, dtype=float)
    n, k, m = secondary.shape
    if values.shape != (n, m):
        raise ValueError("measurements and secondary data are misaligned")
    scatter = np.matmul(secondary.transpose(0, 2, 1), secondary)
    inverse = np.linalg.inv(scatter)
    weighted_y = np.einsum("nij,nj->ni", inverse, values)
    weighted_h = np.einsum("nij,jp->nip", inverse, target)
    gram = np.einsum("ip,nij->npj", target, weighted_h)
    projected = np.einsum("ip,ni->np", target, weighted_y)
    solved = np.linalg.solve(gram, projected[..., None])[..., 0]
    numerator = np.einsum("np,np->n", projected, solved)
    quadratic = np.einsum("ni,ni->n", values, weighted_y)
    tiny = np.finfo(float).tiny
    return {
        "amf": float(k) * numerator,
        "ace": numerator / np.maximum(quadratic, tiny),
        "kelly": numerator / (1.0 + quadratic),
    }


def batched_eigencanceler_f(
    measurements: np.ndarray, secondary: np.ndarray, target: np.ndarray, rank: int
) -> np.ndarray:
    """Per-trial eigencanceler: MSD-F after annihilating the dominant ``rank`` eigenvectors of ``S_K``."""

    values = np.asarray(measurements, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    target = np.asarray(target, dtype=float)
    n, k, m = secondary.shape
    p = target.shape[1]
    q = m - int(rank) - p
    if q <= 0:
        raise ValueError("eigencanceler requires q = m - r - p > 0")
    scatter = np.matmul(secondary.transpose(0, 2, 1), secondary) / float(k)
    _, vectors = np.linalg.eigh(scatter)
    basis = vectors[:, :, -int(rank):]
    residual_y = values - np.einsum("nir,nr->ni", basis, np.einsum("nir,ni->nr", basis, values))
    residual_h = target[None, :, :] - np.matmul(basis, np.matmul(basis.transpose(0, 2, 1), target))
    gram = np.matmul(target.T[None, :, :], residual_h)
    projected = np.einsum("ip,ni->np", target, residual_y)
    numerator = np.einsum("np,np->n", projected, np.linalg.solve(gram, projected[..., None])[..., 0])
    total = np.einsum("ni,ni->n", residual_y, residual_y)
    denominator = np.maximum(total - numerator, np.finfo(float).tiny)
    return (numerator / p) / (denominator / q)


# ---------------------------------------------------------------------------
# Noise laws and the simulation scene
# ---------------------------------------------------------------------------


def draw_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    law: str,
    covariance_root: np.ndarray | None = None,
) -> np.ndarray:
    """Noise with unit variance per sensor (Section 5.4).

    ``gaussian``, ``student_t`` (3 degrees of freedom) and ``bursty``
    (Bernoulli(0.1) activation) are independent per sensor; ``compound_gaussian``
    is a spherically invariant multivariate t vector whose texture is shared by
    all sensors. An optional covariance root colours the draw.
    """

    n, m = shape
    if law == "gaussian":
        noise = rng.normal(size=shape)
    elif law == "student_t":
        noise = rng.standard_t(df=3, size=shape) / np.sqrt(3.0)
    elif law == "bursty":
        mask = rng.binomial(1, 0.1, size=shape)
        noise = mask * rng.normal(size=shape) / np.sqrt(0.1)
    elif law == "compound_gaussian":
        dof = 3.0
        texture = dof / rng.chisquare(dof, size=(n, 1))
        noise = rng.normal(size=shape) * np.sqrt(texture / (dof / (dof - 2.0)))
    else:
        raise ValueError(f"unsupported noise law: {law}")
    if covariance_root is not None:
        noise = noise @ np.asarray(covariance_root, dtype=float).T
    return noise


def ar1_correlation(m: int, coefficient: float) -> np.ndarray:
    index = np.arange(int(m))
    return float(coefficient) ** np.abs(index[:, None] - index[None, :])


class GaussianSubspaceScene:
    """Scene ``y = H theta + U c + e`` of Section 5.

    ``theta ~ N(0, I_p)``, ``c ~ N(0, rho sigma^2 I_r)`` and
    ``e ~ N(0, sigma^2 Psi)`` with ``sigma^2 = p / (m SNR)``. A canonical mimic
    of amplitude ``a`` adds ``a U U' H theta'`` for an independent ``theta'``.
    """

    def __init__(
        self,
        target: np.ndarray,
        interference: np.ndarray,
        snr_db: float,
        inr_db: float,
        noise_correlation: np.ndarray | None = None,
    ):
        self.target = np.asarray(target, dtype=float)
        self.interference = orthonormal_basis(interference)
        self.m, self.p = self.target.shape
        self.r = self.interference.shape[1]
        self.sigma2 = self.p / (self.m * 10.0 ** (float(snr_db) / 10.0))
        self.rho = 10.0 ** (float(inr_db) / 10.0)
        self.psi = np.eye(self.m) if noise_correlation is None else np.asarray(noise_correlation, dtype=float)
        self.psi_root = np.linalg.cholesky(self.psi)
        self.mimic_operator = self.interference @ self.interference.T @ self.target

    @property
    def null_covariance(self) -> np.ndarray:
        return self.rho * self.sigma2 * self.interference @ self.interference.T + self.sigma2 * self.psi

    @property
    def target_covariance(self) -> np.ndarray:
        return self.null_covariance + self.target @ self.target.T

    def mimic_covariance(self, amplitude: float) -> np.ndarray:
        return self.null_covariance + float(amplitude) ** 2 * self.mimic_operator @ self.mimic_operator.T

    def _base(self, rng: np.random.Generator, n: int, law: str = "gaussian") -> np.ndarray:
        c = np.sqrt(self.rho * self.sigma2) * rng.normal(size=(n, self.r))
        noise = np.sqrt(self.sigma2) * draw_noise(rng, (n, self.m), law, self.psi_root)
        return c @ self.interference.T + noise

    def null(self, rng: np.random.Generator, n: int, law: str = "gaussian") -> np.ndarray:
        return self._base(rng, n, law)

    def alternative(self, rng: np.random.Generator, n: int, law: str = "gaussian") -> np.ndarray:
        return self._base(rng, n, law) + rng.normal(size=(n, self.p)) @ self.target.T

    def mimic(self, rng: np.random.Generator, n: int, amplitude: float, law: str = "gaussian") -> np.ndarray:
        return self._base(rng, n, law) + float(amplitude) * rng.normal(size=(n, self.p)) @ self.mimic_operator.T

    def secondary(self, rng: np.random.Generator, n: int, k: int) -> np.ndarray:
        """``n`` independent sets of ``k`` interference-only secondary vectors."""

        return self._base(rng, n * k).reshape(n, k, self.m)


def exact_detector_forms(scene: GaussianSubspaceScene, alpha: float) -> dict[str, tuple[np.ndarray, float]]:
    """Quadratic forms ``(B, t)`` with ``{y'By > t}`` equal to each size-alpha test (Proposition 4)."""

    from scipy.stats import beta as beta_law
    from scipy.stats import f as f_law

    geometry = declared_geometry(scene.target, scene.interference)
    p, q, m = geometry.p, geometry.q, scene.m
    target_projector = geometry.target_basis @ geometry.target_basis.T
    complement_projector = geometry.complement_basis @ geometry.complement_basis.T
    covariance = scene.null_covariance
    amf_form = amf_quadratic_form(covariance, scene.target)
    eta_f = float(f_law.isf(alpha, p, q))
    eta_ace = float(beta_law.isf(alpha, p / 2.0, (m - p) / 2.0))
    return {
        "msd_glr": (target_projector / scene.sigma2, float(chi2.isf(alpha, p))),
        "msd_f": (target_projector / p - eta_f * complement_projector / q, 0.0),
        "amf_clairvoyant": (amf_form, float(chi2.isf(alpha, p))),
        "ace_clairvoyant": (amf_form - eta_ace * np.linalg.inv(covariance), 0.0),
    }
