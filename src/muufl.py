"""MUUFL Gulfport data access, pixel roles and hyperspectral detectors (Section 6).

Data: P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell, "MUUFL Gulfport
Hyperspectral and LiDAR Airborne Data Set", University of Florida Tech. Rep.
REP-2013-570 (2013); MIT-licensed release at
https://github.com/GatorSense/MUUFLGulfport.

Detectors for a query signature ``s`` with declared confuser spectra ``C``
and a rank-``k`` background subspace ``B``:

* ``ace``, ``amf``: background mean and covariance, no confuser knowledge;
* ``tcimf``: background correlation, unit response to ``s`` and zero response
  to every column of ``C`` (Ren and Chang, 2000);
* ``osp``: annihilate ``[B, C]`` and project onto the annihilated query
  signature (Harsanyi and Chang, 1994);
* ``declared_f`` (MSD-F): annihilate ``[B, C]`` and normalize the energy along
  the annihilated query direction by the remaining residual energy;
* ``whitened_declared_ace``: whiten by the background covariance, annihilate
  the whitened, centred confuser spectra and apply ACE;
* ``declared_ace``: the same construction without whitening;
* ``whitened_declared_f``: MSD-F applied to the whitened data and the whitened
  declared span.

``declared_ace``/``whitened_declared_ace`` and ``declared_f``/``whitened_declared_f``
are the matched-subspace whitening ablation pairs: each pair shares its declared
subspace and normalization. The four normalized annihilating detectors assign
the score 0 (non-detection) to a pixel with zero residual after annihilation,
such as a pure declared confuser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import nnls

BANDS = slice(4, 68)  # 64-band subset; the first and last four bands are noise-dominated
DECLARED_COLOURS = ("brown", "dark green", "faux vineyard green", "pea green")
SIGNATURE_NAMES = {
    "brown": "brown",
    "dark green": "dark_green",
    "faux vineyard green": "faux_vineyard_green",
    "pea green": "pea_green",
}
DETECTORS = ("ace", "amf", "tcimf", "osp", "declared_f", "whitened_declared_ace", "declared_ace", "whitened_declared_f")
ZERO_RESIDUAL_TOLERANCE = 1e-20


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    identifier: str
    colour: str
    size: float
    occlusion: int
    confidence: int
    row: int
    col: int


@dataclass(frozen=True)
class Flight:
    name: str
    cube: np.ndarray
    valid: np.ndarray
    targets: tuple[Target, ...]


def halo_radius(size: float) -> int:
    if size >= 6.0:
        return 6
    if size >= 3.0:
        return 2
    return 1


def load_flight(path: Path, name: str) -> Flight:
    hsi = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["hsi"]
    cube = np.asarray(hsi.Data, dtype=float)[:, :, BANDS]
    valid = np.asarray(hsi.valid_mask).astype(bool) & np.isfinite(cube).all(axis=2)
    gt = hsi.groundTruth
    targets = tuple(
        Target(
            identifier=str(gt.Targets_ID[i]),
            colour=str(gt.Targets_Type[i]),
            size=float(gt.Targets_Size[i]),
            occlusion=int(gt.Targets_HumanCat[i]),
            confidence=int(gt.Targets_HumanConf[i]),
            row=int(gt.Targets_rowIndices[i]) - 1,
            col=int(gt.Targets_colIndices[i]) - 1,
        )
        for i in range(len(gt.Targets_Type))
    )
    return Flight(name=name, cube=cube, valid=valid, targets=targets)


def load_signatures(path: Path) -> dict[str, np.ndarray]:
    spec = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["tgt_img_spectra"]
    names = [str(n) for n in spec.names]
    spectra = np.asarray(spec.spectra, dtype=float)[BANDS, :]
    by_name = {n: spectra[:, i] for i, n in enumerate(names)}
    return {colour: by_name[key] for colour, key in SIGNATURE_NAMES.items()}


def halo_pixels(target: Target, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h = halo_radius(target.size)
    rows = np.arange(max(0, target.row - h), min(shape[0], target.row + h + 1))
    cols = np.arange(max(0, target.col - h), min(shape[1], target.col + h + 1))
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return rr.ravel(), cc.ravel()


def background_mask(flight: Flight) -> np.ndarray:
    """Valid pixels outside every target halo."""

    mask = flight.valid.copy()
    for target in flight.targets:
        rr, cc = halo_pixels(target, mask.shape)
        mask[rr, cc] = False
    return mask


def half_image_roles(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Model: top-left quarter; certification: bottom-left quarter; evaluation: right half."""

    rows, cols = np.indices(shape)
    left = cols < shape[1] // 2
    top = rows < shape[0] // 2
    return {"model": left & top, "certification": left & ~top, "evaluation": ~left}


def interleaved_roles(shape: tuple[int, int], block: int = 16) -> dict[str, np.ndarray]:
    """Blocks of ``block x block`` pixels assigned cyclically to the three roles."""

    rows, cols = np.indices(shape)
    label = (rows // int(block) + 2 * (cols // int(block))) % 3
    return {"model": label == 0, "certification": label == 1, "evaluation": label == 2}


def window_maxima(score_map: np.ndarray, usable: np.ndarray, radius: int) -> np.ndarray:
    """Maxima of non-overlapping ``(2r+1) x (2r+1)`` windows whose pixels are all usable."""

    w = 2 * radius + 1
    rows, cols = (score_map.shape[0] // w) * w, (score_map.shape[1] // w) * w
    s = score_map[:rows, :cols].reshape(rows // w, w, cols // w, w)
    u = usable[:rows, :cols].reshape(rows // w, w, cols // w, w)
    keep = u.all(axis=(1, 3))
    return s.max(axis=(1, 3))[keep]


# ---------------------------------------------------------------------------
# Background model and detectors (larger scores mean more target evidence)
# ---------------------------------------------------------------------------


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors / np.sqrt(np.maximum(values, 1e-12 * values.max()))) @ vectors.T


def _basis(matrix: np.ndarray) -> np.ndarray:
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    return left[:, singular > 1e-10 * singular[0]]


class BackgroundModel:
    """Background statistics estimated on model-role background pixels."""

    def __init__(self, pixels: np.ndarray, subspace_rank: int):
        pixels = np.asarray(pixels, dtype=float)
        self.mean = pixels.mean(axis=0)
        centred = pixels - self.mean
        self.covariance = centred.T @ centred / (pixels.shape[0] - 1)
        self.correlation = pixels.T @ pixels / pixels.shape[0]
        _, vectors = np.linalg.eigh(self.correlation)
        self.subspace = vectors[:, ::-1][:, : int(subspace_rank)]
        self.whitener = _inverse_sqrt(self.covariance)
        self.correlation_inverse = np.linalg.inv(self.correlation)


def _zero_residual(residual_rows: np.ndarray, input_rows: np.ndarray) -> np.ndarray:
    return np.sum(residual_rows**2, axis=1) <= ZERO_RESIDUAL_TOLERANCE * np.sum(input_rows**2, axis=1)


def ace_score(x: np.ndarray, signature: np.ndarray, background: BackgroundModel) -> np.ndarray:
    z = (np.asarray(x) - background.mean) @ background.whitener.T
    a = background.whitener @ (signature - background.mean)
    inner = z @ a
    return np.sign(inner) * inner**2 / ((a @ a) * np.maximum(np.sum(z * z, axis=1), 1e-300))


def amf_score(x: np.ndarray, signature: np.ndarray, background: BackgroundModel) -> np.ndarray:
    z = (np.asarray(x) - background.mean) @ background.whitener.T
    a = background.whitener @ (signature - background.mean)
    return (z @ a) / np.sqrt(a @ a)


def tcimf_score(x: np.ndarray, signature: np.ndarray, confusers: np.ndarray, background: BackgroundModel) -> np.ndarray:
    constraint = np.column_stack([signature, confusers])
    weighted = background.correlation_inverse @ constraint
    gram = constraint.T @ weighted
    response = np.zeros(constraint.shape[1])
    response[0] = 1.0
    weights = weighted @ np.linalg.solve(gram, response)
    return np.asarray(x) @ weights


def declared_residual(confusers: np.ndarray, background: BackgroundModel) -> np.ndarray:
    """Projector annihilating the declared span: background subspace plus confuser spectra."""

    span = _basis(np.column_stack([background.subspace, confusers]))
    return np.eye(span.shape[0]) - span @ span.T


def osp_score(x: np.ndarray, signature: np.ndarray, confusers: np.ndarray, background: BackgroundModel) -> np.ndarray:
    residual = declared_residual(confusers, background)
    direction = residual @ signature
    return (np.asarray(x) @ direction) / (direction @ direction)


def declared_f_score(x: np.ndarray, signature: np.ndarray, confusers: np.ndarray, background: BackgroundModel) -> np.ndarray:
    residual = declared_residual(confusers, background)
    direction = residual @ signature
    direction = direction / np.linalg.norm(direction)
    q = int(round(np.trace(residual))) - 1
    x = np.asarray(x, dtype=float)
    values = x @ residual
    target = values @ direction
    remainder = np.maximum(np.sum(values * values, axis=1) - target**2, 1e-300)
    scores = np.sign(target) * (target**2) / (remainder / q)
    scores[_zero_residual(values, x)] = 0.0
    return scores


def whitened_declared_f_score(x: np.ndarray, signature: np.ndarray, confusers: np.ndarray, background: BackgroundModel) -> np.ndarray:
    w = background.whitener
    span = _basis(w @ np.column_stack([background.subspace, confusers]))
    residual = np.eye(span.shape[0]) - span @ span.T
    direction = residual @ (w @ signature)
    direction = direction / np.linalg.norm(direction)
    q = int(round(np.trace(residual))) - 1
    whitened = np.asarray(x) @ w.T
    values = whitened @ residual
    target = values @ direction
    remainder = np.maximum(np.sum(values * values, axis=1) - target**2, 1e-300)
    scores = np.sign(target) * (target**2) / (remainder / q)
    scores[_zero_residual(values, whitened)] = 0.0
    return scores


def _annihilated_ace(z: np.ndarray, a: np.ndarray, nulls: np.ndarray) -> np.ndarray:
    residual = np.eye(a.size) - nulls @ nulls.T
    a = residual @ a
    z_residual = z @ residual
    inner = z_residual @ a
    scores = np.sign(inner) * inner**2 / ((a @ a) * np.maximum(np.sum(z_residual * z_residual, axis=1), 1e-300))
    scores[_zero_residual(z_residual, z)] = 0.0
    return scores


def whitened_declared_ace_score(x: np.ndarray, signature: np.ndarray, confusers: np.ndarray, background: BackgroundModel) -> np.ndarray:
    z = (np.asarray(x) - background.mean) @ background.whitener.T
    a = background.whitener @ (signature - background.mean)
    nulls = _basis(background.whitener @ (confusers - background.mean[:, None]))
    return _annihilated_ace(z, a, nulls)


def declared_ace_score(x: np.ndarray, signature: np.ndarray, confusers: np.ndarray, background: BackgroundModel) -> np.ndarray:
    z = np.asarray(x) - background.mean
    a = signature - background.mean
    nulls = _basis(confusers - background.mean[:, None])
    return _annihilated_ace(z, a, nulls)


def score(name: str, x: np.ndarray, signature: np.ndarray, confusers: np.ndarray, background: BackgroundModel) -> np.ndarray:
    if name == "ace":
        return ace_score(x, signature, background)
    if name == "amf":
        return amf_score(x, signature, background)
    if name == "tcimf":
        return tcimf_score(x, signature, confusers, background)
    if name == "osp":
        return osp_score(x, signature, confusers, background)
    if name == "declared_f":
        return declared_f_score(x, signature, confusers, background)
    if name == "whitened_declared_ace":
        return whitened_declared_ace_score(x, signature, confusers, background)
    if name == "declared_ace":
        return declared_ace_score(x, signature, confusers, background)
    if name == "whitened_declared_f":
        return whitened_declared_f_score(x, signature, confusers, background)
    raise ValueError(f"unknown detector {name}")


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def fully_constrained_mixture(target: np.ndarray, confusers: np.ndarray, delta: float = 1e3) -> np.ndarray:
    """Non-negative, sum-to-one abundances minimizing ``||target - C w||`` (Heinz and Chang, 2001)."""

    scale = delta * np.max(np.abs(confusers))
    design = np.vstack([confusers, scale * np.ones((1, confusers.shape[1]))])
    response = np.append(target, scale)
    weights, _ = nnls(design, response)
    return weights / weights.sum()


def whitened_coherence(spectrum: np.ndarray, signature: np.ndarray, model: BackgroundModel) -> float:
    """Background-whitened squared cosine of a spectrum with the query signature."""

    z = model.whitener @ (spectrum - model.mean)
    a = model.whitener @ (signature - model.mean)
    return float((z @ a) ** 2 / ((z @ z) * (a @ a)))


def candidate_thresholds(model_scores: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Empirical upper quantiles of model-role scores at the given levels, plus their maximum."""

    return np.unique(np.append(np.quantile(model_scores, levels), model_scores.max()))
