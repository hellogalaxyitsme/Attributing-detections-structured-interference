#!/usr/bin/env python3
"""MSD-F under declaration error and coloured noise (Section 5.3).

Error axes: the first declared direction rotated toward the target residual or
toward the complement, omission of a declared direction, and AR(1) noise with
a correct declaration. For each cell: exact false attribution and detection
probability with the analytic F threshold (Proposition 4), the subspace error
delta and target-specific leakage eta_s, and thresholds certified by the
fixed-sequence rule on interference simulated from the declaration or on
measured interference-only data. On the coloured-noise axis, MSD-F is also
computed after whitening by the sample covariance of measured
interference-only vectors. Produces the data of Supplementary Table S4 and
Supplementary Fig. S3.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import f as f_law

from certification import candidates_from_target_scores, fixed_sequence_threshold
from geometry import controlled_operator_pair, declaration_error
from subspace_detection import (
    GaussianSubspaceScene,
    ar1_correlation,
    declared_geometry,
    f_statistic,
    gaussian_quadratic_tail,
    orthonormal_basis,
)


def inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors / np.sqrt(values)) @ vectors.T


def build_cells(config):
    cells = []
    for ii, inr in enumerate(config["inr_db"]):
        for axis, spec in config["axes"].items():
            for li, level in enumerate(spec["levels"]):
                cells.append((config, ii, inr, axis, li, level))
    return cells


def declaration(ops, axis, level):
    """Erroneous declaration (or coloured-noise correlation) of one cell."""

    u = ops.interference
    h = ops.target
    m = u.shape[0]
    residual = np.eye(m) - u @ u.T
    toward_target = residual @ h[:, 0]
    toward_target /= np.linalg.norm(toward_target)
    occupied = orthonormal_basis(np.hstack([u, h]))
    complement = orthonormal_basis(np.eye(m) - occupied @ occupied.T)[:, 0]
    declared = u.copy()
    psi = None
    if axis.startswith("rotation"):
        theta = np.deg2rad(float(level))
        direction = toward_target if axis == "rotation_toward_target" else complement
        declared[:, 0] = np.cos(theta) * u[:, 0] + np.sin(theta) * direction
    elif axis == "omitted_direction":
        declared = u[:, 1:] if level == "target_coupled" else u[:, :-1]
    elif axis == "coloured_noise":
        psi = ar1_correlation(m, float(level))
    return declared, psi


def run_cell(task):
    config, ii, inr, axis, li, level = task
    start = time.perf_counter()
    alpha, beta = float(config["alpha"]), float(config["beta"])
    m, p, r = config["sensor_dim"], config["target_dim"], config["interference_rank"]
    ops = controlled_operator_pair(m, p, r, float(config["overlap"]), int(config["operator_seed"]))
    declared, psi = declaration(ops, axis, level)
    truth = GaussianSubspaceScene(ops.target, ops.interference, config["snr_db"], inr, psi)
    model = GaussianSubspaceScene(ops.target, declared, config["snr_db"], inr)
    geometry = declared_geometry(ops.target, declared)
    eta = float(f_law.isf(alpha, geometry.p, geometry.q))
    form = (
        geometry.target_basis @ geometry.target_basis.T / geometry.p
        - eta * geometry.complement_basis @ geometry.complement_basis.T / geometry.q
    )
    exact = {
        "false_attribution_random": gaussian_quadratic_tail(form, truth.null_covariance),
        "false_attribution_true_mimic": gaussian_quadratic_tail(form, truth.mimic_covariance(1.0)),
        "power": gaussian_quadratic_tail(form, truth.target_covariance),
    }
    error = declaration_error(ops.target, ops.interference, declared)

    axis_index = list(config["axes"]).index(axis)
    base = [int(config["operator_seed"]), ii, axis_index, li]
    test_rng = np.random.default_rng(np.random.SeedSequence(base + [1_000_000]))
    n_null = int(config["n_test_null_per_family"])
    test_random = truth.null(test_rng, n_null)
    test_mimic = truth.mimic(test_rng, n_null, 1.0)
    test_target = truth.alternative(test_rng, int(config["n_test_target"]))

    def summarize(score_random, score_mimic, score_target, threshold):
        if not np.isfinite(threshold):
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(score_random > threshold)),
            float(np.mean(score_mimic > threshold)),
            float(np.mean(score_target > threshold)),
        )

    fixed_scores = [f_statistic(v, geometry) for v in (test_random, test_mimic, test_target)]
    analytic_sim = summarize(*fixed_scores, eta)

    spec = config["candidate_sensitivities"]
    sensitivities = np.linspace(spec["start"], spec["stop"], int(spec["count"]))
    rules = ["certified_on_model", "certified_on_measured"] + (["whitened_certified_on_measured"] if axis == "coloured_noise" else [])
    records = {rule: {"random": [], "mimic": [], "power": []} for rule in rules}
    n_cal = int(config["n_interference_calibration_per_family"])
    for replicate in range(int(config["calibration_replicates"])):
        rng = np.random.default_rng(np.random.SeedSequence(base + [replicate]))
        target_split = truth.alternative(rng, int(config["n_target_calibration"]))
        declared_mimic = model.mimic(rng, n_cal, 1.0)
        model_random = model.null(rng, n_cal)
        measured_random = truth.null(rng, n_cal)
        families = {
            "certified_on_model": [model_random, declared_mimic],
            "certified_on_measured": [measured_random, declared_mimic],
        }
        for rule in ("certified_on_model", "certified_on_measured"):
            candidates = candidates_from_target_scores(f_statistic(target_split, geometry), sensitivities)
            threshold, _ = fixed_sequence_threshold(candidates, [f_statistic(v, geometry) for v in families[rule]], alpha, beta)
            fa_random, fa_mimic, power = summarize(*fixed_scores, threshold)
            records[rule]["random"].append(fa_random)
            records[rule]["mimic"].append(fa_mimic)
            records[rule]["power"].append(power)
        if "whitened_certified_on_measured" in rules:
            secondary = truth.null(rng, int(config["n_secondary_for_whitening"]))
            whitener = inverse_sqrt(secondary.T @ secondary / secondary.shape[0])
            w_geometry = declared_geometry(whitener @ ops.target, whitener @ declared)

            def w_score(values):
                return f_statistic(values @ whitener.T, w_geometry)

            candidates = candidates_from_target_scores(w_score(target_split), sensitivities)
            threshold, _ = fixed_sequence_threshold(candidates, [w_score(measured_random), w_score(declared_mimic)], alpha, beta)
            fa_random, fa_mimic, power = summarize(w_score(test_random), w_score(test_mimic), w_score(test_target), threshold)
            records["whitened_certified_on_measured"]["random"].append(fa_random)
            records["whitened_certified_on_measured"]["mimic"].append(fa_mimic)
            records["whitened_certified_on_measured"]["power"].append(power)

    summary = {}
    for rule, values in records.items():
        random_values = np.asarray(values["random"])
        summary[rule] = {
            "mean_false_attribution_random": float(random_values.mean()),
            "violation_rate_random": float(np.mean(random_values > alpha)),
            "mean_false_attribution_true_mimic": float(np.mean(values["mimic"])),
            "mean_power": float(np.mean(values["power"])),
        }
    return {
        "inr_db": inr,
        "axis": axis,
        "level": level,
        "complement_dim": geometry.q,
        "projector_distance": error["projector_distance"],
        "target_specific_leakage": error["target_specific_leakage"],
        "exact_analytic": exact,
        "simulated_analytic": dict(zip(("false_attribution_random", "false_attribution_true_mimic", "power"), analytic_sim)),
        "certified": summary,
        "seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with Pool(args.workers) as pool:
        cells = pool.map(run_cell, build_cells(config))
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "declaration_error.json").write_text(json.dumps({"cells": cells}, indent=2))
    for c in cells:
        cert = c["certified"]["certified_on_measured"]
        print(
            f"INR={c['inr_db']:.0f} dB {c['axis']} {c['level']}: exact analytic FA {c['exact_analytic']['false_attribution_random']:.4f}, "
            f"Pd {c['exact_analytic']['power']:.3f} | certified on measured data: FA {cert['mean_false_attribution_random']:.4f}, "
            f"violation rate {cert['violation_rate_random']:.3f}, Pd {cert['mean_power']:.3f}"
        )


if __name__ == "__main__":
    main()
