#!/usr/bin/env python3
"""Detection power and mimic false attribution of hard and whitening detectors (Section 5.1).

For every overlap gamma and INR: exact detection probability versus SNR and
exact ROC of the GLR, MSD-F and the clairvoyant AMF and ACE (Proposition 4),
Monte Carlo checks of the exact values, adaptive AMF, ACE, Kelly and the
eigencanceler with independent secondary data per trial, and false
attribution of canonical mimics versus amplitude. Produces the data of Fig. 2,
Supplementary Table S1 and Supplementary Fig. S2.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_law
from scipy.stats import chi2, f

from geometry import controlled_operator_pair
from subspace_detection import (
    GaussianSubspaceScene,
    adaptive_subspace_statistics,
    batched_eigencanceler_f,
    canonical_mimic_coherence,
    clairvoyant_ace,
    clairvoyant_amf,
    declared_geometry,
    exact_detector_forms,
    f_statistic,
    gaussian_quadratic_tail,
    glr_statistic,
    soft_hard_information,
)

EXACT = ("msd_glr", "msd_f", "amf_clairvoyant", "ace_clairvoyant")
CHUNK = 1000


def analytic_thresholds(scene: GaussianSubspaceScene, alpha: float) -> dict[str, float]:
    geometry = declared_geometry(scene.target, scene.interference)
    return {
        "msd_glr": float(chi2.isf(alpha, geometry.p)),
        "msd_f": float(f.isf(alpha, geometry.p, geometry.q)),
        "amf_clairvoyant": float(chi2.isf(alpha, scene.p)),
        "ace_clairvoyant": float(beta_law.isf(alpha, scene.p / 2.0, (scene.m - scene.p) / 2.0)),
    }


def clairvoyant_statistics(scene: GaussianSubspaceScene, values: np.ndarray) -> dict[str, np.ndarray]:
    geometry = declared_geometry(scene.target, scene.interference)
    covariance = scene.null_covariance
    return {
        "msd_glr": glr_statistic(values, geometry, scene.sigma2),
        "msd_f": f_statistic(values, geometry),
        "amf_clairvoyant": clairvoyant_amf(values, covariance, scene.target),
        "ace_clairvoyant": clairvoyant_ace(values, covariance, scene.target),
    }


def adaptive_statistics(scene, values, k, rng) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {"amf_adaptive": [], "ace_adaptive": [], "kelly_adaptive": [], "eigencanceler_f": []}
    for start in range(0, values.shape[0], CHUNK):
        block = values[start : start + CHUNK]
        secondary = scene.secondary(rng, block.shape[0], k)
        stats = adaptive_subspace_statistics(block, secondary, scene.target)
        parts["amf_adaptive"].append(stats["amf"])
        parts["ace_adaptive"].append(stats["ace"])
        parts["kelly_adaptive"].append(stats["kelly"])
        parts["eigencanceler_f"].append(batched_eigencanceler_f(block, secondary, scene.target, scene.r))
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


def upper_quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(values)
    return float(ordered[int(np.ceil((1.0 - alpha) * ordered.size)) - 1])


def run_condition(task):
    config, oi, overlap, ii, inr_db = task
    start_time = time.perf_counter()
    m, p, r = config["sensor_dim"], config["target_dim"], config["interference_rank"]
    alpha = float(config["alpha"])
    ops = controlled_operator_pair(m, p, r, overlap, int(config["operator_seed"]))
    det, mim = config["detection"], config["mimic"]
    child = np.random.SeedSequence([int(config["operator_seed"]), oi, ii]).spawn(8)

    reference = GaussianSubspaceScene(ops.target, ops.interference, mim["snr_db"], inr_db)
    info = soft_hard_information(ops.target, ops.interference, reference.rho * reference.sigma2 * np.eye(r), reference.sigma2)
    kappa2 = float(info["max_mimic_coherence"])
    eta_ace = float(beta_law.isf(alpha, p / 2.0, (m - p) / 2.0))

    # Thresholds of adaptive detectors: empirical (1 - alpha) quantiles of null trials.
    threshold_rng = np.random.default_rng(child[0])
    adaptive_thresholds = {}
    for k in det["secondary_sizes"]:
        null_values = reference.null(threshold_rng, int(det["n_null_trials_for_adaptive_thresholds"]))
        stats = adaptive_statistics(reference, null_values, k, threshold_rng)
        adaptive_thresholds[k] = {name: upper_quantile(v, alpha) for name, v in stats.items()}

    # Detection probability versus SNR.
    curves = []
    sim_rng = np.random.default_rng(child[1])
    for snr_db in det["snr_db"]:
        scene = GaussianSubspaceScene(ops.target, ops.interference, snr_db, inr_db)
        forms = exact_detector_forms(scene, alpha)
        row = {"snr_db": snr_db, "exact_power": {}, "exact_size": {}, "simulated_power": {}}
        for name, (form, threshold) in forms.items():
            row["exact_power"][name] = gaussian_quadratic_tail(form, scene.target_covariance, threshold)
            row["exact_size"][name] = gaussian_quadratic_tail(form, scene.null_covariance, threshold)
        values = scene.alternative(sim_rng, int(det["n_target_trials"]))
        clair = clairvoyant_statistics(scene, values)
        thresholds = analytic_thresholds(scene, alpha)
        for name in EXACT:
            row["simulated_power"][name] = float(np.mean(clair[name] > thresholds[name]))
        for k in det["secondary_sizes"]:
            stats = adaptive_statistics(scene, values, k, sim_rng)
            for name, v in stats.items():
                row["simulated_power"][f"{name}_K{k}"] = float(np.mean(v > adaptive_thresholds[k][name]))
        curves.append(row)

    # Exact ROC at the reference SNR.
    roc_scene = GaussianSubspaceScene(ops.target, ops.interference, det["roc_snr_db"], inr_db)
    spec = det["roc_alphas"]
    roc_alphas = np.logspace(spec["log10_start"], spec["log10_stop"], int(spec["count"]))
    roc = {name: [] for name in EXACT}
    for a in roc_alphas:
        if a >= 1.0:
            for name in EXACT:
                roc[name].append(1.0)
            continue
        for name, (form, threshold) in exact_detector_forms(roc_scene, float(a)).items():
            roc[name].append(gaussian_quadratic_tail(form, roc_scene.target_covariance, threshold))

    # False attribution of canonical mimics versus amplitude.
    mimic_rows = []
    mimic_rng = np.random.default_rng(child[2])
    scene = reference
    forms = exact_detector_forms(scene, alpha)
    k = int(mim["secondary_size"])
    for amplitude in list(mim["amplitudes"]) + [float(mim["large_amplitude_for_exact_limit"])]:
        covariance = scene.mimic_covariance(amplitude)
        row = {"amplitude": amplitude, "exact_false_attribution": {}, "simulated_false_attribution": {}}
        for name, (form, threshold) in forms.items():
            row["exact_false_attribution"][name] = gaussian_quadratic_tail(form, covariance, threshold)
        if amplitude in mim["amplitudes"]:
            values = scene.mimic(mimic_rng, int(mim["n_mimic_trials"]), amplitude)
            clair = clairvoyant_statistics(scene, values)
            thresholds = analytic_thresholds(scene, alpha)
            for name in EXACT:
                row["simulated_false_attribution"][name] = float(np.mean(clair[name] > thresholds[name]))
            stats = adaptive_statistics(scene, values, k, mimic_rng)
            for name, v in stats.items():
                row["simulated_false_attribution"][f"{name}_K{k}"] = float(np.mean(v > adaptive_thresholds[k][name]))
        mimic_rows.append(row)

    return {
        "overlap": overlap,
        "inr_db": inr_db,
        "kappa2": kappa2,
        "kappa2_canonical": canonical_mimic_coherence(overlap, reference.rho),
        "eta_ace": eta_ace,
        "adaptive_thresholds": {str(k): v for k, v in adaptive_thresholds.items()},
        "power_curves": curves,
        "roc": {"alphas": roc_alphas.tolist(), "power": roc},
        "mimic": mimic_rows,
        "seconds": time.perf_counter() - start_time,
    }


def largest_deviation(config, conditions) -> dict[str, float]:
    """Largest |simulated - exact| of the exact detectors, in binomial standard errors and in probability."""

    worst = largest = 0.0
    count = 0
    for cond in conditions:
        for rows, key_e, key_s, n in (
            (cond["power_curves"], "exact_power", "simulated_power", config["detection"]["n_target_trials"]),
            (cond["mimic"], "exact_false_attribution", "simulated_false_attribution", config["mimic"]["n_mimic_trials"]),
        ):
            for row in rows:
                for name in EXACT:
                    if name not in row[key_s]:
                        continue
                    exact, sim = row[key_e][name], row[key_s][name]
                    se = np.sqrt(max(exact * (1 - exact), 1e-12) / n)
                    worst = max(worst, abs(sim - exact) / se)
                    largest = max(largest, abs(sim - exact))
                    count += 1
    return {"comparisons": count, "largest_deviation_se": float(worst), "largest_absolute_deviation": float(largest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tasks = [(config, oi, overlap, ii, inr) for oi, overlap in enumerate(config["overlaps"]) for ii, inr in enumerate(config["inr_db"])]
    with Pool(args.workers) as pool:
        conditions = pool.map(run_condition, tasks)
    results = {
        "exact_vs_simulation": largest_deviation(config, conditions),
        "conditions": conditions,
    }
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "soft_hard_detection.json").write_text(json.dumps(results, indent=2))
    for c in conditions:
        row5 = next(r for r in c["power_curves"] if r["snr_db"] == 5.0)
        mim16 = next(r for r in c["mimic"] if r["amplitude"] == 16.0)
        print(
            f"gamma={c['overlap']} INR={c['inr_db']:.0f} dB kappa2={c['kappa2']:.3f} | Pd(5 dB) "
            + " ".join(f"{n}={row5['exact_power'][n]:.3f}" for n in EXACT)
            + " | FA(a=16) " + " ".join(f"{n}={mim16['simulated_false_attribution'][n]:.3f}" for n in EXACT)
        )


if __name__ == "__main__":
    main()
