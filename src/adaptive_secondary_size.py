#!/usr/bin/env python3
"""Mimic false attribution of adaptive detectors versus the number K of secondary vectors (Section 5.2).

For each overlap--INR condition and K, thresholds are empirical (1 - alpha)
quantiles of null trials and false attribution is measured on canonical
mimics of fixed amplitude, next to the exact value of the clairvoyant ACE.
Produces the data of Supplementary Table S2.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_law

from geometry import controlled_operator_pair
from subspace_detection import (
    GaussianSubspaceScene,
    adaptive_subspace_statistics,
    amf_quadratic_form,
    batched_eigencanceler_f,
    canonical_mimic_coherence,
    gaussian_quadratic_tail,
)

CHUNK = 500


def statistics(scene, values, k, rng):
    parts = {"amf_adaptive": [], "ace_adaptive": [], "kelly_adaptive": [], "eigencanceler_f": []}
    for start in range(0, values.shape[0], CHUNK):
        block = values[start : start + CHUNK]
        secondary = scene.secondary(rng, block.shape[0], k)
        stats = adaptive_subspace_statistics(block, secondary, scene.target)
        parts["amf_adaptive"].append(stats["amf"])
        parts["ace_adaptive"].append(stats["ace"])
        parts["kelly_adaptive"].append(stats["kelly"])
        parts["eigencanceler_f"].append(batched_eigencanceler_f(block, secondary, scene.target, scene.r))
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


def run_task(task):
    config, ci, overlap, inr, k = task
    start = time.perf_counter()
    m, p, r = config["sensor_dim"], config["target_dim"], config["interference_rank"]
    alpha = float(config["alpha"])
    ops = controlled_operator_pair(m, p, r, overlap, int(config["operator_seed"]))
    scene = GaussianSubspaceScene(ops.target, ops.interference, config["snr_db"], inr)
    rng = np.random.default_rng(np.random.SeedSequence([int(config["operator_seed"]), int(config["random_stream"]), ci, int(k)]))
    null_stats = statistics(scene, scene.null(rng, int(config["n_null_trials_for_thresholds"])), k, rng)
    thresholds = {}
    for name, values in null_stats.items():
        ordered = np.sort(values)
        thresholds[name] = float(ordered[int(np.ceil((1 - alpha) * ordered.size)) - 1])
    amplitude = float(config["mimic_amplitude"])
    mimic_stats = statistics(scene, scene.mimic(rng, int(config["n_mimic_trials"]), amplitude), k, rng)
    eta_ace = float(beta_law.isf(alpha, p / 2.0, (m - p) / 2.0))
    covariance = scene.null_covariance
    ace_form = amf_quadratic_form(covariance, scene.target) - eta_ace * np.linalg.inv(covariance)
    return {
        "overlap": overlap,
        "inr_db": inr,
        "secondary_size": int(k),
        "kappa2": canonical_mimic_coherence(overlap, scene.rho),
        "eta_ace": eta_ace,
        "clairvoyant_ace_exact": gaussian_quadratic_tail(ace_form, scene.mimic_covariance(amplitude)),
        "false_attribution": {name: float(np.mean(values > thresholds[name])) for name, values in mimic_stats.items()},
        "seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tasks = [
        (config, ci, float(overlap), float(inr), int(k))
        for ci, (overlap, inr) in enumerate(config["conditions"])
        for k in config["secondary_sizes"]
    ]
    with Pool(args.workers) as pool:
        rows = pool.map(run_task, tasks)
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "adaptive_secondary_size.json").write_text(json.dumps({"rows": rows}, indent=2))
    for row in sorted(rows, key=lambda r: (r["kappa2"], r["secondary_size"])):
        fa = row["false_attribution"]
        print(
            f"gamma={row['overlap']} INR={row['inr_db']:.0f} dB kappa2={row['kappa2']:.3f} K={row['secondary_size']}: "
            f"ACE {fa['ace_adaptive']:.3f} (clairvoyant {row['clairvoyant_ace_exact']:.3f}) "
            f"Kelly {fa['kelly_adaptive']:.3f} eigencanceler {fa['eigencanceler_f']:.3f}"
        )


if __name__ == "__main__":
    main()
