#!/usr/bin/env python3
"""Mimic false attribution of mismatch-selective detectors versus normalized mimic energy (Section 5.2).

For each overlap--INR condition and number K of secondary vectors: AMF, ACE,
Kelly, ABORT, W-ABORT, Rao, the eigencanceler and MSD-F on canonical mimics
whose normalized whitened energy E spans the grid of the configuration, with
the large-sample limits of Proposition 3, and detection probability of every
detector. Every evaluation trial owns one noise draw, one secondary set, one
mimic direction and one target field, so all energies are scored on the same
draws. Produces the data of Fig. 3 and Supplementary Table S3.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from geometry import controlled_operator_pair
from selective_detection import (
    NULL_LIMITS,
    adaptive_energy_grams,
    combine,
    eigencanceler_grams,
    large_sample_mimic_limit,
    statistics_from_energies,
)
from subspace_detection import GaussianSubspaceScene, canonical_mimic_coherence, declared_geometry, f_statistic

CHUNK = 250


def score_weights(scene, geometry, columns, grams, weights, k):
    """Every detector on ``y = columns @ weights`` for one chunk of trials."""

    a, q, ea, et = grams
    out = statistics_from_energies(combine(a, weights), combine(q, weights), k)
    num, total = combine(ea, weights), combine(et, weights)
    q_dim = scene.m - scene.r - scene.p
    out["eigencanceler_f"] = (num / scene.p) / (np.maximum(total - num, np.finfo(float).tiny) / q_dim)
    out["declared_f"] = f_statistic(columns @ np.asarray(weights, dtype=float), geometry)
    return out


def chunked_scores(scene, geometry, rng, n, k, weight_sets, with_fields):
    collected = [dict() for _ in weight_sets]
    for start in range(0, n, CHUNK):
        size = min(CHUNK, n - start)
        base = scene.null(rng, size)
        if with_fields:
            direction = rng.normal(size=(size, scene.p)) @ scene.mimic_operator.T
            field = rng.normal(size=(size, scene.p)) @ scene.target.T
            columns = np.stack([base, direction, field], axis=2)
        else:
            columns = base[:, :, None]
        secondary = scene.secondary(rng, size, k)
        a, q = adaptive_energy_grams(columns, secondary, scene.target)
        ea, et = eigencanceler_grams(columns, secondary, scene.target, scene.r)
        for slot, weights in zip(collected, weight_sets):
            for name, values in score_weights(scene, geometry, columns, (a, q, ea, et), weights, k).items():
                slot.setdefault(name, []).append(values)
    return [{name: np.concatenate(parts) for name, parts in slot.items()} for slot in collected]


def run_task(task):
    config, ci, overlap, inr, k = task
    start = time.perf_counter()
    m, p, r = config["sensor_dim"], config["target_dim"], config["interference_rank"]
    alpha = float(config["alpha"])
    stream = int(config["random_stream"])
    ops = controlled_operator_pair(m, p, r, overlap, int(config["operator_seed"]))
    scene = GaussianSubspaceScene(ops.target, ops.interference, config["snr_db"], inr)
    geometry = declared_geometry(scene.target, scene.interference)
    precision = np.linalg.inv(scene.null_covariance)
    mimic_energy = float(np.trace(scene.mimic_operator.T @ precision @ scene.mimic_operator))
    kappa2 = canonical_mimic_coherence(overlap, scene.rho)

    null_rng = np.random.default_rng(np.random.SeedSequence([int(config["operator_seed"]), stream, ci, int(k), 0]))
    (null_stats,) = chunked_scores(scene, geometry, null_rng, int(config["n_null_trials_for_thresholds"]), k, [(1.0,)], with_fields=False)
    thresholds = {}
    for name, values in null_stats.items():
        ordered = np.sort(values)
        thresholds[name] = float(ordered[int(np.ceil((1 - alpha) * ordered.size)) - 1])

    # The mimic amplitude a = sqrt(E K / tr(M' Sigma^-1 M)), M = U U' H, keeps the
    # whitened mimic energy y' S_K^-1 y close to E for every K.
    energies = [float(e) for e in config["normalized_mimic_energies"]]
    amplitudes = [float(np.sqrt(e * k / mimic_energy)) for e in energies]
    weight_sets = [(1.0, 0.0, 1.0)] + [(1.0, amp, 0.0) for amp in amplitudes]
    eval_rng = np.random.default_rng(np.random.SeedSequence([int(config["operator_seed"]), stream, ci, int(k), 1]))
    scored = chunked_scores(scene, geometry, eval_rng, int(config["n_eval_trials"]), k, weight_sets, with_fields=True)
    power = {name: float(np.mean(values > thresholds[name])) for name, values in scored[0].items()}
    mimic_rows = []
    for energy, amplitude, stats in zip(energies, amplitudes, scored[1:]):
        limits = {name: large_sample_mimic_limit(name, kappa2, energy) for name in NULL_LIMITS}
        mimic_rows.append(
            {
                "energy": energy,
                "amplitude": amplitude,
                "false_attribution": {name: float(np.mean(values > thresholds[name])) for name, values in stats.items()},
                "large_sample_limit": limits,
                "accepts_in_large_sample_limit": {name: bool(limits[name] > NULL_LIMITS[name]) for name in NULL_LIMITS},
            }
        )
    return {
        "overlap": overlap,
        "inr_db": inr,
        "secondary_size": int(k),
        "kappa2": kappa2,
        "thresholds": thresholds,
        "power": power,
        "mimic": mimic_rows,
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
    (out / "selective_detectors.json").write_text(json.dumps({"rows": rows}, indent=2))
    names = ("kelly", "rao", "abort", "w_abort", "eigencanceler_f", "declared_f")
    for row in sorted(rows, key=lambda r: (r["kappa2"], r["secondary_size"])):
        worst = {n: max(e["false_attribution"][n] for e in row["mimic"]) for n in names}
        print(
            f"kappa2={row['kappa2']:.3f} K={row['secondary_size']} | worst-case false attribution "
            + " ".join(f"{n}={worst[n]:.3f}" for n in names)
        )


if __name__ == "__main__":
    main()
