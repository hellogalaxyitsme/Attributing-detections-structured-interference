#!/usr/bin/env python3
"""Threshold rules for MSD-F on matched candidate grids (Section 5.4).

For every seed, overlap and noise law, large test sets are fixed and the
calibration step is repeated with fresh target and interference calibration
data. Rules: the analytic F threshold (and the chi-square threshold of the
known-noise GLR), Bonferroni certification at beta/(LJ) and beta/L, and the
fixed-sequence rule, each on grids of 8 and 199 candidate thresholds. Outputs
mean detection probability, realized false attribution (maximum over the
random-interference and mimic test families) and violation rates, and the
paired sign-flip tests of the fixed-sequence gain over the beta/L rule under
Gaussian noise with Holm adjustment. Produces the data of Table 2,
Supplementary Tables S5--S6 and Supplementary Fig. S4.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import chi2, f

from certification import candidates_from_target_scores, deploy_rules, exceedances
from geometry import controlled_operator_pair
from statistics_tools import holm_adjust, paired_sign_flip_pvalue
from subspace_detection import declared_geometry, draw_noise, f_statistic, glr_statistic, zero_residual_mask

SCORES = ("msd_f", "msd_glr")
RULES = ("bonferroni_LJ", "bonferroni_L", "fixed_sequence")
GRIDS = ("coarse", "fine")


class Scene:
    """Canonical scene with a configurable noise law; mimics are canonical mimics on random interference."""

    def __init__(self, config: dict, seed: int, overlap: float, law: str):
        self.m = int(config["sensor_dim"])
        self.p = int(config["target_dim"])
        self.r = int(config["interference_rank"])
        self.ops = controlled_operator_pair(self.m, self.p, self.r, overlap, seed)
        self.geometry = declared_geometry(self.ops.target, self.ops.interference)
        snr = 10.0 ** (float(config["snr_db"]) / 10.0)
        inr = 10.0 ** (float(config["inr_db"]) / 10.0)
        self.sigma2 = self.p / (self.m * snr)
        self.interference_sd = np.sqrt(inr * self.sigma2)
        self.law = law

    def _noise(self, rng, n):
        return np.sqrt(self.sigma2) * draw_noise(rng, (n, self.m), self.law)

    def target(self, rng, n):
        s = rng.normal(size=(n, self.p))
        c = self.interference_sd * rng.normal(size=(n, self.r))
        return s @ self.ops.target.T + c @ self.ops.interference.T + self._noise(rng, n)

    def random_interference(self, rng, n):
        c = self.interference_sd * rng.normal(size=(n, self.r))
        return c @ self.ops.interference.T + self._noise(rng, n)

    def mimic(self, rng, n):
        s = rng.normal(size=(n, self.p))
        mimic_coefficients = s @ (self.ops.interference.T @ self.ops.target).T
        c = self.interference_sd * rng.normal(size=(n, self.r)) + mimic_coefficients
        return c @ self.ops.interference.T + self._noise(rng, n)

    def score(self, name, values):
        if name == "msd_f":
            return f_statistic(values, self.geometry)
        return glr_statistic(values, self.geometry, self.sigma2)

    def analytic_threshold(self, name, alpha):
        if name == "msd_f":
            return float(f.isf(alpha, self.geometry.p, self.geometry.q))
        return float(chi2.isf(alpha, self.geometry.p))

    def zero_count(self, values) -> int:
        return int(np.sum(zero_residual_mask(values, self.geometry)))


def run_cell(task):
    config, seed, overlap_index, overlap, law_index, law = task
    start = time.perf_counter()
    scene = Scene(config, seed, overlap, law)
    alpha = float(config["alpha"])
    beta = float(config["beta"])
    grids = {
        "coarse": np.asarray(config["coarse_candidate_sensitivities"], dtype=float),
        "fine": np.linspace(
            config["fine_candidate_sensitivities"]["start"],
            config["fine_candidate_sensitivities"]["stop"],
            int(config["fine_candidate_sensitivities"]["count"]),
        ),
    }
    test_rng = np.random.default_rng(np.random.SeedSequence([seed, overlap_index, law_index, 10_000]))
    n_null = int(config["n_test_null_per_family"])
    test_null = [scene.random_interference(test_rng, n_null), scene.mimic(test_rng, n_null)]
    test_target = scene.target(test_rng, int(config["n_test_target"]))
    zero_test_null = sum(scene.zero_count(values) for values in test_null)
    sorted_null = {s: [np.sort(scene.score(s, values)) for values in test_null] for s in SCORES}
    sorted_target = {s: np.sort(scene.score(s, test_target)) for s in SCORES}

    def realized(score, threshold):
        if not np.isfinite(threshold):
            return 0.0
        return max(float(exceedances(v, np.array([threshold]))[0]) / v.size for v in sorted_null[score])

    def power(score, threshold):
        if not np.isfinite(threshold):
            return 0.0
        target = sorted_target[score]
        return float(exceedances(target, np.array([threshold]))[0]) / target.size

    summary = {}
    for score in SCORES:
        threshold = scene.analytic_threshold(score, alpha)
        value = realized(score, threshold)
        summary[f"{score}|analytic"] = {
            "mean_power": power(score, threshold),
            "mean_realized_false_attribution": value,
            "violation_rate": float(value > alpha),
            "abstention_rate": 0.0,
        }

    records = {(rule, grid): {"power": [], "realized": [], "abstain": []} for rule in RULES for grid in GRIDS}
    n_target = int(config["n_target_calibration"])
    n_cal = int(config["n_interference_calibration_per_family"])
    for replicate in range(int(config["calibration_replicates"])):
        rng = np.random.default_rng(np.random.SeedSequence([seed, overlap_index, law_index, replicate]))
        target_split = scene.target(rng, n_target)
        families = [scene.random_interference(rng, n_cal), scene.mimic(rng, n_cal)]
        target_scores = scene.score("msd_f", target_split)
        fam_scores = [np.sort(scene.score("msd_f", fam)) for fam in families]
        for grid, sens in grids.items():
            thresholds = deploy_rules(candidates_from_target_scores(target_scores, sens), fam_scores, alpha, beta)
            for rule, threshold in thresholds.items():
                records[(rule, grid)]["power"].append(power("msd_f", threshold))
                records[(rule, grid)]["realized"].append(realized("msd_f", threshold))
                records[(rule, grid)]["abstain"].append(float(not np.isfinite(threshold)))
    for (rule, grid), rec in records.items():
        realized_values = np.asarray(rec["realized"])
        summary[f"{rule}|{grid}"] = {
            "mean_power": float(np.mean(rec["power"])),
            "mean_realized_false_attribution": float(np.mean(realized_values)),
            "violation_rate": float(np.mean(realized_values > alpha)),
            "violations": int(np.sum(realized_values > alpha)),
            "abstention_rate": float(np.mean(rec["abstain"])),
        }
    return {
        "seed": seed,
        "overlap": overlap,
        "law": law,
        "zero_residual_test_null": zero_test_null,
        "test_null_size": 2 * n_null,
        "summary": summary,
        "seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tasks = [
        (config, seed, oi, overlap, li, law)
        for seed in config["seeds"]
        for oi, overlap in enumerate(config["overlaps"])
        for li, law in enumerate(config["noise_laws"])
    ]
    with Pool(args.workers) as pool:
        cells = pool.map(run_cell, tasks)
    alpha = float(config["alpha"])
    replicates = int(config["calibration_replicates"])

    cell_means = {}
    gains = []
    validity = []
    for law in config["noise_laws"]:
        for overlap in config["overlaps"]:
            chosen = sorted((c for c in cells if c["law"] == law and c["overlap"] == overlap), key=lambda c: c["seed"])
            for key in chosen[0]["summary"]:
                values = [c["summary"][key] for c in chosen]
                entry = {
                    metric: float(np.mean([v[metric] for v in values]))
                    for metric in ("mean_power", "mean_realized_false_attribution", "violation_rate", "abstention_rate")
                }
                entry["realized_over_alpha_min"] = float(min(v["mean_realized_false_attribution"] for v in values) / alpha)
                entry["realized_over_alpha_max"] = float(max(v["mean_realized_false_attribution"] for v in values) / alpha)
                cell_means[f"{law}|{overlap}|{key}"] = entry
            for rule in RULES:
                for grid in GRIDS:
                    violations = sum(c["summary"][f"{rule}|{grid}"]["violations"] for c in chosen)
                    validity.append({"law": law, "overlap": overlap, "rule": rule, "grid": grid,
                                     "pooled_violation_rate": violations / (len(chosen) * replicates)})
            for grid in GRIDS:
                diffs = np.asarray(
                    [c["summary"][f"fixed_sequence|{grid}"]["mean_power"] - c["summary"][f"bonferroni_L|{grid}"]["mean_power"] for c in chosen]
                )
                gains.append({"law": law, "overlap": overlap, "grid": grid, "mean_gain": float(diffs.mean()),
                              "min_gain": float(diffs.min()), "positive_seeds": int(np.sum(diffs > 0)), "seeds": int(diffs.size)})

    tests = []
    gaussian_cells = [(overlap, grid) for grid in GRIDS[::-1] for overlap in config["overlaps"]] if "gaussian" in config["noise_laws"] else []
    for overlap, grid in gaussian_cells:
        chosen = sorted((c for c in cells if c["law"] == "gaussian" and c["overlap"] == overlap), key=lambda c: c["seed"])
        diffs = np.asarray(
            [c["summary"][f"fixed_sequence|{grid}"]["mean_power"] - c["summary"][f"bonferroni_L|{grid}"]["mean_power"] for c in chosen]
        )
        tests.append({"overlap": overlap, "grid": grid, "mean_gain": float(diffs.mean()),
                      "positive_seeds": int(np.sum(diffs > 0)), "p_value": paired_sign_flip_pvalue(diffs, "greater")})
    if tests:
        for test, adjusted in zip(tests, holm_adjust(np.asarray([t["p_value"] for t in tests]))):
            test["holm_p_value"] = float(adjusted)

    bursty = [c for c in cells if c["law"] == "bursty"]
    results = {
        "gaussian_fixed_sequence_vs_bonferroni_L": tests,
        "fixed_sequence_gain_over_bonferroni_L": gains,
        "validity": validity,
        "bursty_zero_residual_fraction": (
            sum(c["zero_residual_test_null"] for c in bursty) / sum(c["test_null_size"] for c in bursty) if bursty else None
        ),
        "cell_means": cell_means,
        "cells": cells,
    }
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "certification_efficiency.json").write_text(json.dumps(results, indent=2))
    for law in config["noise_laws"]:
        for overlap in config["overlaps"]:
            row = " ".join(
                f"{rule}/{grid}={cell_means[f'{law}|{overlap}|{rule}|{grid}']['mean_power']:.3f}" for rule in RULES for grid in GRIDS
            )
            analytic = cell_means[f"{law}|{overlap}|msd_f|analytic"]
            print(f"{law} gamma={overlap}: analytic Pd={analytic['mean_power']:.3f} FA/alpha={analytic['mean_realized_false_attribution'] / alpha:.2f} | {row}")
    for t in tests:
        print(f"Gaussian gamma={t['overlap']} {t['grid']} grid: gain {t['mean_gain']:.3f}, Holm p = {t['holm_p_value']:.2g}")


if __name__ == "__main__":
    main()
