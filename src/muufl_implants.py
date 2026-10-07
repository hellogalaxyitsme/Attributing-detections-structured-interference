#!/usr/bin/env python3
"""Implanted targets and confuser mimics on real MUUFL background (Section 6.1).

Spectra are implanted into evaluation-role background pixels of each held-out
flight (interleaved split) by linear mixing, ``(1 - f) y + f s``. Implants:

* the query signature (detection probability);
* the nearest declared confuser and the fully constrained confuser mixture
  closest to the query (false attribution of declared-subspace mimics);
* confuser spectra measured in the held-out flight (the 3 x 3 core pixels of
  every eligible 3 m cloth);
* the nearest declared confuser rotated out of the declared span, toward the
  residual of the query signature or in random directions.

Thresholds are the fixed-sequence thresholds of the interleaved split. Also
reported: the background-whitened coherence of the nearest confuser with the
query and its variation over half subsamples of the model pixels and under the
half-image background model. Produces the data of Fig. 4, Table 3 and
Supplementary Tables S11--S12.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from certification import fixed_sequence_threshold
from muufl import (
    DECLARED_COLOURS,
    BackgroundModel,
    _basis,
    angle_deg,
    background_mask,
    candidate_thresholds,
    fully_constrained_mixture,
    half_image_roles,
    interleaved_roles,
    load_flight,
    load_signatures,
    score,
    whitened_coherence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_root = Path(config["data_root"])
    signatures = load_signatures(data_root / config["signatures_file"])
    cert = config["certification"]
    alpha, beta = float(cert["per_pixel_alpha"]), float(cert["beta"])
    levels = 1.0 - 10.0 ** np.linspace(-1.0, -5.0, 200)
    rank = int(config["background_subspace_rank"])
    occlusion = set(config["eligibility"]["occlusion_categories"])
    imp = config["implants"]
    abundances = [float(f) for f in imp["abundances"]]
    measured_abundances = [float(f) for f in imp["measured_abundances"]]
    pert = imp["perturbation"]
    variability = config["coherence_variability"]

    cells, coherence = [], []
    for flight_index, (flight_name, file_name) in enumerate(config["flights"].items()):
        flight = load_flight(data_root / file_name, flight_name)
        shape = flight.valid.shape
        background = background_mask(flight)
        pixels = {role: flight.cube[background & mask] for role, mask in interleaved_roles(shape, 16).items()}
        model = BackgroundModel(pixels["model"], rank)
        half_pixels = flight.cube[background & half_image_roles(shape)["model"]]
        half_model = BackgroundModel(half_pixels, rank)

        rng = np.random.default_rng(np.random.SeedSequence([int(imp["host_seed"]), flight_index]))
        n_hosts = min(int(imp["host_pixels_per_flight"]), pixels["evaluation"].shape[0])
        hosts = pixels["evaluation"][rng.choice(pixels["evaluation"].shape[0], size=n_hosts, replace=False)]

        measured = {c: [] for c in DECLARED_COLOURS}
        for target in flight.targets:
            if target.size == 3.0 and target.occlusion in occlusion and target.colour in measured:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        r, c = target.row + dr, target.col + dc
                        if 0 <= r < shape[0] and 0 <= c < shape[1] and flight.valid[r, c]:
                            measured[target.colour].append((target.identifier, flight.cube[r, c].copy()))

        dir_rng = np.random.default_rng(int(pert["random_seed"]) + flight_index)
        for colour in DECLARED_COLOURS:
            signature = signatures[colour]
            others = [c for c in DECLARED_COLOURS if c != colour]
            confusers = np.column_stack([signatures[c] for c in others])
            angles = {c: angle_deg(signature, signatures[c]) for c in others}
            nearest = min(angles, key=angles.get)
            weights = fully_constrained_mixture(signature, confusers)
            library = {"target": signature, "nearest_confuser": signatures[nearest], "confuser_mixture": confusers @ weights}
            span = _basis(np.column_stack([model.subspace, confusers]))
            resid = np.eye(span.shape[0]) - span @ span.T
            toward_target = resid @ signature
            toward_target = toward_target / np.linalg.norm(toward_target)
            randoms = []
            for _ in range(int(pert["random_directions"])):
                d = resid @ dir_rng.normal(size=signature.size)
                randoms.append(d / np.linalg.norm(d))
            c0 = signatures[nearest]
            c_norm = np.linalg.norm(c0)

            def rotate(direction, theta):
                t = np.radians(theta)
                return c_norm * (np.cos(t) * c0 / c_norm + np.sin(t) * direction)

            sub_rng = np.random.default_rng(int(variability["seed"]) + flight_index)
            subsamples = []
            for _ in range(int(variability["half_subsamples"])):
                idx = sub_rng.choice(pixels["model"].shape[0], size=pixels["model"].shape[0] // 2, replace=False)
                subsamples.append(whitened_coherence(c0, signature, BackgroundModel(pixels["model"][idx], rank)))
            coherence.append(
                {
                    "flight": flight_name,
                    "query": colour,
                    "nearest_confuser": nearest,
                    "nearest_angle_deg": angles[nearest],
                    "mixture_weights": dict(zip(others, [float(w) for w in weights])),
                    "kappa2": whitened_coherence(c0, signature, model),
                    "kappa2_half_subsamples_min": float(min(subsamples)),
                    "kappa2_half_subsamples_max": float(max(subsamples)),
                    "kappa2_half_image_model": whitened_coherence(c0, signature, half_model),
                    "measured_spectra": [
                        {
                            "colour": mc,
                            "cloth": cloth,
                            "angle_to_declared_deg": angle_deg(spec, signatures[mc]),
                            "outside_declared_span_fraction": float(np.sum((resid @ spec) ** 2) / np.sum(spec**2)),
                        }
                        for mc in DECLARED_COLOURS
                        for cloth, spec in measured[mc]
                    ],
                }
            )

            for detector in config["detectors"]:
                model_scores = score(detector, pixels["model"], signature, confusers, model)
                cert_scores = score(detector, pixels["certification"], signature, confusers, model)
                threshold, _ = fixed_sequence_threshold(candidate_thresholds(model_scores, levels), [cert_scores], alpha, beta)
                host_accept = score(detector, hosts, signature, confusers, model) > threshold

                def accept(spectrum, f):
                    return score(detector, (1.0 - f) * hosts + f * spectrum[None, :], signature, confusers, model) > threshold

                curves = {}
                for kind, spectrum in library.items():
                    curves[kind] = []
                    for f in abundances:
                        a = accept(spectrum, f)
                        curves[kind].append({"abundance": f, "acceptance": float(np.mean(a)), "agreement_with_host": float(np.mean(a == host_accept))})
                measured_rows = [
                    {"colour": mc, "cloth": cloth, "abundance": f, "acceptance": float(np.mean(accept(spec, f)))}
                    for mc in DECLARED_COLOURS
                    for cloth, spec in measured[mc]
                    for f in measured_abundances
                ]
                perturbed_rows = []
                for theta in pert["angles_deg"]:
                    for f in pert["abundances"]:
                        random_acceptance = [float(np.mean(accept(rotate(d, theta), f))) for d in randoms]
                        perturbed_rows.append(
                            {
                                "angle_deg": theta,
                                "abundance": f,
                                "toward_target_residual": float(np.mean(accept(rotate(toward_target, theta), f))),
                                "random_mean": float(np.mean(random_acceptance)),
                                "random_max": float(np.max(random_acceptance)),
                            }
                        )
                cells.append(
                    {
                        "flight": flight_name,
                        "query": colour,
                        "detector": detector,
                        "threshold": float(threshold),
                        "host_acceptance": float(np.mean(host_accept)),
                        "nearest_confuser": nearest,
                        "curves": curves,
                        "measured": measured_rows,
                        "perturbed": perturbed_rows,
                    }
                )

    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "muufl_implants.json").write_text(json.dumps({"coherence": coherence, "implants": cells}, indent=2))
    for c in coherence:
        print(f"{c['flight']} query {c['query']:<20} nearest {c['nearest_confuser']:<20} whitened coherence {c['kappa2']:.3f}")
    for detector in config["detectors"]:
        pd = [np.mean([next(r["acceptance"] for r in c["curves"]["target"] if r["abundance"] == 0.2)
                       for c in cells if c["detector"] == detector and c["query"] == q]) for q in DECLARED_COLOURS]
        print(f"{detector:<22} Pd at f = 0.2: " + " ".join(f"{v:.2f}" for v in pd))


if __name__ == "__main__":
    main()
