#!/usr/bin/env python3
"""Emplaced cloth targets on the held-out MUUFL flights (Section 6.2).

For each held-out flight, pixel split (interleaved 16 x 16 blocks or half
image), query colour and detector: the fixed-sequence threshold certified on
certification-role background pixels at the per-pixel level alpha, counts and
nominal Clopper--Pearson intervals of the realized background false-alarm rate
on evaluation-role pixels, decisions on the emplaced targets (maximum score
over the target halo) at the certified threshold, at a post hoc oracle
threshold matched to the evaluation-background rate, and at window-level
thresholds certified on background-window maxima of matched size, and the
size-matched detection AUC. Also computed: repeated-measures comparisons of
the declared-confuser decisions (discordant counts per flight and exact
sign-flip tests over physical cloths with Holm adjustment), the sensitivity of
MSD-F to the rank of the background subspace, and the pairwise spectral angles
of the declared signatures. Produces the data of Table 4 and Supplementary
Tables S7--S10 and S13--S14 and Supplementary Fig. S5.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from certification import fixed_sequence_threshold
from muufl import (
    DECLARED_COLOURS,
    BackgroundModel,
    angle_deg,
    background_mask,
    candidate_thresholds,
    half_image_roles,
    halo_pixels,
    halo_radius,
    interleaved_roles,
    load_flight,
    load_signatures,
    score,
    window_maxima,
)
from statistics_tools import clopper_pearson, holm_adjust, paired_sign_flip_pvalue


def analyse_cell(flight, split, roles, background, halos, signature, confusers, colour, detector, model, pixels, config, full):
    """Certified threshold, emplaced decisions and (if ``full``) oracle and window-level results of one cell."""

    cert = config["certification"]
    alpha, beta = float(cert["per_pixel_alpha"]), float(cert["beta"])
    levels = 1.0 - 10.0 ** np.linspace(-1.0, -5.0, 200)
    sizes = set(config["eligibility"]["sizes_m"])
    occlusion = set(config["eligibility"]["occlusion_categories"])
    shape = flight.valid.shape

    model_scores = score(detector, pixels["model"], signature, confusers, model)
    cert_scores = score(detector, pixels["certification"], signature, confusers, model)
    threshold, certified = fixed_sequence_threshold(candidate_thresholds(model_scores, levels), [cert_scores], alpha, beta)
    finite = bool(np.isfinite(threshold))
    eval_scores = score(detector, pixels["evaluation"], signature, confusers, model)
    k_cert = int(np.sum(cert_scores > threshold)) if finite else 0
    k_eval = int(np.sum(eval_scores > threshold)) if finite else 0

    oracle = float(np.quantile(eval_scores, 1.0 - alpha))
    obj_thresholds, obj_rows = {}, {}
    if full:
        obj = config["window_level"]
        w_alpha, w_beta = float(obj["window_alpha"]), float(obj["beta"])
        w_levels = 1.0 - 10.0 ** np.linspace(-1.0, -3.5, 100)
        flat = flight.cube.reshape(-1, flight.cube.shape[2])
        score_map = score(detector, flat, signature, confusers, model).reshape(shape)
        usable = {role: background & mask for role, mask in roles.items()}
        for radius in obj["halo_radii"]:
            wmax = {role: window_maxima(score_map, usable[role], radius) for role in roles}
            t_obj, c_obj = fixed_sequence_threshold(candidate_thresholds(wmax["model"], w_levels), [wmax["certification"]], w_alpha, w_beta)
            fin = bool(np.isfinite(t_obj))
            k_obj = int(np.sum(wmax["evaluation"] > t_obj)) if fin else 0
            obj_thresholds[radius] = t_obj
            obj_rows[str(radius)] = {
                "windows": {role: int(v.size) for role, v in wmax.items()},
                "threshold": float(t_obj) if fin else None,
                "certified_candidates": int(c_obj),
                "evaluation_exceedances": k_obj,
                "realized_window_far": k_obj / wmax["evaluation"].size,
                "evaluation_window_maxima": wmax["evaluation"],
            }

    rows = []
    for target in flight.targets:
        rr, cc = halos[target.identifier]
        valid = flight.valid[rr, cc]
        values = score(detector, flight.cube[rr[valid], cc[valid]], signature, confusers, model)
        halo_max = float(values.max()) if values.size else float("-inf")
        t_obj = obj_thresholds.get(halo_radius(target.size), float("inf"))
        rows.append(
            {
                "split": split,
                "flight": flight.name,
                "query": colour,
                "detector": detector,
                "target_id": target.identifier,
                "target_colour": target.colour,
                "size": target.size,
                "occlusion": target.occlusion,
                "eligible": target.size in sizes and target.occlusion in occlusion,
                "halo_max": halo_max,
                "accepted": bool(finite and halo_max > threshold),
                "accepted_oracle": bool(halo_max > oracle),
                "accepted_window_level": bool(np.isfinite(t_obj) and halo_max > t_obj),
            }
        )
    eligible = [r for r in rows if r["eligible"]]
    pos = [r for r in eligible if r["target_colour"] == colour]
    neg = [r for r in eligible if r["target_colour"] in DECLARED_COLOURS and r["target_colour"] != colour]
    summary = {
        "split": split,
        "flight": flight.name,
        "rank": model.subspace.shape[1],
        "query": colour,
        "detector": detector,
        "n_model": int(model_scores.size),
        "n_certification": int(cert_scores.size),
        "n_evaluation": int(eval_scores.size),
        "threshold": float(threshold) if finite else None,
        "certified_candidates": int(certified),
        "certification_exceedances": k_cert,
        "evaluation_exceedances": k_eval,
        "realized_background_far": k_eval / eval_scores.size,
        "far_ci95_nominal": list(clopper_pearson(k_eval, eval_scores.size)),
        "positives": len(pos),
        "positives_accepted": sum(r["accepted"] for r in pos),
        "confusers": len(neg),
        "confusers_accepted": sum(r["accepted"] for r in neg),
    }
    if full:
        auc_terms = []
        for r in pos:
            ref = obj_rows[str(halo_radius(r["size"]))]["evaluation_window_maxima"]
            auc_terms.append(float(np.mean(ref < r["halo_max"]) + 0.5 * np.mean(ref == r["halo_max"])))
        for radius in obj_rows:
            obj_rows[radius].pop("evaluation_window_maxima")
        summary.update(
            {
                "oracle_threshold": oracle,
                "positives_accepted_oracle": sum(r["accepted_oracle"] for r in pos),
                "confusers_accepted_oracle": sum(r["accepted_oracle"] for r in neg),
                "positives_accepted_window_level": sum(r["accepted_window_level"] for r in pos),
                "confusers_accepted_window_level": sum(r["accepted_window_level"] for r in neg),
                "size_matched_detection_auc": float(np.mean(auc_terms)) if auc_terms else None,
                "window_level": obj_rows,
            }
        )
    return summary, rows


def repeated_measures(decisions, comparisons):
    """Declared-confuser decisions of two detectors: discordant counts and sign-flip tests over cloths."""

    entries = []
    for split in ("interleaved", "half_image"):
        for annihilator, other in comparisons:

            def rows_of(det):
                rs = [
                    d for d in decisions
                    if d["split"] == split and d["detector"] == det and d["eligible"]
                    and d["target_colour"] in DECLARED_COLOURS and d["target_colour"] != d["query"]
                ]
                rs.sort(key=lambda d: (d["flight"], d["query"], d["target_id"]))
                return rs

            a_rows, o_rows = rows_of(annihilator), rows_of(other)
            if [(d["flight"], d["query"], d["target_id"]) for d in a_rows] != [(d["flight"], d["query"], d["target_id"]) for d in o_rows]:
                raise RuntimeError("confuser decisions of the two detectors are not paired")

            def discordant(a, o):
                a, o = np.asarray(a, dtype=bool), np.asarray(o, dtype=bool)
                return {"first_accepted": int(a.sum()), "second_accepted": int(o.sum()),
                        "only_first": int(np.sum(a & ~o)), "only_second": int(np.sum(o & ~a))}

            entry = {"split": split, "first": annihilator, "second": other,
                     "pooled": discordant([d["accepted"] for d in a_rows], [d["accepted"] for d in o_rows]), "per_flight": {}}
            for fl in sorted({d["flight"] for d in a_rows}):
                idx = [i for i, d in enumerate(a_rows) if d["flight"] == fl]
                entry["per_flight"][fl] = discordant([a_rows[i]["accepted"] for i in idx], [o_rows[i]["accepted"] for i in idx])
            clusters = {}
            for a, o in zip(a_rows, o_rows):
                clusters[a["target_id"]] = clusters.get(a["target_id"], 0) + int(o["accepted"]) - int(a["accepted"])
            diffs = np.asarray(list(clusters.values()), dtype=float)
            entry["cloths"] = len(clusters)
            entry["cloths_with_difference"] = int(np.sum(diffs != 0))
            entry["cloth_sign_flip_p"] = paired_sign_flip_pvalue(diffs, "greater") if np.any(diffs != 0) else 1.0
            entries.append(entry)
    for entry, adjusted in zip(entries, holm_adjust(np.asarray([e["cloth_sign_flip_p"] for e in entries]))):
        entry["cloth_sign_flip_p_holm"] = float(adjusted)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_root = Path(config["data_root"])
    signatures = load_signatures(data_root / config["signatures_file"])
    primary = int(config["background_subspace_rank"])
    ranks = [primary] + [int(r) for r in config["rank_sensitivity"]["ranks"]]

    angles = {f"{a}|{b}": angle_deg(signatures[a], signatures[b]) for i, a in enumerate(DECLARED_COLOURS) for b in DECLARED_COLOURS[i + 1 :]}
    summaries, decisions, rank_rows = [], [], []
    for flight_name, file_name in config["flights"].items():
        flight = load_flight(data_root / file_name, flight_name)
        shape = flight.valid.shape
        background = background_mask(flight)
        halos = {t.identifier: halo_pixels(t, shape) for t in flight.targets}
        for split, roles in (("interleaved", interleaved_roles(shape, 16)), ("half_image", half_image_roles(shape))):
            pixels = {role: flight.cube[background & mask] for role, mask in roles.items()}
            for rank in ranks:
                model = BackgroundModel(pixels["model"], rank)
                detectors = config["detectors"] if rank == primary else [config["rank_sensitivity"]["detector"]]
                for colour in DECLARED_COLOURS:
                    signature = signatures[colour]
                    confusers = np.column_stack([signatures[c] for c in DECLARED_COLOURS if c != colour])
                    for detector in detectors:
                        summary, rows = analyse_cell(
                            flight, split, roles, background, halos, signature, confusers, colour, detector, model, pixels, config, rank == primary
                        )
                        if rank == primary:
                            summaries.append(summary)
                            decisions.extend(rows)
                        if detector == config["rank_sensitivity"]["detector"]:
                            rank_rows.append(summary)

    results = {
        "signature_angles_deg": angles,
        "repeated_measures": repeated_measures(decisions, config["repeated_measures"]["comparisons"]),
        "rank_sensitivity": rank_rows,
        "summaries": summaries,
    }
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "muufl_emplaced_targets.json").write_text(json.dumps(results, indent=2))
    with (out / "muufl_target_decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0].keys()))
        writer.writeheader()
        writer.writerows(decisions)
    for split in ("interleaved", "half_image"):
        for detector in config["detectors"]:
            sel = [s for s in summaries if s["split"] == split and s["detector"] == detector]
            print(
                f"{split:<11} {detector:<22} targets {sum(s['positives_accepted'] for s in sel)}/{sum(s['positives'] for s in sel)} "
                f"confusers {sum(s['confusers_accepted'] for s in sel)}/{sum(s['confusers'] for s in sel)} "
                f"cells above alpha {sum(s['realized_background_far'] > config['certification']['per_pixel_alpha'] for s in sel)}/{len(sel)}"
            )
    for e in results["repeated_measures"]:
        print(f"{e['split']:<11} {e['first']} vs {e['second']}: cloth p = {e['cloth_sign_flip_p']:.4f}, Holm {e['cloth_sign_flip_p_holm']:.4f}")


if __name__ == "__main__":
    main()
