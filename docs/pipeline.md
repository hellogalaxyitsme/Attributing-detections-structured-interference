# Analysis Pipeline

This repository contains the analysis code for the associated study:

**Attributing detections in structured interference: subspace mimics, whitening leakage, and certified hard-nulling detectors**

Every analysis is a single script in `src/` driven by a JSON config. Scripts write their results as JSON (and, for the emplaced targets, a CSV of target decisions) to the `output_dir` named in the config. Run all commands from `src/` after installing the dependencies.

## Simulations (Section 5)

The simulation scene has `m = 32` sensors, a target subspace of dimension `p = 4` and a declared interference subspace of rank `r = 8`, with principal-angle cosines `gamma` between them.

```bash
python soft_hard_detection.py --config ../configs/soft_hard_detection.json
python adaptive_secondary_size.py --config ../configs/adaptive_secondary_size.json
python selective_detectors.py --config ../configs/selective_detectors.json
python declaration_error.py --config ../configs/declaration_error.json
python certification_efficiency.py --config ../configs/certification_efficiency.json
```

- `soft_hard_detection.py`: exact and simulated detection probability versus SNR, exact ROC, and false attribution of canonical mimics versus amplitude for the GLR, MSD-F, clairvoyant AMF and ACE, adaptive AMF, ACE and Kelly, and the eigencanceler; also the soft/hard information split and the worst-case mimic coherence.
- `adaptive_secondary_size.py`: false attribution of amplitude-16 mimics for adaptive ACE, AMF, Kelly and the eigencanceler as the number of secondary vectors grows from 64 to 1024.
- `selective_detectors.py`: false attribution of Kelly, Rao, ABORT, W-ABORT, the eigencanceler and MSD-F versus normalized mimic energy, with the large-sample limits of Proposition 3, and detection probability.
- `declaration_error.py`: MSD-F under rotated or omitted declared directions and under AR(1) noise; exact analytic false attribution and thresholds certified on declaration-simulated or measured interference-only data, with and without whitening.
- `certification_efficiency.py`: analytic, Bonferroni and fixed-sequence thresholds on grids of 8 and 199 candidates under Gaussian, compound-Gaussian, Student-t and bursty noise, with paired sign-flip tests and Holm adjustment.

## Hyperspectral experiments (Section 6)

Copy `configs/muufl_example.json`, set `data_root` to the folder holding `muufl_gulfport_campus_3.mat`, `muufl_gulfport_campus_4.mat` and `tgt_img_spectra.mat`, then run:

```bash
python muufl_implants.py --config ../configs/muufl_local.json
python muufl_emplaced_targets.py --config ../configs/muufl_local.json
```

- `muufl_implants.py`: query signatures, declared confusers, confuser mixtures, measured confuser spectra and rotated confusers implanted into held-out background pixels; detection probability and false attribution of the eight detectors at certified thresholds, and the background-whitened coherence of each query with its nearest confuser.
- `muufl_emplaced_targets.py`: per-cell certification records, decisions on the emplaced cloths under the interleaved and half-image splits (certified, oracle and window-level thresholds), size-matched detection AUC, repeated-measures comparisons over physical cloths, sensitivity of MSD-F to the rank of the background subspace, and the spectral angles of the declared signatures.
