# Attributing-detections-structured-interference

Analysis code for **"Attributing detections in structured interference: subspace mimics, whitening leakage, and certified hard-nulling detectors."**

Detectors for signals in structured interference decide whether a target is present; many applications must also decide whether the evidence could have been produced by the interference. The study formulates attribution as a test whose acceptance probability must stay below a prescribed level for every field in a declared interference subspace. It compares detectors that annihilate the declared subspace (GLR, MSD-F, OSP, TCIMF) with detectors that whiten it (AMF, ACE, Kelly) and with mismatch-selective detectors (ABORT, W-ABORT, Rao). The analyses compute exact Gaussian operating characteristics, certify thresholds on interference-only data with a fixed-sequence intersection–union rule, and test the predictions on the MUUFL Gulfport hyperspectral data.

## Repository contents

- `src/`: detector and certification modules, the simulation analyses (Section 5 of the paper) and the MUUFL Gulfport analyses (Section 6).
- `configs/`: JSON configs with the settings used in the paper; `muufl_example.json` has placeholder data paths.
- `docs/pipeline.md`: command-level overview of the reproduction pipeline.

## Dataset

The hyperspectral experiments use the public MUUFL Gulfport data set (MIT licence):

- P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell, *MUUFL Gulfport Hyperspectral and LiDAR Airborne Data Set*, Tech. Rep. REP-2013-570, University of Florida, Gainesville, FL, 2013.
- Repository: <https://github.com/GatorSense/MUUFLGulfport>

The analyses read three files from `MUUFLGulfportDataCollection/`: `muufl_gulfport_campus_3.mat`, `muufl_gulfport_campus_4.mat` and `tgt_img_spectra.mat`. Users are responsible for following the access terms of the data set.

The simulations need no external data.

## Environment

Python 3.10 was used for the analyses.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The only dependencies are NumPy and SciPy.

## Quick start

Run the scripts from `src/`. The simulations use the configs as provided and write JSON results to `outputs/simulations/`:

```bash
cd src
python soft_hard_detection.py --config ../configs/soft_hard_detection.json
python adaptive_secondary_size.py --config ../configs/adaptive_secondary_size.json
python selective_detectors.py --config ../configs/selective_detectors.json
python declaration_error.py --config ../configs/declaration_error.json
python certification_efficiency.py --config ../configs/certification_efficiency.json
```

For the hyperspectral experiments, copy the example config, set `data_root` to the folder holding the MUUFL files, and run:

```bash
cp configs/muufl_example.json configs/muufl_local.json
cd src
python muufl_implants.py --config ../configs/muufl_local.json
python muufl_emplaced_targets.py --config ../configs/muufl_local.json
```

The simulation scripts accept `--workers` to set the number of parallel processes (default 12).

## Scripts and results

| Script | Paper |
|---|---|
| `soft_hard_detection.py` | Section 5.1, Fig. 2, Supplementary Table S1 and Fig. S2 |
| `adaptive_secondary_size.py` | Section 5.2, Supplementary Table S2 |
| `selective_detectors.py` | Section 5.2, Fig. 3, Supplementary Table S3 |
| `declaration_error.py` | Section 5.3, Supplementary Table S4 and Fig. S3 |
| `certification_efficiency.py` | Section 5.4, Table 2, Supplementary Tables S5–S6 and Fig. S4 |
| `muufl_implants.py` | Section 6.1, Fig. 4, Table 3, Supplementary Tables S11–S12 |
| `muufl_emplaced_targets.py` | Section 6.2, Table 4, Supplementary Tables S7–S10 and S13–S14, Fig. S5 |

Library modules:

| Module | Contents |
|---|---|
| `subspace_detection.py` | GLR, MSD-F, clairvoyant and adaptive AMF/ACE/Kelly, eigencanceler, exact tail probabilities of Gaussian quadratic forms (Imhof, Ruben), noise laws, simulation scene |
| `selective_detection.py` | ABORT, W-ABORT and Rao statistics and their large-sample mimic limits |
| `certification.py` | fixed-sequence and Bonferroni threshold certification |
| `geometry.py` | canonical target/interference geometry and declaration-error measures |
| `statistics_tools.py` | exact sign-flip test, Holm adjustment, Clopper–Pearson interval |
| `muufl.py` | MUUFL Gulfport loading, pixel roles, background model and hyperspectral detectors |

## Reproducibility notes

All random seeds are fixed in the configs. The results in the paper were computed with single-threaded BLAS (`OMP_NUM_THREADS=1`) and 12 worker processes; a different BLAS threading configuration can change the last digits of floating-point sums. Each simulation script finishes within minutes on a 12-core workstation. Observations whose residual after annihilation of the declared subspace is zero are scored as non-detections, as defined in Section 2 of the paper. The repository excludes the raw data and generated outputs.
