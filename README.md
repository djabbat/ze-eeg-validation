# Ze EEG Validation

**Experimental EEG validation of Ze Theory (Tkemaladze) across the human lifespan.**

Ze Theory proposes that the *cheating index* χ_Ze — a measure of how close a biosignal's binary switching rate is to the theoretical optimum v* = 0.45631 — serves as a biomarker of neurodynamic efficiency. This repository contains all analysis code supporting the findings reported in:

> Tkemaladze J. *Ze Cheating Index (χ_Ze) as a Biomarker of Neurodynamic Efficiency: Experimental EEG Validation Across the Human Lifespan.* 2026.

---

## Ze Metrics — Core Theory

```
Binary sequence:  x_k = 1  if  sample > median, else 0
Ze velocity:      v = N_S / (N − 1)      N_S = number of switches, v ∈ [0, 1]
Fixed point:      v* = 0.45631           (maximum materialization)
Cheating index:   χ_Ze = 1 − |v − v*| / max(v*, 1−v*)     χ_Ze ∈ [0, 1]
```

**Aging hypothesis:** EEG slows with age → dominant frequency decreases → v moves away from v* → χ_Ze decreases.

### Two validated methods

| Method | Formula | When to use |
|--------|---------|-------------|
| **Proxy** | v_peak = 2·f_peak/f_s | Any dataset with PSD; requires narrow-band alpha |
| **Narrowband Ze** | bandpass 8–12 Hz → binarize → v | Raw EEG with artifact-cleaned recordings |

Note: narrowband Ze applies Ze binarization within the alpha band and is not identical to the theoretical Ze definition on broadband signals.

At 128 Hz, the Ze-optimal frequency is **f_opt = v* × 128/2 ≈ 29.2 Hz** (beta/gamma boundary).
At 100 Hz (Cuban dataset), f_opt ≈ 22.8 Hz.

---

## Results Summary

| Dataset | N | Age range | Key result |
|---------|---|-----------|------------|
| Zenodo 3875159 — EC vs EO | 1 subj | — | Δχ_Ze(EO−EC) = **+0.064** (illustrative) |
| MPI-LEMON alpha peak | 30 | 22–72 yr | d = 0.110, p = 0.765 ⚠️ (underpowered) |
| Dortmund Vital Study ds005385 | **60** | 20–70 yr | **p = 0.006, d = 0.732; ANCOVA p = 0.037; AUC = 0.715** |
| Cuban Normative EEG — Zenodo 4244765 | **196** | 5–97 yr | Inverted-U, peak 36.5 yr, **d = 1.694** |

---

## Repository Structure

| File | Description |
|------|-------------|
| `eeg_ze_processor.py` | **Core library** — Ze metrics, proxy/narrowband methods, Cuban loader, group statistics, EEG file loading, CLI |
| `ze_ec_eo_analysis.py` | Experiment 1: EC vs EO, Zenodo 3875159 (BrainVision, 128 ch) |
| `ze_lemon_analysis.py` | Experiment 2a: MPI-LEMON broadband Ze |
| `ze_bandwise.py` | Experiment 2b: MPI-LEMON per-band Ze (delta/theta/alpha/beta/gamma) |
| `ze_alpha_peak.py` | Experiment 2c: MPI-LEMON alpha peak → χ_Ze, N=30 |
| `ze_batch_pipeline.py` | MPI-LEMON batch download + analysis |
| `ze_dortmund_pipeline.py` | Experiment 3: Dortmund young (20–30) vs old (63–70), N=60 |
| `ze_cuban_analysis.py` | Experiment 4: Cuban lifespan curve, N=196, ages 5–97 |

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `mne`, `numpy`, `scipy`, `matplotlib`.

---

## Usage

### Quickstart — demo (no data required)

```bash
python3 eeg_ze_processor.py --demo
```

Runs on synthetic sinusoidal signals at realistic alpha frequencies (10.5/9.5/8.5 Hz),
demonstrates the proxy and narrowband Ze methods, and confirms the Ze aging hypothesis.

### Single file

```bash
python3 eeg_ze_processor.py --file recording.edf --age 35 --label "Subj01" --resample 128
```

### Batch folder

```bash
python3 eeg_ze_processor.py --batch data/edf/ --resample 128 --out results/
```

### Cuban dataset

```bash
python3 eeg_ze_processor.py --cuban /path/to/EyesClose/ --out results/
```

### Core library — Python API

```python
from eeg_ze_processor import (
    ze_cheating_index, V_STAR,
    alpha_peak_ze,    # proxy method
    narrowband_ze,    # narrowband Ze method
    load_cuban_mcr,   # Cuban .mat loader
    group_statistics, # full stats: t-test, d, CI, AUC, ANCOVA
)

# Proxy method
result = alpha_peak_ze(signal, fs=128.0)
print(f"f_peak={result['f_peak']:.2f} Hz  χ_Ze={result['chi_Ze']:.4f}")

# Narrowband Ze
result = narrowband_ze(signal, fs=128.0, lowcut=8.0, highcut=12.0)
print(f"χ_Ze={result['chi_Ze']:.4f}")

# Cuban dataset
result = load_cuban_mcr('/path/to/sub01_cross.mat')
print(f"age={result['age']}  f_peak={result['f_peak']:.2f} Hz  χ_Ze={result['chi_Ze']:.4f}")

# Group statistics (with sex-adjustment)
import numpy as np
stats = group_statistics(
    chi_young, chi_old,
    sex_young=np.array(['F','M',...]),
    sex_old  =np.array(['M','F',...]),
)
print(f"d={stats['cohens_d']:.3f}  AUC={stats['auc']:.3f}")
print(f"ANCOVA F={stats['ancova']['F_group']:.3f}  p={stats['ancova']['p_group']:.4f}")
```

---

## Dataset-specific analyses

Set data directories via environment variables:

```bash
export ZE_CUBAN_DIR=/path/to/cuban/oldgandalf-FirstWaveCubanHumanNormativeEEGProject-*
export ZE_DORTMUND_DIR=/path/to/dortmund
export ZE_LEMON_DIR=/path/to/lemon
export ZE_ZENODO_VHDR=/path/to/360.vhdr

python3 ze_cuban_analysis.py        # Lifespan curve, N=196
python3 ze_dortmund_pipeline.py     # Young vs old, N=60
python3 ze_alpha_peak.py            # LEMON alpha peak, N=30
python3 ze_ec_eo_analysis.py        # Within-subject EC vs EO
```

---

## Key Results

### Cuban Human Normative EEG — Lifespan Inverted-U (N=196)

| Age Group | N | f_peak (Hz) | χ_Ze (mean ± SD) |
|-----------|---|-------------|-----------------------|
| Children (5–12) | 53 | 9.05 ± 0.81 | 0.4936 ± 0.030 |
| Teens (12–18) | 41 | 9.75 ± 0.87 | 0.5192 ± 0.032 |
| **Young adults (18–35)** | 31 | **10.00 ± 0.98** | **0.5287 ± 0.036** |
| Middle-aged (35–60) | 37 | 9.64 ± 0.90 | 0.5153 ± 0.033 |
| Older adults (60–80) | 34 | 8.94 ± 0.98 | 0.4895 ± 0.036 |

Quadratic peak: **36.5 years** (95% CI: 32.5–39.7) | R² = 0.153 | RMSE = 0.033
Young vs Elderly (≥65 yr, N=22): **d = 1.694 [1.147, 2.487], p < 0.0001, AUC = 0.715**

### Dortmund Vital Study — Young vs Old (N=60)

| Method | Group | χ_Ze | Statistics |
|--------|-------|------|------------|
| Proxy (α-peak) | Young (20–30 yr) | 0.449 ± 0.027 | t = 2.83 |
| | Old (63–70 yr) | 0.429 ± 0.029 | **p = 0.006, d = 0.732 [0.224, 1.387]** |
| Narrowband Ze | Young | 0.450 ± 0.008 | t = 2.19 |
| | Old | 0.444 ± 0.010 | **p = 0.028, d = 0.584 [0.179, 0.943]** |

ANCOVA (sex-adjusted): **F(1,57) = 4.56, p = 0.037**
Group × sex interaction: F(1,56) = 0.600, **p = 0.442** (effect is not sex-specific)
AUC: proxy = **0.715** (acceptable), narrowband Ze = **0.689** (fair)

---

## Datasets (download separately)

| Dataset | Source | N | Format |
|---------|--------|---|--------|
| Zenodo 3875159 | https://zenodo.org/records/3875159 | 1 subj | BrainVision (.vhdr) |
| MPI-LEMON | https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html | 228 | EEGLAB (.set) |
| Dortmund ds005385 | https://openneuro.org/datasets/ds005385 | 608 | EDF/BIDS |
| Cuban Normative EEG | https://zenodo.org/records/4244765 | 211 | MATLAB (.mat) |

---

## Citation

```bibtex
@article{tkemaladze2026ze_eeg,
  title   = {{Ze} Cheating Index ($\chi_{{Ze}}$) as a Biomarker of Neurodynamic
             Efficiency: Experimental {EEG} Validation Across the Human Lifespan},
  author  = {Tkemaladze, Jaba},
  year    = {2026},
  url     = {https://github.com/djabbat/ze-eeg-validation}
}
```

---

## License

MIT License. Datasets must be obtained directly from their original sources under respective licenses.
