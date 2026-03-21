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

For alpha peak frequency f_Hz at sampling rate fs Hz:
```
v_peak = 2 × f_peak / fs
χ_Ze   = 1 − |v_peak − v*| / max(v*, 1−v*)
```

At 128 Hz, the Ze-optimal frequency is **f_opt = 29.2 Hz** (beta/gamma boundary).

---

## Results Summary

| Dataset | N | Age range | Key result |
|---------|---|-----------|------------|
| Zenodo 3875159 — EC vs EO | 1 subj | — | Δχ_Ze(EO−EC) = **+0.064** ✅ |
| MPI-LEMON alpha peak | 30 | 22–72 yr | d = 0.110, p = 0.765 ⚠️ (underpowered) |
| Dortmund Vital Study ds005385 | **60** | 20–70 yr | **p = 0.006, d = 0.732** ✅ |
| Cuban Normative EEG — Zenodo 4244765 | **198** | 5–97 yr | Inverted-U, peak 36.5 yr, **d = 1.694** ✅ |

---

## Repository Structure

| File | Description |
|------|-------------|
| `eeg_ze_processor.py` | Core library: Ze metrics, EEG loaders (EDF/BDF/BrainVision), CLI |
| `ze_ec_eo_analysis.py` | Experiment 1: EC vs EO, Zenodo 3875159 (BrainVision, 128 ch) |
| `ze_lemon_analysis.py` | Experiment 2a: MPI-LEMON broadband Ze |
| `ze_bandwise.py` | Experiment 2b: MPI-LEMON per-band Ze (delta/theta/alpha/beta/gamma) |
| `ze_alpha_peak.py` | Experiment 2c: MPI-LEMON alpha peak → v → χ_Ze, N=30 |
| `ze_batch_pipeline.py` | MPI-LEMON batch download + analysis |
| `ze_dortmund_pipeline.py` | Experiment 3: Dortmund young (20–30) vs old (63–70), N=60 |
| `ze_cuban_analysis.py` | Experiment 4: Cuban lifespan curve, N=198, ages 5–97 |

---

## Datasets (download separately)

| Dataset | Source | N | Format |
|---------|--------|---|--------|
| Zenodo 3875159 | https://zenodo.org/records/3875159 | 1 subj | BrainVision (.vhdr) |
| MPI-LEMON | https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html | 228 | EEGLAB (.set) |
| Dortmund ds005385 | https://openneuro.org/datasets/ds005385 | 608 | EDF/BIDS |
| Cuban Normative EEG | https://zenodo.org/records/4244765 | 211 | MATLAB (.mat) |

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `mne`, `numpy`, `scipy`, `matplotlib`.

---

## Usage

### Core library

```python
from eeg_ze_processor import ze_cheating_index, V_STAR

f_peak = 9.5    # Hz, alpha peak frequency
fs     = 128.0  # Hz, sampling rate
v      = 2 * f_peak / fs
chi    = ze_cheating_index(v)
print(f"v = {v:.4f}  χ_Ze = {chi:.4f}")  # v=0.1484  χ_Ze=0.4299
```

### Command-line interface

```bash
# Demo with synthetic EEG (no data required)
python3 eeg_ze_processor.py --demo

# Single file
python3 eeg_ze_processor.py --file recording.edf --age 35 --label "Subj01" --resample 128

# Batch folder
python3 eeg_ze_processor.py --batch data/edf/ --resample 128 --out results/
```

### Dataset-specific analyses

Set data directories via environment variables (or edit the `_default_*` paths at the top of each script):

```bash
export ZE_CUBAN_DIR=/path/to/cuban/oldgandalf-FirstWaveCubanHumanNormativeEEGProject-3783da7
export ZE_DORTMUND_DIR=/path/to/dortmund
export ZE_LEMON_DIR=/path/to/lemon
export ZE_ZENODO_VHDR=/path/to/360.vhdr

python3 ze_cuban_analysis.py       # Lifespan curve, N=198
python3 ze_dortmund_pipeline.py    # Young vs old, N=60
python3 ze_alpha_peak.py           # LEMON alpha peak, N=30
python3 ze_ec_eo_analysis.py       # Within-subject EC vs EO
```

---

## Key Results

### Cuban Human Normative EEG — Lifespan Inverted-U (N=198)

| Age Group | N | f_peak (Hz) | χ_Ze (mean ± SD) |
|-----------|---|-------------|-----------------|
| Children (5–12) | 53 | 9.05 ± 0.81 | 0.4936 ± 0.030 |
| Teens (12–18) | 41 | 9.75 ± 0.87 | 0.5192 ± 0.032 |
| **Young adults (18–35)** | 31 | **10.00 ± 0.98** | **0.5287 ± 0.036** |
| Middle-aged (35–60) | 37 | 9.64 ± 0.90 | 0.5153 ± 0.033 |
| Older adults (60–80) | 34 | 8.94 ± 0.98 | 0.4895 ± 0.036 |

Quadratic peak: **36.5 years** | Young vs Old: **d = 1.694, p < 0.0001**

### Dortmund Vital Study — Young vs Old (N=60)

| Group | χ_Ze | Statistics |
|-------|------|------------|
| Young (20–30 yr) | 0.4492 ± 0.027 | t = 2.86 |
| Old (63–70 yr) | 0.4285 ± 0.029 | **p = 0.006, d = 0.732** |

### Within-subject EC vs EO (Zenodo 3875159)

| Condition | χ_Ze |
|-----------|------|
| Eyes-Closed ×3 | 0.271 ± 0.004 |
| Eyes-Open ×3 | 0.335 ± 0.022 |
| **Δ EO − EC** | **+0.064** |

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
