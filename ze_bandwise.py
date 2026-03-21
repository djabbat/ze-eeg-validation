#!/usr/bin/env python3
"""
Ze Band-wise Analysis — LEMON dataset
======================================
Tests Ze aging hypothesis per frequency band:
- Alpha (8-12Hz): known to SLOW with aging → lower alpha v → lower alpha χ_Ze in old
- Beta (13-30Hz): often increases with age as broadband noise
- Theta (4-8Hz): increases in mild cognitive impairment

Key insight: broadband EEG has v dominated by slow components.
Bandpass filtering isolates the contribution of each oscillatory band.

After bandpass filter, a band-specific signal has v ≈ 2*f_center/fs.
Ze analysis per band tests: does aging reduce χ_Ze in the ALPHA band?
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import compute_ze_metrics, V_STAR, CHI_MAX_LIVING

import mne
mne.set_log_level('WARNING')

# Frequency bands
BANDS = {
    'delta': (1,   4),
    'theta': (4,   8),
    'alpha': (8,  12),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

SUBJECTS = {
    # old (65-70)
    'sub-032301': {'age': 67, 'group': 'old'},
    'sub-032303': {'age': 67, 'group': 'old'},
    'sub-032305': {'age': 67, 'group': 'old'},
    # young (20-25 / 25-30)
    'sub-032302': {'age': 22, 'group': 'young'},
    'sub-032307': {'age': 27, 'group': 'young'},
    'sub-032310': {'age': 27, 'group': 'young'},
}

LEMON_DIR = Path('/home/oem/Desktop/EEG/data/lemon')
OUT_DIR   = Path('/home/oem/Desktop/EEG/data/lemon/results')
OUT_DIR.mkdir(exist_ok=True)

RESAMPLE_HZ = 128.0
CROP_S = 240  # use first 4 minutes of EC condition


def load_set(path: str) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    raw.resample(RESAMPLE_HZ, npad='auto')
    raw.crop(tmax=min(CROP_S, raw.times[-1]))
    return raw


def bandpass_ze(raw: mne.io.BaseRaw, fmin: float, fmax: float) -> dict:
    """Apply bandpass filter and compute Ze metrics per channel, return summary."""
    raw_filt = raw.copy().filter(fmin, fmax, method='iir', verbose=False)
    picks = mne.pick_types(raw_filt.info, eeg=True, exclude='bads')
    if len(picks) == 0:
        picks = list(range(raw_filt.info['nchan']))
    chi_list, v_list = [], []
    for idx in picks:
        signal = raw_filt.get_data(picks=[idx])[0]
        m = compute_ze_metrics(signal)
        chi_list.append(m['chi_Ze'])
        v_list.append(m['v'])
    return {
        'chi_Ze_mean': float(np.mean(chi_list)),
        'chi_Ze_std':  float(np.std(chi_list)),
        'v_mean':      float(np.mean(v_list)),
    }


print("Ze Band-wise Analysis — MPI-LEMON")
print(f"Subjects: {len(SUBJECTS)} | Crop: {CROP_S}s EC | fs: {RESAMPLE_HZ}Hz\n")

all_results = {}

for sub_id, info in SUBJECTS.items():
    ec_path = LEMON_DIR / sub_id / f"{sub_id}_EC.set"
    if not ec_path.exists():
        print(f"  ⚠️  {sub_id}: EC file not found, skipping")
        continue

    print(f"  {sub_id}  ({info['group']}, age~{info['age']})  loading EC...")
    raw = load_set(str(ec_path))
    print(f"    {raw.info['nchan']}ch, {raw.times[-1]:.0f}s @ {raw.info['sfreq']:.0f}Hz")

    band_results = {}
    for band_name, (fmin, fmax) in BANDS.items():
        bz = bandpass_ze(raw, fmin, fmax)
        band_results[band_name] = bz
        f_center = (fmin + fmax) / 2
        v_theoretical = 2 * f_center / RESAMPLE_HZ
        chi_theoretical = 1 - abs(v_theoretical - V_STAR) / max(V_STAR, 1-V_STAR)
        print(f"    {band_name:6s} [{fmin}-{fmax}Hz]: "
              f"χ_Ze={bz['chi_Ze_mean']:.4f} ±{bz['chi_Ze_std']:.4f}  "
              f"v={bz['v_mean']:.4f}  (theory: v={v_theoretical:.3f} χ={chi_theoretical:.3f})")

    all_results[sub_id] = {**info, 'bands': band_results}
    print()

# Summary table
print("═" * 70)
print("SUMMARY BY BAND")
print(f"{'Band':8s}  {'Young χ_Ze':15s}  {'Old χ_Ze':15s}  {'Δ (Y−O)':10s}  {'Hyp'}")
print("─" * 70)

for band_name in BANDS:
    young_chis = [all_results[s]['bands'][band_name]['chi_Ze_mean']
                  for s in all_results if all_results[s]['group']=='young']
    old_chis   = [all_results[s]['bands'][band_name]['chi_Ze_mean']
                  for s in all_results if all_results[s]['group']=='old']
    if not young_chis or not old_chis:
        continue
    ym, ys = np.mean(young_chis), np.std(young_chis)
    om, os_ = np.mean(old_chis),  np.std(old_chis)
    delta = ym - om
    hyp = "✅ Y>O" if delta > 0.01 else ("⚠️ Y<O" if delta < -0.01 else "≈")
    print(f"  {band_name:6s}  {ym:.4f}±{ys:.4f}    {om:.4f}±{os_:.4f}    {delta:+.4f}    {hyp}")

# Save
out_path = OUT_DIR / 'ze_bandwise_results.json'
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\n  💾 Saved: {out_path}")

# Plot: per-band χ_Ze, young vs old
fig, axes = plt.subplots(1, len(BANDS), figsize=(14, 5), sharey=True)
fig.suptitle('Ze χ_Ze by Frequency Band — MPI-LEMON\nYoung (20-30) vs Old (65-70)',
             fontsize=11, fontweight='bold')

for ax, (band_name, (fmin, fmax)) in zip(axes, BANDS.items()):
    young_vals = [all_results[s]['bands'][band_name]['chi_Ze_mean']
                  for s in all_results if all_results[s]['group']=='young']
    old_vals   = [all_results[s]['bands'][band_name]['chi_Ze_mean']
                  for s in all_results if all_results[s]['group']=='old']

    x_y = np.ones(len(young_vals)) * 0.75
    x_o = np.ones(len(old_vals))   * 1.75
    ax.scatter(x_y + np.random.randn(len(young_vals))*0.05, young_vals,
               color='#2E74B5', s=60, alpha=0.8, zorder=5)
    ax.scatter(x_o + np.random.randn(len(old_vals))*0.05, old_vals,
               color='#C00000', s=60, alpha=0.8, zorder=5)
    ax.errorbar([0.75], [np.mean(young_vals)], yerr=[np.std(young_vals)],
                color='#2E74B5', fmt='D', markersize=10, capsize=5, linewidth=2, zorder=6)
    ax.errorbar([1.75], [np.mean(old_vals)], yerr=[np.std(old_vals)],
                color='#C00000', fmt='D', markersize=10, capsize=5, linewidth=2, zorder=6)
    ax.axhline(CHI_MAX_LIVING, color='green', linestyle=':', linewidth=0.8)
    ax.set_title(f'{band_name}\n{fmin}-{fmax}Hz', fontsize=9)
    ax.set_xticks([0.75, 1.75])
    ax.set_xticklabels(['Young', 'Old'], fontsize=8)
    ax.set_ylim(0, 1)
    if ax == axes[0]:
        ax.set_ylabel('χ_Ze mean')

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#2E74B5', label='Young (20-30)'),
                    Patch(color='#C00000', label='Old (65-70)')],
           loc='lower right', fontsize=9)
plt.tight_layout()
plot_path = OUT_DIR / 'ze_bandwise_young_vs_old.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Plot: {plot_path}")
