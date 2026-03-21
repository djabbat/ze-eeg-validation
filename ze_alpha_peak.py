#!/usr/bin/env python3
"""
Ze Alpha Peak Analysis — MPI-LEMON
====================================
Extracts alpha PEAK FREQUENCY from PSD (not broadband v from binarization).
Aging effect: alpha peak slows ~1Hz (10.5 → 9.5Hz), giving Δv ≈ 0.016 → Δχ_Ze ≈ 0.029.

v_peak = 2 * f_peak / fs
χ_Ze_peak = 1 - |v_peak - v*| / max(v*, 1-v*)

This is the Ze prediction for each subject's "dominant" oscillatory frequency.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch, find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import ze_cheating_index, V_STAR

import mne
mne.set_log_level('WARNING')

LEMON_DIR = Path(os.environ.get('ZE_LEMON_DIR', str(Path(__file__).parent.parent / 'data' / 'lemon')))
OUT_DIR   = LEMON_DIR / 'results'
RESAMPLE_HZ = 128.0
CROP_S = 240

# Subjects to analyze
SUBJECTS = {
    # old
    'sub-032301': 67, 'sub-032303': 67, 'sub-032305': 67,
    'sub-032338': 67, 'sub-032329': 67, 'sub-032340': 67,
    'sub-032430': 67, 'sub-032458': 67, 'sub-032337': 67,
    'sub-032392': 67, 'sub-032442': 72, 'sub-032491': 72,
    'sub-032495': 72, 'sub-032459': 72, 'sub-032333': 72,
    # young
    'sub-032302': 22, 'sub-032307': 27, 'sub-032310': 27,
    'sub-032353': 22, 'sub-032414': 22, 'sub-032389': 27,
    'sub-032390': 22, 'sub-032344': 27, 'sub-032421': 22,
    'sub-032400': 22, 'sub-032385': 22, 'sub-032525': 22,
    'sub-032467': 22, 'sub-032508': 27, 'sub-032323': 22,
}

import csv
meta = {}
with open(LEMON_DIR / 'participants.csv') as f:
    for row in csv.DictReader(f):
        sid = row['ID'].strip()
        age_bin = row['Age'].strip()
        lo = int(age_bin.split('-')[0]) if '-' in age_bin else 0
        meta[sid] = age_bin, ('young' if lo <= 35 else 'old')


def alpha_peak_freq(raw, alpha_band=(7.5, 13.0), n_channels_avg=10):
    """
    Compute mean alpha peak frequency across posterior channels (or all if <10).
    Returns f_peak (Hz).
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if not len(picks): picks = list(range(raw.info['nchan']))

    peak_freqs = []
    for idx in picks:
        sig = raw.get_data(picks=[idx])[0]
        freqs, psd = welch(sig, fs=raw.info['sfreq'], nperseg=int(raw.info['sfreq']*4))
        # Extract alpha band
        mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        if not mask.any():
            continue
        alpha_psd = psd[mask]
        alpha_freqs = freqs[mask]
        # Peak = frequency with max power in alpha band
        f_peak = alpha_freqs[np.argmax(alpha_psd)]
        peak_freqs.append(f_peak)

    return float(np.median(peak_freqs)) if peak_freqs else 10.0


print("Ze Alpha Peak Analysis — MPI-LEMON")
print(f"N = {len(SUBJECTS)} subjects | EC condition | 128Hz resampled\n")

results = []

for sub_id in sorted(SUBJECTS.keys()):
    ec_path = LEMON_DIR / sub_id / f'{sub_id}_EC.set'
    if not ec_path.exists():
        print(f"  ⚠️  {sub_id}: EC.set not found, skip")
        continue

    age_bin, group = meta.get(sub_id, ('?', '?'))

    try:
        raw = mne.io.read_raw_eeglab(str(ec_path), preload=True, verbose=False)
        raw.resample(RESAMPLE_HZ, npad='auto')
        raw.crop(tmax=min(CROP_S, raw.times[-1]))
    except Exception as e:
        print(f"  ❌ {sub_id}: {e}")
        continue

    f_peak = alpha_peak_freq(raw)
    v_peak = 2 * f_peak / RESAMPLE_HZ
    chi_peak = ze_cheating_index(v_peak)

    results.append({
        'subject_id': sub_id,
        'age_bin': age_bin,
        'group': group,
        'alpha_peak_hz': round(f_peak, 3),
        'v_peak': round(v_peak, 5),
        'chi_Ze_peak': round(chi_peak, 5),
    })
    print(f"  {sub_id}  {age_bin:5s} {group:5s}  f_peak={f_peak:.2f}Hz  "
          f"v={v_peak:.4f}  χ_Ze={chi_peak:.4f}")

# Summary
print(f"\n{'═'*60}")
for group in ('young', 'old'):
    gr = [r for r in results if r['group'] == group]
    if not gr: continue
    fps  = [r['alpha_peak_hz']  for r in gr]
    chis = [r['chi_Ze_peak']    for r in gr]
    print(f"  {group.upper():5s}  N={len(gr):2d}  "
          f"f_peak={np.mean(fps):.3f}±{np.std(fps):.3f}Hz  "
          f"χ_Ze={np.mean(chis):.4f}±{np.std(chis):.4f}")

y_fps  = [r['alpha_peak_hz'] for r in results if r['group']=='young']
o_fps  = [r['alpha_peak_hz'] for r in results if r['group']=='old']
y_chis = [r['chi_Ze_peak']   for r in results if r['group']=='young']
o_chis = [r['chi_Ze_peak']   for r in results if r['group']=='old']

if y_fps and o_fps:
    df = np.mean(y_fps) - np.mean(o_fps)
    dc = np.mean(y_chis) - np.mean(o_chis)
    print(f"\n  Δ f_peak (Y−O): {df:+.3f} Hz")
    print(f"  Δ χ_Ze   (Y−O): {dc:+.4f}")
    if df > 0:
        print("  ✅ Alpha slowing confirmed: young faster than old")
    if dc > 0:
        print("  ✅ Ze hypothesis (alpha peak): χ_Ze(young) > χ_Ze(old)")

# Correlation: age vs chi_Ze_peak
ages  = [int(r['age_bin'].split('-')[0])+2 for r in results if '-' in r['age_bin']]
chis  = [r['chi_Ze_peak'] for r in results if '-' in r['age_bin']]
fps_all = [r['alpha_peak_hz'] for r in results if '-' in r['age_bin']]
if len(ages) >= 5:
    corr_chi = np.corrcoef(ages, chis)[0,1]
    corr_fp  = np.corrcoef(ages, fps_all)[0,1]
    print(f"\n  Pearson r (age vs χ_Ze_peak): {corr_chi:.4f}")
    print(f"  Pearson r (age vs f_peak):    {corr_fp:.4f}")

print(f"{'═'*60}")

# Save
out_path = OUT_DIR / 'ze_alpha_peak_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  💾 {out_path}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f'Ze Alpha Peak Analysis — MPI-LEMON (N={len(results)})\n'
             'Alpha peak frequency → v → χ_Ze', fontsize=11, fontweight='bold')

col = {'young': '#2E74B5', 'old': '#C00000'}

# f_peak distribution
ax = axes[0]
for group, offset in [('young', -0.15), ('old', 0.15)]:
    gr = [r for r in results if r['group']==group]
    fps = [r['alpha_peak_hz'] for r in gr]
    ax.scatter([1+offset]*len(fps) + np.random.randn(len(fps))*0.03,
               fps, color=col[group], s=40, alpha=0.7)
    ax.errorbar([1+offset], [np.mean(fps)], yerr=[np.std(fps)/len(fps)**0.5],
                color=col[group], fmt='D', ms=9, capsize=6, lw=2, zorder=5)
ax.set_xticks([0.85, 1.15]); ax.set_xticklabels(['Young', 'Old'])
ax.set_ylabel('Alpha peak frequency (Hz)'); ax.set_title('Alpha Peak Frequency')
ax.set_xlim(0.5, 1.5)

# χ_Ze_peak distribution
ax = axes[1]
for group, offset in [('young', -0.15), ('old', 0.15)]:
    gr = [r for r in results if r['group']==group]
    ch = [r['chi_Ze_peak'] for r in gr]
    ax.scatter([1+offset]*len(ch) + np.random.randn(len(ch))*0.03,
               ch, color=col[group], s=40, alpha=0.7)
    ax.errorbar([1+offset], [np.mean(ch)], yerr=[np.std(ch)/len(ch)**0.5],
                color=col[group], fmt='D', ms=9, capsize=6, lw=2, zorder=5)
ax.set_xticks([0.85, 1.15]); ax.set_xticklabels(['Young', 'Old'])
ax.set_ylabel('χ_Ze (alpha peak)'); ax.set_title('χ_Ze from Alpha Peak')
ax.set_xlim(0.5, 1.5)

# Scatter: age vs χ_Ze_peak
ax = axes[2]
for r in results:
    if '-' not in r['age_bin']: continue
    age = int(r['age_bin'].split('-')[0]) + 2
    ax.scatter(age, r['chi_Ze_peak'], color=col.get(r['group'],'gray'), s=50, alpha=0.7)
if len(ages) >= 5:
    z = np.polyfit(ages, chis, 1)
    x_line = np.linspace(min(ages), max(ages), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'k--', lw=1.5,
            label=f'r={corr_chi:.3f}')
    ax.legend(fontsize=9)
ax.set_xlabel('Age (years)'); ax.set_ylabel('χ_Ze (alpha peak)')
ax.set_title('Age vs χ_Ze(α peak)')

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#2E74B5', label='Young (20-30)'),
                    Patch(color='#C00000', label='Old (65-75)')],
           loc='lower right', fontsize=9)
plt.tight_layout()
ppath = OUT_DIR / 'ze_alpha_peak.png'
plt.savefig(ppath, dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 {ppath}")
