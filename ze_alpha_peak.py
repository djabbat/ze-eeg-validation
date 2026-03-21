#!/usr/bin/env python3
"""
Ze Alpha Peak Analysis — MPI-LEMON
====================================
Extracts alpha peak frequency from PSD, computes proxy Ze velocity and χ_Ze.
Ze aging hypothesis: alpha slowing → v_peak ↓ → χ_Ze ↓

Uses eeg_ze_processor API:
  alpha_peak_ze()   — Welch PSD → α-peak → v_peak = 2f/fs → χ_Ze
  group_statistics()— t-test, Cohen's d, CI, AUC, power

v_peak = 2 * f_peak / fs
χ_Ze   = 1 - |v_peak - v*| / max(v*, 1-v*)

Dataset: https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html
Set ZE_LEMON_DIR or edit default path below.
"""
import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import V_STAR, alpha_peak_ze, group_statistics

import mne
mne.set_log_level('WARNING')

LEMON_DIR   = Path(os.environ.get('ZE_LEMON_DIR',
              str(Path(__file__).parent.parent / 'data' / 'lemon')))
OUT_DIR     = LEMON_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLE_HZ = 128.0
CROP_S      = 240
ALPHA_BAND  = (7.5, 13.0)

SUBJECTS = {
    # old (67–72 yr)
    'sub-032301': 67, 'sub-032303': 67, 'sub-032305': 67,
    'sub-032338': 67, 'sub-032329': 67, 'sub-032340': 67,
    'sub-032430': 67, 'sub-032458': 67, 'sub-032337': 67,
    'sub-032392': 67, 'sub-032442': 72, 'sub-032491': 72,
    'sub-032495': 72, 'sub-032459': 72, 'sub-032333': 72,
    # young (22–27 yr)
    'sub-032302': 22, 'sub-032307': 27, 'sub-032310': 27,
    'sub-032353': 22, 'sub-032414': 22, 'sub-032389': 27,
    'sub-032390': 22, 'sub-032344': 27, 'sub-032421': 22,
    'sub-032400': 22, 'sub-032385': 22, 'sub-032525': 22,
    'sub-032467': 22, 'sub-032508': 27, 'sub-032323': 22,
}

# Load participant metadata (age bins, sex)
meta = {}
meta_csv = LEMON_DIR / 'participants.csv'
if meta_csv.exists():
    with open(meta_csv) as f:
        for row in csv.DictReader(f):
            sid      = row['ID'].strip()
            age_bin  = row.get('Age', '?').strip()
            lo       = int(age_bin.split('-')[0]) if '-' in age_bin else 0
            meta[sid] = {'age_bin': age_bin,
                         'group': 'young' if lo <= 35 else 'old'}

print("Ze Alpha Peak Analysis — MPI-LEMON")
print(f"N = {len(SUBJECTS)} subjects | EC condition | "
      f"Resampled {RESAMPLE_HZ:.0f} Hz\n")

results = []

for sub_id in sorted(SUBJECTS.keys()):
    ec_path = LEMON_DIR / sub_id / f'{sub_id}_EC.set'
    if not ec_path.exists():
        print(f"  ⚠️  {sub_id}: EC.set not found, skip")
        continue

    m     = meta.get(sub_id, {})
    age   = SUBJECTS[sub_id]
    group = 'young' if age <= 35 else 'old'

    try:
        raw = mne.io.read_raw_eeglab(str(ec_path), preload=True, verbose=False)
        raw.resample(RESAMPLE_HZ, npad='auto')
        raw.crop(tmax=min(CROP_S, raw.times[-1]))
    except Exception as exc:
        print(f"  ❌ {sub_id}: {exc}")
        continue

    # Compute proxy Ze per channel, then take median f_peak + mean χ_Ze
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads') or \
            list(range(raw.info['nchan']))
    f_peaks, chi_vals = [], []
    for idx in picks:
        sig = raw.get_data(picks=[idx])[0]
        try:
            px = alpha_peak_ze(sig, RESAMPLE_HZ, f_band=ALPHA_BAND)
            f_peaks.append(px['f_peak'])
            chi_vals.append(px['chi_Ze'])
        except Exception:
            pass

    if not f_peaks:
        print(f"  ❌ {sub_id}: no valid channels")
        continue

    f_peak  = float(np.median(f_peaks))
    v_peak  = 2.0 * f_peak / RESAMPLE_HZ
    chi_Ze  = float(np.mean(chi_vals))

    results.append({
        'subject_id':    sub_id,
        'age':           age,
        'age_bin':       m.get('age_bin', '?'),
        'group':         group,
        'alpha_peak_hz': round(f_peak, 3),
        'v_peak':        round(v_peak, 5),
        'chi_Ze_peak':   round(chi_Ze, 5),
    })
    print(f"  {sub_id}  age={age:2d} {group:5s}  "
          f"f_peak={f_peak:.2f} Hz  v={v_peak:.4f}  χ_Ze={chi_Ze:.4f}")

# ── Group summaries ───────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
for grp in ('young', 'old'):
    gr = [r for r in results if r['group'] == grp]
    if not gr:
        continue
    fps  = [r['alpha_peak_hz'] for r in gr]
    chis = [r['chi_Ze_peak']   for r in gr]
    print(f"  {grp.upper():5s}  N={len(gr):2d}  "
          f"f_peak={np.mean(fps):.3f}±{np.std(fps):.3f} Hz  "
          f"χ_Ze={np.mean(chis):.4f}±{np.std(chis):.4f}")

chi_y = np.array([r['chi_Ze_peak'] for r in results if r['group'] == 'young'])
chi_o = np.array([r['chi_Ze_peak'] for r in results if r['group'] == 'old'])

if len(chi_y) >= 2 and len(chi_o) >= 2:
    st = group_statistics(chi_y, chi_o)
    print(f"\n  Group comparison (young vs old):")
    print(f"  Δ f_peak: {np.mean([r['alpha_peak_hz'] for r in results if r['group']=='young']):.3f}"
          f" − {np.mean([r['alpha_peak_hz'] for r in results if r['group']=='old']):.3f} Hz")
    print(f"  t={st['t']:.3f}  p={st['p']:.4f}  "
          f"d={st['cohens_d']:.3f} [{st['d_ci_95'][0]:.3f}, {st['d_ci_95'][1]:.3f}]  "
          f"power={st['power']:.0%}")
    print(f"  AUC={st['auc']:.3f} [{st['auc_ci_95'][0]:.3f}, "
          f"{st['auc_ci_95'][1]:.3f}]  p_MW={st['auc_p_onesided']:.4f}")
    if st['t'] > 0:
        print(f"  ✅ Ze hypothesis: χ_Ze(young) > χ_Ze(old)")
    else:
        print(f"  ⚠️  No significant young > old difference (likely underpowered)")

print(f"{'═'*64}")

# Save
out_path = OUT_DIR / 'ze_alpha_peak_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  💾 {out_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    f'Ze Alpha Peak Analysis — MPI-LEMON (N={len(results)})\n'
    f'Proxy method: α-peak → v_peak = 2f/{RESAMPLE_HZ:.0f} → χ_Ze',
    fontsize=11, fontweight='bold')

col = {'young': '#2E74B5', 'old': '#C00000'}
np.random.seed(0)

for ax_idx, (ylabel, key) in enumerate([
    ('Alpha peak (Hz)', 'alpha_peak_hz'),
    ('χ_Ze (proxy)',    'chi_Ze_peak'),
]):
    ax = axes[ax_idx]
    for grp, offset in [('young', -0.15), ('old', 0.15)]:
        gr   = [r for r in results if r['group'] == grp]
        vals = [r[key] for r in gr]
        jit  = np.random.uniform(-0.04, 0.04, len(vals))
        ax.scatter([1 + offset + j for j in jit], vals,
                   color=col[grp], s=45, alpha=0.75, zorder=4)
        ax.errorbar([1 + offset], [np.mean(vals)],
                    yerr=[np.std(vals, ddof=1) / len(vals)**0.5],
                    fmt='D', color=col[grp], ms=10, capsize=7, lw=2.5, zorder=5)
    ax.set_xticks([0.85, 1.15])
    ax.set_xticklabels(['Young\n(22–27 yr)', 'Old\n(67–72 yr)'])
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.set_xlim(0.5, 1.5)

# Age vs χ_Ze scatter
ax = axes[2]
for r in results:
    ax.scatter(r['age'], r['chi_Ze_peak'],
               color=col.get(r['group'], 'gray'), s=55, alpha=0.75)
all_ages  = [r['age']        for r in results]
all_chis  = [r['chi_Ze_peak'] for r in results]
if len(all_ages) >= 5:
    z      = np.polyfit(all_ages, all_chis, 1)
    r_corr = np.corrcoef(all_ages, all_chis)[0, 1]
    x_line = np.linspace(min(all_ages), max(all_ages), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'k--', lw=1.5,
            label=f'r = {r_corr:.3f}')
    ax.legend(fontsize=9)
ax.set_xlabel('Age (years)'); ax.set_ylabel('χ_Ze (proxy)')
ax.set_title('Age vs χ_Ze — Alpha Peak')

fig.legend(handles=[Patch(color='#2E74B5', label='Young (22–27 yr)'),
                    Patch(color='#C00000', label='Old (67–72 yr)')],
           loc='lower right', fontsize=9)
plt.tight_layout()
ppath = OUT_DIR / 'ze_alpha_peak.png'
plt.savefig(ppath, dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 {ppath}")
