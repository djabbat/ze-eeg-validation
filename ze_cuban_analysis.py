#!/usr/bin/env python3
"""
Ze Analysis on Cuban Human Normative EEG Database
===================================================
Zenodo 4244765 — N=211, ages 5-97, Eyes Closed + Eyes Open
Pre-computed cross-spectral matrices (Mcr), 19 channels, 0.39-19.11 Hz

Ze alpha peak: f_peak → v = 2*f/100 → χ_Ze
Prediction: inverted-U lifespan curve (development peak ~25yr, aging decline)
"""
import os
import scipy.io, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import ze_cheating_index, V_STAR

_default_cuban = str(Path(__file__).parent.parent / 'data' / 'cuban' /
                     'oldgandalf-FirstWaveCubanHumanNormativeEEGProject-3783da7')
DATA_DIR = Path(os.environ.get('ZE_CUBAN_DIR', _default_cuban))
OUT_DIR = DATA_DIR / 'results'
OUT_DIR.mkdir(exist_ok=True)

FS_ORIGINAL = 100.0  # sampling rate of Cuban EEG data


def alpha_peak_from_mcr(mcr, freqs, band=(7.5, 13.0)):
    """Extract alpha peak frequency from cross-spectral matrix diagonal."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return 10.0
    # Average PSD across all 19 channels (diagonal = power)
    psd_avg = np.mean([mcr[i, i, :].real for i in range(mcr.shape[0])], axis=0)
    return float(freqs[mask][np.argmax(psd_avg[mask])])


def load_subject(mat_path):
    mat = scipy.io.loadmat(str(mat_path))
    age = float(mat['age'].flatten()[0])
    sex = str(mat['sex'].flat[0]).strip()
    freqs = mat['frange'].flatten()
    mcr = mat['Mcr']
    f_peak = alpha_peak_from_mcr(mcr, freqs)
    v = 2 * f_peak / FS_ORIGINAL
    chi = ze_cheating_index(v)
    return {'age': round(age, 2), 'sex': sex, 'alpha_peak_hz': round(f_peak, 4),
            'v': round(v, 5), 'chi_Ze': round(chi, 5)}


# Load all EC files
print("Loading Cuban Normative EEG Database (EC condition)...")
ec_dir = DATA_DIR / 'EyesClose'
results = []
for mat_file in sorted(ec_dir.glob('*_cross.mat')):
    try:
        r = load_subject(mat_file)
        r['file'] = mat_file.name
        r['condition'] = 'EC'
        results.append(r)
    except Exception as e:
        print(f"  Error {mat_file.name}: {e}")

print(f"Loaded {len(results)} EC subjects")
ages = np.array([r['age'] for r in results])
chis = np.array([r['chi_Ze'] for r in results])
fps  = np.array([r['alpha_peak_hz'] for r in results])

print(f"Age range: {ages.min():.1f} – {ages.max():.1f}  mean={ages.mean():.1f}")
print(f"Alpha peak: {fps.mean():.2f} ± {fps.std():.2f} Hz")
print(f"χ_Ze: {chis.mean():.4f} ± {chis.std():.4f}")

# Age groups for comparison
groups = {
    'children (5-12)':  [r for r in results if r['age'] < 12],
    'teens (12-18)':    [r for r in results if 12 <= r['age'] < 18],
    'young (18-35)':    [r for r in results if 18 <= r['age'] < 35],
    'middle (35-60)':   [r for r in results if 35 <= r['age'] < 60],
    'old (60-80)':      [r for r in results if 60 <= r['age'] < 80],
    'oldest (80+)':     [r for r in results if r['age'] >= 80],
}

print(f"\n{'─'*65}")
print(f"{'Group':20s}  {'N':4s}  {'f_peak Hz':12s}  {'χ_Ze':12s}")
print(f"{'─'*65}")
for gname, gr in groups.items():
    if not gr: continue
    fp_g = [r['alpha_peak_hz'] for r in gr]
    ch_g = [r['chi_Ze'] for r in gr]
    print(f"  {gname:20s}  {len(gr):4d}  "
          f"{np.mean(fp_g):.2f}±{np.std(fp_g):.2f}   "
          f"{np.mean(ch_g):.4f}±{np.std(ch_g):.4f}")

# Statistics
r_chi, p_chi = stats.pearsonr(ages, chis)
r_fp,  p_fp  = stats.pearsonr(ages, fps)
print(f"\nLinear correlation (N={len(results)}):")
print(f"  r(age, χ_Ze)   = {r_chi:.4f}  p={p_chi:.5f}  {'✅' if p_chi<0.05 else '⚠️'}")
print(f"  r(age, f_peak) = {r_fp:.4f}  p={p_fp:.5f}  {'✅' if p_fp<0.05 else '⚠️'}")

# Quadratic fit (inverted U expected)
coeffs2 = np.polyfit(ages, chis, 2)
peak_age = -coeffs2[1] / (2*coeffs2[0])
print(f"\nQuadratic fit: χ_Ze = {coeffs2[0]:.6f}·age² + {coeffs2[1]:.5f}·age + {coeffs2[2]:.4f}")
print(f"  Peak age: {peak_age:.1f} years {'✅' if 15 < peak_age < 40 else '⚠️'}")
# R² of quadratic
y_pred2 = np.polyval(coeffs2, ages)
ss_res = np.sum((chis - y_pred2)**2)
ss_tot = np.sum((chis - chis.mean())**2)
r2_quad = 1 - ss_res/ss_tot
print(f"  R² (quadratic) = {r2_quad:.4f}")

# Children vs old t-test
children = [r['chi_Ze'] for r in results if r['age'] < 12]
old      = [r['chi_Ze'] for r in results if r['age'] >= 65]
young    = [r['chi_Ze'] for r in results if 18 <= r['age'] < 35]
if children and young:
    t, p = stats.ttest_ind(young, children)
    pool_sd = np.sqrt((np.var(young,ddof=1)+np.var(children,ddof=1))/2)
    d = (np.mean(young)-np.mean(children))/pool_sd
    print(f"\n  Young adult vs Children: t={t:.3f} p={p:.4f} d={d:.3f}")
if young and old:
    t, p = stats.ttest_ind(young, old)
    pool_sd = np.sqrt((np.var(young,ddof=1)+np.var(old,ddof=1))/2)
    d = (np.mean(young)-np.mean(old))/pool_sd
    print(f"  Young adult vs Old:      t={t:.3f} p={p:.4f} d={d:.3f}")
if children and old:
    t, p = stats.ttest_ind(old, children)
    print(f"  Old vs Children (fp):    similar spectrum, t={t:.3f} p={p:.4f}")

print(f"{'─'*65}")

# Save
(OUT_DIR / 'ze_cuban_ec.json').write_text(json.dumps(results, indent=2))
print(f"\n💾 {OUT_DIR/'ze_cuban_ec.json'}")

# ── PLOT ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Ze Analysis — Cuban Human Normative EEG (N={len(results)})\n'
             'Ages 5–97, Eyes Closed, α-peak frequency → v → χ_Ze',
             fontsize=12, fontweight='bold')

age_range = np.linspace(5, max(ages)+2, 200)

# 1. α-peak vs age (scatter + quadratic)
ax = axes[0,0]
ax.scatter(ages, fps, c=ages, cmap='RdYlBu_r', s=25, alpha=0.7, zorder=5)
z2 = np.polyfit(ages, fps, 2)
ax.plot(age_range, np.polyval(z2, age_range), 'k-', lw=2)
ax.set_xlabel('Age (years)'); ax.set_ylabel('Alpha peak (Hz)')
ax.set_title(f'Alpha Peak Frequency vs Age\nr={r_fp:.3f} p={p_fp:.4f}')
ax.set_xlim(0, 100)

# 2. χ_Ze vs age (scatter + quadratic)
ax = axes[0,1]
sc = ax.scatter(ages, chis, c=ages, cmap='RdYlBu_r', s=25, alpha=0.7, zorder=5)
ax.plot(age_range, np.polyval(coeffs2, age_range), 'k-', lw=2,
        label=f'quadratic R²={r2_quad:.3f}\npeak={peak_age:.0f}yr')
ax.axvline(peak_age, color='green', ls='--', lw=1, alpha=0.7)
ax.set_xlabel('Age (years)'); ax.set_ylabel('χ_Ze (alpha peak)')
ax.set_title(f'χ_Ze vs Age (Ze Hypothesis)\nr={r_chi:.3f} p={p_chi:.4f}')
ax.legend(fontsize=8); ax.set_xlim(0, 100)
plt.colorbar(sc, ax=ax, label='Age')

# 3. Group bar chart
ax = axes[1,0]
gnames = [g for g,v in groups.items() if v]
g_chi_means = [np.mean([r['chi_Ze'] for r in groups[g]]) for g in gnames]
g_chi_errs  = [np.std([r['chi_Ze'] for r in groups[g]])/len(groups[g])**0.5 for g in gnames]
g_ns = [len(groups[g]) for g in gnames]
colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(gnames)))
bars = ax.bar(range(len(gnames)), g_chi_means, yerr=g_chi_errs,
              capsize=5, color=colors, edgecolor='white')
ax.set_xticks(range(len(gnames)))
ax.set_xticklabels([f"{g}\n(N={n})" for g,n in zip(gnames,g_ns)], fontsize=7)
ax.set_ylabel('χ_Ze mean ± SE'); ax.set_title('χ_Ze by Age Group')

# 4. α-peak by group
ax = axes[1,1]
g_fp_means = [np.mean([r['alpha_peak_hz'] for r in groups[g]]) for g in gnames]
g_fp_errs  = [np.std([r['alpha_peak_hz'] for r in groups[g]])/len(groups[g])**0.5 for g in gnames]
ax.bar(range(len(gnames)), g_fp_means, yerr=g_fp_errs,
       capsize=5, color=colors, edgecolor='white')
ax.set_xticks(range(len(gnames)))
ax.set_xticklabels([f"{g}\n(N={n})" for g,n in zip(gnames,g_ns)], fontsize=7)
ax.set_ylabel('Alpha peak (Hz)'); ax.set_title('Alpha Peak by Age Group')

plt.tight_layout()
ppath = OUT_DIR / 'ze_cuban_lifespan.png'
plt.savefig(ppath, dpi=150, bbox_inches='tight')
plt.close()
print(f"📊 {ppath}")
