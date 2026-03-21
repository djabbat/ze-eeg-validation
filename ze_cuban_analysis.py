#!/usr/bin/env python3
"""
Ze Analysis — Cuban Human Normative EEG Database
=================================================
Zenodo 4244765: N=211, ages 5–97, Eyes Closed + Eyes Open
Pre-computed cross-spectral matrices (Mcr), 19 channels, fs=100 Hz

Uses eeg_ze_processor API:
  load_cuban_mcr()  — load .mat, extract α-peak → χ_Ze (proxy method, fs=100 Hz)
  group_statistics()— t-test, Cohen's d, bootstrap CI, AUC, ANCOVA

Prediction: inverted-U lifespan curve (peak ~25–40 yr)

Dataset: https://zenodo.org/records/4244765
Set ZE_CUBAN_DIR to the unzipped dataset folder:
  export ZE_CUBAN_DIR=/path/to/oldgandalf-FirstWaveCubanHumanNormativeEEGProject-*
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import V_STAR, load_cuban_mcr, group_statistics

_default_cuban = str(Path(__file__).parent.parent / 'data' / 'cuban' /
                     'oldgandalf-FirstWaveCubanHumanNormativeEEGProject-3783da7')
DATA_DIR = Path(os.environ.get('ZE_CUBAN_DIR', _default_cuban))
OUT_DIR  = DATA_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_BAND = (7.5, 13.0)

# ── Load all EC files ─────────────────────────────────────────────────────────
print("Loading Cuban Normative EEG Database (EC condition)...")
ec_dir   = DATA_DIR / 'EyesClose'
results  = []
n_errors = 0

for mat_file in sorted(ec_dir.glob('*_cross.mat')):
    try:
        r = load_cuban_mcr(str(mat_file), f_band=ALPHA_BAND)
        r['file']      = mat_file.name
        r['condition'] = 'EC'
        results.append(r)
    except Exception as exc:
        print(f"  Error {mat_file.name}: {exc}")
        n_errors += 1

print(f"Loaded {len(results)} EC subjects  ({n_errors} errors)")

ages = np.array([r['age']    for r in results])
chis = np.array([r['chi_Ze'] for r in results])
fps  = np.array([r['f_peak'] for r in results])

print(f"Age range: {ages.min():.1f} – {ages.max():.1f}  mean={ages.mean():.1f}")
print(f"Alpha peak: {fps.mean():.2f} ± {fps.std():.2f} Hz")
print(f"χ_Ze: {chis.mean():.4f} ± {chis.std():.4f}")

# ── Age group table ───────────────────────────────────────────────────────────
groups = {
    'children (5–12)':   [r for r in results if r['age'] <  12],
    'teens (12–18)':     [r for r in results if 12 <= r['age'] < 18],
    'young (18–35)':     [r for r in results if 18 <= r['age'] < 35],
    'middle (35–60)':    [r for r in results if 35 <= r['age'] < 60],
    'old (60–80)':       [r for r in results if 60 <= r['age'] < 80],
    'oldest (80+)':      [r for r in results if r['age'] >= 80],
}

print(f"\n{'─'*68}")
print(f"{'Group':22s}  {'N':>4}  {'f_peak Hz':>14}  {'χ_Ze':>14}")
print(f"{'─'*68}")
for gname, gr in groups.items():
    if not gr:
        continue
    fp_g = [r['f_peak']  for r in gr]
    ch_g = [r['chi_Ze'] for r in gr]
    print(f"  {gname:22s}  {len(gr):4d}  "
          f"{np.mean(fp_g):.2f}±{np.std(fp_g):.2f}  "
          f"{np.mean(ch_g):.4f}±{np.std(ch_g):.4f}")

# ── Quadratic fit (inverted-U) ────────────────────────────────────────────────
mask_lt90   = ages < 90
coeffs      = np.polyfit(ages[mask_lt90], chis[mask_lt90], 2)
peak_age    = -coeffs[1] / (2 * coeffs[0])
y_pred_quad = np.polyval(coeffs, ages[mask_lt90])
ss_res      = np.sum((chis[mask_lt90] - y_pred_quad) ** 2)
ss_tot      = np.sum((chis[mask_lt90] - chis[mask_lt90].mean()) ** 2)
r2_quad     = 1 - ss_res / ss_tot
rmse        = np.sqrt(ss_res / len(y_pred_quad))

print(f"\nQuadratic fit (ages <90):")
print(f"  χ_Ze = {coeffs[0]:.6f}·age² + {coeffs[1]:.5f}·age + {coeffs[2]:.4f}")
print(f"  Peak age: {peak_age:.1f} yr  {'✅' if 20 < peak_age < 45 else '⚠️'}")
print(f"  R² = {r2_quad:.4f}  RMSE = {rmse:.4f}")

# Bootstrap CI for peak age
rng        = np.random.default_rng(42)
ages_lt90  = ages[mask_lt90]
chis_lt90  = chis[mask_lt90]
peak_boot  = []
for _ in range(10000):
    idx_b = rng.choice(len(ages_lt90), size=len(ages_lt90), replace=True)
    c_b   = np.polyfit(ages_lt90[idx_b], chis_lt90[idx_b], 2)
    if c_b[0] < 0:     # only valid inverted-U
        peak_boot.append(-c_b[1] / (2 * c_b[0]))
peak_ci = np.percentile(peak_boot, [2.5, 97.5]) if peak_boot else [float('nan')]*2
print(f"  Peak 95% CI (bootstrap): [{peak_ci[0]:.1f}, {peak_ci[1]:.1f}] yr")

# Linear correlation
r_chi, p_chi = stats.pearsonr(ages, chis)
r_fp,  p_fp  = stats.pearsonr(ages, fps)
print(f"\nLinear correlation (N={len(results)}):")
print(f"  r(age, χ_Ze)   = {r_chi:.4f}  p={p_chi:.5f}  {'✅' if p_chi < 0.05 else '⚠️ ns'}")
print(f"  r(age, f_peak) = {r_fp:.4f}  p={p_fp:.5f}  {'✅' if p_fp  < 0.05 else '⚠️ ns'}")

# ── Key pairwise comparisons using group_statistics() ────────────────────────
young_chi   = np.array([r['chi_Ze'] for r in results if 18 <= r['age'] < 35])
children_chi= np.array([r['chi_Ze'] for r in results if r['age'] < 12])
elderly_chi = np.array([r['chi_Ze'] for r in results if r['age'] >= 65])

print(f"\nPairwise comparisons (group_statistics, bootstrap 10,000):")
for label, c1, c2 in [
    ('Young (18–35) vs Elderly (≥65)',  young_chi, elderly_chi),
    ('Young (18–35) vs Children (<12)', young_chi, children_chi),
]:
    if len(c1) < 2 or len(c2) < 2:
        print(f"  {label}: insufficient N")
        continue
    st = group_statistics(c1, c2, n_boot=10000)
    print(f"\n  {label}:")
    print(f"    N={st['n_young']} vs N={st['n_old']}")
    print(f"    Mean: {st['mean_young']:.4f}±{st['sd_young']:.4f} vs "
          f"{st['mean_old']:.4f}±{st['sd_old']:.4f}")
    print(f"    t={st['t']:.3f}  p={st['p']:.5f}  "
          f"d={st['cohens_d']:.3f} [{st['d_ci_95'][0]:.3f}, {st['d_ci_95'][1]:.3f}]  "
          f"power={st['power']:.0%}")
    print(f"    AUC={st['auc']:.3f} [{st['auc_ci_95'][0]:.3f}, "
          f"{st['auc_ci_95'][1]:.3f}]  p_MW={st['auc_p_onesided']:.4f}")

print(f"\n{'─'*68}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_json = OUT_DIR / 'ze_cuban_ec.json'
out_json.write_text(json.dumps(results, indent=2))
print(f"💾 {out_json}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'Ze Analysis — Cuban Human Normative EEG (N={len(results)})\n'
    f'Ages 5–97, Eyes Closed | Proxy method (α-peak → v → χ_Ze) | fs=100 Hz',
    fontsize=12, fontweight='bold')

age_line = np.linspace(5, ages.max() + 2, 300)

# 1. Alpha peak vs age
ax = axes[0, 0]
ax.scatter(ages, fps, c=ages, cmap='RdYlBu_r', s=22, alpha=0.7, zorder=5)
z2 = np.polyfit(ages[mask_lt90], fps[mask_lt90], 2)
ax.plot(age_line, np.polyval(z2, age_line), 'k-', lw=2)
ax.set_xlabel('Age (years)'); ax.set_ylabel('Alpha peak (Hz)')
ax.set_title(f'Alpha Peak Frequency vs Age\nr={r_fp:.3f} p={p_fp:.4f}')
ax.set_xlim(0, 100)

# 2. χ_Ze vs age with quadratic fit
ax = axes[0, 1]
sc = ax.scatter(ages, chis, c=ages, cmap='RdYlBu_r', s=22, alpha=0.7, zorder=5)
ax.plot(age_line, np.polyval(coeffs, age_line), 'k-', lw=2,
        label=f'Quadratic fit\nR²={r2_quad:.3f}  RMSE={rmse:.3f}\npeak={peak_age:.0f} yr '
              f'[{peak_ci[0]:.0f}, {peak_ci[1]:.0f}]')
ax.axvline(peak_age, color='green', ls='--', lw=1.2, alpha=0.7)
ax.set_xlabel('Age (years)'); ax.set_ylabel('χ_Ze')
ax.set_title(f'χ_Ze vs Age — Inverted-U Hypothesis\nr={r_chi:.3f} p={p_chi:.4f}')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(0, 100)
plt.colorbar(sc, ax=ax, label='Age')

# 3. Group bar chart (mean ± SE + jittered points)
ax = axes[1, 0]
gnames    = [g for g, v in groups.items() if v and g != 'oldest (80+)']
g_means   = [np.mean([r['chi_Ze'] for r in groups[g]]) for g in gnames]
g_ses     = [np.std([r['chi_Ze']  for r in groups[g]], ddof=1) /
             np.sqrt(len(groups[g])) for g in gnames]
g_ns      = [len(groups[g]) for g in gnames]
colors_g  = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(gnames)))
ax.bar(range(len(gnames)), g_means, yerr=g_ses, capsize=5,
       color=colors_g, edgecolor='white', alpha=0.85)
np.random.seed(1)
for gi, gname in enumerate(gnames):
    vals = np.array([r['chi_Ze'] for r in groups[gname]])
    jit  = np.random.uniform(-0.3, 0.3, len(vals))
    ax.scatter(gi + jit, vals, color=colors_g[gi], s=12, alpha=0.45, zorder=3)
ax.set_xticks(range(len(gnames)))
ax.set_xticklabels([f"{g}\n(N={n})" for g, n in zip(gnames, g_ns)], fontsize=7)
ax.set_ylabel('χ_Ze mean ± SE'); ax.set_title('χ_Ze by Age Group')

# 4. Alpha peak by group
ax = axes[1, 1]
g_fp_means = [np.mean([r['f_peak'] for r in groups[g]]) for g in gnames]
g_fp_ses   = [np.std([r['f_peak'] for r in groups[g]], ddof=1) /
              np.sqrt(len(groups[g])) for g in gnames]
ax.bar(range(len(gnames)), g_fp_means, yerr=g_fp_ses, capsize=5,
       color=colors_g, edgecolor='white', alpha=0.85)
ax.set_xticks(range(len(gnames)))
ax.set_xticklabels([f"{g}\n(N={n})" for g, n in zip(gnames, g_ns)], fontsize=7)
ax.set_ylabel('Alpha peak (Hz)'); ax.set_title('Alpha Peak by Age Group')

plt.tight_layout()
ppath = OUT_DIR / 'ze_cuban_lifespan.png'
plt.savefig(ppath, dpi=150, bbox_inches='tight')
plt.close()
print(f"📊 {ppath}")
