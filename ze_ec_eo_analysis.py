import os
#!/usr/bin/env python3
"""
Ze EC vs EO Analysis — Subject 360
===================================
Eyes-Closed (EC) vs Eyes-Open (EO) comparison using Ze metrics.

Ze hypothesis: Eyes-Open (alpha desynchronized, higher freq) → higher v → higher χ_Ze
Eyes-Closed: dominant alpha ~10Hz → v ≈ 0.156 at 128Hz → χ_Ze ≈ 0.42
Eyes-Open: broader spectrum, alpha suppressed → higher v → higher χ_Ze
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import (
    binarize, ze_velocity, ze_cheating_index, ze_proper_time,
    compute_ze_metrics, V_STAR, CHI_MAX_LIVING
)

import mne
mne.set_log_level('WARNING')

VHDR  = os.environ.get('ZE_ZENODO_VHDR', str(Path(__file__).parent.parent / 'data' / 'zenodo' / '360.vhdr'))
SFREQ_TARGET = 128  # Hz — Ze-compatible rate

# Marker positions (in samples at 512Hz)
# BegF1/EndF1 = eyes-closed; BegO1/EndO1 = eyes-open
SEGMENTS = {
    'EC1': (4592,   35534),
    'EO1': (35535,  66091),
    'EC2': (66092,  97075),
    'EO2': (97076, 127870),
    'EC3': (127871, 158702),
    'EO3': (158703, 189534),
}

print("Loading BrainVision data...")
raw = mne.io.read_raw_brainvision(VHDR, preload=True, verbose=False)
print(f"  {raw.info['nchan']} ch, {raw.info['sfreq']:.0f} Hz, {raw.times[-1]:.1f}s")

# Resample
raw.resample(SFREQ_TARGET, npad='auto')
print(f"  Resampled → {SFREQ_TARGET} Hz")

sfreq = raw.info['sfreq']
scale = sfreq / 512.0  # factor to convert original sample positions to new rate

# Pick EEG channels (all 128 in this dataset)
picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
if len(picks) == 0:
    picks = list(range(raw.info['nchan']))

print(f"  Using {len(picks)} channels\n")

results = {}
for seg_name, (start_orig, end_orig) in SEGMENTS.items():
    start = int(start_orig * scale)
    end   = int(end_orig   * scale)

    # Extract segment data
    data = raw.get_data(picks=picks, start=start, stop=end)  # shape: (n_ch, n_times)
    n_ch, n_times = data.shape

    chi_list = []
    v_list   = []
    for ch in range(n_ch):
        m = compute_ze_metrics(data[ch])
        chi_list.append(m['chi_Ze'])
        v_list.append(m['v'])

    seg_result = {
        'n_times': n_times,
        'duration_s': n_times / sfreq,
        'chi_Ze_mean': float(np.mean(chi_list)),
        'chi_Ze_std':  float(np.std(chi_list)),
        'v_mean':      float(np.mean(v_list)),
        'v_std':       float(np.std(v_list)),
        'condition': 'EC' if seg_name.startswith('EC') else 'EO',
    }
    results[seg_name] = seg_result

    icon = "👁 " if seg_name.startswith('EO') else "😌"
    print(f"  {icon} {seg_name:4s}  χ_Ze={seg_result['chi_Ze_mean']:.4f} ±{seg_result['chi_Ze_std']:.4f}"
          f"  v={seg_result['v_mean']:.4f}  ({seg_result['duration_s']:.1f}s)")

# Aggregate EC vs EO
ec_chis = [results[k]['chi_Ze_mean'] for k in results if k.startswith('EC')]
eo_chis = [results[k]['chi_Ze_mean'] for k in results if k.startswith('EO')]
ec_vs   = [results[k]['v_mean']      for k in results if k.startswith('EC')]
eo_vs   = [results[k]['v_mean']      for k in results if k.startswith('EO')]

print(f"\n{'─'*55}")
print(f"  Eyes-Closed (EC):  χ_Ze = {np.mean(ec_chis):.4f} ± {np.std(ec_chis):.4f}")
print(f"  Eyes-Open   (EO):  χ_Ze = {np.mean(eo_chis):.4f} ± {np.std(eo_chis):.4f}")
delta = np.mean(eo_chis) - np.mean(ec_chis)
print(f"  Δ χ_Ze (EO − EC):  {delta:+.4f}")
if delta > 0:
    print("  ✅ Ze hypothesis SUPPORTED: EO > EC (alpha desync → higher χ_Ze)")
else:
    print("  ⚠️  Ze hypothesis NOT confirmed: EO ≤ EC")
print(f"{'─'*55}")

# Save results
out_path = '/home/oem/Desktop/EEG/data/zenodo/results/ze_ec_eo_360.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  💾 Saved: {out_path}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Ze Analysis: EC vs EO — Subject 360\n'
             '(Jabès et al. 2021, Zenodo 3875159)', fontsize=11, fontweight='bold')

seg_names = list(results.keys())
chis = [results[s]['chi_Ze_mean'] for s in seg_names]
chi_errs = [results[s]['chi_Ze_std'] for s in seg_names]
colors = ['#5B9BD5' if s.startswith('EO') else '#ED7D31' for s in seg_names]

ax = axes[0]
bars = ax.bar(seg_names, chis, color=colors, yerr=chi_errs, capsize=4,
              edgecolor='white', linewidth=0.8)
ax.axhline(CHI_MAX_LIVING, color='green', linestyle='--', linewidth=1,
           label=f'χ_max_living = {CHI_MAX_LIVING}')
ax.axhline(np.mean(eo_chis), color='#5B9BD5', linestyle=':', linewidth=1.5,
           label=f'EO mean = {np.mean(eo_chis):.3f}')
ax.axhline(np.mean(ec_chis), color='#ED7D31', linestyle=':', linewidth=1.5,
           label=f'EC mean = {np.mean(ec_chis):.3f}')
ax.set_ylabel("χ_Ze mean (across 128 channels)")
ax.set_ylim(0, max(chis) * 1.3)
ax.set_title("χ_Ze per segment")
ax.legend(fontsize=7)

from matplotlib.patches import Patch
ax2 = axes[1]
ec_vals = [results[k]['chi_Ze_mean'] for k in ['EC1','EC2','EC3']]
eo_vals = [results[k]['chi_Ze_mean'] for k in ['EO1','EO2','EO3']]
ax2.plot([1,2,3], ec_vals, 'o-', color='#ED7D31', linewidth=2, markersize=8, label='Eyes-Closed (EC)')
ax2.plot([1,2,3], eo_vals, 's-', color='#5B9BD5', linewidth=2, markersize=8, label='Eyes-Open (EO)')
ax2.set_xticks([1,2,3])
ax2.set_xticklabels(['Segment 1', 'Segment 2', 'Segment 3'])
ax2.set_ylabel("χ_Ze mean")
ax2.set_title("EC vs EO trajectory")
ax2.legend(fontsize=9)
ax2.set_ylim(0, max(eo_vals + ec_vals) * 1.3)

# Add legend for bar chart
legend_elements = [Patch(facecolor='#5B9BD5', label='Eyes-Open (EO)'),
                   Patch(facecolor='#ED7D31', label='Eyes-Closed (EC)')]
axes[0].legend(handles=legend_elements + axes[0].get_lines()[:2], fontsize=7)

plt.tight_layout()
plot_path = '/home/oem/Desktop/EEG/data/zenodo/results/ze_ec_eo_360.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Plot: {plot_path}")
