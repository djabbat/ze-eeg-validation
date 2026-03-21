#!/usr/bin/env python3
"""
Ze batch pipeline for Dortmund Vital Study (ds005385).
Downloads EC-pre EDF (22MB), runs Ze alpha peak analysis, deletes EDF.
N=40 subjects (20 young 20-30, 20 old 63-70). Exact ages.
"""
import os, sys, json, subprocess
import numpy as np
from pathlib import Path
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import ze_cheating_index, V_STAR

import mne
mne.set_log_level('WARNING')

BASE = os.environ.get('ZE_DORTMUND_S3', 'https://s3.amazonaws.com/openneuro.org/ds005385')
DORTMUND_DIR = Path(os.environ.get('ZE_DORTMUND_DIR', str(Path(__file__).parent.parent / 'data' / 'dortmund')))
OUT_DIR = DORTMUND_DIR / 'results'
OUT_DIR.mkdir(exist_ok=True)

RESAMPLE_HZ = 128.0
CROP_S = 120  # use 2 minutes

BANDS = {'delta':(1,4),'theta':(4,8),'alpha':(8,12),'beta':(13,30),'gamma':(30,45)}


def alpha_peak_freq(raw, band=(7.5, 13.0)):
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if not len(picks): picks = list(range(raw.info['nchan']))
    peaks = []
    for idx in picks:
        sig = raw.get_data(picks=[idx])[0]
        freqs, psd = welch(sig, fs=raw.info['sfreq'], nperseg=int(raw.info['sfreq']*4))
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if mask.any():
            peaks.append(freqs[mask][np.argmax(psd[mask])])
    return float(np.median(peaks)) if peaks else 10.0


def bandpass_ze_mean(raw, fmin, fmax):
    rf = raw.copy().filter(fmin, fmax, method='iir', verbose=False)
    picks = mne.pick_types(rf.info, eeg=True, exclude='bads')
    if not len(picks): picks = list(range(rf.info['nchan']))
    chis = []
    for idx in picks:
        sig = rf.get_data(picks=[idx])[0]
        N = len(sig)
        if N < 2: continue
        binary = (sig > np.median(sig)).astype(int)
        v = np.sum(binary[1:] != binary[:-1]) / (N-1)
        chis.append(ze_cheating_index(v))
    return float(np.mean(chis)) if chis else 0.0


def process_subject(sub_id, age, sex):
    out_file = OUT_DIR / f'ze_{sub_id}.json'
    if out_file.exists():
        print(f"  ⏭  {sub_id} age={age}: cached")
        return json.loads(out_file.read_text())

    # URL: sub-XXX/ses-1/eeg/sub-XXX_ses-1_task-EyesClosed_acq-pre_eeg.edf
    url = f"{BASE}/{sub_id}/ses-1/eeg/{sub_id}_ses-1_task-EyesClosed_acq-pre_eeg.edf"
    edf_path = DORTMUND_DIR / f'{sub_id}_EC_pre.edf'

    print(f"  ⬇  {sub_id} age={age} {sex}... ", end='', flush=True)
    ret = subprocess.run(['curl', '-s', '-L', url, '-o', str(edf_path)], capture_output=True)
    if ret.returncode != 0 or not edf_path.exists() or edf_path.stat().st_size < 1_000_000:
        print("FAILED (no file or too small)")
        edf_path.unlink(missing_ok=True)
        return None
    print(f"{edf_path.stat().st_size//1024//1024}MB ", end='', flush=True)

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        raw.resample(RESAMPLE_HZ, npad='auto')
        raw.crop(tmax=min(CROP_S, raw.times[-1]))
    except Exception as e:
        print(f"ERR: {e}")
        edf_path.unlink(missing_ok=True)
        return None

    # Delete EDF to save space
    edf_path.unlink()

    # Alpha peak
    f_peak = alpha_peak_freq(raw)
    v_peak = 2 * f_peak / RESAMPLE_HZ
    chi_peak = ze_cheating_index(v_peak)

    # Band-wise
    bands = {}
    for bname, (fmin, fmax) in BANDS.items():
        bands[bname] = bandpass_ze_mean(raw, fmin, fmax)

    group = 'young' if int(age) <= 35 else 'old'
    result = {
        'subject_id': sub_id, 'age': int(age), 'sex': sex, 'group': group,
        'alpha_peak_hz': round(f_peak, 3),
        'v_peak': round(v_peak, 5),
        'chi_Ze_peak': round(chi_peak, 5),
        'bands': bands,
    }
    out_file.write_text(json.dumps(result, indent=2))
    print(f"f_peak={f_peak:.2f}Hz χ_Ze={chi_peak:.4f} γ={bands['gamma']:.4f}")
    return result


# Load batch
batch = []
with open('/tmp/dortmund_batch.tsv') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            batch.append((parts[0], parts[1], parts[2] if len(parts)>2 else '?'))

print(f"Dortmund Ze Pipeline — {len(batch)} subjects\n")

results = []
for sub_id, age, sex in batch:
    r = process_subject(sub_id, age, sex)
    if r:
        results.append(r)

# Statistics
print(f"\n{'═'*65}")
from scipy import stats as scipy_stats

y = [r for r in results if r['group']=='young']
o = [r for r in results if r['group']=='old']

print(f"N: young={len(y)}, old={len(o)}")
print()

# Alpha peak
y_fp = [r['alpha_peak_hz'] for r in y]
o_fp = [r['alpha_peak_hz'] for r in o]
y_chi = [r['chi_Ze_peak'] for r in y]
o_chi = [r['chi_Ze_peak'] for r in o]

if y_fp and o_fp:
    t, p = scipy_stats.ttest_ind(y_fp, o_fp)
    pool_sd = np.sqrt((np.var(y_fp,ddof=1)+np.var(o_fp,ddof=1))/2)
    d = (np.mean(y_fp)-np.mean(o_fp))/pool_sd if pool_sd>0 else 0
    print(f"Alpha peak:  Young={np.mean(y_fp):.3f}±{np.std(y_fp):.3f}  Old={np.mean(o_fp):.3f}±{np.std(o_fp):.3f}  t={t:.3f} p={p:.4f} d={d:.3f}")

    t2, p2 = scipy_stats.ttest_ind(y_chi, o_chi)
    pool_sd2 = np.sqrt((np.var(y_chi,ddof=1)+np.var(o_chi,ddof=1))/2)
    d2 = (np.mean(y_chi)-np.mean(o_chi))/pool_sd2 if pool_sd2>0 else 0
    print(f"χ_Ze(peak):  Young={np.mean(y_chi):.4f}±{np.std(y_chi):.4f}  Old={np.mean(o_chi):.4f}±{np.std(o_chi):.4f}  t={t2:.3f} p={p2:.4f} d={d2:.3f}")

# Age correlation (all subjects)
all_ages = [r['age'] for r in results]
all_chi  = [r['chi_Ze_peak'] for r in results]
all_fp   = [r['alpha_peak_hz'] for r in results]
if len(all_ages) >= 5:
    r_chi, p_chi = scipy_stats.pearsonr(all_ages, all_chi)
    r_fp,  p_fp  = scipy_stats.pearsonr(all_ages, all_fp)
    print(f"\nCorrelation (N={len(results)}):")
    print(f"  r(age, χ_Ze)    = {r_chi:.4f}  p={p_chi:.4f}  {'✅ sig' if p_chi<0.05 else '⚠️ ns'}")
    print(f"  r(age, f_peak)  = {r_fp:.4f}  p={p_fp:.4f}  {'✅ sig' if p_fp<0.05 else '⚠️ ns'}")

# Band-wise
print(f"\nBand-wise χ_Ze (Young vs Old):")
for bname in BANDS:
    yv = [r['bands'][bname] for r in y if bname in r.get('bands',{})]
    ov = [r['bands'][bname] for r in o if bname in r.get('bands',{})]
    if yv and ov:
        delta = np.mean(yv)-np.mean(ov)
        t,p = scipy_stats.ttest_ind(yv, ov)
        sig = '✅' if p<0.05 else '⚠️'
        print(f"  {bname:6s}: Y={np.mean(yv):.4f} O={np.mean(ov):.4f} Δ={delta:+.4f} p={p:.4f} {sig}")

print(f"{'═'*65}")

# Save combined
combined = OUT_DIR / 'ze_dortmund_combined.json'
combined.write_text(json.dumps(results, indent=2))
print(f"\n💾 {combined}")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f'Ze Analysis — Dortmund Vital Study (N={len(results)})\nAge vs α-peak χ_Ze', fontsize=11)

col = {'young': '#2E74B5', 'old': '#C00000'}

ax = axes[0]
for r in results:
    ax.scatter(r['age'], r['alpha_peak_hz'], color=col.get(r['group'],'gray'), s=50, alpha=0.7)
if all_ages:
    z = np.polyfit(all_ages, all_fp, 1)
    xl = np.linspace(min(all_ages), max(all_ages), 100)
    ax.plot(xl, np.poly1d(z)(xl), 'k--', lw=1.5, label=f'r={r_fp:.3f} p={p_fp:.3f}')
    ax.legend(fontsize=8)
ax.set_xlabel('Age'); ax.set_ylabel('Alpha peak (Hz)'); ax.set_title('Age vs Alpha Peak Freq')

ax = axes[1]
for r in results:
    ax.scatter(r['age'], r['chi_Ze_peak'], color=col.get(r['group'],'gray'), s=50, alpha=0.7)
if all_ages:
    z = np.polyfit(all_ages, all_chi, 1)
    ax.plot(xl, np.poly1d(z)(xl), 'k--', lw=1.5, label=f'r={r_chi:.3f} p={p_chi:.3f}')
    ax.legend(fontsize=8)
ax.set_xlabel('Age'); ax.set_ylabel('χ_Ze'); ax.set_title('Age vs χ_Ze (α-peak)')

ax = axes[2]
bands_names = list(BANDS.keys())
y_means = [np.mean([r['bands'][b] for r in y if b in r.get('bands',{})]) for b in bands_names]
o_means = [np.mean([r['bands'][b] for r in o if b in r.get('bands',{})]) for b in bands_names]
x = np.arange(len(bands_names))
ax.bar(x-0.2, y_means, 0.4, label='Young', color='#2E74B5', alpha=0.8)
ax.bar(x+0.2, o_means, 0.4, label='Old',   color='#C00000', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(bands_names, fontsize=8)
ax.set_ylabel('χ_Ze mean'); ax.set_title('Band-wise χ_Ze: Young vs Old')
ax.legend(fontsize=8)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#2E74B5',label='Young (20-30)'),
                    Patch(color='#C00000',label='Old (63-70)')],
           loc='lower right', fontsize=9)
plt.tight_layout()
ppath = OUT_DIR / 'ze_dortmund_results.png'
plt.savefig(ppath, dpi=150, bbox_inches='tight')
plt.close()
print(f"📊 {ppath}")
