#!/usr/bin/env python3
"""
Ze batch pipeline for Dortmund Vital Study (ds005385).
Downloads EC-pre EDF, runs proxy + narrowband Ze analysis, deletes EDF.
N=60 subjects (30 young 20–30 yr, 30 old 63–70 yr).

Uses eeg_ze_processor API:
  alpha_peak_ze()   — proxy method  (PSD α-peak → v_peak → χ_Ze)
  narrowband_ze()   — narrowband Ze (bandpass 8–12 Hz → binarize → χ_Ze)
  group_statistics()— t-test, Cohen's d, CI, AUC, ANCOVA, sex interaction

Dataset: https://openneuro.org/datasets/ds005385
Set ZE_DORTMUND_DIR env var or edit default below.
"""
import os, sys, json, subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import (
    ze_cheating_index, V_STAR,
    alpha_peak_ze, narrowband_ze, group_statistics,
)

import mne
mne.set_log_level('WARNING')

BASE         = os.environ.get('ZE_DORTMUND_S3',
               'https://s3.amazonaws.com/openneuro.org/ds005385')
DORTMUND_DIR = Path(os.environ.get('ZE_DORTMUND_DIR',
               str(Path(__file__).parent.parent / 'data' / 'dortmund')))
OUT_DIR      = DORTMUND_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLE_HZ = 128.0
CROP_S      = 120      # 2-minute EC-pre condition
ALPHA_BAND  = (7.5, 13.0)
NB_LOW, NB_HIGH = 8.0, 12.0
BANDS = {'delta':(1,4), 'theta':(4,8), 'alpha':(8,12),
         'beta':(13,30), 'gamma':(30,45)}


def _channel_ze(raw, compute_fn, **kwargs):
    """Apply a per-channel Ze function and return mean χ_Ze across channels."""
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if not len(picks):
        picks = list(range(raw.info['nchan']))
    vals = []
    for idx in picks:
        sig = raw.get_data(picks=[idx])[0]
        try:
            vals.append(compute_fn(sig, raw.info['sfreq'], **kwargs)['chi_Ze'])
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float('nan')


def process_subject(sub_id, age, sex):
    out_file = OUT_DIR / f'ze_{sub_id}.json'
    if out_file.exists():
        print(f"  ⏭  {sub_id} age={age}: cached")
        return json.loads(out_file.read_text())

    url = (f"{BASE}/{sub_id}/ses-1/eeg/"
           f"{sub_id}_ses-1_task-EyesClosed_acq-pre_eeg.edf")
    edf_path = DORTMUND_DIR / f'{sub_id}_EC_pre.edf'

    print(f"  ⬇  {sub_id} age={age} {sex}... ", end='', flush=True)
    ret = subprocess.run(['curl', '-s', '-L', url, '-o', str(edf_path)],
                        capture_output=True)
    if ret.returncode != 0 or not edf_path.exists() \
            or edf_path.stat().st_size < 1_000_000:
        print("FAILED (no file or too small)")
        edf_path.unlink(missing_ok=True)
        return None
    print(f"{edf_path.stat().st_size // 1024 // 1024}MB ", end='', flush=True)

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        raw.resample(RESAMPLE_HZ, npad='auto')
        raw.crop(tmax=min(CROP_S, raw.times[-1]))
    except Exception as exc:
        print(f"ERR: {exc}")
        edf_path.unlink(missing_ok=True)
        return None
    edf_path.unlink()   # remove EDF to save disk space

    # ── Proxy method: α-peak → v_peak → χ_Ze ──
    chi_proxy = _channel_ze(raw, alpha_peak_ze, f_band=ALPHA_BAND)
    # Also get median f_peak for reporting
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads') or \
            list(range(raw.info['nchan']))
    f_peaks = []
    for idx in picks:
        sig = raw.get_data(picks=[idx])[0]
        try:
            f_peaks.append(alpha_peak_ze(sig, RESAMPLE_HZ, f_band=ALPHA_BAND)['f_peak'])
        except Exception:
            pass
    f_peak = float(np.median(f_peaks)) if f_peaks else float('nan')
    v_peak = 2.0 * f_peak / RESAMPLE_HZ

    # ── Narrowband Ze: bandpass 8–12 Hz → binarize → χ_Ze ──
    chi_nb = _channel_ze(raw, narrowband_ze,
                         lowcut=NB_LOW, highcut=NB_HIGH)

    # ── Band-wise narrowband Ze ──
    bands = {}
    for bname, (flo, fhi) in BANDS.items():
        bands[bname] = _channel_ze(raw, narrowband_ze, lowcut=flo, highcut=fhi)

    group  = 'young' if int(age) <= 35 else 'old'
    result = {
        'subject_id':    sub_id,
        'age':           int(age),
        'sex':           sex,
        'group':         group,
        'alpha_peak_hz': round(f_peak,    3),
        'v_peak':        round(v_peak,    5),
        'chi_Ze_peak':   round(chi_proxy, 5),   # proxy method
        'chi_Ze_nb':     round(chi_nb,    5),   # narrowband Ze
        'bands':         {k: round(v, 5) for k, v in bands.items()},
    }
    out_file.write_text(json.dumps(result, indent=2))
    print(f"f_peak={f_peak:.2f}Hz χ_Ze(proxy)={chi_proxy:.4f} "
          f"χ_Ze(nb)={chi_nb:.4f}")
    return result


# ── Load batch list ──────────────────────────────────────────────────────────
batch = []
batch_tsv = Path('/tmp/dortmund_batch.tsv')
if batch_tsv.exists():
    with open(batch_tsv) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                batch.append((parts[0], parts[1],
                              parts[2] if len(parts) > 2 else '?'))
else:
    # Try to use already-cached results
    cached = list(OUT_DIR.glob('ze_sub-*.json'))
    if cached:
        print(f"  No batch TSV found — loading {len(cached)} cached results")
    else:
        print("⚠️  No batch TSV at /tmp/dortmund_batch.tsv and no cached results.")
        print("   Provide the TSV or set ZE_DORTMUND_DIR to a folder with JSON results.")

print(f"Dortmund Ze Pipeline — {len(batch)} subjects queued\n")

results = []
for sub_id, age, sex in batch:
    r = process_subject(sub_id, age, sex)
    if r:
        results.append(r)

# Load cached results if batch produced nothing
if not results:
    for jf in sorted(OUT_DIR.glob('ze_sub-*.json')):
        try:
            r = json.loads(jf.read_text())
            # Back-fill chi_Ze_nb if absent (old format)
            if 'chi_Ze_nb' not in r and 'bands' in r:
                r['chi_Ze_nb'] = r['bands'].get('alpha', float('nan'))
            results.append(r)
        except Exception:
            pass

if not results:
    print("No results available.")
    raise SystemExit(0)

# ── Save combined JSON ────────────────────────────────────────────────────────
combined_path = OUT_DIR / 'ze_dortmund_combined.json'
combined_path.write_text(json.dumps(results, indent=2))
print(f"\n💾 Combined results: {combined_path}  (N={len(results)})")

# ── Group statistics ──────────────────────────────────────────────────────────
y_list = [r for r in results if r['group'] == 'young']
o_list = [r for r in results if r['group'] == 'old']
print(f"\n{'═'*70}")
print(f"N: young={len(y_list)}, old={len(o_list)}")

chi_y_proxy = np.array([r['chi_Ze_peak'] for r in y_list])
chi_o_proxy = np.array([r['chi_Ze_peak'] for r in o_list])
chi_y_nb    = np.array([r['chi_Ze_nb']   for r in y_list])
chi_o_nb    = np.array([r['chi_Ze_nb']   for r in o_list])
sex_y       = np.array([r['sex']         for r in y_list])
sex_o       = np.array([r['sex']         for r in o_list])

for label, cy, co in [('Proxy (α-peak)', chi_y_proxy, chi_o_proxy),
                       ('Narrowband Ze',  chi_y_nb,    chi_o_nb)]:
    print(f"\n── {label} ──")
    st = group_statistics(cy, co, sex_young=sex_y, sex_old=sex_o)
    print(f"  Young: {st['mean_young']:.4f}±{st['sd_young']:.4f}  "
          f"Old: {st['mean_old']:.4f}±{st['sd_old']:.4f}")
    print(f"  t={st['t']:.3f}  p={st['p']:.4f}  "
          f"d={st['cohens_d']:.3f} [{st['d_ci_95'][0]:.3f}, {st['d_ci_95'][1]:.3f}]  "
          f"power={st['power']:.0%}")
    print(f"  AUC={st['auc']:.3f} [{st['auc_ci_95'][0]:.3f}, "
          f"{st['auc_ci_95'][1]:.3f}]  p_MW={st['auc_p_onesided']:.4f}")
    if 'ancova' in st:
        a = st['ancova']
        print(f"  ANCOVA: {a['df']}={a['F_group']:.3f} p={a['p_group']:.4f}  "
              f"β_group={a['beta_group']:.4f}  R²={a['r2']:.3f}")
        print(f"  Group×Sex: {a['df_interaction']}={a['F_interaction']:.3f} "
              f"p={a['p_interaction']:.4f}")

# Band-wise
print(f"\nBand-wise χ_Ze (narrowband, Young vs Old):")
for bname in BANDS:
    yv = np.array([r['bands'][bname] for r in y_list if bname in r.get('bands', {})])
    ov = np.array([r['bands'][bname] for r in o_list if bname in r.get('bands', {})])
    if len(yv) > 1 and len(ov) > 1:
        st = group_statistics(yv, ov)
        sig = '✅' if st['p'] < 0.05 else '⚠️ ns'
        print(f"  {bname:6s}: Y={st['mean_young']:.4f} O={st['mean_old']:.4f}  "
              f"d={st['cohens_d']:.3f}  p={st['p']:.4f} {sig}")
print(f"{'═'*70}")
