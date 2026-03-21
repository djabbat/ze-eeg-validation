#!/usr/bin/env python3
"""
Batch download + Ze band-wise analysis pipeline for MPI-LEMON.
Downloads, processes (EC only), saves JSON result, removes archive.
"""
import os, sys, json, tarfile, tempfile, subprocess, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import compute_ze_metrics, V_STAR

import mne
mne.set_log_level('WARNING')

BASE_URL = ("https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/MPI-LEMON/"
            "Compressed_tar/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed")
LEMON_DIR = Path('/home/oem/Desktop/EEG/data/lemon')
OUT_DIR   = LEMON_DIR / 'results'
OUT_DIR.mkdir(exist_ok=True)

RESAMPLE_HZ = 128.0
CROP_S = 240

BANDS = {
    'delta': (1,   4),
    'theta': (4,   8),
    'alpha': (8,  12),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

def load_meta():
    import csv
    meta = {}
    with open(LEMON_DIR / 'participants.csv') as f:
        for row in csv.DictReader(f):
            sid = row['ID'].strip()
            age_bin = row['Age'].strip()
            lo = int(age_bin.split('-')[0]) if '-' in age_bin else 0
            meta[sid] = {'age_bin': age_bin, 'age': lo+2,
                         'group': 'young' if lo <= 35 else ('old' if lo >= 60 else 'middle')}
    return meta

def bandpass_ze(raw, fmin, fmax):
    raw_f = raw.copy().filter(fmin, fmax, method='iir', verbose=False)
    picks = mne.pick_types(raw_f.info, eeg=True, exclude='bads')
    if not len(picks): picks = list(range(raw_f.info['nchan']))
    chi_l, v_l = [], []
    for idx in picks:
        m = compute_ze_metrics(raw_f.get_data(picks=[idx])[0])
        chi_l.append(m['chi_Ze']); v_l.append(m['v'])
    return {'chi_Ze_mean': float(np.mean(chi_l)),
            'chi_Ze_std':  float(np.std(chi_l)),
            'v_mean':      float(np.mean(v_l))}

def process_subject(sub_id, meta):
    out_file = OUT_DIR / f'ze_{sub_id}.json'
    if out_file.exists():
        print(f"  ⏭  {sub_id}: already done")
        return json.loads(out_file.read_text())

    # Download
    tar_path = LEMON_DIR / f'{sub_id}.tar.gz'
    if not tar_path.exists() or tar_path.stat().st_size < 1_000_000:
        url = f"{BASE_URL}/{sub_id}.tar.gz"
        print(f"  ⬇  {sub_id}: downloading...", end=' ', flush=True)
        ret = subprocess.run(['curl', '-s', '-L', url, '-o', str(tar_path)],
                             capture_output=True)
        if ret.returncode != 0 or tar_path.stat().st_size < 1_000_000:
            print("FAILED")
            tar_path.unlink(missing_ok=True)
            return None
        print(f"{tar_path.stat().st_size//1024//1024}MB")

    # Extract EC only
    sub_dir = LEMON_DIR / sub_id
    sub_dir.mkdir(exist_ok=True)
    ec_path = sub_dir / f'{sub_id}_EC.set'
    fdt_path = sub_dir / f'{sub_id}_EC.fdt'

    if not ec_path.exists():
        print(f"  📦 {sub_id}: extracting EC...", end=' ', flush=True)
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if '_EC.' in member.name:
                    member.name = Path(member.name).name
                    tf.extract(member, path=sub_dir)
        print("done")

    # Remove archive to save space
    tar_path.unlink(missing_ok=True)

    if not ec_path.exists():
        print(f"  ❌ {sub_id}: EC.set not found after extraction")
        return None

    # Load & resample
    print(f"  🔬 {sub_id}: Ze analysis...", end=' ', flush=True)
    try:
        raw = mne.io.read_raw_eeglab(str(ec_path), preload=True, verbose=False)
        raw.resample(RESAMPLE_HZ, npad='auto')
        raw.crop(tmax=min(CROP_S, raw.times[-1]))
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    bands = {}
    for bname, (fmin, fmax) in BANDS.items():
        bands[bname] = bandpass_ze(raw, fmin, fmax)

    m = meta.get(sub_id, {})
    result = {
        'subject_id': sub_id,
        'age_bin': m.get('age_bin', 'unknown'),
        'age': m.get('age'),
        'group': m.get('group', 'unknown'),
        'sfreq': raw.info['sfreq'],
        'n_channels': raw.info['nchan'],
        'duration_s': round(raw.times[-1], 1),
        'bands': bands,
    }
    out_file.write_text(json.dumps(result, indent=2))
    chi_a = bands['alpha']['chi_Ze_mean']
    chi_g = bands['gamma']['chi_Ze_mean']
    print(f"α={chi_a:.3f} γ={chi_g:.3f} ({m.get('age_bin','?')},{m.get('group','?')})")
    return result

def main():
    meta = load_meta()

    # Load batch list
    batch = []
    with open('/tmp/lemon_batch.txt') as f:
        for line in f:
            sid, age = line.strip().split(',')
            batch.append(sid)

    # Also include already-done subjects
    done_files = sorted(OUT_DIR.glob('ze_sub-*.json'))
    done_subs = [f.stem.replace('ze_','') for f in done_files
                 if f.stem != 'ze_lemon_age_scatter']
    print(f"Already done: {len(done_subs)} subjects")
    print(f"To process: {len(batch)} new subjects\n")

    results = []
    # Load existing
    for sub_id in done_subs:
        f = OUT_DIR / f'ze_{sub_id}.json'
        if f.exists():
            results.append(json.loads(f.read_text()))

    # Process new
    for sub_id in batch:
        r = process_subject(sub_id, meta)
        if r:
            results.append(r)

    # Aggregate
    print(f"\n{'═'*65}")
    print("FINAL SUMMARY BY GROUP AND BAND")
    print(f"{'─'*65}")
    print(f"{'Group':8s}  {'N':3s}  {'delta':7s}  {'theta':7s}  {'alpha':7s}  {'beta':7s}  {'gamma':7s}")
    print(f"{'─'*65}")

    for group in ('young', 'old'):
        gr = [r for r in results if r.get('group') == group]
        if not gr: continue
        row = f"  {group:6s}  {len(gr):3d}  "
        for bname in BANDS:
            vals = [r['bands'][bname]['chi_Ze_mean'] for r in gr
                    if bname in r.get('bands', {})]
            row += f"{np.mean(vals):.3f}    "
        print(row)

    # Delta table (young - old)
    print(f"{'─'*65}")
    row = f"  {'Δ Y-O':6s}       "
    for bname in BANDS:
        y = [r['bands'][bname]['chi_Ze_mean'] for r in results
             if r.get('group')=='young' and bname in r.get('bands',{})]
        o = [r['bands'][bname]['chi_Ze_mean'] for r in results
             if r.get('group')=='old'   and bname in r.get('bands',{})]
        delta = np.mean(y) - np.mean(o) if y and o else 0
        sign = '✅' if delta > 0.01 else ('⚠️' if delta < -0.01 else '≈ ')
        row += f"{delta:+.3f}{sign} "
    print(row)
    print(f"{'═'*65}")

    # Save combined
    combined_path = OUT_DIR / 'ze_lemon_combined.json'
    combined_path.write_text(json.dumps(results, indent=2))
    print(f"\n  💾 {combined_path}  ({len(results)} subjects)")

    # Plot per-band scatter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(BANDS), figsize=(16, 5), sharey=False)
    fig.suptitle(f'Ze χ_Ze by Band — MPI-LEMON (N={len(results)})\n'
                 'Young (20-30) vs Old (65-75), EC condition', fontsize=11)

    for ax, bname in zip(axes, BANDS):
        fmin, fmax = BANDS[bname]
        y_vals = [r['bands'][bname]['chi_Ze_mean'] for r in results
                  if r.get('group')=='young' and bname in r.get('bands',{})]
        o_vals = [r['bands'][bname]['chi_Ze_mean'] for r in results
                  if r.get('group')=='old'   and bname in r.get('bands',{})]

        np.random.seed(0)
        ax.scatter(np.ones(len(y_vals))*1 + np.random.randn(len(y_vals))*0.07,
                   y_vals, color='#2E74B5', s=40, alpha=0.7)
        ax.scatter(np.ones(len(o_vals))*2 + np.random.randn(len(o_vals))*0.07,
                   o_vals, color='#C00000', s=40, alpha=0.7)
        ax.errorbar([1], [np.mean(y_vals)], yerr=[np.std(y_vals)/len(y_vals)**0.5],
                    color='#1a4a80', fmt='D', ms=9, capsize=6, lw=2, zorder=5)
        ax.errorbar([2], [np.mean(o_vals)], yerr=[np.std(o_vals)/len(o_vals)**0.5],
                    color='#800000', fmt='D', ms=9, capsize=6, lw=2, zorder=5)
        delta = np.mean(y_vals) - np.mean(o_vals) if y_vals and o_vals else 0
        sign = '↑Y' if delta > 0.005 else ('↑O' if delta < -0.005 else '≈')
        ax.set_title(f'{bname}\n{fmin}-{fmax}Hz\nΔ={delta:+.3f} {sign}', fontsize=8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Young', 'Old'], fontsize=8)
        ax.set_xlim(0.5, 2.5)
        if ax is axes[0]: ax.set_ylabel('χ_Ze mean')

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#2E74B5', label='Young (20-30)'),
                        Patch(color='#C00000', label='Old (65-75)')],
               loc='lower right', fontsize=9)
    plt.tight_layout()
    ppath = OUT_DIR / 'ze_lemon_bandwise_final.png'
    plt.savefig(ppath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 {ppath}")

if __name__ == '__main__':
    main()
