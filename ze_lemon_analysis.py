#!/usr/bin/env python3
"""
Ze Analysis on MPI-LEMON EEG Dataset
=====================================
Tests Ze aging hypothesis: χ_Ze(young) > χ_Ze(old)

Dataset: MPI-Leipzig Mind-Brain-Body Dataset (LEMON)
- Babayan et al. (2019) Scientific Data 6:308
- 228 subjects, ages 20–77, resting-state EEG 16 min (EC/EO alternating)
- Preprocessed (ICA-cleaned), EEGLAB .set/.fdt format
- 62 channels, 250Hz (preprocessed)

Usage:
    python3 ze_lemon_analysis.py --data data/lemon/ --out data/lemon/results/
    python3 ze_lemon_analysis.py --data data/lemon/ --subjects sub-032301 sub-032302
"""

import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from eeg_ze_processor import (
    compute_ze_metrics, print_summary, plot_ze_channels, plot_group_comparison,
    V_STAR, CHI_MAX_LIVING
)

import mne
mne.set_log_level('WARNING')


# ─── Metadata ────────────────────────────────────────────────────────────────

def load_participants(csv_path: str) -> dict:
    """Load LEMON participants.csv → {sub_id: {age_bin, gender}}"""
    meta = {}
    with open(csv_path) as f:
        lines = f.read().strip().split('\n')
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) < 3:
            continue
        sid, gender, age_bin = parts[0].strip(), parts[1].strip(), parts[2].strip()
        # Parse age: use midpoint of bin
        if '-' in age_bin:
            lo, hi = age_bin.split('-')
            age_mid = (int(lo) + int(hi)) // 2
        else:
            age_mid = None
        meta[sid] = {
            'age_bin': age_bin,
            'age': age_mid,
            'gender': 'F' if gender == '1' else 'M',
        }
    return meta


def age_group(age_bin: str) -> str:
    """Classify age group."""
    if not age_bin or age_bin == 'n/a':
        return 'unknown'
    lo = int(age_bin.split('-')[0])
    if lo <= 35:
        return 'young'
    elif lo >= 60:
        return 'old'
    return 'middle'


# ─── EEGLAB loader ───────────────────────────────────────────────────────────

def load_eeglab_set(set_path: str, resample_hz: float = 128.0) -> mne.io.BaseRaw:
    """Load EEGLAB .set file (with companion .fdt)."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    if resample_hz and raw.info['sfreq'] > resample_hz:
        print(f"    ⟳ {raw.info['sfreq']:.0f} Hz → {resample_hz:.0f} Hz")
        raw.resample(resample_hz, npad='auto')
    return raw


# ─── Per-subject analysis ────────────────────────────────────────────────────

def analyze_subject(sub_dir: Path, meta: dict, resample_hz: float = 128.0) -> list:
    """
    Analyze all EC/EO .set files for a subject.
    Returns list of result dicts (one per condition file).
    """
    sub_id = sub_dir.name
    sub_meta = meta.get(sub_id, {})
    age = sub_meta.get('age')
    age_bin = sub_meta.get('age_bin', 'unknown')
    group = age_group(age_bin)

    # Find all .set files in subject folder
    set_files = sorted(sub_dir.glob('*.set'))
    if not set_files:
        # Also check nested eeg/ folder
        set_files = sorted(sub_dir.rglob('*.set'))

    if not set_files:
        print(f"  ⚠️  No .set files found in {sub_dir}")
        return []

    print(f"\n  Subject: {sub_id}  age={age_bin} ({group})  files={len(set_files)}")
    results = []

    for set_path in set_files:
        condition = 'EC' if '_EC' in set_path.name or 'EC' in set_path.stem else \
                    'EO' if '_EO' in set_path.name or 'EO' in set_path.stem else 'RS'
        print(f"    Loading: {set_path.name} ({condition})")

        try:
            raw = load_eeglab_set(str(set_path), resample_hz)
        except Exception as e:
            print(f"    ❌ Error loading: {e}")
            continue

        # Ze analysis
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(picks) == 0:
            picks = list(range(raw.info['nchan']))

        chi_list, v_list = [], []
        for idx in picks:
            signal = raw.get_data(picks=[idx])[0]
            m = compute_ze_metrics(signal)
            chi_list.append(m['chi_Ze'])
            v_list.append(m['v'])

        result = {
            'subject_id': sub_id,
            'age': age,
            'age_bin': age_bin,
            'group': group,
            'condition': condition,
            'file': set_path.name,
            'sfreq': raw.info['sfreq'],
            'n_channels': len(picks),
            'duration_s': round(raw.times[-1], 2),
            'summary': {
                'chi_Ze_mean': round(float(np.mean(chi_list)), 6),
                'chi_Ze_std':  round(float(np.std(chi_list)),  6),
                'v_mean':      round(float(np.mean(v_list)),   6),
            },
            'timestamp': datetime.now().isoformat(),
        }

        chi = result['summary']['chi_Ze_mean']
        v   = result['summary']['v_mean']
        print(f"    χ_Ze={chi:.4f}  v={v:.4f}  ({raw.times[-1]:.0f}s, {len(picks)}ch)")
        results.append(result)

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ze Analysis on MPI-LEMON EEG")
    parser.add_argument('--data',     default='data/lemon',
                        help="Path to lemon data folder (contains sub-XXXXXX/ dirs)")
    parser.add_argument('--meta',     default=None,
                        help="Path to participants.csv (default: data/participants.csv)")
    parser.add_argument('--subjects', nargs='+', default=None,
                        help="Specific subject IDs to process (default: all found)")
    parser.add_argument('--resample', type=float, default=128.0,
                        help="Resample to Hz before Ze analysis (default: 128)")
    parser.add_argument('--out',      default='data/lemon/results',
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data_dir = Path(args.data)

    # Load metadata
    meta_path = args.meta or str(data_dir / 'participants.csv')
    meta = {}
    if os.path.exists(meta_path):
        meta = load_participants(meta_path)
        print(f"Loaded metadata: {len(meta)} subjects")
    else:
        print(f"⚠️  No participants.csv found at {meta_path}")

    # Find subject directories
    if args.subjects:
        sub_dirs = [data_dir / s for s in args.subjects if (data_dir / s).exists()]
    else:
        sub_dirs = sorted([d for d in data_dir.iterdir()
                           if d.is_dir() and d.name.startswith('sub-')])

    print(f"Processing {len(sub_dirs)} subject(s)...\n")

    all_results = []
    for sub_dir in sub_dirs:
        results = analyze_subject(sub_dir, meta, args.resample)
        all_results.extend(results)

    if not all_results:
        print("No results — check that .set files exist in subject folders.")
        return

    # Save all results
    out_json = os.path.join(args.out, 'ze_lemon_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 All results: {out_json}")

    # Summary by group
    print(f"\n{'═'*55}")
    print("  SUMMARY BY GROUP")
    print(f"{'─'*55}")

    for group in ('young', 'old', 'middle'):
        group_res = [r for r in all_results if r['group'] == group]
        if not group_res:
            continue
        chis = [r['summary']['chi_Ze_mean'] for r in group_res]
        vs   = [r['summary']['v_mean']      for r in group_res]
        age_range = f"{min(r['age_bin'] for r in group_res if r['age_bin'])} — " \
                    f"{max(r['age_bin'] for r in group_res if r['age_bin'])}"
        print(f"  {group.upper():8s}  N={len(group_res):3d}  "
              f"χ_Ze={np.mean(chis):.4f}±{np.std(chis):.4f}  "
              f"v={np.mean(vs):.4f}  ages: {age_range}")

    # Ze hypothesis check
    young_chis = [r['summary']['chi_Ze_mean'] for r in all_results if r['group']=='young']
    old_chis   = [r['summary']['chi_Ze_mean'] for r in all_results if r['group']=='old']

    if young_chis and old_chis:
        delta = np.mean(young_chis) - np.mean(old_chis)
        print(f"\n  Δ χ_Ze (young − old): {delta:+.4f}")
        if delta > 0:
            print("  ✅ Ze aging hypothesis SUPPORTED: χ_Ze(young) > χ_Ze(old)")
        else:
            print("  ⚠️  Ze aging hypothesis NOT confirmed on this sample")

    # Group scatter plot
    if len(all_results) >= 2:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {'young': '#2E74B5', 'old': '#C00000', 'middle': '#7B9FC4', 'unknown': '#999'}
        for r in all_results:
            if r['age'] is not None:
                ax.scatter(r['age'], r['summary']['chi_Ze_mean'],
                           color=colors.get(r['group'], '#999'), s=60, alpha=0.7,
                           label=r['group'])

        # Linear trend
        pts = [(r['age'], r['summary']['chi_Ze_mean'])
               for r in all_results if r['age'] is not None]
        if len(pts) >= 3:
            ages_arr = np.array([p[0] for p in pts])
            chis_arr = np.array([p[1] for p in pts])
            z = np.polyfit(ages_arr, chis_arr, 1)
            p = np.poly1d(z)
            x_line = np.linspace(ages_arr.min(), ages_arr.max(), 100)
            ax.plot(x_line, p(x_line), 'k--', linewidth=2,
                    label=f'trend Δχ/year={z[0]:.5f}')
            corr = np.corrcoef(ages_arr, chis_arr)[0, 1]
            ax.set_title(f'Ze Cheating Index vs Age — MPI-LEMON\n'
                         f'r = {corr:.3f}  |  Ze hypothesis: χ_Ze ↓ with age',
                         fontsize=11)

        ax.set_xlabel("Age (years, midpoint of bin)")
        ax.set_ylabel("χ_Ze mean (across channels)")
        ax.axhline(CHI_MAX_LIVING, color='green', linestyle=':', linewidth=1,
                   label=f'χ_max_living={CHI_MAX_LIVING}')
        ax.set_ylim(0, 1)
        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen = set(); uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); uniq_h.append(h); uniq_l.append(l)
        ax.legend(uniq_h, uniq_l, fontsize=9)
        plt.tight_layout()
        plot_path = os.path.join(args.out, 'ze_lemon_age_scatter.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Plot: {plot_path}")

    print(f"{'═'*55}")


if __name__ == '__main__':
    main()
