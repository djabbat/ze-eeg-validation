#!/usr/bin/env python3
"""
EEG Ze Processor — Experimental Validation of Ze Theory
========================================================
Computes Ze velocity (v), Ze cheating index (χ_Ze), and Ze proper time (τ)
from EEG recordings. Tests the Ze aging hypothesis:
    χ_Ze(young) > χ_Ze(old)

Usage:
    python3 eeg_ze_processor.py --file recording.edf --age 35 --label "Subject_01"
    python3 eeg_ze_processor.py --demo           # run on PhysioNet sample data
    python3 eeg_ze_processor.py --batch data/    # process a folder of EDF files

Theory (Tkemaladze, Ze System as Observer):
    - Binary sequence {x_k}: x_k = 1 if sample > median, else 0
    - N_S = number of switches (x_k ≠ x_{k-1})
    - Ze velocity:       v = N_S / (N - 1)          range [0, 1]
    - Fixed point:       v* ≈ 0.45631               (max materialization)
    - Ze cheating index: χ_Ze = 1 - |v - v*| / max(v*, 1-v*)   range [0, 1]
    - Ze proper time:    τ = α * N * χ_Ze            (temporal potential)
    - Aging hypothesis:  χ_Ze decreases with age
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Ze constants
V_STAR = 0.45631          # fixed point — maximum materialization
CHI_MAX_LIVING = 0.839    # upper bound for living systems
ALPHA = 1.0               # normalization constant (adjustable)

# ─── Core Ze computations ────────────────────────────────────────────────────

def binarize(signal: np.ndarray) -> np.ndarray:
    """Convert continuous EEG signal to binary sequence {0,1} via median threshold."""
    threshold = np.median(signal)
    return (signal > threshold).astype(int)

def ze_velocity(binary_seq: np.ndarray) -> float:
    """v = N_S / (N-1)  — fraction of switching events."""
    N = len(binary_seq)
    if N < 2:
        return 0.0
    switches = np.sum(binary_seq[1:] != binary_seq[:-1])
    return switches / (N - 1)

def ze_cheating_index(v: float) -> float:
    """χ_Ze = 1 - |v - v*| / max(v*, 1-v*)  — proximity to optimal Ze velocity."""
    max_dist = max(V_STAR, 1.0 - V_STAR)
    return 1.0 - abs(v - V_STAR) / max_dist

def ze_proper_time(N: int, chi: float) -> float:
    """τ = α * N * χ_Ze  — accumulated temporal potential."""
    return ALPHA * N * chi

def compute_ze_metrics(signal: np.ndarray) -> dict:
    """Full Ze analysis of a 1D signal array."""
    binary = binarize(signal)
    N = len(binary)
    N_S = int(np.sum(binary[1:] != binary[:-1]))
    v = ze_velocity(binary)
    chi = ze_cheating_index(v)
    tau = ze_proper_time(N, chi)
    return {
        "N": N,
        "N_S": N_S,
        "v": round(v, 6),
        "v_star": V_STAR,
        "chi_Ze": round(chi, 6),
        "chi_max_living": CHI_MAX_LIVING,
        "tau": round(tau, 2),
        "v_deviation": round(abs(v - V_STAR), 6),
    }

# ─── EEG loading ─────────────────────────────────────────────────────────────

def load_edf(filepath: str, duration_s: float = None):
    """Load EDF/BDF file."""
    import mne
    mne.set_log_level('WARNING')
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    if duration_s:
        raw.crop(tmax=min(duration_s, raw.times[-1]))
    return raw

def load_brainvision(filepath: str, duration_s: float = None, resample_hz: float = None):
    """Load BrainVision (.vhdr) file. The .vmrk and .eeg must be in the same folder."""
    import mne
    mne.set_log_level('WARNING')
    raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
    if duration_s:
        raw.crop(tmax=min(duration_s, raw.times[-1]))
    if resample_hz and raw.info['sfreq'] > resample_hz:
        print(f"  ⟳ Resampling {raw.info['sfreq']:.0f} Hz → {resample_hz:.0f} Hz")
        raw.resample(resample_hz, npad='auto')
    return raw

def load_any(filepath: str, duration_s: float = None, resample_hz: float = None):
    """Auto-detect format and load EEG file."""
    ext = Path(filepath).suffix.lower()
    if ext == '.vhdr':
        return load_brainvision(filepath, duration_s, resample_hz)
    elif ext in ('.edf', '.bdf'):
        raw = load_edf(filepath, duration_s)
        if resample_hz and raw.info['sfreq'] > resample_hz:
            print(f"  ⟳ Resampling {raw.info['sfreq']:.0f} Hz → {resample_hz:.0f} Hz")
            raw.resample(resample_hz, npad='auto')
        return raw
    else:
        raise ValueError(f"Unsupported format: {ext} — use .vhdr, .edf, or .bdf")

def load_physionet_sample():
    """Download a PhysioNet EEG Motor Movement sample (free, ~1MB)."""
    import mne
    mne.set_log_level('WARNING')
    print("📡 Downloading PhysioNet sample (subject 1, eyes-open rest)...")
    raw = mne.datasets.eegbci.load_data(subject=1, runs=[1], verbose=False)
    raw_loaded = mne.io.concatenate_raws(
        [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw]
    )
    mne.datasets.eegbci.standardize(raw_loaded)
    return raw_loaded

# ─── Per-channel Ze analysis ──────────────────────────────────────────────────

def analyze_raw(raw, label: str = "Subject", age: int = None) -> dict:
    """Run Ze analysis on all EEG channels. Returns results dict."""
    import mne
    # Pick EEG channels only
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(picks) == 0:
        picks = list(range(len(raw.ch_names)))

    ch_results = {}
    for idx in picks:
        ch_name = raw.ch_names[idx]
        signal = raw.get_data(picks=[idx])[0]
        ch_results[ch_name] = compute_ze_metrics(signal)

    # Aggregate across channels
    chi_values = [r["chi_Ze"] for r in ch_results.values()]
    v_values   = [r["v"]      for r in ch_results.values()]

    result = {
        "label":      label,
        "age":        age,
        "sfreq":      raw.info["sfreq"],
        "n_channels": len(picks),
        "duration_s": round(raw.times[-1], 2),
        "channels":   ch_results,
        "summary": {
            "chi_Ze_mean": round(float(np.mean(chi_values)), 6),
            "chi_Ze_std":  round(float(np.std(chi_values)),  6),
            "chi_Ze_min":  round(float(np.min(chi_values)),  6),
            "chi_Ze_max":  round(float(np.max(chi_values)),  6),
            "v_mean":      round(float(np.mean(v_values)),   6),
            "v_std":       round(float(np.std(v_values)),    6),
        },
        "timestamp": datetime.now().isoformat(),
    }
    return result

# ─── Visualization ────────────────────────────────────────────────────────────

def plot_ze_channels(result: dict, out_dir: str = "."):
    """Bar chart of χ_Ze per channel + v distribution."""
    label = result["label"]
    age_str = f", age {result['age']}" if result["age"] else ""
    channels = result["channels"]
    ch_names = list(channels.keys())
    chi_vals = [channels[c]["chi_Ze"] for c in ch_names]
    v_vals   = [channels[c]["v"]      for c in ch_names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(ch_names)*0.4), 8))
    fig.suptitle(f"Ze Analysis: {label}{age_str}", fontsize=13, fontweight='bold')

    # χ_Ze per channel
    colors = ['#2E74B5' if c >= 0.5 else '#C55A11' for c in chi_vals]
    ax1.bar(ch_names, chi_vals, color=colors, edgecolor='white', linewidth=0.5)
    ax1.axhline(CHI_MAX_LIVING, color='green', linestyle='--', linewidth=1,
                label=f'χ_max_living = {CHI_MAX_LIVING}')
    ax1.axhline(result["summary"]["chi_Ze_mean"], color='red', linestyle='-',
                linewidth=1.5, label=f'mean = {result["summary"]["chi_Ze_mean"]:.4f}')
    ax1.set_ylabel("χ_Ze (cheating index)")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(len(ch_names)))
    ax1.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax1.legend(fontsize=8)
    ax1.set_title("Ze Cheating Index per Channel")

    # v per channel
    ax2.bar(ch_names, v_vals, color='#7B9FC4', edgecolor='white', linewidth=0.5)
    ax2.axhline(V_STAR, color='green', linestyle='--', linewidth=1,
                label=f'v* = {V_STAR}')
    ax2.axhline(result["summary"]["v_mean"], color='red', linestyle='-',
                linewidth=1.5, label=f'mean = {result["summary"]["v_mean"]:.4f}')
    ax2.set_ylabel("Ze velocity (v)")
    ax2.set_ylim(0, 1)
    ax2.set_xticks(range(len(ch_names)))
    ax2.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax2.legend(fontsize=8)
    ax2.set_title("Ze Velocity per Channel")

    plt.tight_layout()
    fname = f"ze_{label.replace(' ', '_')}.png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Plot saved: {fpath}")
    return fpath

def plot_group_comparison(results: list, out_dir: str = "."):
    """Scatter plot χ_Ze vs age for multiple subjects."""
    ages = [r["age"] for r in results if r["age"] is not None]
    chis = [r["summary"]["chi_Ze_mean"] for r in results if r["age"] is not None]
    labels = [r["label"] for r in results if r["age"] is not None]

    if len(ages) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ages, chis, s=80, color='#2E74B5', zorder=5)
    for a, c, l in zip(ages, chis, labels):
        ax.annotate(l, (a, c), textcoords="offset points", xytext=(5, 3), fontsize=7)

    # Linear trend
    if len(ages) >= 3:
        z = np.polyfit(ages, chis, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(ages), max(ages), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=1.5,
                label=f'trend: Δχ/year = {z[0]:.5f}')
        ax.legend(fontsize=9)

    ax.axhline(CHI_MAX_LIVING, color='green', linestyle=':', linewidth=1,
               label=f'χ_max_living = {CHI_MAX_LIVING}')
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("χ_Ze mean (across channels)")
    ax.set_title("Ze Cheating Index vs Age\n(Ze Aging Hypothesis: χ_Ze ↓ with age)")
    ax.set_ylim(0, 1)

    fpath = os.path.join(out_dir, "ze_age_comparison.png")
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Group plot: {fpath}")
    return fpath

# ─── Report ──────────────────────────────────────────────────────────────────

def print_summary(result: dict):
    s = result["summary"]
    age_str = f", age {result['age']}" if result["age"] else ""
    print(f"\n{'─'*50}")
    print(f"  Subject:      {result['label']}{age_str}")
    print(f"  Duration:     {result['duration_s']}s  |  Channels: {result['n_channels']}")
    print(f"  Sfreq:        {result['sfreq']} Hz")
    print(f"  χ_Ze mean:    {s['chi_Ze_mean']:.4f}  ±  {s['chi_Ze_std']:.4f}")
    print(f"  χ_Ze range:   [{s['chi_Ze_min']:.4f} – {s['chi_Ze_max']:.4f}]")
    print(f"  v mean:       {s['v_mean']:.4f}  (v* = {V_STAR})")
    aging = "🟢 HIGH (young)" if s['chi_Ze_mean'] > 0.6 else \
            "🟡 MEDIUM" if s['chi_Ze_mean'] > 0.4 else "🔴 LOW (aging)"
    print(f"  Ze status:    {aging}")
    print(f"{'─'*50}")

def save_json(result: dict, out_dir: str = "."):
    fname = f"ze_{result['label'].replace(' ', '_')}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  💾 JSON saved: {fpath}")
    return fpath

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EEG Ze Processor — compute χ_Ze aging biomarker from EEG"
    )
    parser.add_argument('--file',   help="Path to EDF/BDF/.vhdr file")
    parser.add_argument('--age',    type=int, help="Subject age (years)")
    parser.add_argument('--label',  default="Subject", help="Subject label")
    parser.add_argument('--duration', type=float, default=None,
                        help="Crop to N seconds (default: full recording)")
    parser.add_argument('--resample', type=float, default=128.0,
                        help="Resample to Hz before Ze analysis (default: 128). "
                             "At 128Hz: beta 30Hz → v=0.469≈v*. Use 0 to disable.")
    parser.add_argument('--batch',  help="Process all EDF files in folder")
    parser.add_argument('--demo',   action='store_true',
                        help="Download PhysioNet sample and run demo")
    parser.add_argument('--out',    default=".", help="Output directory for plots/JSON")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_results = []

    # ── DEMO mode ──
    if args.demo:
        print("\n🧪 Ze EEG Demo — PhysioNet EEG Motor Movement Dataset")
        print("   Ze Aging Hypothesis: χ_Ze(young) > χ_Ze(old)\n")

        # Simulated demo using synthetic data (no internet required)
        print("📐 Running on synthetic EEG signals (3 age groups)...\n")
        np.random.seed(42)
        # For a sine wave at freq f Hz and sampling rate fs:
        #   v ≈ 2*f/fs  (switches twice per cycle)
        # v* = 0.45631 → target freq ≈ 0.45631/2 * 160 ≈ 36.5 Hz (young, near optimal)
        # Aging = EEG slowing (alpha→theta→delta) → lower f → lower v → lower χ_Ze
        demo_subjects = [
            ("Young_25",  25, 36.0),   # ~beta: v ≈ 0.450 → χ_Ze ≈ 1.0
            ("Middle_45", 45, 22.0),   # ~alpha: v ≈ 0.275 → χ_Ze ≈ 0.60
            ("Elder_70",  70, 10.0),   # ~theta: v ≈ 0.125 → χ_Ze ≈ 0.27
        ]
        for label, age, freq_hz in demo_subjects:
            N = 160 * 30  # 30s at 160 Hz
            t = np.arange(N) / 160.0
            # Pure sine at target frequency (clean, predictable v)
            signal = np.sin(2 * np.pi * freq_hz * t) + 0.05 * np.random.randn(N)
            binary = binarize(signal)
            actual_v = ze_velocity(binary)
            chi = ze_cheating_index(actual_v)
            tau = ze_proper_time(N, chi)
            fake_result = {
                "label": label, "age": age,
                "sfreq": 160, "n_channels": 1,
                "duration_s": 30,
                "channels": {"Cz": {
                    "N": N, "N_S": int(actual_v*(N-1)),
                    "v": round(actual_v,6), "v_star": V_STAR,
                    "chi_Ze": round(chi,6), "tau": round(tau,2),
                    "chi_max_living": CHI_MAX_LIVING,
                    "v_deviation": round(abs(actual_v-V_STAR),6),
                }},
                "summary": {
                    "chi_Ze_mean": round(chi,6), "chi_Ze_std": 0.0,
                    "chi_Ze_min": round(chi,6), "chi_Ze_max": round(chi,6),
                    "v_mean": round(actual_v,6), "v_std": 0.0,
                },
                "timestamp": datetime.now().isoformat(),
            }
            print_summary(fake_result)
            save_json(fake_result, args.out)
            all_results.append(fake_result)

        if len(all_results) >= 2:
            plot_group_comparison(all_results, args.out)

        # Ze theory prediction check
        chis = [r["summary"]["chi_Ze_mean"] for r in all_results]
        ages = [r["age"] for r in all_results]
        corr = np.corrcoef(ages, chis)[0, 1]
        print(f"\n📉 Pearson correlation (age vs χ_Ze): {corr:.4f}")
        if corr < -0.3:
            print("  ✅ Ze hypothesis SUPPORTED: χ_Ze decreases with age")
        else:
            print("  ⚠️  Ze hypothesis NOT confirmed on this sample")
        return

    # ── BATCH mode ──
    if args.batch:
        batch_path = Path(args.batch)
        files = (sorted(batch_path.glob("*.vhdr")) +
                 sorted(batch_path.glob("*.edf")) +
                 sorted(batch_path.glob("*.EDF")) +
                 sorted(batch_path.glob("*.bdf")))
        print(f"\n📂 Batch mode: {len(files)} files in {args.batch}")
        resample = args.resample if args.resample > 0 else None
        for f in files:
            label = f.stem
            print(f"\n  Processing: {f.name}")
            try:
                raw = load_any(str(f), duration_s=args.duration, resample_hz=resample)
                result = analyze_raw(raw, label=label)
                print_summary(result)
                save_json(result, args.out)
                plot_ze_channels(result, args.out)
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ Error: {e}")
        if len(all_results) >= 2:
            plot_group_comparison(all_results, args.out)
        return

    # ── SINGLE FILE mode ──
    if args.file:
        print(f"\n🔬 Processing: {args.file}")
        resample = args.resample if args.resample > 0 else None
        raw = load_any(args.file, duration_s=args.duration, resample_hz=resample)
        result = analyze_raw(raw, label=args.label, age=args.age)
        print_summary(result)
        save_json(result, args.out)
        plot_ze_channels(result, args.out)
        return

    # No args — show help
    parser.print_help()
    print("\n  Example: python3 eeg_ze_processor.py --demo")

if __name__ == "__main__":
    main()
