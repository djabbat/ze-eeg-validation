#!/usr/bin/env python3
"""
EEG Ze Processor — Experimental Validation of Ze Theory
========================================================
Computes Ze velocity (v), Ze cheating index (χ_Ze), and Ze proper time (τ)
from EEG recordings. Tests the Ze aging hypothesis:
    χ_Ze(young) > χ_Ze(old)

Two validated methods (see Tkemaladze 2026, Sec. 3.3):
  • Proxy method:       α-peak frequency → v_peak = 2·f_peak/f_s → χ_Ze
  • Narrowband Ze:      bandpass 8–12 Hz → median binarization → v → χ_Ze

Usage:
    python3 eeg_ze_processor.py --demo
    python3 eeg_ze_processor.py --file recording.edf --age 35 --label "Subject_01"
    python3 eeg_ze_processor.py --batch data/ --resample 128 --out results/

Theory (Tkemaladze, Ze System as Observer):
    Binary sequence {x_k}: x_k = 1 if sample > median, else 0
    N_S = number of switches (x_k ≠ x_{k-1})
    Ze velocity:       v = N_S / (N - 1)                     range [0, 1]
    Fixed point:       v* ≈ 0.45631                           (max materialization)
    Ze cheating index: χ_Ze = 1 - |v - v*| / max(v*, 1-v*)   range [0, 1]
    Ze proper time:    τ = α * N * χ_Ze                       (temporal potential)
    Proxy:             v_peak = 2·f_peak/f_s  (exact for pure sinusoid; Sec. 2.2)
    Aging hypothesis:  χ_Ze decreases with age
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Ze constants ─────────────────────────────────────────────────────────────

V_STAR = 0.45631
# Upper bound for living systems (empirical, Tkemaladze 2024; not used in main
# calculations — retained for reference and visualization only).
CHI_MAX_LIVING = 0.839
ALPHA = 1.0   # normalization constant for τ (adjustable)

# ─── Core Ze computations ─────────────────────────────────────────────────────

def binarize(signal: np.ndarray) -> np.ndarray:
    """Convert continuous signal to binary {0,1} via median threshold.

    x_k = 1  if  signal[k] > median(signal),  else  0.
    Ties (signal[k] == median) → 0 (conservative convention).
    """
    return (signal > np.median(signal)).astype(np.int8)


def ze_velocity(binary_seq: np.ndarray) -> float:
    """v = N_S / (N-1) — fraction of switching events in [0, 1]."""
    N = len(binary_seq)
    if N < 2:
        return 0.0
    switches = int(np.sum(binary_seq[1:] != binary_seq[:-1]))
    return switches / (N - 1)


def ze_cheating_index(v: float) -> float:
    """χ_Ze = 1 - |v - v*| / max(v*, 1-v*)  ∈ [0, 1]; maximum at v = v*."""
    return 1.0 - abs(v - V_STAR) / max(V_STAR, 1.0 - V_STAR)


def ze_proper_time(N: int, chi: float) -> float:
    """τ = α * N * χ_Ze — accumulated temporal potential."""
    return ALPHA * N * chi


def compute_ze_metrics(signal: np.ndarray) -> dict:
    """Full Ze analysis of a 1D signal array (broadband, unfiltered).

    Returns
    -------
    dict with keys: N, N_S, v, v_star, chi_Ze, tau, v_deviation, chi_max_living
    """
    binary = binarize(signal)
    N = len(binary)
    N_S = int(np.sum(binary[1:] != binary[:-1]))
    v = ze_velocity(binary)
    chi = ze_cheating_index(v)
    return {
        "N":               N,
        "N_S":             N_S,
        "v":               round(v,   6),
        "v_star":          V_STAR,
        "chi_Ze":          round(chi, 6),
        "chi_max_living":  CHI_MAX_LIVING,
        "tau":             round(ze_proper_time(N, chi), 2),
        "v_deviation":     round(abs(v - V_STAR), 6),
    }


# ─── Narrowband Ze method ──────────────────────────────────────────────────────

def narrowband_ze(signal: np.ndarray, fs: float,
                  lowcut: float = 8.0, highcut: float = 12.0) -> dict:
    """Narrowband Ze method: bandpass filter → median binarization → v → χ_Ze.

    This is NOT the theoretical Ze definition (which operates on unfiltered
    signals). It applies Ze binarization within the alpha band only, capturing
    alpha-rhythm dynamics while isolating them from broadband noise.
    See manuscript Sec. 3.3 for the distinction.

    Parameters
    ----------
    signal  : 1D EEG signal array
    fs      : sampling frequency (Hz)
    lowcut  : lower bandpass edge (Hz), default 8.0
    highcut : upper bandpass edge (Hz), default 12.0

    Returns
    -------
    dict with keys: chi_Ze, v, N, N_S, lowcut, highcut, method
    """
    from scipy.signal import butter, filtfilt
    nyq = fs / 2.0
    if highcut >= nyq:
        raise ValueError(f"highcut ({highcut} Hz) must be < Nyquist ({nyq} Hz)")
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    binary = binarize(filtered)
    v = ze_velocity(binary)
    chi = ze_cheating_index(v)
    return {
        "method":   "narrowband_Ze",
        "lowcut":   lowcut,
        "highcut":  highcut,
        "N":        len(binary),
        "N_S":      int(np.sum(binary[1:] != binary[:-1])),
        "v":        round(v,   6),
        "chi_Ze":   round(chi, 6),
    }


# ─── Proxy method ─────────────────────────────────────────────────────────────

def alpha_peak_ze(signal: np.ndarray, fs: float,
                  f_band: tuple = (7.5, 13.0)) -> dict:
    """Proxy Ze method: PSD alpha peak → v_peak = 2·f_peak/f_s → χ_Ze.

    Mathematical basis (Sec. 2.2): for a pure sinusoid at f_peak Hz,
    median thresholding produces exactly v = 2·f_peak/f_s. Real narrow-band
    EEG alpha is approximately sinusoidal within each cycle, making this
    a valid approximation.

    Parameters
    ----------
    signal : 1D EEG signal array
    fs     : sampling frequency (Hz)
    f_band : alpha frequency band bounds (Hz), default (7.5, 13.0)

    Returns
    -------
    dict with keys: chi_Ze, v_peak, f_peak, f_band, method
    """
    from scipy.signal import welch
    nperseg = min(int(4 * fs), len(signal))  # 4-second windows
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= f_band[0]) & (freqs <= f_band[1])
    if not mask.any():
        raise ValueError(f"No PSD bins in band {f_band} Hz at fs={fs} Hz")
    f_peak = float(freqs[mask][np.argmax(psd[mask])])
    v_peak = 2.0 * f_peak / fs
    chi = ze_cheating_index(v_peak)
    return {
        "method":  "proxy",
        "f_peak":  round(f_peak, 4),
        "f_band":  f_band,
        "v_peak":  round(v_peak, 6),
        "chi_Ze":  round(chi,    6),
    }


# ─── Cuban dataset support ────────────────────────────────────────────────────

def load_cuban_mcr(filepath: str, f_band: tuple = (7.5, 13.0)) -> dict:
    """Load a Cuban Normative EEG cross-spectral matrix (.mat) and compute χ_Ze.

    The Cuban dataset (Zenodo 4244765) provides pre-computed cross-spectral
    matrices Mcr (19×19×49) at 100 Hz sampling rate, spectral resolution 0.39 Hz.
    The diagonal Mcr[i,i,:] is the auto-spectrum (PSD) of channel i.

    Parameters
    ----------
    filepath : path to a *_cross.mat file
    f_band   : alpha band for peak detection, default (7.5, 13.0) Hz

    Returns
    -------
    dict with keys: age, sex, f_peak, v_peak, chi_Ze, n_channels, method, filepath
    """
    import scipy.io
    mat = scipy.io.loadmat(str(filepath))

    # Extract age and sex (available in most files)
    age = float(mat['age'].flatten()[0]) if 'age' in mat else None
    sex = str(mat.get('sex', ['?'])[0][0]) if 'sex' in mat else None

    freqs = mat['frange'].flatten().astype(float)
    mcr   = mat['Mcr']
    n_ch  = mcr.shape[0]

    # Mean PSD across all channels (robust to channel-specific artefacts)
    mean_psd = np.mean([mcr[i, i, :].real for i in range(n_ch)], axis=0)

    mask = (freqs >= f_band[0]) & (freqs <= f_band[1])
    if not mask.any():
        raise ValueError(f"No spectral bins in {f_band} Hz in {filepath}")

    f_peak = float(freqs[mask][np.argmax(mean_psd[mask])])
    fs = 100.0   # Cuban dataset sampling rate
    v_peak = 2.0 * f_peak / fs
    chi = ze_cheating_index(v_peak)

    return {
        "method":     "proxy_cuban_mcr",
        "filepath":   str(filepath),
        "age":        age,
        "sex":        sex,
        "n_channels": n_ch,
        "f_peak":     round(f_peak, 4),
        "f_band":     f_band,
        "v_peak":     round(v_peak, 6),
        "chi_Ze":     round(chi,    6),
    }


# ─── Group statistics ──────────────────────────────────────────────────────────

def group_statistics(chi_young: np.ndarray, chi_old: np.ndarray,
                     sex_young: np.ndarray = None,
                     sex_old:   np.ndarray = None,
                     n_boot: int = 10000,
                     rng_seed: int = 42) -> dict:
    """Compute group comparison statistics as reported in the manuscript.

    Includes: t-test, Cohen's d, bootstrap 95% CI for d, statistical power,
    AUC (Mann-Whitney), ANCOVA with sex covariate (if sex arrays provided),
    and group×sex interaction test.

    Parameters
    ----------
    chi_young : χ_Ze values for young group
    chi_old   : χ_Ze values for old group
    sex_young : sex labels ('M'/'F') for young, optional — enables ANCOVA
    sex_old   : sex labels ('M'/'F') for old, optional
    n_boot    : bootstrap resamples for CI (default 10000)
    rng_seed  : random seed for reproducibility

    Returns
    -------
    dict with all statistics (see keys below)
    """
    from scipy import stats as sp_stats

    n1, n2 = len(chi_young), len(chi_old)
    m1, m2 = chi_young.mean(), chi_old.mean()
    s1, s2 = chi_young.std(ddof=1), chi_old.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / pooled_sd

    t_stat, p_val = sp_stats.ttest_ind(chi_young, chi_old)

    # Bootstrap CI for d
    rng = np.random.default_rng(rng_seed)
    d_boot = []
    for _ in range(n_boot):
        y_b = rng.choice(chi_young, size=n1, replace=True)
        o_b = rng.choice(chi_old,   size=n2, replace=True)
        ps  = np.sqrt(((n1-1)*y_b.std(ddof=1)**2 + (n2-1)*o_b.std(ddof=1)**2) / (n1+n2-2))
        d_boot.append((y_b.mean() - o_b.mean()) / ps if ps > 0 else 0.0)
    d_ci = np.percentile(d_boot, [2.5, 97.5])

    # Statistical power (normal approximation)
    ncp   = d * np.sqrt(n1 * n2 / (n1 + n2))
    power = float(1 - sp_stats.norm.cdf(1.96 - ncp) + sp_stats.norm.cdf(-1.96 - ncp))

    # AUC (Mann-Whitney)
    U_stat, p_mw = sp_stats.mannwhitneyu(chi_young, chi_old, alternative='greater')
    auc = U_stat / (n1 * n2)
    auc_boot = []
    for _ in range(n_boot):
        y_b = rng.choice(chi_young, size=n1, replace=True)
        o_b = rng.choice(chi_old,   size=n2, replace=True)
        U_b, _ = sp_stats.mannwhitneyu(y_b, o_b, alternative='greater')
        auc_boot.append(U_b / (n1 * n2))
    auc_ci = np.percentile(auc_boot, [2.5, 97.5])

    result = {
        "n_young": n1, "n_old": n2,
        "mean_young": round(m1, 6), "mean_old": round(m2, 6),
        "sd_young": round(s1, 6),   "sd_old":   round(s2, 6),
        "t": round(t_stat, 4), "p": round(p_val, 6),
        "cohens_d": round(d, 4),
        "d_ci_95": [round(d_ci[0], 4), round(d_ci[1], 4)],
        "power": round(power, 3),
        "auc": round(auc, 4),
        "auc_ci_95": [round(auc_ci[0], 4), round(auc_ci[1], 4)],
        "auc_p_onesided": round(p_mw, 6),
    }

    # ANCOVA + group×sex interaction (requires sex arrays)
    if sex_young is not None and sex_old is not None:
        chis_all   = np.concatenate([chi_young, chi_old])
        group_code = np.concatenate([np.zeros(n1), np.ones(n2)])  # 0=young,1=old
        sex_code   = np.concatenate([
            (sex_young == 'F').astype(float),
            (sex_old   == 'F').astype(float)
        ])
        interaction = group_code * sex_code
        n_all = len(chis_all)

        # Full model: intercept + group + sex + interaction
        X_full = np.column_stack([np.ones(n_all), group_code, sex_code, interaction])
        b_full, _, _, _ = np.linalg.lstsq(X_full, chis_all, rcond=None)
        ss_res_full = np.sum((chis_all - X_full @ b_full) ** 2)

        # Reduced model 1: no interaction (for interaction F-test)
        X_noint = np.column_stack([np.ones(n_all), group_code, sex_code])
        b_noint, _, _, _ = np.linalg.lstsq(X_noint, chis_all, rcond=None)
        ss_res_noint = np.sum((chis_all - X_noint @ b_noint) ** 2)

        # Reduced model 2: no group (for group F-test, sex-adjusted)
        X_nosex = np.column_stack([np.ones(n_all), sex_code])
        b_nosex, _, _, _ = np.linalg.lstsq(X_nosex, chis_all, rcond=None)
        ss_res_nosex = np.sum((chis_all - X_nosex @ b_nosex) ** 2)

        df_res = n_all - X_noint.shape[1]
        F_group = ((ss_res_nosex - ss_res_noint) / 1) / (ss_res_noint / df_res)
        p_group = float(1 - __import__('scipy').stats.f.cdf(F_group, 1, df_res))

        df_res_full = n_all - X_full.shape[1]
        F_inter = ((ss_res_noint - ss_res_full) / 1) / (ss_res_full / df_res_full)
        p_inter = float(1 - __import__('scipy').stats.f.cdf(F_inter, 1, df_res_full))

        r2 = 1 - ss_res_noint / np.sum((chis_all - chis_all.mean()) ** 2)

        result["ancova"] = {
            "beta_group":       round(float(b_noint[1]), 6),
            "beta_sex":         round(float(b_noint[2]), 6),
            "F_group":          round(F_group, 4),
            "p_group":          round(p_group, 6),
            "df":               f"F(1,{df_res})",
            "r2":               round(r2, 4),
            "F_interaction":    round(F_inter, 4),
            "p_interaction":    round(p_inter, 6),
            "df_interaction":   f"F(1,{df_res_full})",
        }

    return result


# ─── EEG loading ──────────────────────────────────────────────────────────────

def _check_mne():
    """Raise ImportError with install instruction if mne is missing."""
    try:
        import mne  # noqa: F401
    except ImportError:
        raise ImportError(
            "mne is required for EEG file loading.\n"
            "Install with:  pip install mne\n"
            "See: https://mne.tools/stable/install/index.html"
        )


def load_edf(filepath: str, duration_s: float = None):
    """Load EDF/BDF file using MNE."""
    _check_mne()
    import mne
    mne.set_log_level('WARNING')
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    if duration_s:
        raw.crop(tmax=min(duration_s, raw.times[-1]))
    return raw


def load_brainvision(filepath: str, duration_s: float = None,
                     resample_hz: float = None):
    """Load BrainVision (.vhdr) file. The .vmrk and .eeg must be co-located."""
    _check_mne()
    import mne
    mne.set_log_level('WARNING')
    raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
    if duration_s:
        raw.crop(tmax=min(duration_s, raw.times[-1]))
    if resample_hz and raw.info['sfreq'] > resample_hz:
        logger.info("Resampling %.0f Hz → %.0f Hz", raw.info['sfreq'], resample_hz)
        raw.resample(resample_hz, npad='auto')
    return raw


def load_any(filepath: str, duration_s: float = None,
             resample_hz: float = None):
    """Auto-detect EEG format and load. Supports .vhdr, .edf, .bdf."""
    ext = Path(filepath).suffix.lower()
    if ext == '.vhdr':
        return load_brainvision(filepath, duration_s, resample_hz)
    elif ext in ('.edf', '.bdf'):
        raw = load_edf(filepath, duration_s)
        if resample_hz and raw.info['sfreq'] > resample_hz:
            logger.info("Resampling %.0f Hz → %.0f Hz", raw.info['sfreq'], resample_hz)
            raw.resample(resample_hz, npad='auto')
        return raw
    else:
        raise ValueError(f"Unsupported format: {ext} — use .vhdr, .edf, or .bdf")


# ─── Per-channel analysis ──────────────────────────────────────────────────────

def analyze_raw(raw, label: str = "Subject", age: int = None,
                compute_proxy: bool = True,
                compute_narrowband: bool = True,
                narrowband_low: float = 8.0,
                narrowband_high: float = 12.0,
                alpha_band: tuple = (7.5, 13.0)) -> dict:
    """Run Ze analysis on all EEG channels of an MNE Raw object.

    For each channel, computes broadband Ze metrics (compute_ze_metrics),
    and optionally the proxy method (alpha_peak_ze) and narrowband Ze
    (narrowband_ze). Metrics are averaged across channels.

    Per-channel median binarization: each channel is binarized independently
    using its own median. This is correct because Ze velocity is defined for
    a single time series; combining channels before binarization would conflate
    different amplitude scales. The subject-level χ_Ze is the mean across channels,
    consistent with the methodology in the manuscript (Sec. 3.2).

    Parameters
    ----------
    raw              : MNE Raw object (EEG channels selected automatically)
    label            : subject identifier string
    age              : subject age in years (optional)
    compute_proxy    : compute proxy method (alpha peak → χ_Ze)
    compute_narrowband: compute narrowband Ze (bandpass → binarize → χ_Ze)
    narrowband_low   : bandpass lower edge for narrowband Ze (Hz)
    narrowband_high  : bandpass upper edge for narrowband Ze (Hz)
    alpha_band       : frequency band for proxy alpha peak detection (Hz)

    Returns
    -------
    dict with keys:
        label, age, sfreq, n_channels, duration_s, timestamp,
        channels  : per-channel broadband Ze metrics
        summary   : mean/std/min/max χ_Ze and v across channels
        proxy     : {chi_Ze, v_peak, f_peak} averaged across channels (if enabled)
        narrowband: {chi_Ze, v} averaged across channels (if enabled)
    """
    _check_mne()
    import mne
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(picks) == 0:
        picks = list(range(len(raw.ch_names)))
    fs = raw.info['sfreq']

    ch_broadband = {}
    proxy_vals, nb_vals = [], []

    for idx in picks:
        ch = raw.ch_names[idx]
        sig = raw.get_data(picks=[idx])[0]

        ch_broadband[ch] = compute_ze_metrics(sig)

        if compute_proxy:
            try:
                px = alpha_peak_ze(sig, fs, f_band=alpha_band)
                proxy_vals.append(px)
            except Exception as exc:
                logger.debug("Proxy failed on %s: %s", ch, exc)

        if compute_narrowband:
            try:
                nb = narrowband_ze(sig, fs, lowcut=narrowband_low,
                                   highcut=narrowband_high)
                nb_vals.append(nb)
            except Exception as exc:
                logger.debug("Narrowband Ze failed on %s: %s", ch, exc)

    chi_vals = [r["chi_Ze"] for r in ch_broadband.values()]
    v_vals   = [r["v"]      for r in ch_broadband.values()]

    result = {
        "label":      label,
        "age":        age,
        "sfreq":      fs,
        "n_channels": len(picks),
        "duration_s": round(raw.times[-1], 2),
        "timestamp":  datetime.now().isoformat(),
        "channels":   ch_broadband,
        "summary": {
            "chi_Ze_mean": round(float(np.mean(chi_vals)), 6),
            "chi_Ze_std":  round(float(np.std(chi_vals)),  6),
            "chi_Ze_min":  round(float(np.min(chi_vals)),  6),
            "chi_Ze_max":  round(float(np.max(chi_vals)),  6),
            "v_mean":      round(float(np.mean(v_vals)),   6),
            "v_std":       round(float(np.std(v_vals)),    6),
        },
    }

    if proxy_vals:
        result["proxy"] = {
            "chi_Ze":  round(float(np.mean([p["chi_Ze"]  for p in proxy_vals])), 6),
            "v_peak":  round(float(np.mean([p["v_peak"]  for p in proxy_vals])), 6),
            "f_peak":  round(float(np.mean([p["f_peak"]  for p in proxy_vals])), 4),
            "n_ch":    len(proxy_vals),
        }

    if nb_vals:
        result["narrowband"] = {
            "chi_Ze":     round(float(np.mean([n["chi_Ze"] for n in nb_vals])), 6),
            "v":          round(float(np.mean([n["v"]      for n in nb_vals])), 6),
            "lowcut":     narrowband_low,
            "highcut":    narrowband_high,
            "n_ch":       len(nb_vals),
        }

    return result


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_ze_channels(result: dict, out_dir: str = "."):
    """Bar chart of χ_Ze and v per channel, with proxy and narrowband summaries."""
    label    = result["label"]
    age_str  = f", age {result['age']}" if result["age"] else ""
    channels = result["channels"]
    ch_names = list(channels.keys())
    chi_vals = [channels[c]["chi_Ze"] for c in ch_names]
    v_vals   = [channels[c]["v"]      for c in ch_names]

    nrows = 3 if ("proxy" in result or "narrowband" in result) else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(max(10, len(ch_names) * 0.4), 4 * nrows))
    fig.suptitle(f"Ze Analysis: {label}{age_str}  (fs={result['sfreq']} Hz)",
                 fontsize=12, fontweight='bold')

    ax1, ax2 = axes[0], axes[1]

    ax1.bar(ch_names, chi_vals,
            color=['#2E74B5' if c >= V_STAR * 0.9 else '#C55A11' for c in chi_vals],
            edgecolor='white', linewidth=0.5)
    ax1.axhline(result["summary"]["chi_Ze_mean"], color='red', lw=1.5,
                label=f'mean={result["summary"]["chi_Ze_mean"]:.4f}')
    if "proxy" in result:
        ax1.axhline(result["proxy"]["chi_Ze"], color='green', ls='--', lw=1.5,
                    label=f'proxy χ_Ze={result["proxy"]["chi_Ze"]:.4f}')
    if "narrowband" in result:
        ax1.axhline(result["narrowband"]["chi_Ze"], color='purple', ls=':', lw=1.5,
                    label=f'narrowband χ_Ze={result["narrowband"]["chi_Ze"]:.4f}')
    ax1.set_ylabel("χ_Ze"); ax1.set_ylim(0, 1)
    ax1.set_xticks(range(len(ch_names)))
    ax1.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax1.legend(fontsize=8); ax1.set_title("Ze Cheating Index per Channel")

    ax2.bar(ch_names, v_vals, color='#7B9FC4', edgecolor='white', linewidth=0.5)
    ax2.axhline(V_STAR, color='green', ls='--', lw=1,
                label=f'v* = {V_STAR}')
    ax2.axhline(result["summary"]["v_mean"], color='red', lw=1.5,
                label=f'mean={result["summary"]["v_mean"]:.4f}')
    ax2.set_ylabel("Ze velocity v"); ax2.set_ylim(0, 1)
    ax2.set_xticks(range(len(ch_names)))
    ax2.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax2.legend(fontsize=8); ax2.set_title("Ze Velocity per Channel")

    if nrows == 3:
        ax3 = axes[2]
        methods, values, colors_m = [], [], []
        if "proxy" in result:
            methods.append(f"Proxy\n(f_peak={result['proxy']['f_peak']:.1f} Hz)")
            values.append(result["proxy"]["chi_Ze"])
            colors_m.append('#2E74B5')
        if "narrowband" in result:
            methods.append(f"Narrowband Ze\n({result['narrowband']['lowcut']}–"
                           f"{result['narrowband']['highcut']} Hz)")
            values.append(result["narrowband"]["chi_Ze"])
            colors_m.append('#7B2D8B')
        methods.append("Broadband Ze\n(unfiltered)")
        values.append(result["summary"]["chi_Ze_mean"])
        colors_m.append('#C55A11')

        ax3.bar(methods, values, color=colors_m, edgecolor='white', width=0.5)
        ax3.axhline(V_STAR, color='green', ls='--', lw=1, label=f'v* = {V_STAR}')
        for xi, v in enumerate(values):
            ax3.text(xi, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        ax3.set_ylabel("χ_Ze"); ax3.set_ylim(0, 1)
        ax3.legend(fontsize=8); ax3.set_title("Method Comparison")

    plt.tight_layout()
    fname = f"ze_{label.replace(' ', '_')}.png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Plot saved: %s", fpath)
    print(f"  📊 Plot saved: {fpath}")
    return fpath


def plot_group_comparison(results: list, out_dir: str = "."):
    """χ_Ze vs age scatter with linear trend and optional group statistics."""
    valid = [r for r in results if r.get("age") is not None]
    if len(valid) < 2:
        return
    ages = np.array([r["age"] for r in valid])
    chis = np.array([r["summary"]["chi_Ze_mean"] for r in valid])
    labels = [r["label"] for r in valid]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(ages, chis, s=80, color='#2E74B5', zorder=5, alpha=0.8)
    for a, c, l in zip(ages, chis, labels):
        ax.annotate(l, (a, c), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color='#333')

    if len(ages) >= 3:
        z = np.polyfit(ages, chis, 1)
        x_line = np.linspace(ages.min(), ages.max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', lw=1.5,
                label=f'Δχ_Ze/year = {z[0]:.5f}')
        # Report Pearson r
        r = np.corrcoef(ages, chis)[0, 1]
        ax.text(0.03, 0.05, f'r = {r:.3f}', transform=ax.transAxes,
                fontsize=10, color='red')

    ax.axhline(V_STAR, color='green', ls=':', lw=1,
               label=f'v* = {V_STAR} (Ze optimal)')
    ax.set_xlabel("Age (years)"); ax.set_ylabel("χ_Ze (mean across channels)")
    ax.set_title("Ze Cheating Index vs Age\n(Ze Aging Hypothesis: χ_Ze ↓ with age)")
    ax.set_ylim(0, 1); ax.legend(fontsize=9)

    fpath = os.path.join(out_dir, "ze_age_comparison.png")
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Group plot: {fpath}")
    return fpath


# ─── Report helpers ───────────────────────────────────────────────────────────

def print_summary(result: dict):
    s = result["summary"]
    age_str = f", age {result['age']}" if result.get("age") else ""
    print(f"\n{'─'*54}")
    print(f"  Subject:      {result['label']}{age_str}")
    print(f"  Duration:     {result.get('duration_s', '?')}s  "
          f"Channels: {result.get('n_channels', 1)}  "
          f"fs: {result.get('sfreq', '?')} Hz")
    print(f"  Broadband Ze: χ_Ze = {s['chi_Ze_mean']:.4f} ± {s['chi_Ze_std']:.4f}  "
          f"v = {s['v_mean']:.4f}")
    if "proxy" in result:
        p = result["proxy"]
        print(f"  Proxy:        χ_Ze = {p['chi_Ze']:.4f}  "
              f"f_peak = {p['f_peak']:.2f} Hz  v = {p['v_peak']:.4f}")
    if "narrowband" in result:
        nb = result["narrowband"]
        print(f"  Narrowband Ze: χ_Ze = {nb['chi_Ze']:.4f}  "
              f"band = {nb['lowcut']}–{nb['highcut']} Hz")
    chi_m = s['chi_Ze_mean']
    status = ("🟢 HIGH"   if chi_m > 0.60 else
              "🟡 MEDIUM" if chi_m > 0.40 else "🔴 LOW")
    print(f"  Status:       {status}  (v* = {V_STAR})")
    print(f"{'─'*54}")


def save_json(result: dict, out_dir: str = "."):
    fname = f"ze_{result['label'].replace(' ', '_')}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  💾 JSON: {fpath}")
    return fpath


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING,
                        format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description="EEG Ze Processor — compute χ_Ze from EEG recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 eeg_ze_processor.py --demo
  python3 eeg_ze_processor.py --file sub-01_eeg.edf --age 28 --resample 128
  python3 eeg_ze_processor.py --batch data/edf/ --resample 128 --out results/
  python3 eeg_ze_processor.py --cuban /path/to/EyesClose/ --out results/
""")
    parser.add_argument('--file',     help="EDF/BDF/.vhdr file path")
    parser.add_argument('--age',      type=int, help="Subject age (years)")
    parser.add_argument('--label',    default="Subject")
    parser.add_argument('--duration', type=float, default=None,
                        help="Crop to N seconds (default: full)")
    parser.add_argument('--resample', type=float, default=128.0,
                        help="Resample to Hz (default: 128; 0 = disabled)")
    parser.add_argument('--batch',    help="Process all EDF files in folder")
    parser.add_argument('--cuban',    help="Process Cuban .mat folder")
    parser.add_argument('--demo',     action='store_true',
                        help="Synthetic EEG demo (no data required)")
    parser.add_argument('--out',      default=".", help="Output directory")
    parser.add_argument('--verbose',  action='store_true',
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(args.out, exist_ok=True)
    all_results = []

    # ── DEMO mode ─────────────────────────────────────────────────────────────
    if args.demo:
        print("\n🧪 Ze EEG Demo — Synthetic signals (fs=128 Hz, no data required)")
        print("   Illustrates Ze aging hypothesis using realistic alpha frequencies.\n")
        print("   Note: these are pure sinusoids, not real EEG — for illustration only.\n")

        np.random.seed(42)
        fs_demo = 128.0
        # Realistic alpha frequencies at fs=128 Hz:
        #   v_peak = 2*f/128;  v* = 0.45631 → f_opt = 29.2 Hz (beta/gamma)
        #   Alpha range: 8–13 Hz → v = 0.125–0.203
        #   Young adult peak: ~10.5 Hz → v ≈ 0.164
        #   Child:            ~ 9.0 Hz → v ≈ 0.141
        #   Elderly:          ~ 8.5 Hz → v ≈ 0.133
        demo_subjects = [
            ("Young_25",  25,  10.5),  # typical young adult alpha peak
            ("Middle_50", 50,   9.5),  # moderate aging
            ("Elder_72",  72,   8.5),  # elderly alpha slowing
        ]
        for label, age, f_hz in demo_subjects:
            N = int(fs_demo * 60)  # 60 s at 128 Hz
            t = np.arange(N) / fs_demo
            signal = np.sin(2 * np.pi * f_hz * t) + 0.08 * np.random.randn(N)

            bb  = compute_ze_metrics(signal)
            prx = alpha_peak_ze(signal, fs_demo)
            nbz = narrowband_ze(signal, fs_demo)

            r = {
                "label": label, "age": age,
                "sfreq": fs_demo, "n_channels": 1, "duration_s": 60,
                "channels": {"Cz": bb},
                "summary": {
                    "chi_Ze_mean": bb["chi_Ze"], "chi_Ze_std": 0.0,
                    "chi_Ze_min":  bb["chi_Ze"], "chi_Ze_max": bb["chi_Ze"],
                    "v_mean": bb["v"], "v_std": 0.0,
                },
                "proxy":      prx,
                "narrowband": nbz,
                "timestamp":  datetime.now().isoformat(),
            }
            print_summary(r)
            save_json(r, args.out)
            all_results.append(r)

        if len(all_results) >= 2:
            plot_group_comparison(all_results, args.out)

        # Ze hypothesis check
        chis = np.array([r["summary"]["chi_Ze_mean"] for r in all_results])
        ages = np.array([r["age"] for r in all_results])
        r_corr = np.corrcoef(ages, chis)[0, 1]
        print(f"\n📉 Pearson r (age vs χ_Ze): {r_corr:.4f}")
        if r_corr < -0.3:
            print("  ✅ Ze hypothesis SUPPORTED: χ_Ze decreases with age")
        else:
            print("  ⚠️  Weak or no aging trend on this demo sample")

        print(f"\n  Ze-optimal frequency at fs={fs_demo:.0f} Hz: "
              f"f_opt = v* × fs/2 = {V_STAR * fs_demo / 2:.1f} Hz (beta/gamma boundary)")
        return

    # ── CUBAN mode ────────────────────────────────────────────────────────────
    if args.cuban:
        cuban_dir = Path(args.cuban)
        mat_files = sorted(cuban_dir.glob("*_cross.mat"))
        print(f"\n📂 Cuban dataset: {len(mat_files)} .mat files in {args.cuban}")
        cuban_results = []
        for f in mat_files:
            try:
                r = load_cuban_mcr(str(f))
                print(f"  {f.name}: age={r['age']:.1f}  "
                      f"f_peak={r['f_peak']:.2f} Hz  χ_Ze={r['chi_Ze']:.4f}")
                cuban_results.append(r)
            except Exception as exc:
                logger.warning("Skipped %s: %s", f.name, exc)
        if cuban_results:
            out_path = os.path.join(args.out, "ze_cuban_summary.json")
            with open(out_path, 'w') as fout:
                json.dump(cuban_results, fout, indent=2)
            print(f"\n  💾 Cuban summary: {out_path}  (N={len(cuban_results)})")
        return

    # ── BATCH mode ────────────────────────────────────────────────────────────
    if args.batch:
        batch_path = Path(args.batch)
        files = (sorted(batch_path.glob("*.vhdr")) +
                 sorted(batch_path.glob("*.edf")) +
                 sorted(batch_path.glob("*.EDF")) +
                 sorted(batch_path.glob("*.bdf")))
        print(f"\n📂 Batch: {len(files)} files in {args.batch}")
        resample = args.resample if args.resample > 0 else None
        for f in files:
            print(f"\n  ▶ {f.name}")
            try:
                raw = load_any(str(f), duration_s=args.duration,
                               resample_hz=resample)
                result = analyze_raw(raw, label=f.stem)
                print_summary(result)
                save_json(result, args.out)
                plot_ze_channels(result, args.out)
                all_results.append(result)
            except Exception as exc:
                logger.error("Error processing %s: %s", f.name, exc)
                print(f"  ❌ Error: {exc}")
        if len(all_results) >= 2:
            plot_group_comparison(all_results, args.out)
        return

    # ── SINGLE FILE mode ──────────────────────────────────────────────────────
    if args.file:
        resample = args.resample if args.resample > 0 else None
        raw = load_any(args.file, duration_s=args.duration, resample_hz=resample)
        result = analyze_raw(raw, label=args.label, age=args.age)
        print_summary(result)
        save_json(result, args.out)
        plot_ze_channels(result, args.out)
        return

    parser.print_help()
    print("\n  Quickstart: python3 eeg_ze_processor.py --demo")


if __name__ == "__main__":
    main()
