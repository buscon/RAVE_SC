#!/usr/bin/env python3
"""
Corpus analysis for concatenative synthesis: cuts a recording into overlapping segments and measures
a few acoustic features of each. Writes <name>_seg<length>.csv next to the recording (or in --out).

Features (computed on the mono mix):
  rms_db      loudness of the segment
  centroid    spectral centroid in Hz (brightness)
  flatness    spectral flatness, 0..1 (1 = noise-like / even spectrum, near 0 = tonal or very uneven)
  low_share   share of the energy below 300 Hz (rumble, body of the waves)
  activity    standard deviation of the loudness in 50 ms frames, in dB (steady = low, bursts = high)
  centroid_hi spectral centroid of the part above 500 Hz (brightness of the hiss, not dominated by the rumble)
  hf_db       energy above 2 kHz relative to the total, in dB (more negative = less hiss)
  timbre1,2   the two main directions in which the segments of this recording differ in spectral shape (MFCC 1-12,
              without MFCC 0 so that loudness is mostly left out; principal components over all segments of
              the table). Only meaningful within one recording. The sign is set so that a higher value is brighter.
  onset_rate  sound events per second: peaks in the loudness that rise at least 6 dB above the local level
              (the median of the surrounding second), at least 0.1 s apart. Counts calls, splashes, bursts.

Usage:  python make_corpus.py audio/Seawaves1.wav --lengths 0.1 0.25 0.5 1 --hop-fraction 0.25
"""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


def onset_times(x, sr):
    """Times (s) of loudness peaks that stand out by 6 dB from the median of the surrounding second."""
    w = int(0.01 * sr)
    db = 20 * np.log10(np.sqrt((x[:len(x) // w * w].reshape(-1, w) ** 2).mean(1)) + 1e-9)
    excess = db - median_filter(db, size=101, mode="nearest")
    peaks, _ = find_peaks(excess, height=6.0, distance=10)
    return peaks * 0.01


def two_scenes(z, pc1, iters=50):
    """Split the segments into two groups of similar spectral shape (k-means with k=2 on the MFCC).
    Returns scene (0 = lower values on the first timbre axis, 1 = higher) and typ (0..1 within each group:
    0 = closest to the group's centre, 1 = farthest)."""
    # start from the two ends of the first axis, so the result is the same every time
    lo, hi = np.argsort(pc1)[:max(1, len(pc1) // 10)], np.argsort(pc1)[-max(1, len(pc1) // 10):]
    c = np.array([z[lo].mean(0), z[hi].mean(0)])
    for _ in range(iters):
        d = ((z[:, None, :] - c[None]) ** 2).sum(2)
        lab = d.argmin(1)
        new = np.array([z[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    d = np.sqrt(((z - c[lab]) ** 2).sum(1))
    typ = np.zeros(len(z))
    for k in range(2):
        idx = np.where(lab == k)[0]
        typ[idx[np.argsort(d[idx])]] = np.linspace(0, 1, len(idx))
    return lab.astype(float), typ


def mel_filterbank(n_fft_bins, sr, n_mels=26, fmax=16000.0):
    """Triangular mel filters on the rfft bins of a segment of n_fft_bins*2-2 samples."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(50.0), mel(min(fmax, sr / 2)), n_mels + 2))
    return pts


def mfcc_of(S, f, sr, n_mels=26, n_coef=13):
    """MFCC 0..12 from a power spectrum S at frequencies f."""
    pts = mel_filterbank(len(f), sr, n_mels)
    E = np.empty(n_mels)
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        w = np.clip(np.minimum((f - lo) / (mid - lo), (hi - f) / (hi - mid)), 0, None)
        E[i] = (w * S).sum() + 1e-18
    logE = np.log(E)
    k = np.arange(n_coef)[:, None]
    dct = np.cos(np.pi * k * (np.arange(n_mels) + 0.5) / n_mels)
    return dct @ logE


def analyse(x, sr, start, length, onsets):
    seg = x[int(start * sr):int((start + length) * sr)]
    n = len(seg)
    rms = np.sqrt((seg ** 2).mean()) + 1e-9
    S = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2 + 1e-18
    f = np.fft.rfftfreq(n, 1 / sr)
    keep = f < 20000
    S, f = S[keep], f[keep]
    P = S / S.sum()
    centroid = float((P * f).sum())
    flat = float(np.exp(np.log(S).mean()) / S.mean())
    low = float(S[f < 300].sum() / S.sum())
    hi = f > 500
    centroid_hi = float((S[hi] * f[hi]).sum() / S[hi].sum())   # brightness of the part above 500 Hz
    hf_db = float(10 * np.log10(S[f > 2000].sum() / S.sum()))   # energy above 2 kHz relative to the total, dB
    w = max(int(0.01 * sr), int(min(0.05, length / 6) * sr))   # 50 ms frames; shorter ones for very short segments
    fr = np.sqrt((seg[:n // w * w].reshape(-1, w) ** 2).mean(1)) + 1e-9
    activity = float(np.std(20 * np.log10(fr)))
    rate = float(((onsets >= start) & (onsets < start + length)).sum() / length)
    return [start, length, 20 * np.log10(rms), centroid, flat, low, activity, centroid_hi, hf_db, rate], mfcc_of(S, f, sr)[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--lengths", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--hop-fraction", type=float, default=0.25, help="hop as a fraction of the segment length")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    x, sr = sf.read(a.audio, always_2d=True)
    x = x.mean(1)
    dur = len(x) / sr
    out = Path(a.out) if a.out else Path(a.audio).parent
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.audio).stem
    print(f"{a.audio}: {dur:.1f} s, {sr} Hz", flush=True)
    onsets = onset_times(x, sr)
    print(f"  {len(onsets)} sound events found ({len(onsets) / dur:.2f} per second)", flush=True)
    for L in a.lengths:
        hop = L * a.hop_fraction
        res = [analyse(x, sr, s, L, onsets) for s in np.arange(0, dur - L, hop)]
        rows = np.array([r[0] for r in res])
        mf = np.array([r[1] for r in res])
        z = (mf - mf.mean(0)) / (mf.std(0) + 1e-9)
        _, sv, vt = np.linalg.svd(z, full_matrices=False)
        pcs = z @ vt[:2].T
        for j in range(2):   # sign: higher = brighter (positive correlation with centroid_hi)
            if np.corrcoef(pcs[:, j], rows[:, 7])[0, 1] < 0:
                pcs[:, j] = -pcs[:, j]
        share = sv ** 2 / (sv ** 2).sum()
        scene, typ = two_scenes(z, pcs[:, 0])
        rows = np.hstack([rows, pcs, scene[:, None], typ[:, None]])
        print(f"     two scenes: {int((scene == 0).sum())} and {int((scene == 1).sum())} segments", flush=True)
        path = out / f"{stem}_seg{L:g}.csv"
        np.savetxt(path, np.array(rows), delimiter=",", fmt="%.4f",
                   header="start,dur,rms_db,centroid,flatness,low_share,activity,centroid_hi,hf_db,onset_rate,timbre1,timbre2,scene,typical", comments="")
        arr = rows
        print(f"     timbre axes explain {share[0]*100:.0f} % and {share[1]*100:.0f} % of the MFCC variance", flush=True)
        print(f"  {path.name}: {len(rows)} segments", flush=True)
        for j, nm in enumerate(["rms_db", "centroid", "flatness", "low_share", "activity", "centroid_hi", "hf_db", "onset_rate", "timbre1", "timbre2", "scene", "typical"], 2):
            q = np.percentile(arr[:, j], [5, 50, 95])
            print(f"     {nm:10s} 5%: {q[0]:9.3f}  median: {q[1]:9.3f}  95%: {q[2]:9.3f}")


if __name__ == "__main__":
    main()
