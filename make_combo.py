#!/usr/bin/env python3
"""
Joins excerpts of several recordings into one WAV, so that concatenative synthesis (rave_concat.scd) can use a
corpus with real contrast, e.g. sea + birds. The analysis (make_corpus.py) then treats it as one recording, so the
timbre axes are computed over all parts together and can separate the recordings from each other.

Usage:  python make_combo.py audio/Seawaves1.wav audio/bluethroat.wav --seconds 60 --out audio/combo.wav

--seconds   length taken from each file. For longer files the stretch with most sound events is taken.
--no-match  do not match the loudness of the files (by default each excerpt gets the same rms).
Prints where each file starts in the result.
"""
import argparse
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from make_corpus import onset_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="audio/combo.wav")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--no-match", action="store_true")
    a = ap.parse_args()
    parts, t = [], 0.0
    for path in a.audio:
        x, sr = sf.read(path, always_2d=True)
        if sr != a.sr:
            g = gcd(sr, a.sr)
            x = resample_poly(x, a.sr // g, sr // g, axis=0)
        L = int(a.seconds * a.sr)
        if len(x) > L:   # the stretch with the most sound events
            on = onset_times(x.mean(1), a.sr)
            best, bs = 0, -1
            for s0 in np.arange(0, len(x) - L, a.sr // 2):
                n = ((on >= s0 / a.sr) & (on < s0 / a.sr + a.seconds)).sum()
                if n > bs: best, bs = s0, n
            x = x[int(best):int(best) + L]
            print(f"{Path(path).name}: using {best / a.sr:.1f}-{best / a.sr + a.seconds:.1f} s ({bs} events)")
        else:
            print(f"{Path(path).name}: using all {len(x) / a.sr:.1f} s")
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x = x[:, :2]
        if not a.no_match:
            x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        fade = int(0.02 * a.sr)
        x[:fade] *= np.linspace(0, 1, fade)[:, None]
        x[-fade:] *= np.linspace(1, 0, fade)[:, None]
        print(f"   starts at {t:.1f} s in the result")
        t += len(x) / a.sr
        parts.append(x)
    y = np.vstack(parts)
    y = y / max(1.0, np.abs(y).max() / 0.95)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.out, y, a.sr, subtype="PCM_16")
    print(f"wrote {a.out}: {len(y) / a.sr:.1f} s")


if __name__ == "__main__":
    main()
