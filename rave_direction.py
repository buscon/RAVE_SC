#!/usr/bin/env python3
"""
Finds, for a RAVE model, the latent direction along which the decoded sound gets louder and brighter, so that the
"latent shift by the value" in rave_grains.scd goes the same way for every model.

How: the recording goes through the encoder. For each latent dimension i the decoder is run again with that dimension
shifted by +delta, and the change of the output (loudness in dB, spectral centroid in octaves) is measured. The
two changes are scaled to the same size over all dimensions and added; the result is the direction.
Models react with very different strength (a shift of 1 can change the level by 3 dB or by 60 dB), so the direction is then
scaled: the scale is chosen so that going from -1 to +1 along it changes "loudness in dB + 6 * octaves of centroid" by about
12, as long as that change rises steadily (-1, 0, +1). A shift of 1 in rave_grains.scd then means about the same strength for
every model. If no scale rises steadily, the model is marked WEAK and the smallest scale is kept.

Usage: python rave_direction.py models/water_pondbrain_b2048_r48000_z16.ts audio/Seawaves1.wav
Writes directions/<model name>.csv (one line, one number per latent dimension) and prints the check.
"""
import argparse, re, warnings
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

def flush(m, sr):
    for _ in range(3):
        z = m.encode(torch.zeros(1, 1, sr)); m.decode(torch.zeros(1, z.shape[1], 40))

def measure(y, sr):
    y = y[int(0.5 * sr):]
    rms = np.sqrt((y ** 2).mean()) + 1e-9
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)))); f = np.fft.rfftfreq(len(y), 1 / sr)
    return 20 * np.log10(rms), float((S * f).sum() / S.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("audio")
    ap.add_argument("--start", type=float, default=5); ap.add_argument("--dur", type=float, default=6)
    ap.add_argument("--level", type=float, default=0.06); ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--out", default="directions"); ap.add_argument("--random", type=int, default=0)
    a = ap.parse_args()
    p = Path(a.model); m = torch.jit.load(str(p), map_location="cpu").eval()
    mm = re.search(r"r(\d{4,6})", p.name); sr = int(mm.group(1)) if mm else 48000
    x, fs = sf.read(a.audio, always_2d=True); x = x.mean(1); g = gcd(int(fs), sr)
    x = resample_poly(x, sr // g, int(fs) // g)[int(a.start * sr):int((a.start + a.dur) * sr)].astype(np.float32)
    x = x / np.sqrt((x ** 2).mean()) * a.level
    def run(shift):
        flush(m, sr)
        z = m.encode(torch.tensor(x)[None, None])
        if shift is not None: z = z + torch.tensor(shift, dtype=torch.float32)[None, :, None]
        return m.decode(z)[0, 0].numpy()
    L0, C0 = measure(run(None), sr); Z = m.encode(torch.tensor(x)[None, None]).shape[1]
    dl, dc = np.zeros(Z), np.zeros(Z)
    for i in range(Z):
        s = np.zeros(Z); s[i] = a.delta
        L, C = measure(run(s), sr); dl[i] = L - L0; dc[i] = np.log2(C / C0)
    e = dl / (dl.std() + 1e-9) + dc / (dc.std() + 1e-9)
    d = e / np.sqrt((e ** 2).mean())
    comp = lambda L, C: L + 6 * np.log2(C / C0)
    best = None
    for c in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
        (Lm, Cm), (Lp, Cp) = measure(run(-c * d), sr), measure(run(c * d), sr)
        im, i0, ip = comp(Lm, Cm), comp(L0, C0), comp(Lp, Cp)
        if im < i0 < ip:
            err = abs((ip - im) - 12)
            if best is None or err < best[0]: best = (err, c, Lp - Lm, Cp / Cm)
    if best is None:
        print("   WEAK: no scale rises steadily; keeping the smallest"); c0 = 0.05
    else:
        c0 = best[1]; print(f"   chosen scale {c0}: -1 -> +1 changes loudness by {best[2]:.1f} dB, centroid x{best[3]:.2f}")
    d = d * c0
    steps = np.array([-2, -1, 0, 1, 2.0])
    def curve(dirv):
        L, C = zip(*[measure(run(dirv * s), sr) for s in steps]); return np.array(L), np.array(C)
    Ld, Cd = curve(d)
    print(f"== {p.stem}: latent size {Z}, input rms {a.level}, output {L0:.1f} dB, centroid {C0:.0f} Hz")
    print("   response direction, shift -2 -1 0 +1 +2: loudness dB", np.round(Ld, 1), "| centroid Hz", np.round(Cd, 0))
    rng = np.random.default_rng(0); up = 0; mono = 0; rows = []
    for _ in range(a.random):
        r = rng.standard_normal(Z); r /= np.sqrt((r ** 2).mean())
        L, C = curve(r); rows.append((L[-1] - L[0], C[-1] / C[0]))
        up += (L[-1] > L[0]); mono += bool(np.all(np.diff(L) > 0))
    rows = np.array(rows)
    if a.random: print(f"   {a.random} random directions: loudness gets higher from -2 to +2 in {up}, steadily rising in {mono}; "
          f"change -2 -> +2: {rows[:,0].min():.1f} .. {rows[:,0].max():.1f} dB (median {np.median(rows[:,0]):.1f}), centroid ratio {rows[:,1].min():.2f} .. {rows[:,1].max():.2f}")
    print(f"   scaled direction: change -2 -> +2: {Ld[-1]-Ld[0]:.1f} dB, centroid ratio {Cd[-1]/Cd[0]:.2f}; steadily rising loudness: {bool(np.all(np.diff(Ld) > 0))}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / (p.stem + ".csv"), d[None], delimiter=",", fmt="%.5f")
    print(f"   written {out / (p.stem + '.csv')}")

if __name__ == "__main__":
    main()
