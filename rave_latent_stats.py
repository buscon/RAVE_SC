#!/usr/bin/env python3
"""
Mean and spread (standard deviation over time) of each latent dimension when a recording goes through a model's encoder.
rave_grains.scd uses them in the source "model only": the decoder is driven by a slow random walk around the mean, with
the size of the walk set by the spread, so that the latent vectors stay in the region where this recording lives.

Usage: python rave_latent_stats.py models/<model>.ts audio/<recording>.wav
Writes directions/<model>__<recording>.stats.csv with two lines: the mean, then the standard deviation.
"""
import argparse, re, warnings
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("audio")
ap.add_argument("--dur", type=float, default=90); ap.add_argument("--level", type=float, default=0.06)
ap.add_argument("--out", default="directions")
a = ap.parse_args()
p = Path(a.model); m = torch.jit.load(str(p), map_location="cpu").eval()
mm = re.search(r"r(\d{4,6})", p.name); sr = int(mm.group(1)) if mm else 48000
x, fs = sf.read(a.audio, always_2d=True); x = x.mean(1); g = gcd(int(fs), sr)
x = resample_poly(x, sr // g, int(fs) // g)[: int(a.dur * sr)].astype(np.float32)
x = x / np.sqrt((x ** 2).mean()) * a.level
for _ in range(3):
    z = m.encode(torch.zeros(1, 1, sr)); m.decode(torch.zeros(1, z.shape[1], 40))
z = m.encode(torch.tensor(x)[None, None])[0].numpy()      # (dimensions, frames)
z = z[:, 10:]                                              # skip the start-up frames
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
f = out / f"{p.stem}__{Path(a.audio).stem}.stats.csv"
np.savetxt(f, np.vstack([z.mean(1), z.std(1)]), delimiter=",", fmt="%.5f")
print(f"{p.stem} / {Path(a.audio).stem}: {z.shape[0]} dimensions, {z.shape[1]} frames, mean spread {z.std(1).mean():.2f} -> {f.name}")
