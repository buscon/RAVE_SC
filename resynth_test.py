#!/usr/bin/env python3
"""
Resynthesis test: recording -> RAVE encode -> decode, for several models and input levels.
Writes the original excerpt and the resynthesis (wav) and prints crude similarity numbers.
Usage: python resynth_test.py --audio audio/Seawaves1.wav --models models --out audio/resynth --start 5 --dur 12
"""
import argparse, re, warnings
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly, stft
warnings.filterwarnings("ignore")

def load(path):
    return torch.jit.load(str(path), map_location="cpu").eval()

@torch.no_grad()
def enc(m, x): return m.encode(torch.tensor(x, dtype=torch.float32)[None, None])
@torch.no_grad()
def dec(m, z): return m.decode(z)[0, 0].numpy()

def flush(m, sr):
    for _ in range(3):
        z = enc(m, np.zeros(sr)); dec(m, torch.zeros(1, z.shape[1], 40))

def rms(x): return float(np.sqrt((x ** 2).mean()))

def spec(x, sr):
    f, t, S = stft(x, sr, nperseg=2048, noverlap=1536)
    return f, np.abs(S) + 1e-9

def compare(a, b, sr):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    f, A = spec(a, sr); _, B = spec(b, sr)
    la, lb = np.log(A), np.log(B)
    # long-term spectrum (dB) correlation, per-frame level correlation, spectral flatness, share above 8 kHz
    lt = np.corrcoef(la.mean(1), lb.mean(1))[0, 1]
    ea, eb = np.log(A.mean(0)), np.log(B.mean(0))
    lev = np.corrcoef(ea, eb)[0, 1]
    # frame-by-frame spectral correlation (how well the moment-to-moment spectral shape is followed)
    fr = np.mean([np.corrcoef(la[:, i], lb[:, i])[0, 1] for i in range(0, la.shape[1], 4)])
    hf = lambda S: float((S[f > 8000] ** 2).sum() / (S ** 2).sum())
    return dict(longterm=lt, level=lev, frame=fr, hf_in=hf(A), hf_out=hf(B))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True); ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="resynth"); ap.add_argument("--start", type=float, default=5)
    ap.add_argument("--dur", type=float, default=12)
    ap.add_argument("--levels", default="0.01,0.03,0.06,0.1,0.2")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    x, fs = sf.read(a.audio, always_2d=True); x = x.mean(1)
    levels = [float(v) for v in a.levels.split(",")]
    for p in sorted(Path(a.models).glob("*.ts")):
        if a.only and a.only not in p.name: continue
        mm = re.search(r"r(\d{4,6})", p.name); sr = int(mm.group(1)) if mm else 48000
        g = gcd(int(fs), sr)
        y = resample_poly(x, sr // g, int(fs) // g)[int(a.start * sr):int((a.start + a.dur) * sr)].astype(np.float32)
        y = y / rms(y)
        m = load(p); best = None
        print(f"== {p.stem}  (sr {sr})", flush=True)
        for lv in levels:
            flush(m, sr)
            try:
                z = enc(m, y * lv); r = dec(m, z)
            except Exception as e:
                print("  error", str(e)[:80]); break
            if not np.isfinite(r).all(): print(f"  level {lv}: NaN"); flush(m, sr); continue
            r = r[int(0.5 * sr):]; ref = (y * lv)[int(0.5 * sr):len(r) + int(0.5 * sr)]
            if rms(r) < 1e-5: print(f"  level {lv}: silent"); continue
            c = compare(ref, r, sr)
            score = c["frame"]
            print(f"  in rms {lv:5.2f} -> out rms {rms(r):.4f} | long-term spectrum r={c['longterm']:.2f}  level-follow r={c['level']:.2f}  frame-spectrum r={c['frame']:.2f}  HF>8k in {c['hf_in']:.3f} out {c['hf_out']:.3f}", flush=True)
            if best is None or score > best[0]: best = (score, lv, r, ref)
        if best:
            _, lv, r, ref = best
            sf.write(str(out / f"{p.stem}_resynth_inrms{lv}.wav"), r / max(1, np.abs(r).max() / 0.9), sr)
            sf.write(str(out / f"{p.stem}_input_inrms{lv}.wav"), ref / max(1, np.abs(ref).max() / 0.9), sr)
            print(f"  -> saved best level {lv}")
if __name__ == "__main__":
    main()
