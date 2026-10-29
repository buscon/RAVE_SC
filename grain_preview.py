"""Offline preview of the grain behaviours of rave_grains.scd (same pools, same value mapping, no RAVE).
Usage: python grain_preview.py audio.wav out_dir"""
import sys, numpy as np, soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt
sys.path.insert(0, str(Path(__file__).parent))
from make_corpus import analyse, onset_times

def ranks(v):
    v = np.asarray(v); o = np.argsort(v, kind="stable"); r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 / max(len(v) - 1, 1); i = j + 1
    return r

wav, outdir = sys.argv[1], Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
x, sr = sf.read(wav, always_2d=True); xm = x.mean(1); dur = len(xm) / sr
on = onset_times(xm, sr); L = 0.5
rows = np.array([analyse(xm, sr, s, L, on)[0] for s in np.arange(0, dur - L, L * 0.25)])
n = len(rows); rAct, rOn, rRms = ranks(rows[:, 6]), ranks(rows[:, 9]), ranks(rows[:, 2])
allv = np.arange(n)
pad = allv[(rAct <= 0.6) & (rOn <= 0.7)]
if len(pad) < 20: pad = allv[np.argsort(rAct)][:max(20, n // 3)]
sc = rAct + rOn + 0.5 * rRms
perc = allv[np.argsort(-sc)][:max(20, int(n * 0.35))]
padkey, perckey = ranks(rows[pad, 8]), ranks(rows[perc, 2])
rng = np.random.default_rng(1)

def params(mode, v):
    e = lambda a, b: a * (b / a) ** v
    if mode == 0: rate, glen, cut, db = e(1.5, 9), 1.6, e(500, 14000), -26 + 22 * v
    else: rate, glen, cut, db = e(0.6, 14), e(0.5, 0.1), e(3000, 16000), -16 + 13 * v
    ov = rate * glen
    gain = 1 / np.sqrt(max(1, ov * .5)) if mode == 0 else 0.8
    return rate, glen, cut, db, gain, ov

def pick(mode, v, recent):
    idx, key, w = (pad, padkey, .15) if mode == 0 else (perc, perckey, .35)
    near = np.where(np.abs(key - v) <= w)[0]
    if len(near) < 6: near = np.argsort(np.abs(key - v))[:6]
    c = [q for q in near if not any(abs(rows[idx[q], 0] - rows[r, 0]) < .3 for r in recent)] or list(near)
    j = idx[rng.choice(c)]; recent.append(j); del recent[:-8]; return j

def grain(i, p, mode, ch):
    rate, glen, cut, db, gain, ov = p
    r = rows[i]; off = rng.uniform(0, max(r[1] - glen, 0)); s0 = int((r[0] + off) * sr); n = int(glen * sr)
    g = x[s0:s0 + n]
    if len(g) < n: g = np.vstack([g, np.zeros((n - len(g), g.shape[1]))])
    g = g[:, :2] if g.shape[1] >= 2 else np.repeat(g, 2, 1)
    t = np.arange(n) / sr
    if mode == 0:
        a = glen / 2; env = np.where(t < a, np.sin(0.5 * np.pi * t / a), np.cos(0.5 * np.pi * (t - a) / (glen - a))) ** 1
    else:
        a = 0.004; env = np.where(t < a, t / a, np.exp(-6 * (t - a) / (glen - a)) * (1 - (t - a) / (glen - a)))
    norm = np.clip(10 ** (((-24 if mode == 0 else -20) - r[2]) / 20), .2, 6)
    pan = rng.uniform(-.6, .6) if mode == 0 else rng.uniform(-.8, .8)
    return g * env[:, None] * gain * norm * rng.uniform(.8, 1) * np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)])

def render(mode, v, secs=12.0):
    out = np.zeros((int(secs * sr) + 4 * sr, 2)); t = 0.0; recent = []; cnt = 0
    p = params(mode, v)
    while t < secs:
        i = pick(mode, v, recent); g = grain(i, p, mode, x.shape[1]); k = int(t * sr); out[k:k + len(g)] += g; cnt += 1
        if mode == 1 and v > .5 and rng.random() < (v - .5) * 1.4:
            j = pick(mode, v, recent); g2 = grain(j, p, mode, x.shape[1]); k2 = k + int(rng.uniform(.01, .04) * sr); out[k2:k2 + len(g2)] += g2
        t += max(.01, 1 / p[0] * rng.uniform(.5, 1.5))
    out = out[int(2 * sr):int(secs * sr)]
    sos = butter(2, p[2] / (sr / 2), output="sos"); out = sosfilt(sos, out, axis=0) * 10 ** ((p[3] + (14 if mode == 0 else 10)) / 20)
    return out

def stats(y):
    m = y.mean(1); rms = 20 * np.log10(np.sqrt((m ** 2).mean()) + 1e-9)
    S = np.abs(np.fft.rfft(m * np.hanning(len(m)))); f = np.fft.rfftfreq(len(m), 1 / sr)
    return rms, (S * f).sum() / S.sum(), len(onset_times(m, sr)) / (len(m) / sr)

stem = Path(wav).stem[:14]
for mode, nm in ((0, "pad"), (1, "perc")):
    for v in (0.1, 0.5, 0.9):
        y = render(mode, v)
        rms, cen, evs = stats(y)
        print(f"{stem} {nm:4s} v={v}: level {rms:6.1f} dB | centroid {cen:6.0f} Hz | events {evs:4.1f}/s | grain rate {params(mode, v)[0]:.1f}/s, length {params(mode, v)[1]*1000:.0f} ms")
        sf.write(outdir / f"{stem}_{nm}_v{v}.wav", np.clip(y, -0.9, 0.9), sr, subtype="PCM_16")
        print(f"      peak {np.abs(y).max():.2f}")
