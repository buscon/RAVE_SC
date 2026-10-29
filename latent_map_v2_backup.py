#!/usr/bin/env python3
"""
Sound map for a RAVE model (TorchScript .ts file).

Every point on the map is a short audio clip with a latent vector (its embedding):
  - Default (no --audio): "decoder map". Latent vectors are sampled around the range that the encoder
    produces for a bank of synthetic sounds, held constant, and decoded to audio. The clips are what the
    DECODER produces. The sampling range is an assumption. Use --audio for real data.
  - --audio FOLDER: "recordings map". Your recordings are cut into clips; the embedding of a clip is the
    mean latent vector of its frames. The clips are the original sound.

The embeddings are grouped into --k clusters (k-means in latent space; default: chosen with the silhouette score). The number of
clusters is unrelated to the number of latent dimensions of the model.
For the 2D layout there are three views:
  islands  : clusters are placed as separate groups; the position of an island follows the distance between
             clusters in latent space, and points inside an island are laid out by a local PCA.
             This is a layout made to show the groups, it is not a neutral projection.
  t-SNE    : neighbourhood-preserving projection of the latent vectors.
  PCA      : linear projection, keeps the global shape.

Outputs in --out:  <name>_views.png, <name>.html (interactive islands), <name>_clips.flac (all clips, one
after the other), <name>_map.csv, <name>_map_groups.txt, <name>_map_info.txt (used by rave_sound_map.scd in
SuperCollider), <name>_scale.json.

Install:  pip install torch numpy scikit-learn matplotlib plotly soundfile scipy
Run:      python latent_map.py --model models/birds_dawnchorus_b2048_r48000_z8.ts
          python latent_map.py --model models/birds_pluma_b2048_r48000_z12.ts --audio ~/recordings
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- model helpers
def load_model(path):
    return torch.jit.load(str(path), map_location="cpu").eval()


def model_sr(path):
    m = re.search(r"r(\d{4,6})", Path(path).name)
    return int(m.group(1)) if m else 48000


@torch.no_grad()
def encode(model, x):
    """x: (T,) numpy -> (D, frames) numpy"""
    t = torch.tensor(x, dtype=torch.float32)[None, None]
    return model.encode(t)[0].numpy()


@torch.no_grad()
def decode(model, z):
    """z: (B, D, frames) numpy -> (B, T) numpy"""
    return model.decode(torch.tensor(z, dtype=torch.float32))[:, 0].numpy()


def latent_hop(model, sr):
    """Samples per latent frame (512 for these models, measured, not assumed)."""
    n = 2 * sr
    return int(round(n / encode(model, np.zeros(n)).shape[1]))


def flush(model, sr):
    """Streaming models keep convolution caches between calls, and a NaN can get stuck in them.
    Pushing silence through (for all 64 cache slots) lets the caches recover."""
    for _ in range(3):
        z = encode(model, np.zeros(sr))
        decode(model, np.zeros((64, z.shape[0], 40)))


# ---------------------------------------------------------------- synthetic bank
def synthetic_bank(sr, seed=0, dur=2.0):
    """Synthetic sounds at levels where RAVE bird models still respond.
    (Tests on birds_dawnchorus: noise above rms ~0.02 and loud tones decode to silence,
    chirps and bursts at rms 0.02-0.05 are reproduced.) Each item is scaled to a target rms."""
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    f = np.fft.rfftfreq(n, 1 / sr)
    bank = []

    def add(x, rms):
        bank.append(x / x.std() * rms)

    for r in (0.002, 0.005):
        add(rng.standard_normal(n), r)
    pink = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.maximum(f, 20.0)), n)
    add(pink, 0.008)
    for f0, f1 in ((500, 3000), (2000, 6000), (6000, 2000), (300, 1500), (3000, 8000)):
        ph = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * dur))
        for r in (0.02, 0.05):
            add(np.sin(ph), r)
    for f0 in (1200, 3000):
        sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        add(sig, 0.01)
    env = np.exp(-((t * 6) % 1.0) * 8)
    add(rng.standard_normal(n) * env, 0.02)
    return bank


# ---------------------------------------------------------------- features and clusters
def features(y, sr):
    w = np.hanning(y.shape[1])
    S = np.abs(np.fft.rfft(y * w, axis=1)) ** 2 + 1e-18
    f = np.fft.rfftfreq(y.shape[1], 1 / sr)
    P = S / S.sum(1, keepdims=True)
    cen = (P * f).sum(1)
    flat = np.exp(np.log(S).mean(1)) / S.mean(1)
    rms = np.sqrt((y**2).mean(1))
    return {"rms": rms, "centroid": cen, "flatness": flat}


def name_clusters(labels, feat):
    """Names are relative to the other clusters (brightness in thirds, tonality above/below the median)."""
    q1, q2 = np.quantile(feat["centroid"], [1 / 3, 2 / 3])
    flat_med = np.median(feat["flatness"])
    names = {}
    for c in np.unique(labels):
        sel = labels == c
        cen = np.median(feat["centroid"][sel])
        fl = np.median(feat["flatness"][sel])
        bright = "dark" if cen < q1 else ("mid" if cen < q2 else "bright")
        tone = "less tonal" if fl > flat_med else "more tonal"
        names[int(c)] = f"{c}: {bright}, {tone} (~{int(cen)} Hz, n={int(sel.sum())})"
    return names


# ---------------------------------------------------------------- 2D layouts
def tsne(X, seed):
    from sklearn.manifold import TSNE

    return TSNE(n_components=2, perplexity=min(30, max(5, len(X) // 50)), init="pca",
                random_state=seed).fit_transform(X)


def pca2(X):
    from sklearn.decomposition import PCA

    return PCA(2).fit_transform(X)


def islands(zn, labels):
    """Clusters as separate groups. Island centres: classical MDS of the cluster centres in latent space,
    pushed apart until they do not overlap. Inside an island: local PCA, scaled to the island radius."""
    ks = sorted(set(labels.tolist()))
    n = len(ks)
    cents = np.array([zn[labels == k].mean(0) for k in ks])
    D2 = ((cents[:, None] - cents[None]) ** 2).sum(-1)
    J = np.eye(n) - 1.0 / n
    w, v = np.linalg.eigh(-0.5 * J @ D2 @ J)
    order = np.argsort(w)[::-1][:2]
    pos = v[:, order] * np.sqrt(np.maximum(w[order], 0))
    if n < 3 or np.abs(pos).max() == 0:
        pos = np.stack([np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False)),
                        np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))], 1)
    pos = pos / np.abs(pos).max() * 2.0
    counts = np.array([(labels == k).sum() for k in ks], dtype=float)
    r = 0.45 * np.sqrt(counts / counts.max()) + 0.1
    for _ in range(800):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d) + 1e-9
                need = r[i] + r[j] + 0.12
                if dist < need:
                    push = (need - dist) / 2 * d / dist
                    pos[i] += push
                    pos[j] -= push
                    moved = True
        if not moved:
            break
    xy = np.zeros((len(zn), 2))
    for i, k in enumerate(ks):
        sel = labels == k
        p = zn[sel] - zn[sel].mean(0)
        if len(p) >= 2:
            _, _, vt = np.linalg.svd(p, full_matrices=False)
            q = p @ vt[:2].T
            if q.shape[1] < 2:
                q = np.pad(q, ((0, 0), (0, 2 - q.shape[1])))
        else:
            q = np.zeros((len(p), 2))
        rad = np.percentile(np.linalg.norm(q, axis=1), 95) + 1e-9
        q = q / rad * r[i] * 0.95
        nrm = np.linalg.norm(q, axis=1)
        big = nrm > r[i]
        q[big] *= (r[i] / nrm[big])[:, None]
        xy[sel] = pos[i] + q
    return xy


def choose_k(zn, seed, fallback=8, kmin=2, kmax=12):
    """Pick the number of clusters with the silhouette score (k-means, k = kmin..kmax).
    Silhouette below ~0.25 means there is no substantial cluster structure (Kaufman & Rousseeuw);
    then every k is arbitrary, and the fallback is used. A structureless Gaussian cloud of the same
    size is scored too, for comparison."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    sample = zn if len(zn) <= 1500 else zn[rng.choice(len(zn), 1500, replace=False)]
    kmax = min(kmax, len(sample) - 1)
    scores = {}
    for k in range(kmin, kmax + 1):
        scores[k] = float(silhouette_score(sample, KMeans(k, n_init=5, random_state=seed).fit_predict(sample)))
    best = max(scores, key=scores.get)
    ref = rng.standard_normal(sample.shape)
    ref_score = float(silhouette_score(ref, KMeans(best, n_init=5, random_state=seed).fit_predict(ref)))
    print("silhouette by k:", {k: round(v, 3) for k, v in scores.items()})
    print(f"best: k={best} ({scores[best]:.3f}); structureless Gaussian cloud at that k: {ref_score:.3f}")
    clear = scores[best] >= 0.25
    if clear:
        print(f"-> clear enough cluster structure, using k={best}")
        return best, scores, ref_score, True
    print(f"-> NO clear cluster structure (silhouette < 0.25), so any k is arbitrary; using k={fallback}")
    return fallback, scores, ref_score, False


# ---------------------------------------------------------------- output
def plot_views(out, stem, views, labels, names, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    fig, axes = plt.subplots(1, len(views), figsize=(6.2 * len(views), 6.2), squeeze=False)
    for ax, (vn, xy) in zip(axes[0], views.items()):
        for g in np.unique(labels):
            sel = labels == g
            ax.scatter(xy[sel, 0], xy[sel, 1], s=7, color=cmap(g % cmap.N), label=names[g], alpha=0.8)
            cx, cy = np.median(xy[sel], 0)
            ax.text(cx, cy, str(g), fontsize=13, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_title(vn)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal" if vn == "islands" else "auto")
    axes[0][0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, markerscale=1.6, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    png = out / f"{stem}_views.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def plot_html(out, stem, xy, labels, names, hover, title):
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    pf = go.Figure()
    for g in np.unique(labels):
        sel = np.where(labels == g)[0]
        col = "rgb(%d,%d,%d)" % tuple(int(255 * c) for c in cmap(g % cmap.N)[:3])
        pf.add_trace(go.Scattergl(x=xy[sel, 0], y=xy[sel, 1], mode="markers", name=names[g],
                                  marker=dict(size=6, color=col, opacity=0.8),
                                  text=[hover[i] for i in sel], hoverinfo="text"))
    pf.update_layout(title=title, template="plotly_white", xaxis=dict(visible=False),
                     yaxis=dict(visible=False, scaleanchor="x"), width=950, height=760)
    html = out / f"{stem}.html"
    pf.write_html(str(html), include_plotlyjs="cdn")
    return html


def export(out, stem, views, labels, feat, Z, clips, sr, names, info):
    import soundfile as sf

    rms_db = 20 * np.log10(feat["rms"] + 1e-6)
    rows = np.column_stack([views["islands"], views["t-SNE"], views["PCA"], labels,
                            feat["centroid"], rms_db, Z])
    np.savetxt(out / f"{stem}_map.csv", rows, delimiter=",", fmt="%.4f")

    # one file with all clips one after the other, same order as the CSV rows
    peaks = np.abs(clips).max(1)
    gain = 0.9 / (np.percentile(peaks, 99) + 1e-9)   # one gain for all clips: keeps the level differences
    flat = np.clip(clips * gain, -1, 1).reshape(-1)
    sf.write(str(out / f"{stem}_clips.flac"), flat, sr, subtype="PCM_16")

    lines = [f"#clip_samples|{clips.shape[1]}", f"#sr|{sr}", f"#audio|{stem}_clips.flac"]
    lines += [f"{g}|{n}" for g, n in sorted(names.items())]
    (out / f"{stem}_map_groups.txt").write_text("\n".join(lines) + "\n")
    (out / f"{stem}_map_info.txt").write_text("\n".join(info) + "\n")


# ---------------------------------------------------------------- the two ways to get clips
def calibrated_latents(model, sr, hop, seed):
    """Encode the synthetic bank at several input levels and keep the latent frames of the (signal, level)
    pairs that the model reproduces audibly. RAVE models only respond to inputs in the range they were
    trained on, so the level that works differs between models."""
    levels = (0.003, 0.01, 0.03, 0.1, 0.3)
    skip = int(0.2 * sr / hop) + 1
    frames, passed, total = [], 0, 0
    for x in synthetic_bank(sr, seed):
        x = x / x.std()
        for lv in levels:
            total += 1
            z = encode(model, x * lv)
            y = decode(model, z[None])[0][int(0.3 * sr):]
            if not (np.isfinite(y).all() and np.isfinite(z).all()):
                flush(model, sr)
                continue
            if y.std() > 5e-4 and y.std() > 0.1 * lv:
                frames.append(z[:, skip:].T)
                passed += 1
    print(f"{passed} of {total} (signal, level) pairs are reproduced audibly by the model")
    return (np.concatenate(frames, 0) if frames else None), passed, total


def decoder_clips(model, sr, hop, args):
    L = int(args.clip * sr)
    F, passed, total = calibrated_latents(model, sr, hop, args.seed)
    if F is None or passed < 5:
        raise SystemExit("This model does not respond to the synthetic test sounds, so a decoder map cannot be "
                         "built. Use --audio with recordings that fit the model.")
    D = F.shape[1]   # the real number of latent channels (model.latent_size can be larger in v1 exports)
    p16, p50, p84 = np.percentile(F, [16, 50, 84], axis=0)
    centre, scale = p50, (p84 - p16) / 2 + 1e-6   # robust: a few extreme frames do not set the range
    print("latent centre:", np.round(centre, 1))
    print("latent scale :", np.round(scale, 1))

    rng = np.random.default_rng(args.seed)
    z = centre + np.clip(rng.standard_normal((args.n, D)), -2.5, 2.5) * scale
    hold = int(np.ceil((L + int(0.4 * sr)) / hop))   # 0.4 s warm-up for the streaming decoder, then the clip
    ys = []
    for i in range(0, args.n, 64):   # the streaming model caches state for at most 64 batch items
        zz = np.repeat(z[i:i + 64, :, None], hold, axis=2)
        out = decode(model, zz)[:, -L:]
        bad = ~np.isfinite(out).all(1)
        if bad.any():
            out[bad] = 0.0
            flush(model, sr)
        ys.append(out)
    clips = np.concatenate(ys, 0).astype(np.float32)
    info = [f"decoded latent point {i}" for i in range(args.n)]
    return clips, z, info, centre, scale


def audio_clips(model, sr, hop, args):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    L = int(args.clip * sr)
    files = sorted(p for p in Path(args.audio).expanduser().rglob("*")
                   if p.suffix.lower() in (".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"))
    if not files:
        raise SystemExit(f"No audio files found in {args.audio}")
    clips, Z, info = [], [], []
    for p in files:
        x, fs = sf.read(str(p), always_2d=True)
        x = x.mean(1)
        if fs != sr:
            g = gcd(int(fs), sr)
            x = resample_poly(x, sr // g, int(fs) // g)
        x = x[: sr * args.max_seconds].astype(np.float32)
        z = encode(model, x)
        hop_f = len(x) / z.shape[1]
        for s0 in range(int(0.1 * sr), len(x) - L, L):
            f0 = max(int(s0 / hop_f), 4)
            f1 = max(int((s0 + L) / hop_f), f0 + 1)
            Z.append(z[:, f0:f1].mean(1))
            clips.append(x[s0:s0 + L])
            info.append(f"{p.parent.name}/{p.name} @ {s0 / sr:.1f} s")
        print(f"{p.name}: done")
    clips, Z = np.array(clips, dtype=np.float32), np.array(Z)
    if len(clips) > args.n:
        idx = np.sort(np.random.default_rng(args.seed).choice(len(clips), args.n, replace=False))
        clips, Z, info = clips[idx], Z[idx], [info[i] for i in idx]
    return clips, Z, info, Z.mean(0), Z.std(0) + 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="latent_maps")
    ap.add_argument("--audio", default=None, help="folder of recordings (recordings map)")
    ap.add_argument("--n", type=int, default=1000, help="number of points / clips")
    ap.add_argument("--k", default="auto", help="number of clusters: an integer, or auto (silhouette score; falls back to 8 if there is no clear structure)")
    ap.add_argument("--clip", type=float, default=0.3, help="clip length in seconds")
    ap.add_argument("--max-seconds", type=int, default=300, help="recordings map: max seconds read per file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="text added to the output names, e.g. _1.0s")
    args = ap.parse_args()

    from sklearn.cluster import KMeans

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)
    sr = model_sr(args.model)
    hop = latent_hop(model, sr)
    print(f"sample rate {sr}, latent frame = {hop} samples = {1000 * hop / sr:.1f} ms")
    stem = Path(args.model).stem + ("_audio" if args.audio else "_decoder") + args.suffix

    clips, Z, info, centre, scale = (audio_clips if args.audio else decoder_clips)(model, sr, hop, args)
    feat = features(clips, sr)
    print(f"{len(clips)} clips, {(feat['rms'] < 1e-4).sum()} of them nearly silent (rms < 1e-4)")

    zn = (Z - centre) / scale
    if str(args.k) == "auto":
        k, k_scores, k_ref, k_clear = choose_k(zn, args.seed)
    else:
        k, k_scores, k_ref, k_clear = int(args.k), {}, None, None
    args.k = k
    labels = KMeans(k, n_init=10, random_state=args.seed).fit_predict(zn)
    names = name_clusters(labels, feat)
    views = {"islands": islands(zn, labels), "t-SNE": tsne(zn, args.seed), "PCA": pca2(zn)}

    what = "decoded latent points" if not args.audio else "clips of recordings"
    title = f"{stem}: {len(clips)} {what}, {args.k} clusters of the latent vectors (colour = cluster)"
    png = plot_views(out, stem, views, labels, names, title)
    rms_db = 20 * np.log10(feat["rms"] + 1e-6)
    hover = [f"{info[i]}<br>cluster {labels[i]}<br>brightness {feat['centroid'][i]:.0f} Hz<br>level {rms_db[i]:.0f} dB"
             for i in range(len(clips))]
    html = plot_html(out, stem, views["islands"], labels, names, hover, title)
    export(out, stem, views, labels, feat, Z, clips, sr, names, info)
    (out / f"{stem}_scale.json").write_text(json.dumps(
        {"model": Path(args.model).name, "sample_rate": sr, "latent_hop": hop,
         "center": centre.round(2).tolist(), "scale": scale.round(2).tolist(),
         "k": k, "k_chosen_by_silhouette": k_clear, "silhouette": {str(a): round(b, 3) for a, b in k_scores.items()},
         "silhouette_structureless_reference": k_ref}, indent=1))
    print("wrote", png, html)
    for g in sorted(names):
        print("  ", names[g])


if __name__ == "__main__":
    main()
