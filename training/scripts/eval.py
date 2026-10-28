#!/usr/bin/env python3
"""
Сравнение на парах: PSNR входа и сети + картинки «вход | сеть | эталон» для глаз.

    python scripts/eval.py runs/m1/best.meow.json --data data/val --out runs/m1/eval --limit 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meow import weights  # noqa: E402
from meow.data import find_pairs, load_rgb, to_tensor  # noqa: E402
from meow.model import MeowRestore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("weights", type=Path, nargs="+", help=".meow.json (можно несколько — сравнить размеры)")
ap.add_argument("--data", type=Path, nargs="+", required=True)
ap.add_argument("--out", type=Path)
ap.add_argument("--limit", type=int, default=50)
ap.add_argument("--crop", type=int, default=360, help="высота вырезки для картинок сравнения")
args = ap.parse_args()

models = []
for w in args.weights:
    meta = weights.load(w)
    m = MeowRestore(meta["channels"], meta["layers"])
    weights.load_into_model(meta, m)
    models.append((meta["name"], m.eval()))

pairs = find_pairs(args.data)[: args.limit]
psnr = lambda a, b: 10 * np.log10(1 / max(np.mean((a - b) ** 2), 1e-10))
scores = {"вход": []} | {n: [] for n, _ in models}
if args.out:
    args.out.mkdir(parents=True, exist_ok=True)
for i, (lq_p, hq_p) in enumerate(pairs):
    lq, hq = to_tensor(load_rgb(lq_p))[None], load_rgb(hq_p).astype(np.float32) / 255
    row = [lq[0].numpy().transpose(1, 2, 0)]
    scores["вход"].append(psnr(row[0], hq))
    for name, m in models:
        with torch.no_grad():
            y = m(lq)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
        scores[name].append(psnr(y, hq))
        row.append(y)
    row.append(hq)
    if args.out and i < 12:
        h, w = hq.shape[:2]
        c = min(args.crop, h)
        y0, x0 = (h - c) // 2, (w - c) // 2
        tiles = [(r[y0 : y0 + c, x0 : x0 + c] * 255 + 0.5).clip(0, 255).astype(np.uint8) for r in row]
        Image.fromarray(np.concatenate(tiles, axis=1)).save(args.out / f"{i:03d}_{lq_p.stem}.png")

print(f"{'':24s} PSNR, дБ   выигрыш")
base = float(np.mean(scores["вход"]))
for k, v in scores.items():
    m = float(np.mean(v))
    print(f"{k:24s} {m:7.2f}   {m - base:+.2f}")
if args.out:
    print(f"картинки (вход | {' | '.join(n for n, _ in models)} | эталон): {args.out}")
