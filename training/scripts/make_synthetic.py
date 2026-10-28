#!/usr/bin/env python3
"""
Синтетическое «мультяшное» видео (заливки, тёмные контуры, градиенты, тонкие линии, движение) —
ТОЛЬКО для проверки, что весь конвейер работает (make_pairs → train → export), когда под рукой нет
аниме-исходников. Для настоящей модели нужны настоящие кадры (docs/TRAINING.md).

    python scripts/make_synthetic.py --out synthetic.mkv --seconds 20
"""
import argparse
import math
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

W, H = args.width, args.height
rng = np.random.default_rng(args.seed)
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
     "-i", "-", "-c:v", "ffv1", args.out],
    stdin=subprocess.PIPE,
)


def palette(n):
    return [tuple(int(c) for c in rng.integers(40, 250, 3)) for _ in range(n)]


scene_len = args.fps * 4
frames = int(args.seconds * args.fps)
for f in range(frames):
    if f % scene_len == 0:  # новая «сцена»
        sky = palette(2)
        shapes = [
            dict(kind=rng.choice(["ellipse", "rect", "poly"]), x=rng.uniform(0, W), y=rng.uniform(0, H),
                 r=rng.uniform(H * 0.04, H * 0.22), col=palette(1)[0], shade=palette(1)[0],
                 vx=rng.uniform(-6, 6), vy=rng.uniform(-3, 3), rot=rng.uniform(0, 6.28), n=int(rng.integers(3, 7)))
            for _ in range(int(rng.integers(8, 18)))
        ]
        hair = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(-1, 1), rng.uniform(H * 0.05, H * 0.25)) for _ in range(40)]
    t = f % scene_len
    grad = np.linspace(0, 1, H)[:, None, None]
    bg = (np.array(sky[0]) * (1 - grad) + np.array(sky[1]) * grad).astype(np.uint8)
    img = Image.fromarray(np.broadcast_to(bg, (H, W, 3)).copy())
    d = ImageDraw.Draw(img)
    lw = max(2, H // 300)
    for s in shapes:
        x, y, r = s["x"] + s["vx"] * t, s["y"] + s["vy"] * t, s["r"]
        if s["kind"] == "ellipse":
            d.ellipse([x - r, y - r * 0.8, x + r, y + r * 0.8], fill=s["col"], outline=(25, 18, 30), width=lw)
            d.chord([x - r, y - r * 0.8, x + r, y + r * 0.8], 20, 160, fill=s["shade"])  # «тень» в стиле cel-shading
        elif s["kind"] == "rect":
            d.rounded_rectangle([x - r, y - r * 0.6, x + r, y + r * 0.6], radius=r * 0.2, fill=s["col"], outline=(25, 18, 30), width=lw)
        else:
            pts = [(x + r * math.cos(s["rot"] + t * 0.01 + k * 6.283 / s["n"]), y + r * math.sin(s["rot"] + t * 0.01 + k * 6.283 / s["n"])) for k in range(s["n"])]
            d.polygon(pts, fill=s["col"], outline=(25, 18, 30), width=lw)
    for hx, hy, k, L in hair:  # тонкие штрихи — как волосы и детали, на них сжатие ошибается сильнее всего
        hx2 = hx + L * k + 3 * math.sin(t * 0.1)
        d.line([(hx, hy), (hx2, hy + L)], fill=(30, 20, 35), width=max(1, lw // 2))
    img = img.filter(ImageFilter.SMOOTH)  # сглаживание контуров, как у отрисованной анимации
    ff.stdin.write(img.tobytes())
ff.stdin.close()
ff.wait()
print(f"{args.out}: {frames} кадров {W}x{H}")
