#!/usr/bin/env python3
"""
Пары для обучения из видео высокого качества (BD/WEB-DL): чистый кадр + тот же кадр, испорченный так,
как портят аниме стриминги (пересжатие H.264, мыло от апскейла 480→720, размытие).

    python scripts/make_pairs.py video1.mkv video2.mkv --out data/bd --height 720 --segments 20

Для каждого видео берётся N случайных отрезков по 4 с:
  1. чистый отрезок → lossless (ffv1) в нужной высоте (lanczos) — это hq;
  2. он же → случайные порчи → libx264 с случайным CRF — это lq;
  3. из обоих извлекаются одни и те же номера кадров (покадрово совпадают, x264 кадры не выкидывает).

Нужен ffmpeg с libx264.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def degrade_filter(height: int, rng: random.Random) -> tuple[str, int, str]:
    """Случайная порча: (фильтр ffmpeg, CRF, описание)."""
    parts = []
    desc = []
    r = rng.random()
    if r < 0.45:
        # Исходник стриминга был ниже разрешением и растянут (самый частый случай: 480p/540p → 720p).
        f = rng.choice([0.5, 0.6, 0.67, 0.75, 0.85])
        lo = max(2, int(height * f) // 2 * 2)
        flags = rng.choice(["bicubic", "bilinear", "lanczos"])
        parts.append(f"scale=-2:{lo}:flags=area,scale=-2:{height}:flags={flags}")
        desc.append(f"rescale{f}/{flags}")
    if rng.random() < 0.3:
        sigma = rng.choice([0.4, 0.6, 0.8])
        parts.append(f"gblur=sigma={sigma}")
        desc.append(f"blur{sigma}")
    if rng.random() < 0.15:
        parts.append("noise=alls=4:allf=t")
        desc.append("noise")
    crf = rng.randint(23, 33)
    desc.append(f"crf{crf}")
    return (",".join(parts) or "null"), crf, "+".join(desc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=int, default=720, help="разрешение, в котором сеть будет работать на сайте")
    ap.add_argument("--segments", type=int, default=12, help="отрезков на видео")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--every", type=int, default=12, help="брать каждый N-й кадр отрезка")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    (args.out / "hq").mkdir(parents=True, exist_ok=True)
    (args.out / "lq").mkdir(parents=True, exist_ok=True)
    log = []
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        for vi, video in enumerate(args.videos):
            dur = ffprobe_duration(video)
            for si in range(args.segments):
                start = rng.uniform(0, max(0.0, dur - args.seconds - 1))
                clean = tmp / "clean.mkv"
                lossy = tmp / "lossy.mp4"
                run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", str(args.seconds), "-i", str(video), "-an", "-sn",
                     "-vf", f"scale=-2:{args.height}:flags=lanczos,format=yuv420p", "-c:v", "ffv1", str(clean)])
                vf, crf, desc = degrade_filter(args.height, rng)
                run(["ffmpeg", "-y", "-i", str(clean), "-vf", f"{vf},format=yuv420p", "-c:v", "libx264",
                     "-preset", rng.choice(["veryfast", "medium"]), "-crf", str(crf), str(lossy)])
                prefix = f"v{vi:03d}s{si:03d}"
                select = f"select='not(mod(n\\,{args.every}))'"
                for src, sub in ((clean, "hq"), (lossy, "lq")):
                    run(["ffmpeg", "-y", "-i", str(src), "-vf", select, "-fps_mode", "passthrough",
                         str(args.out / sub / f"{prefix}_%04d.png")])
                n_hq = len(list((args.out / "hq").glob(f"{prefix}_*.png")))
                n_lq = len(list((args.out / "lq").glob(f"{prefix}_*.png")))
                if n_hq != n_lq:
                    raise SystemExit(f"{video} отрезок {si}: кадров {n_hq} vs {n_lq} — пары разъехались")
                log.append({"video": str(video), "start": round(start, 2), "degrade": desc, "frames": n_hq})
                print(f"{video.name} [{si + 1}/{args.segments}] {desc}: {n_hq} пар")
    (args.out / "pairs.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
