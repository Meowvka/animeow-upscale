#!/usr/bin/env python3
"""
Пары «реальный кадр стриминга → он же, очищенный Real-ESRGAN» — когда чистых исходников (BD) нет.

Учитель — Real-ESRGAN AnimeVideo-v3 (models/realesrgan/*.onnx): кадр ×4 → обратно ×¼ (усреднение).
Получается тот же размер, но без блоков сжатия и с ровными контурами. Маленькая сеть учится
повторять это в реальном времени (дистилляция).

    python scripts/make_teacher.py frames/*.png --out data/teacher
    python scripts/make_teacher.py --video episode.mp4 --every 240 --skip-start 120 --skip-end 150 --out data/teacher

Кадры из видео достаются ffmpeg-ом; на GPU-машине поставьте onnxruntime-gpu — будет в десятки раз быстрее.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "realesrgan" / "animevideov3-x4.fp32.onnx"


def session(model: Path) -> ort.InferenceSession:
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # onnxruntime-gpu[cuda,cudnn]: CUDA/cuDNN из pip-пакетов
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(model), providers=providers)
    used = sess.get_providers()
    print(f"учитель: {used[0]}")
    if used[0] == "CPUExecutionProvider":
        print("  ВНИМАНИЕ: CUDA недоступна — учитель на CPU, ~10 с на кадр 720p (pip install 'onnxruntime-gpu[cuda,cudnn]')")
    return sess


def teacher(sess: ort.InferenceSession, img: np.ndarray, tile: int = 256, pad: int = 8) -> np.ndarray:
    """img HWC uint8 → очищенный HWC uint8 того же размера (×4 по тайлам, затем усреднение 4×4)."""
    h, w = img.shape[:2]
    x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255
    out = np.zeros((3, h, w), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            sy, sx = max(0, y0 - pad), max(0, x0 - pad)
            ey, ex = min(h, y1 + pad), min(w, x1 + pad)
            (y4,) = sess.run(None, {"input": x[:, :, sy:ey, sx:ex]})
            y4 = np.clip(y4[0], 0, 1)
            # ×4 → ×1: среднее по блокам 4×4 (как area-ресайз)
            c, H4, W4 = y4.shape
            y1x = y4.reshape(c, H4 // 4, 4, W4 // 4, 4).mean(axis=(2, 4))
            out[:, y0:y1, x0:x1] = y1x[:, y0 - sy : y0 - sy + (y1 - y0), x0 - sx : x0 - sx + (x1 - x0)]
    return (out.transpose(1, 2, 0) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="*", type=Path)
    ap.add_argument("--video", type=Path, action="append", default=[])
    ap.add_argument("--every", type=int, default=48, help="для --video: каждый N-й кадр")
    ap.add_argument("--skip-start", type=float, default=0, help="для --video: пропустить первые N секунд (опенинг)")
    ap.add_argument("--skip-end", type=float, default=0, help="для --video: пропустить последние N секунд (эндинг)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = ap.parse_args()

    (args.out / "lq").mkdir(parents=True, exist_ok=True)
    (args.out / "hq").mkdir(parents=True, exist_ok=True)
    sess = session(args.model)
    frames = list(args.frames)
    tmp = tempfile.TemporaryDirectory()
    for vi, video in enumerate(args.video):
        pattern = Path(tmp.name) / f"v{vi:03d}_%05d.png"
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                                   check=True, capture_output=True, text=True).stdout.strip())
        cut = ["-ss", str(args.skip_start), "-to", str(max(args.skip_start + 1, dur - args.skip_end))]
        subprocess.run(["ffmpeg", "-v", "error", *cut, "-i", str(video), "-vf", f"select='not(mod(n\\,{args.every}))'",
                        "-fps_mode", "passthrough", str(pattern)], check=True)
        frames += sorted(Path(tmp.name).glob(f"v{vi:03d}_*.png"))
    for i, f in enumerate(frames):
        name = f"{f.stem}.png" if not args.video else f.name
        img = np.asarray(Image.open(f).convert("RGB"))
        Image.fromarray(img).save(args.out / "lq" / name)
        Image.fromarray(teacher(sess, img)).save(args.out / "hq" / name)
        print(f"[{i + 1}/{len(frames)}] {name}")


if __name__ == "__main__":
    main()
