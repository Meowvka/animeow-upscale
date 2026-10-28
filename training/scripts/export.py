#!/usr/bin/env python3
"""
Чекпойнт → веса для сайта (.meow.json) и mpv-хук (.glsl), с проверкой: torch == numpy-эталон на fp16-весах.

    python scripts/export.py runs/m1/best.pt --name meow-restore-m --out ../models/meow
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meow import weights  # noqa: E402
from meow.glsl import to_glsl  # noqa: E402
from meow.model import MeowRestore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", type=Path)
ap.add_argument("--name", required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu")
model = MeowRestore(**ck["config"])
model.load_state_dict(ck["model"])
meta = weights.from_model(model, args.name)
weights.load_into_model(meta, model)  # сверяем уже на fp16-весах — ровно то, что уйдёт в шейдер
x = torch.rand(1, 3, 40, 56)
with torch.no_grad():
    y = model(x)[0].numpy()
err = float(np.abs(y - weights.reference_forward(meta, x[0].numpy())).max())
assert err < 1e-4, f"эталон расходится с torch: {err}"

args.out.mkdir(parents=True, exist_ok=True)
weights.save(meta, args.out / f"{args.name}.meow.json")
(args.out / f"{args.name}.glsl").write_text(to_glsl(meta))
print(f"{args.name}: {model.num_params()} параметров, json {(args.out / f'{args.name}.meow.json').stat().st_size} Б, "
      f"glsl {(args.out / f'{args.name}.glsl').stat().st_size} Б, сверка {err:.1e}")
