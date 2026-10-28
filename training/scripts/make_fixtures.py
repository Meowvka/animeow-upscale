#!/usr/bin/env python3
"""Тестовые данные для web/test: случайные веса всех размеров, вход, эталон (numpy) и GLSL из Python."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meow import weights  # noqa: E402
from meow.glsl import to_glsl  # noqa: E402
from meow.model_sizes import SIZES  # noqa: E402

out_dir = Path(__file__).resolve().parents[2] / "web" / "test" / "fixtures"
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
W, H = 32, 24
img = rng.random((3, H, W), dtype=np.float32)
img = img.astype(np.float16).astype(np.float32)  # вход в тесте живёт в fp16-текстуре

for name, size in SIZES.items():
    C, L = size.channels, size.layers
    cin = [3] + [2 * C] * (L - 1)
    cout = [C] * (L - 1) + [3]
    layers = [
        ((rng.standard_normal((o, i, 3, 3)) * (0.6 / np.sqrt(i * 9))).astype(np.float32), (rng.standard_normal(o) * 0.05).astype(np.float32))
        for i, o in zip(cin, cout)
    ]
    meta = weights.from_arrays(f"meow-restore-{name}", C, layers)
    expected = weights.reference_forward(meta, img)
    (out_dir / f"{name}.json").write_text(json.dumps({
        "weights": meta,
        "width": W,
        "height": H,
        "input": img.transpose(1, 2, 0).ravel().tolist(),  # HWC RGB
        "expected": expected.transpose(1, 2, 0).ravel().tolist(),
        "glsl": to_glsl(meta),
    }))
    print(name, "ok", float(np.abs(expected - img).mean()))
