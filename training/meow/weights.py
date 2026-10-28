"""
Формат весов `.meow.json` — единый источник для всех исполнителей:
Python (эталон на numpy), GLSL (mpv / WebGL2) и WGSL (WebGPU, генерируется в браузере).

{
  "format": "meow-restore/1",
  "name": "meow-restore-m",
  "channels": 8, "layers": 6,
  "dtype": "float16",
  "layers_data": [ {"in": 3, "out": 8, "w": "<base64 fp16, [out,in,3,3]>", "b": "<base64 fp16, [out]>"}, ... ]
}

Веса округляются до fp16 ДО генерации шейдеров: GLSL, WGSL и эталон считают одними и теми же числами.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

FORMAT = "meow-restore/1"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f2").tobytes()).decode()


def _unb64(s: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f2").astype(np.float32).reshape(shape)


def from_model(model, name: str) -> dict:
    layers = []
    for conv in model.convs():
        w = conv.weight.detach().cpu().numpy()
        b = conv.bias.detach().cpu().numpy()
        layers.append({"in": int(w.shape[1]), "out": int(w.shape[0]), "w": _b64(w), "b": _b64(b)})
    return {"format": FORMAT, "name": name, **model.config(), "dtype": "float16", "layers_data": layers}


def from_arrays(name: str, channels: int, layers: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    data = [{"in": int(w.shape[1]), "out": int(w.shape[0]), "w": _b64(w), "b": _b64(b)} for w, b in layers]
    return {"format": FORMAT, "name": name, "channels": channels, "layers": len(layers), "dtype": "float16", "layers_data": data}


def unpack(meta: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(W [out,in,3,3], b [out])] в float32 (значения уже fp16-точные)."""
    if meta.get("format") != FORMAT:
        raise ValueError(f"unknown format {meta.get('format')}")
    return [
        (_unb64(l["w"], (l["out"], l["in"], 3, 3)), _unb64(l["b"], (l["out"],)))
        for l in meta["layers_data"]
    ]


def save(meta: dict, path: Path) -> None:
    path.write_text(json.dumps(meta, separators=(",", ":")))


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_into_model(meta: dict, model) -> None:
    import torch

    for conv, (w, b) in zip(model.convs(), unpack(meta)):
        conv.weight.data = torch.from_numpy(w.copy())
        conv.bias.data = torch.from_numpy(b.copy())


def reference_forward(meta: dict, img: np.ndarray) -> np.ndarray:
    """Эталон на numpy без torch: img [3,H,W] float32 0..1 → [3,H,W]."""
    layers = unpack(meta)
    x = img.astype(np.float32)

    def conv(h, w, b):
        p = np.pad(h, ((0, 0), (1, 1), (1, 1)), mode="edge")
        H, W = h.shape[1:]
        out = np.empty((w.shape[0], H, W), np.float32)
        out[:] = b[:, None, None]
        for ky in range(3):
            for kx in range(3):
                patch = p[:, ky : ky + H, kx : kx + W].reshape(h.shape[0], -1)
                out += (w[:, :, ky, kx] @ patch).reshape(-1, H, W)
        return out

    crelu = lambda h: np.concatenate([np.maximum(h, 0), np.maximum(-h, 0)], 0)
    h = conv(x, *layers[0])
    for w, b in layers[1:-1]:
        h = conv(crelu(h), w, b)
    return x + conv(crelu(h), *layers[-1])
