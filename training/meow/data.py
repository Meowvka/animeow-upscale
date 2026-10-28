"""Пары кадров (испорченный → чистый) для обучения: data/<набор>/lq/*.png и data/<набор>/hq/*.png с одинаковыми именами."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def find_pairs(roots: list[Path]) -> list[tuple[Path, Path]]:
    pairs = []
    for root in roots:
        for hq in sorted((root / "hq").glob("*.png")):
            lq = root / "lq" / hq.name
            if lq.exists():
                pairs.append((lq, hq))
    if not pairs:
        raise SystemExit(f"нет пар в {', '.join(map(str, roots))} (ожидаются папки lq/ и hq/)")
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(a, copy=True)).permute(2, 0, 1).float().div_(255)


class PairDataset(Dataset):
    """Случайные кропы + отражения/повороты. Кадры читаются лениво (сотни 720p-кадров не влезут в память)."""

    def __init__(self, pairs: list[tuple[Path, Path]], crop: int = 128, samples_per_epoch: int = 10_000, cache: bool = False):
        self.pairs = pairs
        self.crop = crop
        self.samples = samples_per_epoch
        # Небольшой набор (до ~1–2 тыс. кадров 720p) выгоднее держать в памяти: PNG декодируется дольше шага обучения.
        self.cached = [(load_rgb(a), load_rgb(b)) for a, b in pairs] if cache else None

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, _i: int):
        i = random.randrange(len(self.pairs))
        lq, hq = self.cached[i] if self.cached else (load_rgb(self.pairs[i][0]), load_rgb(self.pairs[i][1]))
        if lq.shape != hq.shape:
            raise ValueError(f"размеры не совпадают: {self.pairs[i][0]} {lq.shape} vs {hq.shape}")
        h, w = lq.shape[:2]
        c = min(self.crop, h, w)
        y, x = random.randint(0, h - c), random.randint(0, w - c)
        lq, hq = lq[y : y + c, x : x + c], hq[y : y + c, x : x + c]
        if random.random() < 0.5:
            lq, hq = lq[:, ::-1], hq[:, ::-1]
        if random.random() < 0.5:
            lq, hq = lq[::-1], hq[::-1]
        if random.random() < 0.5:
            lq, hq = lq.transpose(1, 0, 2), hq.transpose(1, 0, 2)
        return to_tensor(lq), to_tensor(hq)


def val_crops(pairs: list[tuple[Path, Path]], size: int = 256, limit: int = 32) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Фиксированные центральные кропы для честного сравнения между запусками."""
    out = []
    for lq_path, hq_path in pairs[:limit]:
        lq, hq = load_rgb(lq_path), load_rgb(hq_path)
        h, w = lq.shape[:2]
        s = min(size, h, w)
        y, x = (h - s) // 2, (w - s) // 2
        out.append((to_tensor(lq[y : y + s, x : x + s]), to_tensor(hq[y : y + s, x : x + s])))
    return out
