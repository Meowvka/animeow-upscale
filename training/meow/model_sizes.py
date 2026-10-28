"""Размеры MeowRestore (без зависимости от torch — нужны и экспорту, и тестам)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Size:
    channels: int
    layers: int  # всего свёрток, включая первую и последнюю
    note: str


# Вычисления ≈ числу параметров умножений на пиксель.
SIZES: dict[str, Size] = {
    # ~0.9 тыс. параметров: дешевле Anime4K Restore S — любой телефон с WebGL2.
    "s": Size(4, 4, "телефоны"),
    # ~5.3 тыс.: средние телефоны с WebGPU, ноутбуки, десктоп на WebGL2.
    "m": Size(8, 6, "ноутбуки, сильные телефоны"),
    # ~16.6 тыс. (~65 КБ fp32, ~33 КБ fp16): десктоп с WebGPU.
    "l": Size(12, 8, "десктоп с WebGPU"),
}
