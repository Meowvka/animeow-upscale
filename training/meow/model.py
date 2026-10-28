"""
MeowRestore: восстановление кадра в исходном разрешении (артефакты сжатия, мыло, шум).

Архитектура нарочно повторяет Anime4K Restore, чтобы сеть без изменений исполнялась
шейдерами (mpv-хуки GLSL на WebGL2 и WGSL на WebGPU):

    x ─► conv3×3 (3→C) ─► [CReLU ─► conv3×3 (2C→C)] × (layers-2) ─► CReLU ─► conv3×3 (2C→3) ─► + x

* C кратно 4 — одна текстура RGBA = 4 канала.
* CReLU = concat(relu(h), relu(-h)): в шейдере это max(t, 0) и max(-t, 0) от одной текстуры,
  т.е. 4 канала текстуры дают 8 «нейронов» без лишних проходов.
* padding = replicate — ровно как CLAMP_TO_EDGE у текстур.
* Сеть предсказывает поправку к входу (residual) — поэтому учится быстро и не «плывёт» по цвету.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from .model_sizes import SIZES, Size  # noqa: F401  (реэкспорт)


def crelu(h: torch.Tensor) -> torch.Tensor:
    return torch.cat([F.relu(h), F.relu(-h)], dim=1)


class MeowRestore(nn.Module):
    def __init__(self, channels: int = 8, layers: int = 6):
        super().__init__()
        if channels % 4:
            raise ValueError("channels must be a multiple of 4 (RGBA textures)")
        if layers < 2:
            raise ValueError("layers must be >= 2")
        self.channels = channels
        self.layers = layers
        conv = lambda i, o: nn.Conv2d(i, o, 3, padding=1, padding_mode="replicate")
        self.first = conv(3, channels)
        self.hidden = nn.ModuleList(conv(2 * channels, channels) for _ in range(layers - 2))
        self.last = conv(2 * channels, 3)
        # CReLU сохраняет энергию сигнала (|relu(h)|² + |relu(-h)|² = |h|²), поэтому дисперсия весов 1/fan_in
        # держит масштаб активаций от слоя к слою. Стандартная инициализация PyTorch (≈1/(3·fan_in)) гасит
        # сигнал втрое на каждом слое — у глубокой L он к концу почти нулевой и обучение долго стоит.
        for conv in [self.first, *self.hidden]:
            nn.init.normal_(conv.weight, std=(1 / (conv.in_channels * 9)) ** 0.5)
            nn.init.zeros_(conv.bias)
        # Старт почти с тождественного преобразования: поправка ≈ 0. Не ровно 0 — иначе на первых шагах
        # градиент не доходит до ранних слоев и глубокая сеть (L) долго стоит на месте.
        nn.init.normal_(self.last.weight, std=1e-3)
        nn.init.zeros_(self.last.bias)

    @classmethod
    def from_size(cls, name: str) -> "MeowRestore":
        s = SIZES[name]
        return cls(s.channels, s.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.first(x)
        for conv in self.hidden:
            h = conv(crelu(h))
        return x + self.last(crelu(h))

    def convs(self) -> list[nn.Conv2d]:
        return [self.first, *self.hidden, self.last]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> dict:
        return {"channels": self.channels, "layers": self.layers}
