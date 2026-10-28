# animeow-upscale

Пайплайн качества картинки для animeow: **кадры и картинки — сервером один раз, видео — на устройстве
каждым кадром**, своей маленькой сетью перед Anime4K.

```
картинки:  _orig ──► Real-ESRGAN AnimeVideo-v3 (сервер, CPU) ──► готовый WebP          (устройству — 0)
видео:     кадр  ──► MeowRestore (2–45 КБ весов) ──► Anime4K Upscale ──► экран        (WebGL2 / WebGPU)
```

Подробно и с цифрами — [docs/PIPELINE.md](docs/PIPELINE.md).
Встраивание в animeow — [docs/INTEGRATION.md](docs/INTEGRATION.md).
Обучение своей сети — [docs/TRAINING.md](docs/TRAINING.md).

## Что внутри

| Папка | Что |
|---|---|
| `training/` | MeowRestore на PyTorch: подготовка пар из видео (ffmpeg), учитель Real-ESRGAN, обучение, выгрузка, сравнение |
| `web/` | Рантайм для браузера: веса `.meow.json` → mpv-хук GLSL (WebGL2-исполнитель animeow) и WGSL (WebGPU, в цепочке с `anime4k-webgpu`) |
| `server/` | Real-ESRGAN через onnxruntime-node для `media-enhance` (та же форма, что `enhanceRgb` из `anime4k-cpu.ts`) |
| `models/realesrgan/` | Real-ESRGAN AnimeVideo-v3 в ONNX (fp32 2.5 МБ, fp16 1.25 МБ) — для сервера и как учитель |
| `models/meow/` | Веса MeowRestore |

## MeowRestore

Сеть восстановления в архитектуре Anime4K (свёртки 3×3, CReLU, остаток к входу), поэтому исполняется теми
же шейдерами, что уже работают в плеере, — без onnxruntime и без новых зависимостей.

| Размер | Параметров | `.meow.json` | Для кого |
|---|---|---|---|
| S | 915 | 2.7 КБ | телефоны, WebGL2 |
| M | 5 299 | 14 КБ | ноутбуки, сильные телефоны |
| L | 16 611 | 45 КБ (≈65 КБ в fp32) | десктоп, WebGPU |

Отличие от Anime4K Restore — **обучается на ваших потоках**: на тех артефактах, что реально приходят
со стримингов (пересжатие x264, апскейл 480→720, размытие).

> ⚠️ `models/meow/proof/` — веса пробного прогона на **синтетических** кадрах (проверка конвейера,
> не для продакшена). Настоящие веса — после обучения на аниме-кадрах: [docs/TRAINING.md](docs/TRAINING.md).

## Проверено

- Все три исполнителя — GLSL из Python, GLSL из TS, WGSL — совпадают с эталоном на numpy в настоящем Chromium
  (WebGL2 и WebGPU), погрешность ≤ 5·10⁻⁴ (предел fp16-текстур): `cd web && npm run verify`.
- `MeowPreset` собирается в одну цепочку с `anime4k-webgpu` (ClampHighlights → Meow → CNNx2VL → …).
- Обучение сходится на парах из ffmpeg-конвейера: пробный прогон на CPU — S +1.25, M +2.19, L +2.90 дБ против +1.2…1.55 у Anime4K Restore на тех же кадрах (синтетика, см. `models/meow/proof/README.md`).
- Сервер: Real-ESRGAN, постер 460→920 ≈ 4.4 с на одно ядро (1.5 с на четыре).

## Быстрые команды

```bash
# браузерный рантайм
cd web && npm i && npm test && npm run verify          # verify нужен playwright + chromium

# сервер
cd server && ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm i
npx tsx src/cli.ts --scale 2 poster.webp poster-x2.webp
npx tsx src/bench.ts 1

# обучение
cd training && pip install -r requirements.txt
python scripts/make_pairs.py /media/bd/*.mkv --out data/bd
python scripts/train.py --data data/bd --size m --out runs/m
```

## Лицензии

Код — MIT. `models/realesrgan/` — производные Real-ESRGAN (BSD-3-Clause), см. [NOTICE](NOTICE).
Формат хуков и подход — Anime4K (MIT, bloc97).
