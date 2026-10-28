# animeow-upscale — для ассистента

Ты уже знаешь сайт animeow и его Anime4K (`app/lib/anime4k-enhance.ts`, `anime4k-webgl.ts`,
`metadata/lib/anime4k-cpu.ts`, `media-enhance.ts`). Этот репозиторий — отдельный проект для качества
картинки на сайте. Ниже — что в нём и как он связан с сайтом.

## Идея

- **Картинки** (постеры, баннеры, кадры, пре-кадры) — на сервере, один раз: Real-ESRGAN AnimeVideo-v3
  вместо или вместе с Anime4K L в `media-enhance`. Зрителю — готовый WebP.
- **Видео** (Kodik 720p → режимы 1080/2K) — в браузере: **MeowRestore** (наша маленькая сеть восстановления)
  → потом Anime4K Upscale. Своя сеть нужна, потому что её можно обучить на реальных артефактах потоков Kodik.

MeowRestore — та же архитектура, что Anime4K Restore (свёртки 3×3, CReLU, остаток к входу). Поэтому она
выгружается в **те же mpv-хуки**, и WebGL-исполнитель сайта запускает её без изменений. На WebGPU есть
`MeowPreset` — это ModeA, где CNNVL заменён на нашу сеть.

## Карта

| Папка | Что |
|---|---|
| `web/src/` | Рантайм для браузера, без зависимостей. `weights.ts` — формат `.meow.json` (fp16 base64); `glsl.ts` — `toMpvHook()`; `webgpu.ts` — `MeowRestoreGPU`; `preset.ts` — `MeowPreset(lib, …)`; `tiers.ts` — `pickSize`/`stepDown`; `hooks-gl.ts` — мини-исполнитель хуков (для тестов, повторяет семантику `anime4k-webgl.ts`) |
| `server/src/enhance.ts` | Real-ESRGAN через onnxruntime-node. `enhancer.enhanceRgb(rgb, w, h, { upscale })` — та же форма, что `enhanceRgb` в `anime4k-cpu.ts`, но async |
| `training/` | PyTorch: `meow/model.py` (сеть), `meow/glsl.py` (экспорт хуков), `meow/weights.py` (формат + эталон на numpy); `scripts/` — пары из видео, учитель, обучение, выгрузка, сравнение; `gpu/` — установка и прогон на арендованном сервере |
| `models/realesrgan/` | Real-ESRGAN в ONNX (fp32 — сервер, fp16) |
| `models/meow/proof/` | **Пробные веса, обучены на синтетике — НЕ для продакшена.** Настоящие появятся в `models/meow/` после обучения |
| `docs/` | `PIPELINE.md` — схема и цифры; `INTEGRATION.md` — **что менять в animeow**; `TRAINING.md`; `RUNBOOK.md` — обучение на GPU по SSH |

## Размеры сети

| | Параметров | `.meow.json` | Где |
|---|---|---|---|
| S | 915 | 2.7 КБ | телефоны, WebGL2 |
| M | 5 299 | 14 КБ | ноутбуки, сильные телефоны |
| L | 16 611 | 45 КБ | десктоп, WebGPU |

Сайт качает только `.meow.json`, шейдеры генерируются в браузере (`toMpvHook`, `layerWGSL`).
По скорости: L ≈ 0.75 от CNNVL, M ≈ CNNM, S легче Anime4K Restore S.

## Соглашения (важно не сломать)

- `mat4` в GLSL/WGSL — **по столбцам**, как у Anime4K: `out[i] = Σj m[j*4+i]·v[j]`.
- Координаты как в mpv: строка 0 — верх кадра, `+y` вниз; `W[o,i,ky,kx]` ↔ смещение `(kx-1, ky-1)`.
- CReLU в шейдере: `go_{2g} = max(t_g, 0)`, `go_{2g+1} = max(-t_g, 0)`; в torch — `cat([relu(h), relu(-h)])`.
- Padding — `replicate` (= CLAMP_TO_EDGE). Веса округляются до fp16 **до** генерации шейдеров.
- Инициализация в `model.py` (дисперсия 1/fan_in) — не менять: со стандартной PyTorch глубокая L не учится.
- Датасеты и серии **никогда не коммитить** (`training/data` в .gitignore).

## Проверки

```bash
cd web && npm i && npm run typecheck && npm test        # юнит-тесты, в т.ч. GLSL из TS == GLSL из Python
cd web && npm run verify                                 # Chromium: GLSL(Py), GLSL(TS), WGSL против эталона numpy
python -m unittest discover training/tests               # без torch
cd server && ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm i && npm run typecheck
```

`verify` и `test/compare.mjs` нужен playwright с Chromium; WebGPU в headless — через SwiftShader (флаги уже в
`web/test/browser.mjs`). Без `ONNXRUNTIME_NODE_INSTALL_CUDA=skip` установка onnxruntime-node на Linux падает.

## Статус

Сделано и проверено:
- рантайм (WebGL2 + WebGPU совпадают с эталоном до 5e-4) и цепочка с `anime4k-webgpu`;
- серверный Real-ESRGAN: постер 460→920 ≈ 4.4 с на ядро;
- конвейер обучения целиком (на синтетике: S +1.25, M +2.19, L +2.90 дБ против +1.2…1.55 у Anime4K Restore).

Не сделано:
1. **Обучение на настоящих кадрах** — владелец арендует GPU, сценарий в `docs/RUNBOOK.md`.
2. **Встраивание в animeow** — по `docs/INTEGRATION.md`: хук вместо `RESTORE_CNN_S` в `anime4k-webgl.ts`,
   `MeowPreset` вместо `lib.ModeA/ModeAA` в `anime4k-enhance.ts`, шаг вниз L→M→S вместо выключения,
   `--net ESRGAN` в `media-enhance.ts`. Код из `web/src/` копируется в `app/lib/meow/`, а не ставится пакетом.
3. Ставить на сайт только веса, которые на отложенных тайтлах лучше Anime4K Restore хотя бы на +0.3 дБ
   и глазами чище. Пробные (`proof/`) — не ставить.
