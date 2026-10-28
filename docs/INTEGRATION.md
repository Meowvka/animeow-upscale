# Встраивание в animeow

> Для ассистента, который будет встраивать: в animeow своя (новая) версия Next.js — сверяйтесь
> с `node_modules/next/dist/docs/`, а не с памятью. Ниже — точки входа по текущему коду ветки `test`.

Состав (всё без внешних зависимостей, кроме уже стоящих в animeow `sharp` и `anime4k-webgpu`):

| Что | Откуда | Куда в animeow |
|---|---|---|
| Рантайм сети (TS, ~600 строк) | `web/src/*.ts` | `app/lib/meow/` (скопировать) |
| Веса | `models/meow/*.meow.json` | `public/models/meow/` |
| Серверный Real-ESRGAN | `server/src/enhance.ts` + `models/realesrgan/*.fp32.onnx` | `metadata/lib/esrgan.ts` + `metadata/lib/models/` |

Копировать, а не ставить пакетом: файлов мало, а сборка Next не будет зависеть от чужого репозитория.

## 1. Видео, WebGL2 (телефоны) — `app/lib/anime4k-webgl.ts`

Исполнитель уже понимает наш формат (те же mpv-хуки, что у Anime4K). Нужно только дать ему
другой хук восстановления вместо `RESTORE_CNN_S`:

```ts
// anime4k-webgl.ts
export function createAnime4KGL(gl: WebGL2RenderingContext, opts: { restoreHook?: string } = {}): Anime4KGL {
  …
  const restore = parseHooks(opts.restoreHook ?? RESTORE_CNN_S).map(build);
```

```ts
// anime4k-enhance.ts → startGL()
import { loadWeights, toMpvHook, pickSize } from "./meow";
const size = pickSize({ webgpu: false, mobile: isMobile, memoryGb: navigator.deviceMemory, frameHeight: video.videoHeight });
const restoreHook = toMpvHook(await loadWeights(`/models/meow/meow-restore-${size}.meow.json`));
const a4k = net ? createAnime4KGL(gl, { restoreHook }) : null;
```

`startGL` сейчас синхронный — загрузку весов сделать заранее (при монтировании плеера, это 2–14 КБ)
и передать хук параметром.

## 2. Видео, WebGPU (десктоп) — `app/lib/anime4k-enhance.ts → build()`

```ts
import { MeowPreset, parseWeights } from "./meow";

// вместо: const Preset = mode === "1440" ? lib.ModeAA : lib.ModeA; new Preset({...})
const preset = new MeowPreset(lib, {
  device, inputTexture: input,
  nativeDimensions: { width: w, height: h }, targetDimensions: target,
  weights: meowWeights,                       // заранее: await loadWeights("/models/meow/meow-restore-l.meow.json")
  upscale: mode === "1440" ? "UL" : "VL",
});
```

Интерфейс `MeowPreset` тот же, что у `lib.ModeA` (`pass(encoder)`, `getOutputTexture()`), остальной
код (`copyExternalImageToTexture`, blit, счётчик пропусков) не меняется.

## 3. Шаг вниз вместо выключения

Сейчас при «видеокарта не успевает» улучшение выключается целиком. Лучше на ступень вниз:

```ts
import { stepDown } from "./meow";
// в обработчике tooSlow: size = stepDown(size); если null — как сейчас (резкость / выключить)
```

L → M → S → резкость (CAS) → выкл.

## 4. Пре-кадры серий

`frames/` уже обрабатывает `media-enhance` на сервере; браузерный `enhanceStills()` нужен только для
ещё не обработанных. С ESRGAN на сервере стоит оставить его как есть (Anime4K в браузере) — или
вовсе отключить и показывать исходник, пока сервер не догнал: это экономит GPU телефона до старта серии.

## 5. Сервер — `metadata/scripts/metadata/media-enhance.ts`

`server/src/enhance.ts` даёт ту же форму, что `enhanceRgb` из `anime4k-cpu.ts`, только async:

```ts
import { createEnhancer } from "../../lib/esrgan";

let esrgan: Awaited<ReturnType<typeof createEnhancer>> | null = null;

async function enhanceOne(item: Item, net: NetSize | "ESRGAN") {
  …
  const r = net === "ESRGAN"
    ? await (esrgan ??= await createEnhancer({ threads: 1 })).enhanceRgb(new Uint8Array(data), info.width, info.height, { upscale })
    : enhanceRgb(new Uint8Array(data), info.width, info.height, { size: net, upscale });
```

и `--net ESRGAN` в разборе аргументов. `threads: 1` — потому что процессов и так «ядра минус одно».

Зависимость: `onnxruntime-node`. Ставить с `ONNXRUNTIME_NODE_INSTALL_CUDA=skip` — иначе установщик
на Linux пытается скачать CUDA-пакет с NuGet и падает:

```bash
ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm i onnxruntime-node
```

Чтобы перегнать уже обработанное Anime4K: вернуть файлы из `_orig/<папка>` (как описано в
`docs/guide/16-media.md`) и запустить `media-enhance --net ESRGAN`. Начать с `--what covers --limit 20`
и сравнить глазами.

## 6. Проверка после встраивания

- В плеере: «Картинка» → 1080 на десктопе — в DevTools Performance кадр не дольше, чем с ModeA.
- Телефон: включить улучшение на 720p — нет «телефон не успевает» в консоли за 4+ минуты.
- mpv с теми же весами (`models/meow/*.glsl`) — эталон того, как должно выглядеть.
