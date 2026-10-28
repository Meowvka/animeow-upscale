# Обучение MeowRestore

Всё в `training/`. PyTorch — под свою видеокарту (https://pytorch.org/get-started/locally/), остальное:

```bash
cd training
pip install -r requirements.txt
```

Пошаговый сценарий «датасет на своём ПК → обучение на арендованном GPU по SSH» — [RUNBOOK.md](RUNBOOK.md)
(скрипты `training/gpu/setup.sh` и `training/gpu/run.sh`).

Обучение S/M идёт и на CPU (медленно), L — лучше на GPU. Бесплатного Google Colab (T4) хватает:
M/L обучаются за 1–3 часа.

## 1. Данные — главное

Сеть учится превращать «как приходит со стриминга» в «как должно быть». Два способа получить пары:

### A. Есть чистые исходники (BD-remux / WEB-DL 1080p) — лучший вариант

```bash
python scripts/make_pairs.py /media/bd/*.mkv --out data/bd --height 720 --segments 20
```

Скрипт режет случайные отрезки, делает чистый кадр (lanczos до 720p) и испорченный —
те порчи, что встречаются у стримингов: апскейл с 480/540p, размытие, шум и пересжатие x264 с CRF 23–33.
Кадры совпадают покадрово. 10–20 серий разных тайтлов/студий → 2–5 тыс. пар — хватает.

`--height` — разрешение, в котором сеть будет работать на сайте: 720 для потоков 720p.
Можно сделать два набора (720 и 480) и учить на обоих.

### B. Чистых исходников нет — дистилляция от Real-ESRGAN

```bash
python scripts/make_teacher.py --video episode01.mp4 --video episode02.mp4 --every 48 --out data/teacher
```

Вход — реальные кадры ваших потоков (именно те артефакты, что видят зрители), цель — тот же кадр,
очищенный Real-ESRGAN (×4 → обратно ×¼). Маленькая сеть учится повторять большую в реальном времени.
На CPU учитель медленный (~10 с на кадр 720p) — используйте `onnxruntime-gpu` на машине с GPU.

Наборы можно смешивать: `--data data/bd data/teacher`.

## 2. Обучение

```bash
python scripts/train.py --data data/bd data/teacher --size l --steps 60000 --out runs/l
python scripts/train.py --data data/bd data/teacher --size m --steps 60000 --out runs/m
# S — с подсказками от M (дистилляция): маленькой сети так проще
python scripts/train.py --data data/bd --size s --steps 40000 --teacher runs/m/best.pt --out runs/s
```

Каждые `--val-every` шагов печатается PSNR входа и сети на отложенных кадрах — «выигрыш, дБ».
Лучшее сохраняется в `runs/<имя>/best.pt` + сразу `best.meow.json` и `best.glsl`.

Ориентиры: +0.5 дБ — заметно глазу на артефактах; +1–2 дБ — хорошо.
Пробный прогон на CPU (4000 шагов, ~150 пар): S +1.0, M +1.6, L +2.1 дБ на валидации.
Если выигрыш долго стоит около нуля — сеть «не завелась»: проверьте данные (пары должны совпадать
покадрово, `make_pairs` это проверяет) и не меняйте инициализацию в `meow/model.py` (там объяснено почему).

## 3. Смотреть глазами (обязательно)

PSNR не всё. Два способа:

```bash
# картинки «вход | сеть | эталон» и таблица PSNR по размерам
python scripts/eval.py runs/s/best.meow.json runs/m/best.meow.json --data data/val --out runs/eval
```

и **в mpv на настоящей серии** — хук готов к использованию:

```
mpv --glsl-shaders="runs/m/best.glsl;~~/shaders/Anime4K_Upscale_CNN_x2_M.glsl" episode.mp4
```

Сравнивайте с `Anime4K_Restore_CNN_M.glsl` вместо нашего — это и есть «было/стало» для сайта.

## 4. Выгрузка для сайта

```bash
python scripts/export.py runs/m/best.pt --name meow-restore-m --out ../models/meow
```

Проверяет, что эталон на fp16-весах совпадает с torch, и пишет `models/meow/meow-restore-m.meow.json` (+ `.glsl`).
Затем из корня репозитория: `cd web && npm run verify` — прогонит шейдеры в Chromium против эталона.

## Свой размер

`--channels 8 --layers 8` и т.п. (каналы кратны 4). Стоимость на пиксель — `web/src/tiers.ts → macsPerPixel()`.
На WebGPU каналов не больше `4 × maxStorageTexturesPerShaderStage` (обычно 16).
