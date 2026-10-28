# RUNBOOK: обучение MeowRestore на арендованном GPU

Инструкция для **Claude Code на компьютере владельца** (скорее всего Windows; там лежат исходники серий).
Задача: собрать датасет локально → залить на арендованный GPU-сервер по SSH → обучить → забрать веса.

Почему датасет собирается локально: исходники BD/WEB-DL весят 50–100 ГБ, а готовые пары — ~12 ГБ;
к тому же у арендованных серверов старые процессоры (E5-2666v3), а аренда идёт поминутно.

## Правила

- Репозиторий `animeow` (сайт) **не трогать**. Работа только в `animeow-upscale`.
- Серии и датасет **не коммитить и никуда не выкладывать** (авторские права): только локальный диск и арендованный сервер.
  После обучения — удалить с сервера (шаг 5).
- Пароль SSH не записывать в файлы. Если владелец даёт пароль — предложить настроить вход по ключу.
- Сервер арендует и выключает владелец; напомнить выключить, когда всё скачано.

## 0. Спросить у владельца (один раз, одним сообщением)

1. SSH: адрес, порт, пользователь; ключ или пароль.
2. Папка **TRAIN** — исходники BD-remux / WEB-DL 1080p: 8–10 разных тайтлов по 2–3 серии
   (разные студии и годы, яркие и тёмные сцены). Чем разнообразнее, тем лучше — не 12 серий одного тайтла.
3. Папка **VAL** — 2–3 **других** тайтла по 1 серии (их не будет в обучении — по ним честная оценка).
4. Папка **KODIK** (необязательно, но полезно) — 5–10 серий разных тайтлов, скачанных с Kodik как есть (720p).
   Из них учитель Real-ESRGAN сделает пары с настоящими артефактами потоков сайта.
5. Длина опенинга/эндинга, если нестандартная (по умолчанию пропускаются первые 120 с и последние 150 с).

## 1. Локально: инструменты

```powershell
python --version            # нужен 3.10+
ffmpeg -hide_banner -encoders | Select-String libx264   # нужен ffmpeg с libx264 (и ffprobe рядом)
```

Если чего-то нет — с согласия владельца: `winget install Python.Python.3.12`, `winget install Gyan.FFmpeg`
(после установки открыть новый терминал).

```powershell
git clone https://github.com/Meowvka/animeow-upscale
cd animeow-upscale\training
python -m pip install numpy pillow
```

## 2. Локально: датасет

Поток на сайте — **720p** (Kodik), поэтому `--height 720`. Каждая папка — **одной командой**
(или разными `--prefix`, иначе кадры перезапишутся).

```powershell
# обучение: ~20 отрезков по 4 с с каждой серии, без опенинга и эндинга
python scripts/make_pairs.py (Get-ChildItem "D:\anime\train" -Recurse -Include *.mkv,*.mp4).FullName `
  --out data/train --height 720 --segments 20 --skip-start 120 --skip-end 150

# проверка: другие тайтлы, другой seed
python scripts/make_pairs.py (Get-ChildItem "D:\anime\val" -Recurse -Include *.mkv,*.mp4).FullName `
  --out data/val --height 720 --segments 6 --skip-start 120 --skip-end 150 --seed 1

# серии с Kodik — просто положить, учитель запустится на сервере
New-Item -ItemType Directory -Force data/kodik | Out-Null
Copy-Item "D:\anime\kodik\*" data/kodik/
```

Проверить перед заливкой:
- число файлов в `data/train/hq` и `data/train/lq` одинаковое (скрипт и сам падает, если пары разъехались);
- открыть 2–3 пары глазами: `lq` заметно хуже `hq`, кадры совпадают;
- ориентиры: 30 серий × 20 отрезков × 8 кадров ≈ 4800 пар ≈ 12 ГБ; ~1–2 мин на серию.

Упаковать (tar есть в Windows 10+):

```powershell
tar -cf meow-data.tar data
```

## 3. Сервер

Лучше конфигурация с **64 ГБ памяти** (кадры держатся в памяти, обучение быстрее). На 32 ГБ тоже
работает — `run.sh` сам выключит кэш, если не хватает.

```powershell
# PORT, user, HOST — подставить
ssh -p PORT user@HOST nvidia-smi          # видеокарта видна?
ssh -p PORT user@HOST "git clone https://github.com/Meowvka/animeow-upscale && bash animeow-upscale/training/gpu/setup.sh"
```

`setup.sh` ставит ffmpeg, PyTorch под версию CUDA драйвера и onnxruntime-gpu, в конце проверяет, что GPU видно.

Залить датасет и распаковать (порт у scp — заглавная `-P`):

```powershell
scp -P PORT meow-data.tar user@HOST:animeow-upscale/training/
ssh -p PORT user@HOST "cd animeow-upscale/training && tar -xf meow-data.tar && rm meow-data.tar && ls data"
```

Запустить в tmux (обрыв SSH ничего не прервёт):

```powershell
ssh -p PORT user@HOST "cd animeow-upscale && tmux new -d -s meow 'bash training/gpu/run.sh 2>&1 | tee training/run.log'"
```

## 4. Следить

Раз в 10–15 минут:

```powershell
ssh -p PORT user@HOST "tail -n 15 animeow-upscale/training/run.log"
```

Что должно быть: строки `[шаг/всего] … вход X дБ → сеть Y дБ (+выигрыш)`, выигрыш растёт и выходит на плато.
Порядок: учитель (если есть Kodik, ~10–20 мин) → L → M → S → проверка → `== ГОТОВО`.
Всего ориентировочно 1.5–3 часа.

Тревожные признаки — остановить (`tmux kill-session -t meow`) и сообщить владельцу:
- после 10 тыс. шагов выигрыш ≤ +0.1 дБ — с данными что-то не так;
- «ВНИМАНИЕ: CUDA недоступна» у учителя — он будет считать часами; можно убрать `data/kodik` и перезапустить;
- `Killed` / нехватка памяти — перезапустить с `CACHE=0`: `tmux new -d -s meow 'CACHE=0 bash training/gpu/run.sh 2>&1 | tee -a training/run.log'`
  (уже обученные размеры пропускаются).

## 5. Забрать результат и убрать за собой

```powershell
scp -P PORT user@HOST:animeow-upscale/training/results.tar .
tar -xf results.tar
ssh -p PORT user@HOST "rm -rf animeow-upscale/training/data animeow-upscale/training/runs"
```

Показать владельцу:
- `out/eval.txt` — PSNR и выигрыш S/M/L на отложенных тайтлах;
- 3–4 картинки из `out/eval/` (слева направо: вход | S | M | L | эталон).

Напомнить владельцу выключить аренду в панели хостинга.

## 6. Сохранить веса

С согласия владельца — в репозиторий `animeow-upscale`, отдельной веткой:

```powershell
git checkout -b weights/$(Get-Date -Format yyyy-MM-dd)
Copy-Item out/models/* ../models/meow/
# в ../models/meow/README.md: дата, какие тайтлы (без файлов), сколько пар, таблица из out/eval.txt
git add ../models/meow
git commit -m "MeowRestore weights trained on real anime frames"
git push -u origin HEAD
```

Чекпойнты `out/*.pt` (десятки КБ) оставить локально — пригодятся для дообучения (`train.py --init`).

## 7. (По желанию) сравнить с Anime4K

Нужны Node.js 22, `npm i -D playwright && npx playwright install chromium` в `web/`, и GLSL Anime4K
(в репозитории сайта: `metadata/lib/anime4k/Anime4K_Restore_CNN_{S,M,L}.glsl` — только прочитать/скопировать).

```powershell
cd ..\web; npm i
node test/compare.mjs --pairs ../training/data/val --limit 24 --out ../training/out/compare `
  meow-L=../models/meow/meow-restore-l.meow.json a4k-L=C:\path\Anime4K_Restore_CNN_L.glsl `
  meow-S=../models/meow/meow-restore-s.meow.json a4k-S=C:\path\Anime4K_Restore_CNN_S.glsl
```

Решение: если наша сеть на отложенных тайтлах не лучше Anime4K Restore хотя бы на +0.3 дБ и глазами не чище —
на сайт не ставить, сообщить владельцу.
