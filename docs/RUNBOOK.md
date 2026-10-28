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

1. SSH: адрес, порт, пользователь; ключ или пароль. Сервер пускает **только с IP этого ПК** —
   если SSH не подключается, сначала проверить текущий IP (`(Invoke-RestMethod ifconfig.me/ip)`)
   и попросить владельца обновить его в панели хостинга.
2. Папка с исходниками BD-remux / WEB-DL 1080p — 10 тайтлов по 3 серии (уже собраны).
3. **Какие 2 тайтла из 10 отложить в VAL** (см. ниже). Если владелец не выберет — предложить
   2 самых непохожих на остальные (другая студия/эпоха/стиль) и согласовать.
4. Адрес сайта animeow (прод или локальный `npm run dev`) и **номера тайтлов** для Kodik — 8–10 тайтлов
   каталога, по 1–2 серии. Номер — число в адресе страницы тайтла на сайте. Можно те же тайтлы, что в BD,
   и несколько других — разнообразие важнее.

### Что такое VAL и зачем

VAL — «экзамен» для сети: серии, которые она **ни разу не видела** при обучении. По ним честно видно,
стала ли картинка лучше на любом аниме, а не только на тех тайтлах, что она запомнила.

- Откладывать нужно **тайтлы целиком**, не отдельные серии: серии одного тайтла почти одинаковы по стилю,
  и проверка на них завысит результат.
- 10 тайтлов → **8 в обучение (TRAIN), 2 в проверку (VAL)**. Если владелец может добавить ещё 2–3 тайтла
  по одной серии только для VAL — ещё лучше: тогда все 10 идут в обучение.
- VAL не улучшает сеть — он только измеряет. Решение «ставить на сайт или нет» принимается по нему.

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
# обучение: 8 тайтлов (папка TRAIN), ~20 отрезков по 4 с с каждой серии, без опенинга и эндинга
python scripts/make_pairs.py (Get-ChildItem "D:\anime\train" -Recurse -Include *.mkv,*.mp4).FullName `
  --out data/train --height 720 --segments 20 --skip-start 120 --skip-end 150

# проверка: 2 отложенных тайтла (папка VAL), другой seed
python scripts/make_pairs.py (Get-ChildItem "D:\anime\val" -Recurse -Include *.mkv,*.mp4).FullName `
  --out data/val --height 720 --segments 6 --skip-start 120 --skip-end 150 --seed 1
```

Серии с Kodik — **с этого ПК** (ссылки потоков сайта привязаны к зрителю, с сервера они не скачаются):

```powershell
# kodik.txt — по строке: «номер_тайтла серии [озвучка]», например:
#   1234 1-2
#   5678 3
python scripts/fetch_kodik.py --site https://АДРЕС_САЙТА --list kodik.txt --out data/kodik
```

Скрипт спрашивает тот же API, что плеер (`/api/v1/stream/episode`), берёт поток **720p**, сохраняет только
видео (~150–300 МБ на серию) и рядом `<имя>.skips.json` — отрезки опенинга/эндинга от Kodik; учитель на
сервере их выкинет. Если API отвечает `403` — сайт не принял запрос как «свою страницу»: попробовать
локальный `npm run dev` в репозитории animeow (`--site http://localhost:3000`) — **ничего в нём не меняя**.
Если ffmpeg не качает поток (403 от шлюза) — передать куки браузера с сайта: `--header "Cookie: …"`.

Проверить перед заливкой:
- число файлов в `data/train/hq` и `data/train/lq` одинаковое (скрипт и сам падает, если пары разъехались);
- открыть 2–3 пары глазами: `lq` заметно хуже `hq`, кадры совпадают;
- ориентиры: TRAIN 24 серии × 20 отрезков × 8 кадров ≈ 3800 пар ≈ 10 ГБ; VAL ≈ 300 пар; ~1–2 мин на серию;
  Kodik 8–10 тайтлов по 1–2 серии ≈ 2–4 ГБ видео (учитель сделает из них ещё ~1000–1500 пар на сервере).

Упаковать (tar есть в Windows 10+):

```powershell
tar -cf meow-data.tar data
```

## 3. Сервер

Образ при аренде — **Ubuntu** (22.04/24.04, лучше «с драйвером NVIDIA»), не Windows: скрипты под Linux и SSH.
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
