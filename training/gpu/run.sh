#!/usr/bin/env bash
# Полный прогон на GPU-сервере: (учитель) → обучение L, M, S → сравнение → выгрузка → results.tar.
# Запускать в tmux, чтобы обрыв SSH ничего не прервал:
#   tmux new -d -s meow 'bash training/gpu/run.sh 2>&1 | tee training/run.log'
#   tail -f training/run.log
#
# Ожидает (относительно training/):
#   data/train/{hq,lq}   — пары из make_pairs.py (обязательно)
#   data/val/{hq,lq}     — пары с ДРУГИХ тайтлов (желательно)
#   data/kodik/*.mp4     — серии с Kodik для учителя Real-ESRGAN (необязательно; fetch_kodik.py кладёт рядом
#                          <имя>.skips.json — тогда опенинг/эндинг выкидываются по точной разметке Kodik)
# Настройки — переменными окружения: STEPS_L=60000 STEPS_M=60000 STEPS_S=40000 CACHE=1 WORKERS=10
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

STEPS_L=${STEPS_L:-60000}
STEPS_M=${STEPS_M:-60000}
STEPS_S=${STEPS_S:-40000}
WORKERS=${WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}
[ -d data/train/hq ] || { echo "нет data/train — сначала залейте датасет"; exit 1; }

DATA="data/train"
if compgen -G "data/kodik/*.mp4" >/dev/null || compgen -G "data/kodik/*.mkv" >/dev/null; then
  if [ ! -d data/teacher/hq ]; then
    echo "== учитель Real-ESRGAN по сериям Kodik"
    args=(); for v in data/kodik/*.mp4 data/kodik/*.mkv; do [ -f "$v" ] && args+=(--video "$v"); done
    python scripts/make_teacher.py "${args[@]}" --every 240 --skip-start 120 --skip-end 150 --out data/teacher
  fi
  DATA="$DATA data/teacher"
fi

# Кадры в памяти: ~5.5 МБ на пару 720p. Без кэша PNG читаются с диска (медленнее, но памяти не нужно).
PAIRS=$(find data/*/hq -name '*.png' 2>/dev/null | wc -l)
MEM_GB=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
NEED_GB=$(( PAIRS * 55 / 10000 + 4 ))
CACHE=${CACHE:-$([ "$NEED_GB" -lt "$MEM_GB" ] && echo 1 || echo 0)}
CACHE_FLAG=$([ "$CACHE" = 1 ] && echo --cache || echo "")
echo "пар: $PAIRS, памяти: ${MEM_GB} ГБ (нужно ~${NEED_GB} ГБ для кэша) → cache=$CACHE, workers=$WORKERS"


ckpt() { [ -f "runs/$1/best.pt" ] && echo "runs/$1/best.pt" || echo "runs/$1/last.pt"; }

train() { # размер шаги [доп. аргументы]
  local size=$1 steps=$2; shift 2
  if [ -f "runs/$size/last.pt" ]; then echo "== $size уже обучена, пропуск"; return; fi
  echo "== обучение $size ($steps шагов)"
  python scripts/train.py --data $DATA --size "$size" --steps "$steps" --batch 32 --crop 128 \
    --val-every 2000 --workers "$WORKERS" $CACHE_FLAG --out "runs/$size" "$@"
}
train l "$STEPS_L"
train m "$STEPS_M"
train s "$STEPS_S" --teacher "$(ckpt m)"

mkdir -p out/models
for z in s m l; do python scripts/export.py "$(ckpt $z)" --name "meow-restore-$z" --out out/models; done

if [ -d data/val/hq ]; then
  echo "== проверка на отложенных тайтлах"
  python scripts/eval.py out/models/meow-restore-{s,m,l}.meow.json --data data/val --limit 200 --out out/eval | tee out/eval.txt
fi
for z in s m l; do cp "runs/$z/log.csv" "out/log_$z.csv"; cp "$(ckpt $z)" "out/meow-restore-$z.pt"; done
tar -cf results.tar out
echo "== ГОТОВО: training/results.tar ($(du -h results.tar | cut -f1))"
