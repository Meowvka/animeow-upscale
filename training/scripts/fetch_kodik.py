#!/usr/bin/env python3
"""
Серии с Kodik в 720p — через API самого сайта animeow (/api/v1/stream/episode), как их получает плеер.
Сохраняет только видео (звук не нужен) и рядом — отрезки опенинга/эндинга от Kodik (<имя>.skips.json),
которые make_teacher.py потом выкидывает.

    python scripts/fetch_kodik.py --site https://САЙТ --title 123 --episodes 1-3 --out data/kodik
    python scripts/fetch_kodik.py --site https://САЙТ --list kodik.txt --out data/kodik

kodik.txt — по строке на тайтл: «номер_тайтла серии [озвучка]», например:
    123 1-3
    456 2,5 610

Номер тайтла — как в адресе страницы тайтла на сайте; озвучка — номер из выбора озвучки (необязательно).
Эндпоинт отдаёт ссылки только «своим страницам» (lib/own-page.ts) — скрипт представляется своей страницей
(Sec-Fetch-Site: same-origin). Это сайт владельца; на чужих сайтах так не делать.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"


def parse_episodes(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part.strip():
            out.append(int(part))
    return out


def resolve(site: str, title: int, episode: int, voice: int | None, extra_headers: list[str]) -> dict:
    url = f"{site}/api/v1/stream/episode?title={title}&episode={episode}" + (f"&voice={voice}" if voice else "")
    headers = {"User-Agent": UA, "Sec-Fetch-Site": "same-origin", "Origin": site, "Referer": f"{site}/", "Accept": "application/json"}
    for h in extra_headers:
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf8", "replace")
            if e.code in (502, 503, 429) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise SystemExit(f"{url}: HTTP {e.code} {body}")
    raise SystemExit(f"{url}: не ответил")


def pick_quality(qualities: dict[str, str], want: int) -> tuple[str, str]:
    nums = sorted((int(q), u) for q, u in qualities.items() if q.isdigit())
    if not nums:
        raise SystemExit(f"нет потоков: {qualities}")
    below = [x for x in nums if x[0] <= want]
    q, u = (below[-1] if below else nums[0])
    return str(q), u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="адрес сайта, например https://animeow.example (или http://localhost:3000)")
    ap.add_argument("--title", type=int)
    ap.add_argument("--episodes", default="1")
    ap.add_argument("--voice", type=int)
    ap.add_argument("--list", type=Path)
    ap.add_argument("--quality", type=int, default=720)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--header", action="append", default=[], help="доп. заголовок к API и к потоку, «Имя: значение» (например Cookie)")
    args = ap.parse_args()
    site = args.site.rstrip("/")

    jobs: list[tuple[int, int, int | None]] = []
    if args.list:
        for line in args.list.read_text(encoding="utf8").splitlines():
            parts = line.split("#")[0].split()
            if parts:
                jobs += [(int(parts[0]), e, int(parts[2]) if len(parts) > 2 else None) for e in parse_episodes(parts[1] if len(parts) > 1 else "1")]
    elif args.title:
        jobs = [(args.title, e, args.voice) for e in parse_episodes(args.episodes)]
    else:
        raise SystemExit("нужно --title или --list")

    args.out.mkdir(parents=True, exist_ok=True)
    for i, (title, ep, voice) in enumerate(jobs, 1):
        name = f"t{title}_e{ep:03d}"
        dst = args.out / f"{name}.mp4"
        if dst.exists():
            print(f"[{i}/{len(jobs)}] {name}: уже есть")
            continue
        data = resolve(site, title, ep, voice, args.header)
        if data.get("status") != "ok":
            print(f"[{i}/{len(jobs)}] {name}: {data.get('status')} — пропуск", file=sys.stderr)
            continue
        q, url = pick_quality(data.get("qualities") or {}, args.quality)
        if url.startswith("/"):
            url = site + url
        tmp = dst.with_suffix(".part.mp4")
        ff_headers = f"Referer: {site}/\r\n" + "".join(f"{h.strip()}\r\n" for h in args.header)
        cmd = ["ffmpeg", "-y", "-v", "error", "-user_agent", UA, "-headers", ff_headers,
               "-i", url, "-map", "0:v:0", "-c", "copy", str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 100_000:
            tmp.unlink(missing_ok=True)
            print(f"[{i}/{len(jobs)}] {name}: ffmpeg не скачал ({r.stderr.strip()[-300:]})", file=sys.stderr)
            continue
        tmp.rename(dst)
        (args.out / f"{name}.skips.json").write_text(json.dumps(data.get("skips") or []))
        print(f"[{i}/{len(jobs)}] {name}: {q}p, {dst.stat().st_size / 1e6:.0f} МБ, заставок: {len(data.get('skips') or [])}")
        time.sleep(2)  # у эндпоинта лимит запросов


if __name__ == "__main__":
    main()
