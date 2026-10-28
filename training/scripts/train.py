#!/usr/bin/env python3
"""
Обучение MeowRestore.

    python scripts/train.py --data data/bd data/teacher --size m --steps 60000 --out runs/m1
    python scripts/train.py --data data/bd --size s --init runs/m1/best.pt --distill ...   (см. docs/TRAINING.md)

В конце (и при каждом улучшении на валидации) сохраняет runs/<имя>/best.pt и выгружает
best.meow.json (веса для сайта) и best.glsl (mpv-хук) — их сразу можно смотреть в mpv.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meow import weights  # noqa: E402
from meow.data import PairDataset, find_pairs, val_crops  # noqa: E402
from meow.glsl import to_glsl  # noqa: E402
from meow.model import SIZES, MeowRestore  # noqa: E402


def charbonnier(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


_SOBEL = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)


def edges(x: torch.Tensor) -> torch.Tensor:
    k = torch.stack([_SOBEL, _SOBEL.T])[:, None].to(x)  # [2,1,3,3]
    g = F.conv2d(x.mean(1, keepdim=True), k, padding=1)
    return g


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b).item()
    return 99.0 if mse == 0 else 10 * math.log10(1 / mse)


@torch.no_grad()
def evaluate(model: MeowRestore, val, device) -> tuple[float, float]:
    model.eval()
    p_in = p_out = 0.0
    for lq, hq in val:
        lq, hq = lq[None].to(device), hq[None].to(device)
        p_in += psnr(lq, hq)
        p_out += psnr(model(lq), hq)
    model.train()
    return p_in / len(val), p_out / len(val)


def export(model: MeowRestore, out: Path, name: str) -> dict:
    meta = weights.from_model(model, name)
    weights.save(meta, out.with_suffix(".meow.json"))
    out.with_suffix(".glsl").write_text(to_glsl(meta))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--size", choices=list(SIZES), default="m")
    ap.add_argument("--channels", type=int, help="свой размер вместо --size")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--edge-weight", type=float, default=0.1, help="вес потерь на контурах (Собель)")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--init", type=Path, help="продолжить с чекпойнта (того же размера)")
    ap.add_argument("--teacher", type=Path, help="чекпойнт большей MeowRestore: учиться и на её ответах (дистилляция)")
    ap.add_argument("--teacher-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", action="store_true", help="держать кадры в памяти (небольшие наборы)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.data)
    random.Random(1).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:] or pairs
    val = val_crops(val_pairs)
    print(f"пар: {len(train_pairs)} обучение, {len(val_pairs)} валидация; устройство: {device}")

    model = MeowRestore(args.channels, args.layers) if args.channels else MeowRestore.from_size(args.size)
    name = f"meow-restore-{args.size if not args.channels else f'c{args.channels}l{args.layers}'}"
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    model.to(device)
    teacher = None
    if args.teacher:
        ck = torch.load(args.teacher, map_location="cpu")
        teacher = MeowRestore(**ck["config"])
        teacher.load_state_dict(ck["model"])
        teacher.to(device).eval().requires_grad_(False)
    print(f"{name}: {model.num_params()} параметров ({model.num_params() * 2 / 1024:.1f} КБ fp16)")

    loader = DataLoader(
        PairDataset(train_pairs, args.crop, samples_per_epoch=args.steps * args.batch, cache=args.cache),
        batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    p_in, p_out = evaluate(model, val, device)
    best = p_out
    print(f"старт: вход {p_in:.2f} дБ, сеть {p_out:.2f} дБ")
    log = csv.writer(open(args.out / "log.csv", "w", newline=""))
    log.writerow(["step", "loss", "psnr_in", "psnr_out", "gain_db", "sec"])
    t0 = time.time()
    running = 0.0
    for step, (lq, hq) in enumerate(loader, 1):
        lq, hq = lq.to(device, non_blocking=True), hq.to(device, non_blocking=True)
        out = model(lq)
        loss = charbonnier(out, hq) + args.edge_weight * charbonnier(edges(out), edges(hq))
        if teacher is not None:
            with torch.no_grad():
                t_out = teacher(lq)
            loss = loss + args.teacher_weight * charbonnier(out, t_out)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        if step % args.val_every == 0 or step == args.steps:
            p_in, p_out = evaluate(model, val, device)
            gain = p_out - p_in
            print(f"[{step}/{args.steps}] loss {running / args.val_every:.4f}  вход {p_in:.2f} дБ → сеть {p_out:.2f} дБ ({gain:+.2f})  {time.time() - t0:.0f} с")
            log.writerow([step, f"{running / args.val_every:.5f}", f"{p_in:.3f}", f"{p_out:.3f}", f"{gain:.3f}", int(time.time() - t0)])
            running = 0.0
            if p_out > best:
                best = p_out
                torch.save({"model": model.state_dict(), "config": model.config(), "step": step, "psnr": p_out, "gain": gain}, args.out / "best.pt")
                export(model.cpu(), args.out / "best", name)
                model.to(device)
        if step >= args.steps:
            break
    torch.save({"model": model.state_dict(), "config": model.config(), "step": args.steps}, args.out / "last.pt")
    if not (args.out / "best.pt").exists():
        print("сеть не обогнала вход на валидации — best не сохранён (мало данных/шагов?)")
    else:
        print(f"готово: {args.out / 'best.meow.json'}, {args.out / 'best.glsl'}")


if __name__ == "__main__":
    main()
