"""
Генерация mpv-хуков GLSL (формат Anime4K) из весов `.meow.json`.

Результат работает:
  * в mpv: --glsl-shaders="meow-restore-m.glsl;Anime4K_Upscale_CNN_x2_M.glsl" — удобно смотреть на реальных сериях;
  * в WebGL2-исполнителе animeow (app/lib/anime4k-webgl.ts) — вместо RESTORE_CNN_S, без правок исполнителя.

Соглашения, общие с Anime4K:
  * mat4 в GLSL — по столбцам: out[i] = Σj m[j*4+i] · v[j];
  * texOff(vec2(x, y)): +y — вниз (строка 0 — верх кадра), ядро свёртки W[.., ky, kx] берёт смещение (kx-1, ky-1);
  * go_{2g} = max(t_g, 0), go_{2g+1} = max(-t_g, 0) — положительная и отрицательная части CReLU.
"""
from __future__ import annotations

import numpy as np

from .weights import unpack

OFFSETS = [(kx - 1, ky - 1, ky, kx) for kx in range(3) for ky in range(3)]  # порядок как у Anime4K: x-major


def _f(v: float) -> str:
    """Кратчайшая запись, которая округляется в то же fp16 (веса и так fp16) — текст шейдера в ~3 раза короче."""
    s = str(np.float16(v))
    if "e" in s and "." not in s.split("e")[0]:
        m, e = s.split("e")
        s = f"{m}.0e{e}"
    return s if ("." in s or "e" in s) else s + ".0"


def tex_name(layer: int, t: int) -> str:
    return f"meow_{layer}_{t}"


def _input_cols(w: np.ndarray, out_lo: int, cin: int, crelu_src_channels: int | None):
    """
    Для 4 выходных каналов [out_lo, out_lo+4) — список (go-индекс, ky, kx, 16 чисел mat4 по столбцам).
    crelu_src_channels=None — вход сырой RGB(A) (первый слой).
    """
    groups = []
    if crelu_src_channels is None:
        groups.append((0, list(range(3)) + [None]))  # RGB, альфа с нулевым весом
    else:
        C = crelu_src_channels
        for g in range(C // 4):
            pos = [g * 4 + k for k in range(4)]
            neg = [C + g * 4 + k for k in range(4)]
            groups.append((2 * g, pos))
            groups.append((2 * g + 1, neg))
    for go, in_idx in groups:
        for x_off, y_off, ky, kx in OFFSETS:
            m = []
            for j in range(4):  # столбец = входной компонент
                for i in range(4):  # строка = выходной компонент
                    o = out_lo + i
                    ci = in_idx[j]
                    m.append(0.0 if (ci is None or o >= w.shape[0]) else float(w[o, ci, ky, kx]))
            yield go, x_off, y_off, m


def to_glsl(meta: dict) -> str:
    layers = unpack(meta)
    C = meta["channels"]
    T = C // 4
    name = meta.get("name", "meow-restore")
    out = [
        f"// {name}: MeowRestore {C}ch × {meta['layers']} свёрток, формат mpv-хуков (как Anime4K).",
        "// Сгенерировано animeow-upscale/training/meow/glsl.py из .meow.json — не править руками.",
        "",
    ]
    n = len(layers)
    for li, (w, b) in enumerate(layers):
        last = li == n - 1
        src = None if li == 0 else [tex_name(li - 1, t) for t in range(T)]
        for t in range(1 if last else T):
            binds = ["MAIN"] if li == 0 else (["MAIN", *src] if last else src)
            size_from = "MAIN" if li == 0 else src[0]
            save = "MAIN" if last else tex_name(li, t)
            out.append(f"//!DESC {name}-L{li}-T{t}")
            out.append("//!HOOK MAIN")
            out += [f"//!BIND {x}" for x in binds]
            out.append(f"//!SAVE {save}")
            out.append(f"//!WIDTH {size_from}.w")
            out.append(f"//!HEIGHT {size_from}.h")
            if not last:
                out.append("//!COMPONENTS 4")
            if li == 0:
                out.append("#define go_0(x_off, y_off) (MAIN_texOff(vec2(x_off, y_off)))")
            else:
                for g, s in enumerate(src):
                    out.append(f"#define go_{2 * g}(x_off, y_off) (max(({s}_texOff(vec2(x_off, y_off))), 0.0))")
                    out.append(f"#define go_{2 * g + 1}(x_off, y_off) (max(-({s}_texOff(vec2(x_off, y_off))), 0.0))")
            out.append("vec4 hook() {")
            first = True
            for go, x_off, y_off, m in _input_cols(w, t * 4, w.shape[1], None if li == 0 else C):
                expr = f"mat4({', '.join(_f(v) for v in m)}) * go_{go}({_f(x_off)}, {_f(y_off)})"
                out.append(f"    {'vec4 result = ' if first else 'result += '}{expr};")
                first = False
            bias = [float(b[t * 4 + k]) if t * 4 + k < len(b) else 0.0 for k in range(4)]
            out.append(f"    result += vec4({', '.join(_f(v) for v in bias)});")
            out.append("    return result + MAIN_tex(MAIN_pos);" if last else "    return result;")
            out.append("}")
            out.append("")
    return "\n".join(out)
