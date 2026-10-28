import { shortF16, wAt, type MeowConv, type MeowWeights } from "./weights";

/**
 * mpv-хуки GLSL (формат Anime4K) из весов — то же, что training/meow/glsl.py, но в браузере:
 * сайт качает компактный .meow.json (2–33 КБ), а не текст шейдера (20–120 КБ).
 * Результат можно отдать WebGL2-исполнителю animeow (anime4k-webgl.ts → parseHooks) вместо RESTORE_CNN_S.
 */
export function toMpvHook(weights: MeowWeights): string {
  const C = weights.channels;
  const T = C / 4;
  const n = weights.convs.length;
  const lines: string[] = [`// ${weights.name}: MeowRestore ${C}ch × ${weights.layers}, mpv-хуки.`, ""];
  weights.convs.forEach((conv, li) => {
    const last = li === n - 1;
    const src = li === 0 ? [] : range(T).map((t) => texName(li - 1, t));
    for (let t = 0; t < (last ? 1 : T); t++) {
      const binds = li === 0 ? ["MAIN"] : last ? ["MAIN", ...src] : src;
      const sizeFrom = li === 0 ? "MAIN" : src[0]!;
      lines.push(`//!DESC ${weights.name}-L${li}-T${t}`, "//!HOOK MAIN", ...binds.map((b) => `//!BIND ${b}`));
      lines.push(`//!SAVE ${last ? "MAIN" : texName(li, t)}`, `//!WIDTH ${sizeFrom}.w`, `//!HEIGHT ${sizeFrom}.h`);
      if (!last) lines.push("//!COMPONENTS 4");
      if (li === 0) lines.push("#define go_0(x_off, y_off) (MAIN_texOff(vec2(x_off, y_off)))");
      else
        src.forEach((s, g) => {
          lines.push(`#define go_${2 * g}(x_off, y_off) (max((${s}_texOff(vec2(x_off, y_off))), 0.0))`);
          lines.push(`#define go_${2 * g + 1}(x_off, y_off) (max(-(${s}_texOff(vec2(x_off, y_off))), 0.0))`);
        });
      lines.push("vec4 hook() {");
      let first = true;
      for (const { go, dx, dy, m } of mat4Terms(conv, t * 4, li === 0 ? null : C)) {
        const expr = `mat4(${m.map(shortF16).join(", ")}) * go_${go}(${dx.toFixed(1)}, ${dy.toFixed(1)})`;
        lines.push(`    ${first ? "vec4 result = " : "result += "}${expr};`);
        first = false;
      }
      lines.push(`    result += vec4(${bias4(conv, t * 4).map(shortF16).join(", ")});`);
      lines.push(last ? "    return result + MAIN_tex(MAIN_pos);" : "    return result;", "}", "");
    }
  });
  return lines.join("\n");
}

export interface Mat4Term {
  /** Индекс входа: 2g — max(t_g, 0), 2g+1 — max(-t_g, 0); у первого слоя — 0 (MAIN). */
  go: number;
  dx: number;
  dy: number;
  /** 16 чисел по столбцам (столбец = входной компонент). */
  m: number[];
}

/**
 * Все слагаемые «mat4 × вход» для 4 выходных каналов [outLo, outLo+4).
 * cReluFrom = null — вход сырой RGB(A), иначе — число каналов C предыдущего слоя.
 */
export function* mat4Terms(conv: MeowConv, outLo: number, cReluFrom: number | null): Generator<Mat4Term> {
  const groups: [number, (number | null)[]][] = [];
  if (cReluFrom === null) groups.push([0, [0, 1, 2, null]]);
  else
    for (let g = 0; g < cReluFrom / 4; g++) {
      groups.push([2 * g, range(4).map((k) => g * 4 + k)]);
      groups.push([2 * g + 1, range(4).map((k) => cReluFrom + g * 4 + k)]);
    }
  for (const [go, inIdx] of groups)
    for (let kx = 0; kx < 3; kx++)
      for (let ky = 0; ky < 3; ky++) {
        const m: number[] = [];
        for (let j = 0; j < 4; j++)
          for (let i = 0; i < 4; i++) {
            const o = outLo + i;
            const ci = inIdx[j];
            m.push(ci === null || ci === undefined || o >= conv.cout ? 0 : wAt(conv, o, ci, ky, kx));
          }
        yield { go, dx: kx - 1, dy: ky - 1, m };
      }
}

export function bias4(conv: MeowConv, lo: number): number[] {
  return range(4).map((k) => (lo + k < conv.cout ? conv.b[lo + k]! : 0));
}

export function texName(layer: number, t: number): string {
  return `meow_${layer}_${t}`;
}

export function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}
