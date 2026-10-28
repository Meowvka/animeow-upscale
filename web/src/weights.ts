/** Формат `.meow.json` (см. training/meow/weights.py). */
export interface MeowWeightsJson {
  format: "meow-restore/1";
  name: string;
  channels: number;
  layers: number;
  dtype: "float16";
  layers_data: { in: number; out: number; w: string; b: string }[];
}

export interface MeowConv {
  cin: number;
  cout: number;
  /** [out][in][ky][kx], float32 (значения fp16-точные). */
  w: Float32Array;
  b: Float32Array;
}

export interface MeowWeights {
  name: string;
  channels: number;
  layers: number;
  convs: MeowConv[];
}

export function parseWeights(json: MeowWeightsJson | string): MeowWeights {
  const meta: MeowWeightsJson = typeof json === "string" ? JSON.parse(json) : json;
  if (meta.format !== "meow-restore/1") throw new Error(`unknown weights format: ${meta.format}`);
  if (meta.channels % 4) throw new Error("channels must be a multiple of 4");
  return {
    name: meta.name,
    channels: meta.channels,
    layers: meta.layers,
    convs: meta.layers_data.map((l) => ({
      cin: l.in,
      cout: l.out,
      w: decodeF16(l.w, l.out * l.in * 9),
      b: decodeF16(l.b, l.out),
    })),
  };
}

export async function loadWeights(url: string): Promise<MeowWeights> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`weights fetch failed: ${res.status}`);
  return parseWeights(await res.json());
}

/** Вес свёртки W[o][i][ky][kx]. */
export function wAt(c: MeowConv, o: number, i: number, ky: number, kx: number): number {
  return c.w[((o * c.cin + i) * 3 + ky) * 3 + kx]!;
}

function decodeF16(b64: string, n: number): Float32Array {
  const bin = atob(b64);
  if (bin.length !== n * 2) throw new Error(`bad weights length: ${bin.length} != ${n * 2}`);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = f16ToF32(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8));
  return out;
}

export function f16ToF32(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export function f32ToF16(v: number): number {
  const f = new Float32Array([v]);
  const x = new Uint32Array(f.buffer)[0]!;
  const sign = (x >> 16) & 0x8000;
  const exp = ((x >> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (((x >> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let half = mant >> shift;
    const rem = mant & ((1 << shift) - 1);
    const mid = 1 << (shift - 1);
    if (rem > mid || (rem === mid && half & 1)) half++;
    return sign | half;
  }
  let half = sign | (exp << 10) | (mant >> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && half & 1)) half++;
  return half;
}

/** Кратчайшая десятичная запись, которая округляется в то же fp16 — компактный текст шейдера. */
export function shortF16(v: number): string {
  if (v === 0) return "0.0";
  const target = f32ToF16(v);
  for (let p = 1; p <= 6; p++) {
    const s = v.toPrecision(p);
    if (f32ToF16(Number(s)) === target) return normalizeFloat(s);
  }
  return normalizeFloat(String(v));
}

function normalizeFloat(s: string): string {
  return /[.e]/.test(s) ? s : `${s}.0`;
}
