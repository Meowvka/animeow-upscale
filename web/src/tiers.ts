/**
 * Какой размер MeowRestore брать для видео. Решает «потолок», а реальную скорость проверяет плеер
 * (как сейчас в anime4k-enhance.ts: пропущена треть кадров — шаг вниз).
 *
 *   l — десктоп с WebGPU (≈16.6k MAC/пиксель, ~30 GFLOP на кадр 720p)
 *   m — WebGPU на ноутбуке/сильном телефоне, или WebGL2 на десктопе (≈5.3k MAC/пиксель)
 *   s — телефоны на WebGL2 (≈0.9k MAC/пиксель, дешевле Anime4K Restore S)
 */
export type MeowSize = "s" | "m" | "l";

export interface DeviceHints {
  webgpu: boolean;
  mobile: boolean;
  /** navigator.deviceMemory, ГБ. */
  memoryGb?: number;
  /** Высота кадра видео. */
  frameHeight: number;
}

export function pickSize(d: DeviceHints): MeowSize {
  // Чем больше кадр, тем дороже каждый слой: 1080p — в 2.25 раза дороже 720p.
  const heavy = d.frameHeight > 800;
  if (d.mobile) {
    if (d.webgpu && (d.memoryGb ?? 4) >= 6 && !heavy) return "m";
    return "s";
  }
  if (d.webgpu) return heavy ? "m" : "l";
  return heavy ? "s" : "m";
}

/** На ступень легче — когда устройство не успевает за видео. */
export function stepDown(s: MeowSize): MeowSize | null {
  return s === "l" ? "m" : s === "m" ? "s" : null;
}

/** Умножений на пиксель для модели C каналов и L свёрток (для своих размеров). */
export function macsPerPixel(channels: number, layers: number): number {
  return 3 * channels * 9 + (layers - 2) * 2 * channels * channels * 9 + 2 * channels * 3 * 9;
}
