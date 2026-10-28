/**
 * Улучшение картинок на сервере: Real-ESRGAN AnimeVideo-v3 (×4) через onnxruntime-node на CPU.
 * Один раз на картинку, пользователю — готовый файл: ни веса, ни нагрузки на устройстве.
 *
 * Сеть всегда делает ×4; нужный масштаб получаем уменьшением результата (lanczos):
 *   scale 2 — постер 460 → 920 (вместо Anime4K Restore+Upscale);
 *   scale 1 — то же разрешение, но без блоков сжатия и мыла (баннеры, кадры) — сеть ×4, затем обратно.
 */
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import sharp from "sharp";

export const DEFAULT_MODEL = fileURLToPath(new URL("../../models/realesrgan/animevideov3-x4.fp32.onnx", import.meta.url));
const NET_SCALE = 4;

export interface EnhancerOptions {
  /** Путь к .onnx (по умолчанию fp32 — на CPU fp16 не быстрее). */
  model?: string;
  /** Потоков ORT на картинку. На сервере с N ядрами: процессов × threads ≈ N - 1. */
  threads?: number;
  /** Тайл во входных пикселях: меньше — меньше памяти, чуть медленнее. */
  tile?: number;
}

export interface EnhanceOptions {
  /** Итоговый масштаб к исходнику (1..4). */
  scale: number;
  /** Не больше этой стороны на выходе (защита от гигантских картинок). */
  maxSide?: number;
}

export interface RawImage {
  data: Buffer;
  width: number;
  height: number;
  channels: 3;
}

export interface Enhancer {
  /**
   * Та же форма, что у enhanceRgb из animeow (metadata/lib/anime4k-cpu.ts): RGB по 3 байта, строка 0 — верх.
   * upscale: true — ×2 (постеры), false — тот же размер, только восстановление.
   */
  enhanceRgb(rgb: Uint8Array, w: number, h: number, opts: { upscale: boolean }): Promise<{ rgb: Uint8Array; w: number; h: number }>;
  /** Любой вход, который понимает sharp (путь, Buffer) → сырые RGB-пиксели результата. */
  enhance(input: string | Buffer, opts: EnhanceOptions): Promise<RawImage>;
  /** То же, сразу в файл (формат по расширению: .webp/.jpg/.png). */
  enhanceFile(input: string | Buffer, output: string, opts: EnhanceOptions & { quality?: number }): Promise<RawImage>;
  close(): Promise<void>;
}

export async function createEnhancer(opts: EnhancerOptions = {}): Promise<Enhancer> {
  const session = await ort.InferenceSession.create(opts.model ?? DEFAULT_MODEL, {
    executionProviders: ["cpu"],
    intraOpNumThreads: opts.threads ?? 0,
    graphOptimizationLevel: "all",
  });
  const tile = opts.tile ?? 192;
  const pad = 8;

  async function upscale4(rgb: Uint8Array, w: number, h: number): Promise<Float32Array> {
    const W = w * NET_SCALE;
    const H = h * NET_SCALE;
    const out = new Float32Array(W * H * 3); // HWC
    for (let y0 = 0; y0 < h; y0 += tile) {
      for (let x0 = 0; x0 < w; x0 += tile) {
        const y1 = Math.min(h, y0 + tile);
        const x1 = Math.min(w, x0 + tile);
        const sy = Math.max(0, y0 - pad);
        const sx = Math.max(0, x0 - pad);
        const ey = Math.min(h, y1 + pad);
        const ex = Math.min(w, x1 + pad);
        const tw = ex - sx;
        const th = ey - sy;
        const plane = tw * th;
        const input = new Float32Array(plane * 3);
        for (let y = 0; y < th; y++)
          for (let x = 0; x < tw; x++) {
            const s = ((sy + y) * w + sx + x) * 3;
            const d = y * tw + x;
            input[d] = rgb[s]! / 255;
            input[d + plane] = rgb[s + 1]! / 255;
            input[d + 2 * plane] = rgb[s + 2]! / 255;
          }
        const res = await session.run({ input: new ort.Tensor("float32", input, [1, 3, th, tw]) });
        const o = res.output!.data as Float32Array;
        const ow = tw * NET_SCALE;
        const oplane = ow * th * NET_SCALE;
        // Пишем только ядро тайла (без перекрытия) — швов нет.
        for (let y = (y0 - sy) * NET_SCALE; y < (y1 - sy) * NET_SCALE; y++)
          for (let x = (x0 - sx) * NET_SCALE; x < (x1 - sx) * NET_SCALE; x++) {
            const s = y * ow + x;
            const d = ((sy * NET_SCALE + y) * W + sx * NET_SCALE + x) * 3;
            out[d] = o[s]!;
            out[d + 1] = o[s + oplane]!;
            out[d + 2] = o[s + 2 * oplane]!;
          }
      }
    }
    return out;
  }

  async function scaleRaw(rgb: Uint8Array, w: number, h: number, o: EnhanceOptions): Promise<RawImage> {
    let tw = Math.round(w * o.scale);
    let th = Math.round(h * o.scale);
    if (o.maxSide && Math.max(tw, th) > o.maxSide) {
      const k = o.maxSide / Math.max(tw, th);
      tw = Math.round(tw * k);
      th = Math.round(th * k);
    }
    const big = await upscale4(rgb, w, h);
    const u8 = Buffer.alloc(big.length);
    for (let i = 0; i < big.length; i++) u8[i] = Math.max(0, Math.min(255, Math.round(big[i]! * 255)));
    const resized = await sharp(u8, { raw: { width: w * NET_SCALE, height: h * NET_SCALE, channels: 3 } })
      .resize(tw, th, { kernel: "lanczos3" })
      .raw()
      .toBuffer();
    return { data: resized, width: tw, height: th, channels: 3 };
  }

  async function enhance(input: string | Buffer, o: EnhanceOptions): Promise<RawImage> {
    const { data, info } = await sharp(input).removeAlpha().toColourspace("srgb").raw().toBuffer({ resolveWithObject: true });
    return scaleRaw(data, info.width, info.height, o);
  }

  return {
    enhance,
    async enhanceRgb(rgb, w, h, { upscale }) {
      const r = await scaleRaw(rgb, w, h, { scale: upscale ? 2 : 1 });
      return { rgb: new Uint8Array(r.data.buffer, r.data.byteOffset, r.data.length), w: r.width, h: r.height };
    },
    async enhanceFile(input, output, o) {
      const img = await enhance(input, o);
      let s = sharp(img.data, { raw: { width: img.width, height: img.height, channels: 3 } });
      const q = o.quality ?? 85;
      if (/\.webp$/i.test(output)) s = s.webp({ quality: q, effort: 5 });
      else if (/\.jpe?g$/i.test(output)) s = s.jpeg({ quality: q, mozjpeg: true });
      else s = s.png();
      await s.toFile(output);
      return img;
    },
    close: () => session.release(),
  };
}
