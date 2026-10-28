/** Скорость на этой машине: постер 460×690 → 920×1380 и кадр 1280×720 (восстановление, scale 1). */
import sharp from "sharp";
import { createEnhancer } from "./enhance.ts";

const threads = Number(process.argv[2] ?? 0) || undefined;
const enhancer = await createEnhancer({ threads });
const noise = (w: number, h: number) =>
  sharp({ create: { width: w, height: h, channels: 3, background: "#8af" } })
    .composite([{ input: Buffer.from(`<svg width="${w}" height="${h}"><circle cx="${w / 2}" cy="${h / 2}" r="${h / 3}" fill="#f6a" stroke="#222" stroke-width="4"/></svg>`) }])
    .png()
    .toBuffer();

for (const [name, w, h, scale] of [["постер 460×690 ×2", 460, 690, 2], ["кадр 1280×720 ×1", 1280, 720, 1]] as const) {
  const img = await noise(w, h);
  const t0 = performance.now();
  await enhancer.enhance(img, { scale });
  console.log(`${name}: ${((performance.now() - t0) / 1000).toFixed(1)} с (потоков: ${threads ?? "все"})`);
}
await enhancer.close();
