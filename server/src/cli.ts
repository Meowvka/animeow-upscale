/**
 * npx tsx src/cli.ts --scale 2 in.webp out.webp
 * npx tsx src/cli.ts --scale 1 --in media/_orig/banners --out media/banners   (папка; уже сделанные пропускает)
 */
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createEnhancer } from "./enhance.ts";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    scale: { type: "string", default: "2" },
    "max-side": { type: "string" },
    quality: { type: "string", default: "85" },
    threads: { type: "string" },
    model: { type: "string" },
    in: { type: "string" },
    out: { type: "string" },
  },
});

const enhancer = await createEnhancer({ model: values.model, threads: values.threads ? Number(values.threads) : undefined });
const opts = { scale: Number(values.scale), maxSide: values["max-side"] ? Number(values["max-side"]) : undefined, quality: Number(values.quality) };

const jobs: [string, string][] = [];
if (values.in && values.out) {
  mkdirSync(values.out, { recursive: true });
  for (const f of readdirSync(values.in).filter((f) => /\.(webp|jpe?g|png)$/i.test(f)).sort()) {
    const dst = join(values.out, f);
    if (!existsSync(dst)) jobs.push([join(values.in, f), dst]);
  }
} else if (positionals.length === 2) {
  jobs.push([positionals[0]!, positionals[1]!]);
} else {
  console.error("usage: cli.ts [--scale 2] in out  |  cli.ts --in dir --out dir");
  process.exit(2);
}

for (const [i, [src, dst]] of jobs.entries()) {
  const t0 = performance.now();
  const r = await enhancer.enhanceFile(src, dst, opts);
  console.log(`[${i + 1}/${jobs.length}] ${src} → ${dst} ${r.width}×${r.height} ${((performance.now() - t0) / 1000).toFixed(1)} с`);
}
await enhancer.close();
