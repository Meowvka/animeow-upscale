// Сравнение сетей восстановления на парах кадров в настоящем Chromium (WebGL2, тот же исполнитель, что у сайта):
//   node test/compare.mjs --pairs ../training/data/val --limit 12 \
//     meow-s=../models/meow/meow-restore-s.meow.json a4k-S=/path/Anime4K_Restore_CNN_S.glsl …
// Печатает PSNR (дБ) против эталона и выигрыш относительно входа; --out dir — картинки «вход | сети… | эталон».
import { build } from "esbuild";
import { createServer } from "node:http";
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { pairs: { type: "string" }, limit: { type: "string", default: "12" }, crop: { type: "string", default: "384" }, out: { type: "string" } },
});
const pairsDir = resolve(values.pairs);
const names = readdirSync(join(pairsDir, "hq")).filter((f) => f.endsWith(".png")).sort();
const step = Math.max(1, Math.floor(names.length / Number(values.limit)));
const frames = names.filter((_, i) => i % step === 0).slice(0, Number(values.limit));
const models = positionals.map((p) => {
  const [name, file] = p.split("=");
  return { name, kind: file.endsWith(".json") ? "json" : "glsl", text: readFileSync(resolve(file), "utf8") };
});

const root = new URL(".", import.meta.url).pathname;
await build({
  stdin: {
    contents: `
      import { createHookRunner, parseWeights, toMpvHook } from "../src/index";
      window.__run = async ({ models, frames, crop }) => {
        const gl = document.createElement("canvas").getContext("webgl2");
        const load = async (url) => {
          const bmp = await createImageBitmap(await (await fetch(url)).blob());
          const s = Math.min(crop, bmp.width, bmp.height);
          const c = new OffscreenCanvas(s, s).getContext("2d");
          c.drawImage(bmp, (bmp.width - s) >> 1, (bmp.height - s) >> 1, s, s, 0, 0, s, s);
          const d = c.getImageData(0, 0, s, s).data;
          return { s, rgba: Float32Array.from(d, (v) => v / 255) };
        };
        const psnr = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) if (i % 4 !== 3) { const e = Math.min(1, Math.max(0, a[i])) - b[i]; m += e * e; } return 10 * Math.log10(1 / (m / (a.length * 0.75))); };
        const runners = models.map((m) => ({ name: m.name, r: createHookRunner(gl, m.kind === "json" ? toMpvHook(parseWeights(m.text)) : m.text) }));
        const scores = { input: [] }; runners.forEach((x) => (scores[x.name] = []));
        const shots = [];
        for (const f of frames) {
          const lq = await load("/pairs/lq/" + f), hq = await load("/pairs/hq/" + f);
          scores.input.push(psnr(lq.rgba, hq.rgba));
          const outs = [lq.rgba];
          for (const { name, r } of runners) { const o = r.run(lq.rgba, lq.s, lq.s); scores[name].push(psnr(o.data, hq.rgba)); outs.push(o.data); }
          outs.push(hq.rgba);
          if (shots.length < 4) {
            const s = lq.s, W = s * outs.length, img = new ImageData(W, s);
            outs.forEach((o, k) => { for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) for (let c = 0; c < 4; c++) img.data[(y * W + k * s + x) * 4 + c] = c === 3 ? 255 : Math.round(Math.min(1, Math.max(0, o[(y * s + x) * 4 + c])) * 255); });
            const cv = new OffscreenCanvas(W, s); cv.getContext("2d").putImageData(img, 0, 0);
            const b = await cv.convertToBlob({ type: "image/png" });
            shots.push(Array.from(new Uint8Array(await b.arrayBuffer())));
          }
        }
        return { scores, shots };
      };`,
    resolveDir: root,
    loader: "js",
  },
  bundle: true,
  format: "esm",
  outfile: join(root, ".out/compare.js"),
  logLevel: "warning",
});

const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p === "/") return res.writeHead(200, { "content-type": "text/html" }).end(`<script type="module" src="/.out/compare.js"></script>`);
  try {
    const file = p.startsWith("/pairs/") ? join(pairsDir, basename(p.split("/")[2]), basename(p)) : join(root, p);
    res.writeHead(200, { "content-type": p.endsWith(".js") ? "text/javascript" : "image/png" }).end(readFileSync(file));
  } catch {
    res.writeHead(404).end();
  }
}).listen(0);

const browser = await chromium.launch({ args: ["--enable-unsafe-swiftshader"] });
const page = await browser.newPage();
await page.goto(`http://localhost:${server.address().port}/`);
await page.waitForFunction(() => window.__run);
const { scores, shots } = await page.evaluate((a) => window.__run(a), { models, frames, crop: Number(values.crop) });
await browser.close();
server.close();

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const base = mean(scores.input);
console.log(`кадров: ${frames.length}, вырезка ${values.crop}px`);
for (const [k, v] of Object.entries(scores)) console.log(`${k.padEnd(14)} ${mean(v).toFixed(2)} дБ  ${(mean(v) - base >= 0 ? "+" : "") + (mean(v) - base).toFixed(2)}`);
if (values.out) {
  mkdirSync(values.out, { recursive: true });
  shots.forEach((s, i) => writeFileSync(join(values.out, `cmp_${i}.png`), Buffer.from(s)));
  console.log(`картинки (вход | ${models.map((m) => m.name).join(" | ")} | эталон): ${values.out}`);
}
