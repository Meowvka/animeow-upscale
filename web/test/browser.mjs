// Проверка в настоящем Chromium (WebGL2 + WebGPU через SwiftShader):
//   node test/browser.mjs        — нужен playwright (npm i -D playwright && npx playwright install chromium)
import { build } from "esbuild";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const root = new URL(".", import.meta.url).pathname;
await build({ entryPoints: [join(root, "page.ts")], bundle: true, format: "esm", outfile: join(root, ".out/page.js"), logLevel: "warning" });

const html = `<!doctype html><meta charset="utf-8"><script type="module" src="./.out/page.js"></script>`;
const server = createServer(async (req, res) => {
  const path = new URL(req.url, "http://x").pathname;
  if (path === "/") return res.writeHead(200, { "content-type": "text/html" }).end(html);
  const file = normalize(join(root, decodeURIComponent(path)));
  if (!file.startsWith(root)) return res.writeHead(403).end();
  try {
    const type = { ".js": "text/javascript", ".json": "application/json" }[extname(file)] ?? "application/octet-stream";
    res.writeHead(200, { "content-type": type }).end(await readFile(file));
  } catch {
    res.writeHead(404).end();
  }
}).listen(0);

const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage();
page.on("console", (m) => m.type() === "error" && console.log("[page]", m.text()));
await page.goto(`http://localhost:${server.address().port}/`);
await page.waitForFunction(() => window.__done, null, { timeout: 600_000 });
const r = await page.evaluate(() => window.__done);
await browser.close();
server.close();

console.log(JSON.stringify(r, null, 2));
if (r.error) process.exit(1);
// fp16-текстуры между слоями дают погрешность ~1e-3; больше — ошибка в генерации шейдера.
const TOL = 4e-3;
let failed = false;
for (const s of ["s", "m", "l"]) {
  for (const k of ["glslPython", "glslTs", "webgpu"]) {
    const v = r[s]?.[k];
    if (v === undefined) continue;
    const ok = v < TOL;
    failed ||= !ok;
    console.log(`${ok ? "OK  " : "FAIL"} ${s} ${k}: max error ${v.toExponential(2)}`);
  }
}
if (r.preset && !(r.preset.finite && r.preset.out[0] === 64)) {
  failed = true;
  console.log("FAIL preset chain", r.preset);
}
process.exit(failed ? 1 : 0);
