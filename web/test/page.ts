/// <reference types="@webgpu/types" />
// Страница проверки в Chromium: корректность всех исполнителей против эталона numpy + замеры скорости.
import * as lib from "anime4k-webgpu";
import { createHookRunner, MeowPreset, MeowRestoreGPU, parseWeights, toMpvHook, f16ToF32, f32ToF16, type MeowWeightsJson } from "../src/index";

interface Fixture {
  weights: MeowWeightsJson;
  width: number;
  height: number;
  input: number[];
  expected: number[];
  glsl: string;
}

const toRGBA = (rgb: number[], n: number) => {
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = rgb[i * 3]!;
    out[i * 4 + 1] = rgb[i * 3 + 1]!;
    out[i * 4 + 2] = rgb[i * 3 + 2]!;
    out[i * 4 + 3] = 1;
  }
  return out;
};

const maxErr = (rgba: Float32Array, expected: number[], n: number) => {
  let m = 0;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) m = Math.max(m, Math.abs(rgba[i * 4 + c]! - expected[i * 3 + c]!));
  return m;
};

async function gpuDevice(): Promise<GPUDevice | null> {
  const adapter = await navigator.gpu?.requestAdapter();
  return adapter ? adapter.requestDevice() : null;
}

function uploadF16(device: GPUDevice, rgba: Float32Array, w: number, h: number): GPUTexture {
  const tex = device.createTexture({
    size: [w, h],
    format: "rgba16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
  });
  const half = new Uint16Array(rgba.length);
  for (let i = 0; i < rgba.length; i++) half[i] = f32ToF16(rgba[i]!);
  device.queue.writeTexture({ texture: tex }, half, { bytesPerRow: w * 8 }, [w, h]);
  return tex;
}

async function readF16(device: GPUDevice, tex: GPUTexture): Promise<Float32Array> {
  const { width: w, height: h } = tex;
  const bpr = Math.ceil((w * 8) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const raw = new Uint16Array(buf.getMappedRange().slice(0));
  buf.destroy();
  const out = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w * 4; x++) out[y * w * 4 + x] = f16ToF32(raw[(y * bpr) / 2 + x]!);
  return out;
}

async function timeGpu(device: GPUDevice, p: { pass(e: GPUCommandEncoder): void }, frames: number): Promise<number> {
  const run = async () => {
    const enc = device.createCommandEncoder();
    p.pass(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  };
  await run();
  const t0 = performance.now();
  for (let i = 0; i < frames; i++) await run();
  return (performance.now() - t0) / frames;
}

async function main() {
  const results: Record<string, unknown> = {};
  const device = await gpuDevice();
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2")!;
  const sizes = ["s", "m", "l"];

  for (const size of sizes) {
    const fx: Fixture = await (await fetch(`./fixtures/${size}.json`)).json();
    const n = fx.width * fx.height;
    const input = toRGBA(fx.input, n);
    const weights = parseWeights(fx.weights);
    const r: Record<string, unknown> = {};

    const pyGL = createHookRunner(gl, fx.glsl);
    r.glslPython = maxErr(pyGL.run(input, fx.width, fx.height).data, fx.expected, n);
    pyGL.dispose();

    const tsHook = toMpvHook(weights);
    const tsGL = createHookRunner(gl, tsHook);
    r.glslTs = maxErr(tsGL.run(input, fx.width, fx.height).data, fx.expected, n);
    tsGL.dispose();
    r.hookBytes = tsHook.length;
    r.weightsJsonBytes = JSON.stringify(fx.weights).length;

    if (device) {
      const tex = uploadF16(device, input, fx.width, fx.height);
      const restore = new MeowRestoreGPU({ device, inputTexture: tex, weights });
      const enc = device.createCommandEncoder();
      restore.pass(enc);
      device.queue.submit([enc.finish()]);
      r.webgpu = maxErr(await readF16(device, restore.getOutputTexture()), fx.expected, n);
      restore.destroy();
      tex.destroy();
    }
    results[size] = r;
  }

  // Цепочка с anime4k-webgpu: MeowRestore → CNNx2VL → … до ×2.
  if (device) {
    const fx: Fixture = await (await fetch(`./fixtures/m.json`)).json();
    const tex = uploadF16(device, toRGBA(fx.input, fx.width * fx.height), fx.width, fx.height);
    const preset = new MeowPreset(lib as never, {
      device,
      inputTexture: tex,
      nativeDimensions: { width: fx.width, height: fx.height },
      targetDimensions: { width: fx.width * 2, height: fx.height * 2 },
      weights: parseWeights(fx.weights),
    });
    const enc = device.createCommandEncoder();
    preset.pass(enc);
    device.queue.submit([enc.finish()]);
    const out = await readF16(device, preset.getOutputTexture());
    results.preset = {
      out: [preset.getOutputTexture().width, preset.getOutputTexture().height],
      finite: out.every(Number.isFinite),
      stages: preset.pipelines.length,
    };
  }

  // Замеры на кадре 1280×720 (в headless это программный рендер — важны соотношения, не абсолютные мс).
  const bench: Record<string, number> = {};
  const W = 1280, H = 720;
  const frame = new Float32Array(W * H * 4).map((_, i) => ((i * 2654435761) % 1000) / 1000);
  for (const size of sizes) {
    const fx: Fixture = await (await fetch(`./fixtures/${size}.json`)).json();
    const weights = parseWeights(fx.weights);
    const runner = createHookRunner(gl, toMpvHook(weights));
    runner.run(frame, W, H);
    const t0 = performance.now();
    for (let i = 0; i < 3; i++) runner.rerun();
    runner.finish();
    bench[`webgl2 meow-${size}`] = (performance.now() - t0) / 3;
    runner.dispose();
    if (device) {
      const tex = uploadF16(device, frame, W, H);
      const restore = new MeowRestoreGPU({ device, inputTexture: tex, weights });
      bench[`webgpu meow-${size}`] = await timeGpu(device, restore, 3);
      restore.destroy();
      tex.destroy();
    }
  }
  if (device) {
    for (const name of ["CNNM", "CNNVL", "CNNx2M"] as const) {
      const tex = uploadF16(device, frame, W, H);
      const p = new (lib as unknown as Record<string, new (d: object) => { pass(e: GPUCommandEncoder): void }>)[name]!({ device, inputTexture: tex });
      bench[`webgpu anime4k ${name}`] = await timeGpu(device, p, 3);
      tex.destroy();
    }
  }
  results.bench = bench;
  return results;
}

main().then(
  (r) => ((window as unknown as { __done: unknown }).__done = r),
  (e) => ((window as unknown as { __done: unknown }).__done = { error: String(e?.stack ?? e) }),
);
