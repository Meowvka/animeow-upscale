/// <reference types="@webgpu/types" />
import { MeowRestoreGPU, type Anime4KPipeline } from "./webgpu";
import type { MeowWeights } from "./weights";

/** Минимум из anime4k-webgpu, который нужен пресету (передаётся модулем: `await import("anime4k-webgpu")`). */
export interface Anime4KLib {
  ClampHighlights: new (d: { device: GPUDevice; inputTexture: GPUTexture }) => Anime4KPipeline;
  CNNx2M: new (d: { device: GPUDevice; inputTexture: GPUTexture }) => Anime4KPipeline;
  CNNx2VL: new (d: { device: GPUDevice; inputTexture: GPUTexture }) => Anime4KPipeline;
  CNNx2UL: new (d: { device: GPUDevice; inputTexture: GPUTexture }) => Anime4KPipeline;
  Downscale: new (d: { device: GPUDevice; inputTexture: GPUTexture; targetDimensions: Dim }) => Anime4KPipeline;
}

interface Dim {
  width: number;
  height: number;
}

export interface MeowPresetDescriptor {
  device: GPUDevice;
  inputTexture: GPUTexture;
  nativeDimensions: Dim;
  targetDimensions: Dim;
  weights: MeowWeights;
  /** Первое увеличение ×2: VL — как в ModeA, M — легче, UL — тяжелее. */
  upscale?: "M" | "VL" | "UL";
}

/**
 * Как Anime4K ModeA, но восстановление — наша MeowRestore вместо CNNVL:
 * ClampHighlights → MeowRestore → CNNx2{VL} → (Downscale) → CNNx2M.
 * Интерфейс тот же, что у lib.ModeA — в anime4k-enhance.ts это замена одной строки.
 */
export class MeowPreset implements Anime4KPipeline {
  readonly pipelines: Anime4KPipeline[] = [];
  readonly restore: MeowRestoreGPU;
  private readonly output: GPUTexture;

  constructor(lib: Anime4KLib, d: MeowPresetDescriptor) {
    const { device, nativeDimensions: nat, targetDimensions: tgt } = d;
    const push = (p: Anime4KPipeline) => {
      this.pipelines.push(p);
      return p.getOutputTexture();
    };
    let tex = push(new lib.ClampHighlights({ device, inputTexture: d.inputTexture }));
    this.restore = new MeowRestoreGPU({ device, inputTexture: tex, weights: d.weights });
    tex = push(this.restore);
    let w = nat.width;
    let h = nat.height;
    const Up = lib[`CNNx2${d.upscale ?? "VL"}`];
    if (tgt.width > 1.2 * w && tgt.height > 1.2 * h) {
      tex = push(new Up({ device, inputTexture: tex }));
      w *= 2;
      h *= 2;
    }
    if (tgt.width > 1.2 * nat.width && tgt.height > 1.2 * nat.height && tgt.width < 2 * nat.width && tgt.height < 2 * nat.height) {
      tex = push(new lib.Downscale({ device, inputTexture: tex, targetDimensions: tgt }));
      w = tgt.width;
      h = tgt.height;
    }
    if (tgt.width > 2.4 * nat.width && tgt.height > 2.4 * nat.height && tgt.width < 4 * nat.width && tgt.height < 4 * nat.height) {
      const half = { width: Math.ceil(tgt.width / 2), height: Math.ceil(tgt.height / 2) };
      tex = push(new lib.Downscale({ device, inputTexture: tex, targetDimensions: half }));
      w = half.width;
      h = half.height;
    }
    if (tgt.width > 1.2 * w && tgt.height > 1.2 * h) tex = push(new lib.CNNx2M({ device, inputTexture: tex }));
    this.output = tex;
  }

  updateParam(): void {
    throw new Error("Preset has no param");
  }

  pass(encoder: GPUCommandEncoder): void {
    for (const p of this.pipelines) p.pass(encoder);
  }

  getOutputTexture(): GPUTexture {
    return this.output;
  }
}
