/// <reference types="@webgpu/types" />
import { bias4, mat4Terms, range } from "./glsl";
import { shortF16, type MeowWeights } from "./weights";

/** Интерфейс пайплайна anime4k-webgpu — чтобы ставить MeowRestore в одну цепочку с его пресетами. */
export interface Anime4KPipeline {
  updateParam(param: string, value: unknown): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}

export interface MeowRestoreGPUDescriptor {
  device: GPUDevice;
  inputTexture: GPUTexture;
  weights: MeowWeights;
}

const WG = 8;
const FORMAT: GPUTextureFormat = "rgba16float";

/**
 * MeowRestore на WebGPU: по одному compute-проходу на слой. В отличие от GLSL (одна текстура на проход),
 * здесь проход пишет сразу все C/4 текстуры слоя — вход читается один раз, а не C/4 раз.
 * Выход — rgba16float того же размера, что вход: дальше его можно отдать CNNx2M/CNNx2VL из anime4k-webgpu.
 */
export class MeowRestoreGPU implements Anime4KPipeline {
  private readonly passes: { pipeline: GPUComputePipeline; bind: GPUBindGroup }[] = [];
  private readonly textures: GPUTexture[] = [];
  private readonly output: GPUTexture;
  private readonly w: number;
  private readonly h: number;

  constructor({ device, inputTexture, weights }: MeowRestoreGPUDescriptor) {
    const T = weights.channels / 4;
    if (T > device.limits.maxStorageTexturesPerShaderStage) {
      throw new Error(`MeowRestore: ${T} storage textures per pass > device limit`);
    }
    this.w = inputTexture.width;
    this.h = inputTexture.height;
    const make = (label: string) => {
      const t = device.createTexture({
        label,
        size: [this.w, this.h],
        format: FORMAT,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      this.textures.push(t);
      return t;
    };

    let prev: GPUTexture[] = [];
    const n = weights.convs.length;
    weights.convs.forEach((_, li) => {
      const last = li === n - 1;
      const inputs = li === 0 ? [inputTexture] : last ? [inputTexture, ...prev] : prev;
      const outs = last ? [make("meow-out")] : range(T).map((t) => make(`meow-${li}-${t}`));
      const code = layerWGSL(weights, li);
      const module = device.createShaderModule({ label: `meow-L${li}`, code });
      const pipeline = device.createComputePipeline({ label: `meow-L${li}`, layout: "auto", compute: { module, entryPoint: "main" } });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [...inputs, ...outs].map((t, i) => ({ binding: i, resource: t.createView() })),
      });
      this.passes.push({ pipeline, bind });
      prev = outs;
    });
    this.output = prev[0]!;
  }

  updateParam(): void {
    throw new Error("MeowRestore has no params");
  }

  pass(encoder: GPUCommandEncoder): void {
    const p = encoder.beginComputePass({ label: "meow-restore" });
    for (const { pipeline, bind } of this.passes) {
      p.setPipeline(pipeline);
      p.setBindGroup(0, bind);
      p.dispatchWorkgroups(Math.ceil(this.w / WG), Math.ceil(this.h / WG));
    }
    p.end();
  }

  getOutputTexture(): GPUTexture {
    return this.output;
  }

  destroy(): void {
    for (const t of this.textures) t.destroy();
  }
}

/** WGSL одного слоя. Экспортируется для тестов и для тех, кто собирает свой рендер. */
export function layerWGSL(weights: MeowWeights, li: number): string {
  const C = weights.channels;
  const T = C / 4;
  const n = weights.convs.length;
  const conv = weights.convs[li]!;
  const last = li === n - 1;
  const first = li === 0;
  // Входы: 0 — MAIN (первый и последний слой), дальше — текстуры предыдущего слоя.
  const srcCount = first ? 1 : T;
  const srcBase = first ? 0 : last ? 1 : 0;
  const outCount = last ? 1 : T;
  const decl: string[] = [];
  if (first || last) decl.push("@group(0) @binding(0) var main_tex: texture_2d<f32>;");
  if (!first) for (let g = 0; g < T; g++) decl.push(`@group(0) @binding(${srcBase + g}) var src${g}: texture_2d<f32>;`);
  const outBase = (first || last ? 1 : 0) + (first ? 0 : T);
  for (let t = 0; t < outCount; t++)
    decl.push(`@group(0) @binding(${outBase + t}) var out${t}: texture_storage_2d<rgba16float, write>;`);

  const body: string[] = [];
  for (let t = 0; t < outCount; t++) body.push(`  var r${t} = vec4f(${bias4(conv, t * 4).map(shortF16).join(", ")});`);
  // Слагаемые группируем по смещению, чтобы каждую текстуру читать один раз на смещение.
  const terms = range(outCount).map((t) => [...mat4Terms(conv, t * 4, first ? null : C)]);
  for (let kx = 0; kx < 3; kx++)
    for (let ky = 0; ky < 3; ky++) {
      const dx = kx - 1;
      const dy = ky - 1;
      body.push("  {");
      for (let g = 0; g < srcCount; g++) {
        const tex = first ? "main_tex" : `src${g}`;
        body.push(`    let v${g} = textureLoad(${tex}, clamp(p + vec2i(${dx}, ${dy}), vec2i(0), lim), 0);`);
        if (!first) body.push(`    let a${g} = max(v${g}, vec4f(0.0));`, `    let n${g} = max(-v${g}, vec4f(0.0));`);
      }
      for (let t = 0; t < outCount; t++)
        for (const term of terms[t]!) {
          if (term.dx !== dx || term.dy !== dy) continue;
          if (term.m.every((x) => x === 0)) continue;
          const g = Math.floor(term.go / 2);
          const v = first ? "v0" : term.go % 2 ? `n${g}` : `a${g}`;
          body.push(`    r${t} += mat4x4f(${term.m.map(shortF16).join(", ")}) * ${v};`);
        }
      body.push("  }");
    }
  if (last) body.push("  textureStore(out0, p, r0 + textureLoad(main_tex, p, 0));");
  else for (let t = 0; t < outCount; t++) body.push(`  textureStore(out${t}, p, r${t});`);

  const dimsOf = first || last ? "main_tex" : "src0";
  return `${decl.join("\n")}
@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2i(textureDimensions(${dimsOf}));
  let p = vec2i(gid.xy);
  if (p.x >= dims.x || p.y >= dims.y) { return; }
  let lim = dims - vec2i(1);
${body.join("\n")}
}
`;
}
