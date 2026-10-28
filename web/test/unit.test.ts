import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { f16ToF32, f32ToF16, layerWGSL, macsPerPixel, parseHooks, parseWeights, pickSize, shortF16, stepDown, toMpvHook } from "../src/index";

const fixture = (s: string) => JSON.parse(readFileSync(new URL(`./fixtures/${s}.json`, import.meta.url), "utf8"));
const numbers = (src: string) => (src.match(/-?\d+\.\d*(?:e[+-]?\d+)?|-?\d+e[+-]?\d+/g) ?? []).map(Number);

describe("fp16", () => {
  it("round-trips representative values", () => {
    for (const v of [0, 1, -2.5, 0.1, 65504, 6e-8, -1.234e-5, 3.14159]) {
      expect(f32ToF16(f16ToF32(f32ToF16(v)))).toBe(f32ToF16(v));
    }
  });
  it("shortest text maps back to the same half", () => {
    for (let h = 0; h < 0x7c00; h += 37) {
      const v = f16ToF32(h);
      expect(f32ToF16(Number(shortF16(v)))).toBe(h);
    }
  });
});

describe("toMpvHook", () => {
  for (const size of ["s", "m", "l"]) {
    it(`matches the Python generator numerically (${size})`, () => {
      const fx = fixture(size);
      const ts = toMpvHook(parseWeights(fx.weights));
      const py: string = fx.glsl;
      const tsPasses = parseHooks(ts);
      const pyPasses = parseHooks(py);
      expect(tsPasses.map((p) => [p.binds, p.save, p.from])).toEqual(pyPasses.map((p) => [p.binds, p.save, p.from]));
      const a = tsPasses.flatMap((p) => numbers(p.code));
      const b = pyPasses.flatMap((p) => numbers(p.code));
      expect(a.length).toBe(b.length);
      for (let i = 0; i < a.length; i++) expect(f32ToF16(a[i]!)).toBe(f32ToF16(b[i]!));
    });
  }

  it("produces Anime4K-style passes the animeow executor understands", () => {
    const passes = parseHooks(toMpvHook(parseWeights(fixture("m").weights)));
    expect(passes[0]!.binds).toEqual(["MAIN"]);
    expect(passes.at(-1)!.save).toBe("MAIN");
    expect(passes.at(-1)!.binds[0]).toBe("MAIN");
    expect(passes.at(-1)!.code).toContain("return result + MAIN_tex(MAIN_pos);");
    expect(passes.every((p) => p.scale === 1)).toBe(true);
  });
});

describe("layerWGSL", () => {
  it("binds main + previous layer textures + outputs in order", () => {
    const w = parseWeights(fixture("l").weights);
    const last = layerWGSL(w, w.convs.length - 1);
    expect(last).toMatch(/@binding\(0\) var main_tex/);
    expect(last).toMatch(/@binding\(3\) var src2/);
    expect(last).toMatch(/@binding\(4\) var out0/);
    const mid = layerWGSL(w, 1);
    expect(mid).toMatch(/@binding\(5\) var out2/);
  });
});

describe("tiers", () => {
  it("picks sizes by device", () => {
    expect(pickSize({ webgpu: true, mobile: false, frameHeight: 720 })).toBe("l");
    expect(pickSize({ webgpu: true, mobile: false, frameHeight: 1080 })).toBe("m");
    expect(pickSize({ webgpu: false, mobile: true, frameHeight: 720 })).toBe("s");
    expect(pickSize({ webgpu: true, mobile: true, memoryGb: 8, frameHeight: 720 })).toBe("m");
  });
  it("steps down to null", () => {
    expect([stepDown("l"), stepDown("m"), stepDown("s")]).toEqual(["m", "s", null]);
  });
  it("counts MACs per pixel like the parameter count minus biases", () => {
    expect(macsPerPixel(8, 6)).toBe(5299 - 8 * 5 - 3);
  });
});
