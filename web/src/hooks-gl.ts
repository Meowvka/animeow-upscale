/**
 * Мини-исполнитель mpv-хуков на WebGL2 с той же семантикой, что у animeow (app/lib/anime4k-webgl.ts):
 * RGBA16F-текстуры, строка 0 — верх кадра, NAME_texOff / NAME_tex / NAME_pos / NAME_pt.
 * Нужен для проверки шейдеров и замеров скорости; на сайте используется его собственный исполнитель.
 */
export interface HookPass {
  binds: string[];
  save: string;
  from: string;
  scale: number;
  code: string;
}

export function parseHooks(src: string): HookPass[] {
  return src
    .split("//!DESC")
    .slice(1)
    .map((chunk) => {
      const binds: string[] = [];
      let save = "MAIN";
      let from = "MAIN";
      let scale = 1;
      const code: string[] = [];
      for (const line of chunk.split("\n").slice(1)) {
        const m = line.match(/^\/\/!(\w+)\s*(.*)$/);
        if (!m) {
          code.push(line);
          continue;
        }
        const key = m[1]!;
        const value = m[2]!.trim();
        if (key === "BIND") binds.push(value);
        else if (key === "SAVE") save = value;
        else if (key === "WIDTH") {
          const parts = value.split(/\s+/);
          from = parts[0]!.replace(/\.w$/, "");
          scale = parts.length === 3 && parts[2] === "*" ? Number(parts[1]) : 1;
        }
      }
      return { binds, save, from, scale, code: code.join("\n") };
    });
}

const VS = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

function fragmentOf(p: HookPass): string {
  const decl = p.binds
    .map(
      (n) => `uniform sampler2D ${n};
uniform vec2 ${n}_size;
#define ${n}_pos (gl_FragCoord.xy / out_size)
#define ${n}_pt (1.0 / ${n}_size)
#define ${n}_tex(p) texture(${n}, p)
#define ${n}_texOff(o) texture(${n}, ${n}_pos + (o) * ${n}_pt)`,
    )
    .join("\n");
  return `#version 300 es
precision highp float;
precision highp int;
uniform vec2 out_size;
out vec4 frag;
${decl}
${p.code}
void main() { frag = hook(); }`;
}

interface Tex {
  tex: WebGLTexture;
  w: number;
  h: number;
}

export interface HookRunner {
  /** Прогнать RGBA float32 (строка 0 — верх) через хуки, вернуть RGBA float32 результата. */
  run(rgba: Float32Array, w: number, h: number): { data: Float32Array; w: number; h: number };
  /** Только прогон без чтения результата (для замеров) — вход уже загружен прошлым run. */
  rerun(): void;
  finish(): void;
  dispose(): void;
}

export function createHookRunner(gl: WebGL2RenderingContext, hooks: string | string[]): HookRunner {
  if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("WebGL2: нет EXT_color_buffer_float");
  const compile = (type: number, code: string) => {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, code);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) ?? "shader");
    return sh;
  };
  const vs = compile(gl.VERTEX_SHADER, VS);
  const passes = (Array.isArray(hooks) ? hooks : [hooks]).flatMap(parseHooks).map((pass) => {
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragmentOf(pass)));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) ?? "link");
    return {
      pass,
      prog,
      outSize: gl.getUniformLocation(prog, "out_size"),
      sizes: pass.binds.map((n) => gl.getUniformLocation(prog, `${n}_size`)),
      units: pass.binds.map((n) => gl.getUniformLocation(prog, n)),
      out: null as Tex | null,
    };
  });
  const makeTex = (filter: number): WebGLTexture => {
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  };
  const source: Tex = { tex: makeTex(gl.LINEAR), w: 0, h: 0 };
  const fbo = gl.createFramebuffer();
  let result: Tex = source;

  const execute = () => {
    const named = new Map<string, Tex>([["MAIN", source]]);
    for (const p of passes) {
      const base = named.get(p.pass.from)!;
      const w = base.w * p.pass.scale;
      const h = base.h * p.pass.scale;
      if (!p.out) p.out = { tex: makeTex(p.pass.save === "MAIN" ? gl.LINEAR : gl.NEAREST), w: 0, h: 0 };
      if (p.out.w !== w || p.out.h !== h) {
        gl.bindTexture(gl.TEXTURE_2D, p.out.tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
        p.out.w = w;
        p.out.h = h;
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, p.out.tex, 0);
      gl.viewport(0, 0, w, h);
      gl.useProgram(p.prog);
      gl.uniform2f(p.outSize, w, h);
      p.pass.binds.forEach((name, i) => {
        const t = named.get(name);
        if (!t) throw new Error(`hook binds unknown texture ${name}`);
        gl.activeTexture(gl.TEXTURE0 + i);
        gl.bindTexture(gl.TEXTURE_2D, t.tex);
        gl.uniform1i(p.units[i]!, i);
        gl.uniform2f(p.sizes[i]!, t.w, t.h);
      });
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      named.set(p.pass.save, p.out);
    }
    result = named.get("MAIN")!;
  };

  return {
    run(rgba, w, h) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, source.tex);
      // RGBA16F, а не RGBA32F: 32-битные текстуры без OES_texture_float_linear не фильтруются и читаются нулями.
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.FLOAT, rgba);
      source.w = w;
      source.h = h;
      execute();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, result.tex, 0);
      const data = new Float32Array(result.w * result.h * 4);
      gl.readPixels(0, 0, result.w, result.h, gl.RGBA, gl.FLOAT, data);
      return { data, w: result.w, h: result.h };
    },
    rerun: execute,
    finish() {
      // Дождаться GPU: readPixels одного пикселя синхронизирует очередь.
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, result.tex, 0);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, new Float32Array(4));
    },
    dispose() {
      for (const p of passes) {
        gl.deleteProgram(p.prog);
        if (p.out) gl.deleteTexture(p.out.tex);
      }
      gl.deleteTexture(source.tex);
      gl.deleteFramebuffer(fbo);
    },
  };
}
