"""python -m unittest discover training/tests — без torch (эталон на numpy, генерация GLSL)."""
import re
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meow import weights  # noqa: E402
from meow.glsl import to_glsl  # noqa: E402
from meow.model_sizes import SIZES  # noqa: E402


def random_meta(name: str, seed: int = 0) -> dict:
    s = SIZES[name]
    rng = np.random.default_rng(seed)
    cin = [3] + [2 * s.channels] * (s.layers - 1)
    cout = [s.channels] * (s.layers - 1) + [3]
    layers = [(rng.standard_normal((o, i, 3, 3)).astype(np.float32) * 0.1, rng.standard_normal(o).astype(np.float32) * 0.01) for i, o in zip(cin, cout)]
    return weights.from_arrays(f"meow-restore-{name}", s.channels, layers)


class ExportTest(unittest.TestCase):
    def test_roundtrip_fp16(self):
        meta = random_meta("m")
        for (w, b), l in zip(weights.unpack(meta), meta["layers_data"]):
            self.assertEqual(w.shape, (l["out"], l["in"], 3, 3))
            np.testing.assert_array_equal(w, w.astype(np.float16).astype(np.float32))

    def test_zero_last_layer_is_identity(self):
        meta = random_meta("s")
        layers = weights.unpack(meta)
        layers[-1] = (np.zeros_like(layers[-1][0]), np.zeros_like(layers[-1][1]))
        meta = weights.from_arrays("id", SIZES["s"].channels, layers)
        x = np.random.default_rng(1).random((3, 9, 11), dtype=np.float32)
        np.testing.assert_allclose(weights.reference_forward(meta, x), x, atol=1e-7)

    def test_glsl_structure(self):
        for name, s in SIZES.items():
            g = to_glsl(random_meta(name))
            passes = g.count("//!DESC")
            self.assertEqual(passes, (s.layers - 1) * (s.channels // 4) + 1, name)
            self.assertEqual(g.count("//!SAVE MAIN"), 1)
            self.assertIn("return result + MAIN_tex(MAIN_pos);", g)
            # каждое число в mat4 — валидный GLSL-литерал
            for lit in re.findall(r"mat4\(([^)]*)\)", g)[:50]:
                for v in lit.split(", "):
                    self.assertRegex(v, r"^-?\d+(\.\d*)?(e[+-]?\d+)?$")

    def test_glsl_reproduces_reference(self):
        """Мини-интерпретатор хука на numpy: гарантирует порядок mat4 (по столбцам) и смещений."""
        meta = random_meta("m", seed=3)
        x = np.random.default_rng(2).random((3, 7, 8), dtype=np.float32)
        ref = weights.reference_forward(meta, x)
        tex = {"MAIN": np.concatenate([x, np.ones((1, 7, 8), np.float32)])}
        for block in to_glsl(meta).split("//!DESC")[1:]:
            save = re.search(r"//!SAVE (\S+)", block).group(1)
            defs = dict(re.findall(r"#define (go_\d+)\(x_off, y_off\) \((.*)\)$", block, re.M))
            acc = np.zeros((4, 7, 8), np.float32)
            for nums, go, dx, dy in re.findall(r"mat4\(([^)]*)\) \* (go_\d+)\(([-\d.]+), ([-\d.]+)\)", block):
                m = np.array([float(v) for v in nums.split(", ")], np.float32).reshape(4, 4).T  # столбцы → матрица
                src = defs[go]
                name = re.search(r"(\w+)_texOff", src).group(1)
                t = np.pad(tex[name], ((0, 0), (1, 1), (1, 1)), mode="edge")
                ix, iy = int(float(dx)), int(float(dy))
                v = t[:, 1 + iy : 1 + iy + 7, 1 + ix : 1 + ix + 8]
                if "max(-(" in src:
                    v = np.maximum(-v, 0)
                elif "max((" in src:
                    v = np.maximum(v, 0)
                acc += np.einsum("ij,jhw->ihw", m, v)
            bias = re.search(r"result \+= vec4\(([^)]*)\);", block).group(1)
            acc += np.array([float(v) for v in bias.split(", ")], np.float32)[:, None, None]
            if "MAIN_tex(MAIN_pos)" in block:
                acc += tex["MAIN"]
            tex[save] = acc
        np.testing.assert_allclose(tex["MAIN"][:3], ref, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
