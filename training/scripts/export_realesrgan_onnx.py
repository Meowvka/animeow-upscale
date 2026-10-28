#!/usr/bin/env python3
"""
Конвертация Real-ESRGAN AnimeVideo-v3 (SRVGGNetCompact) в ONNX для браузера.

Без PyTorch: веса читаются прямо из .pth (zip + pickle), граф собирается через onnx.helper.
Зависимости: pip install numpy onnx onnxruntime onnxconverter-common

    python training/scripts/export_realesrgan_onnx.py                     # скачает веса и соберёт models/*.onnx
    python training/scripts/export_realesrgan_onnx.py --weights my.pth --num-conv 16 --scale 4

Выход:
    models/realesrgan/animevideov3-x4.fp32.onnx  (~2.4 МБ) — для WASM (CPU)
    models/realesrgan/animevideov3-x4.fp16.onnx  (~1.2 МБ) — для WebGPU (вход/выход остаются float32)

Почему эта модель: SRVGGNetCompact — самая лёгкая из «настоящих» SR-сетей Real-ESRGAN
(~0.6 млн параметров против 17 млн у RRDB), обучена именно на аниме, только conv+PReLU —
отлично ложится на WebGPU и WASM SIMD.
"""
from __future__ import annotations

import argparse
import io
import pickle
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"

_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
}


def load_pth(path: Path) -> "OrderedDict[str, np.ndarray]":
    """Минимальный загрузчик torch.save(zip) без torch."""
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split("/")[0]

    def rebuild_tensor(storage, offset, size, stride, *_args):
        itemsize = storage.dtype.itemsize
        arr = np.lib.stride_tricks.as_strided(
            storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
        )
        return np.ascontiguousarray(arr)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild_tensor
            if module == "torch" and name in _DTYPES:
                return _DTYPES[name]
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

        def persistent_load(self, pid):
            _, dtype, key, _location, _numel = pid
            return np.frombuffer(zf.read(f"{prefix}/data/{key}"), dtype=dtype)

    obj = Unpickler(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()
    for k in ("params_ema", "params"):
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return obj


def build_graph(sd: dict, num_conv: int, scale: int) -> onnx.ModelProto:
    nodes, inits = [], []

    def init(name: str, arr: np.ndarray) -> str:
        inits.append(numpy_helper.from_array(arr.astype(np.float32), name))
        return name

    x = "input"
    idx = 0
    # conv → prelu, затем num_conv × (conv → prelu), затем conv в 3·scale² каналов
    for i in range(num_conv + 1):
        w, b = sd[f"body.{idx}.weight"], sd[f"body.{idx}.bias"]
        y = f"conv{i}"
        nodes.append(helper.make_node("Conv", [x, init(f"w{i}", w), init(f"b{i}", b)], [y], pads=[1, 1, 1, 1]))
        slope = sd[f"body.{idx + 1}.weight"].reshape(-1, 1, 1)
        x = f"act{i}"
        nodes.append(helper.make_node("PRelu", [y, init(f"a{i}", slope)], [x]))
        idx += 2
    w, b = sd[f"body.{idx}.weight"], sd[f"body.{idx}.bias"]
    nodes.append(helper.make_node("Conv", [x, init("w_last", w), init("b_last", b)], ["feat"], pads=[1, 1, 1, 1]))
    # PixelShuffle в PyTorch == DepthToSpace(mode=CRD)
    nodes.append(helper.make_node("DepthToSpace", ["feat"], ["up"], blocksize=scale, mode="CRD"))
    # Остаточная связь: nearest-увеличенный вход
    init("scales", np.array([1, 1, scale, scale], dtype=np.float32))
    nodes.append(helper.make_node("Resize", ["input", "", "scales"], ["base"], mode="nearest"))
    nodes.append(helper.make_node("Add", ["up", "base"], ["output"]))

    for i, n in enumerate(nodes):
        n.name = f"{n.op_type.lower()}_{i}"  # уникальные имена нужны конвертеру fp16

    graph = helper.make_graph(
        nodes,
        "animevideov3",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="animeow-upscale")
    model.ir_version = 8  # совместимо с onnxruntime-web
    onnx.checker.check_model(model)
    return model


def to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    from onnxconverter_common import float16

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # «число N будет обрезано» — норма для мелких весов
        return float16.convert_float_to_float16(model, keep_io_types=True)


def verify(path: Path, scale: int) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((1, 3, 37, 53), dtype=np.float32)
    (y,) = sess.run(None, {"input": x})
    assert y.shape == (1, 3, 37 * scale, 53 * scale), y.shape
    # Ровная серая заливка должна остаться почти серой — ловит перепутанные каналы/веса.
    flat = np.full((1, 3, 32, 32), 0.5, dtype=np.float32)
    (yf,) = sess.run(None, {"input": flat})
    err = float(np.abs(yf - 0.5).mean())
    print(f"  {path.name}: out {y.shape}, flat-gray error {err:.4f}")
    assert err < 0.05, "model output looks wrong"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, help="путь к .pth (по умолчанию скачать animevideov3)")
    ap.add_argument("--num-conv", type=int, default=16)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--name", default="animevideov3-x4")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "models" / "realesrgan")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    weights = args.weights
    if weights is None:
        weights = args.out / "realesr-animevideov3.pth"
        if not weights.exists():
            print(f"downloading {WEIGHTS_URL}")
            urllib.request.urlretrieve(WEIGHTS_URL, weights)

    sd = load_pth(weights)
    model = build_graph(sd, args.num_conv, args.scale)
    fp32 = args.out / f"{args.name}.fp32.onnx"
    onnx.save(model, fp32)
    fp16 = args.out / f"{args.name}.fp16.onnx"
    onnx.save(to_fp16(model), fp16)
    for p in (fp32, fp16):
        print(f"saved {p} ({p.stat().st_size / 1e6:.2f} MB)")
        verify(p, args.scale)


if __name__ == "__main__":
    main()
