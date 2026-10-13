#!/usr/bin/env python3
"""Convierte los pesos descargados de DeepSeek-V4-Flash (formato original, mixto
fp8/bf16/f32) a un único dtype, renombrando las claves para que carguen
directamente en `Attention` (`src/pipelines.py`).

Uso
---
    python scripts/convert_attention_weights.py \
        --input-dir  data/deepseek-v4-flash-csa \
        --output-dir data/deepseek-v4-flash-csa-converted \
        --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# src/ en el path (para reutilizar la config real de pipeline_utils)
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline_utils import attention_kwargs  # noqa: E402  (scripts/)

FP8_BLOCK = 128
LAYER_TYPE = "csa"  # CSA usa TokenCompressor con overlap (ratio 4)

# clave HF (sin el prefijo `layers.<n>.`)  ->  clave del módulo Attention
KEY_MAP: dict[str, str] = {
    "attn.wq_a.weight": "mqa_proj.linear_1.weight",
    "attn.q_norm.weight": "mqa_proj.norm_1.weight",
    "attn.wq_b.weight": "mqa_proj.linear_2.weight",
    "attn.wkv.weight": "window_kv.linear.weight",
    "attn.kv_norm.weight": "window_kv.norm.weight",
    "attn.compressor.wkv.weight": "compressor.linear_value.weight",
    "attn.compressor.wgate.weight": "compressor.linear_score.weight",
    "attn.compressor.ape": "compressor.weighted_sum.ape",
    "attn.compressor.norm.weight": "compressor.norm.weight",
    "attn.indexer.wq_b.weight": "indexer.wq_b.weight",
    "attn.indexer.weights_proj.weight": "indexer.weights_proj.weight",
    "attn.indexer.compressor.wkv.weight": "indexer.compressor.linear_value.weight",
    "attn.indexer.compressor.wgate.weight": "indexer.compressor.linear_score.weight",
    "attn.indexer.compressor.ape": "indexer.compressor.weighted_sum.ape",
    "attn.indexer.compressor.norm.weight": "indexer.compressor.norm.weight",
    "attn.attn_sink": "output.attention.attention_sink",
    "attn.wo_a.weight": "output.wo_a.weight",
    "attn.wo_b.weight": "output.wo_b.weight",
}

# pesos cuyo layout de checkpoint [ratio, coff*head_dim] se reordena a
# [slots, hidden_dim] (layout nativo de WeightedSum) al convertir.
APE_KEYS = {"attn.compressor.ape", "attn.indexer.compressor.ape"}

# pesos FP8 -> clave de la escala con la que dequantizar
FP8_SCALE: dict[str, str] = {
    "attn.wq_a.weight": "attn.wq_a.scale",
    "attn.wq_b.weight": "attn.wq_b.scale",
    "attn.wkv.weight": "attn.wkv.scale",
    "attn.wo_a.weight": "attn.wo_a.scale",
    "attn.wo_b.weight": "attn.wo_b.scale",
    "attn.indexer.wq_b.weight": "attn.indexer.wq_b.scale",
}

DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32, "fp16": torch.float16}
_KEY_RE = re.compile(r"^layers\.(\d+)\.(.*)$")


# --------------------------------------------------------------------------- #
# Dequantización / reordenado
# --------------------------------------------------------------------------- #
def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """W_fp8 [out,in] + scale [ceil(out/b), ceil(in/b)] -> f32 [out,in]."""
    out_f, in_f = weight.shape
    sf = scale.to(torch.float32)
    sf = sf.repeat_interleave(FP8_BLOCK, dim=0)[:out_f]
    sf = sf.repeat_interleave(FP8_BLOCK, dim=1)[:, :in_f]
    return weight.to(torch.float32) * sf


def ape_to_slots(ape: torch.Tensor) -> torch.Tensor:
    """[ratio, coff*head_dim] -> [slots, hidden_dim].

    Los slots de solape (0..ratio-1) toman la primera mitad del checkpoint
    (ape[:, :d]) y los actuales (ratio..) la segunda (ape[:, d:])."""
    d = ape.shape[1] // 2
    return torch.cat([ape[:, :d], ape[:, d:]], dim=0)


# --------------------------------------------------------------------------- #
# Conversión / verificación de una capa
# --------------------------------------------------------------------------- #
def convert_layer(sd: dict[str, torch.Tensor], dtype: torch.dtype
                  ) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Renombra + dequantiza + reordena + castea el state_dict de una capa."""
    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, value in sd.items():
        m = _KEY_RE.match(key)
        if not m:
            unmapped.append(key)
            continue
        layer_idx, rest = m.group(1), m.group(2)
        if rest.endswith(".scale"):           # la escala se consume con su peso
            continue
        target = KEY_MAP.get(rest)
        if target is None:
            unmapped.append(key)
            continue
        if rest in FP8_SCALE and f"layers.{layer_idx}.{FP8_SCALE[rest]}" in sd:
            value = dequant_fp8(value, sd[f"layers.{layer_idx}.{FP8_SCALE[rest]}"])
        if rest in APE_KEYS:
            value = ape_to_slots(value)
        out[target] = value.to(dtype).contiguous()

    return out, unmapped


def build_attention(dtype: torch.dtype, max_seq_len: int = 256, device: str | None = None):
    """Construye `Attention` con la config real y parámetros en `dtype`."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from pipelines import Attention
    model = Attention(**attention_kwargs("real", LAYER_TYPE, max_seq_len, dtype, device))
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.to(dtype)
    return model


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=Path("data/deepseek-v4-flash-csa"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/deepseek-v4-flash-csa-converted"))
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    args = ap.parse_args(argv)

    dtype = DTYPES[args.dtype]
    files = sorted(args.input_dir.glob("layer_*.safetensors"))
    if not files:
        raise SystemExit(f"no hay layer_*.safetensors en {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_attention(dtype)
    print(f"device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"convirtiendo {len(files)} capa(s) -> {args.dtype} (verificación obligatoria)")

    manifest: dict = {
        "source_dir": str(args.input_dir),
        "dtype": args.dtype,
        "block_size": FP8_BLOCK,
        "layer_type": LAYER_TYPE,
        "key_map": KEY_MAP,
        "layers": {},
    }

    for f in files:
        module_sd, unmapped = convert_layer(load_file(str(f)), dtype)
        out_path = args.output_dir / f.name
        save_file(module_sd, str(out_path))

        # verificación obligatoria: la capa debe cargar en Attention (strict)
        try:
            model.load_state_dict(module_sd)
        except RuntimeError as exc:
            raise SystemExit(f"{f.name}: verificación falló -> {exc}")
        print(f"  {f.name}: {len(module_sd)} tensores, load OK")

        manifest["layers"][f.stem] = {"output": out_path.name,
                                      "num_tensors": len(module_sd),
                                      "unmapped": unmapped, "load_ok": True}

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {(args.output_dir / 'manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
