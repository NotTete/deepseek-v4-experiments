#!/usr/bin/env python3
"""Runner unificado para kernels, piezas de pipeline y el pipeline completo.

    python scripts/run.py <target> <chunks...> [opciones]

`<target>`: `pipeline`, `module/<nombre>`, `kernel/<nombre>` (o el nombre a secas).
`<chunks...>`: schedule, p.ej. `p12 p8 d1 d1 d4` o `10 2 3 1 3 4 2` (`N` = `dN`);
`r` reinicia buffers. Los buffers se mantienen vivos entre pasos.

Escribe tres JSON legibles: `<output>.in.json` (tokens de entrada),
`<output>.out.json` (salida por token, paso a paso) y `<output>.state.json`
(estado de todos los buffers tras el run).

Opciones:
    --range X / --range A B   rango de entrada ([-X,X] o [A,B]); default 1 -> [-1,1]
    --output F                prefijo de los .json (default: output)
    --seed N                  semilla; por defecto aleatoria (se imprime)
    --weights FILE            pesos de la capa (.safetensors); obligatorio en módulos
    --random                  pesos sintéticos en vez de --weights
    --size tiny|real, --layer-type csa|hca|window, --dtype, --device, --ratio
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kernels import (  # noqa: E402
    CheapMultiQueryAttention, Hadamard, Linear, RMSNorm, RoPE,
    SafeMultiQueryAttention, WeightedSum, precompute_rope_freqs,
)
from pipelines import (  # noqa: E402
    Attention, Indexer, MultiQueryProyection, SharedKeyValueMultiQueryAttention,
    TokenCompressor, WindowKV, get_window_topk_idxs,
)

KERNELS = ["linear", "rmsnorm", "weighted_sum", "hadamard", "rope",
           "safe_attention", "cheap_attention"]
MODULES = ["mqa_proj", "window_kv", "token_compressor", "indexer_compressor",
           "indexer", "sparse_attention", "output", "attention"]

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
RATIO = {"csa": 4, "hca": 128, "window": 0}   # ratio -> tipo de atención
PRESETS = {
    "tiny": dict(dim=128, q_lora_rank=32, n_heads=4, head_dim=32, rope_dim=8,
                 window_size=16, index_n_heads=4, index_head_dim=16, index_topk=8,
                 o_groups=2, o_lora_rank=16),
    "real": dict(dim=4096, q_lora_rank=1024, n_heads=64, head_dim=512, rope_dim=64,
                 window_size=128, index_n_heads=64, index_head_dim=128, index_topk=512,
                 o_groups=8, o_lora_rank=1024),
}
# prefijo en el fichero de pesos convertido -> módulo del runner
SUBMODULE_PREFIX = {
    "mqa_proj": "mqa_proj.", "window_kv": "window_kv.", "token_compressor": "compressor.",
    "indexer_compressor": "indexer.compressor.", "indexer": "indexer.",
    "sparse_attention": "output.attention.", "output": "output.",
}


# --------------------------------------------------------------------------- #
# Configuración / utilidades
# --------------------------------------------------------------------------- #
def init_weights(module: torch.nn.Module, seed: int, std: float = 0.02) -> None:
    """Pesos sintéticos: norms -> 1, resto N(0, std)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.is_floating_point():
                p.fill_(1.0) if (name.endswith("weight") and "norm" in name) else p.normal_(0.0, std)


def make_attention(size: str, layer_type: str, max_seq_len: int,
                   dtype: torch.dtype, device: str, seed: int) -> Attention:
    kw = dict(PRESETS[size])
    kw.update(ratio=RATIO[layer_type], max_seq_len=max_seq_len, compress_rope_theta=160000.0,
              rope_theta=10000.0, rope_factor=16.0, original_seq_len=65536,
              beta_fast=32, beta_slow=1, compute_dtype=dtype, device=device)
    torch.manual_seed(seed)
    model = Attention(**kw)
    init_weights(model, seed)
    return model


def make_freqs(rope_dim: int, max_seq_len: int, layer_type: str, device: str) -> torch.Tensor:
    """Tabla cos/sin [max_seq_len, rope/2, 2]."""
    args = (65536, 160000.0, 16.0) if layer_type in ("csa", "hca") else (0, 10000.0, 1.0)
    return precompute_rope_freqs(rope_dim, max_seq_len, *args, 32, 1, device=device)


def gen_range(n: int, dim: int, seed: int, lo: float, hi: float,
              dtype: torch.dtype, device: str) -> torch.Tensor:
    """n vectores [n, dim] uniformes en [lo, hi] (reproducible)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(n, dim, generator=g) * (hi - lo) + lo).to(dtype=dtype, device=device)


def parse_schedule(spec: str) -> list[tuple[str, int]]:
    """`p4 d1 d1 r p8 d2` -> pasos; `N` es atajo de `dN`, `r` reinicia."""
    steps: list[tuple[str, int]] = []
    for tok in spec.replace(",", " ").replace("|", " r ").split():
        tok = tok.lower()
        if tok in ("r", "reset"):
            steps.append(("reset", 0))
        elif m := re.fullmatch(r"([pd]?)(\d+)", tok):
            steps.append(("prefill" if m[1] == "p" else "decode", int(m[2])))
        else:
            raise ValueError(f"paso de schedule inválido: {tok!r}")
    if not steps:
        raise ValueError("schedule vacío")
    return steps


def run_schedule(module, tokens, steps, forward, on_reset=None):
    """Ejecuta los pasos manteniendo los buffers vivos; devuelve lista de (kind, n, start, out)."""
    results, g = [], 0
    for kind, n in steps:
        if kind == "reset":
            getattr(module, "reset", lambda: None)()
            on_reset and on_reset()
            results.append(("reset", 0, 0, None))
        else:
            results.append((kind, n, g, forward(module, tokens[g:g + n], g)))
            g += n
    return results


# --------------------------------------------------------------------------- #
# Serialización
# --------------------------------------------------------------------------- #
def _json(t: torch.Tensor):
    t = t.detach()
    return t.float().tolist() if t.is_floating_point() else t.tolist()


def rows(t: torch.Tensor) -> list:
    """[n, ...] -> una fila aplanada por token."""
    return [_json(t.detach())] if t.dim() == 0 else _json(t.reshape(t.shape[0], -1))


def buffer_state(module: torch.nn.Module) -> dict:
    return {name: {"shape": list(b.shape), "dtype": str(b.dtype).replace("torch.", ""),
                   "finite": bool(torch.isfinite(b).all()) if b.is_floating_point() else True,
                   "values": _json(b)}
            for name, b in module.named_buffers()}


def print_state(module: torch.nn.Module, title: str) -> None:
    print(f"\n[{title}]")
    for name, b in module.named_buffers():
        finite = bool(torch.isfinite(b).all()) if b.is_floating_point() else True
        print(f"  {name:28s} {tuple(b.shape)} {str(b.dtype).replace('torch.', '')} finite={finite}")


# --------------------------------------------------------------------------- #
# Plan de ejecución
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    module: torch.nn.Module
    forward: Callable                # (module, chunk, pos) -> tensor
    inputs: dict                     # tensores de entrada (para .in.json)
    tokens: torch.Tensor             # tensor primario que se trocea
    on_reset: Callable | None = None


def _finish(model, seed, forward, inputs, tokens, on_reset=None) -> Plan:
    init_weights(model, seed)
    return Plan(model, forward, inputs, tokens, on_reset)


def build_plan(target: str, name: str, cfg: dict, tokens: torch.Tensor) -> Plan:
    kw, dtype, device, seed = dict(PRESETS[cfg["size"]]), cfg["dtype"], cfg["device"], cfg["seed"]
    lo, hi, total, lt = cfg["lo"], cfg["hi"], tokens.shape[0], cfg["layer_type"]
    S = max(total, 1)
    torch.manual_seed(seed)

    def rnd(n, dim, off=0):
        return gen_range(n, dim, seed + off, lo, hi, dtype, device)

    if target == "kernel":
        return _build_kernel(name, kw, S, rnd, dtype, device, seed)

    if name == "attention":
        model = make_attention(cfg["size"], lt, max(total + 2, 4), dtype, device, seed)
        return Plan(model, lambda m, c, p: m(c, start_pos=p), {"x": tokens}, tokens)

    if name in ("mqa_proj", "window_kv"):
        freqs = make_freqs(kw["rope_dim"], total + 2, lt, device)
        cut = lambda c, p: freqs[p:p + c.shape[0]]  # noqa: E731
        if name == "mqa_proj":
            model = MultiQueryProyection(kw["dim"], kw["q_lora_rank"], kw["head_dim"],
                                         kw["n_heads"], kw["rope_dim"], compute_dtype=dtype, device=device)
            fn = lambda m, c, p: m(p, c, cut(c, p))[1]  # noqa: E731
        else:
            model = WindowKV(kw["dim"], kw["head_dim"], kw["rope_dim"], compute_dtype=dtype, device=device)
            fn = lambda m, c, p: m(p, c, cut(c, p))  # noqa: E731
        return _finish(model, seed, fn, {"x": tokens, "freqs": freqs}, tokens)

    if name in ("token_compressor", "indexer_compressor"):
        ratio = cfg["ratio"] or RATIO[lt] or 4
        if ratio <= 0:
            raise SystemExit("el compresor necesita ratio > 0 (usa --ratio o csa/hca)")
        hidden = kw["index_head_dim"] if name == "indexer_compressor" else kw["head_dim"]
        freqs = make_freqs(kw["rope_dim"], total + 2, "csa" if ratio == 4 else "hca", device)
        model = TokenCompressor(kw["dim"], hidden, kw["rope_dim"], ratio=ratio, overlap=ratio == 4,
                                 rotate=name == "indexer_compressor", compute_dtype=dtype, device=device)
        cache = torch.zeros(max(total // ratio, 1) + 1, hidden, dtype=dtype, device=device)
        return _finish(model, seed, lambda m, c, p: m(p, c, freqs, kv_cache=cache),
                       {"x": tokens, "freqs": freqs}, tokens, cache.zero_)

    if name == "indexer":
        ratio = cfg["ratio"] or 4
        freqs = make_freqs(kw["rope_dim"], total + 2, "csa", device)
        shared = rnd(total, kw["q_lora_rank"], 1)
        model = Indexer(kw["dim"], kw["q_lora_rank"], kw["index_n_heads"], kw["index_head_dim"],
                        kw["rope_dim"], kw["index_topk"], ratio, total + 2,
                        compute_dtype=dtype, device=device)
        cache = torch.zeros(max(total // ratio, 1) + 1, kw["index_head_dim"], dtype=dtype, device=device)
        return _finish(model, seed, lambda m, c, p: m(p, c, shared[p:p + c.shape[0]], freqs, cache),
                       {"x": tokens, "shared": shared, "freqs": freqs}, tokens, cache.zero_)

    if name in ("sparse_attention", "output"):
        C = max(1, S // 4)
        q = rnd(S, kw["n_heads"] * kw["head_dim"], 1).reshape(S, kw["n_heads"], kw["head_dim"])
        window, compressed = rnd(S, kw["head_dim"], 2), rnd(C, kw["head_dim"], 3)
        win_index = get_window_topk_idxs(min(kw["window_size"], S), S, 0).to(device)
        topk_index = torch.randint(0, C, (S, min(kw["index_topk"], C)), device=device)
        inputs = {"q": q, "window": window, "compressed": compressed,
                  "win_index": win_index, "topk_index": topk_index}
        if name == "sparse_attention":
            model = SafeMultiQueryAttention(kw["n_heads"], compute_dtype=dtype, device=device)
            fn = lambda m, c, p: m(c, window[:p + c.shape[0]], compressed,  # noqa: E731
                                   topk_index[p:p + c.shape[0]], win_index[p:p + c.shape[0]])
        else:
            freqs = make_freqs(kw["rope_dim"], S + 2, lt, device)
            model = SharedKeyValueMultiQueryAttention(
                kw["n_heads"], kw["head_dim"], kw["rope_dim"], kw["o_groups"],
                kw["o_lora_rank"], kw["dim"], compute_dtype=dtype, device=device)
            fn = lambda m, c, p: m(c, window[:p + c.shape[0]], compressed,  # noqa: E731
                                   topk_index[p:p + c.shape[0]], win_index[p:p + c.shape[0]],
                                   freqs[p:p + c.shape[0]])
            inputs["freqs"] = freqs
        return _finish(model, seed, fn, inputs, q)

    raise SystemExit(f"módulo desconocido: {name}")


def _build_kernel(name, kw, S, rnd, dtype, device, seed) -> Plan:
    if name in ("linear", "rmsnorm", "hadamard"):
        dim = kw["dim"]
        x = rnd(S, dim, 1)
        model = {"linear": lambda: Linear(dim, dim * 2, compute_dtype=dtype, device=device),
                 "rmsnorm": lambda: RMSNorm(dim, compute_dtype=dtype, weighted=True, device=device),
                 "hadamard": lambda: Hadamard(dim, compute_dtype=dtype, device=device)}[name]()
        return _finish(model, seed, lambda m, c, p: m(c), {"x": x}, x)

    if name == "weighted_sum":
        dim, slots = kw["dim"], 8
        vals = rnd(S, slots * dim, 1).reshape(S, slots, dim)
        scores = rnd(S, slots * dim, 2).reshape(S, slots, dim)
        model = WeightedSum(dim, slots, compute_dtype=dtype, device=device)
        return _finish(model, seed, lambda m, c, p: m(c, scores[p:p + c.shape[0]]),
                       {"values": vals, "scores": scores}, vals)

    if name == "rope":
        dim = kw["rope_dim"]
        freqs = make_freqs(dim, S + 2, "window", device)
        x = rnd(S * 3, dim, 1).reshape(S, 3, dim)
        model = RoPE(dim, compute_dtype=dtype, device=device)
        fn = lambda m, c, p: m(c, freqs[p:p + c.shape[0]].unsqueeze(-3))  # noqa: E731
        return _finish(model, seed, fn, {"x": x, "freqs": freqs}, x)

    if name == "cheap_attention":
        dim = kw["dim"]
        H = max(2, dim // 4)
        q = rnd(S * H, dim, 1).reshape(S, H, dim)
        kv, w = rnd(max(1, S // 2), dim, 2), rnd(S * H, 1, 3).reshape(S, H)
        model = CheapMultiQueryAttention(compute_dtype=dtype, device=device)
        return _finish(model, seed, lambda m, c, p: m(c, kv, w[p:p + c.shape[0]]),
                       {"q": q, "kv": kv, "weights": w}, q)

    if name == "safe_attention":
        h, D, C = 3, kw["head_dim"], 5
        q = rnd(S * h, D, 1).reshape(S, h, D)
        window, compressed = rnd(S, D, 2), rnd(C, D, 3)
        win_index = torch.randint(-1, S, (S, min(4, S)), device=device)
        win_index[:, 0] = torch.arange(S, device=device)
        topk_index = torch.randint(-1, C, (S, min(3, C)), device=device)
        model = SafeMultiQueryAttention(h, compute_dtype=dtype, device=device)
        fn = lambda m, c, p: m(c, window, compressed,  # noqa: E731
                               topk_index[p:p + c.shape[0]], win_index[p:p + c.shape[0]])
        return _finish(model, seed, fn, {"q": q, "window": window, "compressed": compressed,
                                         "win_index": win_index, "topk_index": topk_index}, q)

    raise SystemExit(f"kernel desconocido: {name}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_target(spec: str) -> tuple[str, str]:
    if spec == "pipeline":
        return "module", "attention"
    if "/" in spec:
        kind, name = spec.split("/", 1)
        if (kind == "kernel" and name in KERNELS) or (kind == "module" and name in MODULES):
            return kind, name
        raise SystemExit(f"target inválido: {spec!r}")
    for kind, names in (("kernel", KERNELS), ("module", MODULES)):
        if spec in names:
            return kind, spec
    raise SystemExit(f"target desconocido: {spec!r} (kernels: {KERNELS}; modules: {MODULES})")


def parse_range(parts: list[str]) -> tuple[float, float]:
    """`x` -> [-x, x]; `a b` -> [a, b]; ignora ruido tipo `=`/`default`."""
    vals = [float(q) for p in parts for q in p.replace("=", " ").split() if q.lower() != "default"]
    if not vals:
        return -1.0, 1.0
    return (-abs(vals[0]), abs(vals[0])) if len(vals) == 1 else (vals[0], vals[1])


def load_weights(module: torch.nn.Module, path: Path, name: str) -> int:
    """Carga el .safetensors convertido (completo, o recortado por prefijo)."""
    sd = load_file(str(path))
    if prefix := SUBMODULE_PREFIX.get(name):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not sd:
            raise SystemExit(f"{path} no contiene pesos para {name} (prefijo {prefix!r})")
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"carga de pesos incompleta: missing={missing} unexpected={unexpected}")
    return len(sd)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target")
    ap.add_argument("schedule", nargs="*", help="chunks, p.ej. 'p12 d1 d1' o '10 2 3 1'")
    ap.add_argument("--range", nargs="+", default=["1"], help="x -> [-x,x]; a b -> [a,b]")
    ap.add_argument("-o", "--output", default="output", help="prefijo de los .json")
    ap.add_argument("--size", choices=list(PRESETS), default="tiny")
    ap.add_argument("--layer-type", choices=list(RATIO), default="csa")
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    ap.add_argument("--device", default=None, help="cuda si está disponible, si no cpu")
    ap.add_argument("--seed", type=int, default=None, help="por defecto aleatoria")
    ap.add_argument("--ratio", type=int, default=None)
    ap.add_argument("--weights", type=Path, default=None, help="pesos de la capa (obligatorio en módulos)")
    ap.add_argument("--random", action="store_true", help="pesos sintéticos en vez de --weights")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target, name = parse_target(args.target)
    if target == "kernel" and args.weights:
        raise SystemExit("--weights no aplica a kernels")
    if target == "module" and not (args.weights or args.random):
        raise SystemExit("falta --weights FILE con los pesos (usa --random para sintéticos)")
    if args.weights and args.random:
        raise SystemExit("--weights y --random son incompatibles")

    lo, hi = parse_range(args.range)
    dtype = DTYPES[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    cfg = {"size": args.size, "layer_type": args.layer_type, "dtype": dtype,
           "device": device, "seed": seed, "ratio": args.ratio, "lo": lo, "hi": hi}

    steps = parse_schedule(" ".join(args.schedule) or "p16 d1")
    total = sum(n for k, n in steps if k != "reset")
    tokens = gen_range(total, PRESETS[args.size]["dim"], seed, lo, hi, dtype, device)

    print("=" * 72)
    print(f"target={target}/{name} | {args.size} | layer={args.layer_type} | "
          f"dtype={args.dtype} | device={device}")
    print(f"schedule: {' '.join(args.schedule) or '(default)'}  total={total}  "
          f"range=[{lo:g}, {hi:g}]  seed={seed}")
    print("=" * 72)

    plan = build_plan(target, name, cfg, tokens)
    if args.weights:
        print(f"pesos cargados: {args.weights} ({load_weights(plan.module, args.weights, name)} tensores)")
    results = run_schedule(plan.module, plan.tokens, steps, plan.forward, plan.on_reset)

    in_doc = {"target": args.target, "resolved": f"{target}/{name}", "size": args.size,
              "layer_type": args.layer_type, "dtype": args.dtype, "device": device,
              "seed": seed, "range": [lo, hi],
              "schedule": [f"{k}{n}" if k != "reset" else "r" for k, n in steps],
              "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", ""),
                             "tokens": rows(v)} for k, v in plan.inputs.items()}}

    out_steps = []
    for kind, n, start, out in results:
        if out is None:
            print(f"  {'reset':<8} buffers reiniciados")
            out_steps.append({"step": "reset"})
            continue
        out = out[0] if isinstance(out, tuple) else out
        print(f"  {kind:<8} n={n:<3} start={start:<4} out={tuple(out.shape)} "
              f"dtype={str(out.dtype).replace('torch.', '')} "
              f"norm={float(out.detach().float().norm()):.4f}")
        out_steps.append({"kind": kind, "n": n, "start": start,
                          "output": {"shape": list(out.shape),
                                     "dtype": str(out.dtype).replace("torch.", ""),
                                     "tokens": rows(out)}})

    print_state(plan.module, "buffers tras el run")
    head = {"target": args.target, "resolved": f"{target}/{name}"}
    base = Path(args.output)
    for suffix, doc in ((".in.json", in_doc), (".out.json", {**head, "steps": out_steps}),
                        (".state.json", {**head, "buffers": buffer_state(plan.module)})):
        path = base.with_name(base.name + suffix)
        path.write_text(json.dumps(doc, indent=2))
        print(f"escrito: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
