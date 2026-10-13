#!/usr/bin/env python3
"""Download the attention weights of the Compressed Sparse Attention (CSA) layers of
``deepseek-ai/DeepSeek-V4-Flash``.

DeepSeek-V4-Flash stores its weights in 46 safetensors shards.  Each transformer
layer lives in a single shard, and a shard is ~3.6 GB because of the MoE
experts, while the *attention* tensors of a layer are only ~137 MB.  Downloading
whole shards would pull ~75 GB; this script pulls only the attention part of the
CSA layers (~3 GB total).

Examples
--------
    python scripts/download_csa_layers.py                 # all CSA layers
    python scripts/download_csa_layers.py --layers 2,4    # subset
    python scripts/download_csa_layers.py -o /tmp/out     # custom output dir
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "main"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
INDEX_URL = BASE + "model.safetensors.index.json"

# Tensors selected per layer: the attention module plus its input norm.
INCLUDE = re.compile(r"^layers\.\d+\.attn(\.|_norm\.weight$)")
# Merge byte ranges closer than 8 MiB to avoid many tiny HTTP requests.
MAX_GAP = 8 << 20


def get_range(session: requests.Session, url: str, start: int, end: int, retries: int = 5) -> bytes:
    """Fetch ``bytes=start-end`` (inclusive).  Never downloads the whole file."""
    want = end - start + 1
    for attempt in range(retries):
        try:
            with session.get(url, headers={"Range": f"bytes={start}-{end}"},
                             stream=True, timeout=(20, 180)) as r:
                if r.status_code == 200:
                    raise RuntimeError("server ignored Range header")
                r.raise_for_status()
                data = b"".join(r.iter_content(1 << 20))
            if len(data) != want:
                raise RuntimeError(f"short read: got {len(data)}, want {want}")
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def read_header(session: requests.Session, url: str) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) of a remote safetensors file."""
    n = struct.unpack("<Q", get_range(session, url, 0, 7))[0]
    if not 0 < n < 500_000_000:
        raise ValueError(f"implausible safetensors header length {n}")
    header = json.loads(get_range(session, url, 8, 8 + n - 1))
    header.pop("__metadata__", None)
    return header, 8 + n


def layer_shards(index: dict) -> dict[int, set[str]]:
    """Map every layer index to the shard file(s) holding its tensors."""
    out: dict[int, set[str]] = {}
    for key, shard in index["weight_map"].items():
        if m := re.match(r"layers\.(\d+)\.", key):
            out.setdefault(int(m.group(1)), set()).add(shard)
    return out


def csa_layers(index: dict) -> list[int]:
    """CSA layers are exactly the layers carrying an attention indexer."""
    return sorted({
        int(m.group(1))
        for key in index["weight_map"]
        if (m := re.match(r"layers\.(\d+)\.attn\.indexer\.", key))
    })


def merge_ranges(spans: list[tuple[int, int]], gap: int = MAX_GAP) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    """Write (key, dtype, shape, raw_bytes) tuples to a safetensors file."""
    offset = 0
    header: dict[str, dict] = {}
    for key, dtype, shape, data in tensors:
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)  # 8-byte aligned header

    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for *_, data in tensors:
            fh.write(data)
    tmp.replace(path)


def download_layer(session: requests.Session, shards: set[str], layer: int):
    """Download the selected tensors of one layer.  Returns (tensors, shard_info)."""
    tensors: list[tuple[str, str, list[int], bytes]] = []
    info: list[dict] = []
    prefix = f"layers.{layer}."

    for shard in sorted(shards):
        url = BASE + shard
        header, data_start = read_header(session, url)
        keys = [k for k in header if k.startswith(prefix) and INCLUDE.match(k)]
        if not keys:
            continue

        merged = merge_ranges([tuple(header[k]["data_offsets"]) for k in keys])
        info.append({"shard": shard, "num_tensors": len(keys),
                     "fetch_bytes": sum(e - s for s, e in merged),
                     "num_requests": len(merged)})

        blobs = [(s, e, get_range(session, url, data_start + s, data_start + e - 1))
                 for s, e in merged]
        for k in keys:
            s, e = header[k]["data_offsets"]
            for ms, me, blob in blobs:
                if ms <= s and e <= me:
                    tensors.append((k, header[k]["dtype"], header[k]["shape"],
                                    blob[s - ms: s - ms + (e - s)]))
                    break
            else:
                raise RuntimeError(f"no blob covering {k}")
    return tensors, info


def parse_layers(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return sorted(out)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download CSA-layer attention weights.")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("data/deepseek-v4-flash-csa"))
    ap.add_argument("--layers", help="explicit layer list, e.g. '2,4,6' or '2-10' (default: all CSA)")
    args = ap.parse_args(argv)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    if token := os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        session.headers["Authorization"] = f"Bearer {token}"

    print(f"repo      : {REPO}@{REVISION}")
    print(f"output dir: {out_dir.resolve()}")

    index = session.get(INDEX_URL, timeout=(20, 120)).json()
    shards = layer_shards(index)
    csa = csa_layers(index)
    targets = parse_layers(args.layers) if args.layers else csa
    if unknown := [i for i in targets if i not in shards]:
        ap.error(f"layers not present in the checkpoint: {unknown}")

    print(f"CSA layers : {csa}  ({len(csa)})")
    print(f"downloading: {targets}")

    manifest = {"repo": REPO, "revision": REVISION,
                "created": datetime.now(timezone.utc).isoformat(),
                "csa_layers": csa, "layers": {}}

    total = 0
    for i in targets:
        t0 = time.time()
        tensors, info = download_layer(session, shards[i], i)
        if not tensors:
            print(f"layer {i:>2}: no matching tensors, skipping")
            continue

        nbytes = sum(len(d) for *_, d in tensors)
        total += nbytes
        print(f"layer {i:>2}: {len(tensors):>3} tensors, {human(nbytes)} "
              f"({len(info)} shard(s), {sum(s['num_requests'] for s in info)} requests) "
              f"[{time.time() - t0:.1f}s]")

        write_safetensors(out_dir / f"layer_{i:04d}.safetensors", tensors)
        manifest["layers"][str(i)] = {"num_tensors": len(tensors),
                                      "bytes": nbytes, "shards": info}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ntotal: {human(total)} written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
