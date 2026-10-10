# DeepSeek V4 experiments — devenv

Fast, reproducible Python env powered by **devenv 2.x + uv**, with **CUDA-enabled
PyTorch** (cu128 wheels) for the RTX 3060 Mobile (Ampere, sm_86).

## What's in here

| File | Purpose |
|------|---------|
| `devenv.nix` | Python 3.12 + uv, `uv sync` on shell entry, CUDA libs on `LD_LIBRARY_PATH` |
| `pyproject.toml` | Dependencies + the PyTorch cu128 wheel index |
| `uv.lock` | Fully pinned / hashed resolution (161 pkgs, torch 2.11.0+cu128) |
| `devenv.lock` | Pins the nixpkgs input |
| `justfile` | `just sync`, `just lock`, `just add …`, `just gpu` |

## 1. NVIDIA driver (lives in ~/dotfiles)

A devenv cannot ship the CUDA **driver** — `libcuda.so.1` must come from the
running kernel. This machine (`malenia`) now gets it from the dotfiles flake:

- `~/dotfiles/modules/features/gpu.nix` — fixed so `"nvidia"` is added to
  `services.xserver.videoDrivers` (nixpkgs only enables the driver when that
  list contains `"nvidia"`) and a per-host `gpu.nvidia.enable` flag was added.
- `~/dotfiles/modules/hosts/malenia/configuration.nix` — sets
  `gpu.nvidia.enable = true` + hybrid offload (iGPU keeps driving the display,
  `nvidia-offload` for GPU apps).
- `gpu.compute.enable` stays **false**: no `nixpkgs.config.cudaSupport`, so
  nothing in nixpkgs is rebuilt.

Deploy it:

```sh
sudo nixos-rebuild switch --flake ~/dotfiles#malenia
```

Afterwards `nvidia-smi` works and `/run/opengl-driver/lib/libcuda.so.1` exists,
so the RTX 3060 is usable through the prebuilt PyTorch wheels. If the display
regresses, set `gpu.hybrid.enable = false` and rebuild — CUDA keeps working.

## 2. Enter the devenv

```sh
cd ~/Research/deepseek-v4-experiments
devenv shell          # first run downloads torch (a few GB), then it's cached
```

`uv sync` is **checksum-gated** on `pyproject.toml`, so re-entering the shell is
instant after the first time.

Or with direnv (optional): install direnv, hook it into your shell, then
`direnv allow` once — the env activates automatically on `cd`.

## 3. Verify CUDA

```sh
gpu-check     # or: just gpu
```

Expected: `cuda build : 12.8`, `available : True`, and your `NVIDIA GeForce
RTX 3060 ...`.

## Day-to-day

```sh
just add wandb          # uv add + updates uv.lock
just add-dev pytest-cov
just lock               # re-resolve
just sync               # apply lockfile
uv run python train.py  # run inside the env
```

## Tuning CUDA

- The wheel index is set in `pyproject.toml` (`cu128`). To change CUDA version,
  edit the index URL + `[tool.uv.sources]`, then `just lock && just sync`.
  cu128 needs driver ≥ 525; all current NixOS `nvidiaPackages.stable` satisfy it.
- `torch`/`torchvision` ship their own CUDA runtime/cuDNN/NCCL, so the multi-GB
  CUDA toolkit is **not** needed for training. If you compile custom kernels
  (flash-attn, apex…), set `withCudaToolkit = true;` in `devenv.nix` to add
  `cudaPackages.cudatoolkit` / `cuda_nvcc` and `CUDA_HOME`.
- `manylinux.enable` + `libraries = [ "/run/opengl-driver/lib" ]` make third-party
  wheels find the glibc-compatible libs and the driver on NixOS.

## Python version

Pinned to nixpkgs `python312` (torch wheel support). The venv lives in
`.devenv/state/venv`; `uv` never downloads its own interpreter
(`UV_PYTHON_DOWNLOADS=never`).

## FAQ

**Does this recompile my programs with CUDA?**
No. `nixpkgs.config.cudaSupport` is intentionally **not** set. CUDA is provided
by prebuilt PyTorch pip wheels (via uv) plus the prebuilt NVIDIA driver binary.
No nixpkgs package is rebuilt. If you ever see thousands of builds queued, you
(or another module) enabled `cudaSupport` — remove it.

**What actually uses the GPU at runtime?**
torch `dlopen`s `libcuda.so.1` from the host driver; the wheel bundles the CUDA
runtime, cuDNN, and NCCL. That's why the host only needs the driver.
