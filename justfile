# devenv recipes — run with `just <recipe>`

# Show available recipes
default:
    @just --list

# Install/refresh the venv from uv.lock (fast)
sync:
    uv sync

# Re-resolve dependencies and rewrite uv.lock
lock:
    uv lock

# Add one or more packages, e.g. `just add transformers wandb`
add *pkgs:
    uv add {{pkgs}}

# Add to the dev group: `just add-dev pytest-cov`
add-dev *pkgs:
    uv add --dev {{pkgs}}

# CUDA / PyTorch sanity check
gpu:
    gpu-check
