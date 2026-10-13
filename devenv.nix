{ pkgs, lib, config, inputs, ... }:

let
  nvidiaDriver = "/run/opengl-driver";
  withCudaToolkit = false;
in
{
  # ------------------------------- Python ---------------------------------
  languages.python = {
    enable = true;
    package = pkgs.python312;

    venv.enable = true;

    manylinux.enable = true;

    libraries = [ nvidiaDriver ];

    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  # ------------------------------- Packages -------------------------------
  packages = with pkgs; [
    git
  ] ++ lib.optionals withCudaToolkit [
    cudaPackages.cudatoolkit
    cudaPackages.cuda_nvcc
  ];

  # ------------------------------ CUDA env --------------------------------
  env = lib.optionalAttrs withCudaToolkit {
    CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
    CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
  };

  # ------------------------------- Shell ----------------------------------
  enterShell = ''
    echo "python $(python --version 2>&1 | awk '{print $2}')   uv $(uv --version | awk '{print $2}')"
    if [ -e ${nvidiaDriver}/lib/libcuda.so.1 ]; then
      echo "cuda driver  : present (${nvidiaDriver}/lib)"
    else
      echo "cuda driver  : MISSING"
    fi
  '';

  # Quick sanity check: run `gpu-check` inside the shell.
  scripts.gpu-check.exec = ''
    python - <<'PY'
    import torch
    print("torch      :", torch.__version__)
    print("cuda build :", torch.version.cuda)
    print("available  :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    i = torch.cuda.current_device()
    print("device     :", torch.cuda.get_device_name(i))
    print("capability :", torch.cuda.get_device_capability(i))
    print("cudnn      :", torch.backends.cudnn.version())
    a = torch.randn(4096, 4096, device="cuda")
    print("matmul     :", float((a @ a).sum()))
    PY
  '';

  # ------------------------------- Checks ---------------------------------
  enterTest = ''
    python -c "import torch; print(torch.__version__)"
  '';
}
