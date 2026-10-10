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
