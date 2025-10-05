import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for GPU Acceleration")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")
