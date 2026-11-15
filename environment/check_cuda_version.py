import torch

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA available:", torch.cuda.is_available())
print("Compute Capability:", torch.cuda.get_device_capability(0))
print("PyTorch CUDA:", torch.version.cuda)
print("Properties:", torch.cuda.get_device_properties(0))

