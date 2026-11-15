import torch

A = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
B = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    C = torch.matmul(A, B)
    torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=20
))
