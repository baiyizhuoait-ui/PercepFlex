"""Hardware-friendly profiling: params, model size, FLOPs, latency (P50/P95), peak GPU memory."""
import statistics
import time

import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    return count_params(model) * 4 / 1e6  # fp32 bytes


def flops(model, input_size=(1, 3, 640, 640), device="cuda"):
    try:
        from thop import profile
    except ImportError:
        return None
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(x,), verbose=False)
    return 2 * macs


def latency_ms(model, input_size=(1, 3, 640, 640), device="cuda",
               warmup=10, reps=100, half=False):
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)
    if half:
        model = model.half()
        x = x.half()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    times = sorted(times)
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "fps": 1000.0 / statistics.mean(times),
    }


def peak_gpu_memory_mb(model, input_size=(1, 3, 640, 640), device="cuda"):
    torch.cuda.reset_peak_memory_stats()
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)
    with torch.no_grad():
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def profile_model(model, input_size=(1, 3, 640, 640), device="cuda", reps=100):
    return {
        "parameters": count_params(model),
        "model_size_mb": round(model_size_mb(model), 3),
        "flops": flops(model, input_size, device),
        **latency_ms(model, input_size, device, reps=reps),
        "peak_gpu_memory_mib": round(peak_gpu_memory_mb(model, input_size, device), 1),
    }
