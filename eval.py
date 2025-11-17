import time
import torch

def benchmark_forward(model, input_sample, device="cpu", runs=20):
    model = model.to(device)
    input_sample = input_sample.to(device)
    model.eval()

    torch.set_grad_enabled(False)

    # warmup
    for _ in range(3):
        _ = model(input_sample)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = time.time()
        _ = model(input_sample)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    avg = sum(times) / len(times)
    return avg

if __name__ == "__main__":
    model1 = ...
    model2 = ...

    dummy = torch.randint(0, 30000, (1, 2_000))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t1 = benchmark_forward(model1, dummy, device)
    t2 = benchmark_forward(model2, dummy, device)

    print(f"Model 1 avg time: {t1*1000:.3f} ms")
    print(f"Model 2 avg time: {t2*1000:.3f} ms")
    print(f"Speedup: {t2/t1:.2f}x (model1 vs model2)")
