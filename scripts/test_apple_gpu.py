"""Train a tiny regression model on random synthetic data using Apple MPS.

Run with: bash scripts/test-gpu.sh
Or, with the simulator venv active: python scripts/test_apple_gpu.py
"""

import argparse
import math
import os
import platform
import subprocess
import time

# Unsupported GPU operations must fail instead of silently using the CPU.
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

import torch
from torch import nn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200, help="optimizer steps (default: 200)")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, capture_output=True, check=False,
        )
        print(f"Chip: {result.stdout.strip() if result.returncode == 0 else 'unavailable (restricted process)'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        raise SystemExit("FAIL: Apple GPU is unavailable to this process. No CPU training performed.")

    device = torch.device("mps")
    torch.manual_seed(42)
    torch.mps.manual_seed(42)

    # Random inputs and random hidden coefficients with a learnable relationship.
    # Independent random labels would only test fitting noise.
    features, outputs = 32, 4
    true_weight = torch.randn(features, outputs) / math.sqrt(features)
    true_bias = torch.randn(outputs)

    def dataset(samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(samples, features)
        y = x @ true_weight + true_bias + 0.01 * torch.randn(samples, outputs)
        return x.to(device), y.to(device)

    x_train, y_train = dataset(2048)
    x_test, y_test = dataset(512)
    model = nn.Linear(features, outputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.MSELoss()
    starting_weight = model.weight.detach().clone()

    print(f"Training device: {model.weight.device}")
    print(f"Data device: {x_train.device}; CPU fallback disabled")
    print("Task: learn y = X @ W + b + small noise from 2,048 random samples")
    with torch.no_grad():
        initial_loss = loss_fn(model(x_test), y_test).item()
    print(f"Initial held-out loss: {initial_loss:.6f}")

    torch.mps.synchronize()
    start = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is None or parameter.grad.device.type != "mps":
                raise SystemExit("FAIL: gradient missing or not on Apple GPU")
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            value = loss.item()
            if not math.isfinite(value):
                raise SystemExit("FAIL: training loss became non-finite")
            print(f"Step {step:3d}/{args.steps}: training loss = {value:.6f}")
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        final_loss = loss_fn(model(x_test), y_test).item()
        weight_change = (model.weight - starting_weight).abs().max().item()

    improvement = 100 * (1 - final_loss / initial_loss)
    print(f"Final held-out loss: {final_loss:.6f} ({improvement:.4f}% reduction)")
    print(f"Largest weight change: {weight_change:.6f}")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"MPS tensor memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MiB")
    if not math.isfinite(final_loss) or final_loss >= initial_loss * 0.2 or weight_change <= 0:
        raise SystemExit("FAIL: model did not learn enough; try the default 200 steps")
    print("PASS: forward pass, gradients and optimizer updates ran on Apple GPU; loss decreased.")


if __name__ == "__main__":
    main()
