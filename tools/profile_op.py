"""Profile repeated CPU calls on a large contiguous float32 tensor."""

import platform

import torch

from fse_ops import leaky_relu


def run_calls(x, repeats):
    for _ in range(repeats):
        leaky_relu(x)


def main():
    torch.set_num_threads(1)
    x = torch.linspace(-5.0, 5.0, steps=10_000_000, dtype=torch.float32)
    repeats = 20

    # Warm up before the run_calls interval; cProfile also records this call.
    leaky_relu(x)
    print(f"Python {platform.python_version()}, PyTorch {torch.__version__}")
    print(f"CPU float32: {x.numel():,} elements, {repeats} calls, 1 PyTorch thread")
    run_calls(x, repeats)


if __name__ == "__main__":
    main()
