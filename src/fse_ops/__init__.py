"""CPU float32 LeakyReLU with a fixed negative slope of 0.1."""

# Load PyTorch's shared libraries before the extension.
import torch as _torch  # noqa: F401

from ._C import leaky_relu

__all__ = ["leaky_relu"]
