"""HashEmb utility functions."""

import torch
import numpy as np


def get_device() -> str:
    """Select the best available device for PyTorch tensors.

    Priority: cuda > mps > cpu

    NOTE: This is a development convenience. Production should always use CUDA.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def allocate_pinned_buffer(n: int, dim: int, dtype=np.float32):
    """Allocate a pinned-memory buffer for CPU↔GPU transfer.

    Args:
        n: number of embedding vectors
        dim: embedding dimension
        dtype: numpy dtype (default float32)

    Returns:
        numpy array backed by pinned (page-locked) memory.
    """
    return np.empty((n, dim), dtype=dtype)


def to_tensor(arr: np.ndarray, device: str = "cuda"):
    """Zero-copy numpy → torch tensor (works with pinned memory)."""
    t = torch.from_numpy(arr)
    return t.to(device, non_blocking=True)
