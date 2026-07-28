"""HashEmb utility functions."""

from pathlib import Path
from typing import Optional

import numpy as np


NUM_BUCKETS = 16


def export_hashemb_to_npz(
    save_path: str,
    output_path: str,
    embedding_dim: int,
    capacity: int = 10_000_000,
    block_size: int = 1_000_000,
) -> int:
    """Read a saved ``.hashemb`` file and export keys + embeddings to NPZ.

    Writes one NPZ file per hash bucket (``_bucket_00.npz`` through
    ``_bucket_15.npz``), so peak memory is bounded to a single bucket.
    Export order within each file is unspecified — consumers should
    align vectors by ``keys``.

    Args:
        save_path: Path to the ``.hashemb`` binary written by
            :meth:`hashemb.HashEmbedding.save`.
        output_path: Output NPZ base path (must end with ``.npz``).
            Actual files written are ``{stem}_bucket_00.npz`` through
            ``{stem}_bucket_15.npz``.
        embedding_dim: Embedding dimension (must match the saved table).
        capacity: Initial hash table capacity hint.  Only needs to be
            larger than the number of entries in the save file.
        block_size: Slots per memory block.

    Returns:
        Total number of exported entries across all buckets.
    """
    # Lazy import to avoid circular dependencies and allow this module
    # to be imported without torch for metadata-only uses.
    from hashemb import HashEmbedding  # noqa: PLC0415

    if not str(output_path).endswith(".npz"):
        raise ValueError("output_path must end with '.npz'")

    emb = HashEmbedding(
        embedding_dim=embedding_dim,
        capacity=capacity,
        block_size=block_size,
    )
    emb.load(save_path)

    base = Path(output_path)
    total = 0
    for bucket_id in range(NUM_BUCKETS):
        raw = emb._table.export_bucket_arrays(bucket_id)
        keys = raw["keys"].astype(np.int64, copy=False)
        embeddings = raw["embeddings"].astype(np.float32, copy=False)
        bucket_path = base.with_name(
            f"{base.stem}_bucket_{bucket_id:02d}{base.suffix}"
        )
        np.savez(
            bucket_path,
            keys=keys,
            embeddings=embeddings,
            dim=np.array(embedding_dim, dtype=np.int64),
            num_entries=np.array(len(keys), dtype=np.int64),
            format_version=np.array(2, dtype=np.int32),
            bucket_id=np.array(bucket_id, dtype=np.int32),
            num_buckets=np.array(NUM_BUCKETS, dtype=np.int32),
        )
        total += int(len(keys))
    return total


def get_device() -> str:
    """Select the best available device for PyTorch tensors.

    Priority: cuda > mps > cpu

    NOTE: This is a development convenience. Production should always use CUDA.
    """
    import torch

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
    import torch

    t = torch.from_numpy(arr)
    return t.to(device, non_blocking=True)
