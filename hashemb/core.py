"""HashEmb — PyTorch wrapper for CPU host-memory hash embedding table."""

import torch
import numpy as np
from . import _hashemb_cpp


class HashEmbeddingFunction(torch.autograd.Function):
    """Autograd function: CPU hash table lookup + gradient accumulation.

    Forward:
        keys_cpu (int64 CPU, 1-D) → embeddings (float32 CPU, 2-D)

    Backward:
        grad_output (float32 CPU, 2-D) → scatter_add into C++ grad_buffer
        (actual optimizer step happens later in ``HashEmbedding.step()``)
    """

    @staticmethod
    def forward(ctx, keys_cpu, table, lr, flow):
        N = keys_cpu.size(0)
        keys_np = keys_cpu.numpy()
        emb_np, slot_np = table.lookup_and_gather(keys_np)

        ctx.table = table
        ctx.lr = lr
        ctx.slot_indices = torch.from_numpy(slot_np.copy())
        ctx.N = N

        emb = torch.from_numpy(emb_np)                     # (N, D) CPU
        return emb + 0.0 * flow                            # zero contribution, ensures grad graph

    @staticmethod
    def backward(ctx, grad_output):
        # Ensure contiguous layout: grad_output may be a non-contiguous view
        # (e.g., from strided / transposed operations).
        grad_np = grad_output.contiguous().numpy()
        slot_np = ctx.slot_indices.numpy()

        # Accumulate gradients into C++ grad_buffer (no update yet).
        ctx.table.scatter_add_grad(slot_np, grad_np)

        return None, None, None, grad_output.sum().unsqueeze(0)


class HashEmbedding(torch.nn.Module):
    """PyTorch module wrapping a CPU host-memory hash embedding table
    with native SGD / Adam optimisers and checkpoint serialisation.

    Usage:
        emb = HashEmbedding(embedding_dim=64, capacity=10_000_000,
                            optimizer='adam', lr=0.001)

        for batch in loader:
            out = emb(keys)                          # forward + grad accumulate
            loss = F.cross_entropy(dense(out), y)
            loss.backward()                          # grads → C++ grad_buffer
            dense_optim.step()                       # update dense params
            emb.step()                               # update hash table (SGD/Adam)

        # Save / load
        torch.save({'dense': dense.state_dict(),
                     'hash_emb': emb.state_dict()}, 'ckpt.pt')
        emb.load_state_dict(torch.load('ckpt.pt')['hash_emb'])

    Supports input ranks ≤ 3:
        (B,)       → (B, D)
        (B, S)     → (B, S, D)
        (B, F, S)  → (B, F, S, D)

    Device:
        Output is on the same device as ``keys``.  Device transfers happen
        outside the ``autograd.Function`` for CUDA / MPS compatibility.
    """

    def __init__(self, embedding_dim: int, capacity: int,
                 optimizer: str = "sgd",
                 lr: float = 0.01,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8,
                 block_size: int = 10_000_000):
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be ≥ 1")
        if capacity < 1:
            raise ValueError("capacity must be ≥ 1")
        if optimizer not in ("sgd", "adam"):
            raise ValueError(f"unknown optimizer '{optimizer}', use 'sgd' or 'adam'")

        self.embedding_dim = embedding_dim
        self.capacity = capacity
        self.optimizer = optimizer
        self.lr = lr

        # C++ hash embedding table (CPU memory)
        self._table = _hashemb_cpp.HashEmbeddingTable(
            capacity, embedding_dim,
            optimizer=optimizer, lr=lr,
            beta1=beta1, beta2=beta2, eps=eps,
            block_size=block_size,
        )

        # Dummy parameter for autograd graph (see HashEmbeddingFunction).
        self._flow = torch.nn.Parameter(torch.zeros(1))

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings for keys.

        Args:
            keys: int64 tensor, shape (...,) with rank ≤ 3.

        Returns:
            float32 tensor on the same device as keys, shape (..., embedding_dim).
        """
        orig_shape = keys.shape
        dev = keys.device
        keys_flat = keys.contiguous().view(-1).cpu()

        emb_cpu = HashEmbeddingFunction.apply(
            keys_flat, self._table, self.lr, self._flow.cpu(),
        )

        out_shape = orig_shape + (self.embedding_dim,)
        return emb_cpu.to(dev).view(out_shape)

    # ── Optimizer step ─────────────────────────────────────────────────

    def step(self):
        """Apply accumulated gradients (SGD or Adam) and clear grad buffer.

        Call this after ``loss.backward()`` and before zeroing the dense
        model's gradients.
        """
        self._table.step()

    def zero_grad(self):
        """Clear the internal gradient buffer.  (Not usually needed;
        ``step()`` automatically zeroes gradients after applying them.)
        """
        self._table.zero_grad()

    # ── Checkpoint ─────────────────────────────────────────────────────

    def save(self, path: str):
        """Save hash table to binary file (bucket-by-bucket, zero extra memory).

        Args:
            path: File path to write.
        """
        self._table.save(path)

    def load(self, path: str):
        """Load hash table from binary file written by :meth:`save`.

        Args:
            path: File path to read.
        """
        self._table.load(path)

    def state_dict(self, *args, **kwargs):
        """Return the hash table state as a dict of CPU ``torch.Tensor``.

        Compatible with ``torch.save`` / ``torch.load``.  Accepts extra
        keyword arguments so it does not crash when the module is nested
        inside another ``nn.Module``, but note that parent modules will
        **not** automatically include this state — call
        ``hash_emb.state_dict()`` explicitly when building checkpoints.
        """
        raw = self._table.state_dict()
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in raw.items()}

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Restore hash table state from a dict returned by ``state_dict()``.

        Accepts both tensor-valued and numpy-valued input.
        """
        # Convert torch tensors → numpy for the C++ layer.
        converted = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                converted[k] = v.cpu().numpy()
            else:
                converted[k] = v
        self._table.load_state_dict(converted)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def num_entries(self) -> int:
        """Number of unique feat_ids currently stored in the table."""
        return self._table.num_entries

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"capacity={self.capacity}, "
            f"optimizer={self.optimizer}, "
            f"lr={self.lr}, "
            f"num_entries={self.num_entries}"
        )
