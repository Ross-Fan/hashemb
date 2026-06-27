#!/usr/bin/env python3
"""HashEmb demo: save/load checkpoint, training loop with native optimisers."""

import numpy as np
from hashemb import _hashemb_cpp, HashEmbedding
from hashemb.utils import get_device


def demo_cpp():
    print("=" * 60)
    print("C++ native API with SGD / Adam + checkpoint")
    print("=" * 60)

    # SGD
    t = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="sgd", lr=0.1)
    t.lookup_and_gather(np.array([1, 2], dtype=np.int64))
    sd_sgd = t.state_dict()
    print(f"SGD state_dict keys: {list(sd_sgd.keys())}")
    print(f"  opt_type={sd_sgd['opt_type']}, t={sd_sgd['t']}")

    # Adam
    t2 = _hashemb_cpp.HashEmbeddingTable(
        100, 4, optimizer="adam", lr=0.001)
    t2.lookup_and_gather(np.array([10], dtype=np.int64))
    _, slot = t2.lookup_and_gather(np.array([10], dtype=np.int64))
    t2.scatter_add_grad(slot, np.ones((1, 4), dtype=np.float32))
    t2.step()
    sd_adam = t2.state_dict()
    print(f"Adam after 1 step: opt_type={sd_adam['opt_type']}, t={sd_adam['t']}")
    print(f"  m[:4] = {sd_adam['m'][0][:4]}")
    print(f"  v[:4] = {sd_adam['v'][0][:4]}")

    # Roundtrip
    t3 = _hashemb_cpp.HashEmbeddingTable(
        100, 4, optimizer="adam", lr=0.001)
    t3.load_state_dict(sd_adam)
    w1 = t2.lookup(np.array([0], dtype=np.int32))
    w3 = t3.lookup(np.array([0], dtype=np.int32))
    assert np.allclose(w1, w3), "Roundtrip mismatch"
    print("✓ Save → load → weights match")


def demo_pytorch():
    """Full PyTorch training loop with emb.step() after backward()."""
    import torch

    device = get_device()
    has_accel = device != "cpu"
    print(f"\nDevice: {device}")

    # ── Create model ───────────────────────────────────────────────────
    emb = HashEmbedding(
        embedding_dim=64, capacity=100_000,
        optimizer="adam", lr=0.001, beta1=0.9, beta2=0.999,
    )

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        def forward(self, x): return self.net(x)

    dense = MLP().to(device)
    dense_opt = torch.optim.Adam(dense.parameters(), lr=0.001)

    # ── Training loop ──────────────────────────────────────────────────
    print("\nTraining (3 steps) ...")
    for step in range(3):
        keys = torch.randint(0, 10000, (64,), dtype=torch.int64, device=device)
        labels = torch.randint(0, 2, (64,), device=device)

        logits = dense(emb(keys))                  # CPU hash lookup → GPU forward
        loss = torch.nn.functional.cross_entropy(logits, labels)

        loss.backward()                            # accumulates grads in C++
        dense_opt.step()                           # dense model update (PyTorch)
        emb.step()                                 # hash table update (C++ Adam)

        print(f"  Step {step}: loss={loss.item():.4f}, entries={emb.num_entries}")

    # ── Save checkpoint ────────────────────────────────────────────────
    ckpt = {
        "dense": dense.state_dict(),
        "hash_emb": emb.state_dict(),              # keys, slots, weight, grad, m, v, t
    }
    print(f"\nCheckpoint keys: {list(ckpt['hash_emb'].keys())}")
    print(f"  num_entries = {ckpt['hash_emb']['keys'].shape[0]}")

    # ── Load & continue ────────────────────────────────────────────────
    emb2 = HashEmbedding(64, 100_000, optimizer="adam", lr=0.001)
    emb2.load_state_dict(ckpt["hash_emb"])
    dense2 = MLP().to(device)
    dense2.load_state_dict(ckpt["dense"])

    # Verify same weights = same loss.
    with torch.no_grad():
        keys2 = torch.randint(0, 10000, (64,), dtype=torch.int64, device=device)
        l1 = torch.nn.functional.cross_entropy(dense(emb(keys2)), labels)
        l2 = torch.nn.functional.cross_entropy(dense2(emb2(keys2)), labels)
        assert torch.allclose(l1, l2, atol=1e-5), "Checkpoint restore mismatch"

    print("✓ Checkpoint save → load → continue training (weights match)")
    print(f"  Total entries after save/load: {emb2.num_entries}")


if __name__ == "__main__":
    demo_cpp()
    demo_pytorch()
    print("\n✅ All demos passed.")
