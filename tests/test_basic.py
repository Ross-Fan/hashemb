"""Tests for HashEmb C++ extension and PyTorch wrapper.

GPU/MPS tests are skipped when no accelerator is available.
"""

import sys
import numpy as np
import pytest

from hashemb import _hashemb_cpp

# ===========================================================================
# C++ Extension Tests (always run)
# ===========================================================================


class TestCppHashTable:
    def test_create(self):
        table = _hashemb_cpp.HashEmbeddingTable(1000, 64)
        assert table.embedding_dim == 64
        assert table.num_entries == 0

    def test_find_or_create_dedup(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 8)
        keys = np.array([10, 20, 30, 10], dtype=np.int64)
        slot = table.find_or_create(keys)
        assert slot.shape == (4,)
        assert slot[0] == slot[3]
        assert table.num_entries == 3

    def test_find_or_create_repeat_call(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 8)
        s1 = table.find_or_create(np.array([42], dtype=np.int64))
        s2 = table.find_or_create(np.array([42], dtype=np.int64))
        assert s1[0] == s2[0]

    def test_lookup_and_gather(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4)
        keys = np.array([1, 2, 3], dtype=np.int64)
        emb, slot = table.lookup_and_gather(keys)
        assert emb.shape == (3, 4)
        assert np.allclose(emb, 0.0)

    def test_many_keys(self):
        n = 5000
        table = _hashemb_cpp.HashEmbeddingTable(n, 16)
        keys = np.arange(n, dtype=np.int64)
        emb, slot = table.lookup_and_gather(keys)
        assert emb.shape == (n, 16)
        assert table.num_entries == n

    def test_auto_grow(self):
        """Buckets auto-grow when full — no hard capacity limit."""
        table = _hashemb_cpp.HashEmbeddingTable(5, 4)
        keys = np.arange(100, dtype=np.int64)
        emb, slot = table.lookup_and_gather(keys)
        assert emb.shape == (100, 4)
        assert table.num_entries == 100
        # Verify all lookups return valid (zero-initialized) embeddings.
        assert np.allclose(emb, 0.0)

    # ── Optimiser: SGD ─────────────────────────────────────────────────

    def test_sgd_step(self):
        """scatter_add_grad + step() = equivalent to old sgd_update."""
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="sgd", lr=0.1)
        keys = np.array([42], dtype=np.int64)
        emb, slot = table.lookup_and_gather(keys)

        grad = np.ones((1, 4), dtype=np.float32) * 0.5
        table.scatter_add_grad(slot, grad)
        table.step()

        emb2 = table.lookup(slot)
        expected = np.zeros((1, 4), dtype=np.float32) - 0.1 * 0.5
        assert np.allclose(emb2, expected, atol=1e-6)

    def test_sgd_accumulate_then_step(self):
        """Multiple backward passes → gradients accumulate → applied once."""
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="sgd", lr=0.1)
        keys = np.array([99], dtype=np.int64)
        _, slot = table.lookup_and_gather(keys)
        sid = slot[0]

        keys_dup = np.array([99, 99, 99], dtype=np.int64)
        _, slot_dup = table.lookup_and_gather(keys_dup)
        assert np.all(slot_dup == sid)

        grads = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        table.scatter_add_grad(slot_dup, grads)
        table.step()

        emb = table.lookup(np.array([sid], dtype=np.int32))
        assert np.allclose(emb[0, 0], -0.6, atol=1e-6)   # sum=6 * -0.1 = -0.6
        assert np.allclose(emb[0, 1:], 0.0, atol=1e-6)

    # ── Optimiser: Adam ────────────────────────────────────────────────

    def test_adam_creates_mv_buffers(self):
        """Adam table should have working m/v buffers."""
        table = _hashemb_cpp.HashEmbeddingTable(
            100, 4, optimizer="adam", lr=0.001, beta1=0.9, beta2=0.999)
        keys = np.array([10, 20], dtype=np.int64)
        emb, slot = table.lookup_and_gather(keys)

        grads = np.ones((2, 4), dtype=np.float32) * 1.0
        table.scatter_add_grad(slot, grads)
        table.step()

        # After Adam step, values should differ from SGD (non-uniform update
        # due to moment estimates).  Just check they're non-zero and
        # negative (gradients were positive, so update should decrease).
        emb2 = table.lookup(slot)
        assert np.all(emb2 < 0), "Adam step should decrease weights"
        # Also verify that different dims have the same value (uniform grad
        # → uniform m/v → uniform update for first step).
        row0 = emb2[0]
        assert np.allclose(row0, row0[0], atol=1e-6), "Uniform grad → uniform update"

    def test_adam_t_step_increments(self):
        """Check t increments properly (affects bias correction)."""
        table = _hashemb_cpp.HashEmbeddingTable(
            10, 2, optimizer="adam", lr=0.1)
        keys = np.array([1], dtype=np.int64)
        _, slot = table.lookup_and_gather(keys)

        grads = np.ones((1, 2), dtype=np.float32)
        for _ in range(3):
            table.scatter_add_grad(slot, grads)
            table.step()

        # Verify non-zero values after 3 steps.
        emb = table.lookup(slot)
        assert not np.allclose(emb, 0.0, atol=1e-6)

    # ── Serialisation ──────────────────────────────────────────────────

    def test_state_dict_sgd(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="sgd", lr=0.1)
        keys = np.array([1, 2, 3], dtype=np.int64)
        table.lookup_and_gather(keys)

        sd = table.state_dict()
        assert set(sd.keys()) >= {"keys", "slots", "weight", "grad", "m", "v",
                                   "t", "opt_type", "dim"}
        assert sd["opt_type"] == "sgd"
        assert sd["t"] == 0
        assert len(sd["keys"]) == 3
        assert sd["dim"] == 4

    def test_state_dict_roundtrip_sgd(self):
        """Save → load → continue training → verify consistency."""
        t1 = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="sgd", lr=0.1)
        keys = np.array([10, 20], dtype=np.int64)
        slots1 = t1.find_or_create(keys)   # → [slot_10, slot_20] (order depends on impl)

        # Train one step on the slots that were assigned.
        grads = np.ones((2, 4), dtype=np.float32) * 0.5
        t1.scatter_add_grad(slots1, grads)
        t1.step()

        sd = t1.state_dict()

        # Load into a new table with same capacity/dim.
        t2 = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="sgd", lr=0.1)
        t2.load_state_dict(sd)
        assert t2.num_entries == 2

        # Verify mappings: same keys → same slots as before (slot order is an
        # implementation detail, only consistency matters).
        s2 = t2.find_or_create(np.array([10, 20], dtype=np.int64))
        assert np.all(s2 == slots1)

        # Verify weights preserved (lookup by slot, not key).
        w2 = t2.lookup(s2)
        w1 = t1.lookup(slots1)
        assert np.allclose(w1, w2, atol=1e-6)

        # Continue training on the restored table (single slot update).
        t2_slots = t2.find_or_create(np.array([10], dtype=np.int64))
        t2.scatter_add_grad(t2_slots[0:1],
                            np.ones((1, 4), dtype=np.float32))
        t2.step()

    def test_state_dict_adam(self):
        table = _hashemb_cpp.HashEmbeddingTable(
            100, 4, optimizer="adam", lr=0.001)
        table.lookup_and_gather(np.array([1], dtype=np.int64))
        _, slot = table.lookup_and_gather(np.array([1], dtype=np.int64))
        table.scatter_add_grad(slot, np.ones((1, 4), dtype=np.float32))
        table.step()

        sd = table.state_dict()
        assert sd["opt_type"] == "adam"
        assert sd["t"] == 1
        # Adam m and v should now be non-zero.
        assert not np.allclose(sd["m"], 0.0, atol=1e-6)
        assert not np.allclose(sd["v"], 0.0, atol=1e-6)

    def test_state_dict_roundtrip_adam(self):
        t1 = _hashemb_cpp.HashEmbeddingTable(
            100, 4, optimizer="adam", lr=0.001)
        t1.lookup_and_gather(np.array([42], dtype=np.int64))
        _, slot = t1.lookup_and_gather(np.array([42], dtype=np.int64))
        t1.scatter_add_grad(slot, np.ones((1, 4), dtype=np.float32))
        t1.step()

        sd = t1.state_dict()

        t2 = _hashemb_cpp.HashEmbeddingTable(
            100, 4, optimizer="adam", lr=0.001)
        t2.load_state_dict(sd)
        assert t2.num_entries == 1

        # Continue training on t2 — should be equivalent to continuing on t1.
        _, slot2 = t2.lookup_and_gather(np.array([42], dtype=np.int64))
        t2.scatter_add_grad(slot2, np.ones((1, 4), dtype=np.float32))
        t2.step()

        # t1 continuing
        t1.scatter_add_grad(slot, np.ones((1, 4), dtype=np.float32))
        t1.step()

        w1 = t1.lookup(np.array([0], dtype=np.int32))
        w2 = t2.lookup(np.array([0], dtype=np.int32))
        assert np.allclose(w1, w2, atol=1e-5)


# ===========================================================================
# PyTorch Wrapper Tests (device-agnostic: CUDA > MPS > skip)
# ===========================================================================

torch = pytest.importorskip("torch")
from hashemb.utils import get_device
DEVICE = get_device()
HAS_ACCELERATOR = DEVICE != "cpu"


class TestHashEmbeddingCPU:
    """CPU-level tests (no accelerator needed)."""

    def test_cpp_path_via_embedding(self):
        from hashemb import HashEmbedding
        emb = HashEmbedding(embedding_dim=8, capacity=100)
        assert emb.embedding_dim == 8
        assert emb.capacity == 100
        assert emb.num_entries == 0

    def test_optimizer_params(self):
        from hashemb import HashEmbedding
        emb = HashEmbedding(4, 100, optimizer="adam", lr=0.001,
                            beta1=0.9, beta2=0.999, eps=1e-8)
        assert emb.optimizer == "adam"
        assert emb.lr == 0.001

    def test_forward_basic(self):
        from hashemb import HashEmbedding
        emb = HashEmbedding(embedding_dim=8, capacity=100)
        keys = torch.tensor([1, 2, 3, 1], dtype=torch.int64, device=DEVICE)
        out = emb(keys)
        assert out.shape == (4, 8)
        assert out.device.type == DEVICE
        assert torch.allclose(out[0], out[3])

    def test_backward_gradient_sum(self):
        """Same feat_id → gradients summed, then step() applies."""
        from hashemb import HashEmbedding
        emb = HashEmbedding(embedding_dim=4, capacity=100, optimizer="sgd", lr=0.1)

        keys = torch.tensor([99, 99, 1], dtype=torch.int64, device=DEVICE)
        out = emb(keys)
        loss = out.sum()
        loss.backward()
        emb.step()  # Apply accumulated gradients

        out2 = emb(keys)
        expected_99 = torch.full((4,), -0.2, device=DEVICE, dtype=torch.float32)
        assert torch.allclose(out2[0], expected_99, atol=1e-6)
        assert torch.allclose(out2[1], expected_99, atol=1e-6)
        expected_1 = torch.full((4,), -0.1, device=DEVICE, dtype=torch.float32)
        assert torch.allclose(out2[2], expected_1, atol=1e-6)

    def test_multi_step_accumulate(self):
        """Multiple backward passes before single step() accumulate grads."""
        from hashemb import HashEmbedding
        emb = HashEmbedding(4, 100, optimizer="sgd", lr=0.1)

        keys = torch.tensor([99], dtype=torch.int64, device=DEVICE)

        # Simulate gradient clipping / micro-batching:
        # Two backward passes, one step.
        out = emb(keys)
        out.sum().backward()

        out = emb(keys)
        out.sum().backward()

        emb.step()

        # Two backward passes → grad=2 accumulated → update = -0.1 * 2 = -0.2
        out2 = emb(keys)
        expected = torch.full((4,), -0.2, device=DEVICE, dtype=torch.float32)
        assert torch.allclose(out2[0], expected, atol=1e-6)

    @pytest.mark.skipif(not HAS_ACCELERATOR, reason="No accelerator available")
    def test_training_step(self):
        """End-to-end training step."""
        from hashemb import HashEmbedding
        emb = HashEmbedding(16, 1000, optimizer="adam", lr=0.001)
        dense = torch.nn.Linear(16, 2).to(DEVICE)
        opt = torch.optim.Adam(dense.parameters(), lr=0.001)

        keys = torch.randint(0, 100, (32,), dtype=torch.int64, device=DEVICE)
        labels = torch.randint(0, 2, (32,), device=DEVICE)

        logits = dense(emb(keys))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        emb.step()

    def test_state_dict_roundtrip(self):
        """PyTorch-level state_dict save/load."""
        from hashemb import HashEmbedding
        import copy

        emb = HashEmbedding(4, 100, optimizer="adam", lr=0.001)
        keys = torch.tensor([1, 2, 3], dtype=torch.int64, device=DEVICE)
        out = emb(keys)
        out.sum().backward()
        emb.step()

        sd = emb.state_dict()

        # New module, load state.
        emb2 = HashEmbedding(4, 100, optimizer="adam", lr=0.001)
        emb2.load_state_dict(sd)
        assert emb2.num_entries == 3

        # Verify same weights.
        out1 = emb(keys)
        out2 = emb2(keys)
        assert torch.allclose(out1, out2, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
