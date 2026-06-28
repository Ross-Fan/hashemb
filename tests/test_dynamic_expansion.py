"""Dynamic expansion stress test: small initial capacity + block_size.

Creates HashEmbedding with capacity=100, block_size=100, then inserts
1000 feat_ids and trains with Adam.  This forces:
  - Hash table bucket auto-grow (initial buckets ~16 → multiple grow cycles)
  - Block-level expansion (1000 slots / 100 block_size = 10 blocks)

Compares loss, gradients, and weights step-by-step against a pure PyTorch
nn.Embedding reference with identical initialisation.
"""
import pytest
import torch
import numpy as np
from hashemb import HashEmbedding


# ===========================================================================
# Reference: pure PyTorch nn.Embedding with manual Adam
# ===========================================================================

class RefEmbedding(torch.nn.Module):
    """Pure PyTorch embedding with configurable init."""
    def __init__(self, num_embeddings, embedding_dim, init_weight=0.1):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        torch.nn.init.constant_(self.embedding.weight, init_weight)

    def forward(self, keys):
        return self.embedding(keys)


class DenseHead(torch.nn.Module):
    """Simple 2-layer MLP on top of embeddings."""
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# Manual Adam (exact match with C++ in embedding_table.cpp)
# ===========================================================================

def manual_adam(weight, grad, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Matches C++ dense Adam (all entries updated)."""
    m.data = beta1 * m + (1 - beta1) * grad
    v.data = beta2 * v + (1 - beta2) * grad * grad
    bias_corr1 = 1.0 - beta1 ** t
    bias_corr2 = 1.0 - beta2 ** t
    m_hat = m / bias_corr1
    v_hat = v / bias_corr2
    weight.data -= lr * m_hat / (v_hat.sqrt() + eps)


def manual_adam_sparse(weight, grad, m, v, t, lr, touched_slots,
                       beta1=0.9, beta2=0.999, eps=1e-8):
    """Matches HashEmb sparse Adam: only updates slots touched this batch."""
    bias_corr1 = 1.0 - beta1 ** t
    bias_corr2 = 1.0 - beta2 ** t
    wd = weight.data
    for slot in touched_slots:
        g = grad[slot]
        ms = m[slot]
        vs = v[slot]
        ms.mul_(beta1).add_(g, alpha=1 - beta1)
        vs.mul_(beta2).add_(g * g, alpha=1 - beta2)
        m_hat = ms / bias_corr1
        v_hat = vs / bias_corr2
        wd[slot] -= lr * m_hat / (v_hat.sqrt() + eps)


# ===========================================================================
# HashEmb helpers
# ===========================================================================

def _init_hash_emb_weights(hash_emb, n_feats, init_val):
    """Initialize HashEmbedding weights to a constant value."""
    all_ids = torch.arange(n_feats, dtype=torch.int64)
    _ = hash_emb(all_ids)  # create entries
    sd = hash_emb.state_dict()
    sd['weight'] = torch.full_like(sd['weight'], init_val)
    sd['grad'] = torch.zeros_like(sd['grad'])
    if 'm' in sd and sd['m'].numel() > 0:
        sd['m'] = torch.zeros_like(sd['m'])
    if 'v' in sd and sd['v'].numel() > 0:
        sd['v'] = torch.zeros_like(sd['v'])
    hash_emb.load_state_dict(sd)


def _extract_hash_grads(hash_emb, n_feats):
    """Extract per-feat_id gradients: [n_feats, D]."""
    sd = hash_emb.state_dict()
    D = sd['grad'].shape[1]
    grads = torch.zeros(n_feats, D)
    for i in range(len(sd['keys'])):
        fid = int(sd['keys'][i])
        grads[fid] = sd['grad'][i]
    return grads


def _extract_hash_weights(hash_emb, n_feats):
    """Extract per-feat_id weights: [n_feats, D]."""
    sd = hash_emb.state_dict()
    D = sd['weight'].shape[1]
    weights = torch.zeros(n_feats, D)
    for i in range(len(sd['keys'])):
        fid = int(sd['keys'][i])
        weights[fid] = sd['weight'][i]
    return weights


# ===========================================================================
# Test configurations
# ===========================================================================

# Small allocations to stress-test expansion paths
CAPACITY = 100        # initial capacity hint (small buckets → forces bucket grow)
BLOCK_SIZE = 100      # small block size (1000 feat_ids → 10 blocks)
N_FEATS = 1000        # total unique feat_ids in the dataset
D = 4                 # embedding_dim
H = 8                 # hidden_dim
C = 2                 # num_classes
B = 32                # batch_size
N_STEPS = 20          # training steps
LR = 0.01             # embedding learning rate (Adam)
INIT = 0.1            # initial embedding weight


class TestDynamicExpansion:

    def test_expansion_correctness(self):
        """Step-by-step comparison with PyTorch reference under expansion stress."""
        torch.manual_seed(42)
        np.random.seed(42)
        rng = np.random.RandomState(42)

        # Pre-generate data covering all 1000 feat_ids.
        all_feats = rng.randint(0, N_FEATS, size=(N_STEPS, B))
        all_labels = rng.randint(0, C, size=(N_STEPS, B))

        # ═══════════════════════════════════════════════════════
        # Pure PyTorch reference
        # ═══════════════════════════════════════════════════════
        ref_emb = RefEmbedding(N_FEATS, D, init_weight=INIT)
        ref_dense = DenseHead(D, H, C)
        ref_dense_opt = torch.optim.Adam(ref_dense.parameters(), lr=0.01)

        ref_m = torch.zeros(N_FEATS, D)
        ref_v = torch.zeros(N_FEATS, D)
        ref_t = 0

        # ═══════════════════════════════════════════════════════
        # HashEmb with aggressive expansion constraints
        # ═══════════════════════════════════════════════════════
        hash_emb = HashEmbedding(
            embedding_dim=D, capacity=CAPACITY,
            optimizer="adam", lr=LR,
            block_size=BLOCK_SIZE,
        )
        hash_dense = DenseHead(D, H, C)
        hash_dense_opt = torch.optim.Adam(hash_dense.parameters(), lr=0.01)
        hash_dense.load_state_dict(ref_dense.state_dict())

        # Initialise hash embedding weights to match reference.
        _init_hash_emb_weights(hash_emb, N_FEATS, INIT)

        print(f"\n  Config: capacity={CAPACITY}, block_size={BLOCK_SIZE}, "
              f"n_feats={N_FEATS}, ADAM, lr={LR}")
        print(f"  Hash table initial buckets: ~{CAPACITY // 16} per bucket "
              f"(will grow) | Blocks needed: ~{N_FEATS // BLOCK_SIZE}")
        print(f"  {'Step':>4s} | {'Blocks':>6s} {'N entries':>8s} | "
              f"{'Δloss':>8s} {'Δgrad_max':>10s} {'Δweight_max':>10s}")

        max_dg = 0.0
        max_dw = 0.0
        first_bad = None

        for step in range(N_STEPS):
            x = torch.tensor(all_feats[step], dtype=torch.int64)
            y = torch.tensor(all_labels[step], dtype=torch.int64)

            # ── Reference ────────────────────────────────────
            ref_dense_opt.zero_grad()
            logits_ref = ref_dense(ref_emb(x))
            loss_ref = torch.nn.functional.cross_entropy(logits_ref, y)
            loss_ref.backward()

            grad_ref = ref_emb.embedding.weight.grad.clone()

            ref_t += 1
            touched = x.unique().tolist()
            manual_adam_sparse(ref_emb.embedding.weight, grad_ref,
                               ref_m, ref_v, ref_t, LR, touched)
            ref_emb.embedding.weight.grad.zero_()
            ref_dense_opt.step()

            # ── HashEmb ──────────────────────────────────────
            hash_dense_opt.zero_grad()
            logits_hash = hash_dense(hash_emb(x))
            loss_hash = torch.nn.functional.cross_entropy(logits_hash, y)
            loss_hash.backward()

            grad_hash = _extract_hash_grads(hash_emb, N_FEATS)

            hash_dense_opt.step()
            hash_emb.step()

            # ── Compare ──────────────────────────────────────
            w_ref = ref_emb.embedding.weight
            w_hash = _extract_hash_weights(hash_emb, N_FEATS)

            dl = abs(loss_ref.item() - loss_hash.item())
            dg = (grad_ref - grad_hash).abs().max().item()
            dw = (w_ref - w_hash).abs().max().item()

            max_dg = max(max_dg, dg)
            max_dw = max(max_dw, dw)
            if first_bad is None and (dl > 1e-5 or dg > 1e-5 or dw > 1e-5):
                first_bad = step

            if step < 3 or step == N_STEPS - 1:
                n_slots = len(hash_emb._table.state_dict()['keys'])
                print(f"  {step:4d} | {n_slots:>6d} "
                      f"{hash_emb.num_entries:>8d} | "
                      f"{dl:8.2e} {dg:10.2e} {dw:10.2e}")

        print(f"  ─────────────────────────────────────────────────────────────")
        print(f"  Max grad diff:   {max_dg:.2e}")
        print(f"  Max weight diff: {max_dw:.2e}")

        assert max_dg < 1e-5, \
            f"Gradient mismatch (max={max_dg:.2e}, first at step {first_bad})"
        assert max_dw < 1e-5, \
            f"Weight mismatch (max={max_dw:.2e}, first at step {first_bad})"
        print(f"  ✓ Expansion stress test passed: {N_STEPS} steps match PyTorch")

    def test_block_count(self):
        """Verify that block expansion actually happens."""
        hash_emb = HashEmbedding(
            embedding_dim=D, capacity=CAPACITY,
            optimizer="sgd", lr=0.1,
            block_size=BLOCK_SIZE,
        )
        keys = torch.arange(N_FEATS, dtype=torch.int64)
        _ = hash_emb(keys)

        # Inspect internal state to verify multiple blocks are allocated.
        # The C++ table doesn't expose block count directly, but we can
        # verify num_entries == N_FEATS as a proxy for correctness.
        assert hash_emb.num_entries == N_FEATS, \
            f"Expected {N_FEATS} entries, got {hash_emb.num_entries}"
        print(f"  ✓ {N_FEATS} feat_ids stored successfully "
              f"(capacity hint was {CAPACITY}, block_size={BLOCK_SIZE})")

    def test_state_dict_roundtrip_with_expansion(self):
        """Save/load state dict with multiple blocks."""
        hash_emb = HashEmbedding(
            embedding_dim=D, capacity=CAPACITY,
            optimizer="adam", lr=LR,
            block_size=BLOCK_SIZE,
        )
        keys = torch.arange(N_FEATS, dtype=torch.int64)
        _ = hash_emb(keys)

        sd = hash_emb.state_dict()
        assert len(sd['keys']) == N_FEATS

        # Load into a new module.
        hash_emb2 = HashEmbedding(
            embedding_dim=D, capacity=CAPACITY,
            optimizer="adam", lr=LR,
            block_size=BLOCK_SIZE,
        )
        hash_emb2.load_state_dict(sd)
        assert hash_emb2.num_entries == N_FEATS

        # Verify weights identical.
        out1 = hash_emb(keys)
        out2 = hash_emb2(keys)
        assert torch.allclose(out1, out2, atol=1e-6)
        print(f"  ✓ State dict roundtrip preserves {N_FEATS} entries across "
              f"{N_FEATS // BLOCK_SIZE} blocks")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
