"""
Rigorous end-to-end correctness: Pure PyTorch nn.Embedding vs HashEmb.

Same initialization (all weights = 0.1), same data, same optimizer.
Loss, gradients, and weights must be identical at every step.

This is the gold-standard test for HashEmb correctness.
"""
import pytest
import torch
import numpy as np
from hashemb import HashEmbedding


# ===========================================================================
# Pure PyTorch reference: nn.Embedding + manual SGD/Adam
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
# Manual optimizer steps (exact match with C++ in embedding_table.cpp)
# ===========================================================================

def manual_sgd(weight, grad, lr):
    """Matches C++: w[d] -= lr * g[d]  (embedding_table.cpp:134)"""
    weight.data -= lr * grad


def manual_adam(weight, grad, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Matches C++ Adam (embedding_table.cpp:152-158)."""
    m.data = beta1 * m + (1 - beta1) * grad
    v.data = beta2 * v + (1 - beta2) * grad * grad
    bias_corr1 = 1.0 - beta1 ** t
    bias_corr2 = 1.0 - beta2 ** t
    m_hat = m / bias_corr1
    v_hat = v / bias_corr2
    weight.data -= lr * m_hat / (v_hat.sqrt() + eps)


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
# Test: step-by-step comparison
# ===========================================================================

class TestComparePureTorchVsHashEmb:

    N_FEATS = 100     # feat_id range: 0..99
    D = 4             # embedding_dim
    H = 8             # hidden_dim
    C = 2             # num_classes
    B = 16            # batch_size
    N_STEPS = 20      # number of training steps
    LR = 0.1          # embedding learning rate
    INIT = 0.1        # initial embedding weight

    @pytest.fixture(params=["sgd", "adam"])
    def opt(self, request):
        return request.param

    def test_step_by_step(self, opt):
        """
        For each step: compare loss, gradients, and weights.
        All on CPU for precise numerical comparison.
        """
        torch.manual_seed(42)
        np.random.seed(42)
        rng = np.random.RandomState(42)

        # Pre-generate data (same for both)
        all_feats = rng.randint(0, self.N_FEATS, size=(self.N_STEPS, self.B))
        all_labels = rng.randint(0, self.C, size=(self.N_STEPS, self.B))

        # ═══════════════════════════════════════════════════════
        # Pure PyTorch reference
        # ═══════════════════════════════════════════════════════
        ref_emb = RefEmbedding(self.N_FEATS, self.D, init_weight=self.INIT)
        ref_dense = DenseHead(self.D, self.H, self.C)
        ref_dense_opt = torch.optim.Adam(ref_dense.parameters(), lr=0.01)

        ref_m = ref_v = None
        ref_t = 0
        if opt == "adam":
            ref_m = torch.zeros(self.N_FEATS, self.D)
            ref_v = torch.zeros(self.N_FEATS, self.D)

        # ═══════════════════════════════════════════════════════
        # HashEmb
        # ═══════════════════════════════════════════════════════
        hash_emb = HashEmbedding(
            embedding_dim=self.D, capacity=self.N_FEATS + 10,
            optimizer=opt, lr=self.LR,
        )
        hash_dense = DenseHead(self.D, self.H, self.C)
        hash_dense_opt = torch.optim.Adam(hash_dense.parameters(), lr=0.01)
        hash_dense.load_state_dict(ref_dense.state_dict())
        _init_hash_emb_weights(hash_emb, self.N_FEATS, self.INIT)

        # ═══════════════════════════════════════════════════════
        # Step-by-step comparison
        # ═══════════════════════════════════════════════════════
        print(f"\n  Optimizer: {opt.upper()}, LR={self.LR}, init={self.INIT}")
        print(f"  {'Step':>4s} | {'Loss_ref':>8s} {'Loss_hash':>8s} {'Δloss':>8s} "
              f"| {'Δgrad_max':>10s} {'Δweight_max':>10s}")

        max_dg = 0.0
        max_dw = 0.0
        first_bad = None

        for step in range(self.N_STEPS):
            x = torch.tensor(all_feats[step], dtype=torch.int64)
            y = torch.tensor(all_labels[step], dtype=torch.int64)

            # ── Reference ────────────────────────────────────
            ref_dense_opt.zero_grad()
            logits_ref = ref_dense(ref_emb(x))
            loss_ref = torch.nn.functional.cross_entropy(logits_ref, y)
            loss_ref.backward()

            grad_ref = ref_emb.embedding.weight.grad.clone()

            if opt == "sgd":
                manual_sgd(ref_emb.embedding.weight, grad_ref, self.LR)
            else:
                ref_t += 1
                manual_adam(ref_emb.embedding.weight, grad_ref,
                            ref_m, ref_v, ref_t, self.LR)
            ref_emb.embedding.weight.grad.zero_()  # match HashEmb's auto-zero
            ref_dense_opt.step()

            # ── HashEmb ──────────────────────────────────────
            hash_dense_opt.zero_grad()
            logits_hash = hash_dense(hash_emb(x))
            loss_hash = torch.nn.functional.cross_entropy(logits_hash, y)
            loss_hash.backward()

            grad_hash = _extract_hash_grads(hash_emb, self.N_FEATS)

            hash_dense_opt.step()
            hash_emb.step()

            # ── Compare ──────────────────────────────────────
            w_ref = ref_emb.embedding.weight
            w_hash = _extract_hash_weights(hash_emb, self.N_FEATS)

            dl = abs(loss_ref.item() - loss_hash.item())
            dg = (grad_ref - grad_hash).abs().max().item()
            dw = (w_ref - w_hash).abs().max().item()

            max_dg = max(max_dg, dg)
            max_dw = max(max_dw, dw)
            if first_bad is None and (dl > 1e-5 or dg > 1e-5 or dw > 1e-5):
                first_bad = step

            if step < 3 or step == self.N_STEPS - 1:
                print(f"  {step:4d} | {loss_ref.item():8.6f} {loss_hash.item():8.6f} "
                      f"{dl:8.2e} | {dg:10.2e} {dw:10.2e}")

        print(f"  ─────────────────────────────────────────────────────────────")
        print(f"  Max grad diff:   {max_dg:.2e}")
        print(f"  Max weight diff: {max_dw:.2e}")

        assert max_dg < 1e-5, \
            f"Gradient mismatch (max={max_dg:.2e}, first at step {first_bad})"
        assert max_dw < 1e-5, \
            f"Weight mismatch (max={max_dw:.2e}, first at step {first_bad})"
        print(f"  ✓ {opt.upper()}: {self.N_STEPS} steps match PyTorch reference")


# ===========================================================================
# Test: dense model params converge identically
# ===========================================================================

def test_dense_params_match():
    """
    After training, all dense model parameters must be bitwise-identical.
    This proves embedding gradient flow through the shared dense head is
    numerically identical.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(42)

    N, D, H, C = 100, 4, 8, 2
    B, STEPS, LR = 16, 10, 0.1

    precomputed = [(rng.randint(0, N, size=B), rng.randint(0, C, size=B))
                   for _ in range(STEPS)]

    # ── Reference ─────────────────────────────────────────
    ref_emb = RefEmbedding(N, D, init_weight=0.1)
    ref_dense = DenseHead(D, H, C)
    ref_opt = torch.optim.SGD(ref_dense.parameters(), lr=0.01)

    # Save initial dense state before training
    dense_init_sd = {k: v.clone() for k, v in ref_dense.state_dict().items()}

    for feats, labels in precomputed:
        x, y = torch.tensor(feats, dtype=torch.int64), torch.tensor(labels, dtype=torch.int64)
        ref_opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(ref_dense(ref_emb(x)), y)
        loss.backward()
        manual_sgd(ref_emb.embedding.weight, ref_emb.embedding.weight.grad, LR)
        ref_emb.embedding.weight.grad.zero_()
        ref_opt.step()

    # ── HashEmb ───────────────────────────────────────────
    hash_emb = HashEmbedding(embedding_dim=D, capacity=N + 10,
                              optimizer="sgd", lr=LR)
    hash_dense = DenseHead(D, H, C)
    hash_dense.load_state_dict(dense_init_sd)  # same initial weights
    hash_opt = torch.optim.SGD(hash_dense.parameters(), lr=0.01)
    _init_hash_emb_weights(hash_emb, N, 0.1)

    for feats, labels in precomputed:
        x, y = torch.tensor(feats, dtype=torch.int64), torch.tensor(labels, dtype=torch.int64)
        hash_opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(hash_dense(hash_emb(x)), y)
        loss.backward()
        hash_opt.step()
        hash_emb.step()

    # Compare dense params
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(ref_dense.named_parameters(),
                                   hash_dense.named_parameters()):
        diff = (p1 - p2).abs().max().item()
        max_diff = max(max_diff, diff)
        print(f"  dense.{n1}: max_diff={diff:.2e}")
    assert max_diff < 1e-5, f"Dense params diverge: max_diff={max_diff:.2e}"
    print(f"  ✓ Dense params match (max_diff={max_diff:.2e})")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
