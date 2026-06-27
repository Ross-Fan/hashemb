#!/usr/bin/env python3
"""
Adam optimizer correctness verification.

Compares C++ Adam step-by-step against a hand-written PyTorch reference
implementation using the exact same gradients, so any deviation is a bug.

Dataset: 100 binary-classification samples, single feat_id per sample,
          embedding_dim=4, learning_rate=0.1 (amplified for visibility).
"""

import sys
import numpy as np
import torch


def reference_adam(weight, grads, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    """Manual Adam step-by-step.  Returns list of (weight, m, v) after each step."""
    d = weight.shape[0]
    m = torch.zeros(d)
    v = torch.zeros(d)
    t = 0
    history = []

    for g in grads:
        t += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        weight = weight - lr * m_hat / (v_hat.sqrt() + eps)
        history.append((weight.clone(), m.clone(), v.clone(), t))

    return history


def test_adam_against_reference():
    from hashemb import _hashemb_cpp

    D = 4
    LR = 0.1
    B1, B2, EPS = 0.9, 0.999, 1e-8

    # ── Build synthetic gradients ──────────────────────────────────────
    # 100 steps, each with a random gradient vector.
    # Using a fixed seed ensures reproducibility.
    rng = np.random.RandomState(42)
    grads_np = rng.randn(100, D).astype(np.float32)

    # ── Reference: manual Adam in PyTorch ──────────────────────────────
    w_ref = torch.zeros(D)
    history_ref = reference_adam(w_ref, [torch.from_numpy(g) for g in grads_np],
                                 lr=LR, beta1=B1, beta2=B2, eps=EPS)

    # ── C++ HashEmbeddingTable with Adam ───────────────────────────────
    table = _hashemb_cpp.HashEmbeddingTable(
        capacity=100, embedding_dim=D,
        optimizer="adam", lr=LR, beta1=B1, beta2=B2, eps=EPS,
    )
    # Insert a single key so we have slot 0.
    table.lookup_and_gather(np.array([42], dtype=np.int64))
    slot = np.array([0], dtype=np.int32)

    # ── Step-by-step comparison ────────────────────────────────────────
    max_diff_weight = 0.0
    max_diff_m = 0.0
    max_diff_v = 0.0
    first_mismatch = None

    for step in range(100):
        # Feed gradient
        table.scatter_add_grad(slot, grads_np[step:step + 1])
        table.step()

        # Fetch C++ state
        sd = table.state_dict()
        w_cpp = torch.from_numpy(sd["weight"][0])    # slot 0's embedding
        m_cpp = torch.from_numpy(sd["m"][0])
        v_cpp = torch.from_numpy(sd["v"][0])
        t_cpp = sd["t"]

        # Reference values after this step
        w_ref_i, m_ref_i, v_ref_i, t_ref_i = history_ref[step]

        dw = (w_cpp - w_ref_i).abs().max().item()
        dm = (m_cpp - m_ref_i).abs().max().item()
        dv = (v_cpp - v_ref_i).abs().max().item()

        max_diff_weight = max(max_diff_weight, dw)
        max_diff_m = max(max_diff_m, dm)
        max_diff_v = max(max_diff_v, dv)

        if first_mismatch is None and (dw > 1e-5 or dm > 1e-5 or dv > 1e-5):
            first_mismatch = step

        if step < 3 or step == 99:
            print(f"  Step {step:3d}: t={t_cpp:3d} | "
                  f"w[0]={w_cpp[0]:+.6f} (ref={w_ref_i[0]:+.6f}) | "
                  f"m[0]={m_cpp[0]:+.6f} (ref={m_ref_i[0]:+.6f}) | "
                  f"v[0]={v_cpp[0]:+.6f} (ref={v_ref_i[0]:+.6f})")

    print(f"\n  Max diff: weight={max_diff_weight:.2e}, m={max_diff_m:.2e}, v={max_diff_v:.2e}")

    # ── Assertions ────────────────────────────────────────────────────
    assert t_cpp == 100, f"t counter mismatch: {t_cpp} vs 100"
    assert max_diff_weight < 1e-5, (
        f"Weight divergence at step {first_mismatch}: max_diff={max_diff_weight:.2e}")
    assert max_diff_m < 1e-5, (
        f"m divergence at step {first_mismatch}: max_diff={max_diff_m:.2e}")
    assert max_diff_v < 1e-5, (
        f"v divergence at step {first_mismatch}: max_diff={max_diff_v:.2e}")
    print("  ✓ All 100 steps match reference Adam within 1e-5")


def test_adam_training_convergence():
    """
    End-to-end: a simple binary classifier *must* converge when trained
    with HashEmbedding + Adam.

    A small embedding_dim=4 should be enough to overfit 100 samples.
    """
    from hashemb import HashEmbedding
    import torch.nn.functional as F

    DEVICE = "cpu"  # keep it simple for verification

    # ── Synthetic dataset ──────────────────────────────────────────────
    N = 100
    rng = np.random.RandomState(1)
    feat_ids = rng.randint(0, 10000, size=N).astype(np.int64)
    labels = (feat_ids % 2).astype(np.int64)  # deterministic label

    # ── Model ──────────────────────────────────────────────────────────
    emb = HashEmbedding(
        embedding_dim=4, capacity=10_000,
        optimizer="adam", lr=0.01,
    )
    classifier = torch.nn.Linear(4, 2)
    opt_cls = torch.optim.Adam(classifier.parameters(), lr=0.01)

    losses = []
    for epoch in range(20):
        epoch_loss = 0.0
        for i in range(N):
            key = torch.tensor([feat_ids[i]], dtype=torch.int64, device=DEVICE)
            label = torch.tensor([labels[i]], dtype=torch.int64, device=DEVICE)

            out = emb(key)                     # (1, 4)  embedding
            logits = classifier(out)           # (1, 2)
            loss = F.cross_entropy(logits, label)

            loss.backward()                    # accumulate grad in C++
            opt_cls.step()
            emb.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / N)
        if epoch < 3 or epoch == 19:
            print(f"  Epoch {epoch:2d}: avg_loss = {losses[-1]:.4f}")

    # Must converge: loss after 20 epochs should be much lower than start.
    assert losses[-1] < losses[0] * 0.5, (
        f"Loss did not converge: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  ✓ Loss converged: {losses[0]:.4f} → {losses[-1]:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Step-by-step Adam vs reference")
    print("=" * 60)
    test_adam_against_reference()

    print("\n" + "=" * 60)
    print("Test 2: Training convergence (20 epochs)")
    print("=" * 60)
    test_adam_training_convergence()

    print("\n✅ All Adam verification tests passed.")
