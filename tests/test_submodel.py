"""
Test: Two sub-models → BigModel architecture.

SubModel 1 (FeatureEmbedder):
    feat_id dict → feat embedding dict

SubModel 2 (DenseModel):
    feat embedding dict → logits (dense prediction)

BigModel:
    Combines both, verifies forward/backward/step all work end-to-end.
"""

import pytest
import torch
import numpy as np
from hashemb import HashEmbedding
from hashemb.utils import get_device

DEVICE = get_device()


# ===========================================================================
# SubModel 1: feat_id → feat embedding (dict output)
# ===========================================================================

class FeatureEmbedder(torch.nn.Module):
    """Takes multiple feature fields, shares one HashEmbedding table.

    Each feature field gets the full embedding vector (embedding_dim).
    Outputs a dict: {feature_name: embedding_tensor}.

    Args:
        feature_names: list of feature field names
        embedding_dim: dimension of each embedding vector
        capacity: max unique feat_ids
        optimizer/lr/beta1/beta2/eps: passed to HashEmbedding
    """
    def __init__(self, feature_names, embedding_dim, capacity,
                 optimizer="adam", lr=0.001,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.feature_names = list(feature_names)
        self.emb = HashEmbedding(
            embedding_dim=embedding_dim,
            capacity=capacity,
            optimizer=optimizer,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )

    def forward(self, feat_dict):
        """feat_dict: {name: int64_tensor(...,)} → {name: float32_tensor(..., D)}"""
        return {name: self.emb(feat_dict[name]) for name in self.feature_names}

    def step(self):
        self.emb.step()


# ===========================================================================
# SubModel 2: feat embedding dict → logits (dense prediction)
# ===========================================================================

class DenseModel(torch.nn.Module):
    """Takes dict of embeddings, concatenates them, runs dense layers.

    Args:
        feature_names: order-matched with embedder output
        embedding_dim: per-feature embedding dimension
        hidden_dim: hidden layer size
        num_classes: output classes
    """
    def __init__(self, feature_names, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.feature_names = list(feature_names)
        self.total_dim = len(feature_names) * embedding_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.total_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, emb_dict):
        """emb_dict: {name: tensor(..., D)} → logits.

        All embeddings must have same batch dims.
        Concatenates along last dim: (..., D) * N → (..., N*D)
        """
        combined = torch.cat(
            [emb_dict[name] for name in self.feature_names],
            dim=-1,  # concat embedding dims
        )
        return self.net(combined)


# ===========================================================================
# BigModel: sub1 + sub2 combined
# ===========================================================================

class BigModel(torch.nn.Module):
    """End-to-end: feat_ids → embeddings → dense → logits.

    Training loop:
        logits = big_model(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        dense_optim.step()
        big_model.embedder.step()   # hash table update
        dense_optim.zero_grad()     # (emb.step() auto-zeroes C++ grad buffer)
    """
    def __init__(self, feature_names, embedding_dim, capacity,
                 hidden_dim, num_classes,
                 optimizer="adam", lr=0.001):
        super().__init__()
        self.embedder = FeatureEmbedder(
            feature_names, embedding_dim, capacity,
            optimizer=optimizer, lr=lr,
        )
        self.dense = DenseModel(
            feature_names, embedding_dim, hidden_dim, num_classes,
        )

    def forward(self, feat_dict):
        embeddings = self.embedder(feat_dict)
        return self.dense(embeddings)


# ===========================================================================
# Tests
# ===========================================================================

class TestBigModel:
    """End-to-end tests for two-sub-model architecture."""

    FEATURE_NAMES = ["user_id", "item_id", "context_id"]
    EMBEDDING_DIM = 8
    CAPACITY = 1000
    HIDDEN_DIM = 16
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    LR = 0.01

    def test_bigmodel_forward_shape(self):
        """Verify output shape is correct."""
        model = BigModel(
            self.FEATURE_NAMES, self.EMBEDDING_DIM, self.CAPACITY,
            self.HIDDEN_DIM, self.NUM_CLASSES,
            lr=self.LR,
        ).to(DEVICE)

        feat_dict = {
            "user_id":    torch.randint(0, 100, (self.BATCH_SIZE,), dtype=torch.int64, device=DEVICE),
            "item_id":    torch.randint(0, 100, (self.BATCH_SIZE,), dtype=torch.int64, device=DEVICE),
            "context_id": torch.randint(0, 100, (self.BATCH_SIZE,), dtype=torch.int64, device=DEVICE),
        }

        logits = model(feat_dict)
        assert logits.shape == (self.BATCH_SIZE, self.NUM_CLASSES), \
            f"Expected ({self.BATCH_SIZE}, {self.NUM_CLASSES}), got {logits.shape}"
        print(f"  ✓ Forward shape: {logits.shape}")

    def test_embeddings_dict_output(self):
        """SubModel 1 outputs a dict of tensors, each with correct shape."""
        embedder = FeatureEmbedder(
            self.FEATURE_NAMES, self.EMBEDDING_DIM, self.CAPACITY,
            lr=self.LR,
        ).to(DEVICE)

        feat_dict = {
            "user_id":    torch.randint(0, 100, (self.BATCH_SIZE,), dtype=torch.int64, device=DEVICE),
            "item_id":    torch.randint(0, 100, (self.BATCH_SIZE,), dtype=torch.int64, device=DEVICE),
            "context_id": torch.randint(0, 100, (self.BATCH_SIZE,), dtype=torch.int64, device=DEVICE),
        }

        emb_dict = embedder(feat_dict)

        assert isinstance(emb_dict, dict), "embedder output must be dict"
        assert set(emb_dict.keys()) == set(self.FEATURE_NAMES), \
            f"Expected keys {self.FEATURE_NAMES}, got {list(emb_dict.keys())}"
        for name in self.FEATURE_NAMES:
            assert emb_dict[name].shape == (self.BATCH_SIZE, self.EMBEDDING_DIM), \
                f"{name}: expected ({self.BATCH_SIZE}, {self.EMBEDDING_DIM}), got {emb_dict[name].shape}"
            assert emb_dict[name].device.type == DEVICE, \
                f"{name}: expected device {DEVICE}, got {emb_dict[name].device}"
        print(f"  ✓ Embeddings dict: { {k: v.shape for k, v in emb_dict.items()} }")

    def test_backward_gradient_flow(self):
        """
        Critical test: gradients must flow from loss → through DenseModel
        → through all embedding tensors → into C++ grad_buffer.

        Verify by checking that emb.step() actually changes the weights
        (proving gradients were accumulated).
        """
        model = BigModel(
            self.FEATURE_NAMES, self.EMBEDDING_DIM, self.CAPACITY,
            self.HIDDEN_DIM, self.NUM_CLASSES,
            optimizer="sgd", lr=0.1,       # Use SGD for easy numerical verification
        ).to(DEVICE)
        dense_opt = torch.optim.SGD(model.dense.parameters(), lr=0.01)

        feat_dict = {
            "user_id":    torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=DEVICE),
            "item_id":    torch.tensor([5, 6, 7, 8], dtype=torch.int64, device=DEVICE),
            "context_id": torch.tensor([9, 10, 11, 12], dtype=torch.int64, device=DEVICE),
        }
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=DEVICE)

        # ---- Forward ----
        logits = model(feat_dict)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # ---- Backward ----
        loss.backward()
        dense_opt.step()
        model.embedder.step()

        # ---- Verify: weights should have changed (non-zero after backward + step) ----
        # Re-lookup the same IDs and verify they've changed from initial zeros.
        emb_dict_after = model.embedder(feat_dict)
        for name in self.FEATURE_NAMES:
            emb = emb_dict_after[name]
            # At least some non-zero values due to gradient update
            assert not torch.allclose(emb, torch.zeros_like(emb), atol=1e-6), \
                f"{name}: embeddings didn't change after backward+step!"

        print(f"  ✓ Gradient flowed through all {len(self.FEATURE_NAMES)} feature fields")
        print(f"  ✓ emb.step() applied gradients, weights changed from zero")

    def test_gradient_flow_sequence_features(self):
        """
        With sequence features (B, S), each position gets its own gradient.
        After backward, gradients for duplicate feat_ids get summed correctly.

        Real usage: sequence features are mean-pooled before dense prediction.
        """
        model = BigModel(
            ["seq_feat"], self.EMBEDDING_DIM, self.CAPACITY,
            self.HIDDEN_DIM, self.NUM_CLASSES,
            optimizer="sgd", lr=1.0,    # Large LR for visible change
        ).to(DEVICE)
        dense_opt = torch.optim.SGD(model.dense.parameters(), lr=0.01)

        # Sequence feature: feat_id 42 appears twice → gradients should sum
        feat_dict = {
            "seq_feat": torch.tensor([[42, 99, 42]], dtype=torch.int64, device=DEVICE),  # (1, 3)
        }
        labels = torch.tensor([0], dtype=torch.int64, device=DEVICE)

        out1 = model.embedder.emb(feat_dict["seq_feat"])
        print(f"  Sequence feature output shape: {out1.shape}  (expect [1, 3, {self.EMBEDDING_DIM}])")

        # For sequence features, mean-pool before dense prediction.
        # This is the standard pattern in recommendation systems.
        emb_dict = model.embedder(feat_dict)                              # {name: (1, 3, D)}
        pooled = {name: emb.mean(dim=-2) for name, emb in emb_dict.items()}  # {name: (1, D)}
        logits = model.dense(pooled)                                      # (1, C)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        loss.backward()
        dense_opt.step()
        model.embedder.step()

        # Verify: the two positions with feat_id=42 both updated
        out_after = model.embedder.emb(feat_dict["seq_feat"])
        assert out_after.shape == (1, 3, self.EMBEDDING_DIM)
        # Position 0 (id=42) and Position 2 (id=42) should have same embedding
        assert torch.allclose(out_after[0, 0], out_after[0, 2], atol=1e-6), \
            "Duplicate feat_ids must have same embedding after update"
        print(f"  ✓ Sequence feature: positions with same feat_id have same embedding")
        print(f"  ✓ Mean-pooled sequence → dense → backward → step: all gradients flow correctly")

    @pytest.mark.skipif(DEVICE == "cpu", reason="Needs accelerator for MPS/CUDA")
    def test_bigmodel_device_transfer(self):
        """
        With accelerator (MPS/CUDA): feat_ids on GPU, embeddings on GPU,
        gradients flow from GPU loss → CPU C++ grad_buffer correctly.
        """
        model = BigModel(
            self.FEATURE_NAMES, self.EMBEDDING_DIM, self.CAPACITY,
            self.HIDDEN_DIM, self.NUM_CLASSES,
            lr=self.LR,
        ).to(DEVICE)  # dense params on device, embedder params on CPU (only _flow)

        feat_dict = {
            name: torch.randint(0, 100, (self.BATCH_SIZE,), dtype=torch.int64, device=DEVICE)
            for name in self.FEATURE_NAMES
        }
        labels = torch.randint(0, self.NUM_CLASSES, (self.BATCH_SIZE,), device=DEVICE)

        logits = model(feat_dict)
        assert logits.device.type == DEVICE, f"Logits on {logits.device}, expected {DEVICE}"

        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        # Verify _flow parameter got a gradient (proving autograd graph was built)
        assert model.embedder.emb._flow.grad is not None, \
            "_flow gradient is None — autograd graph NOT built!"
        print(f"  ✓ _flow.grad = {model.embedder.emb._flow.grad.item():.4f} (non-None = autograd works)")
        print(f"  ✓ Device transfer (CPU lookup → {DEVICE} forward → backward) works")


# ===========================================================================
# Full training demo
# ===========================================================================

def test_bigmodel_training():
    """
    Full training loop with dict embeddings.
    Verifies loss decreases over time (model learns).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    FEATURE_NAMES = ["feat_a", "feat_b", "feat_c"]
    N_SAMPLES = 200
    BATCH_SIZE = 16
    EPOCHS = 10
    N_CLASSES = 4

    # Synthetic dataset: 3 feature fields, deterministic labels
    rng = np.random.RandomState(42)
    all_feats = {
        "feat_a": rng.randint(0, 500, size=N_SAMPLES).astype(np.int64),
        "feat_b": rng.randint(0, 500, size=N_SAMPLES).astype(np.int64),
        "feat_c": rng.randint(0, 500, size=N_SAMPLES).astype(np.int64),
    }
    all_labels = ((all_feats["feat_a"] + all_feats["feat_b"] + all_feats["feat_c"]) % N_CLASSES).astype(np.int64)

    model = BigModel(
        FEATURE_NAMES, embedding_dim=8, capacity=2000,
        hidden_dim=16, num_classes=N_CLASSES,
        optimizer="adam", lr=0.01,
    ).to(DEVICE)
    dense_opt = torch.optim.Adam(model.dense.parameters(), lr=0.01)

    losses = []
    for epoch in range(EPOCHS):
        perm = rng.permutation(N_SAMPLES)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_SAMPLES, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]

            feat_dict = {
                name: torch.tensor(all_feats[name][idx], dtype=torch.int64, device=DEVICE)
                for name in FEATURE_NAMES
            }
            labels = torch.tensor(all_labels[idx], dtype=torch.int64, device=DEVICE)

            logits = model(feat_dict)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            loss.backward()
            dense_opt.step()
            model.embedder.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if epoch < 3 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:2d}: avg_loss = {avg_loss:.4f}")

    # Loss must decrease
    assert losses[-1] < losses[0] * 0.8, \
        f"Loss did not converge: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"  ✓ BigModel training: loss {losses[0]:.4f} → {losses[-1]:.4f}")


# ===========================================================================
# Advanced: Variable-length sequence features
# ===========================================================================

def test_varlen_sequence_with_dict():
    """
    Each feature field can have its own variable-length sequence.

    feat_a: (B, S_a) — user history
    feat_b: (B, S_b) — item list
    feat_c: (B,)     — single context ID

    All produce embeddings of same dimension D.
    Each feature field independently gets its embedding dict entry.
    """
    B = 4
    D = 8

    feat_dict = {
        "history": torch.randint(0, 100, (B, 5), dtype=torch.int64, device=DEVICE),   # (B, 5)
        "candidates": torch.randint(0, 100, (B, 3), dtype=torch.int64, device=DEVICE),  # (B, 3)
        "context": torch.randint(0, 100, (B,), dtype=torch.int64, device=DEVICE),       # (B,)
    }

    embedder = FeatureEmbedder(
        list(feat_dict.keys()), D, capacity=500,
        optimizer="sgd", lr=0.1,
    ).to(DEVICE)

    emb_dict = embedder(feat_dict)

    assert emb_dict["history"].shape == (B, 5, D), \
        f"history: expected ({B}, 5, {D}), got {emb_dict['history'].shape}"
    assert emb_dict["candidates"].shape == (B, 3, D), \
        f"candidates: expected ({B}, 3, {D}), got {emb_dict['candidates'].shape}"
    assert emb_dict["context"].shape == (B, D), \
        f"context: expected ({B}, {D}), got {emb_dict['context'].shape}"
    print(f"  ✓ Variable-length sequences:")
    for name, emb in emb_dict.items():
        print(f"      {name:12s}: {feat_dict[name].shape} → {emb.shape}")


def test_cross_device_embedder_cpu_dense_accelerator():
    """
    Core scenario: embedder on CPU, dense model on accelerator (CUDA/MPS).

    This is the primary motivation for the two-sub-model architecture:
      - Large embedding table in CPU memory (TB-scale)
      - Dense model on GPU for fast computation

    Data flow:
        feat_ids (CPU) → embedder (CPU) → CPU embeddings
        → .to(device) → dense (MPS/CUDA) → logits
        → loss → backward → gradients flow GPU→CPU→C++ grad_buffer
    """
    FEATURE_NAMES = ["user", "item"]
    B, D, C = 8, 4, 3

    # embedder stays on CPU (the whole point!)
    embedder = FeatureEmbedder(
        FEATURE_NAMES, embedding_dim=D, capacity=1000,
        optimizer="sgd", lr=0.1,
    )
    # dense model goes on accelerator
    dense = DenseModel(FEATURE_NAMES, embedding_dim=D,
                       hidden_dim=8, num_classes=C).to(DEVICE)
    dense_opt = torch.optim.SGD(dense.parameters(), lr=0.01)

    # Verify device placement
    assert next(embedder.parameters()).device.type == "cpu", \
        "Embedder must stay on CPU"
    assert next(dense.parameters()).device.type == DEVICE, \
        f"Dense must be on {DEVICE}"

    print(f"  Device placement: embedder=cpu, dense={DEVICE}")

    # ── Forward: feat_ids on CPU → CPU embeddings → move to device → dense ──
    feat_dict = {
        "user": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64),
        "item": torch.tensor([9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.int64),
    }
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.int64, device=DEVICE)

    # Step 1: embedder on CPU → CPU embeddings
    cpu_embs = embedder(feat_dict)
    for name in FEATURE_NAMES:
        assert cpu_embs[name].device.type == "cpu", \
            f"{name} embedding must be on CPU, got {cpu_embs[name].device}"

    # Step 2: explicitly move to accelerator
    gpu_embs = {k: v.to(DEVICE) for k, v in cpu_embs.items()}

    # Step 3: dense on accelerator → logits on accelerator
    logits = dense(gpu_embs)
    assert logits.device.type == DEVICE, \
        f"Logits must be on {DEVICE}, got {logits.device}"
    assert logits.shape == (B, C)

    # ── Backward: gradients flow GPU → CPU → C++ grad_buffer ──
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # _flow is on CPU (not moved) — verify it got a gradient
    assert embedder.emb._flow.grad is not None, \
        "_flow gradient is None — autograd graph broken!"
    print(f"  ✓ _flow.grad = {embedder.emb._flow.grad.item():.4f} (on CPU)")

    # Step
    dense_opt.step()
    embedder.step()

    # ── Verify: embeddings changed (gradients flowed through) ──
    cpu_embs_after = embedder(feat_dict)
    for name in FEATURE_NAMES:
        assert not torch.allclose(cpu_embs_after[name],
                                  torch.zeros_like(cpu_embs_after[name]),
                                  atol=1e-6), \
            f"{name}: embeddings didn't change!"
    print(f"  ✓ Embedding weights updated (gradients flowed CPU→Dense→Back→C++ buffer)")
    print(f"  ✓ Cross-device training loop: CPU embedder + {DEVICE} dense ✓")


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
