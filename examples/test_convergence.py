#!/usr/bin/env python3
"""
Convergence test: HashEmb vs nn.Embedding on synthetic recommendation data.

Uses synthetic data with a learnable signal: label is generated from
a ground-truth embedding table, so models can actually learn the pattern.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hashemb import HashEmbedding

# ============================================================================
# Config
# ============================================================================
N_USERS = 500
N_ITEMS = 300
EMBEDDING_DIM = 16
N_SAMPLES = 3000
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.01
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("Convergence Test: HashEmb vs nn.Embedding")
print("=" * 70)
print(f"Users={N_USERS}, Items={N_ITEMS}, Dim={EMBEDDING_DIM}, "
      f"Samples={N_SAMPLES}, Epochs={EPOCHS}")
print()

# ============================================================================
# Generate synthetic data with a LEARNABLE signal
# ============================================================================
# Create ground-truth embedding vectors for each user and item
gt_user = torch.randn(N_USERS + 1, EMBEDDING_DIM)
gt_item = torch.randn(N_ITEMS + 1, EMBEDDING_DIM)

# Generate sample pairs and labels using dot-product similarity + noise
user_ids = np.random.randint(1, N_USERS + 1, size=N_SAMPLES, dtype=np.int64)
item_ids = np.random.randint(1, N_ITEMS + 1, size=N_SAMPLES, dtype=np.int64)

scores = (gt_user[user_ids] * gt_item[item_ids]).sum(1).numpy()
probs = 1.0 / (1.0 + np.exp(-scores))  # sigmoid
labels = (np.random.random(N_SAMPLES) < probs).astype(np.float32)

split = int(0.8 * N_SAMPLES)
train_u = torch.from_numpy(user_ids[:split])
train_i = torch.from_numpy(item_ids[:split])
train_y = torch.from_numpy(labels[:split])
val_u = torch.from_numpy(user_ids[split:])
val_i = torch.from_numpy(item_ids[split:])
val_y = torch.from_numpy(labels[split:])

print(f"Train: {len(train_y)}, Val: {len(val_y)}")
print(f"Train label balance: pos={train_y.sum():.0f}/{len(train_y)} "
      f"({train_y.mean():.2%})")
print()

# ============================================================================
# Model definitions
# ============================================================================

class TorchModel(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pred = nn.Linear(dim * 2, 1)

    def forward(self, uid, iid):
        u = self.user_emb(uid)
        i = self.item_emb(iid)
        return self.pred(torch.cat([u, i], dim=-1)).squeeze(-1)


class HashModel(nn.Module):
    def __init__(self, dim, capacity, lr):
        super().__init__()
        self.user_emb = HashEmbedding(dim, capacity, optimizer="adam", lr=lr)
        self.item_emb = HashEmbedding(dim, capacity, optimizer="adam", lr=lr)
        self.pred = nn.Linear(dim * 2, 1)

    def forward(self, uid, iid):
        u = self.user_emb(uid)
        i = self.item_emb(iid)
        return self.pred(torch.cat([u, i], dim=-1)).squeeze(-1)

    def step(self):
        self.user_emb.step()
        self.item_emb.step()


def train_one_model(name, model, opt, step_fn):
    """Train for EPOCHS, return (train_losses, val_losses)."""
    print(f"Training {name}...")
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()

        perm = torch.randperm(len(train_y))
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(train_y), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            logits = model(train_u[idx], train_i[idx])
            loss = F.binary_cross_entropy_with_logits(logits, train_y[idx])
            loss.backward()
            opt.step()
            step_fn()
            total_loss += loss.item()
            n_batches += 1

        train_losses.append(total_loss / n_batches)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_u, val_i)
            val_acc = ((torch.sigmoid(val_logits) > 0.5).float() == val_y).float().mean().item()
            val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y).item()
            val_losses.append(val_loss)

        print(f"  Epoch {epoch+1:2d}: train={train_losses[-1]:.4f}  "
              f"val={val_losses[-1]:.4f}  acc={val_acc:.3f}")

    return train_losses, val_losses


# ============================================================================
# Run both models
# ============================================================================

# nn.Embedding baseline
torch.manual_seed(SEED)
model_torch = TorchModel(N_USERS, N_ITEMS, EMBEDDING_DIM)
opt_torch = torch.optim.Adam(model_torch.parameters(), lr=LR)
torch_losses, torch_val = train_one_model(
    "nn.Embedding", model_torch, opt_torch,
    step_fn=lambda: opt_torch.zero_grad()
)
print()

# HashEmb
torch.manual_seed(SEED)
np.random.seed(SEED)
model_hash = HashModel(dim=EMBEDDING_DIM,
                       capacity=N_USERS + N_ITEMS + 100, lr=LR)
opt_hash = torch.optim.Adam(model_hash.pred.parameters(), lr=LR)
hash_losses, hash_val = train_one_model(
    "HashEmb", model_hash, opt_hash,
    step_fn=model_hash.step
)

# ============================================================================
# Results
# ============================================================================
print()
print("=" * 70)
print("Convergence Summary")
print("=" * 70)
print(f"{'Ep':>3s} | {'Torch train':>12s} {'Hash train':>12s} | "
      f"{'Torch val':>12s} {'Hash val':>12s}")
print("-" * 65)

for ep in range(EPOCHS):
    print(f"{ep+1:3d} | {torch_losses[ep]:12.6f} {hash_losses[ep]:12.6f} | "
          f"{torch_val[ep]:12.6f} {hash_val[ep]:12.6f}")

# Criteria: loss must decrease from epoch 1 to epoch EPOCHS
t_ok = torch_losses[-1] < torch_losses[0] * 0.8
h_ok = hash_losses[-1] < hash_losses[0] * 0.8

print()
if t_ok and h_ok:
    print("✓ PASS: Both models converged")
    print(f"  nn.Embedding: train {torch_losses[0]:.4f} → {torch_losses[-1]:.4f}")
    print(f"  HashEmb:      train {hash_losses[0]:.4f} → {hash_losses[-1]:.4f}")
elif t_ok:
    print("✗ FAIL: HashEmb did not converge")
elif h_ok:
    print("✗ FAIL: nn.Embedding did not converge")
else:
    print("✗ FAIL: Neither model converged")
print("=" * 70)
