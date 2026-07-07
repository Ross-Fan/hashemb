#!/usr/bin/env python3
"""
Convergence test: HashEmb vs nn.Embedding on synthetic recommendation data.

Equivalent models, same data, both should converge to low loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hashemb import HashEmbedding

# ============================================================================
# Config
# ============================================================================
N_USERS = 1000
N_MOVIES = 500
EMBEDDING_DIM = 16
N_SAMPLES = 5000
BATCH_SIZE = 128
EPOCHS = 15
LR = 0.01
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("Convergence Test: HashEmb vs nn.Embedding")
print("=" * 70)
print(f"Users={N_USERS}, Movies={N_MOVIES}, Dim={EMBEDDING_DIM}, "
      f"Samples={N_SAMPLES}, Epochs={EPOCHS}")
print()

# ============================================================================
# Synthetic data: user_id + movie_id → binary label
# ============================================================================
user_ids = np.random.randint(1, N_USERS + 1, size=N_SAMPLES, dtype=np.int64)
movie_ids = np.random.randint(1, N_MOVIES + 1, size=N_SAMPLES, dtype=np.int64)
labels = np.random.randint(0, 2, size=N_SAMPLES).astype(np.float32)

split = int(0.8 * N_SAMPLES)
train_u = torch.from_numpy(user_ids[:split])
train_m = torch.from_numpy(movie_ids[:split])
train_y = torch.from_numpy(labels[:split])
val_u = torch.from_numpy(user_ids[split:])
val_m = torch.from_numpy(movie_ids[split:])
val_y = torch.from_numpy(labels[split:])

print(f"Train: {len(train_y)}, Val: {len(val_y)}")
print()

# ============================================================================
# Model definitions
# ============================================================================

class TorchModel(nn.Module):
    def __init__(self, n_users, n_movies, dim):
        super().__init__()
        # Use all 0.1 init so both models start identically
        self.user_emb = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.movie_emb = nn.Embedding(n_movies + 1, dim, padding_idx=0)
        nn.init.constant_(self.user_emb.weight, 0.1)
        nn.init.constant_(self.movie_emb.weight, 0.1)
        self.pred = nn.Linear(dim * 2, 1)

    def forward(self, uid, mid):
        u = self.user_emb(uid)
        m = self.movie_emb(mid)
        return self.pred(torch.cat([u, m], dim=-1)).squeeze(-1)


class HashModel(nn.Module):
    def __init__(self, dim, capacity, lr):
        super().__init__()
        self.user_emb = HashEmbedding(dim, capacity, optimizer="adam", lr=lr)
        self.movie_emb = HashEmbedding(dim, capacity, optimizer="adam", lr=lr)
        self.pred = nn.Linear(dim * 2, 1)

    def forward(self, uid, mid):
        u = self.user_emb(uid)
        m = self.movie_emb(mid)
        return self.pred(torch.cat([u, m], dim=-1)).squeeze(-1)

    def step(self):
        self.user_emb.step()
        self.movie_emb.step()


# ============================================================================
# Train nn.Embedding baseline
# ============================================================================
print("Training nn.Embedding baseline...")
torch.manual_seed(SEED)
model_torch = TorchModel(N_USERS, N_MOVIES, EMBEDDING_DIM)
opt_torch = torch.optim.Adam(model_torch.parameters(), lr=LR)

torch_losses = []

for epoch in range(EPOCHS):
    model_torch.train()
    opt_torch.zero_grad()

    # Shuffle
    perm = torch.randperm(len(train_y))
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(train_y), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        logits = model_torch(train_u[idx], train_m[idx])
        loss = F.binary_cross_entropy_with_logits(logits, train_y[idx])
        loss.backward()
        opt_torch.step()
        opt_torch.zero_grad()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    torch_losses.append(avg_loss)

    # Val
    model_torch.eval()
    with torch.no_grad():
        val_logits = model_torch(val_u, val_m)
        val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y).item()

    print(f"  Epoch {epoch+1:2d}: train={avg_loss:.4f}  val={val_loss:.4f}")

print()

# ============================================================================
# Train HashEmb
# ============================================================================
print("Training HashEmb...")
torch.manual_seed(SEED)
np.random.seed(SEED)

model_hash = HashModel(
    dim=EMBEDDING_DIM,
    capacity=N_USERS + N_MOVIES + 100,
    lr=LR,
)
opt_hash = torch.optim.Adam(model_hash.pred.parameters(), lr=LR)

hash_losses = []

for epoch in range(EPOCHS):
    model_hash.train()
    opt_hash.zero_grad()

    perm = torch.randperm(len(train_y))
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(train_y), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        logits = model_hash(train_u[idx], train_m[idx])
        loss = F.binary_cross_entropy_with_logits(logits, train_y[idx])
        loss.backward()
        opt_hash.step()
        model_hash.step()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    hash_losses.append(avg_loss)

    # Val
    model_hash.eval()
    with torch.no_grad():
        val_logits = model_hash(val_u, val_m)
        val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y).item()

    print(f"  Epoch {epoch+1:2d}: train={avg_loss:.4f}  val={val_loss:.4f}")

# ============================================================================
# Results
# ============================================================================
print()
print("=" * 70)
print("Convergence Summary")
print("=" * 70)
print(f"{'Epoch':>5s} | {'Torch':>10s} {'Hash':>10s} | {'Both ↓':>10s}")
print("-" * 45)

for epoch in range(EPOCHS):
    t = torch_losses[epoch]
    h = hash_losses[epoch]
    mark = "✓" if (t < torch_losses[0] * 0.5 and h < hash_losses[0] * 0.5) else ""
    print(f"{epoch+1:5d} | {t:10.4f} {h:10.4f} | {mark:>10s}")

# Both should converge (final loss < 50% of initial)
torch_converged = torch_losses[-1] < torch_losses[0] * 0.5
hash_converged = hash_losses[-1] < hash_losses[0] * 0.5

print()
if torch_converged and hash_converged:
    print("✓ PASS: Both models converged")
    print(f"  nn.Embedding: {torch_losses[0]:.4f} → {torch_losses[-1]:.4f} ({torch_losses[-1]/torch_losses[0]*100:.1f}%)")
    print(f"  HashEmb:      {hash_losses[0]:.4f} → {hash_losses[-1]:.4f} ({hash_losses[-1]/hash_losses[0]*100:.1f}%)")
else:
    print("✗ FAIL: One or both models did not converge")
    print(f"  nn.Embedding: {torch_losses[0]:.4f} → {torch_losses[-1]:.4f}")
    print(f"  HashEmb:      {hash_losses[0]:.4f} → {hash_losses[-1]:.4f}")

print("=" * 70)
