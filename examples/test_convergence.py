#!/usr/bin/env python3
"""
Convergence test: Compare HashEmb with nn.Embedding using synthetic data.

This test verifies that the sort-based deduplication optimization
maintains numerical correctness by comparing with nn.Embedding baseline.
"""
import sys
sys.path.insert(0, '/Users/fanwei/study/HKV/hashemb')

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
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.01
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("Convergence Test: HashEmb vs nn.Embedding")
print("=" * 70)
print(f"Config: users={N_USERS}, movies={N_MOVIES}, dim={EMBEDDING_DIM}, "
      f"batch={BATCH_SIZE}, epochs={EPOCHS}")
print()

# ============================================================================
# Generate synthetic data
# ============================================================================
print("Generating synthetic data...")

N_SAMPLES = 5000
user_ids = np.random.randint(1, N_USERS + 1, size=N_SAMPLES, dtype=np.int64)
movie_ids = np.random.randint(1, N_MOVIES + 1, size=N_SAMPLES, dtype=np.int64)
# Random binary labels
labels = np.random.randint(0, 2, size=N_SAMPLES, dtype=np.int32).astype(np.float32)

# Split train/val (80/20)
split = int(0.8 * N_SAMPLES)
train_user_ids = torch.from_numpy(user_ids[:split])
train_movie_ids = torch.from_numpy(movie_ids[:split])
train_labels = torch.from_numpy(labels[:split])

val_user_ids = torch.from_numpy(user_ids[split:])
val_movie_ids = torch.from_numpy(movie_ids[split:])
val_labels = torch.from_numpy(labels[split:])

print(f"  Train samples: {len(train_labels)}")
print(f"  Val samples: {len(val_labels)}")
print()

# ============================================================================
# Model 1: nn.Embedding baseline
# ============================================================================
print("Training nn.Embedding baseline...")

class EmbeddingModel(nn.Module):
    def __init__(self, n_users, n_movies, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.movie_emb = nn.Embedding(n_movies + 1, dim, padding_idx=0)
        self.pred = nn.Linear(dim * 2, 1)

    def forward(self, user_id, movie_id):
        u = self.user_emb(user_id)
        m = self.movie_emb(movie_id)
        concat = torch.cat([u, m], dim=-1)
        return self.pred(concat).squeeze(-1)

model_torch = EmbeddingModel(N_USERS, N_MOVIES, EMBEDDING_DIM)
optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=LR)

torch_losses = []
torch_val_losses = []

for epoch in range(EPOCHS):
    # Train
    model_torch.train()
    optimizer_torch.zero_grad()
    logits = model_torch(train_user_ids, train_movie_ids)
    loss = F.binary_cross_entropy_with_logits(logits, train_labels)
    loss.backward()
    optimizer_torch.step()
    torch_losses.append(loss.item())

    # Val
    model_torch.eval()
    with torch.no_grad():
        val_logits = model_torch(val_user_ids, val_movie_ids)
        val_loss = F.binary_cross_entropy_with_logits(val_logits, val_labels).item()
        torch_val_losses.append(val_loss)

    print(f"  Epoch {epoch+1:2d}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

print()

# ============================================================================
# Model 2: HashEmb with sort-based deduplication
# ============================================================================
print("Training HashEmb...")

class HashEmbeddingModel(nn.Module):
    def __init__(self, capacity, dim, optimizer, lr):
        super().__init__()
        self.user_emb = HashEmbedding(dim, capacity, optimizer=optimizer, lr=lr)
        self.movie_emb = HashEmbedding(dim, capacity, optimizer=optimizer, lr=lr)
        self.pred = nn.Linear(dim * 2, 1)

    def forward(self, user_id, movie_id):
        u = self.user_emb(user_id)
        m = self.movie_emb(movie_id)
        concat = torch.cat([u, m], dim=-1)
        return self.pred(concat).squeeze(-1)

    def step(self):
        self.user_emb.step()
        self.movie_emb.step()

model_hash = HashEmbeddingModel(
    capacity=N_USERS + N_MOVIES + 100,
    dim=EMBEDDING_DIM,
    optimizer="adam",
    lr=LR
)
optimizer_hash = torch.optim.Adam(model_hash.pred.parameters(), lr=LR)

# Initialize HashEmb with same weights as nn.Embedding
print("  Initializing HashEmb with same weights...")
init_user_ids = torch.arange(1, N_USERS + 1, dtype=torch.int64)
init_movie_ids = torch.arange(1, N_MOVIES + 1, dtype=torch.int64)

with torch.no_grad():
    # Initialize user embeddings
    user_emb = model_hash.user_emb(init_user_ids)
    user_emb.copy_(model_torch.user_emb.weight[1:N_USERS+1].detach().cpu())
    sd = model_hash.user_emb.state_dict()
    sd['weight'] = torch.from_numpy(user_emb.numpy())
    sd['grad'] = torch.zeros_like(sd['weight'])
    model_hash.user_emb.load_state_dict(sd)

    # Initialize movie embeddings
    movie_emb = model_hash.movie_emb(init_movie_ids)
    movie_emb.copy_(model_torch.movie_emb.weight[1:N_MOVIES+1].detach().cpu())
    sd = model_hash.movie_emb.state_dict()
    sd['weight'] = torch.from_numpy(movie_emb.numpy())
    sd['grad'] = torch.zeros_like(sd['grad'])
    model_hash.movie_emb.load_state_dict(sd)

hash_losses = []
hash_val_losses = []

for epoch in range(EPOCHS):
    # Train
    model_hash.train()
    optimizer_hash.zero_grad()
    logits = model_hash(train_user_ids, train_movie_ids)
    loss = F.binary_cross_entropy_with_logits(logits, train_labels)
    loss.backward()
    optimizer_hash.step()
    model_hash.step()
    hash_losses.append(loss.item())

    # Val
    model_hash.eval()
    with torch.no_grad():
        val_logits = model_hash(val_user_ids, val_movie_ids)
        val_loss = F.binary_cross_entropy_with_logits(val_logits, val_labels).item()
        hash_val_losses.append(val_loss)

    print(f"  Epoch {epoch+1:2d}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

print()

# ============================================================================
# Compare results
# ============================================================================
print("=" * 70)
print("Convergence Comparison")
print("=" * 70)

print(f"{'Epoch':>5s} | {'Torch Loss':>12s} {'Hash Loss':>12s} {'Diff':>10s} | {'Torch Val':>12s} {'Hash Val':>12s} {'Diff':>10s}")
print("-" * 80)

max_diff = 0.0
max_val_diff = 0.0

for epoch in range(EPOCHS):
    train_diff = abs(torch_losses[epoch] - hash_losses[epoch])
    val_diff = abs(torch_val_losses[epoch] - hash_val_losses[epoch])
    max_diff = max(max_diff, train_diff)
    max_val_diff = max(max_val_diff, val_diff)

    print(f"{epoch+1:5d} | {torch_losses[epoch]:12.6f} {hash_losses[epoch]:12.6f} {train_diff:10.6f} | "
          f"{torch_val_losses[epoch]:12.6f} {hash_val_losses[epoch]:12.6f} {val_diff:10.6f}")

print()
print(f"Max train loss diff: {max_diff:.8f}")
print(f"Max val loss diff:   {max_val_diff:.8f}")

# Pass if differences are very small (numerical tolerance)
# Note: Small numerical differences are expected due to different implementations
TOLERANCE = 1e-2  # 0.01 is acceptable for convergence comparison
if max_diff < TOLERANCE and max_val_diff < TOLERANCE:
    print()
    print("✓ PASS: Convergence matches within tolerance")
    print(f"  Max diff is {max_diff:.6f} < {TOLERANCE}")
else:
    print()
    print(f"✗ FAIL: Differences exceed tolerance {TOLERANCE}")

print()
print("=" * 70)