#!/usr/bin/env python3
"""
ML-1M comparison: nn.Embedding vs HashEmb BigModel.

Architecture (NFM-style, single linear layer after embedding concat):
    UserID → embed ─┐
                     ├─ concat → Linear(2*D, 1) → logit
    MovieID → embed ─┘

Data: MovieLens 1M (ratings.dat), label = rating >= 4.

Download:
    # Download ml-1m.zip from https://grouplens.org/datasets/movielens/1m/
    # or use wget:
    wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
    unzip ml-1m.zip

Usage:
    python examples/compare_ml1m.py
    # or specify data path:
    python examples/compare_ml1m.py --data /path/to/ml-1m/ratings.dat
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from hashemb import HashEmbedding

# ===========================================================================
# Config
# ===========================================================================
EMBEDDING_DIM = 16
BATCH_SIZE = 1024
EPOCHS = 10
LR = 0.01
SEED = 42

# ML-1M dataset constants
N_USERS = 6040
N_MOVIES = 3952

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# Data loading
# ===========================================================================
class ML1M_Dataset(Dataset):
    """Load ratings.dat, binary label: rating >= 4 -> 1."""
    def __init__(self, path, train=True, split=0.8, seed=SEED):
        data = []
        with open(path) as f:
            for line in f:
                uid, mid, rating, _ = line.strip().split("::")
                uid = int(uid)
                mid = int(mid)
                label = 1 if int(rating) >= 4 else 0
                data.append((uid, mid, label))

        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(data))
        n_train = int(len(data) * split)
        indices = perm[:n_train] if train else perm[n_train:]
        self.users = np.array([data[i][0] for i in indices], dtype=np.int64)
        self.movies = np.array([data[i][1] for i in indices], dtype=np.int64)
        self.labels = np.array([data[i][2] for i in indices], dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]


def collate_fn(batch):
    users = torch.tensor([b[0] for b in batch], dtype=torch.int64)
    movies = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return users, movies, labels


# ===========================================================================
# Pure PyTorch baseline
# ===========================================================================
class PureEmbeddingModel(torch.nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users + 1, embedding_dim, padding_idx=0)
        self.movie_emb = torch.nn.Embedding(n_movies + 1, embedding_dim, padding_idx=0)
        torch.nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.movie_emb.weight, mean=0, std=0.01)
        self.user_emb.weight.data[0] = 0
        self.movie_emb.weight.data[0] = 0
        total_dim = 2 * embedding_dim
        self.predict = torch.nn.Linear(total_dim, 1)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)

    def forward(self, uid, mid):
        u = self.user_emb(uid)
        m = self.movie_emb(mid)
        return self.predict(torch.cat([u, m], dim=-1)).squeeze(-1)


# ===========================================================================
# HashEmb BigModel
# ===========================================================================
class HashBigModel(torch.nn.Module):
    def __init__(self, dim, capacity_per_field, lr):
        super().__init__()
        self.user_emb = HashEmbedding(dim, capacity_per_field, optimizer="adam", lr=lr)
        self.movie_emb = HashEmbedding(dim, capacity_per_field, optimizer="adam", lr=lr)
        self.predict = torch.nn.Linear(dim * 2, 1)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)

    def forward(self, uid, mid):
        u = self.user_emb(uid)
        m = self.movie_emb(mid)
        return self.predict(torch.cat([u, m], dim=-1)).squeeze(-1)

    def step(self):
        self.user_emb.step()
        self.movie_emb.step()


# ===========================================================================
# Training & evaluation
# ===========================================================================
def train_epoch(model, loader, optimizer, is_hash):
    model.train()
    total_loss, n_batches = 0.0, 0
    for users, movies, labels in loader:
        optimizer.zero_grad()
        logits = model(users, movies)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        if is_hash:
            model.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_scores, all_labels = [], []
    for users, movies, labels in loader:
        logits = model(users, movies)
        all_scores.append(torch.sigmoid(logits).numpy())
        all_labels.append(labels.numpy())
    return roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores))


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", help="Path to ratings.dat")
    args = parser.parse_args()

    # Try common locations for ratings.dat
    candidates = [args.data] if args.data else []
    candidates += [
        "ml-1m/ratings.dat",
        os.path.expanduser("~/ml-1m/ratings.dat"),
        os.path.expanduser("~/data/ml-1m/ratings.dat"),
        "/tmp/ml-1m/ratings.dat",
    ]
    data_path = None
    for p in candidates:
        if p and os.path.isfile(p):
            data_path = p
            break

    if not data_path:
        print("ERROR: ratings.dat not found. Download it:")
        print("  wget https://files.grouplens.org/datasets/movielens/ml-1m.zip")
        print("  unzip ml-1m.zip")
        print("Then run: python examples/compare_ml1m.py --data ml-1m/ratings.dat")
        sys.exit(1)

    print("=" * 60)
    print("ML-1M: nn.Embedding vs HashEmb")
    print("=" * 60)
    print(f"  Data: {data_path}")
    print(f"  dim={EMBEDDING_DIM}, batch={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}")
    print()

    # ── Load data ──
    train_ds = ML1M_Dataset(data_path, train=True)
    val_ds = ML1M_Dataset(data_path, train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Train label balance: {(sum(train_ds.labels)):.0f}/{len(train_ds)}")
    print()

    # ── Pure PyTorch ──
    print("=" * 60)
    print("nn.Embedding baseline")
    print("=" * 60)
    torch.manual_seed(SEED)
    pure_model = PureEmbeddingModel(N_USERS, N_MOVIES, EMBEDDING_DIM)
    pure_opt = torch.optim.Adam(pure_model.parameters(), lr=LR)

    pure_aucs, pure_times = [], []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(pure_model, train_loader, pure_opt, is_hash=False)
        auc = evaluate(pure_model, val_loader)
        elapsed = time.time() - t0
        pure_aucs.append(auc)
        pure_times.append(elapsed)
        print(f"  Ep {epoch:2d}: loss={loss:.4f}  val_auc={auc:.4f}  ({elapsed:.1f}s)")

    print(f"  Best AUC: {max(pure_aucs):.4f}")
    print()

    # ── HashEmb ──
    print("=" * 60)
    print("HashEmb")
    print("=" * 60)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    hash_model = HashBigModel(
        dim=EMBEDDING_DIM,
        capacity_per_field=10000,
        lr=LR,
    )
    hash_opt = torch.optim.Adam(hash_model.predict.parameters(), lr=LR)

    hash_aucs, hash_times = [], []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(hash_model, train_loader, hash_opt, is_hash=True)
        auc = evaluate(hash_model, val_loader)
        elapsed = time.time() - t0
        hash_aucs.append(auc)
        hash_times.append(elapsed)
        print(f"  Ep {epoch:2d}: loss={loss:.4f}  val_auc={auc:.4f}  ({elapsed:.1f}s)")

    print(f"  Best AUC: {max(hash_aucs):.4f}")
    print()

    # ── Comparison ──
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"{'Ep':>3s} | {'Pure':>8s}  {'Hash':>8s} | {'ΔAUC':>8s}")
    print("-" * 38)

    for ep in range(EPOCHS):
        d = pure_aucs[ep] - hash_aucs[ep]
        pm = "★" if pure_aucs[ep] == max(pure_aucs) else " "
        hm = "★" if hash_aucs[ep] == max(hash_aucs) else " "
        print(f"{ep+1:3d} | {pure_aucs[ep]:.4f}{pm} {hash_aucs[ep]:.4f}{hm} | {d:+.4f}")

    print(f"Best | {max(pure_aucs):.4f}   {max(hash_aucs):.4f}   "
          f"{max(pure_aucs)-max(hash_aucs):+.4f}")

    print()
    avg_pure = np.mean(pure_times)
    avg_hash = np.mean(hash_times)
    print(f"Avg epoch: Pure={avg_pure:.2f}s  HashEmb={avg_hash:.2f}s")
    print(f"Speed:     Pure={avg_hash/avg_pure:.2f}x slower" if avg_hash > avg_pure
          else f"Speed:     HashEmb={avg_pure/avg_hash:.2f}x faster")

    if abs(max(pure_aucs) - max(hash_aucs)) < 0.01:
        print("✓ HashEmb AUC matches nn.Embedding (Δ < 0.01)")
    else:
        print(f"⚠ AUC diff = {max(pure_aucs)-max(hash_aucs):.4f} (> 0.01)")
    print("=" * 60)


if __name__ == "__main__":
    main()
