#!/usr/bin/env python3
"""
ML-1M comparison: nn.Embedding vs HashEmb BigModel.

Architecture (NFM-style, single linear layer after embedding concat):
    UserID → embed ─┐
                     ├─ concat → Linear(2*D, 1) → logit
    MovieID → embed ─┘

Single linear层确保dense部分无法"补偿"embedding不更新——embedding直接决定预测质量。
对比 AUC 以验证 HashEmb 在真实数据上效果等价于 nn.Embedding。
"""
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hashemb import HashEmbedding

# ===========================================================================
# Config
# ===========================================================================
EMBEDDING_DIM = 16
BATCH_SIZE = 1024
EPOCHS = 20
LR = 0.01
SEED = 42

DATA_PATH = "/Users/rexus/works/hkv_pproject/ml-1m/ratings.dat"
N_USERS = 6040   # UserID 1..6040
N_MOVIES = 3952  # MovieID 1..3952
DEVICE = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# Data loading
# ===========================================================================
class ML1M_Dataset(Dataset):
    """Load ratings.dat, binary label: rating >= 4 → 1."""
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
        return {
            "user_id": self.users[idx],
            "movie_id": self.movies[idx],
            "label": self.labels[idx],
        }


def collate_fn(batch):
    users = torch.tensor([b["user_id"] for b in batch], dtype=torch.int64)
    movies = torch.tensor([b["movie_id"] for b in batch], dtype=torch.int64)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return {"user_id": users, "movie_id": movies, "label": labels}


# ===========================================================================
# Pure PyTorch baseline
# ===========================================================================
class PureEmbeddingModel(torch.nn.Module):
    """Standard nn.Embedding baseline."""
    def __init__(self, n_users, n_movies, embedding_dim):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users + 1, embedding_dim, padding_idx=0)
        self.movie_emb = torch.nn.Embedding(n_movies + 1, embedding_dim, padding_idx=0)
        torch.nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.movie_emb.weight, mean=0, std=0.01)
        self.user_emb.weight.data[0] = 0
        self.movie_emb.weight.data[0] = 0

        # Single linear layer (NFM-style: concat → 1 layer)
        total_dim = 2 * embedding_dim
        self.predict = torch.nn.Linear(total_dim, 1)

        # Init linear layer
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)

    def forward(self, feat_dict):
        u = self.user_emb(feat_dict["user_id"])
        m = self.movie_emb(feat_dict["movie_id"])
        h = torch.cat([u, m], dim=-1)
        return self.predict(h).squeeze(-1)


# ===========================================================================
# HashEmb BigModel (two-sub-model)
# ===========================================================================
class FeatureEmbedder(torch.nn.Module):
    """SubModel 1: feat_id dict → feat embedding dict.

    Each feature field gets its own HashEmbedding table (per-field tables,
    since feature ID spaces may overlap — e.g. user_id and movie_id).
    """
    def __init__(self, feature_names, embedding_dim, capacity_per_field,
                 optimizer="adam", lr=0.001):
        super().__init__()
        self.feature_names = list(feature_names)
        # Per-field HashEmbedding tables
        for name in feature_names:
            self.__setattr__(f"emb_{name}", HashEmbedding(
                embedding_dim=embedding_dim,
                capacity=capacity_per_field,
                optimizer=optimizer,
                lr=lr,
            ))

    def forward(self, feat_dict):
        return {name: self.__getattr__(f"emb_{name}")(feat_dict[name])
                for name in self.feature_names}

    def step(self):
        for name in self.feature_names:
            self.__getattr__(f"emb_{name}").step()


class DenseModel(torch.nn.Module):
    """SubModel 2: embedding dict → logits. Single linear layer."""
    def __init__(self, feature_names, embedding_dim):
        super().__init__()
        self.feature_names = list(feature_names)
        total_dim = len(feature_names) * embedding_dim
        self.predict = torch.nn.Linear(total_dim, 1)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)

    def forward(self, emb_dict):
        combined = torch.cat(
            [emb_dict[name] for name in self.feature_names], dim=-1)
        return self.predict(combined).squeeze(-1)


def init_hash_emb_like_pure(embedder, field_name, n_ids, mean=0, std=0.01):
    """Initialize a single field's HashEmbedding to match PureEmbeddingModel init."""
    emb = embedder.__getattr__(f"emb_{field_name}")
    feat_ids = torch.arange(1, n_ids + 1, dtype=torch.int64)
    _ = emb(feat_ids)  # create entries
    sd = emb.state_dict()
    rng = np.random.RandomState(SEED)
    D = sd['weight'].shape[1]
    raw = rng.randn(len(sd['keys']), D).astype(np.float32) * std + mean
    sd['weight'] = torch.from_numpy(raw)
    sd['grad'] = torch.zeros_like(sd['grad'])
    if 'm' in sd and sd['m'].numel() > 0:
        sd['m'] = torch.zeros_like(sd['m'])
    if 'v' in sd and sd['v'].numel() > 0:
        sd['v'] = torch.zeros_like(sd['v'])
    emb.load_state_dict(sd)


# ===========================================================================
# Training & evaluation
# ===========================================================================
def train_epoch(model, loader, optimizer, is_hash_model=False):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        feat_dict = {
            "user_id": batch["user_id"],
            "movie_id": batch["movie_id"],
        }
        labels = batch["label"]

        optimizer.zero_grad()
        logits = model(feat_dict)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        if is_hash_model:
            model.embedder.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_scores = []
    all_labels = []
    for batch in loader:
        feat_dict = {
            "user_id": batch["user_id"],
            "movie_id": batch["movie_id"],
        }
        logits = model(feat_dict)
        scores = torch.sigmoid(logits)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    return roc_auc_score(y_true, y_score)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("ML-1M: nn.Embedding vs HashEmb BigModel")
    print("=" * 60)
    print(f"  embedding_dim={EMBEDDING_DIM}, batch_size={BATCH_SIZE}, "
          f"epochs={EPOCHS}, lr={LR}")
    print(f"  Users: {N_USERS}, Movies: {N_MOVIES}")
    print(f"  Architecture: concat → Linear({2*EMBEDDING_DIM}, 1) → sigmoid")
    print()

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    train_dataset = ML1M_Dataset(DATA_PATH, train=True)
    val_dataset = ML1M_Dataset(DATA_PATH, train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print()

    # ── Pure PyTorch baseline ──────────────────────────────────────────
    print("=" * 60)
    print("Model 1: Pure PyTorch nn.Embedding")
    print("=" * 60)
    pure_model = PureEmbeddingModel(N_USERS, N_MOVIES, EMBEDDING_DIM)
    pure_opt = torch.optim.Adam(pure_model.parameters(), lr=LR)

    pure_aucs = []
    pure_times = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(pure_model, train_loader, pure_opt, is_hash_model=False)
        auc = evaluate(pure_model, val_loader)
        elapsed = time.time() - t0
        pure_aucs.append(auc)
        pure_times.append(elapsed)
        print(f"  Epoch {epoch:2d}: loss={loss:.4f}, val_auc={auc:.4f} "
              f"({elapsed:.1f}s)")
    print(f"  → Best val AUC: {max(pure_aucs):.4f}")
    print()

    # ── HashEmb BigModel ───────────────────────────────────────────────
    print("=" * 60)
    print("Model 2: HashEmb BigModel")
    print("=" * 60)
    FEATURE_NAMES = ["user_id", "movie_id"]

    hash_embedder = FeatureEmbedder(
        FEATURE_NAMES, EMBEDDING_DIM, capacity_per_field=10000,
        optimizer="adam", lr=LR,
    )
    hash_dense = DenseModel(FEATURE_NAMES, EMBEDDING_DIM)

    # Initialize each field's HashEmbedding to match PureEmbeddingModel
    init_hash_emb_like_pure(hash_embedder, "user_id", N_USERS, mean=0, std=0.01)
    init_hash_emb_like_pure(hash_embedder, "movie_id", N_MOVIES, mean=0, std=0.01)

    hash_opt = torch.optim.Adam(hash_dense.parameters(), lr=LR)

    class HashBigModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedder = hash_embedder
            self.dense = hash_dense

        def forward(self, feat_dict):
            return self.dense(self.embedder(feat_dict))

    hash_model = HashBigModel()

    hash_aucs = []
    hash_times = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(hash_model, train_loader, hash_opt, is_hash_model=True)
        auc = evaluate(hash_model, val_loader)
        elapsed = time.time() - t0
        hash_aucs.append(auc)
        hash_times.append(elapsed)
        print(f"  Epoch {epoch:2d}: loss={loss:.4f}, val_auc={auc:.4f} "
              f"({elapsed:.1f}s)")
    print(f"  → Best val AUC: {max(hash_aucs):.4f}")
    print()

    # ── Comparison summary ─────────────────────────────────────────────
    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"  {'Epoch':>5s} | {'Pure AUC':>8s} {'Hash AUC':>8s} {'ΔAUC':>8s}")
    print(f"  {'─'*5}-|-{'─'*8}-{'─'*8}-{'─'*8}")
    for epoch in range(EPOCHS):
        delta = pure_aucs[epoch] - hash_aucs[epoch]
        marker = " ★" if epoch == np.argmax(pure_aucs) else "  "
        marker2 = " ★" if epoch == np.argmax(hash_aucs) else "  "
        print(f"  {epoch+1:5d} | {pure_aucs[epoch]:.4f}{marker} "
              f"{hash_aucs[epoch]:.4f}{marker2} {delta:+.4f}")

    best_pure = max(pure_aucs)
    best_hash = max(hash_aucs)
    print(f"  {'─'*5}-|-{'─'*8}-{'─'*8}-{'─'*8}")
    print(f"  {'Best':>5s} | {best_pure:.4f}   {best_hash:.4f}   "
          f"{best_pure - best_hash:+.4f}")

    print()
    if abs(best_pure - best_hash) < 0.01:
        print("  ✓ HashEmb AUC matches nn.Embedding within 0.01")
    else:
        print(f"  ⚠ AUC diff = {best_pure - best_hash:.4f}")

    avg_time_pure = np.mean(pure_times)
    avg_time_hash = np.mean(hash_times)
    print(f"\n  Avg epoch time: Pure={avg_time_pure:.2f}s, "
          f"HashEmb={avg_time_hash:.2f}s")


if __name__ == "__main__":
    main()
